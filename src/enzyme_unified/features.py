from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


def preprocess_protein_sequence(seq: str) -> str:
    # ProtT5 通常要求氨基酸间空格分隔，并将非常规字符映射为 X
    seq = seq.upper().replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return " ".join(list(seq))


def compute_maccs(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    for i in range(fp.GetNumBits()):
        arr[i] = float(fp.GetBit(i))
    if arr.shape[0] == 167:
        return arr
    # 保险处理，确保固定 167 维
    out = np.zeros(167, dtype=np.float32)
    out[: min(167, arr.shape[0])] = arr[: min(167, arr.shape[0])]
    return out


def compute_physchem_22(seq: str) -> np.ndarray:
    seq = seq.upper()
    analysis = ProteinAnalysis(seq)
    aa_percent = analysis.get_amino_acids_percent()
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    composition = np.array([aa_percent.get(aa, 0.0) for aa in aa_order], dtype=np.float32)
    extras = np.array(
        [
            analysis.molecular_weight(),
            analysis.isoelectric_point(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([composition, extras], axis=0)  # 20 + 2 = 22


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        protein_model_name: str,
        substrate_model_name: str,
        use_prostt5: bool,
        freeze_encoders: bool = True,
        max_protein_length: int = 1024,
        max_smiles_length: int = 256,
    ):
        super().__init__()
        self.max_protein_length = max_protein_length
        self.max_smiles_length = max_smiles_length
        self.use_prostt5 = use_prostt5
        self.freeze_encoders = freeze_encoders

        self.protein_tokenizer = self._build_tokenizer(protein_model_name, is_protein=True)
        self.protein_model = self._build_encoder_model(protein_model_name)
        self.substrate_tokenizer = self._build_tokenizer(substrate_model_name, is_protein=False)
        self.substrate_model = self._build_encoder_model(substrate_model_name)

        self._protein_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._substrate_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._maccs_cache: Dict[str, torch.Tensor] = {}
        self._phys_cache: Dict[str, torch.Tensor] = {}

        if self.freeze_encoders:
            for p in self.protein_model.parameters():
                p.requires_grad = False
            for p in self.substrate_model.parameters():
                p.requires_grad = False

    @staticmethod
    def _build_tokenizer(model_name: str, is_protein: bool):
        """
        强制优先使用 slow tokenizer，避免某些 T5 权重触发 fast tokenizer 转换报错：
        'Unigram model ... trained with a different algorithm'
        """
        base_kwargs = {"use_fast": False}
        if is_protein:
            base_kwargs["do_lower_case"] = False
            # 对 T5 家族 tokenizer 显式指定 legacy，消除行为歧义
            base_kwargs["legacy"] = True
        try:
            return AutoTokenizer.from_pretrained(model_name, **base_kwargs)
        except TypeError:
            # 某些 tokenizer 不接受 legacy 参数，回退到最小参数集合
            fallback = {"use_fast": False}
            if is_protein:
                fallback["do_lower_case"] = False
            return AutoTokenizer.from_pretrained(model_name, **fallback)

    @staticmethod
    def _build_encoder_model(model_name: str):
        """
        ProtT5/ProstT5/MolT5 本质是 T5 家族。我们只需要 encoder 表征，
        若误用完整 T5Model 会在 forward 时要求 decoder_input_ids。
        """
        cfg = AutoConfig.from_pretrained(model_name)
        if getattr(cfg, "is_encoder_decoder", False):
            return T5EncoderModel.from_pretrained(model_name)
        return AutoModel.from_pretrained(model_name)

    def _encode_protein(self, sequences: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        missing = [seq for seq in sequences if seq not in self._protein_cache]
        if missing:
            texts = [preprocess_protein_sequence(seq) for seq in missing]
            tokenized = self.protein_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_protein_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
                outputs = self.protein_model(**tokenized)
            hidden = outputs.last_hidden_state.detach().cpu()
            attn_mask = tokenized["attention_mask"].detach().cpu()
            lengths = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (hidden * attn_mask.unsqueeze(-1)).sum(dim=1) / lengths
            for i, seq in enumerate(missing):
                self._protein_cache[seq] = {
                    "token": hidden[i],
                    "mask": attn_mask[i],
                    "pool": pooled[i],
                }

        token_list = [self._protein_cache[seq]["token"] for seq in sequences]
        mask_list = [self._protein_cache[seq]["mask"] for seq in sequences]
        pool_list = [self._protein_cache[seq]["pool"] for seq in sequences]
        token_batch = nn.utils.rnn.pad_sequence(token_list, batch_first=True)
        mask_batch = nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
        pool_batch = torch.stack(pool_list, dim=0)
        return {
            "token": token_batch.to(device),
            "mask": mask_batch.to(device),
            "pool": pool_batch.to(device),
        }

    def _encode_substrate(self, smiles_list: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        missing = [s for s in smiles_list if s not in self._substrate_cache]
        if missing:
            tokenized = self.substrate_tokenizer(
                missing,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_smiles_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
                outputs = self.substrate_model(**tokenized)
            hidden = outputs.last_hidden_state.detach().cpu()
            attn_mask = tokenized["attention_mask"].detach().cpu()
            lengths = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (hidden * attn_mask.unsqueeze(-1)).sum(dim=1) / lengths
            for i, s in enumerate(missing):
                self._substrate_cache[s] = {
                    "token": hidden[i],
                    "mask": attn_mask[i],
                    "pool": pooled[i],
                }

        token_list = [self._substrate_cache[s]["token"] for s in smiles_list]
        mask_list = [self._substrate_cache[s]["mask"] for s in smiles_list]
        pool_list = [self._substrate_cache[s]["pool"] for s in smiles_list]
        token_batch = nn.utils.rnn.pad_sequence(token_list, batch_first=True)
        mask_batch = nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
        pool_batch = torch.stack(pool_list, dim=0)
        return {
            "token": token_batch.to(device),
            "mask": mask_batch.to(device),
            "pool": pool_batch.to(device),
        }

    def _maccs(self, smiles_list: List[str], device: torch.device) -> torch.Tensor:
        vectors = []
        for smiles in smiles_list:
            if smiles not in self._maccs_cache:
                self._maccs_cache[smiles] = torch.tensor(compute_maccs(smiles), dtype=torch.float32)
            vectors.append(self._maccs_cache[smiles])
        return torch.stack(vectors, dim=0).to(device)

    def _physchem(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        vectors = []
        for seq in sequences:
            if seq not in self._phys_cache:
                feat = compute_physchem_22(seq)
                self._phys_cache[seq] = torch.tensor(feat, dtype=torch.float32)
            vectors.append(self._phys_cache[seq])
        return torch.stack(vectors, dim=0).to(device)

    def encode_batch(
        self, sequences: List[str], smiles_list: List[str], use_physchem: bool, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        protein = self._encode_protein(sequences, device=device)
        substrate = self._encode_substrate(smiles_list, device=device)
        maccs = self._maccs(smiles_list, device=device)
        out = {
            "protein_token": protein["token"],
            "protein_mask": protein["mask"],
            "protein_pool": protein["pool"],
            "substrate_token": substrate["token"],
            "substrate_mask": substrate["mask"],
            "substrate_pool": substrate["pool"],
            "maccs": maccs,
        }
        if use_physchem:
            out["physchem"] = self._physchem(sequences, device=device)
        return out

