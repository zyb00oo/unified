from typing import Dict, Optional

import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: 1 for valid tokens
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denom


class BiCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn_e = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_s = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_e1 = nn.LayerNorm(hidden_dim)
        self.norm_s1 = nn.LayerNorm(hidden_dim)
        self.ffn_e = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_e2 = nn.LayerNorm(hidden_dim)
        self.norm_s2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        e: torch.Tensor,
        s: torch.Tensor,
        e_padding_mask: torch.Tensor,
        s_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MultiheadAttention 里 key_padding_mask=True 表示 padding 位
        e2, _ = self.attn_e(
            query=e,
            key=s,
            value=s,
            key_padding_mask=s_padding_mask,
            need_weights=False,
        )
        s2, _ = self.attn_s(
            query=s,
            key=e,
            value=e,
            key_padding_mask=e_padding_mask,
            need_weights=False,
        )
        e = self.norm_e1(e + e2)
        s = self.norm_s1(s + s2)
        e = self.norm_e2(e + self.ffn_e(e))
        s = self.norm_s2(s + self.ffn_s(s))
        return e, s


class EnzymeUnifiedModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        substrate_dim: int,
        maccs_dim: int = 167,
        physchem_dim: int = 22,
        hidden_dim: int = 768,
        num_heads: int = 8,
        cross_layers: int = 1,
        dropout: float = 0.1,
        use_physchem: bool = False,
    ):
        super().__init__()
        self.use_physchem = use_physchem
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.substrate_proj = nn.Linear(substrate_dim, hidden_dim)
        self.maccs_proj = nn.Linear(maccs_dim, hidden_dim)

        self.cross_blocks = nn.ModuleList(
            [BiCrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(cross_layers)]
        )

        self.attn_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        global_in = protein_dim + substrate_dim + maccs_dim + (physchem_dim if use_physchem else 0)
        self.bn_protein = nn.BatchNorm1d(protein_dim)
        self.bn_substrate = nn.BatchNorm1d(substrate_dim)
        self.concat_head = nn.Sequential(
            nn.Linear(global_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 门控参数 alpha，对应论文式(4)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.protein_proj(feat["protein_token"])
        s = self.substrate_proj(feat["substrate_token"])
        e_mask = feat["protein_mask"] > 0
        s_mask = feat["substrate_mask"] > 0
        e_padding_mask = ~e_mask
        s_padding_mask = ~s_mask

        for block in self.cross_blocks:
            e, s = block(e, s, e_padding_mask=e_padding_mask, s_padding_mask=s_padding_mask)

        e_pool = masked_mean(e, e_mask)
        s_pool = masked_mean(s, s_mask)
        maccs_proj = self.maccs_proj(feat["maccs"])
        attn_input = torch.cat([e_pool, s_pool, maccs_proj], dim=-1)
        y_attn = self.attn_head(attn_input)

        p_global = self.bn_protein(feat["protein_pool"])
        s_global = self.bn_substrate(feat["substrate_pool"])
        concat_parts = [p_global, s_global, feat["maccs"]]
        if self.use_physchem:
            concat_parts.append(feat["physchem"])
        concat_input = torch.cat(concat_parts, dim=-1)
        y_concat = self.concat_head(concat_input)

        gate = torch.sigmoid(self.alpha)
        y = gate * y_attn + (1.0 - gate) * y_concat
        return y.squeeze(-1)

