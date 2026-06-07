from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_task_dataframe(csv_path: str, label_col: str, log_target: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"Sequence", "Smiles", "fold", label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺失列: {sorted(missing)}")

    df = df.dropna(subset=["Sequence", "Smiles", "fold", label_col]).copy()
    df["fold"] = df["fold"].astype(int)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    if log_target:
        df = df[df[label_col] > 0].copy()
    return df.reset_index(drop=True)


def build_fold_split(
    df: pd.DataFrame,
    total_folds: int,
    test_fold: int,
    strategy: str,
    seed: int,
) -> SplitData:
    if not (0 <= test_fold < total_folds):
        raise ValueError(f"test_fold={test_fold} 超出范围 [0, {total_folds - 1}]")

    test_df = df[df["fold"] == test_fold].copy()
    pool_df = df[df["fold"] != test_fold].copy()
    if strategy == "modulo1":
        val_fold = (test_fold + 1) % total_folds
        val_df = pool_df[pool_df["fold"] == val_fold].copy()
        train_df = pool_df[pool_df["fold"] != val_fold].copy()
    elif strategy == "random90_10":
        rng = np.random.default_rng(seed + test_fold)
        idx = np.arange(len(pool_df))
        rng.shuffle(idx)
        cut = int(len(idx) * 0.9)
        train_idx = idx[:cut]
        val_idx = idx[cut:]
        train_df = pool_df.iloc[train_idx].copy()
        val_df = pool_df.iloc[val_idx].copy()
    else:
        raise ValueError(f"未知划分策略: {strategy}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("划分后存在空集合，请检查 fold 或策略。")
    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


class EnzymeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        return {
            "sequence": str(row["Sequence"]),
            "smiles": str(row["Smiles"]),
            "label_raw": float(row[self.label_col]),
        }


def collate_samples(samples: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "sequence": [item["sequence"] for item in samples],
        "smiles": [item["smiles"] for item in samples],
        "label_raw": torch.tensor([item["label_raw"] for item in samples], dtype=torch.float32),
    }

