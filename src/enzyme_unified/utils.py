import math
import os
import random
from typing import Dict, Iterable, List

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def transform_targets(y: torch.Tensor, log_target: bool) -> torch.Tensor:
    if log_target:
        return torch.log10(y)
    return y


def inverse_transform_targets(y_hat: np.ndarray, log_target: bool) -> np.ndarray:
    if log_target:
        return np.power(10.0, y_hat)
    return y_hat


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_corr(
    fn, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return float("nan")
    value, _ = fn(y_true, y_pred)
    return float(value)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.array(list(y_true), dtype=np.float64)
    y_pred_arr = np.array(list(y_pred), dtype=np.float64)
    return {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "pcc": safe_corr(pearsonr, y_true_arr, y_pred_arr),
        "scc": safe_corr(spearmanr, y_true_arr, y_pred_arr),
    }


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out

