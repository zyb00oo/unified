import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import inverse_transform_targets, regression_metrics, transform_targets


@dataclass
class TrainResult:
    best_epoch: int
    best_val_rmse: float
    test_metrics: Dict[str, float]
    history: List[Dict[str, float]]


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _is_main_process() -> bool:
    return _rank() == 0


def _all_reduce_sum(value: float, device: torch.device) -> float:
    if not _is_dist():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _gather_1d_array(values: np.ndarray, device: torch.device) -> np.ndarray:
    if not _is_dist():
        return values
    tensor = torch.tensor(values, device=device, dtype=torch.float32)
    length = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    world_size = dist.get_world_size()
    lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(lengths, length)
    lengths = [int(x.item()) for x in lengths]
    max_len = max(lengths)
    if tensor.numel() < max_len:
        pad = torch.zeros(max_len - tensor.numel(), device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    gathered = [torch.zeros(max_len, device=device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    out = []
    for g, l in zip(gathered, lengths):
        out.append(g[:l].detach().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.array([], dtype=np.float32)


def _run_epoch(
    model,
    feature_encoder,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    use_physchem: bool,
    log_target: bool,
    scaler,
    train_mode: bool,
    grad_accum_steps: int = 1,
    eval_in_log_space_for_log_target: bool = True,
) -> Dict[str, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    y_true_raw_all = []
    y_pred_raw_all = []
    y_true_trans_all = []
    y_pred_trans_all = []

    sampler = getattr(loader, "sampler", None)
    if train_mode and sampler is not None and hasattr(sampler, "set_epoch"):
        # 由外层 epoch 控制 set_epoch，这里不设置
        pass

    pbar = tqdm(loader, desc="train" if train_mode else "eval", leave=False, disable=not _is_main_process())
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(pbar, start=1):
        labels_raw = batch["label_raw"].to(device)
        labels_trans = transform_targets(labels_raw, log_target=log_target)

        with torch.set_grad_enabled(train_mode):
            feats = feature_encoder.encode_batch(
                sequences=batch["sequence"],
                smiles_list=batch["smiles"],
                use_physchem=use_physchem,
                device=device,
            )
            with torch.amp.autocast("cuda", enabled=(scaler is not None and device.type == "cuda")):
                pred_trans = model(feats)
                loss = F.mse_loss(pred_trans, labels_trans)

            if train_mode:
                backward_loss = loss / max(1, grad_accum_steps)
                if scaler is not None:
                    scaler.scale(backward_loss).backward()
                    if step_idx % grad_accum_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    backward_loss.backward()
                    if step_idx % grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * labels_raw.shape[0]
        pred_raw = inverse_transform_targets(pred_trans.detach().cpu().numpy(), log_target=log_target)
        y_true_raw_all.extend(labels_raw.detach().cpu().numpy().tolist())
        y_pred_raw_all.extend(pred_raw.tolist())
        y_true_trans_all.extend(labels_trans.detach().cpu().numpy().tolist())
        y_pred_trans_all.extend(pred_trans.detach().cpu().numpy().tolist())

    if train_mode and (len(loader) % max(1, grad_accum_steps) != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    y_true_arr = np.array(y_true_raw_all, dtype=np.float32)
    y_pred_arr = np.array(y_pred_raw_all, dtype=np.float32)
    y_true_trans_arr = np.array(y_true_trans_all, dtype=np.float32)
    y_pred_trans_arr = np.array(y_pred_trans_all, dtype=np.float32)
    y_true_global = _gather_1d_array(y_true_arr, device=device)
    y_pred_global = _gather_1d_array(y_pred_arr, device=device)
    y_true_trans_global = _gather_1d_array(y_true_trans_arr, device=device)
    y_pred_trans_global = _gather_1d_array(y_pred_trans_arr, device=device)

    total_loss_sum = _all_reduce_sum(running_loss, device=device)
    total_count = _all_reduce_sum(float(len(y_true_raw_all)), device=device)

    raw_metrics = regression_metrics(y_true_global, y_pred_global)
    trans_metrics = regression_metrics(y_true_trans_global, y_pred_trans_global)
    if log_target and eval_in_log_space_for_log_target:
        metrics = {
            "rmse": trans_metrics["rmse"],
            "pcc": trans_metrics["pcc"],
            "scc": trans_metrics["scc"],
        }
    else:
        metrics = {
            "rmse": raw_metrics["rmse"],
            "pcc": raw_metrics["pcc"],
            "scc": raw_metrics["scc"],
        }
    metrics["raw"] = raw_metrics
    metrics["transformed"] = trans_metrics
    metrics["loss"] = total_loss_sum / max(total_count, 1.0)
    return metrics


def train_and_evaluate_one_fold(
    model,
    feature_encoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
    use_physchem: bool,
    log_target: bool,
    mixed_precision: bool,
    grad_accum_steps: int = 1,
    eval_in_log_space_for_log_target: bool = True,
) -> TrainResult:
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    if _is_dist():
        dist.barrier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision and device.type == "cuda"))

    best_epoch = -1
    best_val_rmse = float("inf")
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(1, max_epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            model=model,
            feature_encoder=feature_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_physchem=use_physchem,
            log_target=log_target,
            scaler=scaler,
            train_mode=True,
            grad_accum_steps=grad_accum_steps,
            eval_in_log_space_for_log_target=eval_in_log_space_for_log_target,
        )
        val_metrics = _run_epoch(
            model=model,
            feature_encoder=feature_encoder,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            use_physchem=use_physchem,
            log_target=log_target,
            scaler=None,
            train_mode=False,
            eval_in_log_space_for_log_target=eval_in_log_space_for_log_target,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "val_pcc": val_metrics["pcc"],
            "val_scc": val_metrics["scc"],
        }
        history.append(row)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            if _is_main_process():
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state, best_ckpt_path)
        else:
            bad_epochs += 1

        if _is_main_process():
            print(
                f"[epoch {epoch}] train_rmse={train_metrics['rmse']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} best_val_rmse={best_val_rmse:.6f}"
            )

        if bad_epochs >= patience:
            break

    if _is_dist():
        dist.barrier()
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics = _run_epoch(
        model=model,
        feature_encoder=feature_encoder,
        loader=test_loader,
        optimizer=optimizer,
        device=device,
        use_physchem=use_physchem,
        log_target=log_target,
        scaler=None,
        train_mode=False,
        eval_in_log_space_for_log_target=eval_in_log_space_for_log_target,
    )

    if _is_main_process():
        payload = {
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "test_metrics": test_metrics,
            "history": history,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return TrainResult(
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
        test_metrics=test_metrics,
        history=history,
    )

