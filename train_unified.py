import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.enzyme_unified.config import TASK_CONFIGS, VARIANTS
from src.enzyme_unified.dataset import EnzymeDataset, build_fold_split, collate_samples, load_task_dataframe
from src.enzyme_unified.features import FeatureEncoder
from src.enzyme_unified.model import EnzymeUnifiedModel
from src.enzyme_unified.trainer import train_and_evaluate_one_fold
from src.enzyme_unified.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Enzyme-Unified on one fold.")
    parser.add_argument("--task", choices=list(TASK_CONFIGS.keys()), required=True)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="hybrid")
    parser.add_argument("--csv", type=str, default=None, help="Override CSV path.")
    parser.add_argument("--test_fold", type=int, required=True)
    parser.add_argument("--split_strategy", choices=["modulo1", "random90_10"], default="modulo1")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument(
        "--eval_raw_for_log_target",
        action="store_true",
        help="若设置，则在 log-target 任务上也用原尺度指标；默认用 log 空间指标（更贴近论文表格量级）。",
    )
    parser.add_argument("--freeze_encoders", action="store_true")
    parser.add_argument("--protein_model_name", type=str, default="/mnt/data/oyangcan/prot_t5_xl_uniref50")
    parser.add_argument("--prostt5_model_name", type=str, default="/mnt/data/oyangcan/ProstT5")
    parser.add_argument("--substrate_model_name", type=str, default="/mnt/data/oyangcan/molt5-base-smiles2caption")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--cross_layers", type=int, default=1)
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--max_smiles_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def main():
    args = parse_args()
    set_seed(args.seed)
    distributed, rank, local_rank, world_size, device = setup_distributed()

    task_cfg = TASK_CONFIGS[args.task]
    variant_cfg = VARIANTS[args.variant]
    use_physchem = variant_cfg["use_physchem"]

    csv_path = args.csv or task_cfg["csv_path"]
    lr = args.lr if args.lr is not None else task_cfg["default_lr"]
    batch_size = args.batch_size if args.batch_size is not None else task_cfg["default_batch_size"]
    log_target = task_cfg["log_target"]
    eval_in_log_space_for_log_target = not args.eval_raw_for_log_target

    df = load_task_dataframe(csv_path=csv_path, label_col=task_cfg["label_col"], log_target=log_target)
    split = build_fold_split(
        df=df,
        total_folds=task_cfg["folds"],
        test_fold=args.test_fold,
        strategy=args.split_strategy,
        seed=args.seed,
    )

    train_ds = EnzymeDataset(split.train_df, label_col=task_cfg["label_col"])
    val_ds = EnzymeDataset(split.val_df, label_col=task_cfg["label_col"])
    test_ds = EnzymeDataset(split.test_df, label_col=task_cfg["label_col"])

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        pin_memory=True,
    )

    protein_model_name = args.prostt5_model_name if variant_cfg["use_prostt5"] else args.protein_model_name
    feature_encoder = FeatureEncoder(
        protein_model_name=protein_model_name,
        substrate_model_name=args.substrate_model_name,
        use_prostt5=variant_cfg["use_prostt5"],
        freeze_encoders=args.freeze_encoders,
        max_protein_length=args.max_protein_length,
        max_smiles_length=args.max_smiles_length,
    ).to(device)

    protein_dim = feature_encoder.protein_model.config.hidden_size
    substrate_dim = feature_encoder.substrate_model.config.hidden_size

    model = EnzymeUnifiedModel(
        protein_dim=protein_dim,
        substrate_dim=substrate_dim,
        maccs_dim=167,
        physchem_dim=22,
        hidden_dim=args.hidden_dim,
        num_heads=args.cross_heads,
        cross_layers=args.cross_layers,
        dropout=args.dropout,
        use_physchem=use_physchem,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        config_payload = dict(vars(args))
        config_payload["world_size"] = world_size
        config_payload["ddp_enabled"] = distributed
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(config_payload, f, ensure_ascii=False, indent=2)
    if distributed:
        dist.barrier()

    result = train_and_evaluate_one_fold(
        model=model,
        feature_encoder=feature_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        device=device,
        lr=lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        use_physchem=use_physchem,
        log_target=log_target,
        mixed_precision=args.mixed_precision,
        grad_accum_steps=args.grad_accum_steps,
        eval_in_log_space_for_log_target=eval_in_log_space_for_log_target,
    )
    if is_main_process(rank):
        print(
            f"[DONE] task={args.task} variant={args.variant} fold={args.test_fold} "
            f"test_rmse={result.test_metrics['rmse']:.6f} "
            f"test_pcc={result.test_metrics['pcc']:.6f} "
            f"test_scc={result.test_metrics['scc']:.6f}"
        )

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

