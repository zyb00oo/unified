import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.enzyme_unified.config import TASK_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Run Table 2 experiments across folds and variants.")
    parser.add_argument("--variants", nargs="+", default=["hybrid", "hybrid_prostt5", "hybrid_pp"])
    parser.add_argument("--tasks", nargs="+", default=["kcat_km", "ph", "topt"])
    parser.add_argument("--split_strategy", choices=["modulo1", "random90_10"], default="modulo1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--freeze_encoders", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=-1,
        help=">0 时固定值；<=0 时按任务默认全局batch自动计算。",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Per-process batch size.")
    parser.add_argument("--output_root", type=str, default="results/table2")
    parser.add_argument("--python_exec", type=str, default="python")
    parser.add_argument("--launcher", choices=["python", "torchrun"], default="python")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    return parser.parse_args()


def summarize_fold_metrics(task_dir: Path, folds: int) -> dict:
    rows = []
    for fold in range(folds):
        metrics_path = task_dir / f"fold_{fold}" / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        test_metrics = payload["test_metrics"]
        rows.append(
            {
                "fold": fold,
                "rmse": test_metrics["rmse"],
                "pcc": test_metrics["pcc"],
                "scc": test_metrics["scc"],
            }
        )
    df = pd.DataFrame(rows)
    return {
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std": float(df["rmse"].std(ddof=0)),
        "pcc_mean": float(df["pcc"].mean()),
        "pcc_std": float(df["pcc"].std(ddof=0)),
        "scc_mean": float(df["scc"].mean()),
        "scc_std": float(df["scc"].std(ddof=0)),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for task in args.tasks:
        task_cfg = TASK_CONFIGS[task]
        folds = task_cfg["folds"]
        for variant in args.variants:
            variant_task_dir = output_root / task / variant
            variant_task_dir.mkdir(parents=True, exist_ok=True)
            for fold in range(folds):
                fold_dir = variant_task_dir / f"fold_{fold}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                if args.launcher == "torchrun":
                    cmd = [
                        "torchrun",
                        "--nproc_per_node",
                        str(args.nproc_per_node),
                        "train_unified.py",
                    ]
                else:
                    cmd = [args.python_exec, "train_unified.py"]

                if args.grad_accum_steps > 0:
                    grad_accum_steps = args.grad_accum_steps
                else:
                    per_proc_bs = args.batch_size if args.batch_size is not None else task_cfg["default_batch_size"]
                    global_bs = task_cfg["default_batch_size"]
                    world = args.nproc_per_node if args.launcher == "torchrun" else 1
                    grad_accum_steps = max(1, global_bs // max(1, per_proc_bs * world))

                cmd += [
                    "--task",
                    task,
                    "--variant",
                    variant,
                    "--test_fold",
                    str(fold),
                    "--split_strategy",
                    args.split_strategy,
                    "--seed",
                    str(args.seed),
                    "--max_epochs",
                    str(args.max_epochs),
                    "--patience",
                    str(args.patience),
                    "--grad_accum_steps",
                    str(grad_accum_steps),
                    "--output_dir",
                    str(fold_dir),
                ]
                if args.batch_size is not None:
                    cmd += ["--batch_size", str(args.batch_size)]
                if args.freeze_encoders:
                    cmd.append("--freeze_encoders")
                if args.mixed_precision:
                    cmd.append("--mixed_precision")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)

            metrics = summarize_fold_metrics(variant_task_dir, folds=folds)
            summary_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["task", "variant"]).reset_index(drop=True)
    summary_csv = output_root / "table2_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[DONE] summary saved to {summary_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()

