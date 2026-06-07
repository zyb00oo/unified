# Enzyme-Unified Table2 Reproduction

本项目实现了论文 **Enzyme-Unified** 在已处理数据上的模型与训练流程，目标是复现 **Table 2**（`kcat_km`、`ph`、`topt` 三个任务）。

## 1. 目录说明

- `dataset/`: 已处理好的 CSV 数据（你已提供）
- `src/enzyme_unified/`: 模型、特征、训练逻辑
- `train_unified.py`: 单任务单折训练入口
- `run_table2.py`: 按任务 × 变体 × 折批量运行并汇总结果
- `results/`: 输出目录（checkpoint、metrics、table2_summary）

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 2 卡单折训练（推荐起步）

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node=2 train_unified.py \
  --task kcat_km \
  --variant hybrid \
  --test_fold 0 \
  --split_strategy modulo1 \
  --batch_size 8 \
  --grad_accum_steps 32 \
  --freeze_encoders \
  --mixed_precision \
  --output_dir results/debug/kcat_km_hybrid_fold0
```

## 4. 批量跑 Table2

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km ph topt \
  --variants hybrid hybrid_prostt5 hybrid_pp \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks kcat_km \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/kcat_km
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks ph \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/ph
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=1 \
python run_table2.py \
  --tasks topt \
  --variants hybrid \
  --launcher torchrun \
  --nproc_per_node 2 \
  --batch_size 8 \
  --grad_accum_steps -1 \
  --split_strategy modulo1 \
  --freeze_encoders \
  --mixed_precision \
  --output_root results/table2/topt

```

输出文件：

- 每折指标：`results/table2/<task>/<variant>/fold_<k>/metrics.json`
- 汇总表：`results/table2/table2_summary.csv`

## 5. 与论文对齐的关键设置

- 任务级配置：`src/enzyme_unified/config.py`
  - `kcat_km`: 10 折，`log10` 标签变换，默认 `lr=1e-5`
  - `ph`: 5 折，默认 `lr=5e-4`
  - `topt`: 5 折，默认 `lr=1e-3`
- 模型：
  - 双路径（Cross-Attention + Global Concat）
  - 门控融合：`sigmoid(alpha) * y_attn + (1-sigmoid(alpha)) * y_concat`
- 指标：
  - `kcat_km` 训练在 `log10(y)` 空间；默认在 **log 空间**汇报指标（更贴近论文量级）
  - pH / `topt` 在原尺度汇报指标

## 6. 复现一致性建议

- 固定随机种子：`--seed`
- 固定划分策略：`--split_strategy`
- 固定预训练模型版本（`train_unified.py` 参数）
- 记录运行配置：每次运行会在输出目录写入 `run_config.json`

## 7. 注意事项

- ProtT5 / ProstT5 / MolT5 体积较大，请保证显存与磁盘缓存空间。
- 如果结果与论文有偏差，优先检查：
  1. 使用的预训练模型版本是否一致；
  2. 变体设置（hybrid / hybrid_prostt5 / hybrid_pp）是否对应；
  3. 早停策略与 batch size 是否与论文设定一致。

