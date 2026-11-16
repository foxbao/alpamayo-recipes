# 本地 CoC 数据的 Alpamayo-1.5 训练

本文说明如何在训练服务器上启动 `LocalRecordDataset` 的 Stage 1 SFT。模型会同时学习 CoC 文本和未来轨迹 token；这不是 CoC-only 训练，也不是 Stage 2 的连续轨迹 diffusion expert 训练。

## 训练前需要准备

训练服务器需要有：

1. `alpamayo-recipes` 仓库中本地 CoC 适配代码（包含 `recipes/alpamayo1_5_sft/datasets/` 和 `configs/sft_stage1_local_coc.yaml`）。
2. Alpamayo-1.5 转换后的 A1-format checkpoint。
3. 原始样本目录，结构为 `<sample_root>/<group_id>/<record_id>/`，其中有 `camera/front_image`、`camera/rear_image` 和 `localization/`。
4. 从 CoC 标注仓库复制过来的正式标签版本目录。每个版本至少要保留 `manifest.json` 和 `annotations.jsonl`；建议完整复制整个 `versions/vNNN/` 目录。
5. 与依赖匹配的 GPU 环境。官方 recipe 使用 `uv`，要求 Python 3.12；请按本目录 README 建立 `a1_5_sft` 环境并执行 `uv sync --active`。

Git 同步只会带来代码，不会带来被 `.gitignore` 排除的本地数据、模型权重或 `outputs/.../formal_labels/` 产物。请通过数据盘、共享存储或受控文件传输把三类文件分别放到服务器上，并核对容量和路径。

如果两台机器可以通过 SSH 互通，可以按实际用户名和主机名复制这两组样本及其正式标签版本。正式标签目录要保留版本目录本身，不要只拷贝 JSONL：

```bash
rsync -a --info=progress2 /mnt/disk3/vla_kl/sample/20261019_lin_007_out/ \
  USER@A800_HOST:/data/vla_kl/sample/20261019_lin_007_out/
rsync -a --info=progress2 /mnt/disk3/vla_kl/sample/20261111_BB_008_out/ \
  USER@A800_HOST:/data/vla_kl/sample/20261111_BB_008_out/

rsync -a --info=progress2 /path/to/formal_labels/versions/v002/ \
  USER@A800_HOST:/data/formal_labels/20261019_lin_007_out/policy_v010/v002/
rsync -a --info=progress2 /path/to/formal_labels/versions/v001/ \
  USER@A800_HOST:/data/formal_labels/20261111_BB_008_out/policy_v001/v001/
```

checkpoint 可以由目标服务器单独下载并转换，或从共享模型盘读取；不要通过 Git 或把 checkpoint 放进样本目录来传输。

## 准备 checkpoint

Stage 1 使用 A1-format checkpoint。若服务器上只有原始 `nvidia/Alpamayo-1.5-10B` checkpoint，在 `alpamayo-recipes` 根目录执行：

```bash
python scripts/convert_checkpoint.py to-a1 \
  --input /models/Alpamayo-1.5-10B \
  --output /models/Alpamayo-1.5-10B-A1-format
```

训练时把 `ALPAMAYO_A1_CHECKPOINT` 指向转换后的目录。不要把权重复制到 Git 仓库。

检查 GPU 和系统环境：

```bash
nvidia-smi
ldd --version
```

当前训练配置从 checkpoint 的 `config.json` 读取 attention 实现。若使用 `flash_attention_2`，服务器上的 `flash-attn` 扩展必须能被当前系统和 CUDA/PyTorch 环境加载。此前开发机的 glibc 2.31 无法加载要求 glibc 2.32 的扩展；本次单样本验证通过临时 SDPA 配置副本完成。A800 服务器应使用它自身兼容的环境和 checkpoint 配置，不要依赖开发机 `/tmp` 中的副本。

## 设置数据路径并启动

先进入 recipe 目录并激活环境：

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_5_sft
source a1_5_sft/bin/activate
```

设置数据根目录、训练/验证正式版本和 checkpoint。正式版本可以填写一个或多个路径，多个路径用逗号分隔；路径中不要包含逗号。训练集与验证集必须使用不同的完整 `record` 或数据组，不能把同一记录的相邻事件随机拆开。

例如，当前两组数据可以按组留出：用 `20261019_lin_007_out` 的 9 条训练，用 `20261111_BB_008_out` 的 24 条验证。标签路径须在服务器上实际存在：

```bash
export LOCAL_COC_SAMPLE_ROOT=/data/vla_kl/sample
export LOCAL_COC_TRAIN_VERSIONS=/data/formal_labels/20261019_lin_007_out/policy_v010/v002
export LOCAL_COC_VAL_VERSIONS=/data/formal_labels/20261111_BB_008_out/policy_v001/v001
export ALPAMAYO_A1_CHECKPOINT=/models/Alpamayo-1.5-10B-A1-format
export NPROC_PER_NODE=8
export LOCAL_COC_OUTPUT_DIR=/data/training_runs/local_coc_stage1/run_001
```

先做一个限步训练运行，确认多卡启动、数据读取、保存路径和训练日志：

```bash
LOCAL_COC_MAX_STEPS=10 ./scripts/run_local_coc_sft.sh
```

这 10 步会更新完整 Stage 1 模型，和先前只训练末端归一化层的 5 步内存安全 smoke test 不同。它只用于验证训练入口，不代表训练效果。确认日志、GPU 显存和输出目录都正常后，准备完整训练时不设置 `LOCAL_COC_MAX_STEPS`，脚本会使用配置中的 epoch 数和训练参数：

```bash
./scripts/run_local_coc_sft.sh
```

启动脚本还支持以下可选覆盖：

```bash
export LOCAL_COC_NUM_EPOCHS=3
export LOCAL_COC_DATALOADER_WORKERS=4
```

`NPROC_PER_NODE` 应与本次分配给任务的 GPU 数量一致。若作业系统通过 `CUDA_VISIBLE_DEVICES` 分配 GPU，保持其设置，并令 `NPROC_PER_NODE` 等于可见卡数。不要在 GPU 被其他任务占用时直接启动。

## 输出与验证边界

运行日志会显示样本根目录、输出目录和 train/validation 版本数量。Trainer 的配置、日志和 checkpoint 写入 `LOCAL_COC_OUTPUT_DIR`。每个样本会校验正式标签 manifest/hash、双重人工接受状态，并从原始记录读取 front/rear 图片和 localization。

当前正式标签共 33 条，只适合验证训练链路和探索性小实验。两组数据按组留出后，训练集只有 9 条，不能据此评估泛化，也不应把模型用于生产。正式微调前应继续积累多个独立记录/数据组，并确认定位姿态坐标约定及希望模型学习的倒车、停车行为。

这份配置监督 `cot` 和 `traj_future`。若只想训练 CoC 文本，需要另行修改模型的轨迹 token 融合与 loss 路径；不能通过删除 processor 配置中的轨迹字段来得到 CoC-only 训练。
