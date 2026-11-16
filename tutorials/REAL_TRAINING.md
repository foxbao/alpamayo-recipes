# 真实训练路线

这份文档把仓库里的三个真实训练 recipe 串起来：

| 路线 | 目录 | 主要目标 | 训练方法 |
| --- | --- | --- | --- |
| Alpamayo 1 SFT | `recipes/alpamayo1_sft` | CoC / 轨迹 SFT | next-token cross-entropy，随后可训练 action expert |
| Alpamayo 1.5 SFT | `recipes/alpamayo1_5_sft` | 导航轨迹、LingoQA、action expert | Stage 1 VLM SFT + Stage 2 flow matching |
| Alpamayo 1.x RL | `recipes/alpamayo1_x_rl` | 用驾驶效果优化 VLM | Cosmos-RL + GRPO，奖励来自 rollout |

它是**研究和准备真实训练的操作手册**，不是保证任意机器都能完成训练的脚本。特别是，仓库官方的真实训练配置面向 80 GB 级 GPU；当前这台机器的 8 张 RTX 4090 每张约 24 GB，不能把官方 10B SFT/RL 配置当作已验证可运行的任务。

## 0. 先判断机器

官方 README 中的参考环境是：

- SFT：8 × H100 80 GB；
- RL local test：至少约 5 张 80 GB GPU，示例使用单节点 8 × A100/H100 80 GB。

当前机器是 8 × RTX 4090（约 24 GB/卡）。因此建议这样分工：

| 当前机器可以做 | 当前机器不应直接承诺 |
| --- | --- |
| 阅读和修改 dataset / processor / reward | 按原配置完成 10B Stage 1/Stage 2 全训练 |
| 下载并检查小规模数据 | 按官方配置完成 Cosmos-RL local test |
| 转换模型、解析 Hydra/TOML 配置 | 仅靠修改 `nproc_per_node` 就解决显存问题 |
| 运行 toy tutorials 和静态检查 | 把 `qwen_vl` 环境当成真实 recipe 环境 |

10B 模型的 bf16 权重、梯度和 Adam 状态本身就可能超过单卡 24 GB；视觉输入、激活和临时 buffer 还没有计算在内。ZeRO-3、CPU offload 或量化可以作为研究性改造，但不是仓库已经验证的标准路径。

磁盘空间会随任务变化。开始下载模型或数据前，先用 `df -h / /mnt/disk1` 检查实际余量；本指南编写时根盘约剩 55 GB，`/mnt/disk1` 约剩 230 GB。若只是准备环境和小数据，建议先把缓存放到后者：

```bash
export REAL_TRAINING_ROOT=/mnt/disk1/alpamayo-training
export UV_CACHE_DIR="$REAL_TRAINING_ROOT/.cache/uv"
export HF_HOME="$REAL_TRAINING_ROOT/.cache/huggingface"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
```

reasoning mini 数据的 16 clips 可能接近 85--90 GB；下载前要给模型和 checkpoint 预留空间，不要在没有确认磁盘余量时执行大规模下载。

## 1. 推荐学习顺序

不要一开始同时安装三个环境。推荐顺序是：

1. 先完成 `alpamayo1_5_sft` 的配置阅读和数据检查；
2. 在 80 GB 级节点上跑 1.5 的 Stage 1 nav smoke test；
3. 再跑 Stage 2，理解 VLM 冻结和 action expert；
4. 需要研究 CoC 时，再切到 `alpamayo1_sft` 的 reasoning 数据和 `coc.yaml`；
5. 最后安装 `alpamayo1_x_rl`，先跑 motion reward，再跑 joint reasoning reward。

这里的 smoke test 是为了验证数据、processor、模型加载和训练链路，不等于发布模型所用的完整训练集或训练超参数。

## 2. 安装真实环境

真实 recipe 使用 `uv`，不是 `alpamayo-tutorials` conda 环境。后者只用于 CPU toy 教程。

### Alpamayo 1.5 SFT

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_5_sft
uv venv a1_5_sft
source a1_5_sft/bin/activate
uv sync --active
```

`pyproject.toml` 固定了 Python 3.12、PyTorch 2.8、Transformers 4.57.1、DeepSpeed 0.19.1 和 flash-attn 等依赖。

### Alpamayo 1 SFT / CoC

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_sft
uv venv a1_sft
source a1_sft/bin/activate
uv sync --active
```

### Alpamayo 1.x RL

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_x_rl
uv venv a1x_rl
source a1x_rl/bin/activate
uv sync --active --no-install-package flash-attn
uv sync --active
```

RL 还依赖 Cosmos-RL、vLLM 0.11.0、Transformers 4.57.1 和 flash-attn。flash-attn 的编译必须和当前 PyTorch/CUDA 环境匹配，遇到编译错误时先确认 GPU 节点、驱动和 CUDA，而不是把错误归因于训练代码。

每条 recipe 都有独立的 `pyproject.toml`。不要在 `alpamayo-tutorials` 或已有的 Python 3.10 `qwen_vl` 环境中直接启动真实训练。

## 3. 模型准备

### Alpamayo 1.5 SFT

先从 Hugging Face 下载发布 checkpoint，然后转换为 Stage 1 使用的 A1 格式：

```bash
hf auth login

hf download nvidia/Alpamayo-1.5-10B \
  --local-dir /path/to/Alpamayo-1.5-10B

cd /path/to/alpamayo-recipes
python scripts/convert_checkpoint.py to-a1 \
  --input /path/to/Alpamayo-1.5-10B \
  --output /path/to/Alpamayo-1.5-10B-A1-format
```

Stage 1 的 `model.checkpoint_path` 指向转换后的目录。这里不是从 Qwen3-VL 重新训练；Qwen3-VL 主要用于构造兼容的 VLM processor/架构。

### Alpamayo 1 SFT / CoC

Alpamayo 1 的 A1 checkpoint 可以直接用于这个 recipe，不需要上面的 `to-a1` 转换：

```bash
hf download nvidia/Alpamayo-R1-10B \
  --local-dir /path/to/Alpamayo-R1-10B
```

### RL

RL 需要把 release checkpoint 转换成 Cosmos-RL 的训练格式：

```bash
cd /path/to/alpamayo-recipes
python scripts/convert_release_config_to_training.py \
  --output-dir /path/to/alpamayo_model_converted_from_hf
```

默认转换 Alpamayo 1.5；研究 Alpamayo 1 时显式指定：

```bash
python scripts/convert_release_config_to_training.py \
  --alpamayo-model nvidia/Alpamayo-R1-10B \
  --output-dir /path/to/alpamayo_model_converted_from_hf
```

不要把 Cosmos-RL 产生的 DTensor 分片直接交给 `from_pretrained()`；训练结束后需要使用 `scripts/convert_cosmos_rl_checkpoint.py` 导出标准 Hugging Face checkpoint，见第 8 节。

## 4. 训练数据

### 4.1 PAI 导航数据（Alpamayo 1.5 SFT）

Stage 1 nav 和 Stage 2 nav 使用 Physical AI Autonomous Vehicles（PAI）数据，主要需要：

- 相机：`camera_front_wide_120fov`、`camera_cross_left_120fov`、`camera_cross_right_120fov`、`camera_front_tele_30fov`；
- 标定：`camera_intrinsics`、`sensor_extrinsics`；
- 标签：`egomotion`；
- 导航 annotation：`clip_id`、`t0_relative`、`nav_text`，可选 `cot`。

仓库中的 nav 示例 annotation 来自 Alpamayo 1.5 仓库的 `notebooks/nav_demo_samples.json`。其中列出的 20 个 chunk 是 smoke/overfit 数据，不是发布模型的完整训练集。

```bash
export HF_TOKEN=<your-token>
cd /path/to/alpamayo-recipes

python scripts/download_pai.py \
  --chunk-ids "214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir /path/to/pai_dataset
```

`--chunk-ids` 是一个字符串参数，多个 chunk 要放在同一对引号中。训练前应确认 `local_dir` 下确实有相机、标定和标签文件，并且 annotation 的 `clip_id` 能在 `clip_index.parquet` 中找到。

### 4.2 LingoQA

LingoQA 用于 1.5 的 Stage 1 VQA：

```text
LingoQA/
├── train.parquet
└── images/train/
```

`train.parquet` 约有 148k 个 QA pair、约 3.5k 个 video segment；每个 segment 使用 5 张前视 JPEG。真实评估的 `val.parquet` 来自单独的 evaluation 下载目录。只有 train 数据时可以临时把验证集也指向 `train.parquet`，但那不能作为真实泛化评估。

### 4.3 CoC reasoning 数据

CoC 是 Alpamayo 1 recipe 中的可选 SFT 变体。它需要：

```text
/path/to/pai_reasoning/reasoning/ood_reasoning.parquet
/path/to/pai_reasoning/clip_index_reasoning_mini.parquet
```

下载一个小的 reasoning 子集：

```bash
cd /path/to/alpamayo-recipes
python scripts/download_pai.py \
  --only-reasoning-chunks \
  --num-reasoning-clips 16 \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --reasoning ood_reasoning.parquet \
  --output-dir /path/to/pai_reasoning
```

CoC 训练同时要求 processor 的 `components_order`、`components_prompt`、`label_components` 都包含 `cot`。对应配置见 `recipes/alpamayo1_sft/configs/vla_processor/coc.yaml`。如果模板打开了 `cot`，但数据没有 reasoning metadata，通常会得到 `cot not found in data`；如果仍使用默认 keyframe，可能找不到事件时间戳。

## 5. Alpamayo 1.5 SFT 命令

### Stage 1 nav：先做这个 smoke test

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_5_sft

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_nav \
  model.checkpoint_path=/path/to/Alpamayo-1.5-10B-A1-format \
  data.train_dataset.local_dir=/path/to/pai_dataset \
  data.val_dataset.local_dir=/path/to/pai_dataset \
  data.train_dataset.annotations_path=/path/to/nav_demo_samples.json \
  data.val_dataset.annotations_path=/path/to/nav_demo_samples.json
```

Stage 1 训练 VLM，监督目标是离散的 `traj_future` token，损失是 next-token cross-entropy。上面使用的 20 条 annotation 适合验证链路和观察 loss 下降，不代表完整训练规模。

### Stage 2 nav：训练连续 action expert

Stage 2 需要 Stage 1 的 Trainer 输出目录，并且目录中有 `model.safetensors.index.json` 及其 shard：

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_5_sft

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  model.pretrained_model_name_or_path=/path/to/Alpamayo-1.5-10B-A1-format \
  model.stage1_vlm_checkpoint_path=/path/to/output_stage1_nav/checkpoint-xxxx \
  data.train_dataset.local_dir=/path/to/pai_dataset \
  data.val_dataset.local_dir=/path/to/pai_dataset \
  data.train_dataset.annotations_path=/path/to/nav_demo_samples.json \
  data.val_dataset.annotations_path=/path/to/nav_demo_samples.json
```

Stage 2 冻结 Stage 1 VLM，使用同一类 PAI 数据训练连续 action expert 的 flow-matching 目标。它不是再做一次 token CE。

### LingoQA Stage 1

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_5_sft

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_lingoqa \
  model.checkpoint_path=/path/to/Alpamayo-1.5-10B-A1-format \
  data.train_dataset.data_root=/path/to/LingoQA \
  data.val_dataset.data_root=/path/to/LingoQA \
  data.val_dataset.parquet_name=train.parquet
```

最后一行只是没有独立 validation parquet 时的 smoke 配置；真实评估应使用独立的 validation/evaluation 数据。

## 6. Alpamayo 1 CoC SFT 命令

CoC 不是特殊优化器，而是普通 SFT：`cot` 和 `traj_future` 都作为 assistant 侧 label，逐 token 计算 cross-entropy。完整 processor 和 metadata 配置需要手动对齐：

```bash
cd /path/to/alpamayo-recipes/recipes/alpamayo1_sft

torchrun --nproc_per_node 8 \
  -m alpamayo1_sft.train_hf \
  --config-path pkg://alpamayo1_sft/configs \
  --config-name sft_stage1_coc \
  model.checkpoint_path=/path/to/Alpamayo-R1-10B \
  data.train_dataset.local_dir=/path/to/pai_reasoning \
  data.val_dataset.local_dir=/path/to/pai_reasoning \
  +data.train_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet \
  +data.val_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet \
  +data.train_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet \
  +data.val_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false
```

这里的 `+` 是 Hydra 语法：只有当字段没有在 YAML 中预先声明时才使用。`reasoning_metadata` 和 `clip_index_metadata` 相对于 dataset `local_dir`。

Alpamayo 1.5 当前公开的 `nav.yaml` 不含 CoC；不要仅把 1.5 的 nav 命令改名为 CoC 就认为它会训练 reasoning。

## 7. RL：motion 与 joint reasoning

RL 的流程是：policy 用 FSDP 训练 VLM，rollout 用 vLLM 采样，controller 汇总 reward 和 advantage，再同步权重。RL 训练的是离散 trajectory-token 路径，不训练连续 action expert。

先设置路径：

```bash
export ALPAMAYO_WORKSPACE=/path/to/alpamayo-recipes
export ALPAMAYO_MODEL_DIR=/path/to/alpamayo_model_converted_from_hf
export ALPAMAYO_PAI_LOCAL_DIR=/path/to/PAI_mini
export ALPAMAYO_LOG_DIR=/path/to/alpamayo_cosmos_rl_job/logs
```

### Motion reward 数据和命令

```bash
cd "$ALPAMAYO_WORKSPACE"
python scripts/download_pai.py \
  --chunk-ids "3116" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"

python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk 3116 \
  --num-samples 16 \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index_mini.parquet"
```

编辑 `recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test.toml`：

```toml
[train]
output_dir = "/path/to/output"

[policy]
model_name_or_path = "/path/to/alpamayo_model_converted_from_hf"

[policy.parallelism]
dp_shard_size = 4
```

然后启动：

```bash
cd "$ALPAMAYO_WORKSPACE"
cosmos-rl \
  --config recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

这个 local test 的官方硬件前提仍然是 80 GB 级 GPU；`--policy 1 --rollout 1` 不会把 10B 模型变成小模型。

### Joint reasoning reward

下载 reasoning 数据，并额外准备 Lingo-Judge：

```bash
python scripts/download_pai.py \
  --only-reasoning-chunks \
  --num-reasoning-clips 16 \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion egomotion.offline obstacle.offline \
  --reasoning ood_reasoning.parquet \
  --output-dir /path/to/PAI_Reasoning_mini

hf download wayveai/Lingo-Judge \
  --local-dir /path/to/lingo_judge_model
```

编辑 `alpamayo_rvla_rl_local_test_with_reasoning.toml` 中的 reasoning grader 路径和 reward 权重，然后启动：

```bash
cosmos-rl \
  --config recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test_with_reasoning.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py
```

这里的 reasoning 是**整段 rollout 完成后的 reward**，不是 CoC SFT 那种 token-level label。reasoning score 的阈值归一化定义在 `recipes/alpamayo1_x_rl/rewards/aggregated_reward_with_reasoning.py`。

## 8. 导出 RL checkpoint

Cosmos-RL checkpoint 是按 rank 保存的 DTensor 分片，不能直接作为普通 Hugging Face 模型加载：

```bash
cd "$ALPAMAYO_WORKSPACE"
python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt /path/to/output/checkpoints/step_<N>/policy \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir /path/to/exported_model
```

导出后再用对应 recipe 的 inference/evaluation 代码加载。

## 9. 低风险验证清单

在申请 80 GB GPU 节点或开始大规模下载前，先做这些不占大显存的检查：

```bash
conda run --no-capture-output -n alpamayo-tutorials \
  python -m compileall -q tutorials recipes tests

git diff --check
```

在真实 `uv` 环境中，再按以下顺序检查：

1. `python -c "import torch; print(torch.__version__, torch.cuda.device_count())"`；
2. 用 `hf auth whoami` 确认 Hugging Face 权限；
3. 检查模型目录是否包含 `config.json` 和 safetensors；
4. 检查 PAI 的 `clip_index.parquet`、相机、标定、labels 是否齐全；
5. 用 `python scripts/curate_pai_samples.py` 生成 16 条 mini index；
6. 只解析 Hydra/TOML 配置，不启动训练，确认所有路径已替换；
7. 在目标 GPU 节点上先跑一个短的 nav smoke test，再决定是否下载完整数据。

当前仓库的 toy 环境没有安装 pytest，因此不要把 `compileall` 结果称为完整单元测试；reasoning reward 的数值 sanity check 可以单独运行，完整 recipe 仍需真实依赖和 GPU 才能验证。

## 10. 一句话理解三种训练

```text
Stage 1 SFT:  真值轨迹 -> 离散 token -> next-token CE -> 训练 VLM
Stage 2 SFT:  真值轨迹 -> 连续 action -> flow-matching loss -> 训练 expert
RL:           模型 rollout -> ADE/comfort/(reasoning) reward -> GRPO -> 训练 VLM
CoC SFT:      真值 cot + 真值轨迹 -> token labels -> 普通 next-token CE
```

先把这四行和每个 recipe 的配置对应起来，再阅读完整训练代码，会比直接从 `cosmos-rl` 入口跳进去容易很多。
