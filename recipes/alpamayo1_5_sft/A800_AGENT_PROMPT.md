# A800 上启动 VLA-KL CoC 训练的说明

请先阅读并按照以下文档操作：

1. `recipes/alpamayo1_5_sft/SKILL.md`
2. `recipes/alpamayo1_5_sft/LOCAL_COC_TRAINING.md`
3. `recipes/alpamayo1_5_sft/README.md`
4. `data/vla_kl/README.md`
5. `recipes/alpamayo1_5_sft/scripts/run_local_coc_sft.sh`

当前 `alpamayo-recipes` 已同步到 commit：

```text
6457ee417327290a6e1742dcebcef12b426c9c9a
```

本次任务是使用 VLA-KL 数据进行 Alpamayo-1.5 Stage-1 CoC SFT 训练。不要重新下载数据或模型，不要修改原始 checkpoint，不要改动 Alpamayo 源代码。

训练项目目录：

```text
/home/baojiali/Downloads/public_code/alpamayo-recipes
```

Recipe 目录：

```text
/home/baojiali/Downloads/public_code/alpamayo-recipes/recipes/alpamayo1_5_sft
```

数据入口已经配置好：

```text
data/vla_kl/sample       -> /cloud/cloud-ssd2/data/vla_kl/sample
data/vla_kl/train_labels -> /cloud/cloud-ssd2/data/vla_kl/train_labels
data/vla_kl/val_labels   -> /cloud/cloud-ssd2/data/vla_kl/val_labels
data/vla_kl/checkpoint    -> /cloud/cloud-ssd2/data/models/Alpamayo-1.5-10B-A1-format
```

数据包含 74 条记录：

- `20261019_lin_007_out`：4 条
- `20261105_008_out`：58 条
- `20261111_BB_008_out`：12 条

当前正式标签：

- train：9 条
- validation：24 条

训练环境：

```text
/home/baojiali/Downloads/public_code/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft
```

环境已验证：

- Python 3.12.15
- PyTorch 2.8.0 + CUDA 12.8
- flash-attn 2.8.3
- CUDA 可用
- 依赖检查通过
- A800 当前只显示 1 张 GPU，因此使用 `NPROC_PER_NODE=1`

请先执行以下检查：

```bash
cd /home/baojiali/Downloads/public_code/alpamayo-recipes/recipes/alpamayo1_5_sft
source a1_5_sft/bin/activate

python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__, torch.cuda.is_available())"

uv pip check --python ./a1_5_sft/bin/python
```

然后先进行 10 步 smoke test，不要直接开始完整训练：

```bash
export LOCAL_COC_SAMPLE_ROOT="$PWD/data/vla_kl/sample"
export LOCAL_COC_TRAIN_VERSIONS="$PWD/data/vla_kl/train_labels"
export LOCAL_COC_VAL_VERSIONS="$PWD/data/vla_kl/val_labels"
export ALPAMAYO_A1_CHECKPOINT="$PWD/data/vla_kl/checkpoint"
export NPROC_PER_NODE=1
export LOCAL_COC_OUTPUT_DIR=/cloud/cloud-ssd2/workspace/vla_sft/outputs/smoke_local_coc

LOCAL_COC_MAX_STEPS=10 ./scripts/run_local_coc_sft.sh
```

请重点检查：

1. 数据是否能正常读取；
2. front/rear 图片和 localization 是否能加载；
3. checkpoint 是否使用 `flash_attention_2`；
4. 多卡启动参数是否正确；
5. loss、显存和输出 checkpoint 是否正常；
6. 是否出现路径、标签 manifest 或数据格式错误。

smoke test 成功后，再汇报训练日志、显存使用、输出路径和是否适合启动完整训练。完整训练暂时不要覆盖 smoke test 输出目录。
