# Alpamayo 训练教程（从零理解 SFT）

配合本仓库 `recipes/alpamayo1_5_sft/` 的真实代码，一步步手写并理解 Alpamayo 的训练。

> 前置：建议先完成 `alpamayo1.5/tutorials/` 的**推理教程**（stage0~15），理解 VLA 的架构。
> 本教程讲的是**怎么训练**——那些权重是怎么来的。

---

## ⚠️ 命名说明（避免混淆）

有三个「stage」容易混，本教程统一约定：

| 说法 | 含义 |
|---|---|
| `train1` ~ `trainN` | **本教程的第 N 课**（教学步骤），文件名 `trainN_*.py` |
| **SFT stage1_lingoqa / stage1_nav / stage2** | 真实训练流程的三个阶段（见下面的课程图） |
| `stage0`~`stage15` | `alpamayo1.5/tutorials/` 的推理教程，与本教程无关 |

---

## 训练全景（先建立整体感）

真实训练是**三段课程**，全部在 `recipes/alpamayo1_5_sft/configs/` 里能看出来
（每段一个 yaml）：

```
起点：nvidia/Alpamayo-1.5-10B 的 VLM 权重
      ⚠️ 不是从原始 Qwen3-VL-8B 开始训。`from_alpamayo_checkpoint` 的 docstring：
         "Creates the VLM architecture from config (without loading base weights),
          then loads the fine-tuned vlm.* weights from the Alpamayo checkpoint."
         Qwen/Qwen3-VL-8B-Instruct 只用来建 tokenizer / processor 的架构。
      ⚠️ 所以 recipes 的本质是「在已发布模型上继续训」，不是「从头训一个 Alpamayo」。
   │
   ├─ stage1_lingoqa  (sft_stage1_lingoqa.yaml)   模型：TrainableReasoningVLA
   │    数据: LingoQA（VQA 问答）
   │    目标: 通用「看图说话」——先把视觉语言对齐练好
   │
   ├─ stage1_nav      (sft_stage1_nav.yaml)       模型：TrainableReasoningVLA
   │    数据: PAI nav + 导航标注
   │    目标: 驾驶推理（CoC）+ 未来轨迹的离散 token
   │    损失: next-token 交叉熵（拆成 future_traj / others 两段）
   │    技巧: lr_multiplier {vlm.model.visual: 0.1}  ← 视觉塔只给 1/10 学习率
   │
   └─ stage2_nav      (sft_stage2_nav.yaml)       模型：TrainableAlpamayoR1
        起点: nvidia/Alpamayo-1.5-10B（含 expert）
        注入: stage1_vlm_checkpoint_path ← 把上面 stage1 训出的 VLM 塞进来
        cotrain_vlm: false               ← 冻结 VLM
        目标: 训 action expert（flow matching）
        技巧: deepspeed: null（不用 ZeRO），gradient_checkpointing: false
```

入口：`recipes/alpamayo1_5_sft/train_hf.py`（Hydra 配置）+ `trainer.py`（HF Trainer 子类）。

### Alpamayo 有**两条动作通路**，本教程只讲其中一条

这是读 configs 之前最容易搞错的一件事：

| | 通路 A：**离散轨迹 token** | 通路 B：**连续 action expert** |
|---|---|---|
| 轨迹表示 | VLM 自回归生成 4000 词表里的轨迹 token | flow-matching 去噪出连续动作 `(加速度, 曲率)` |
| 谁在训 | **RL**（`alpamayo1_x_rl`，GRPO） | **SFT 阶段二**（本教程） |
| 发布推理用哪个 | ❌ 被 `MaskDiscreteTrajectoryLogitsProcessor` 屏蔽 | ✅ 用它 |

RL 只训 **VLM backbone**（FAQ 原文："The action expert head ... is **not** trained by
this RL pipeline"），导出的 checkpoint 也只有 VLM 权重；拿回去 SFT 训 expert 时
**expert 是随机初始化的**。所以闭环是：

```
SFT stage1 (VLM) → RL (VLM + 离散轨迹) → SFT stage2 (expert 重新随机初始化)
```

> **RL 是独立的一整套**：`recipes/alpamayo1_x_rl/`（GRPO via Cosmos-RL，
> policy/rollout/controller 异步架构，需要 ≥5×80GB 显存）。本教程的 `train1~8`
> 只讲 SFT；RL 单独开 `rl*` 轨道，且只讲**算法层**（基础设施教学价值低）。

---

## 课程路线

| 课 | 文件 | 学什么 |
|---|---|---|
| 1 | `train1_lm_loss.py` | 语言建模损失：next-token 交叉熵（阶段一的本质）✅ |
| 2 | `train2_label_mask.py` | 标签掩码 + 错一位对齐：哪些 token 该算 loss ✅ |
| 3 | `train3_two_losses.py` | 两段损失：future_traj + others 拆开算 ✅ |
| 4 | `train4_sft_stage1.py` | 阶段一的机制组装：迷你 VLM SFT ✅ |
| 5 | `train5_sft_stage2.py` | **阶段二机制**：冻结 VLM + 训 expert（crop cache、`cotrain_vlm` 梯度对照）✅ |
| 6 | `train6_curriculum.py` | **三段课程与训练技巧**：LingoQA 为什么在前、`lr_multiplier` 视觉塔 0.1、checkpoint 交接 ✅ |
| 7 | `train7_memory.py` | 显存与分布式：显存花在哪、ZeRO 1/2/3、为什么 stage2 不用 DeepSpeed ✅ |
| 8 | `train8_real_code.py` | 逐行对照真实代码（把 1~7 的机制映射到行号）✅ |

> `train1~4` 是**导入**（先讲清损失长什么样），`train5` 是**关键一跳**（阶段二怎么接上），
> `train6~7` 是**工程**（课程设计、跑得动吗），`train8` 收口。

**后续轨道（尚未开始）**：`rl1_grpo.py` / `rl2_reward.py` / `rl3_real_code.py` ——
只讲 GRPO 算法层与 Alpamayo 的 reward 设计，不碰 Cosmos-RL 那套基础设施。

> 每课都能独立运行（纯 CPU、秒级），重点看**张量 shape 和数值变化**。

**复习题**：跑完一课做一遍 `REVIEW.md` 里对应的题（题目在上、答案折叠在文末），
比再读一遍代码管用。

## 真实代码对照表

| 教程讲的 | 真实代码 |
|---|---|
| next-token 损失 | `recipes/alpamayo1_5_sft/models/sft_base_model.py:315` `_compute_next_token_loss` |
| 标签掩码 | `src/alpamayo/utils/get_label_mask.py`（`fill_masks_between_special_tokens`） |
| `<eos>` 也监督 | 同文件的 `get_role_eos_mask` / `get_assistant_mask`（注释：*include the eos (+1) for supervision*） |
| 两段损失 | `sft_base_model.py:388-407`（`future_traj` / `others`） |
| 轨迹 token 融进序列 | `sft_base_model.py` `fuse_traj_tokens` / `tokenize_history_trajectory` |
| 阶段二（冻结 + expert） | `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:125` `forward` |
| 阶段二的 cache 裁剪 | 同文件 `kv_cache.crop(last_traj_future_start_idx)` |
| `cotrain_vlm` / `stop_grad_from_vlm` | 同文件 `__init__`（默认 `False` / `True`） |
| 训练入口 | `recipes/alpamayo1_5_sft/train_hf.py` |
| 自定义学习率 | `recipes/alpamayo1_5_sft/trainer.py:58` `create_optimizer`（`lr_multiplier`） |
| 三段课程 | `recipes/alpamayo1_5_sft/configs/sft_stage1_lingoqa.yaml` / `sft_stage1_nav.yaml` / `sft_stage2_nav.yaml` |
