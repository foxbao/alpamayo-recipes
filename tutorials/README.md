# Alpamayo 训练教程（从零理解 SFT）

配合本仓库 `recipes/alpamayo1_5_sft/` 的真实代码，一步步手写并理解 Alpamayo 的训练。

> 前置：建议先完成 `alpamayo1.5/tutorials/` 的**推理教程**（stage0~15），理解 VLA 的架构。
> 本教程讲的是**怎么训练**——那些权重是怎么来的。
>
> ⚠️ 但不用把 16 个 stage 全背下来再开始。**逐课真正需要哪几节，见下面
> [「每课需要推理教程的哪几节」](#-每课需要推理教程alpamayo15tutorials的哪几节) 那张表。**

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
   │    组件: [image, question, answer]  ← 只训 answer 段
   │    目标: 通用「看图回答驾驶问题」——先把视觉语言对齐练好
   │          ⚠️ answer 用 <answer_start>/<answer_end> 包，**不是 CoC 格式**
   │
   ├─ stage1_nav      (sft_stage1_nav.yaml)       模型：TrainableReasoningVLA
   │    数据: PAI nav + 导航标注
   │    组件: [image, traj_history, route, prompt, traj_future]
   │          用户侧: 图 + 历史轨迹 + 导航指令 + "output the future trajectory."
   │          助手侧: <traj_future_start> <i..>×128 <traj_future_end>
   │          ⚠️ **序列里没有 CoC**（见下面「CoC 到底在哪」）
   │    目标: 未来轨迹的离散 token
   │    损失: next-token 交叉熵（拆成 future_traj / others 两段）
   │          ⚠️ 本配置下 label_components=["traj_future"]，
   │             所以 others 段实际只剩 assistant 的 <|im_end|> 一个 token
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

### ★ 哪个阶段训哪个部件（别被模型类名绕进去）

上面写了每段的模型**类名**，但光看名字看不出训的是什么。真实情况是：

| 阶段 | config 里的模型类 | 模型里**有 `expert` 吗** | 训什么 | 目标函数 |
|---|---|---|---|---|
| **SFT 阶段一** | `TrainableReasoningVLA.from_alpamayo_checkpoint` | ❌ **没有** | **VLM** | next-token 交叉熵 |
| **RL**（`alpamayo1_x_rl`） | `RLWrapperReasoningVLA` | ❌ 没有 | **VLM** | GRPO（reward） |
| **SFT 阶段二** | `TrainableAlpamayoR1.from_pretrained` | ✅ 有 | **expert**（VLM 冻结） | flow matching |

证据在类定义上：

```python
# sft_base_model.py:201 —— 阶段一用的
class TrainableReasoningVLA(ReasoningVLA, TrajectoryFusionWithFutureMixin):
                            ↑ ReasoningVLA 里【没有 expert】

# sft_alpamayo_r1.py:33 —— 阶段二用的
class TrainableAlpamayoR1(AlpamayoR1):
                            ↑ 有 self.expert（第 192 行用了）
```

```python
# src/alpamayo1_5/models/base_model.py:292   ← 只有 VLM
class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin):

# src/alpamayo1_5/models/alpamayo1_5.py:82    ← 在它基础上加了 expert
class Alpamayo1_5(ReasoningVLA):
    self.expert = AutoModel.from_config(expert_config)     # :105
```

**要点**：`VLM` 被训**两次**（SFT 阶段一 → RL），`expert` 被训**一次**（SFT 阶段二）。

所以完整的闭环其实是：

```
              ┌────── 训 VLM ──────┐
SFT 阶段一  ──▶   RL   ──▶  SFT 阶段二  ──▶  发布模型
（next-token CE）(GRPO)      （flow matching）
                            └─ 训 expert ─┘
                              VLM 冻结，且 expert 是【随机初始化】的
```

⚠️ **RL 和 SFT 阶段一训的是同一个部件**，差别不在「训谁」而在「怎么训」：

| | SFT 阶段一 | RL |
|---|---|---|
| 数据从哪来 | 数据集里的**真值**序列 | **模型自己采样**出来的 |
| 目标函数 | next-token CE | reward（GRPO） |
| 在教什么 | 「**像不像**真值」 | 「车**开得好不好**」 |
| 采样可导吗 | 不需要（有真值） | **不可导** ⟹ 要 score function |

这正是 RL 存在的理由：阶段一已经把 VLM 训到会输出轨迹 token 了，
但「像真值」≠「开得好」——RL 用 reward 补上这个缺口。

### ★ CoC 到底在哪（一个查过才发现的事实）

**推理时模型确实会先写 CoC 再出轨迹**（推教程 stage13/14 讲的都对）。但——
**本仓库的这两个 SFT 阶段，训练序列里都没有 CoC。**

`components_order` 决定序列里放哪些组件，而 `build_conversation` 就是按它遍历的：

```python
# chat_template/r1.py
for component in components_order:          # ← 只遍历这个列表
    self._add_component(component=component, ...)
# 里面是 match component:  case "cot": ...
# 不在列表里 → 永不匹配 → 永远不放进序列
```

两个 shipped 配置：

| 阶段 | `components_order` | 有 `cot` 吗 |
|---|---|---|
| stage1_lingoqa | `["image", "question", "answer"]` | ❌ |
| stage1_nav | `["image", "traj_history", "route", "prompt", "traj_future"]` | ❌ |

（`tests/test_recipe_static_contracts.py:170-171` 专门断言了这两个列表）

所以 `stage1_nav` 的**实际序列**是：

```
system   : "You are a driving assistant that generates safe and accurate actions."
user     : [图片] [历史轨迹×48] <route_start>…<route_end> "output the future trajectory."
assistant: <traj_future_start> <i..>×128 <traj_future_end> <|im_end|>
```

连**系统提示都没提 CoC**。`label_components: ["traj_future"]` 也只圈了轨迹区间，
所以那「两段损失」里的 `others` 段**实际只剩 assistant 的 `<|im_end|>` 这一个 token**。

**那 CoC 是哪来的？** 两份证据指向同一个答案——**继承自发布 checkpoint**：

1. 本仓库定位是「在已发布模型上继续训」：`from_alpamayo_checkpoint` 的 docstring 写明
   只加载 checkpoint 里 fine-tuned 的 `vlm.*` 权重
2. 数据集里**有** `cot` 字段（`pai_nav.py:25`），但它在**RL 的 reward** 里被用
   （`gt_cot = reference.get("cot", "")`），不是当 SFT 目标

**为什么这不影响「模型会写 CoC」**：CoC 能力来自发布 checkpoint。
而且发布推理**明确要求**它输出——`alpamayo1_5/helper.py:104-143`：

```python
prompt_text = ("output the chain-of-thought reasoning of the driving process, "
               "then output the future trajectory.")      # ← 明确要 CoC
...
{"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]}  # ← 预热
```

> **旁证**：既然推理指令里白纸黑字要求「先输出 CoC」，模型也照做——
> 那它**一定**在某个阶段被训练过这件事。所以基本可以认定：
> **NVIDIA 的原始训练带 CoC，开源的 recipe 配置简化掉了这一段**
> （`cot` 的机器造好了却没启用：`components.py:81` 有同样的模板串、
> `construct_cot` 也实现了）。但这是**旁证而非仓库内的直接证据**，所以标在这里。

> **本教程的处理**：`train1`/`train2`/`train3`/`train5`/`train7` 的 toy 序列里
> **保留了 CoC**，因为它是讲清「哪些 token 算 loss」的好素材，而且掩码/两段损失的
> **机制本身是真的**。但请记住：**那是教学装置，不是 nav 的真实序列结构。**

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
> policy/rollout/controller 异步架构，需要 ≥5×80GB 显存）。本教程的 `train1~10`
> 只讲 SFT；RL 单独开 `rl*` 轨道，且只讲**算法层**（基础设施教学价值低）。

---

## 课程路线

| 课 | 文件 | 学什么 |
|---|---|---|
| 1 | `train1_lm_loss.py` | 语言建模损失：next-token 交叉熵（阶段一的本质）✅ |
| 2 | `train2_label_mask.py` | 标签掩码 + 错一位对齐：哪些 token 该算 loss ✅ |
| 3 | `train3_two_losses.py` | 两段损失：future_traj + others 拆开算 ✅ |
| 4 | `train4_traj_tokens.py` | 轨迹怎么量化成一串 token（`<iK>` 到底是什么）✅ |
| 5 | `train5_sft_stage1.py` | 阶段一的机制组装：迷你 VLM SFT ✅ |
| 6 | `train6_flow_matching.py` | flow matching 是什么（训练目标 + 欧拉采样）✅ |
| 7 | `train7_sft_stage2.py` | **阶段二机制**：冻结 VLM + 训 expert（crop cache、`cotrain_vlm` 梯度对照）✅ |
| 8 | `train8_curriculum.py` | **三段课程与训练技巧**：LingoQA 为什么在前、`lr_multiplier` 视觉塔 0.1、checkpoint 交接 ✅ |
| 9 | `train9_memory.py` | 显存与分布式：显存花在哪、ZeRO 1/2/3、为什么 stage2 不用 DeepSpeed ✅ |
| 10 | `train10_real_code.py` | 逐行对照真实代码（把 1~9 的机制映射到行号）✅ |

> **编号就是阅读顺序**，没有字母后缀 —— 一课的粒度就是「一个能独立讲清的概念」。
> 结构上：`train1~5` 是**导入**（先讲清 loss 和序列长什么样），
> `train6~7` 是**关键一跳**（flow matching + 阶段二怎么接上），
> `train8~9` 是**工程**（课程设计、跑得动吗），`train10` 收口。
>
> `train4`（轨迹 token）和 `train6`（flow matching）之所以单独成课，是因为
> 它们各自是**独立主题**，混进别课会让人卡在一个没讲过的东西上。

### ★ 每课需要推理教程（`alpamayo1.5/tutorials/`）的哪几节

一句「先做完 stage0~15」粒度太粗——卡住时不知道该回去看哪节。逐课对上：

| 课 | 需要推理教程的 | 为什么 |
|---|---|---|
| train1~3 | — | 只用 `embed + head`，没有 attention |
| **train4** | real1 | `<i0>`…`<i3999>` 那个词表怎么来的 |
| **train5** | **stage9, stage13** | 突然出现 `CausalBlock`（自注意力）和因果 mask |
| **train6** | **stage2, stage6** | flow matching 的采样循环与训练目标 |
| **train7** | **stage13（尤其 Part 2）** | KV cache 当条件 —— 和 Part 2 是同一件事 |
| train8 | — | 全是配置与课程设计，没有新机制 |
| train9 | — | 显存算术 |
| train10 | — | 纯映射 |
| rl1 | — | 算法本身自足 |
| rl2 | real5 | ADE / FDE 这些指标在 real5 里出现过 |
| rl3 | — | 纯映射 |

### 共享模块

| 文件 | 是什么 | 谁在用 |
|---|---|---|
| `toy_traj_tokens.py` | 玩具版离散轨迹 tokenizer（镜像 `DiscreteTrajectoryTokenizer`） | train4 / train5 / train7 |
| `toy_stage2.py` | 阶段二的模型与数据（`TinyVLM` / `TinyExpert` / `Stage2Model` / `make_sample`） | train7（讲）/ train8（用） |

> 抽成模块是为了**别让一课硬依赖另一课的实现细节**：`train8_curriculum.py` 原本
> 直接 `from train7_sft_stage2 import ...` 了 7 个名字，train7 一改它就可能悄悄坏掉。
> 放共享模块里，改 API 两边会一起编译不过，不会漏。

---

## RL 轨道（rl1 ~ rl3）

SFT 的损失优化的是「像不像真值」，但「像真值」≠「车开得好」——
SFT 的 loss 里根本没有「好不好」这一项。RL 就是把 reward 变成梯度的办法。

真实代码在 `recipes/alpamayo1_x_rl/`（GRPO via Cosmos-RL，**需要 ≥5×80GB**，
真实配置是 64 个 policy replica + 128 个 rollout replica）。
所以本轨道**只做算法层**，不碰 Cosmos-RL 那套基础设施。

| 课 | 文件 | 学什么 |
|---|---|---|
| 1 | `rl1_grpo.py` | **GRPO 算法核心**：reward 不可导 → score function → 组内归一化省掉 critic ✅ |
| 2 | `rl2_reward.py` | **reward 设计**：ADE + comfort、gate 的形状、reasoning reward ✅ |
| 3 | `rl3_real_code.py` | 逐行对照真实代码 + SFT/RL 差别表 ✅ |

> `rl1` 实测：GRPO 的梯度方差是 REINFORCE 的 **1/36**；给所有 reward 加同一个常数，
> advantage **完全不变**（这就是「相对」的含义，也是它能省掉 critic 的原因）。

> ⚠️ RL 训的是 **VLM backbone**（走**离散轨迹 token** 那条通路），
> 不是 SFT 阶段二那个连续 action expert。所以 RL 导出的 checkpoint
> **只含 VLM 权重**，拿回去 SFT 训 expert 时 expert 是**随机初始化**的。

> 每课都能独立运行（纯 CPU，单课 2~20 秒），重点看**张量 shape 和数值变化**。

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
| **轨迹量化成 token** | `src/alpamayo1_5/action_space/discrete_action_space.py:47` `DiscreteTrajectoryTokenizer.encode`（train4；toy 版是 `toy_traj_tokens.py`） |
| 造 `<i0>`…`<i3999>` 词表 | `src/alpamayo1_5/models/base_model.py:271`（train4） |
| 推理时屏蔽这些 token | `src/alpamayo1_5/models/alpamayo1_5.py:44` `ExpertLogitsProcessor`（train4） |
| 阶段二（冻结 + expert） | `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:125` `forward` |
| 阶段二的 cache 裁剪 | 同文件 `kv_cache.crop(last_traj_future_start_idx)` |
| `cotrain_vlm` / `stop_grad_from_vlm` | 同文件 `__init__`（默认 `False` / `True`） |
| 训练入口 | `recipes/alpamayo1_5_sft/train_hf.py` |
| 自定义学习率 | `recipes/alpamayo1_5_sft/trainer.py:58` `create_optimizer`（`lr_multiplier`） |
| 三段课程 | `recipes/alpamayo1_5_sft/configs/sft_stage1_lingoqa.yaml` / `sft_stage1_nav.yaml` / `sft_stage2_nav.yaml` |
