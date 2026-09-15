# Alpamayo 训练教程（从零理解 SFT）

配合本仓库 `recipes/alpamayo1_5_sft/` 的真实代码，一步步手写并理解 Alpamayo 的训练。

> 前置：建议先完成 `alpamayo1.5/tutorials/` 的**推理教程**（stage0~15），理解 VLA 的架构。
> 本教程讲的是**怎么训练**——那些权重是怎么来的。
>
> ⚠️ 但不用把 16 个 stage 全背下来再开始。**逐课真正需要哪几节，见下面
> [「每课需要推理教程的哪几节」](#-每课需要推理教程alpamayo15tutorials的哪几节) 那张表。**

---

## ⚠️ 开始之前：本教程引用了**两个不同的仓库**

本教程会反复引用另一个仓库 `alpamayo1.5`。**它和本仓库不在同一棵树里**，
需要**分别 clone**，并建议放在**同一个父目录下**：

```
你放代码的地方/
├─ alpamayo1.5/                    ← 另一个仓库（模型 + 推理教程）
│   ├─ src/alpamayo1_5/            ← 模型本体（`src/alpamayo1_5/...` 开头的引用在这）
│   └─ tutorials/                  ← 推理教程 stage0~15（本教程的前置）
│
└─ alpamayo-recipes/               ← 本仓库
    ├─ recipes/alpamayo1_5_sft/    ← 训练配置（训练教程的主线）
    ├─ src/alpamayo/               ← 数据 / processor / chat_template / utils
    └─ tutorials/                  ← 👈 你正在读的这个训练教程
```

**路径约定（读本教程时请按这个理解）**：

⚠️ **两个仓库都有 `src/` 目录**，靠**子包名**区分：

| 路径开头 | 在哪个仓库 | 例子 |
|---|---|---|
| `src/alpamayo1_5/...` | **`alpamayo1.5`** | `src/alpamayo1_5/models/base_model.py:292` |
| `src/alpamayo/...` | **本仓库** | `src/alpamayo/utils/get_label_mask.py` |
| `recipes/...` | **本仓库** | `recipes/alpamayo1_5_sft/models/sft_base_model.py:315` |

两个仓库的 `src/` 各存各的：

```
alpamayo1.5/src/          →  alpamayo1_5/     （模型本体）
alpamayo-recipes/src/     →  alpamayo/        （数据 / processor / chat_template / utils）
```

⚠️ **三个长得很像的名字，指的是三个不同的东西**——别混：

| 名字 | 是什么 |
|---|---|
| `alpamayo1.5/` | 另一个**仓库**的根目录 |
| `alpamayo1_5/` | 上面那个仓库里的 **python 包**目录（即 `src/alpamayo1_5/`） |
| `recipes/alpamayo1_5_sft/` | **本仓库内**的训练配置目录 |

以 `/home/you/code/` 为例：

```
src/alpamayo1_5/models/base_model.py:292   →  /home/you/code/alpamayo1.5/src/alpamayo1_5/models/base_model.py
src/alpamayo/utils/get_label_mask.py       →  /home/you/code/alpamayo-recipes/src/alpamayo/utils/get_label_mask.py
recipes/alpamayo1_5_sft/models/…           →  /home/you/code/alpamayo-recipes/recipes/alpamayo1_5_sft/models/…
```

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
   │          ⚠️ **这个配置里没有 CoC**（1.x 的 recipe 有 opt-in 的 coc.yaml，见下）
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

### ★ 三种监督：同一个 GT 走三条不同的路

**同一个「真值未来轨迹」，三个阶段里去了完全不同的地方**——这张图值得先记住：

```
                  同一个 GT future trajectory
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       trajectory tokenizer          action_space
              │                           │
              ▼                           ▼
       离散 trajectory tokens        连续 action
              │                           │
              ▼                           ▼
        VLM next-token CE          flow-matching MSE
          （Stage 1）                  （Stage 2）

   ── RL 走的是另一条路：GT 不产生 loss，只产生 reward ──

       model rollout  →  predicted trajectory + CoT
                                ↓
                reward（ADE + comfort + reasoning）
                                ↓
                     advantage  →  GRPO loss
```

对照表（**这是本教程最值得记的一张**）：

| | **Stage 1**（VLM） | **Stage 2**（expert） | **RL**（VLM） |
|---|---|---|---|
| **训练对象** | VLM | action expert | VLM policy |
| **轨迹表示** | 离散 token | 连续 action `(accel, κ)` | 离散 token（**rollout 出来**） |
| **GT 未来轨迹的去向** | `tokenize_future_trajectory` 后当 **label** | `traj_to_action` 后当**回归目标** | 只用来**算 reward** |
| **GT CoC 的去向** | ⚠️ **只在 opt-in 的 `coc` 变体里**当 label（默认关，1.5 无此选项） | 不参与 | reasoning reward（Lingo-Judge） |
| **Loss 来源** | next-token 交叉熵 | flow-matching MSE | GRPO policy loss |
| **直接和 GT 比较？** | ✅ | ✅ | ❌ |
| **要 rollout？** | ❌ | ❌ | ✅ |
| **要 reward？** | ❌ | ❌ | ✅ |
| **梯度从哪来** | CE → backprop | FM MSE → backprop | reward → advantage → GRPO → backprop |

★ **最该记住的一条**：

> **Stage 1 / Stage 2 是「真值 → loss」；RL 是「真值 → reward → loss」。**
>
> RL 里**没有一个 token-level 的「正确答案」**——只有「这一整条 rollout 好不好」。
> 这就是为什么 RL 必须用 score function 估计量（见 rl1），
> 也是为什么它需要「一组」采样来比较（组内归一化）。

代码位置（都在 `recipes/alpamayo1_5_sft/models/`）：

| 阶段 | 关键行 |
|---|---|
| Stage 1 的 GT → token | `sft_base_model.py:43` `tokenize_future_trajectory` |
| Stage 1 的 loss | `sft_base_model.py:315` `_compute_next_token_loss`，`:388-407` 两段拆分 |
| Stage 2 的 GT → action | `sft_alpamayo_r1.py:95` `traj_to_action` → `:102` `construct_training_data` |
| Stage 2 的 loss | `sft_alpamayo_r1.py:204` `compute_loss_from_pred`（flow matching） |

### ★ CoC 到底在哪、怎样训练（**Alpamayo 1 可选；1.5 当前 recipe 不提供**）

CoC（chain-of-causation，因果链推理）本质上不是一个特殊的优化器，而是普通的
**监督微调（SFT）**：数据集为每个驾驶片段提供一段文字推理 `cot`，模板把它放进
`<cot_start> ... <cot_end>`，然后用 next-token cross-entropy 训练模型逐 token 生成。

一条启用 CoC 的训练序列大致是：

```text
图片 + 历史轨迹 + 导航指令
assistant: <cot_start> ...推理文本... <cot_end>
           <traj_future_start> ...轨迹 token... <traj_future_end> <eos>
```

训练时需要同时满足三件事：

1. 数据样本中真的有 `cot` 字段。PAI 数据集通过 `reasoning_metadata` 读取它，并用
   `clip_index_metadata` 只枚举有推理标注的片段。
2. chat template 的 `components_order` / `components_prompt` 包含 `cot`。
3. `label_components` 包含 `cot`，这样 `<cot_start>`、推理内容和 `<cot_end>` 才会进入
   loss；通常还会同时包含 `traj_future`。

Alpamayo 1 的开源 recipe 提供了现成的 `recipes/alpamayo1_sft/configs/vla_processor/coc.yaml`，
它把这三个列表设置为：

```yaml
components_order: [image, traj_history, prompt, cot, traj_future]
components_prompt: [cot, traj_future]
label_components: [cot, traj_future]
```

并且要按 `recipes/alpamayo1_sft/SKILL.md` 配好 reasoning parquet、reasoning clip index，
以及 `use_default_keyframe=false`。漏掉数据侧配置时，常见报错是
`cot not found in data` 或 `Event timestamp ... not found`。

Alpamayo 1.5 的本仓库 recipe 没有 `coc.yaml`，其 `nav.yaml` 默认只监督
`traj_future`。因此这里的 `train5` toy 序列虽然保留了 CoC，主要是为了讲清 label mask；
它不是 1.5 当前公开 nav 配置的逐字复现。发布模型已有的 CoC 能力来自带 CoC 标注的训练
checkpoint，RL 阶段则可以把 rollout 生成的 CoC 交给 Lingo-Judge 作为 reward，但那是
**奖励模型打分**，不是把 CoC 再当作 token-level SFT 标签。

最小的 CoC SFT 逻辑可以概括为：

```python
labels = torch.where(cot_mask | traj_mask, input_ids, IGNORE_INDEX)
loss = cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=IGNORE_INDEX)
```

也就是说，模型学习的是“在当前图片、轨迹和提示条件下，生成这段标注推理以及后续轨迹”。
它不会自动保证推理真的有用；要优化“推理是否帮助驾驶”，才需要后续 RL 的 reasoning reward。

**推理时模型确实会先写 CoC 再出轨迹**（推教程 stage13/14 讲的都对）。
但训练侧，「CoC 是不是监督目标」**取决于配置**，而这里有个容易踩的坑——
**仓库里有两个 SFT recipe，情况不一样**。

#### 机制：`components_order` 决定序列里放什么

```python
# chat_template/r1.py:build_conversation
for component in components_order:          # ← 只遍历这个列表
    self._add_component(component=component, ...)
# 里面是 match component:  case "cot": ...
# 不在列表里 → 永不匹配 → 永远不放进序列
```

#### 两个 recipe 的差别（**这是关键**）

```
recipes/
├─ alpamayo1_sft/          ← Alpamayo 1
│   └─ configs/vla_processor/
│       ├─ default.yaml    components_order: [image, traj_history, prompt, traj_future]
│       │                  label_components: [traj_future]                    ← 默认
│       └─ coc.yaml        components_order: [image, traj_history, prompt, cot, traj_future]
│                          components_prompt: [cot, traj_future]
│                          label_components:  [cot, traj_future]              ← opt-in ⭐
│
└─ alpamayo1_5_sft/        ← Alpamayo 1.5
    └─ configs/vla_processor/
        ├─ default.yaml \ nav.yaml \ vqa.yaml   三个都【没有 cot】
        └─ （连 coc.yaml 都不存在）              ← 这个选项被去掉了
```

**所以：**

| | Alpamayo **1** recipe | Alpamayo **1.5** recipe |
|---|---|---|
| CoC 作为 Stage-1 监督目标 | ✅ **支持，但 opt-in** | ❌ **没有这个选项** |
| 怎么开 | 把 `default.yaml` 换成 `coc.yaml` 的三个列表 + 数据集侧三件套 | — |
| 默认值 | **关**（`ENABLE_COC=no`） | — |

**证据（都在 `alpamayo1_sft/`）**：
- `README.md:116` `### Stage 1 (CoC reasoning disabled)`
- `README.md:146` `### Stage 1 (CoC reasoning enabled)`
- `SKILL.md:92` 把它做成开关，**默认值写的是 `no`**
- `SKILL.md:352` 「Apply this **only if** `$ENABLE_COC = yes`」——要三处齐全，
  缺一个就报 `AssertionError: cot not found in data but 'cot' in components_order`
- `benchmark/run_benchmarks.py:51-56` 有 `no_coc` / `coc` 两组对照实验

#### 本教程对应的 1.5 配置（`nav.yaml`）

`stage1_nav` 的**实际序列**（不带 CoC）：

```
system   : "You are a driving assistant that generates safe and accurate actions."
user     : [图片] [历史轨迹×48] <route_start>…<route_end> "output the future trajectory."
assistant: <traj_future_start> <i..>×128 <traj_future_end> <|im_end|>
```

`components_order` / `label_components` 见上（`tests/test_recipe_static_contracts.py:170-171`
专门断言了这两个列表）。⚠️ 因为 `label_components: ["traj_future"]` 只圈了轨迹区间，
那「两段损失」里的 `others` 段**实际只剩 assistant 的 `<|im_end|>` 这一个 token**。

#### 那发布模型的 CoC 能力是哪来的

发布推理**明确要求**模型输出 CoC，而模型照做——`alpamayo1_5/helper.py:104-143`：

```python
prompt_text = ("output the chain-of-thought reasoning of the driving process, "
               "then output the future trajectory.")      # ← 明确要 CoC
...
{"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]}  # ← 预热
```

既然模型能被这个指令驱动，说明**它在某个阶段被训过写 CoC** ——
而 `alpamayo1_sft` 里的 `coc.yaml` 就是那条路径。也就是说：

> **发布模型是用「开了 CoC 的 Stage 1」训出来的**；1.5 的开源 recipe
> 只保留了轨迹部分（`coc.yaml` 被去掉），CoC 能力由 checkpoint 继承。
> （`cot` 的机器还在——`components.py:81` 有模板串 `"output the chain-of-thought
> reasoning of the driving process"`、`construct_cot` 也实现了——因为它是给 1.x 用的。）

> **本教程的处理**：`train1`/`train2`/`train3`/`train5`/`train7` 的 toy 序列里
> **保留了 CoC**，因为它是讲清「哪些 token 算 loss」的好素材，而且掩码/两段损失的
> **机制本身是真的**。但请记住：**那是教学装置，1.5 的 nav 序列里没有那一段。**

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

本教程有**两层**：

- **主课（1~10）** —— 按编号顺序读，一课的粒度是「一个能独立讲清的概念」
- **拆分课（5a~5d / 7a~7d）** —— 把主课 `train5` / `train7` 里的**四个前置概念**
  拆出来单独演示。**它们不是独立的一课，是主课的前置**

> ⚠️ **推荐先读拆分课，再读主课。**
> `train5_sft_stage1.py` 和 `train7_sft_stage2.py` 是「完整实现对照课」——
> 一次性塞进了注意力、teacher forcing、图片前缀、两段损失四件事。
> 先跟着 5a~5d 一件件看过，再回到 `train5` 看它们怎么组装起来，就不会觉得是黑盒。

### 拆分课（建议先读）

| 顺序 | 文件 | 只解决一个问题 |
|---|---|---|
| 5a | `train5a_attention.py` | causal self-attention 为什么不能看未来 |
| 5b | `train5b_teacher_forcing.py` | teacher forcing 和自回归生成的区别 |
| 5c | `train5c_image_conditioning.py` | 图片作为前缀 token 时怎样对齐 logits/labels |
| 5d | `train5d_stage1_sft.py` | 把 mask、轨迹 token、图片和两段 loss 组装起来 |
| 7a | `train7a_kv_cache.py` | KV cache 是怎样保存前缀条件的 |
| 7b | `train7b_cache_crop.py` | 为什么要裁掉 teacher-forcing 的真值轨迹 |
| 7c | `train7c_freeze_gradient.py` | 冻结 VLM 后梯度到底去哪了 |
| 7d | `train7d_stage2_expert.py` | 阶段二 expert 训练的完整流程骨架 |

每个拆分课都能单独运行。读完它们再跑 `train5_sft_stage1.py` / `train7_sft_stage2.py`。

### 主课（完整路线）

| 课 | 文件 | 学什么 |
|---|---|---|
| 1 | `train1_lm_loss.py` | 语言建模损失：next-token 交叉熵（阶段一的本质）✅ |
| 2 | `train2_label_mask.py` | 标签掩码 + 错一位对齐：哪些 token 该算 loss ✅ |
| 3 | `train3_two_losses.py` | 两段损失：future_traj + others 拆开算 ✅ |
| 4 | `train4_traj_tokens.py` | 轨迹怎么量化成一串 token（`<iK>` 到底是什么）✅ |
| 5 | `train5_sft_stage1.py` | 阶段一的机制组装：迷你 VLM SFT（**前置：5a~5d**）✅ |
| 6 | `train6_flow_matching.py` | flow matching 是什么（训练目标 + 欧拉采样）✅ |
| 7 | `train7_sft_stage2.py` | **阶段二机制**：冻结 VLM + 训 expert（**前置：7a~7d**）✅ |
| 8 | `train8_curriculum.py` | **三段课程与训练技巧**：LingoQA 为什么在前、`lr_multiplier` 视觉塔 0.1、checkpoint 交接 ✅ |
| 9 | `train9_memory.py` | 显存与分布式：显存花在哪、ZeRO 1/2/3、为什么 stage2 不用 DeepSpeed ✅ |
| 10 | `train10_real_code.py` | 逐行对照真实代码（把 1~9 的机制映射到行号）✅ |

> **主课的编号（1~10）就是阅读顺序**，本身没有字母后缀 ——
> 结构上：`1~5` 是**导入**（先讲清 loss 和序列长什么样），
> `6~7` 是**关键一跳**（flow matching + 阶段二怎么接上），
> `8~9` 是**工程**（课程设计、跑得动吗），`10` 收口。
>
> `train4`（轨迹 token）和 `train6`（flow matching）之所以单独成主课，是因为
> 它们各自是**独立主题**，混进别课会让人卡在一个没讲过的东西上。
> 字母课（5a~5d / 7a~7d）则是把 `train5`/`train7` 里的概念**再拆细一层**——
> 两者不是并列关系，是「前置 → 组装」的关系。

### ★ 每课需要推理教程（`alpamayo1.5/tutorials/`）的哪几节

一句「先做完 stage0~15」粒度太粗——卡住时不知道该回去看哪节。逐课对上：

| 课 | 需要推理教程的 | 为什么 |
|---|---|---|
| train1~3 | — | 只用 `embed + head`，没有 attention |
| **train4** | real1 | `<i0>`…`<i3999>` 那个词表怎么来的 |
| **train5a** | **stage9, stage13** | causal mask 是 5a 的全部内容；stage9 讲注意力本身 |
| **train5b** | stage13 | teacher forcing 就是 stage13 的自回归对照 |
| **train5c** | stage14 | 图片拼进多模态前缀，stage14 讲的就是这个 |
| **train5d** | — | 只是把 5a~5c 组装起来 |
| **train5**（主课） | 同上 | 组装版，前置知识同 5a~5d |
| **train6** | **stage2, stage6** | flow matching 的采样循环与训练目标 |
| **train7a** | **stage13 Part 2** | KV cache 当条件就是 Part 2 的做法 |
| **train7b** | — | crop 是训练侧特有的，推理教程没有对应物 |
| **train7c** | — | 梯度分析，纯 PyTorch |
| **train7d** | stage6 | expert 的 flow matching 训练 |
| **train7**（主课） | 同上 | 组装版，前置知识同 7a~7d |
| train8 | — | 全是配置与课程设计，没有新机制 |
| train9 | — | 显存算术 |
| train10 | — | 纯映射 |
| rl1a~rl1c | — | 算法本身自足；`train1` 的 log-prob 可选回看 |
| rl2a~rl2b | real5 | ADE / FDE 这些指标在 real5 里出现过 |
| rl2c | — | reasoning reward 独立成进阶课 |
| rl3a~rl3b | — | 纯系统概览与代码导航 |

### 教程共享模块

这些不是额外课程，而是多个 toy 课共用的最小实现。放进 `tutorial_support/` 是为了把
“教材共享代码”和“按顺序阅读的课程”区分开。

| 文件 | 是什么 | 谁在用 |
|---|---|---|
| `tutorial_support/trajectory_tokenizer.py` | 玩具版离散轨迹 tokenizer（镜像 `DiscreteTrajectoryTokenizer`） | train4 / train5 / train7 |
| `tutorial_support/stage2_model.py` | 阶段二的模型与数据（`TinyVLM` / `TinyExpert` / `Stage2Model` / `make_sample`） | train7（讲）/ train8（用） |

> 抽成模块是为了**别让一课硬依赖另一课的实现细节**：`train8_curriculum.py` 原本
> 直接 `from train7_sft_stage2 import ...` 了 7 个名字，train7 一改它就可能悄悄坏掉。
> 放共享模块里，改 API 两边会一起编译不过，不会漏。

---

## RL 轨道（rl1a ~ rl3b）

SFT 的损失优化的是「像不像真值」，但「像真值」≠「车开得好」——
SFT 的 loss 里根本没有「好不好」这一项。RL 就是把 reward 变成梯度的办法。

真实代码在 `recipes/alpamayo1_x_rl/`（GRPO via Cosmos-RL，**需要 ≥5×80GB**，
真实配置是 64 个 policy replica + 128 个 rollout replica）。
所以本轨道**只做算法层**，不碰 Cosmos-RL 那套基础设施。

### RL 轨道的推荐前置和阅读顺序

不需要先完成 `train1~9`。最低前置是：

| 课 | 真正需要的知识 | 建议回看 |
|---|---|---|
| `rl1a` | 采样、reward、log-prob、梯度下降 | `train1` 可选 |
| `rl1b` | rl1a 的 log-prob 梯度、组内比较 | `rl1a` |
| `rl1c` | rl1b 的方差实验 | `rl1b` |
| `rl2a` | ADE、comfort 两个原始指标 | `real5` 可选，用来回看 ADE/FDE |
| `rl2b` | 指标归一化、权重和 ADE gate | `rl2a` |
| `rl2c` | CoC 抽取、grader、reasoning gate | `rl2b` |
| `rl3a` | policy / rollout / controller 的系统分工 | `rl1c`、`rl2b` |
| `rl3b` | 概念到真实文件的最短阅读路径 | `rl3a` |

`rl1_grpo.py`、`rl2_reward.py`、`rl3_real_code.py` 仍然保留，作为每条细分路线完成后的
“完整总览课”。第一次学习不建议从这三个总览文件开始。

| 课 | 文件 | 学什么 |
|---|---|---|
| 1a | `rl1a_score_function.py` | reward 不可导时，如何用 score function 更新策略 |
| 1b | `rl1b_grpo_advantage.py` | 组内均值、标准化和没有相对信号的边界 |
| 1c | `rl1c_grpo_variance.py` | REINFORCE vs GRPO 的方差实验 |
| 2a | `rl2a_motion_metrics.py` | ADE 与 comfort 的原始测量 |
| 2b | `rl2b_reward_aggregation.py` | 权重、归一化和 ADE gate |
| 2c | `rl2c_reasoning_reward.py` | CoC、Lingo-Judge 和 reasoning gate（进阶） |
| 3a | `rl3a_system_overview.py` | policy / rollout / controller 的数据流 |
| 3b | `rl3b_code_walkthrough.py` | 概念对应的真实文件和最短阅读路径 |

推荐顺序：

```text
rl1a -> rl1b -> rl1c -> rl2a -> rl2b -> rl2c -> rl3a -> rl3b
```

其中 `rl2c` 可以在先完成 `rl3b` 后再回看；它需要额外理解外部 grader，
不是 GRPO 主算法的前置。

> `rl1c` 在当前 toy、随机种子和 `G=12` 下实测：GRPO 的梯度方差约为 REINFORCE 的 **1/33**；
> 这个比例不是 GRPO 的固定常数。给所有 reward 加同一个常数，
> advantage **完全不变**（这就是「相对」的含义，也是它能省掉 critic 的原因）。

> RL 还有一个容易忽略的边界：如果同一个 prompt 的所有候选 reward 都相同，组内
> advantage 就全是 0，GRPO 没有相对学习信号。这也是 `n_generation` 不能简单设成 1 的原因。

## 运行环境

仓库原来没有统一的 conda environment 文件；完整 recipe 的依赖写在各自的
`recipes/*/pyproject.toml` 中。教程 toy 代码不需要 `alpamayo_r1`、vLLM、DeepSpeed
或 Cosmos-RL，只需要 Python 和 PyTorch。仓库现在提供了：

```text
tutorials/environment-tutorials.yml   # conda 环境
tutorials/requirements.txt            # pip 最小依赖
```

新建环境：

```bash
conda env create -f tutorials/environment-tutorials.yml
conda activate alpamayo-tutorials
python tutorials/train1_lm_loss.py
```

你机器上已有的 `qwen_vl` 也可以直接运行 toy 教程（已检查：Python 3.10、PyTorch 2.6、CUDA 可用）：

```bash
conda activate qwen_vl
python tutorials/train5a_attention.py
```

如果只想运行新增的演示，推荐按 `5a -> 5b -> 5c -> 5d`，再按 `7a -> 7b -> 7c -> 7d`。
从仓库根目录运行即可；这些脚本不依赖另一个 `alpamayo1.5` 仓库。

完整 SFT / RL recipe 是另一种环境：需要 Python 3.12、PyTorch 2.8，以及
`flash-attn`、DeepSpeed、vLLM、Cosmos-RL 等依赖；其中 RL 的真实配置还需要多卡 GPU。
不要为了运行 toy 教程安装这套环境。

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
| **轨迹量化成 token** | `src/alpamayo1_5/action_space/discrete_action_space.py:47` `DiscreteTrajectoryTokenizer.encode`（train4；toy 版是 `tutorial_support/trajectory_tokenizer.py`） |
| 造 `<i0>`…`<i3999>` 词表 | `src/alpamayo1_5/models/base_model.py:271`（train4） |
| 推理时屏蔽这些 token | `src/alpamayo1_5/models/alpamayo1_5.py:48` `ExpertLogitsProcessor`（train4） |
| 阶段二（冻结 + expert） | `recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py:125` `forward` |
| 阶段二的 cache 裁剪 | 同文件 `kv_cache.crop(last_traj_future_start_idx)` |
| `cotrain_vlm` / `stop_grad_from_vlm` | 同文件 `__init__`（默认 `False` / `True`） |
| 训练入口 | `recipes/alpamayo1_5_sft/train_hf.py` |
| 自定义学习率 | `recipes/alpamayo1_5_sft/trainer.py:58` `create_optimizer`（`lr_multiplier`） |
| 三段课程 | `recipes/alpamayo1_5_sft/configs/sft_stage1_lingoqa.yaml` / `sft_stage1_nav.yaml` / `sft_stage2_nav.yaml` |
