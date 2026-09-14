# Alpamayo 训练教程复习题（train1 ~ train10）


> **用法**：先自己答，再翻到文末对答案。每题标了「考点」，答不上来就回去看那一行代码。
> 答案在意不在字——说出关键点即可。

---

## train1 — next-token CE（阶段一的本质）

**1-1** `shift_logits = logits[:, :-1, :]`、`shift_labels = input_ids[:, 1:]`。
为什么要**错开一位**？不错开、直接拿 `logits` 和 `input_ids` 算 CE，会怎样？
<sub>考点：`logits[i]` 本来就看过 `input_ids[i]`，不错位就是让模型「抄当前位置的输入」</sub>

**1-2** 初始 loss 是 `2.6187`，注释说「应接近 ln(11) = 2.40」。
**为什么是 ln(词表大小)？** 词表变成 100，初始 loss 大概是多少？
<sub>考点：随机初始化 → 输出近似均匀分布 `1/V` → CE = -log(1/V) = ln V</sub>

**1-3** 训练后贪心续写一字不差地复现了训练序列。这能说明模型「学会了这条 CoC 的含义」吗？
它实际说明的是什么？
<sub>考点：单条样本 = 过拟合；说明的是「训练链路是通的」，是冒烟测试不是能力测试</sub>

---

## train2 — 标签掩码 + 那个「错一位」

**2-1** mask 覆盖的是 `<cot_start> ~ <cot_end>`（**含两端**）。
只覆盖中间的内容词（`construction blocks right lane change left`）不行吗？
为什么头尾两个特殊 token 也要算 loss？
<sub>考点：含 `<cot_end>` ⟹ 它成为目标 ⟹ 模型学会输出它 ⟹ 生成时知道该停；
只 mask 内容词的话模型根本停不下来</sub>

**2-2** 打印里「`9 lane` → `10 <cot_end>` ✓」被监督了。
**被监督的是哪个位置的 label**——来源位置 9 的，还是目标位置 10 的？
<sub>考点：`shift_labels = labels[:, 1:]`，目标永远取 i+1</sub>

**2-3** ④ 里位置 2（`are`，被屏蔽）梯度是 `0.0000`。
如果我**不加 mask**，但把整条 loss 乘个 `0.1`，位置 2 的梯度会是 0 吗？为什么？
<sub>考点：乘系数改变梯度的【大小】不改变【有无】；`ignore_index` 是把该位置整个剔出平均，
所以 `d(loss)/d(logits[2])` 精确为 0</sub>

---

## train3 — 两段损失

**3-1** 「合起来算」时 traj 段只占 **33%**，「拆开算」变成 **50%**。
33% 这个数**具体是从哪来的**？
<sub>考点：4/12 —— 被 token 数量决定，而不是被「这段重不重要」决定</sub>

**3-2** 拆开算是 `mean(cot 8个) + mean(traj 4个)` —— 注意是**相加**，不是加权平均。
那么 traj 段的总权重是多少？cot 段呢？这个和「按 token 数加权平均」差在哪？
<sub>考点：两段各占 1.0（相等），拆开 = 强行让轨迹段和文本段平起平坐</sub>

**3-3** 你 stage13 里的 `loss = cot_loss + fm_loss` 和这里的「两段损失」是同一招吗？
想清楚：`cot_loss` 是对哪些位置平均的？`fm_loss` 是对什么平均的？
<sub>考点：同一招——每段各自求平均、再相加，等于人为指定各段权重</sub>

---

## train4 — 轨迹怎么变成 token

**4-1** 你看到序列里出现 `<i32>`，能判断它属于哪个场景吗？为什么？
<sub>考点：不能。三个场景的第一个 token 都是 `<i32>`（accel 都是 0）。
含义在【整串】里，不在单个 token 上</sub>

**4-2** 真实里一条轨迹是几个 token？**128 这个数是怎么来的**？
<sub>考点：64 waypoint × 2 维（加速度、曲率）。toy 里是 8 × 2 = 16 个</sub>

**4-3** `T.encode` 返回的是 bin 号（0~63），塞进序列前为什么要 `+ V["<i0>"]`？
<sub>考点：`<i0>` 不在词表下标 0（前面还有 12 个固定 token）。
真实里这一步叫 `tokenizer.traj_token_start_idx`</sub>

**4-4** 真实为什么用 3000 个 bin，不用 64 个？
<sub>考点：**只有一个理由——精度**。bin 数直接决定分辨率，进而决定最大误差（≈ 分辨率的一半）。
真实分辨率 0.0045 m/s² / 0.000174 1/m；64 个 bin 会粗约 68 倍。
代价上：词表从 ~15 万涨到 ~15.3 万（只多 2%），而轨迹 token 只占序列里 128 个位置——
所以「贵」得起。
⚠️ **别答成**「128 个 token 一起表达，所以单个 bin 可以不准」——
那**是错的**：误差只由 num_bins 决定，和 token 总数无关（train4 里实测过）</sub>

---

## train5 — 阶段一完整 SFT

**5-1** `others_mask = cot | eos_mask` —— `<eos>` 为什么要单独 `|` 进来？
**注意 train2 里 `<cot_end> → <eos>` 是 ✗**（不算 loss），两个文件对 `<eos>` 的处理不一致。
哪个对？
<sub>考点：真实代码里有两套 mask——`get_label_mask` 管区间，`get_assistant_mask` 管
assistant 回复的 eos（注释写着 "include the eos (+1) for supervision"）。train5 更接近真实</sub>

**5-2** 图片 embedding 拼在序列**最前面**，而 `head` 输出时又 `x[:, 1:, :]` 把第一个位置切掉。
为什么可以切？不切会怎样？
<sub>考点：切掉的是「图片位置」的输出。图片只是被「看」，不需要被「预测」；
不切的话 logits 会多一位，和 labels 对不上</sub>

**5-3** 训练时：**图片位置**的输出参与 loss 吗？**`<bos>` 位置**的输出参与吗？
还有——`<bos>` 本身作为**目标**被预测过吗？（三个小问，答案不一样）
<sub>考点：图片输出被切掉了，不参与；`<bos>` 的输出【参与】（用来预测 `<cot_start>`）；
但 `<bos>` 从不作为目标（labels 位置 0 是 IGNORE）</sub>

---

## train6 — flow matching

**6-1** flow matching 为什么**不直接让模型输出轨迹**，而要绕一圈学「速度场」？
<sub>考点：直接回归只会得到「条件均值」——多个合理的未来会被平均成
一条不存在的轨迹（推理教程 stage8 讲过）</sub>

**6-2** `v_target = x1 - x0` 和 `t` 有关系吗？这说明什么？
<sub>考点：**没有关系**——直线路径上速度恒定。这是 FM 比 DDPM 简单的地方
（DDPM 要预测的是和 t 有关的东西）</sub>

**6-3** 采样 `x ← x + dt · v_model(x, t)` 里有随机性吗？那「生成多样性」从哪来？
<sub>考点：没有随机性——同样的起点一定得到同样的终点。
多样性在【起点】上做（初始噪声）</sub>

**6-4** 训练时 t 从 U(0,1) 采，采样时只积到 t=0.5 会怎样？
<sub>考点：只走完路径一半，停在噪声和数据之间。
模型没错，是你没走到终点。这类「训练/推理必须对齐」的坑不止这一处</sub>

---

## train7 — SFT 阶段二（冻结 VLM + 训 expert）

**7-1** `train_expert` 里 expert 的**输入**是什么？它有 `condition` 参数吗？
<sub>考点：真实 `sft_alpamayo_r1.py:125` 的 forward 里，expert 收的是
`inputs_embeds=action_embeds` + `past_key_values=kv_cache`</sub>

**7-2** 为什么必须 `kv_cache.crop(...)`？**不裁**会怎样？（提示：训练时的序列里有什么）
<sub>考点：teacher forcing 的序列里 `<traj_future_start>` 后面跟着【真值轨迹 token】，
不裁就是把标准答案放进了 expert 的前缀里</sub>

**7-3** 本课实测里，「不裁」那一版的**训练 fm_loss 和「裁」几乎一样**，但推理时明显更差。
为什么训练 loss 看不出来？
<sub>考点：训练时 CoC 和轨迹 token 永远一致，expert 没有理由偏爱任何一个，
于是学成「一半看 CoC、一半抄答案」——两边都用上了，loss 自然不会变差</sub>

**7-4** `cotrain_vlm=False` 时 VLM 的梯度是多少？靠哪**两件**事保证？
<sub>考点：`torch.no_grad()` 包住 VLM forward + `cache.detach()`（`stop_grad_from_vlm`）</sub>

**7-5** 既然 `cotrain_vlm=False` 已经用了 `torch.no_grad()`，那 `stop_grad_from_vlm`
（把 cache 的 K/V detach）是不是多余的？它真正防的是哪种情况？
<sub>考点：`cotrain_vlm=True` 时 forward 不在 no_grad 里，cache 是带图的；
这时 detach 才是必须的——它让【expert 的 loss 不回流到 VLM】，
而 VLM 仍然能被 `vlm_outputs.loss` 训练。两个开关各管一条路</sub>

---

## train8 — 三段课程与训练技巧

**8-1** 三段课程分别是什么？为什么 LingoQA 排在最前？
<sub>考点：stage1_lingoqa（通用 VQA，先建立视觉-语言对齐）→ stage1_nav（驾驶推理 + 轨迹 token）
→ stage2_nav（冻结 VLM，训 expert）</sub>

**8-2** `lr_multiplier: {vlm.model.visual: 0.1}` —— 为什么**视觉塔**只给 1/10 学习率？
<sub>考点：视觉特征来自预训练、最通用也最容易被打坏；驾驶数据规模远小于预训练，
大 lr 会把 ViT 拉偏。语言塔要学新的 CoC 表达，得给足 lr</sub>

**8-3** 如果同时写了 `{vlm: 0.5, vlm.model.visual: 0.1}`，视觉塔的参数会用哪个倍率？
<sub>考点：**最长前缀匹配**——用更长的那条（0.1）</sub>

**8-4** `stage1_vlm_checkpoint_path` 交接时，哪些权重被换掉、哪些保持原样？
<sub>考点：只换 VLM；expert 保持发布模型（`nvidia/Alpamayo-1.5-10B`）里那份。
（RL 那条线反过来：只导出 VLM，拿回 SFT 时 expert 随机初始化）</sub>

---

## train9 — 显存与分布式

**9-1** 训练时一个参数占几份显存？**哪一块最大**？
<sub>考点：权重 2B + 梯度 2B + 优化器状态 12B（fp32 主权重 4 + 一阶动量 4 + 二阶动量 4）
= 16 字节；优化器状态占 75%</sub>

**9-2** ZeRO 1 / 2 / 3 分别把什么切开？
<sub>考点：1 切优化器状态；2 再切梯度；3 再切参数本身</sub>

**9-3** 为什么 stage2 的 config 是 `deepspeed: null`（而 stage1 用 zero2）？
<sub>考点：VLM 冻结 → 8B 参数不要梯度也不要优化器状态，只占权重；
开销大头只剩 expert 的 ~27GB，一张 80GB 卡放得下</sub>

**9-4** `ddp_find_unused_parameters` 为什么只在 **stage2** 是 `true`？
<sub>考点：VLM 冻结 → 它的参数永远拿不到梯度。DDP 必须开这个开关去搜索
「哪些参数没被用到」，否则直接报错。stage1 全模型都参与 loss，不需要</sub>

**9-5** 既然要更大的 batch，为什么不直接把 `micro_batch_size` 调大，
而要用 `gradient_accumulation_steps`？
<sub>考点：【激活】显存 ∝ micro_batch_size × 序列长度，是唯一随 batch 线性涨的一块；
梯度累积把等效 batch 做大，但同一时刻只有一个 micro-batch 的激活在显存里。代价是慢</sub>

---

## train10 — 逐行对照

**10-1** 默写阶段二 forward 的**五步**（不必逐字，说清顺序和每步在干嘛）。
<sub>考点：fuse_traj_tokens → labels(掩码) → VLM forward(no_grad) 拿 past_key_values
→ crop + detach → expert(动作, cache) → action_out_proj → flow matching loss</sub>

**10-2** 为什么说 `alpamayo-recipes` 是「**在已发布模型上继续训**」而不是「从头训一个 Alpamayo」？
<sub>考点：`from_alpamayo_checkpoint` 的 docstring——按 config 建 VLM 架构（不加载 base 权重），
再加载 checkpoint 里 fine-tuned 的 `vlm.*` 权重；`Qwen/Qwen3-VL-8B-Instruct` 只用来建
tokenizer / processor 的架构</sub>

---

# 参考答案

<details>
<summary>点开查看（先自己答完再看）</summary>

**1-1** 必须错开。`logits[i]` 是**看着 `input_ids[i]`** 算出来的（train1 里更是直接
`head(embed(input_ids))`，第 i 个输出就是第 i 个输入的函数）。
不错位的话，模型只要学会「把当前位置的输入抄一遍」就能把 loss 打到 0，**根本没在预测**。
移位后 `logits[i]` 要预测 `input_ids[i+1]`，而这个 token 它没见过，只能靠上下文推。

**1-2** 随机初始化时 logits 接近相同，softmax 出来接近**均匀分布** `1/V`，交叉熵
`-log(1/V) = ln V`。V=11 → `ln 11 = 2.398`，实测 `2.6187`。
实测**略大于** ln V 是因为 logits 不是完全均匀的：`-log` 是凸函数，由 Jensen 不等式
`E[-log p] ≥ -log E[p]`，只要 p 不是严格均匀，CE 就 ≥ ln V。
这个检查的意思是：初始 loss ≈ ln(词表大小) ⟹ 模型「均匀地懵逼」，没有初始化偏置。
V=100 → `ln 100 = 4.605`。

**1-3** 不能。单条样本训 300 步，模型基本是**背下来**了，这**不能**说明任何能力。
它实际说明的是**训练链路是通的**：梯度能流、loss 能降、`argmax` 贪心解码能跑出完整序列。
这是冒烟测试。（另外这里贪心解码**没用 KV cache**——每次都把 `gen` 整个重喂；
stage13 的 `generate` 是同一个东西，只是多了 cache。）

**2-1** mask 含 `<cot_end>` ⟹ `<cot_end>` 成为**目标** ⟹ 模型被监督着学会**输出**它
⟹ 生成时知道该停。**只 mask 中间内容词的话**，`<cot_end>` 永远不是目标，
生成时**根本停不下来**，一路跑到 `max_len`。
`<cot_start>` 同理：监督它 = 学会从「读指令」切换到「开始推理」。
这两个特殊 token 不是「顺便」算进去的，它们是**「何时开始 / 何时停」的唯一监督信号**。

**2-2** 被监督的是 **10 号位置的 label**。机制在 `shift_labels = labels[:, 1:]`——
目标永远取 `i+1`。对应行为：看着 `right lane`（到位置 9），输出 `<cot_end>`（位置 10）。

**2-3** 不会是 0。**乘系数改变的是梯度的「大小」，不是「有无」**——`0.1 × (非零) ≠ 0`。
而加了 mask 时梯度**精确等于 0.0000**（不是「很小」）：`cross_entropy(..., ignore_index=-100)`
把 label=-100 的位置**从平均里整个剔除**，loss 里根本没有一项涉及位置 2 的 logits，
所以 `d(loss)/d(logits[2]) = 0`。乘系数做不到这件事。

**3-1** `4/(4+8) = 1/3`。关键在于：混合平均时**每个位置权重都是 1/12**，
traj 段 4 个位置加起来就是 `4 × 1/12 = 4/12 = 33%`。
所以 **traj 段的总权重 = traj 的 token 数 / 总 token 数**——完全由「这段碰巧有几个 token」决定，
**跟这段重不重要毫无关系**。这就是真实代码要拆开算的原因。

**3-2** 两段**各占 1.0**，不是各占 0.5。看数字：

```
mean(cot 8个) + mean(traj 4个) = 2.8066 + 2.8668 = 5.6734   ← 代码给的（纯相加）
(mean(cot) + mean(traj)) / 2   = 2.8368                     ← 「相加再平均」会得到这个
```

代码给的是 **5.6734 = 纯相加，没有再平均**。所以每段的绝对权重是 **1.0**，
两段的**相对比例是 1:1**。

**这个区别实际影响什么**：只影响 loss 的绝对大小（2 倍），不影响比例。
而比例才是关键——`mean_cot + mean_traj` 和 `0.5·mean_cot + 0.5·mean_traj`
**梯度方向完全一样**，只差 2 倍大小（等价于把 lr 翻倍）。

**3-3** 结构上**是同一招**（每段各自求平均、再相加），但**两段的性质不同**：

| | train3 的两段 | stage13 的两段 |
|---|---|---|
| 段一 | cot 的 8 个 token | `cot_loss`：对**非屏蔽的 token 位置**求平均 |
| 段二 | traj 的 4 个 token | `fm_loss`：对 **batch × 64 waypoint × 2 维**求平均 |
| 差别 | 两段都是**同一条序列里的 token**，只是位置不同 | 一段是**离散 token 的 CE**，一段是**连续回归的 MSE**，对象根本不是一类 |

train3 是把一条序列切成两段；stage13 是把两种完全不同的损失绑在一起。
**两个「平均」的分母完全不同**，所以那个 1:1 的比例是**人为设定**的，不是从数据里算出来的。

**5-1** **(a) 为什么 `<eos>` 必须算 loss**：和 2-1 的 `<cot_end>` 同一个道理——
一个 token 只有成为**目标**，模型才会被教着**输出**它。`<eos>` 不算 loss ⟹
模型从没学过吐 `<eos>` ⟹ **生成时不知道何时结束**，一路跑到 `max_len`。

**(b) 哪个对**：**train5 对**。真实代码 `get_assistant_mask` 的注释原文：
*"Offsets: skip the bos + 'assistant\n' (always 3 tokens) and include the eos (+1) for supervision"*。

**(c) train2 不算错，是只讲了一半**：真实代码有**两套 mask**——
`get_label_mask` 按 `{label}_start`/`{label}_end` 划**区间**（train2 对应这套），
`get_assistant_mask` 单独管 assistant 回复里的 `<eos>`（train5 补上这套）。

**5-2** 图片是**额外拼在最前面**的，`input_ids` 有 L 个 token（现在是 24，
因为轨迹是 16 个 token 而不是 1 个）。
**不切的话** `head(x)` 给出 **L+1 个** logits，而 `labels` 只有 L 个：
`logits[:, :-1]` → L 个，`labels[:, 1:]` → L-1 个，**「错一位」直接崩掉**。

**为什么可以切**：本质是**图片位置是「输入」，不是「要被预测的东西」**——
它只有「被看」这一个身份，没有对应的监督目标，输出本来就不该进 loss。
（对比 `<bos>`：它也是特殊的，但它的**输出要被用**，见 5-3。）

**5-3** 三个小题答案不同，关键是把「**输出**」和「**目标**」分开：

| 位置 | 输出参与 loss 吗 | 作为目标被预测过吗 |
|---|---|---|
| 图片 | ❌ 被 `x[:, 1:, :]` 切掉了 | —（根本不在 `input_ids` 里） |
| `<bos>` | ✅ `logits[0]` 经 shift，目标是 `<cot_start>` | ❌ `labels[0]` 是 IGNORE，且 `shift_labels` 从位置 1 才开始 |

**4-1** **不能。** 关键不是「要 decode」，而是「**一个 token 不够**」。
看 train4 的输出：三个场景的**第一个 token 都是 `<i32>`**（accel 都是 0 → 都落在 bin 32）。
而且 `decode([<i32>])` 也没意义——你会得到一个孤立的数，不知道它对应哪个 waypoint、哪一维。
**含义在整串里**：看第二个 token 才知道方向（`<i41>` 左 / `<i22>` 右 / `<i32>` 直）。
这正是「`<iK>` 是量化刻度，不是类别标签」的直接证据。

**4-2** `64 waypoint × 2 维 = 128`。两维是**加速度**和**曲率**。
（train4 的 toy 是 `8 × 2 = 16`。）

**4-3** 不是 prefix 的问题，是**词表偏移**。`train4_traj_tokens.py` 的词表：

```python
VOCAB = ["<pad>", "<bos>", "<eos>", "<cot_start>", "<cot_end>",
         "<traj_start>", "<traj_end>",
         "turn", "left", "right", "go", "straight"]   # ← 12 个固定 token
        + T.bin_names()                                # ← <i0>…<i63> 从下标 12 开始
```

`T.encode` 返回 bin 号 **0~63**，但词表里 `<i0>` **不在下标 0**（前面占了 12 个位置），
所以要 `+ V["<i0>"]` 才对得上。
真实里这一步有名字：**`tokenizer.traj_token_start_idx`**（`base_model.py:274`）：

```python
tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
```

**4-4** **只有一个理由——精度**，不是两层理由。

bin 数决定分辨率，分辨率决定最大误差（≈ 分辨率的一半）：

| num_bins | κ 分辨率 | 每个 waypoint 的 κ 误差 |
|---|---|---|
| 64（toy） | 0.010476 | 0.000476 |
| 3000（真实） | 0.000220 | 0.000023 |

真实的分辨率换算成物理量是 **0.0045 m/s² / 0.000174 1/m**
（`alpamayo1.5/tutorials/real4_traj_roundtrip.py` 打印的）。64 个 bin 粗约 68 倍。

那为什么「贵」得起？代价只有两处：词表从 ~15 万涨到 ~15.3 万（多 2%），
轨迹 token 也只占序列里 128 个位置。

⚠️ **一个容易想歪的说法**：「128 个 token 一起表达一条轨迹，所以单个 bin 可以不准」——
**这是错的**。train4 实测过：误差只由 `num_bins` 决定，和 token 总数无关
（num_bins=64 时误差 0.000476，=3000 时 0.000023，两次都是 16 个 token）。

**6-1** 因为**直接回归只会得到「条件均值」**。MSE 回归的最优解就是条件期望
`E[轨迹 | 条件]`。看到路口时真值可能是左转也可能是右转，回归会把两簇都学一点，
输出**中间那条**——而那是两簇的均值，**可能压根不是一条合法的轨迹**。

所以不是「少了点多样性」，是「输出了一条不存在的路」（推理教程 stage8 讲过）。
用速度场 + 从**不同噪声**出发积分，才能表达「多峰」这件事。

**6-2** **和 t 无关**。看代码：

```python
t   = torch.rand(batch, 1)
x_t = (1 - t) * x0 + t * x1                # ← x_t 和 t 有关
loss = F.mse_loss(model(x_t, t), x1 - x0)  # ← v_target 和 t 无关
```

对同一对 `(x0, x1)`，**无论 t 取多少，`v_target` 都是同一个值**。

「t 表示修到哪一步」是对 t **含义**的正确描述，但不是这题的答案。要点在于：
`x_t` 和 t 有关（路径上的位置），`v_target` 和 t 无关（直线路径速度恒定），
但**模型仍然要接收 t** —— 它得知道自己在路径的哪一段，才能从 `x_t` 反推 `x0`。

这正是 **FM 比 DDPM 简单的地方**：DDPM 要预测的量带 t 相关的噪声调度系数。

**6-3** **没有随机性**。「`x` 里面有噪声」和「这一步是随机的」不是一回事：

| | 是随机的吗 |
|---|---|
| `x` 的**起点**（t=0 的初始值） | ✅ `torch.randn` 采的 |
| 更新式 `x ← x + dt·v_model(x, t)` | ❌ **完全确定**，就是一个 ODE 积分 |

给定同一个起点，**整条积分轨迹完全确定**。多样性全在**起点**上。

（对比 DDPM：它的反向每一步都**额外加噪声**，那才是「每步都有随机性」。）

**6-4** 会得到**噪声和数据的中点**：`x_0.5 = 0.5·(x0 + x1)`。
实测（理论值：停在 t 时均值恰好是 `3t`）：

| 积到 t | 实测均值 | 理论均值 `3t` | 标准差 |
|---|---|---|---|
| 1.00 | 2.991 | 3.00 | 0.481 |
| 0.75 | 2.250 | 2.25 | 0.453 |
| 0.50 | 1.497 | 1.50 | 0.572 |
| 0.25 | 0.749 | 0.75 | 0.779 |

注意停在 0.5 时标准差反而**变大**（0.572 > 目标的 0.5）——那是两团的混合，更「糊」。

⚠️ **别把训练和采样搞混**：题目说的是「**训练时** t ~ U(0,1)，**采样时**只积到 0.5」。
**训练完全不受影响**（全程都覆盖了），只有**采样**被截断。所以「后半程没训练过」
是错的——是「后半程没走到」。

**模型没错，是你没走到终点。** 这类「训练/推理必须对齐」的坑不止这一处
（推理教程 stage2 的 `temperature`、stage13 的 `MAX_LEN` 都是）。

</details>
