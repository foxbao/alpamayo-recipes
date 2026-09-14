"""train9: 显存与分布式 —— 跑得动吗

train1~6 讲的是「训练在算什么」。这一课讲「训练跑不跑得起来」。

一个参数在训练时占几份显存（以 Adam + bf16 混合精度为例）：

    ┌─ 权重（bf16）            2 字节
    ├─ 梯度（bf16）            2 字节
    └─ 优化器状态（fp32）      12 字节   ← fp32 主权重 4 + 一阶动量 4 + 二阶动量 4
    ─────────────────────────────────
    合计                      16 字节 / 参数

注意【优化器状态是最大的一块】，是权重本身的 6 倍。这就是 ZeRO 要切它的原因。

ZeRO 三个阶段分别把什么切开（切了的部分各卡只存 1/N）：

    ZeRO-1   切优化器状态          → 每卡 2 + 2 + 12/N
    ZeRO-2   再切梯度              → 每卡 2 + 2/N + 12/N
    ZeRO-3   再切参数本身          → 每卡 (2 + 2 + 12)/N

★ 本课能实测的：
    ① 用真实模型的参数名 / 层数，把上面这张表算成具体数字，逐级看 ZeRO 省多少
    ② 为什么 stage1 用 zero2、stage2 却 deepspeed: null
    ③ `ddp_find_unused_parameters: true` 为什么只在 stage2 出现（和冻结 VLM 直接相关）
    ④ gradient_accumulation_steps 在模拟什么

⚠️ 本课的数字是**按公式算的**，不是真的上 GPU 实测的（教程设计就是纯 CPU）。
   假设写在代码里，你可以按自己的卡改。

对应真实代码：
    recipes/alpamayo1_5_sft/configs/deepspeed/zero2.json
    recipes/alpamayo1_5_sft/configs/sft_stage1_nav.yaml     (deepspeed: zero2)
    recipes/alpamayo1_5_sft/configs/sft_stage2_nav.yaml     (deepspeed: null)
    recipes/alpamayo1_5_sft/train_hf.py

【简化】纯 CPU、秒级。
"""

import torch
import torch.nn as nn

GB = 1024 ** 3


# ═══════════════════════════════════════════════════════════════
# 显存公式
# ═══════════════════════════════════════════════════════════════

def param_bytes(n_params, weight_bytes=2, grad_bytes=2, optim_bytes=12):
    """一个训练中的参数占的字节数（权重 + 梯度 + Adam 的 fp32 状态）。"""
    return n_params * (weight_bytes + grad_bytes + optim_bytes)


def zero_per_gpu(n_params, stage, n_gpus, weight_bytes=2, grad_bytes=2, optim_bytes=12):
    """ZeRO-1/2/3 下每卡的模型状态显存（不含激活）。"""
    w = n_params * weight_bytes
    g = n_params * grad_bytes
    o = n_params * optim_bytes
    if stage == 0:                      # 纯 DDP
        return w + g + o
    if stage == 1:
        return w + g + o / n_gpus
    if stage == 2:
        return w + g / n_gpus + o / n_gpus
    if stage == 3:
        return (w + g + o) / n_gpus
    raise ValueError(stage)


def count_params_by_prefix(model, prefix):
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 74)
    print("① 四块显存里的「模型状态」—— 一个参数 16 字节是怎么来的")
    print("=" * 74)
    rows = [("权重 (bf16)", 2), ("梯度 (bf16)", 2),
            ("优化器 fp32 主权重", 4), ("优化器 一阶动量 m", 4), ("优化器 二阶动量 v", 4)]
    total = 0
    for name, b in rows:
        total += b
        print(f"  {name:<24}{b:>3} 字节/参数   累计 {total:>3} 字节")
    print(f"\n  → 合计 {total} 字节/参数，其中优化器状态占 {12}/{total} = {12/total:.0%}。")
    print("     这就是为什么 ZeRO 第一刀就砍它。\n")

    # ── 用真实尺寸算 ──
    print("=" * 74)
    print("② 套到 Alpamayo 的真实尺寸上")
    print("=" * 74)
    # 真实 config 里读到的数字（见 README §训练全景 和各 config）
    VLM_PARAMS = 8.0e9      # Cosmos-Reason2-8B / Qwen3-VL-8B 量级
    EXPERT_LAYERS = 36
    EXPERT_HIDDEN = 2048
    EXPERT_FFN = 8256
    per_layer = 4 * EXPERT_HIDDEN ** 2 + 2 * EXPERT_HIDDEN * EXPERT_FFN
    EXPERT_PARAMS = per_layer * EXPERT_LAYERS
    print(f"  VLM    ≈ {VLM_PARAMS/1e9:.1f}B 参数（Cosmos-Reason2-8B 量级）")
    print(f"  expert ≈ {EXPERT_PARAMS/1e9:.2f}B 参数")
    print(f"           算法: 每层 self_attn 4·H² + MLP 2·H·F = "
          f"4·{EXPERT_HIDDEN}² + 2·{EXPERT_HIDDEN}·{EXPERT_FFN} = {per_layer/1e6:.1f}M")
    print(f"                 × {EXPERT_LAYERS} 层 = {EXPERT_PARAMS/1e9:.2f}B\n")

    # ── 阶段二：只训 expert（VLM 冻结）──
    vlm_frozen = VLM_PARAMS * 2                       # 冻结 → 只要权重，不需要梯度/优化器
    print("  阶段二（train7 那一步）：VLM 冻结，只训 expert\n")
    print(f"  {'':<30}{'每卡模型状态(不含激活)':>20}")
    print("  " + "-" * 46)
    print(f"  {'VLM 权重（冻结，bf16，不参与优化）':<30}{vlm_frozen/GB:>11.1f} GB")
    print(f"  {'expert 全量（= 纯 DDP）':<30}{param_bytes(EXPERT_PARAMS)/GB:>11.1f} GB")
    print("  " + "-" * 46)
    for n_gpu in (1, 8):
        print(f"  {n_gpu} 卡下 expert 的模型状态：")
        for stage in (0, 1, 2, 3):
            m = zero_per_gpu(EXPERT_PARAMS, stage, n_gpu)
            tag = "纯 DDP" if stage == 0 else f"ZeRO-{stage}"
            print(f"     {tag:<12}{m/GB:>11.1f} GB   合计 {(vlm_frozen + m)/GB:>5.1f} GB")
    print("\n  → 注意 ZeRO-1 到 2 省得最多（优化器状态 12 字节 > 梯度 2 字节）；")
    print("     ZeRO-3 把权重也切了，省得更多，但每次前向都要 all-gather，通信开销最大。\n")
    print("  那为什么 stage2 的 config 还是 `deepspeed: null`？")
    print("    ① VLM 冻结 → 它的 8B 参数【不要梯度也不要优化器状态】，")
    print(f"       只占权重 {vlm_frozen/GB:.0f}GB；开销大头只剩 expert 的 {param_bytes(EXPERT_PARAMS)/GB:.0f}GB。")
    print(f"    ② 一张 80GB 卡就放得下（模型状态约 {(vlm_frozen + param_bytes(EXPERT_PARAMS))/GB:.0f}GB，"
          f"不含激活），不需要跨卡切。")
    print("    ③ 省掉了 DeepSpeed 的通信开销，而 stage2 的 batch 更小，本来也不缺这点显存。\n")

    # ── ddp_find_unused_parameters ──
    print("=" * 74)
    print("③ `ddp_find_unused_parameters` 为什么只在 stage2 是 true")
    print("=" * 74)
    print("  对比两个 config：")
    print("     sft_stage1_nav.yaml →  ddp_find_unused_parameters: false")
    print("     sft_stage2_nav.yaml →  ddp_find_unused_parameters: true\n")
    print("  原因很直接：")
    print("     stage1：整个模型都参与 loss，每个参数都拿得到梯度，DDP 不用去找。")
    print("     stage2：VLM 被冻了（cotrain_vlm: false），它的参数【永远不产生梯度】，")
    print("             DDP 必须开这个开关去搜索「哪些参数没用上」，否则会直接报错。")
    print("  → 这个开关是「模型里有一部分不参与前向的 loss 计算」的直接信号。\n")

    # ── 实测一下：冻结的子模块确实没有梯度 ──
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = nn.Linear(4, 4)
            self.expert = nn.Linear(4, 4)

        def forward(self, x):
            return self.expert(self.vlm(x).detach())      # detach = 冻结 VLM

    m = Toy()
    for p in m.vlm.parameters():
        p.requires_grad_(False)
    m(torch.randn(2, 4)).sum().backward()
    print("  实测（toy，VLM 用 detach + requires_grad_(False)）：")
    for name, mod in [("vlm", m.vlm), ("expert", m.expert)]:
        gs = [p.grad is not None for p in mod.parameters()]
        print(f"     {name:<8} 有梯度的参数 {sum(gs)}/{len(gs)}")
    print()

    # ── gradient accumulation ──
    print("=" * 74)
    print("④ gradient_accumulation_steps 在模拟什么")
    print("=" * 74)
    print("  configs 里 stage1 是 gradient_accumulation_steps: 4。")
    print("  它做的事：连续算 4 个 micro-batch 的梯度、【累积】起来不 step，")
    print("           第 4 次之后才 optimizer.step() 一次。")
    print()
    for accum in (1, 4):
        eff = 2 * accum * 8      # micro_bs × accum × 卡数
        print(f"  micro_bs=2, accum={accum}, 8 卡  →  等效 batch = 2×{accum}×8 = {eff}")
    print()
    print("  为什么不用更大的 micro_bs 直接搞定？—— **激活**。")
    print("  激活显存 ∝ micro_batch_size × 序列长度，是唯一随 batch 线性涨的一块；")
    print("  梯度累积把 batch 做大，但同一时刻只有一个 micro-batch 的激活在显存里。")
    print("  → 代价是慢：累积 N 次就是 N 次前向+反向，只是少做了 optimizer.step()。")
