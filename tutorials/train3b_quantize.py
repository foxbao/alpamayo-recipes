"""train3b: 轨迹怎么变成 token —— 离散轨迹 tokenizer

（补课。建议在 train4 之前或之后读：train4/train5 的序列里那些 `<i32>` 是怎么来的，
  就是本课要讲的事。不读也能跑，但读了才不会有「`<iK>` 是个类别标签」的误解。）

train4 起的困惑：VOCAB 里有 `<i0> <i1> <i2>`，看着像三个「类别」。
真实不是——它们是 **bin 刻度**，而且一条轨迹要 **128 个**。

为什么 VLM 需要这个：语言模型的输出是【下一个 token 的概率分布】，
它吐不出浮点数。所以想让 VLM 直接输出轨迹，就得先把连续的轨迹**量化成 token**。

真实代码（`src/alpamayo1_5/action_space/discrete_action_space.py:47`）就四步：

    action = action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)  ① 轨迹→动作
    action = (action - dims_min) / (dims_max - dims_min)                       ② 归一化到 [0,1]
    action = (action * (num_bins - 1)).round().long()                          ③ 量化
    action = action.clamp(0, num_bins - 1)                                     ④ 夹住
    return action.reshape(B, -1)                                  → (B, tokens_per_trajectory)

decode 是反过来（`discrete_action_space.py:80`），多了第 ① 步的逆运算 action_to_traj。

真实数字（release config，由 `alpamayo1.5/tutorials/real4_traj_roundtrip.py` 打印）：
    num_bins = 3000,  dims = [[-10, -10], [10, 10]]（动作是【归一化后】的量，两维都写成 ±10）
    未来 token 数 = 64 waypoint × 2 维 = 128

本课实测：
    ① 一条轨迹走完四步，逐步打印中间值
    ② 三个场景各自变成什么 token 序列
    ③ decode 回来，看量化误差
    ④ ★ num_bins 的权衡：词表大小 ↔ 精度，真实为什么选 3000

对应真实代码：src/alpamayo1_5/action_space/discrete_action_space.py
            src/alpamayo1_5/models/base_model.py:271（造 <i0>…<i3999>）
            src/alpamayo1_5/models/alpamayo1_5.py:44 ExpertLogitsProcessor（推理时屏蔽它们）

【简化】纯 CPU、秒级。toy 用 8 waypoint / num_bins=64；真实是 64 waypoint / 3000。
"""

import torch

import toy_traj_tokens as T

REAL_NUM_BINS = 3000
REAL_TOKENS_PER_TRAJ = 128


def encode_stepwise(action, num_bins):
    """把 encode 的四步拆开，逐步返回中间值，方便打印。"""
    lo = torch.tensor(T.DIMS_MIN)
    hi = torch.tensor(T.DIMS_MAX)
    norm = (action - lo) / (hi - lo)                 # ②
    scaled = norm * (num_bins - 1)
    rounded = scaled.round().long()                  # ③
    clamped = rounded.clamp(0, num_bins - 1)         # ④
    return norm, scaled, rounded, clamped


if __name__ == "__main__":
    print("=" * 74)
    print("① 四步走一遍（以「左转」场景的第一个 waypoint 为例）")
    print("=" * 74)
    action = torch.zeros(1, T.N_WAYPOINTS, 2)
    action[0, :, 1] = 0.10                            # κ = +0.10
    wp = action[0, 0]                                 # 第一个 waypoint: (accel=0, κ=0.10)

    norm, scaled, rounded, clamped = encode_stepwise(action, T.NUM_BINS)
    print(f"  waypoint 0 的动作       : accel={wp[0]:.4f}, κ={wp[1]:.4f}")
    print(f"  dims_min / dims_max     : {T.DIMS_MIN} / {T.DIMS_MAX}")
    print()
    print(f"  {'维度':<10}{'① 原值':>10}{'② 归一化':>12}{'③ ×(N-1)':>12}{'④ round':>10}{'clamp':>8}")
    print("  " + "-" * 62)
    for d, name in enumerate(["accel", "κ"]):
        print(f"  {name:<10}{wp[d]:>10.4f}{norm[0,0,d]:>12.4f}"
              f"{scaled[0,0,d]:>12.4f}{rounded[0,0,d]:>10d}{clamped[0,0,d]:>8d}")
    print(f"\n  num_bins = {T.NUM_BINS}，所以归一化值 ×{T.NUM_BINS-1} 就是「第几个 bin」")
    print(f"  → accel 落在 bin {clamped[0,0,0]}，κ 落在 bin {clamped[0,0,1]}")
    print(f"  → 对应的 token 名就是 <i{clamped[0,0,0]}> 和 <i{clamped[0,0,1]}>"
          f"（真实里是 <i0>…<i{REAL_NUM_BINS-1}>）\n")

    # ── ② 三个场景的完整 token 序列 ──
    print("=" * 74)
    print("② 一条轨迹 = 一串 token（不是一个！）")
    print("=" * 74)
    scenes = [("左转", +0.10), ("右转", -0.10), ("直行", 0.00)]
    print(f"  每个 waypoint 出 {T.DIMS} 个 token（accel, κ），"
          f"{T.N_WAYPOINTS} 个 waypoint → {T.TOKENS_PER_TRAJ} 个 token")
    print(f"  真实：64 waypoint × 2 = {REAL_TOKENS_PER_TRAJ} 个\n")
    for name, kappa in scenes:
        a = torch.zeros(1, T.N_WAYPOINTS, 2)
        a[0, :, 1] = kappa
        ids = T.encode(a)[0].tolist()
        print(f"  {name:<4} κ={kappa:+.2f}")
        print(f"       <traj_start> {' '.join(f'<i{v}>' for v in ids)} <traj_end>")
    print("\n  ★ 注意看：三个场景【第一个 token 都是 <i32>】——它们 accel 都是 0。")
    print("     差别只在第二个 token（κ 的 bin）：<i41> / <i22> / <i32>。")
    print("     这就是「一个 token 没有语义」的直接证据：<i32> 单独出现时，")
    print("     你分不清它属于哪个场景。含义在【整串】里。\n")

    # ── ③ 量化误差 ──
    print("=" * 74)
    print("③ decode 回来：量化是有损的")
    print("=" * 74)
    print(f"  {'场景':<6}{'目标 accel':>12}{'解码后':>12}{'误差':>10}"
          f"{'目标 κ':>10}{'解码后':>10}{'误差':>10}")
    print("  " + "-" * 70)
    for name, kappa in scenes:
        a = torch.zeros(1, T.N_WAYPOINTS, 2)
        a[0, :, 1] = kappa
        back = T.decode(T.encode(a))
        print(f"  {name:<6}{0.0:>12.4f}{back[0,0,0]:>12.4f}{(back[0,0,0]-0).abs():>10.4f}"
              f"{kappa:>10.4f}{back[0,0,1]:>10.4f}{(back[0,0,1]-kappa).abs():>10.4f}")
    print(f"\n  分辨率（每个 bin 跨多大）：accel {T.resolution()[0]:.4f} m/s², "
          f"κ {T.resolution()[1]:.6f} 1/m")
    print("  → 最大误差 ≈ 分辨率的一半。accel 误差大是因为 toy 只有 64 个 bin")
    print("     却要覆盖 ±9.8 这么宽的范围。\n")

    # ── ④ num_bins 的权衡 ──
    print("=" * 74)
    print("④ ★ num_bins 的权衡：词表大小 ↔ 精度（真实为什么选 3000）")
    print("=" * 74)
    a = torch.zeros(1, T.N_WAYPOINTS, 2)
    a[0, :, 1] = 0.10
    print(f"  {'num_bins':>10}{'accel 分辨率':>16}{'κ 分辨率':>14}{'最大误差(κ)':>14}"
          f"{'词表占比':>12}")
    print("  " + "-" * 66)
    for nb in [8, 16, 64, 256, 1000, REAL_NUM_BINS]:
        lo = torch.tensor(T.DIMS_MIN)
        hi = torch.tensor(T.DIMS_MAX)
        norm = (a - lo) / (hi - lo)
        ids = (norm * (nb - 1)).round().long().clamp(0, nb - 1)
        back = ids.float().reshape(-1, T.N_WAYPOINTS, 2) / (nb - 1) * (hi - lo) + lo
        err_k = (back[0, 0, 1] - 0.10).abs().item()
        res_a = (hi[0] - lo[0]).item() / (nb - 1)
        res_k = (hi[1] - lo[1]).item() / (nb - 1)
        mark = "  ← 真实" if nb == REAL_NUM_BINS else ""
        print(f"  {nb:>10}{res_a:>16.5f}{res_k:>14.7f}{err_k:>14.6f}"
              f"{nb/150000:>11.2%}{mark}")
    print("\n  → 词表不是白给的：num_bins 每翻一倍，精度翻一倍，但词表也多一份。")
    print("     真实选 3000，是因为要在「轨迹够准」和「词表别太大」之间平衡")
    print("     （相对 ~15 万的自然语言词表，3000 只占约 2%）。")
    print("  → 而且真实还有第二层：**128 个 token 描述一条轨迹**——")
    print("     单个 bin 不需要多准，因为整条轨迹是 128 个 bin 一起表达的。")
    print()
    print("=" * 74)
    print("⑤ 这串 token 在模型里怎么用")
    print("=" * 74)
    print("  · VLM 自回归【生成】它们（通路 A）—— 这就是 RL 训的那条路，")
    print("    因为离散 token 的 log-prob 好算，策略梯度直接可得。")
    print("  · 发布推理走通路 B（连续 expert）时，`ExpertLogitsProcessor` 会把这")
    print(f"    {REAL_NUM_BINS} 个 token 全 mask 成 -inf，生成【停在 <traj_future_start>】，")
    print("    交给 flow-matching expert 出连续动作。")
    print("  · train5 里 crop 掉的「真值轨迹 token」，真实里就是这 128 个。")
