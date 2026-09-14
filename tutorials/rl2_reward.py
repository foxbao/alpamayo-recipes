"""rl2: reward 怎么设计 —— ADE + comfort，以及那个「gate」

【前置】rl1（GRPO 为什么需要 reward）。本课讲 reward 本身。

真实代码两份（`recipes/alpamayo1_x_rl/rewards/`）：

    aggregated_reward.py                  只打分轨迹 —— README 说这是【demo 版】
    aggregated_reward_with_reasoning.py   还打分 CoC 推理 —— Alpamayo 1.5 实际用的是这版

两者结构一样，后者多一项 reasoning。本课先讲清轨迹那部分，最后说 reasoning。

真实代码（`aggregated_reward.py` 的 `compute_reward`）：

    l2_dist = calculate_ade(pred, gt)                      ① ADE：XY 上的平均 L2
    comfort_score = mean(compute_comfort(...)) - 1.0       ② comfort：6 个指标，∈[-1,0]
    ade_threshold = 3.0
    if l2_dist < ade_threshold:                            ③ gate
        final_reward = (-0.4 * (l2_dist / 3.0)             ④ 轨迹项
                        + 0.1 * comfort_score)             ⑤ comfort 项
    else:
        final_reward = -1.0                                ⑥ 撞上 gate：直接 -1

权重来自 TOML `[custom.alpamayo.reward]`：traj_l2_weight = 0.4, comfort_weight = 0.1

★ 三个容易看漏的点，本课都会实测：
    ① comfort 不是「比例」——README 说「Fraction of timesteps within bounds」，
       但代码是 `torch.all(...)`：**6 个指标每个必须【全程】在界内才得 1 分**。
       文档和代码不一致，代码为准。
    ② 两项都【除以了自己的阈值】（l2/3.0、comfort 已经是归一化的），
       所以两段量纲可比，权重才说得通。
    ③ l2 = 3.0 处有一个【跳变】：从约 -0.5 直接掉到 -1.0。

【简化】纯 CPU、秒级。用合成的轨迹/动力学数据，不跑真模型。
"""

import torch

# 真实 TOML 里的权重
TRAJ_L2_WEIGHT = 0.4
COMFORT_WEIGHT = 0.1
ADE_THRESHOLD = 3.0
REASONING_THRESHOLD = -0.4

# 真实 comfort 的 6 个指标（`comfort_reward.py`）
COMFORT_METRICS = [
    ("comfort/lon_accel", -4.05, 2.40, "m/s²"),
    ("comfort/lat_accel", -4.89, 4.89, "m/s²"),
    ("comfort/jerk", -8.37, 8.37, "m/s³"),
    ("comfort/lon_jerk", -4.13, 4.13, "m/s³"),
    ("comfort/yaw_accel", -1.93, 1.93, "rad/s²"),
    ("comfort/yaw_rate", -0.95, 0.95, "rad/s"),
]


def calculate_ade(pred_xyz, gt_xyz):
    """真实 `traj_reward.py:calculate_ade` —— 只取 XY，逐 waypoint 求距离再平均。"""
    d = torch.linalg.norm(pred_xyz[..., :2] - gt_xyz[..., :2], dim=-1)
    return float(d.mean())


def comfort_score(metric_values):
    """真实 `comfort_reward.py` 的判据：每个指标【全程】在界内才得 1，否则 0。

    注意是 `torch.all` —— 不是「在界内的时间比例」。
    """
    scores = []
    for (name, lo, hi, _), vals in zip(COMFORT_METRICS, metric_values):
        scores.append(1.0 if bool(((vals > lo) & (vals < hi)).all()) else 0.0)
    return sum(scores) / len(scores) - 1.0        # ∈ [-1, 0]


def reward(l2_dist, comfort, use_gate=True,
           traj_w=TRAJ_L2_WEIGHT, comfort_w=COMFORT_WEIGHT):
    """真实 `aggregated_reward.compute_reward` 的聚合逻辑（不含 reasoning）。"""
    if use_gate and not (l2_dist < ADE_THRESHOLD):
        return -1.0
    return -traj_w * (l2_dist / ADE_THRESHOLD) + comfort_w * comfort


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 74)
    print("① ADE —— 轨迹项")
    print("=" * 74)
    print("  真实 `calculate_ade`：只取 XY、逐 waypoint 求 L2、再平均\n")
    gt = torch.tensor([[[i * 1.0, 0.0, 0.0] for i in range(10)]]).float()
    pred_close = gt + torch.tensor([0.0, 0.5, 0.0])          # 横向偏 0.5m
    pred_far = gt + torch.tensor([0.0, 2.5, 0.0])            # 横向偏 2.5m
    for name, p in [("横向偏 0.5m", pred_close), ("横向偏 2.5m", pred_far)]:
        print(f"  {name:<14} ADE = {calculate_ade(p, gt):.3f} m")
    print("  （只算 XY，Z 完全不看）\n")

    # ── comfort ──
    print("=" * 74)
    print("② comfort —— 6 个指标，硬边界")
    print("=" * 74)
    print(f"  {'指标':<20}{'下界':>9}{'上界':>9}   单位")
    print("  " + "-" * 52)
    for name, lo, hi, unit in COMFORT_METRICS:
        print(f"  {name:<20}{lo:>9.2f}{hi:>9.2f}   {unit}")
    print()

    T = 10
    ok = [torch.zeros(T, dtype=torch.float64)] * 6            # 全部在界内
    print(f"  全部 6 项都在界内     → comfort_score = {comfort_score(ok):+.2f}")
    one_bad = list(ok)
    one_bad[2] = torch.zeros(T, dtype=torch.float64)
    one_bad[2][-1] = 9.0                                      # 只有【最后一个时刻】jerk 超了
    print(f"  只有 jerk 的最后一帧超 → comfort_score = {comfort_score(one_bad):+.2f}")
    print(f"  全部 6 项都超         → comfort_score = "
          f"{comfort_score([torch.full((T,), 99.0, dtype=torch.float64)] * 6):+.2f}")
    print("\n  ★ 注意第二行：**只超了一帧，这一项就整项作废**（`torch.all`）。")
    print("     README 写的是「Fraction of timesteps within bounds」（比例），")
    print("     但代码不是——它是 all-or-nothing。以代码为准。\n")

    # ── 聚合 + gate ──
    print("=" * 74)
    print("③ 聚合：两项各自归一化，再加，然后撞 gate")
    print("=" * 74)
    print(f"  reward = -{TRAJ_L2_WEIGHT}·(ADE/{ADE_THRESHOLD}) + {COMFORT_WEIGHT}·comfort"
          f"     （当 ADE < {ADE_THRESHOLD}）")
    print(f"  reward = -1.0                                    （当 ADE ≥ {ADE_THRESHOLD}）\n")
    print(f"  {'ADE':>6}{'comfort=0':>14}{'comfort=-0.5':>16}{'comfort=-1':>14}")
    print("  " + "-" * 52)
    for l2 in [0.0, 0.5, 1.0, 2.0, 2.5, 2.9, 2.99, 3.0, 3.5, 5.0]:
        row = f"  {l2:>6.2f}"
        for c in [0.0, -0.5, -1.0]:
            row += f"{reward(l2, c):>14.3f}"
        print(row)
    print("\n  ★ 看 ADE = 2.99 → 3.0 那一跳：reward 从约 -0.40 直接掉到 -1.00。")
    print("     这就是 gate 的形状：**过了阈值就没有「差不多好」，一律 -1**。")
    print("     它的作用是把「根本没走对」和「走得还行」彻底分开，")
    print("     避免模型在错误轨迹上做微调。")

    # ── reasoning ──
    print()
    print("=" * 74)
    print("④ 带 reasoning 的那版（Alpamayo 1.5 实际用的）")
    print("=" * 74)
    print("  `aggregated_reward_with_reasoning.py` 多一项，gate 也多一个条件：\n")
    print("      reasoning_score = grader.score(pred_cot, gt_cot) - 1.0     ∈ [-1, 0]")
    print("      # grader = LingoJudgeGrader（wayveai/Lingo-Judge 序列分类器）")
    print("      # pred_cot 是从 rollout 里 extract_between_special_tokens(..., 'cot') 抠出来的")
    print()
    print(f"      if 抠到 CoC  and  reasoning_score > {REASONING_THRESHOLD}  and  ADE < {ADE_THRESHOLD}:")
    print("          reward = -0.4·(ADE/3) + 0.1·comfort + w_reasoning·(reasoning_score / -0.4)")
    print("      else:")
    print("          reward = -1.0")
    print()
    print(f"  → 两个阈值各自归一化：ADE 除 3.0，reasoning 除 {REASONING_THRESHOLD}。")
    print(f"  → reasoning_score 必须 > {REASONING_THRESHOLD}（即原始打分 > "
          f"{1 + REASONING_THRESHOLD:.1f}），否则整个样本 -1。")
    print("  → 这条 gate 的意思是：**推理写得不对，轨迹再好也不给分**。")
    print("     这正是「让 reasoning 真的参与优化」的机制——SFT 阶段做不到这件事")
    print("     （SFT 只能让 CoC 像真值，不能让它'对结果有用'）。")
    print()
    print("  ⚠️ demo 版（`aggregated_reward.py`）**只**打分轨迹，不碰 CoC。")
    print("     README 明确写了这一点，并说 Alpamayo 1.5 是用 reasoning reward 训的。")
    print("     所以想复现 1.5 的效果，光用 demo 的 reward 是不够的。")
