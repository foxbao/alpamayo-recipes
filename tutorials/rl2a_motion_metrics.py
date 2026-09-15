"""rl2a: 运动 reward 的两个原始指标 —— ADE 与 comfort。

本课只做测量，不把指标聚合成 reward。下一课再讲权重和 ADE gate。
"""

import torch

COMFORT_METRICS = [
    ("comfort/lon_accel", -4.05, 2.40, "m/s²"),
    ("comfort/lat_accel", -4.89, 4.89, "m/s²"),
    ("comfort/jerk", -8.37, 8.37, "m/s³"),
    ("comfort/lon_jerk", -4.13, 4.13, "m/s³"),
    ("comfort/yaw_accel", -1.93, 1.93, "rad/s²"),
    ("comfort/yaw_rate", -0.95, 0.95, "rad/s"),
]


def calculate_ade(pred_xyz, gt_xyz):
    """只取 XY，逐 waypoint 求 L2 后平均。"""
    distances = torch.linalg.norm(pred_xyz[..., :2] - gt_xyz[..., :2], dim=-1)
    return float(distances.mean())


def comfort_score(metric_values):
    """每个动力学指标必须在所有时间点都过界内，最后返回 [-1, 0]。"""
    passed = []
    for (_, lower, upper, _), values in zip(COMFORT_METRICS, metric_values):
        passed.append(float(((values > lower) & (values < upper)).all()))
    return sum(passed) / len(passed) - 1.0


if __name__ == "__main__":
    print("=" * 72)
    print("① ADE：轨迹位置偏不偏")
    print("=" * 72)
    ground_truth = torch.zeros(1, 10, 3)
    close = ground_truth + torch.tensor([0.0, 0.5, 0.0])
    far = ground_truth + torch.tensor([0.0, 2.5, 0.0])
    print(f"  横向偏 0.5m -> ADE = {calculate_ade(close, ground_truth):.3f} m")
    print(f"  横向偏 2.5m -> ADE = {calculate_ade(far, ground_truth):.3f} m")
    print("  Z 坐标不会进入 ADE。")

    print("\n② comfort：动作是否越过动力学边界")
    print(f"  {'指标':<22}{'下界':>9}{'上界':>9}   单位")
    print("  " + "-" * 54)
    for name, lower, upper, unit in COMFORT_METRICS:
        print(f"  {name:<22}{lower:>9.2f}{upper:>9.2f}   {unit}")

    time_steps = 10
    all_ok = [torch.zeros(time_steps)] * len(COMFORT_METRICS)
    one_bad = list(all_ok)
    one_bad[2] = torch.zeros(time_steps)
    one_bad[2][-1] = 9.0
    all_bad = [torch.full((time_steps,), 99.0)] * len(COMFORT_METRICS)
    print(f"\n  全部通过       -> comfort_score = {comfort_score(all_ok):+.2f}")
    print(f"  jerk 仅一帧越界 -> comfort_score = {comfort_score(one_bad):+.2f}")
    print(f"  全部越界       -> comfort_score = {comfort_score(all_bad):+.2f}")
    print("\n  注意：真实实现使用 torch.all，不是‘通过时间点的比例’。")
    print("  下一课 rl2b：把 ADE、comfort 和阈值 gate 合成一个 scalar reward。")
