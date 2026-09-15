"""rl3a: 真实 RL 系统的最小结构。

本课不逐文件读代码，只建立一条数据流：
prompt -> rollout 采样 -> reward -> advantage -> policy 更新 -> 权重同步。
"""

PIPELINE = [
    ("prompt", "数据集给出的图片、历史轨迹和导航信息"),
    ("rollout replica", "用 vLLM 生成一组 completion"),
    ("reward", "把 completion 解码成轨迹，再计算 ADE/comfort/reasoning"),
    ("advantage", "Cosmos-RL 在组内比较 reward，准备 rollout.advantage"),
    ("policy replica", "读取 log-prob，调用 GRPO loss，更新 VLM"),
    ("weight sync", "定期把 policy 权重同步回 rollout replica"),
]


if __name__ == "__main__":
    print("=" * 72)
    print("① Alpamayo RL 的最小数据流")
    print("=" * 72)
    for index, (name, description) in enumerate(PIPELINE, 1):
        arrow = "  │\n  ▼\n" if index < len(PIPELINE) else ""
        print(f"  {index}. {name:<18} {description}")
        print(arrow, end="")

    print("\n② 为什么不是一个普通 Trainer？")
    print("  rollout 生成是慢且不可导的，policy 更新必须等 reward 算完；")
    print("  因此真实系统把采样、训练和调度拆成异步 replica。")
    print("\n③ 三个角色先记住")
    print("  policy replica   负责前向、GRPO loss 和反向")
    print("  rollout replica  负责采样 completion，通常由 vLLM 加速")
    print("  controller       负责派发任务、收集结果、协调权重同步")
    print("\n  下一课 rl3b：沿着这条数据流定位真实代码文件。")
