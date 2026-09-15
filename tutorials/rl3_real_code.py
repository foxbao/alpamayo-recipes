"""rl3: RL 逐行对照 —— 收口

【前置】rl1（GRPO 算法）、rl2（reward 设计）。本课把它们映射回真实文件。
不要求你把 `recipes/alpamayo1_x_rl/` 全部读完；先沿着下面的最短路径走，再按需要展开。

真实 RL 代码在 `recipes/alpamayo1_x_rl/`，架构和 SFT 完全不同：

    SFT：一个 Trainer + DataLoader，模型、损失、优化器全在一个进程里
    RL ：policy replica（训） + rollout replica（采样） + controller（调度）
         三者【异步】跑，靠 Cosmos-RL 协调、靠 vLLM 做高速采样

★ 为什么 RL 要这么复杂的架子？
   因为一个训练步需要：给 48 个 prompt 各采 12 条完整轨迹（每条几千 token），
   算 reward，再回去更新。采样量比 SFT 大得多，而且 policy 和 rollout
   要不断同步权重（`sync_weight_interval = 2`）。
   把这套塞进一个进程里是跑不动的——所以拆成独立的 replica。

【本课】
    ① 代码地图：每个文件干什么
    ② rl1/rl2 的每个概念 → 落在哪个文件的哪一段
    ③ SFT vs RL 的关键差别（一张表）
    ④ 最短阅读路径，以及哪些工程文件可以暂时跳过

【简化】纯 CPU、秒级。本课不跑训练，只做映射与对照。
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.abspath(os.path.join(HERE, "..", "recipes", "alpamayo1_x_rl"))

# ═══════════════════════════════════════════════════════════════
# ① 代码地图
# ═══════════════════════════════════════════════════════════════

CODE_MAP = [
    ("入口与配置", [
        ("launcher.py", "共享启动逻辑，初始化 state 后调 Cosmos worker"),
        ("models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py",
         "注册模型 / rollout / trainer / reward —— cosmos-rl 从这里读"),
        ("toml/alpamayo_rvla_rl_local_test.toml",
         "★ 所有超参都在这：组大小、裁剪、lr、reward 权重、replica 数"),
        ("hydra_configs/alpamayo1_5_rvla_rl_pai.yaml", "PAI 数据集与预处理配置"),
    ]),
    ("模型包装", [
        ("models/reasoning_vla/cosmos_wrapper.py", "把模型包成 Cosmos-RL 认识的接口"),
        ("models/reasoning_vla/base_model.py", "策略模型本体"),
        ("models/reasoning_vla/weight_mapper.py", "权重名映射（RL 用的是【离散通路】的模型）"),
        ("utils/weight_loading.py", "加载 SFT checkpoint 转过来的权重"),
    ]),
    ("rollout（采样）", [
        ("models/reasoning_vla/rollout.py", "★ 531 行，rollout 的主逻辑"),
        ("models/reasoning_vla/vllm_wrapper.py", "用 vLLM 加速生成"),
        ("utils/trajectory_decode.py", "把 rollout 出的 token 解码回 xyz 轨迹（reward 要用）"),
        ("utils/logit_processor.py", "MaskDiscreteTrajectoryLogitsProcessor（生成 CoC 时屏蔽轨迹 token）"),
    ]),
    ("训练", [
        ("models/reasoning_vla/trainer.py", "★ 387 行，GRPO 的 loss（调 cosmos_rl 的 compute_loss）"),
        ("utils/fsdp.py", "FSDP 切分"),
    ]),
    ("reward", [
        ("rewards/traj_reward.py", "29 行，就是 `calculate_ade`"),
        ("rewards/comfort_reward.py", "132 行，6 个指标的硬边界"),
        ("rewards/aggregated_reward.py", "98 行，demo 版（只打分轨迹）"),
        ("rewards/aggregated_reward_with_reasoning.py", "132 行，加 reasoning 项"),
        ("utils/light_weight_reasoning_grading_model.py", "246 行，Lingo-Judge 打分器"),
    ]),
    ("数据", [
        ("models/reasoning_vla/data_packer.py", "把样本打包给 rollout"),
        ("prefetch/server.py", "★ 1321 行，节点级预取服务（共享内存广播，44s → 5s）"),
        ("prefetch/shm.py", "共享内存那层"),
    ]),
]

# ═══════════════════════════════════════════════════════════════
# ② 概念 → 代码
# ═══════════════════════════════════════════════════════════════

MAPPING = [
    # (课, 概念, 文件, 定位 / 说明)
    ("rl1", "组大小 G（GRPO 的「组」）", "toml/alpamayo_rvla_rl_local_test.toml",
     "[rollout] n_generation = 12 —— 每个 prompt 采 12 条"),
    ("rl1", "advantages 从哪来", "models/reasoning_vla/trainer.py",
     "`advantages_list = [rollout.advantage for rollout in rollouts]`"),
    ("rl1", "GRPO 的 loss", "models/reasoning_vla/trainer.py",
     "`from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import compute_loss`"),
    ("rl1", "组内归一化", "（在 cosmos_rl 包里，不在本仓库）",
     "advantage 由 Cosmos-RL 算好放进 rollout，trainer 直接取"),
    ("rl1", "不对称裁剪", "toml", "epsilon_low = 0.2 / epsilon_high = 0.28"),
    ("rl1", "没有 KL", "toml", "kl_beta = 0.0"),
    ("rl1", "内层迭代", "toml", "mu_iterations = 1（PPO 那种多 epoch 这里没有）"),
    ("rl1", "同步权重到 rollout", "toml", "train.sync_weight_interval = 2"),
    ("rl2", "ADE", "rewards/traj_reward.py", "`calculate_ade` —— 只有 29 行"),
    ("rl2", "comfort 的 6 个指标", "rewards/comfort_reward.py",
     "`COMFORT_METRIC_CONFIG_DICT` + `_within_bound`（用的是 torch.all）"),
    ("rl2", "gate + 聚合", "rewards/aggregated_reward.py",
     "`if l2_dist < ade_threshold: ... else: final_reward = -1.0`"),
    ("rl2", "reasoning reward", "rewards/aggregated_reward_with_reasoning.py",
     "`reasoning_score = raw - 1.0`，阈值 -0.4，gate 多一个条件"),
    ("rl2", "打分器", "utils/light_weight_reasoning_grading_model.py",
     "`LingoJudgeGrader`（wayveai/Lingo-Judge 序列分类器）"),
    ("rl2", "抽 CoC 文本", "aggregated_reward_with_reasoning.py",
     "`extract_between_special_tokens([...], token='cot')`"),
    ("rl2", "把 token 解码回轨迹", "utils/trajectory_decode.py", "`decode_rollout_trajectory`"),
]

# 这是“先理解算法，再理解系统”的最短路径。完整 CODE_MAP 适合查文件，
# 不适合第一次阅读时从头到尾执行。
MINIMUM_PATH = [
    ("toml/alpamayo_rvla_rl_local_test.toml", "先看 G=12、clip、KL、同步频率等训练旋钮"),
    ("rewards/traj_reward.py", "知道 rollout 如何变成 ADE"),
    ("rewards/aggregated_reward.py", "看 comfort、权重和 ADE gate 如何合成 scalar reward"),
    ("utils/trajectory_decode.py", "确认离散轨迹 token 如何还原成 xyz"),
    ("models/reasoning_vla/rollout.py", "只读 rollout_generation：采样、logprob、reward 前的数据形态"),
    ("models/reasoning_vla/trainer.py", "只读 step_training：old/ref/current logprob 和 GRPO loss"),
    ("models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py",
     "最后看 Cosmos-RL 如何注册模型、rollout、trainer 和 reward"),
]

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 78)
    print("① RL 代码地图")
    print("=" * 78)
    for group, files in CODE_MAP:
        print(f"\n  【{group}】")
        for rel, why in files:
            p = os.path.join(RL, rel)
            n = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
            mark = "✅" if os.path.exists(p) else "❌"
            print(f"    {mark} {rel:<52}{n:>5} 行")
            print(f"       {why}")

    print("\n" + "=" * 78)
    print("② rl1 / rl2 的概念 → 真实代码位置")
    print("=" * 78)
    cur = None
    for lesson, concept, rel, note in MAPPING:
        if lesson != cur:
            print(f"\n  【{lesson}】")
            cur = lesson
        print(f"    {concept}")
        print(f"        {rel}")
        print(f"        {note}")

    print("\n" + "=" * 78)
    print("③ SFT vs RL 的关键差别")
    print("=" * 78)
    rows = [
        ("训练信号", "teacher forcing 的 token 交叉熵", "reward（从采样结果上算）"),
        ("数据从哪来", "数据集里的真值序列", "模型自己采样出来的轨迹"),
        ("一个样本", "一条真值序列", "同一个 prompt 采 12 条（GRPO 的「组」）"),
        ("能不能并行", "整条序列一次前向反向", "必须先生成（不可导），再回传"),
        ("损失里的项", "Stage 1: token CE；Stage 2: flow MSE", "advantage × log p（score function）"),
        ("训什么", "阶段一整个 VLM / 阶段二只训 expert", "★ 只训 VLM backbone（离散通路）"),
        ("要不要 critic", "不需要", "GRPO 也不需要（用组内均值当 baseline）"),
        ("学习率", "1e-5 量级", "2e-6（小一个量级）"),
        ("硬件", "单机可跑", "≥5×80GB；真实配置 64 policy + 128 rollout replica"),
        ("框架", "HF Trainer + DeepSpeed", "Cosmos-RL + vLLM + FSDP"),
    ]
    print(f"  {'':<12}{'SFT':<38}{'RL'}")
    print("  " + "-" * 74)
    for a, b, c in rows:
        print(f"  {a:<12}{b:<38}{c}")

    print("\n  ★ 最容易忽略的一条：「训什么」不同。")
    print("     SFT 阶段二训的是连续 action expert；RL 训的是【VLM backbone】，")
    print("     走的是【离散轨迹 token】那条通路。所以：")
    print("       · RL 导出的 checkpoint 【只含 VLM 权重】")
    print("       · 拿回去 SFT 训 expert 时，expert 是【随机初始化】的")
    print("     这也是为什么 `utils/weight_mapper.py` 存在——两边权重名不一样。")

    print("\n" + "=" * 78)
    print("④ 最短阅读路径")
    print("=" * 78)
    for i, (rel, why) in enumerate(MINIMUM_PATH, 1):
        print(f"  {i}. {rel:<52} {why}")

    print("\n  第一遍可以暂时跳过：")
    print("     · rewards/comfort_reward.py：rl2 已经把判据复刻成 CPU toy；需要核对阈值时再读")
    print("     · models/reasoning_vla/weight_mapper.py / utils/weight_loading.py：权重交接细节")
    print("     · utils/fsdp.py：分布式切分细节，不改变 GRPO 数学")
    print("     · prefetch/server.py：性能工程（1321 行），不是 reward 或 policy loss")

    print("\n  如果要追踪 rl2 的 reasoning 分支，再从这里展开：")
    print("     · rewards/aggregated_reward_with_reasoning.py：在 ADE + comfort 外加入 CoT grader")
    print("     · models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py：")
    print("       注册 reasoning reward，并切到 reasoning 数据集")

    print("\n  ⚠️ 读之前记住两件事：")
    print("     · 这个仓库的真实 RL 跑起来要多卡大显存（local test 至少 5×80GB，")
    print("       README 展示的完整配置还有 64 policy + 128 rollout replica），")
    print("       所以本教程的 rl1/rl2 都刻意只做【算法层】，不碰基础设施。")
    print("     · 组内归一化那步在 `cosmos_rl` 包里，不在本仓库——")
    print("       trainer 只是取现成的 `rollout.advantage`。")
