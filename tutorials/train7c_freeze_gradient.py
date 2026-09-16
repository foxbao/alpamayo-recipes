"""train7c: cotrain_vlm 与 stop_grad_from_vlm 的梯度实验。

阶段二有两件容易混淆的事：`cotrain_vlm=False` 通常意味着 VLM 在 no_grad 下运行；
`stop_grad_from_vlm=True` 则是在已经建图时，用 detach 阻止 expert loss 沿 cache 回传。
本课把四种组合全部跑一遍。

⚠️ 本课【只算 expert 的 flow-matching loss】，不额外加 LM loss ——
   否则 VLM 的梯度会被 LM loss 主导，看不出 cache 这条路通不通（见下）。
   真实代码里 `cotrain_vlm=True` 时还会 `loss += vlm_outputs.loss`，
   那条 LM 路径独立于 cache，本课的表格刻意把它排除在外。
"""

import torch
import torch.nn.functional as F

if __package__:
    from .tutorial_support.stage2_model import (
        ACTION_DIM, DATA, N_WAYPOINTS, Stage2Model, V, make_sample
    )
else:
    from tutorial_support.stage2_model import (
        ACTION_DIM, DATA, N_WAYPOINTS, Stage2Model, V, make_sample
    )

# ★ 固定种子：四种组合必须看【同一批】输入和【同一个初始权重】。
#   否则行与行之间的差异里混着采样噪声和初始化噪声，比不出开关的作用。
#   这两个种子固定后，整张表可复现，而且四行的 expert grad 应该完全相同 ——
#   唯一变量就只剩那两个开关。
SEED = 0            # 采 x0 / t 用的种子
SEED_MODEL = 1      # 初始化模型用的种子


def gradient_norm(module: torch.nn.Module) -> float:
    """整体梯度范数 sqrt(Σ‖g_i‖²)。

    ⚠️ 别用「逐参数范数的平均」——那会被参数个数影响，不同模块之间不可比。
    """
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += parameter.grad.norm().item() ** 2
    return total ** 0.5


def make_batch():
    """四种组合共用同一批输入（含同一个噪声起点和时间步）。"""
    samples = [make_sample(*item) for item in DATA]
    input_ids = torch.stack([item[0] for item in samples])
    images = torch.stack([item[1] for item in samples])
    target_actions = torch.stack([item[2] for item in samples])   # (B, 8, 2)

    torch.manual_seed(SEED)
    x0 = torch.randn_like(target_actions)
    t = torch.rand(target_actions.shape[0])
    return input_ids, images, x0, t, target_actions


def run(cotrain_vlm: bool, stop_grad: bool, batch) -> tuple[float, float]:
    input_ids, images, x0, t, target_actions = batch
    # ★ 每次都用同一个种子初始化 —— 四种组合拿到的是【同一份权重】，
    #   这样四行 expert grad 相同，VLM grad 的差异才纯粹来自开关。
    torch.manual_seed(SEED_MODEL)
    model = Stage2Model(cotrain_vlm=cotrain_vlm, stop_grad_from_vlm=stop_grad)
    start = int((input_ids[0] == V["<traj_future_start>"]).nonzero()[0, 0])
    caches, _ = model.cache_for_expert(input_ids, images, crop_at=start + 2)

    # ★ 和 train7/train7d 同款：真实的 flow matching 目标
    #   x_t = (1-t)·x0 + t·x1，v_target = x1 - x0
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * target_actions
    velocity = model.expert(x_t, t, caches)
    loss = F.mse_loss(velocity, target_actions - x0)

    model.zero_grad()
    loss.backward()
    return gradient_norm(model.vlm), gradient_norm(model.expert)


if __name__ == "__main__":
    batch = make_batch()

    print(f"{'cotrain_vlm':<14}{'stop_grad':<12}{'VLM grad':>14}{'expert grad':>16}")
    print("-" * 58)
    for cotrain in (False, True):
        for stop_grad in (False, True):
            vlm_grad, expert_grad = run(cotrain, stop_grad, batch)
            print(f"{str(cotrain):<14}{str(stop_grad):<12}{vlm_grad:>14.3e}{expert_grad:>16.3e}")

    print("\n读法（这张表只含 expert loss，不含 LM loss）:")
    print("  cotrain=False（任意 stop_grad）→ VLM grad 恒为 0")
    print("     因为 VLM 在 `torch.no_grad()` 下跑，cache 根本没有计算图。")
    print("     此时 stop_grad_from_vlm 取哪个值都一样 —— 它【无事可做】。")
    print()
    print("  cotrain=True + stop_grad=False → VLM grad 非零")
    print("     VLM 建了图，expert 的 flow-matching loss 顺着 cache 回传到它。")
    print()
    print("  cotrain=True + stop_grad=True  → VLM grad 回到 0")
    print("     `cache.detach()` 把这条路切断。")
    print()
    print("  ★ expert grad 四行【完全相同】——因为四种组合用的是同一份权重、")
    print("    同一批输入（固定了两个种子）。所以 VLM grad 的差异")
    print("    【只来自那两个开关】，不是采样或初始化噪声。")
    print()
    print("  ⚠️ 真实代码里 cotrain_vlm=True 时还有一条独立的路：")
    print("     `loss += vlm_outputs.loss`（LM 交叉熵）——它不经过 cache，")
    print("     所以即使 stop_grad=True，VLM 仍然会被 LM loss 训练。")
    print("     本表为了隔离出 cache 那条路，刻意没有加它。")
