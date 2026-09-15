"""train7c: cotrain_vlm 与 stop_grad_from_vlm 的梯度实验。

阶段二有两件容易混淆的事：`cotrain_vlm=False` 通常意味着 VLM 在 no_grad 下运行；
`stop_grad_from_vlm=True` 则是在已经建图时，用 detach 阻止 expert loss 沿 cache 回传。
本课把组合全部跑一遍，并且只计算 expert loss，避免 LM loss 掩盖差异。
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


def gradient_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.norm().item() for parameter in module.parameters() if parameter.grad is not None]
    return sum(values) / len(values) if values else 0.0


def run(cotrain_vlm: bool, stop_grad: bool) -> tuple[float, float]:
    samples = [make_sample(*item) for item in DATA]
    input_ids = torch.stack([item[0] for item in samples])
    images = torch.stack([item[1] for item in samples])
    model = Stage2Model(cotrain_vlm=cotrain_vlm, stop_grad_from_vlm=stop_grad)
    start = int((input_ids[0] == V["<traj_future_start>"]).nonzero()[0, 0])
    caches, _ = model.cache_for_expert(input_ids, images, crop_at=start + 2)
    action = torch.randn(len(samples), N_WAYPOINTS, ACTION_DIM)
    time = torch.rand(len(samples))
    prediction = model.expert(action, time, caches)
    loss = F.mse_loss(prediction, torch.zeros_like(prediction))
    model.zero_grad()
    loss.backward()
    return gradient_norm(model.vlm), gradient_norm(model.expert)


if __name__ == "__main__":
    print(f"{'cotrain_vlm':<14}{'stop_grad':<12}{'VLM grad':>14}{'expert grad':>16}")
    print("-" * 58)
    for cotrain in (False, True):
        for stop_grad in (False, True):
            vlm_grad, expert_grad = run(cotrain, stop_grad)
            print(f"{str(cotrain):<14}{str(stop_grad):<12}{vlm_grad:>14.3e}{expert_grad:>16.3e}")
    print("\n读法:")
    print("  False + 任意 stop_grad: no_grad 已经切断 VLM")
    print("  True + False: expert loss 可以回传到 VLM")
    print("  True + True: cache.detach() 切断回传，但 VLM 仍可由额外 LM loss 训练")
