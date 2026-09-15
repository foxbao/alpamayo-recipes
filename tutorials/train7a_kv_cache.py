"""train7a: KV cache 是怎样保存前缀条件的。

本课使用 `tutorial_support/stage2_model.py` 的 toy 近似：VLM 对图片、CoC 和提示做一次前向，逐层返回 K/V；
action expert 只输入连续动作 token，并把 K/V 作为前缀条件。重点看 cache 的形状和复用。
"""

import torch

if __package__:
    from .tutorial_support.stage2_model import DATA, TinyExpert, TinyVLM, make_sample
else:
    from tutorial_support.stage2_model import DATA, TinyExpert, TinyVLM, make_sample


if __name__ == "__main__":
    sample = make_sample(*DATA[0])
    input_ids = sample[0][None, :]
    image = sample[1][None, :]
    vlm = TinyVLM()
    expert = TinyExpert()

    logits, caches = vlm(input_ids, image)
    print("VLM 输入 token 数:", input_ids.shape[1], "（模型内部还加 1 个图片 token）")
    print("VLM logits shape:", tuple(logits.shape))
    print("这里故意先使用完整 cache；下一课再处理答案泄漏和 crop。")
    print("\n逐层 cache:")
    for layer, (key, value) in enumerate(caches):
        print(f"  layer {layer}: K={tuple(key.shape)} V={tuple(value.shape)}")
    print("  形状含义: (batch, heads, prefix_length, head_dim)")

    action = torch.randn(1, 8, 2)
    time = torch.tensor([0.4])
    velocity = expert(action, time, caches)
    print("\nexpert 输入 action:", tuple(action.shape), "time:", tuple(time.shape))
    print("expert 输出 velocity:", tuple(velocity.shape))
    print("expert.forward 没有 condition 参数；它通过每层 CacheBlock 的 K/V 读取 VLM 条件。")

    velocity_again = expert(torch.randn_like(action), time, caches)
    print("同一 cache 的第二次采样输出:", tuple(velocity_again.shape))
