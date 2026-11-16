from __future__ import annotations

import importlib.util
from pathlib import Path


_REWARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "recipes"
    / "alpamayo1_x_rl"
    / "rewards"
    / "aggregated_reward_with_reasoning.py"
)
_SPEC = importlib.util.spec_from_file_location("_reasoning_reward", _REWARD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
normalize_reasoning_score = _MODULE.normalize_reasoning_score


def test_reasoning_score_normalization_preserves_quality_order() -> None:
    assert normalize_reasoning_score(-0.4) == 0.0
    assert normalize_reasoning_score(-0.2) == 0.5
    assert normalize_reasoning_score(0.0) == 1.0


def test_reasoning_score_normalization_rejects_non_negative_threshold() -> None:
    try:
        normalize_reasoning_score(-0.1, threshold=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-negative reasoning threshold must be rejected")
