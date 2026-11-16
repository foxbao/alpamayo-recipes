# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from alpamayo1_5_sft.datasets.local_record_geometry import (
    poses_in_event_frame,
    sample_pose_at,
    yaw_from_rotation_matrix,
)
from alpamayo1_5_sft.datasets.local_record_pose_audit import (
    _audit_pai_image_grid,
    _audit_recent_image_history,
)


def test_sample_pose_at_interpolates_translation_and_rotation() -> None:
    timestamps = [Decimal(0), Decimal("0.1")]
    positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Identity, then 90 degrees about map z, in wxyz order.
    orientations = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]]
    )

    position, rotation = sample_pose_at(
        timestamps, positions, orientations, Decimal("0.05")
    )

    np.testing.assert_allclose(position, [1.0, 0.0, 0.0])
    assert yaw_from_rotation_matrix(rotation.as_matrix()) == pytest.approx(np.pi / 4)


def test_sample_pose_at_rejects_large_interpolation_gap() -> None:
    with pytest.raises(ValueError, match="exceeds limit"):
        sample_pose_at(
            [Decimal(0), Decimal("0.2")],
            np.zeros((2, 3)),
            np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
            Decimal("0.1"),
            max_gap_sec=Decimal("0.1"),
        )


def test_poses_in_event_frame_preserves_forward_and_reverse_signs() -> None:
    # Event heading is +90 degrees in map. Its local +x points along map +y.
    half = np.sqrt(0.5)
    event_orientation = [half, 0.0, 0.0, half]
    positions = np.asarray([[0.0, 1.0, 0.0], [0.0, -2.0, 0.0]])
    orientations = np.asarray([event_orientation, event_orientation])

    local_positions, local_rotations = poses_in_event_frame(
        positions,
        orientations,
        [0.0, 0.0, 0.0],
        event_orientation,
    )

    np.testing.assert_allclose(local_positions[:, 0], [1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(local_positions[:, 1], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(local_rotations[0], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(local_rotations[1], np.eye(3), atol=1e-12)


def test_pai_image_grid_audit_matches_previous_frame_sampling() -> None:
    timestamps = [Decimal("99.7") + Decimal("0.1") * index for index in range(4)]

    max_lateness, unique_count = _audit_pai_image_grid(timestamps, Decimal("100.0"))

    assert max_lateness == Decimal(0)
    assert unique_count == 4


def test_recent_image_history_uses_four_distinct_frames_before_event() -> None:
    timestamps = [Decimal("99.55"), Decimal("99.66"), Decimal("99.78"), Decimal("99.91")]

    oldest_age, latest_age, max_interval, count = _audit_recent_image_history(
        timestamps, Decimal("100.0")
    )

    assert oldest_age == Decimal("0.45")
    assert latest_age == Decimal("0.09")
    assert max_interval == Decimal("0.13")
    assert count == 4
