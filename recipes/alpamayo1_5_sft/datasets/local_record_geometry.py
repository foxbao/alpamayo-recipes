# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose sampling and event-local frame conversion for local record data."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from decimal import Decimal

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _rotation_from_wxyz(quaternion_wxyz: Sequence[float]) -> Rotation:
    """Build a SciPy rotation from localization's named ``w,x,y,z`` fields."""
    if len(quaternion_wxyz) != 4:
        raise ValueError("A pose quaternion must have four components")
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    values = np.asarray([x, y, z, w], dtype=np.float64)
    if not np.isfinite(values).all() or np.linalg.norm(values) <= 1e-12:
        raise ValueError("Pose quaternion must be finite and non-zero")
    return Rotation.from_quat(values)


def sample_pose_at(
    timestamps: Sequence[Decimal],
    positions_xyz: np.ndarray,
    orientations_wxyz: np.ndarray,
    timestamp: Decimal,
    *,
    max_gap_sec: Decimal = Decimal("0.1"),
) -> tuple[np.ndarray, Rotation]:
    """Interpolate position and orientation at a timestamp without crossing large gaps."""
    if not timestamps:
        raise ValueError("No localization poses are available")
    if len(timestamps) != len(positions_xyz) or len(timestamps) != len(orientations_wxyz):
        raise ValueError("Pose timestamps, positions, and orientations must have equal lengths")
    if positions_xyz.shape != (len(timestamps), 3):
        raise ValueError("Position array must have shape (num_samples, 3)")
    if orientations_wxyz.shape != (len(timestamps), 4):
        raise ValueError("Quaternion array must have shape (num_samples, 4) in wxyz order")

    right = bisect.bisect_left(timestamps, timestamp)
    if right < len(timestamps) and timestamps[right] == timestamp:
        return positions_xyz[right].copy(), _rotation_from_wxyz(orientations_wxyz[right])
    if right == 0 or right == len(timestamps):
        raise ValueError(f"Timestamp {timestamp} is outside localization coverage")

    left = right - 1
    gap = timestamps[right] - timestamps[left]
    if gap <= 0 or gap > max_gap_sec:
        raise ValueError(f"Localization gap {gap} exceeds limit {max_gap_sec}")
    fraction = float((timestamp - timestamps[left]) / gap)
    position = positions_xyz[left] + fraction * (positions_xyz[right] - positions_xyz[left])
    endpoint_rotations = Rotation.from_quat(
        np.stack(
            [
                _rotation_from_wxyz(orientations_wxyz[left]).as_quat(),
                _rotation_from_wxyz(orientations_wxyz[right]).as_quat(),
            ]
        )
    )
    rotation = Slerp([0.0, 1.0], endpoint_rotations)([fraction])[0]
    return position, rotation


def poses_in_event_frame(
    positions_xyz: np.ndarray,
    orientations_wxyz: np.ndarray,
    event_position_xyz: Sequence[float],
    event_orientation_wxyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform world/map poses into the vehicle frame at the event.

    This matches Alpamayo-1.5's local-pose transform:
    ``p_local = R_event.inv().apply(p_world - p_event)`` and
    ``R_local = R_event.inv() * R_world``.

    The caller must supply positions and orientations in the same world/map frame,
    with localization quaternions ordered ``w,x,y,z``. For the current VLA-KL
    records, interpreting the body axes as FLU is the data-supported working
    assumption; this function does not infer or validate that physical convention.
    """
    positions = np.asarray(positions_xyz, dtype=np.float64)
    quaternions = np.asarray(orientations_wxyz, dtype=np.float64)
    event_position = np.asarray(event_position_xyz, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (num_samples, 3)")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("Quaternions must have shape (num_samples, 4) in wxyz order")
    if event_position.shape != (3,) or not np.isfinite(event_position).all():
        raise ValueError("Event position must contain three finite values")

    event_rotation = _rotation_from_wxyz(event_orientation_wxyz)
    world_rotations = Rotation.from_quat(
        np.stack([_rotation_from_wxyz(value).as_quat() for value in quaternions])
    )
    event_inverse = event_rotation.inv()
    local_positions = event_inverse.apply(positions - event_position)
    local_rotations = (event_inverse * world_rotations).as_matrix()
    return local_positions, local_rotations


def yaw_from_rotation_matrix(rotation_matrix: np.ndarray) -> float:
    """Return planar yaw in radians from a local-to-event rotation matrix."""
    return float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
