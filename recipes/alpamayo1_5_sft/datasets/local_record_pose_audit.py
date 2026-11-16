# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare approved CoC reverse descriptions with event-local trajectory direction."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from alpamayo1_5_sft.datasets.local_record_geometry import (
    poses_in_event_frame,
    sample_pose_at,
    yaw_from_rotation_matrix,
)


def _read_poses(localization_dir: Path) -> tuple[list[Decimal], np.ndarray, np.ndarray]:
    rows = []
    for path in localization_dir.glob("*.json"):
        timestamp = Decimal(path.stem)
        payload = json.loads(path.read_text(encoding="utf-8"))
        position = payload["position"]
        orientation = payload["orientation"]
        rows.append(
            (
                timestamp,
                [position[axis] for axis in ("x", "y", "z")],
                [orientation[axis] for axis in ("w", "x", "y", "z")],
            )
        )
    rows.sort(key=lambda row: row[0])
    return (
        [row[0] for row in rows],
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def _read_formal_rows(version_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    annotation_path = version_dir / "annotations.jsonl"
    rows = [
        json.loads(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.get("annotation_count") != len(rows):
        raise ValueError(f"Annotation count mismatch in {version_dir}")
    return rows


def _read_image_timestamps(camera_dir: Path) -> list[Decimal]:
    if not camera_dir.is_dir():
        return []
    return sorted(
        Decimal(path.stem)
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def _audit_pai_image_grid(
    timestamps: list[Decimal], event_timestamp: Decimal
) -> tuple[Decimal | None, int]:
    """Mirror PAI-AV decoding: select the closest frame at or before each target time."""
    offsets = (Decimal("-0.3"), Decimal("-0.2"), Decimal("-0.1"), Decimal(0))
    if not timestamps:
        return None, 0
    selected = []
    errors = []
    for offset in offsets:
        target = event_timestamp + offset
        index = bisect.bisect_right(timestamps, target) - 1
        if index < 0:
            return None, len(set(selected))
        selected.append(timestamps[index])
        errors.append(target - timestamps[index])
    max_error = max(errors)
    return max_error, len(set(selected))


def _audit_recent_image_history(
    timestamps: list[Decimal], event_timestamp: Decimal
) -> tuple[Decimal | None, Decimal | None, Decimal | None, int]:
    """Measure the four latest distinct frames at or before the event time."""
    prior = [timestamp for timestamp in timestamps if timestamp <= event_timestamp]
    selected = prior[-4:]
    if len(selected) < 4:
        return None, None, None, len(selected)
    intervals = [later - earlier for earlier, later in pairwise(selected)]
    return (
        event_timestamp - selected[0],
        event_timestamp - selected[-1],
        max(intervals),
        len(selected),
    )


def audit_pose(
    annotation: dict[str, Any],
    sample_root: Path,
    cache: dict[Path, tuple[list[Decimal], np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    qualified_id = str(annotation["source_record"])
    parts = Path(qualified_id).parts
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid source record: {qualified_id}")
    record_dir = sample_root.joinpath(*parts)
    if record_dir not in cache:
        cache[record_dir] = _read_poses(record_dir / "localization")
    timestamps, positions, orientations = cache[record_dir]

    camera_root = record_dir / "camera"
    front_dir = camera_root / "front_image"
    if not front_dir.is_dir():
        front_dir = camera_root / "front"
    rear_dir = camera_root / "rear_image"
    if not rear_dir.is_dir():
        rear_dir = camera_root / "rear"
    front_timestamps = _read_image_timestamps(front_dir)
    rear_timestamps = _read_image_timestamps(rear_dir)
    if not front_timestamps:
        raise ValueError(f"No front image timestamps in {front_dir}")
    event_time = Decimal(str(annotation["event_time_sec"]))
    event_timestamp = front_timestamps[0] + event_time
    front_grid_error, front_grid_unique = _audit_pai_image_grid(
        front_timestamps, event_timestamp
    )
    rear_grid_error, rear_grid_unique = _audit_pai_image_grid(
        rear_timestamps, event_timestamp
    )
    front_recent_oldest_age, front_recent_latest_age, front_recent_max_interval, front_recent_count = (
        _audit_recent_image_history(front_timestamps, event_timestamp)
    )
    rear_recent_oldest_age, rear_recent_latest_age, rear_recent_max_interval, rear_recent_count = (
        _audit_recent_image_history(rear_timestamps, event_timestamp)
    )
    event_position, event_rotation = sample_pose_at(
        timestamps, positions, orientations, event_timestamp
    )

    offsets = [Decimal("0.5"), Decimal("1.0"), Decimal("3.0"), Decimal("6.4")]
    sampled = [
        sample_pose_at(timestamps, positions, orientations, event_timestamp + offset)
        for offset in offsets
    ]
    future_positions = np.asarray([position for position, _ in sampled])
    future_quaternions = np.asarray(
        [rotation.as_quat()[[3, 0, 1, 2]] for _, rotation in sampled]
    )
    local_positions, local_rotations = poses_in_event_frame(
        future_positions,
        future_quaternions,
        event_position,
        event_rotation.as_quat()[[3, 0, 1, 2]],
    )

    text = annotation["final_content"]["ego_behavior_schema"]["effect_on_ego_behavior"]
    return {
        "item_id": annotation["item_id"],
        "qualified_id": qualified_id,
        "candidate_id": annotation["candidate_id"],
        "event_time_sec": str(event_time),
        "coc_text": text,
        "coc_mentions_reverse": "revers" in text.lower() or "backward" in text.lower(),
        "front_pai_grid_max_lateness_sec": (
            str(front_grid_error) if front_grid_error is not None else None
        ),
        "front_pai_grid_unique_image_count": front_grid_unique,
        "rear_pai_grid_max_lateness_sec": (
            str(rear_grid_error) if rear_grid_error is not None else None
        ),
        "rear_pai_grid_unique_image_count": rear_grid_unique,
        "front_recent4_oldest_age_sec": (
            str(front_recent_oldest_age) if front_recent_oldest_age is not None else None
        ),
        "front_recent4_latest_age_sec": (
            str(front_recent_latest_age) if front_recent_latest_age is not None else None
        ),
        "front_recent4_max_interval_sec": (
            str(front_recent_max_interval) if front_recent_max_interval is not None else None
        ),
        "front_recent4_count": front_recent_count,
        "rear_recent4_oldest_age_sec": (
            str(rear_recent_oldest_age) if rear_recent_oldest_age is not None else None
        ),
        "rear_recent4_latest_age_sec": (
            str(rear_recent_latest_age) if rear_recent_latest_age is not None else None
        ),
        "rear_recent4_max_interval_sec": (
            str(rear_recent_max_interval) if rear_recent_max_interval is not None else None
        ),
        "rear_recent4_count": rear_recent_count,
        **{
            f"x_{offset}s_m": float(local_positions[index, 0])
            for index, offset in enumerate(offsets)
        },
        **{
            f"y_{offset}s_m": float(local_positions[index, 1])
            for index, offset in enumerate(offsets)
        },
        **{
            f"yaw_{offset}s_deg": float(np.degrees(yaw_from_rotation_matrix(local_rotations[index])))
            for index, offset in enumerate(offsets)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-version", action="append", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cache: dict[Path, tuple[list[Decimal], np.ndarray, np.ndarray]] = {}
    audited = []
    for version_dir in args.formal_version:
        audited.extend(
            audit_pose(row, args.sample_root, cache) for row in _read_formal_rows(version_dir)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "local_record_pose_conversion_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)

    reverse_rows = [row for row in audited if row["coc_mentions_reverse"]]
    moving_reverse_rows = [row for row in reverse_rows if abs(row["x_6.4s_m"]) >= 1.0]
    complete_pai_camera_windows = [
        row
        for row in audited
        if row["front_pai_grid_unique_image_count"] == 4
        and row["rear_pai_grid_unique_image_count"] == 4
    ]
    complete_recent_camera_windows = [
        row
        for row in audited
        if row["front_recent4_count"] == 4 and row["rear_recent4_count"] == 4
    ]
    summary = {
        "sample_count": len(audited),
        "record_count": len(cache),
        "samples_with_four_distinct_front_and_rear_frames_on_pai_fixed_grid": len(
            complete_pai_camera_windows
        ),
        "samples_with_four_distinct_front_and_rear_recent_frames": len(
            complete_recent_camera_windows
        ),
        "max_front_pai_grid_lateness_sec": max(
            float(row["front_pai_grid_max_lateness_sec"]) for row in audited
        ),
        "max_rear_pai_grid_lateness_sec": max(
            float(row["rear_pai_grid_max_lateness_sec"]) for row in audited
        ),
        "max_front_recent4_oldest_age_sec": max(
            float(row["front_recent4_oldest_age_sec"]) for row in audited
        ),
        "max_rear_recent4_oldest_age_sec": max(
            float(row["rear_recent4_oldest_age_sec"]) for row in audited
        ),
        "reverse_descriptions": len(reverse_rows),
        "reverse_descriptions_with_at_least_1m_motion": len(moving_reverse_rows),
        "reverse_descriptions_with_negative_local_x_at_6_4s": sum(
            row["x_6.4s_m"] < 0 for row in moving_reverse_rows
        ),
        "max_abs_local_y_at_6_4s_m": max(abs(row["y_6.4s_m"]) for row in audited),
        "coordinate_assumption": (
            "Localization orientation is interpreted as body-FLU to map; poses are transformed "
            "into the event-time body frame using the same inverse-rotation operation as the "
            "Alpamayo-1.5 PAI loader. This assumption is supported by prior VLA-KL checks, "
            "but is not declared in the localization JSON."
        ),
        "csv": str(csv_path),
    }
    json_path = args.output_dir / "local_record_pose_conversion_audit.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(audited)} approved CoC samples")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
