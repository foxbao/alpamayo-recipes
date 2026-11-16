# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from alpamayo1_5_sft.datasets.local_record_audit import audit_annotation


def _write_pose(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            }
        ),
        encoding="utf-8",
    )


def _make_record(tmp_path: Path, *, localization_gap: bool) -> Path:
    record_dir = tmp_path / "group" / "record"
    front_dir = record_dir / "camera" / "front_image"
    rear_dir = record_dir / "camera" / "rear_image"
    localization_dir = record_dir / "localization"
    front_dir.mkdir(parents=True)
    rear_dir.mkdir(parents=True)
    localization_dir.mkdir(parents=True)
    (front_dir / "1000.000.jpg").touch()
    (front_dir / "1001.000.jpg").touch()
    (rear_dir / "1000.020.jpg").touch()
    (rear_dir / "1001.020.jpg").touch()

    timestamp = Decimal("999.3")
    final_time = Decimal("1007.5")
    while timestamp <= final_time:
        if not localization_gap or not Decimal("1003.9") < timestamp < Decimal("1004.3"):
            _write_pose(localization_dir / f"{timestamp}.json")
        timestamp += Decimal("0.05")
    return record_dir


def _annotation() -> dict[str, str]:
    return {
        "item_id": "item-1",
        "source_record": "group/record",
        "qualified_id": "group/record",
        "candidate_id": "candidate_001",
        "event_time_sec": "1.0",
    }


def test_audit_accepts_complete_camera_and_pose_coverage(tmp_path: Path) -> None:
    _make_record(tmp_path, localization_gap=False)
    row = audit_annotation(
        _annotation(),
        tmp_path,
        history_sec=Decimal("1.6"),
        future_sec=Decimal("6.4"),
        interval_sec=Decimal("0.1"),
        max_localization_gap_sec=Decimal("0.1"),
        record_cache={},
    )
    assert row.record_status == "pass"
    assert row.camera_status == "pass"
    assert row.trajectory_coverage_status == "pass"
    assert row.valid_grid_points == row.required_grid_points == 81


def test_audit_flags_large_localization_gap(tmp_path: Path) -> None:
    _make_record(tmp_path, localization_gap=True)
    row = audit_annotation(
        _annotation(),
        tmp_path,
        history_sec=Decimal("1.6"),
        future_sec=Decimal("6.4"),
        interval_sec=Decimal("0.1"),
        max_localization_gap_sec=Decimal("0.1"),
        record_cache={},
    )
    assert row.trajectory_coverage_status == "fail"
    assert row.valid_grid_points < row.required_grid_points
    assert "localization grid coverage" in row.issues
