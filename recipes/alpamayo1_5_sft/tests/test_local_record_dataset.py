# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from alpamayo1_5_sft.datasets.local_record import LocalRecordDataset
from PIL import Image


def _write_formal_version(version_dir: Path) -> None:
    version_dir.mkdir(parents=True)
    annotations_path = version_dir / "annotations.jsonl"
    annotation = {
        "item_id": "sample-1",
        "source_record": "group/record",
        "candidate_id": "candidate_001",
        "event_time_sec": "1.0",
        "final_content": {
            "ego_behavior_schema": {"effect_on_ego_behavior": "Continue forward."}
        },
        "candidate_review_decision": "accept",
        "coc_review_decision": "accept",
    }
    annotations_path.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
    digest = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
    (version_dir / "manifest.json").write_text(
        json.dumps({"annotation_count": 1, "annotations_sha256": digest}),
        encoding="utf-8",
    )


def _make_record(sample_root: Path) -> None:
    record_dir = sample_root / "group" / "record"
    for view, color in (("front_image", (20, 40, 60)), ("rear_image", (60, 40, 20))):
        camera_dir = record_dir / "camera" / view
        camera_dir.mkdir(parents=True)
        timestamp = Decimal("1000.0")
        for _ in range(41):
            Image.new("RGB", (12, 8), color).save(camera_dir / f"{timestamp}.jpg")
            timestamp += Decimal("0.1")

    localization_dir = record_dir / "localization"
    localization_dir.mkdir(parents=True)
    timestamp = Decimal("999.5")
    while timestamp <= Decimal("1007.5"):
        elapsed = float(timestamp - Decimal("1000.0"))
        payload = {
            "position": {"x": elapsed, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        (localization_dir / f"{timestamp}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        timestamp += Decimal("0.05")


def test_local_record_dataset_builds_two_view_coc_and_trajectory_sample(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample"
    _make_record(sample_root)
    version_dir = tmp_path / "formal" / "versions" / "v001"
    _write_formal_version(version_dir)

    dataset = LocalRecordDataset([version_dir], sample_root)
    item = dataset[0]

    assert item["image_frames"].shape == (2, 4, 3, 8, 12)
    assert item["camera_indices"].tolist() == [1, 4]
    assert item["relative_timestamps"].shape == (2, 4)
    assert item["relative_timestamps"].max().item() == 0.0
    assert item["ego_history_xyz"].shape == (1, 16, 3)
    assert item["ego_history_rot"].shape == (1, 16, 3, 3)
    assert item["ego_future_xyz"].shape == (1, 64, 3)
    assert item["ego_future_rot"].shape == (1, 64, 3, 3)
    np.testing.assert_allclose(item["ego_history_xyz"][0, -1].numpy(), [0, 0, 0])
    np.testing.assert_allclose(item["ego_future_xyz"][0, 0].numpy(), [0.1, 0, 0])
    assert item["cot"] == "Continue forward."
    assert item["label_components"] == ["cot", "traj_future"]


def test_local_record_dataset_rejects_unapproved_annotation(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample"
    _make_record(sample_root)
    version_dir = tmp_path / "formal" / "versions" / "v001"
    _write_formal_version(version_dir)
    annotation_path = version_dir / "annotations.jsonl"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["coc_review_decision"] = "pending"
    annotation_path.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
    digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    (version_dir / "manifest.json").write_text(
        json.dumps({"annotation_count": 1, "annotations_sha256": digest}),
        encoding="utf-8",
    )

    try:
        LocalRecordDataset([version_dir], sample_root)
    except ValueError as error:
        assert "not approved" in str(error)
    else:
        raise AssertionError("Unapproved formal annotation should be rejected")
