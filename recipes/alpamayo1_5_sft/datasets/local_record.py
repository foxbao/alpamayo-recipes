# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter for approved CoC annotations on local recording data."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from alpamayo1_5_sft.datasets.local_record_geometry import (
    poses_in_event_frame,
    sample_pose_at,
)
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset


class LocalRecordDataset(Dataset):
    """Load formal local CoC labels as Alpamayo-1.5 SFT samples.

    Each item contains four recent distinct history frames per camera, event-local
    ego history/future pose tensors, and the accepted CoC sentence. Front/rear use
    the pretrained Alpamayo camera IDs 1 and 4 respectively.
    """

    HISTORY_STEPS: ClassVar[int] = 16
    FUTURE_STEPS: ClassVar[int] = 64
    TIME_STEP_SEC: ClassVar[Decimal] = Decimal("0.1")
    CAMERA_IDS: ClassVar[dict[str, int]] = {"front": 1, "rear": 4}

    def __init__(
        self,
        formal_versions: list[str | Path],
        sample_root: str | Path,
        *,
        local_dir: str | Path | None = None,
        chunk_ids: Any | None = None,
        use_default_keyframe: bool | None = None,
        model_config: Any | None = None,
        vla_preprocess_args: dict[str, Any] | None = None,
        max_localization_gap_sec: Decimal = Decimal("0.1"),
    ) -> None:
        # ``sft_base`` historically supplies these PAI-only options to datasets.
        # Accept them for Hydra config compatibility; local records use
        # ``sample_root`` and their approved event times instead.
        del local_dir, chunk_ids, use_default_keyframe
        self.sample_root = Path(sample_root)
        self.max_localization_gap_sec = max_localization_gap_sec
        self.annotations = [
            annotation
            for version in formal_versions
            for annotation in self._read_version(Path(version))
        ]
        self._pose_cache: dict[
            Path, tuple[list[Decimal], np.ndarray, np.ndarray]
        ] = {}
        self._image_path_cache: dict[Path, list[tuple[Decimal, Path]]] = {}
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.preprocess = (
            instantiate(vla_preprocess_args, model_config=model_config)
            if vla_preprocess_args is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.annotations)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _read_version(cls, version_dir: Path) -> list[dict[str, Any]]:
        manifest_path = version_dir / "manifest.json"
        annotations_path = version_dir / "annotations.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_hash = manifest.get("annotations_sha256")
        if expected_hash and cls._sha256(annotations_path) != expected_hash:
            raise ValueError(f"Formal annotation hash mismatch: {annotations_path}")
        rows = [
            json.loads(line)
            for line in annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if manifest.get("annotation_count") != len(rows):
            raise ValueError(f"Formal annotation count mismatch: {annotations_path}")
        for row in rows:
            if (
                row.get("candidate_review_decision") != "accept"
                or row.get("coc_review_decision") != "accept"
            ):
                raise ValueError(f"Formal annotation is not approved: {row.get('item_id')}")
        return rows

    @staticmethod
    def _record_parts(annotation: dict[str, Any]) -> tuple[str, str]:
        source_record = str(annotation["source_record"])
        parts = Path(source_record).parts
        if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Invalid source_record: {source_record}")
        return parts[0], parts[1]

    @staticmethod
    def _image_paths(camera_dir: Path) -> list[tuple[Decimal, Path]]:
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"Camera directory does not exist: {camera_dir}")
        paths = [
            (Decimal(path.stem), path)
            for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ]
        return sorted(paths)

    def _timed_images(self, camera_dir: Path) -> list[tuple[Decimal, Path]]:
        if camera_dir not in self._image_path_cache:
            self._image_path_cache[camera_dir] = self._image_paths(camera_dir)
        return self._image_path_cache[camera_dir]

    @staticmethod
    def _camera_dir(record_dir: Path, view: str) -> Path:
        camera_root = record_dir / "camera"
        normalized = camera_root / f"{view}_image"
        legacy = camera_root / view
        return normalized if normalized.is_dir() else legacy

    @staticmethod
    def _recent_four_images(
        timed_images: list[tuple[Decimal, Path]], event_timestamp: Decimal, view: str
    ) -> list[tuple[Decimal, Path]]:
        prior = [item for item in timed_images if item[0] <= event_timestamp]
        selected = prior[-4:]
        if len(selected) != 4:
            raise ValueError(f"Only {len(selected)} {view} frames precede event time")
        return selected

    def _load_record_poses(
        self, record_dir: Path
    ) -> tuple[list[Decimal], np.ndarray, np.ndarray]:
        if record_dir not in self._pose_cache:
            rows = []
            for path in (record_dir / "localization").glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    (
                        Decimal(path.stem),
                        [payload["position"][axis] for axis in ("x", "y", "z")],
                        [
                            payload["orientation"][axis]
                            for axis in ("w", "x", "y", "z")
                        ],
                    )
                )
            rows.sort(key=lambda row: row[0])
            self._pose_cache[record_dir] = (
                [row[0] for row in rows],
                np.asarray([row[1] for row in rows], dtype=np.float64),
                np.asarray([row[2] for row in rows], dtype=np.float64),
            )
        return self._pose_cache[record_dir]

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def _trajectory(
        self,
        record_dir: Path,
        event_timestamp: Decimal,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        timestamps, positions, orientations = self._load_record_poses(record_dir)
        history_offsets = [
            -self.TIME_STEP_SEC * (self.HISTORY_STEPS - 1 - index)
            for index in range(self.HISTORY_STEPS)
        ]
        future_offsets = [
            self.TIME_STEP_SEC * (index + 1) for index in range(self.FUTURE_STEPS)
        ]
        all_offsets = history_offsets + future_offsets
        sampled = [
            sample_pose_at(
                timestamps,
                positions,
                orientations,
                event_timestamp + offset,
                max_gap_sec=self.max_localization_gap_sec,
            )
            for offset in all_offsets
        ]
        event_position, event_orientation = sampled[self.HISTORY_STEPS - 1]
        sampled_positions = np.asarray([position for position, _ in sampled])
        sampled_quaternions = np.asarray(
            [rotation.as_quat()[[3, 0, 1, 2]] for _, rotation in sampled]
        )
        local_positions, local_rotations = poses_in_event_frame(
            sampled_positions,
            sampled_quaternions,
            event_position,
            event_orientation.as_quat()[[3, 0, 1, 2]],
        )
        history_slice = slice(0, self.HISTORY_STEPS)
        future_slice = slice(self.HISTORY_STEPS, None)
        return (
            torch.from_numpy(local_positions[history_slice]).float().unsqueeze(0),
            torch.from_numpy(local_rotations[history_slice]).float().unsqueeze(0),
            torch.from_numpy(local_positions[future_slice]).float().unsqueeze(0),
            torch.from_numpy(local_rotations[future_slice]).float().unsqueeze(0),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        annotation = self.annotations[index]
        group, record = self._record_parts(annotation)
        record_dir = self.sample_root / group / record
        front_images = self._timed_images(self._camera_dir(record_dir, "front"))
        rear_images = self._timed_images(self._camera_dir(record_dir, "rear"))
        event_time = Decimal(str(annotation["event_time_sec"]))
        event_timestamp = front_images[0][0] + event_time
        front_selected = self._recent_four_images(front_images, event_timestamp, "front")
        rear_selected = self._recent_four_images(rear_images, event_timestamp, "rear")

        camera_images = [
            torch.stack([self._load_image(path) for _, path in selected])
            for selected in (front_selected, rear_selected)
        ]
        image_frames = torch.stack(camera_images)
        image_timestamps = torch.tensor(
            [[float(timestamp - event_timestamp) for timestamp, _ in selected]
             for selected in (front_selected, rear_selected)],
            dtype=torch.float32,
        )
        camera_indices = torch.tensor(
            [self.CAMERA_IDS["front"], self.CAMERA_IDS["rear"]], dtype=torch.int64
        )
        history_xyz, history_rot, future_xyz, future_rot = self._trajectory(
            record_dir, event_timestamp
        )
        item = {
            "item_id": annotation["item_id"],
            "source_record": annotation["source_record"],
            "candidate_id": annotation["candidate_id"],
            "event_timestamp_sec": float(event_timestamp),
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "relative_timestamps": image_timestamps,
            "absolute_timestamps": torch.tensor(
                [[int(timestamp * 1_000_000) for timestamp, _ in selected]
                 for selected in (front_selected, rear_selected)],
                dtype=torch.int64,
            ),
            "ego_history_xyz": history_xyz,
            "ego_history_rot": history_rot,
            "ego_future_xyz": future_xyz,
            "ego_future_rot": future_rot,
            "cot": annotation["final_content"]["ego_behavior_schema"][
                "effect_on_ego_behavior"
            ],
            "label_components": ["cot", "traj_future"],
            "generation_mode": False,
        }
        if self.preprocess is not None:
            item["tokenized_data"] = self.preprocess(data=item)
        return item
