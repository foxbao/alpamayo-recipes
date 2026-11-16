# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit local CoC labels against camera and localization data before training.

This is a read-only feasibility check. It does not prepare model tensors or assert that
the localization coordinate convention is correct for Alpamayo.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRow:
    """Per-label availability and temporal-coverage result."""

    item_id: str
    qualified_id: str
    candidate_id: str
    event_time_sec: str
    coc_text: str
    record_exists: bool
    front_image_count: int
    rear_image_count: int
    front_event_error_sec: str | None
    rear_event_error_sec: str | None
    localization_count: int
    localization_max_gap_sec: str | None
    required_grid_points: int
    valid_grid_points: int
    missing_position_points: int
    missing_orientation_points: int
    oversized_bracket_points: int
    record_status: str
    camera_status: str
    trajectory_coverage_status: str
    issues: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_from_filename(path: Path) -> Decimal:
    try:
        timestamp = Decimal(path.stem)
    except InvalidOperation as exc:
        raise ValueError(f"Timestamp filename is not numeric: {path}") from exc
    if not timestamp.is_finite():
        raise ValueError(f"Timestamp filename is not finite: {path}")
    return timestamp


def _timestamped_images(camera_dir: Path) -> list[tuple[Decimal, Path]]:
    if not camera_dir.is_dir():
        return []
    images = []
    for path in camera_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            images.append((_timestamp_from_filename(path), path))
    return sorted(images)


def _camera_directory(record_dir: Path, normalized: str, legacy: str) -> Path:
    camera_root = record_dir / "camera"
    preferred = camera_root / normalized
    fallback = camera_root / legacy
    return preferred if preferred.is_dir() or not fallback.is_dir() else fallback


def _nearest_error(timestamps: Sequence[Decimal], target: Decimal) -> Decimal | None:
    if not timestamps:
        return None
    index = bisect.bisect_left(timestamps, target)
    options = [timestamps[i] for i in (index - 1, index) if 0 <= i < len(timestamps)]
    return min(abs(value - target) for value in options)


def _numeric_vector(payload: Any, key: str, fields: Sequence[str]) -> tuple[float, ...] | None:
    if not isinstance(payload, dict):
        return None
    vector = payload.get(key)
    if not isinstance(vector, dict):
        return None
    values = []
    for field in fields:
        value = vector.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def _load_localization(localization_dir: Path) -> list[tuple[Decimal, bool, bool]]:
    """Read filename sampling times and check whether each pose has xyz + quaternion."""
    if not localization_dir.is_dir():
        return []
    rows = []
    for path in localization_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        timestamp = _timestamp_from_filename(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        position_ok = _numeric_vector(payload, "position", ("x", "y", "z")) is not None
        orientation = _numeric_vector(payload, "orientation", ("w", "x", "y", "z"))
        orientation_ok = orientation is not None
        if orientation is not None:
            norm = math.sqrt(sum(component * component for component in orientation))
            orientation_ok = norm > 1e-8
        rows.append((timestamp, position_ok, orientation_ok))
    return sorted(rows)


def _grid_coverage(
    localization: Sequence[tuple[Decimal, bool, bool]],
    event_timestamp: Decimal,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    max_gap_sec: Decimal,
) -> dict[str, Any]:
    offsets = []
    count = int((history_sec + future_sec) / interval_sec)
    # Include both ends: [-1.6, +6.4] at 0.1 s gives 81 targets.
    offsets = [-history_sec + index * interval_sec for index in range(count + 1)]
    timestamps = [row[0] for row in localization]
    missing_position = missing_orientation = oversized = valid = 0
    max_bracket_gap = Decimal(0)

    for offset in offsets:
        target = event_timestamp + offset
        index = bisect.bisect_left(timestamps, target)
        if index < len(timestamps) and timestamps[index] == target:
            bracket = [localization[index]]
            gap = Decimal(0)
        elif index == 0 or index == len(timestamps):
            bracket = []
            gap = None
        else:
            bracket = [localization[index - 1], localization[index]]
            gap = bracket[1][0] - bracket[0][0]

        if gap is None:
            missing_position += 1
            missing_orientation += 1
            continue
        max_bracket_gap = max(max_bracket_gap, gap)
        position_ok = all(row[1] for row in bracket)
        orientation_ok = all(row[2] for row in bracket)
        if not position_ok:
            missing_position += 1
        if not orientation_ok:
            missing_orientation += 1
        if gap > max_gap_sec:
            oversized += 1
        if position_ok and orientation_ok and gap <= max_gap_sec:
            valid += 1

    gaps_in_span = [
        later - earlier
        for index in range(len(timestamps) - 1)
        for earlier, later in ((timestamps[index], timestamps[index + 1]),)
        if later >= event_timestamp - history_sec and earlier <= event_timestamp + future_sec
    ]
    return {
        "required_grid_points": len(offsets),
        "valid_grid_points": valid,
        "missing_position_points": missing_position,
        "missing_orientation_points": missing_orientation,
        "oversized_bracket_points": oversized,
        "localization_max_gap_sec": max(gaps_in_span) if gaps_in_span else None,
        "max_bracket_gap_sec": max_bracket_gap,
    }


def _annotation_rows(version_dir: Path) -> list[dict[str, Any]]:
    annotations_path = version_dir / "annotations.jsonl"
    manifest_path = version_dir / "manifest.json"
    if not annotations_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Expected annotations.jsonl and manifest.json under {version_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("annotations_sha256")
    actual_hash = _sha256(annotations_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(f"Annotation hash mismatch: {annotations_path}")
    rows = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if manifest.get("annotation_count") is not None and manifest["annotation_count"] != len(rows):
        raise ValueError(f"Annotation count does not match manifest: {annotations_path}")
    for row in rows:
        if row.get("candidate_review_decision") != "accept" or row.get("coc_review_decision") != "accept":
            raise ValueError(f"Formal annotation is not doubly accepted: {row.get('item_id')}")
    return rows


def audit_annotation(
    annotation: dict[str, Any],
    sample_root: Path,
    *,
    history_sec: Decimal,
    future_sec: Decimal,
    interval_sec: Decimal,
    max_localization_gap_sec: Decimal,
    record_cache: dict[Path, tuple[list[tuple[Decimal, Path]], list[tuple[Decimal, Path]], list[tuple[Decimal, bool, bool]]]],
) -> AuditRow:
    qualified_id = str(annotation.get("source_record") or annotation.get("qualified_id") or "")
    parts = Path(qualified_id).parts
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid source_record in formal annotation: {qualified_id!r}")
    record_dir = sample_root.joinpath(*parts)
    issues: list[str] = []
    record_exists = record_dir.is_dir()
    if not record_exists:
        issues.append("record directory missing")

    event_time = Decimal(str(annotation.get("event_time_sec", "")))
    if not event_time.is_finite() or event_time < 0:
        raise ValueError(f"Invalid event_time_sec for {annotation.get('item_id')}")

    cache_key = record_dir.resolve()
    if cache_key not in record_cache:
        front_dir = _camera_directory(record_dir, "front_image", "front")
        rear_dir = _camera_directory(record_dir, "rear_image", "rear")
        front = _timestamped_images(front_dir)
        rear = _timestamped_images(rear_dir)
        localization = _load_localization(record_dir / "localization") if record_exists else []
        record_cache[cache_key] = (front, rear, localization)
    front, rear, localization = record_cache[cache_key]
    front_times = [item[0] for item in front]
    rear_times = [item[0] for item in rear]
    front_event_error = rear_event_error = None
    if not front:
        issues.append("front images missing")
    else:
        # Candidate event times are relative to the first front-camera sample.
        event_timestamp = front_times[0] + event_time
        front_event_error = _nearest_error(front_times, event_timestamp)
        if front_event_error is not None and front_event_error > Decimal("0.5"):
            issues.append("front camera has no image within 0.5 s of event")
    if not rear:
        issues.append("rear images missing")
    elif front:
        rear_event_error = _nearest_error(rear_times, event_timestamp)
        if rear_event_error is not None and rear_event_error > Decimal("0.5"):
            issues.append("rear camera has no image within 0.5 s of event")

    if not localization:
        coverage = {
            "required_grid_points": 0,
            "valid_grid_points": 0,
            "missing_position_points": 0,
            "missing_orientation_points": 0,
            "oversized_bracket_points": 0,
            "localization_max_gap_sec": None,
        }
        issues.append("localization samples missing")
    elif not front:
        coverage = {
            "required_grid_points": 0,
            "valid_grid_points": 0,
            "missing_position_points": 0,
            "missing_orientation_points": 0,
            "oversized_bracket_points": 0,
            "localization_max_gap_sec": None,
        }
        issues.append("cannot align localization event without front-camera timestamps")
    else:
        coverage = _grid_coverage(
            localization,
            event_timestamp,
            history_sec,
            future_sec,
            interval_sec,
            max_localization_gap_sec,
        )
        if coverage["valid_grid_points"] != coverage["required_grid_points"]:
            issues.append(
                f"localization grid coverage {coverage['valid_grid_points']}/"
                f"{coverage['required_grid_points']}"
            )

    camera_ok = bool(front and rear) and not any(
        issue.startswith(("front camera has", "rear camera has", "front images", "rear images"))
        for issue in issues
    )
    trajectory_ok = (
        coverage["required_grid_points"] > 0
        and coverage["valid_grid_points"] == coverage["required_grid_points"]
    )
    return AuditRow(
        item_id=str(annotation.get("item_id", "")),
        qualified_id=qualified_id,
        candidate_id=str(annotation.get("candidate_id", "")),
        event_time_sec=str(event_time),
        coc_text=json.dumps(annotation.get("final_content", {}), ensure_ascii=False),
        record_exists=record_exists,
        front_image_count=len(front),
        rear_image_count=len(rear),
        front_event_error_sec=(str(front_event_error) if front_event_error is not None else None),
        rear_event_error_sec=(str(rear_event_error) if rear_event_error is not None else None),
        localization_count=len(localization),
        localization_max_gap_sec=(
            str(coverage["localization_max_gap_sec"])
            if coverage["localization_max_gap_sec"] is not None
            else None
        ),
        required_grid_points=coverage["required_grid_points"],
        valid_grid_points=coverage["valid_grid_points"],
        missing_position_points=coverage["missing_position_points"],
        missing_orientation_points=coverage["missing_orientation_points"],
        oversized_bracket_points=coverage["oversized_bracket_points"],
        record_status="pass" if record_exists else "fail",
        camera_status="pass" if camera_ok else "fail",
        trajectory_coverage_status="pass" if trajectory_ok else "fail",
        issues="; ".join(issues),
    )


def _write_outputs(rows: Sequence[AuditRow], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "local_record_training_audit.csv"
    row_dicts = [asdict(row) for row in rows]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(AuditRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(row_dicts)

    summary = {
        **metadata,
        "sample_count": len(rows),
        "record_pass_count": sum(row.record_status == "pass" for row in rows),
        "camera_pass_count": sum(row.camera_status == "pass" for row in rows),
        "trajectory_coverage_pass_count": sum(
            row.trajectory_coverage_status == "pass" for row in rows
        ),
        "samples": row_dicts,
        "note": (
            "Coverage only checks timestamped image availability and interpolatable pose fields. "
            "It does not validate coordinate-frame semantics or training-target correctness."
        ),
    }
    (output_dir / "local_record_training_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _decimal_arg(value: str) -> Decimal:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("value must be a decimal number") from exc
    if not number.is_finite() or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-version",
        action="append",
        type=Path,
        required=True,
        help="Formal labels version directory; repeat for each source version.",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        required=True,
        help="Dataset sample directory, containing <group>/<record>/ directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-sec", type=_decimal_arg, default=Decimal("1.6"))
    parser.add_argument("--future-sec", type=_decimal_arg, default=Decimal("6.4"))
    parser.add_argument("--interval-sec", type=_decimal_arg, default=Decimal("0.1"))
    parser.add_argument("--max-localization-gap-sec", type=_decimal_arg, default=Decimal("0.1"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (args.history_sec + args.future_sec) % args.interval_sec != 0:
        raise SystemExit("history + future duration must be divisible by interval")
    sample_root = args.sample_root.resolve()
    rows: list[AuditRow] = []
    versions = []
    seen_items: set[str] = set()
    record_cache: dict[
        Path,
        tuple[
            list[tuple[Decimal, Path]],
            list[tuple[Decimal, Path]],
            list[tuple[Decimal, bool, bool]],
        ],
    ] = {}
    for version_dir_arg in args.formal_version:
        version_dir = version_dir_arg.resolve()
        annotations = _annotation_rows(version_dir)
        manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
        versions.append(
            {
                "path": str(version_dir),
                "dataset_id": manifest.get("dataset_id"),
                "group_id": manifest.get("group_id"),
                "run_id": manifest.get("run_id"),
                "policy_id": manifest.get("policy_id"),
                "version": manifest.get("version"),
                "annotation_count": len(annotations),
                "annotations_sha256": _sha256(version_dir / "annotations.jsonl"),
            }
        )
        for annotation in annotations:
            item_id = str(annotation.get("item_id", ""))
            if not item_id or item_id in seen_items:
                raise ValueError(f"Missing or duplicate item_id across formal versions: {item_id!r}")
            seen_items.add(item_id)
            rows.append(
                audit_annotation(
                    annotation,
                    sample_root,
                    history_sec=args.history_sec,
                    future_sec=args.future_sec,
                    interval_sec=args.interval_sec,
                    max_localization_gap_sec=args.max_localization_gap_sec,
                    record_cache=record_cache,
                )
            )

    _write_outputs(
        rows,
        args.output_dir,
        {
            "schema_version": 1,
            "sample_root": str(sample_root),
            "window": {
                "history_sec": str(args.history_sec),
                "future_sec": str(args.future_sec),
                "interval_sec": str(args.interval_sec),
                "max_localization_gap_sec": str(args.max_localization_gap_sec),
            },
            "formal_versions": versions,
        },
    )
    print(f"Audited {len(rows)} labels")
    print(f"CSV: {args.output_dir / 'local_record_training_audit.csv'}")
    print(f"JSON: {args.output_dir / 'local_record_training_audit.json'}")


if __name__ == "__main__":
    main()
