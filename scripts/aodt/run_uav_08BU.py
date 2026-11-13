#!/usr/bin/env python3
"""Validate a complete 07BU AODT GPX/UAV raw CIR collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FILE_PATTERN = re.compile(r"^cir_timestep_(\d{4})_ru_(\d{2})\.npz$")
PAYLOAD_FIELDS = (
    "values",
    "delays",
    "angles_of_departure",
    "angles_of_arrival",
)
IDENTITY_FIELDS = (
    "batch",
    "timestep",
    "ru_tensor_index",
    "ru_scenario_id",
    "ru_aodt_id",
    "ue_tensor_indices",
    "ue_scenario_ids",
    "ue_aodt_ids",
    "full_antenna_pair",
)
EXPECTED_DTYPES = {
    "values": "complex64",
    "delays": "float32",
    "angles_of_departure": "float32",
    "angles_of_arrival": "float32",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(array: np.ndarray) -> Any:
    if array.shape != ():
        raise ValueError(f"expected scalar, got shape {array.shape}")
    return array.item()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p05": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def latest_07bu_run(repo_root: Path) -> Path:
    results_root = repo_root / "results" / "aodt_uav"
    candidates: list[tuple[float, Path]] = []
    if results_root.is_dir():
        for run_dir in results_root.iterdir():
            manifest_path = run_dir / "run_manifest.json"
            if not run_dir.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = json_load(manifest_path)
            except (OSError, ValueError):
                continue
            if manifest.get("stages", {}).get("07BU", {}).get("status") == "pass":
                candidates.append((manifest_path.stat().st_mtime, run_dir))
    if not candidates:
        raise FileNotFoundError("No completed AODT 07BU run was found.")
    return max(candidates, key=lambda item: item[0])[1]


def validate(run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage_07 = run_dir / "07BU_raw_collection"
    raw_dir = stage_07 / "raw"
    required = {
        "manifest": run_dir / "run_manifest.json",
        "hashes": stage_07 / "raw_file_hashes.json",
        "schema": stage_07 / "raw_schema_inventory.json",
        "inventory": stage_07 / "raw_artifact_inventory.json",
        "positions": run_dir / "06BU_execution" / "position_registry.json",
        "topology": run_dir / "03BU_topology" / "aodt_node_mapping.json",
        "gpx_binding": run_dir / "04BU_config" / "gpx_binding_manifest.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required BU input not found: {path}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw BU NPZ directory not found: {raw_dir}")
    manifest = json_load(required["manifest"])
    hashes_document = json_load(required["hashes"])
    declared_schema = json_load(required["schema"])
    raw_inventory = json_load(required["inventory"])
    positions = json_load(required["positions"])
    topology = json_load(required["topology"])
    gpx_binding = json_load(required["gpx_binding"])
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def record(
        target: list[dict[str, Any]],
        code: str,
        message: str,
        relative_path: str | None = None,
    ) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if relative_path:
            item["relative_path"] = relative_path
        if len(target) < 100:
            target.append(item)

    scope = manifest.get("scope", {})
    expected_scope = {
        "ru": 8,
        "ue": 54,
        "ground_ue": 50,
        "uav": 4,
        "gpx": 4,
        "time_steps": 256,
    }
    for key, expected in expected_scope.items():
        if int(scope.get(key, -1)) != expected:
            record(errors, "SCOPE_MISMATCH", f"scope.{key} must be {expected}.")
    if manifest.get("stages", {}).get("07BU", {}).get("status") != "pass":
        record(errors, "07BU_NOT_PASS", "The source run does not have passing 07BU.")
    if positions.get("validation", {}).get("status") != "pass":
        record(errors, "POSITION_VALIDATION", "06BU position validation is not pass.")
    if not positions.get("validation", {}).get(
        "trajectory_runtime_verified"
    ):
        record(
            errors,
            "RUNTIME_GPX_MISMATCH",
            "06BU did not confirm runtime positions against canonical AODT ENU.",
        )
    if gpx_binding.get("trajectory_hashes_verified") is not True:
        record(errors, "GPX_BINDING_MISMATCH", "04BU GPX binding is not AODT-shared.")
    validation = topology.get("validation", {})
    if (
        validation.get("ru_count") != 8
        or validation.get("ground_ue_count") != 50
        or validation.get("uav_count") != 4
        or validation.get("total_ue_count") != 54
        or validation.get("uav_aodt_ids") != [51, 52, 53, 54]
    ):
        record(errors, "TOPOLOGY_MISMATCH", "03BU topology mapping is invalid.")

    declared_hashes = hashes_document.get("files", {})
    if hashes_document.get("algorithm") != "sha256" or not isinstance(
        declared_hashes, dict
    ):
        record(errors, "INVALID_HASH_MANIFEST", "07BU hash manifest is invalid.")
        declared_hashes = {}
    actual_files = sorted(raw_dir.glob("*.npz"))
    actual_relpaths = {path.relative_to(stage_07).as_posix() for path in actual_files}
    expected_relpaths = set(declared_hashes)
    if actual_relpaths != expected_relpaths:
        record(
            errors,
            "RAW_FILE_SET_MISMATCH",
            f"Raw file set differs: actual={len(actual_relpaths)} expected={len(expected_relpaths)}.",
        )
    selected = [
        int(value)
        for value in raw_inventory.get("selection", {}).get("time_steps", [])
    ]
    expected_rus = [
        int(value)
        for value in raw_inventory.get("selection", {}).get(
            "ru_tensor_indices", []
        )
    ]
    if selected != list(range(256)):
        record(errors, "INCOMPLETE_TIME_STEPS", "08BU requires time steps 0..255.")
    if expected_rus != list(range(8)):
        record(errors, "INCOMPLETE_RUS", "08BU requires RU tensor indices 0..7.")
    expected_pairs = {(time_step, ru) for time_step in selected for ru in expected_rus}
    observed_pairs: set[tuple[int, int]] = set()
    shapes: dict[str, set[tuple[int, ...]]] = {
        field: set() for field in PAYLOAD_FIELDS
    }
    dtypes: dict[str, set[str]] = {field: set() for field in PAYLOAD_FIELDS}
    nan_counts = {field: 0 for field in PAYLOAD_FIELDS}
    inf_counts = {field: 0 for field in PAYLOAD_FIELDS}
    active_coefficient_count = 0
    logical_link_rows = 0
    uav_logical_link_rows = 0
    link_total_powers: list[float] = []
    active_delays: list[float] = []
    uav_power_signatures: dict[tuple[int, int], set[float]] = defaultdict(set)
    identity_expected = {
        "ue_tensor_indices": np.arange(54, dtype=np.int64),
        "ue_scenario_ids": np.arange(54, dtype=np.int64),
        "ue_aodt_ids": np.arange(1, 55, dtype=np.int64),
    }

    for file_path in actual_files:
        relative = file_path.relative_to(stage_07).as_posix()
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            record(errors, "INVALID_FILENAME", "Unexpected BU raw filename.", relative)
            continue
        time_step = int(match.group(1))
        filename_ru = int(match.group(2))
        pair = (time_step, filename_ru)
        if pair in observed_pairs:
            record(errors, "DUPLICATE_PAIR", f"Duplicate raw pair {pair}.", relative)
        observed_pairs.add(pair)
        if file_path.stat().st_size == 0:
            record(errors, "ZERO_BYTE", "Raw file is empty.", relative)
            continue
        if sha256_file(file_path) != declared_hashes.get(relative):
            record(errors, "HASH_MISMATCH", "Raw SHA-256 differs.", relative)
        try:
            with np.load(file_path, allow_pickle=False) as payload:
                if set(payload.files) != set(PAYLOAD_FIELDS + IDENTITY_FIELDS):
                    record(errors, "FIELD_SET", "NPZ field set differs.", relative)
                    continue
                arrays = {field: np.asarray(payload[field]) for field in PAYLOAD_FIELDS}
                if int(scalar(np.asarray(payload["batch"]))) != 0:
                    record(errors, "BATCH_ID", "batch must be zero.", relative)
                if int(scalar(np.asarray(payload["timestep"]))) != time_step:
                    record(errors, "TIMESTEP_ID", "timestep identity differs.", relative)
                if int(scalar(np.asarray(payload["ru_tensor_index"]))) != filename_ru:
                    record(errors, "RU_INDEX", "RU tensor identity differs.", relative)
                if int(scalar(np.asarray(payload["ru_scenario_id"]))) != filename_ru:
                    record(errors, "RU_SCENARIO_ID", "RU scenario identity differs.", relative)
                if int(scalar(np.asarray(payload["ru_aodt_id"]))) != filename_ru + 1:
                    record(errors, "RU_AODT_ID", "RU AODT identity differs.", relative)
                for field, expected in identity_expected.items():
                    if not np.array_equal(np.asarray(payload[field]), expected):
                        record(errors, "UE_IDENTITY", f"{field} differs.", relative)
                if np.asarray(payload["full_antenna_pair"]).shape != ():
                    record(errors, "FULL_PAIR_SHAPE", "full_antenna_pair is not scalar.", relative)
        except (OSError, ValueError, EOFError) as exc:
            record(errors, "UNREADABLE_NPZ", f"Unable to read NPZ: {exc}", relative)
            continue
        for field, array in arrays.items():
            shapes[field].add(tuple(array.shape))
            dtypes[field].add(str(array.dtype))
            if str(array.dtype) != EXPECTED_DTYPES[field]:
                record(
                    errors,
                    "DTYPE_MISMATCH",
                    f"{field} dtype {array.dtype} differs.",
                    relative,
                )
            if not array.shape or array.shape[0] != 54:
                record(
                    errors,
                    "UE_AXIS_MISMATCH",
                    f"{field} first dimension must be 54; got {array.shape}.",
                    relative,
                )
            if np.iscomplexobj(array):
                nan_counts[field] += int(
                    np.count_nonzero(np.isnan(array.real))
                    + np.count_nonzero(np.isnan(array.imag))
                )
                inf_counts[field] += int(
                    np.count_nonzero(np.isinf(array.real))
                    + np.count_nonzero(np.isinf(array.imag))
                )
            else:
                nan_counts[field] += int(np.count_nonzero(np.isnan(array)))
                inf_counts[field] += int(np.count_nonzero(np.isinf(array)))
        values = arrays["values"]
        if values.shape[0] == 54:
            power_per_ue = np.sum(np.abs(values) ** 2, axis=tuple(range(1, values.ndim)))
            link_total_powers.extend(float(value) for value in power_per_ue)
            for uav_offset in range(4):
                signature = float(power_per_ue[50 + uav_offset])
                uav_power_signatures[(filename_ru, uav_offset)].add(
                    round(signature, 18)
                )
            active_coefficient_count += int(np.count_nonzero(np.abs(values) > 0))
            logical_link_rows += 54
            uav_logical_link_rows += 4
        delays = arrays["delays"]
        valid_delays = delays[np.isfinite(delays) & (delays >= 0)]
        if valid_delays.size:
            active_delays.extend(float(value) for value in valid_delays.ravel())

    missing_pairs = expected_pairs - observed_pairs
    unexpected_pairs = observed_pairs - expected_pairs
    if missing_pairs:
        record(errors, "MISSING_PAIRS", f"Missing {len(missing_pairs)} time/RU pairs.")
    if unexpected_pairs:
        record(
            errors, "UNEXPECTED_PAIRS", f"Unexpected {len(unexpected_pairs)} time/RU pairs."
        )
    if len(actual_files) != 2048:
        record(errors, "FILE_COUNT", f"Expected 2048 NPZ files, got {len(actual_files)}.")
    if logical_link_rows != 256 * 8 * 54:
        record(
            errors,
            "LOGICAL_LINK_COUNT",
            f"Expected 110592 logical links, got {logical_link_rows}.",
        )
    if uav_logical_link_rows != 256 * 8 * 4:
        record(
            errors,
            "UAV_LINK_COUNT",
            f"Expected 8192 UAV logical links, got {uav_logical_link_rows}.",
        )
    for field in PAYLOAD_FIELDS:
        if nan_counts[field] or inf_counts[field]:
            record(
                errors,
                "NONFINITE_PAYLOAD",
                f"{field} has NaN={nan_counts[field]} Inf={inf_counts[field]}.",
            )
    moving_uav_channel_pairs = sum(
        len(signatures) > 1 for signatures in uav_power_signatures.values()
    )
    for uav_offset in range(4):
        if not any(
            len(uav_power_signatures[(ru, uav_offset)]) > 1 for ru in range(8)
        ):
            record(
                errors,
                "UAV_CHANNEL_CONSTANT",
                f"UAV tensor index {50 + uav_offset} has no time-varying RU channel.",
            )
    if active_coefficient_count <= 0:
        record(errors, "NO_ACTIVE_CHANNEL", "No nonzero AODT CIR coefficient exists.")

    schema_report = {
        "schema_version": "aodt-uav-raw-schema-report-v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not errors else "fail",
        "source_declared_schema": declared_schema,
        "observed": {
            "file_count": len(actual_files),
            "shapes": {
                field: [list(shape) for shape in sorted(values)]
                for field, values in shapes.items()
            },
            "dtypes": {field: sorted(values) for field, values in dtypes.items()},
            "identity": {
                "ru_tensor_indices": [0, 7],
                "ue_tensor_indices": [0, 53],
                "uav_tensor_indices": [50, 51, 52, 53],
                "uav_aodt_ids": [51, 52, 53, 54],
            },
        },
    }
    quality_report = {
        "schema_version": "aodt-uav-raw-quality-report-v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not errors else "fail",
        "counts": {
            "raw_npz": len(actual_files),
            "logical_link_rows": logical_link_rows,
            "uav_logical_link_rows": uav_logical_link_rows,
            "active_coefficient_count": active_coefficient_count,
            "moving_uav_ru_channel_pairs": moving_uav_channel_pairs,
        },
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "link_total_power_quantiles": quantiles(link_total_powers),
        "nonnegative_delay_quantiles_s": quantiles(active_delays),
        "position_validation": positions["validation"],
        "gpx_lineage": {
            "trajectory_id": gpx_binding.get("trajectory_id"),
            "trajectory_hashes_verified": gpx_binding.get("trajectory_hashes_verified"),
            "runtime_confirmed": positions["validation"].get(
                "trajectory_runtime_verified"
            ),
        },
        "errors": errors,
        "warnings": warnings,
    }
    rf_inventory = {
        "schema_version": "aodt-uav-rf-field-inventory-v1",
        "generated_at_utc": utc_now(),
        "source": "AODT direct-gRPC raw tensors",
        "available": {
            "values": "linear complex channel coefficient",
            "delays": "seconds",
            "angles_of_departure": "radians",
            "angles_of_arrival": "radians",
            "ue_positions": "AODT scene-local metres for all 256 time steps",
        },
        "not_derived": [
            "path_loss_db",
            "received_power_dbm",
            "snr_db",
            "sinr_db",
            "cqi",
            "serving_ru",
        ],
        "fallback_used": False,
        "external_rf_values_used": False,
    }
    stage_status = {
        "stage": "08BU",
        "status": "pass" if not errors else "fail",
        "generated_at_utc": utc_now(),
        "source_07bu": str(stage_07.resolve()).replace("\\", "/"),
        "raw_npz_count": len(actual_files),
        "logical_link_row_count": logical_link_rows,
        "uav_logical_link_row_count": uav_logical_link_rows,
        "active_coefficient_count": active_coefficient_count,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "elapsed_seconds": time.perf_counter() - started,
        "trajectory_hashes_verified": not errors
        and gpx_binding.get("trajectory_hashes_verified") is True,
    }
    return {
        "stage_status": stage_status,
        "schema_report": schema_report,
        "quality_report": quality_report,
        "rf_inventory": rf_inventory,
    }


def write_outputs(run_dir: Path, result: dict[str, Any]) -> Path:
    stage_dir = run_dir / "08BU_raw_validation"
    try:
        stage_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite existing 08BU: {stage_dir}") from exc
    json_dump(stage_dir / "stage_status.json", result["stage_status"])
    json_dump(stage_dir / "aodt_uav_raw_schema_report.json", result["schema_report"])
    json_dump(stage_dir / "aodt_uav_raw_quality_report.json", result["quality_report"])
    json_dump(stage_dir / "aodt_uav_rf_field_inventory.json", result["rf_inventory"])
    report = [
        "# 08BU AODT GPX/UAV raw validation report",
        "",
        f"- Status: **{result['stage_status']['status']}**",
        f"- Raw NPZ files: {result['stage_status']['raw_npz_count']}",
        f"- Logical RU-UE-time rows: {result['stage_status']['logical_link_row_count']}",
        f"- UAV-only RU-UE-time rows: {result['stage_status']['uav_logical_link_row_count']}",
        f"- Trajectory hashes verified: {result['stage_status']['trajectory_hashes_verified']}",
        f"- Errors: {result['stage_status']['error_count']}",
        "",
        "08BU validates AODT raw channel tensors and GPX lineage. It does not "
        "substitute external RF/KPI values or derive SINR/CQI/serving-RU fields.",
        "",
    ]
    (stage_dir / "aodt_uav_raw_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json_load(manifest_path)
    manifest.setdefault("stages", {})["08BU"] = result["stage_status"]
    manifest["overall_status"] = (
        "pass_through_08BU"
        if result["stage_status"]["status"] == "pass"
        else "failed_at_08BU"
    )
    manifest["updated_at_utc"] = utc_now()
    manifest["finished_at_utc"] = utc_now()
    json_dump(manifest_path, manifest)
    return stage_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a completed AODT 07BU GPX/UAV raw collection."
    )
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Completed BU run. Defaults to the newest passing 07BU run.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_07bu_run(repo_root)
    print(f"[08BU] validating {run_dir}", flush=True)
    result = validate(run_dir)
    status = result["stage_status"]
    print(
        f"[08BU] status={status['status']} files={status['raw_npz_count']} "
        f"links={status['logical_link_row_count']} errors={status['error_count']}",
        flush=True,
    )
    if not args.no_write:
        output = write_outputs(run_dir, result)
        print(f"[08BU] outputs={output}", flush=True)
    return 0 if status["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
