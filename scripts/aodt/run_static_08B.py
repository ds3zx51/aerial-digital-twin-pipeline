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


STAGE_NAME = "08B_raw_validation"
FILE_PATTERN = re.compile(r"^cir_slot_(\d{4})_ru_(\d{2})\.npz$")
PAYLOAD_FIELDS = (
    "values",
    "delays",
    "angles_of_departure",
    "angles_of_arrival",
)
IDENTITY_FIELDS = (
    "batch",
    "slot",
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
UNITS = {
    "values": "linear complex channel coefficient",
    "delays": "seconds",
    "angles_of_departure": "radians (azimuth, zenith), TX-panel local frame",
    "angles_of_arrival": "radians (azimuth, zenith), RX-panel local frame",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def text_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def scalar(array: np.ndarray) -> Any:
    if array.shape != ():
        raise ValueError(f"expected scalar, got shape {array.shape}")
    return array.item()


def latest_07b_run(repo_root: Path) -> Path:
    results_root = repo_root / "results" / "aodt_static"
    candidates: list[tuple[float, Path]] = []
    if not results_root.is_dir():
        raise FileNotFoundError(f"AODT results root not found: {results_root}")
    for run_dir in results_root.iterdir():
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json_load(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("stages", {}).get("07B", {}).get("status") == "pass":
            candidates.append((manifest_path.stat().st_mtime, run_dir))
    if not candidates:
        raise FileNotFoundError("No completed AODT 07B run was found.")
    return max(candidates, key=lambda item: item[0])[1]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")


def validate(run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage_07b = run_dir / "07B_raw_collection"
    raw_dir = stage_07b / "raw"
    manifest_path = run_dir / "run_manifest.json"
    hashes_path = stage_07b / "raw_file_hashes.json"
    schema_path = stage_07b / "raw_schema_inventory.json"
    inventory_path = stage_07b / "raw_artifact_inventory.json"
    positions_path = run_dir / "06B_execution" / "position_registry.json"
    topology_path = run_dir / "03B_topology" / "aodt_node_mapping.json"

    for path in (
        manifest_path,
        hashes_path,
        schema_path,
        inventory_path,
        positions_path,
        topology_path,
    ):
        require_file(path)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw NPZ directory not found: {raw_dir}")

    manifest = json_load(manifest_path)
    hashes_document = json_load(hashes_path)
    declared_schema = json_load(schema_path)
    raw_inventory = json_load(inventory_path)
    positions = json_load(positions_path)
    topology = json_load(topology_path)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def record(
        target: list[dict[str, Any]],
        code: str,
        message: str,
        relative_path: str | None = None,
    ) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if relative_path is not None:
            item["relative_path"] = relative_path
        if len(target) < 100:
            target.append(item)

    if manifest.get("stages", {}).get("07B", {}).get("status") != "pass":
        record(errors, "07B_NOT_PASS", "The source run does not have a passing 07B stage.")

    expected_ru_count = int(manifest.get("scope", {}).get("ru", 0))
    expected_ue_count = int(manifest.get("scope", {}).get("ue", 0))
    if expected_ru_count <= 0 or expected_ue_count <= 0:
        record(errors, "INVALID_SCOPE", "Run manifest has invalid RU/UE counts.")

    declared_hashes = hashes_document.get("files", {})
    if hashes_document.get("algorithm") != "sha256" or not isinstance(declared_hashes, dict):
        record(errors, "INVALID_HASH_MANIFEST", "07B raw_file_hashes.json is invalid.")
        declared_hashes = {}

    actual_files = sorted(raw_dir.glob("*.npz"))
    actual_relpaths = {path.relative_to(stage_07b).as_posix() for path in actual_files}
    expected_relpaths = set(declared_hashes)
    missing_files = sorted(expected_relpaths - actual_relpaths)
    unexpected_files = sorted(actual_relpaths - expected_relpaths)
    if missing_files:
        record(
            errors,
            "MISSING_RAW_FILES",
            f"{len(missing_files)} files declared by 07B are missing; first={missing_files[0]}",
        )
    if unexpected_files:
        record(
            errors,
            "UNEXPECTED_RAW_FILES",
            f"{len(unexpected_files)} undeclared NPZ files exist; first={unexpected_files[0]}",
        )

    expected_slots = [
        int(value) for value in raw_inventory.get("selection", {}).get("slots", [])
    ]
    expected_rus = [
        int(value)
        for value in raw_inventory.get("selection", {}).get("ru_tensor_indices", [])
    ]
    expected_ues = np.arange(expected_ue_count, dtype=np.int64)
    expected_aodt_ues = np.arange(1, expected_ue_count + 1, dtype=np.int64)
    expected_pairs = {(slot, ru) for slot in expected_slots for ru in expected_rus}

    observed_pairs: set[tuple[int, int]] = set()
    duplicate_file_pairs = 0
    hash_mismatch_count = 0
    zero_byte_count = 0
    unreadable_count = 0
    schema_mismatch_count = 0
    identity_mismatch_count = 0
    nan_counts = {field: 0 for field in PAYLOAD_FIELDS}
    inf_counts = {field: 0 for field in PAYLOAD_FIELDS}
    shapes: dict[str, set[tuple[int, ...]]] = {
        field: set() for field in PAYLOAD_FIELDS
    }
    dtypes: dict[str, set[str]] = {field: set() for field in PAYLOAD_FIELDS}
    active_path_count = 0
    zero_path_link_count = 0
    logical_link_rows = 0
    link_path_counts: list[float] = []
    link_total_powers: list[float] = []
    nonzero_abs_h_min = math.inf
    nonzero_abs_h_max = -math.inf
    active_delay_min = math.inf
    active_delay_max = -math.inf
    active_delay_sum = 0.0
    active_delay_observations = 0
    active_angle_min = {
        "aod_azimuth": math.inf,
        "aod_zenith": math.inf,
        "aoa_azimuth": math.inf,
        "aoa_zenith": math.inf,
    }
    active_angle_max = {
        "aod_azimuth": -math.inf,
        "aod_zenith": -math.inf,
        "aoa_azimuth": -math.inf,
        "aoa_zenith": -math.inf,
    }
    delay_range_violation_count = 0
    angle_range_violation_count = 0
    temporal_signatures: dict[int, set[tuple[int, float]]] = defaultdict(set)

    for file_path in actual_files:
        relative_path = file_path.relative_to(stage_07b).as_posix()
        if file_path.stat().st_size == 0:
            zero_byte_count += 1
            record(errors, "ZERO_BYTE_RAW_FILE", "Raw NPZ is empty.", relative_path)
            continue

        match = FILE_PATTERN.match(file_path.name)
        if match is None:
            record(
                errors,
                "INVALID_RAW_FILENAME",
                "Filename does not encode slot and RU as required.",
                relative_path,
            )
            continue
        filename_slot = int(match.group(1))
        filename_ru = int(match.group(2))
        pair = (filename_slot, filename_ru)
        if pair in observed_pairs:
            duplicate_file_pairs += 1
        observed_pairs.add(pair)

        declared_hash = declared_hashes.get(relative_path)
        if declared_hash is not None and sha256_file(file_path) != declared_hash:
            hash_mismatch_count += 1
            record(
                errors,
                "RAW_HASH_MISMATCH",
                "File SHA-256 differs from the immutable 07B inventory.",
                relative_path,
            )

        try:
            with np.load(file_path, allow_pickle=False) as npz:
                available = set(npz.files)
                missing_arrays = (set(PAYLOAD_FIELDS) | set(IDENTITY_FIELDS)) - available
                if missing_arrays:
                    schema_mismatch_count += 1
                    record(
                        errors,
                        "MISSING_ARRAYS",
                        f"Missing arrays: {sorted(missing_arrays)}",
                        relative_path,
                    )
                    continue

                arrays = {field: np.asarray(npz[field]) for field in PAYLOAD_FIELDS}
                for field, array in arrays.items():
                    shapes[field].add(tuple(int(value) for value in array.shape))
                    dtypes[field].add(str(array.dtype))
                    if str(array.dtype) != EXPECTED_DTYPES[field]:
                        schema_mismatch_count += 1
                        record(
                            errors,
                            "DTYPE_MISMATCH",
                            f"{field} is {array.dtype}, expected {EXPECTED_DTYPES[field]}.",
                            relative_path,
                        )

                values = arrays["values"]
                delays = arrays["delays"]
                aod = arrays["angles_of_departure"]
                aoa = arrays["angles_of_arrival"]
                shape_ok = (
                    values.ndim == 9
                    and delays.ndim == 7
                    and aod.ndim == 8
                    and aoa.ndim == 8
                    and values.shape[0] == expected_ue_count
                    and delays.shape[0] == expected_ue_count
                    and aod.shape[0] == expected_ue_count
                    and aoa.shape[0] == expected_ue_count
                    and values.shape[-1] == delays.shape[-1]
                    and values.shape[-1] == aod.shape[-2]
                    and values.shape[-1] == aoa.shape[-2]
                    and aod.shape[-1] == 2
                    and aoa.shape[-1] == 2
                )
                if not shape_ok:
                    schema_mismatch_count += 1
                    record(
                        errors,
                        "SHAPE_MISMATCH",
                        "CIR tensor ranks, UE dimension, tap dimensions, or angle components disagree.",
                        relative_path,
                    )
                    continue

                for field, array in arrays.items():
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

                try:
                    payload_slot = int(scalar(np.asarray(npz["slot"])))
                    payload_ru = int(scalar(np.asarray(npz["ru_tensor_index"])))
                    identity_ok = (
                        int(scalar(np.asarray(npz["batch"]))) == 0
                        and payload_slot == filename_slot
                        and payload_ru == filename_ru
                        and int(scalar(np.asarray(npz["ru_scenario_id"]))) == filename_ru
                        and int(scalar(np.asarray(npz["ru_aodt_id"]))) == filename_ru + 1
                        and np.array_equal(
                            np.asarray(npz["ue_tensor_indices"]), expected_ues
                        )
                        and np.array_equal(
                            np.asarray(npz["ue_scenario_ids"]), expected_ues
                        )
                        and np.array_equal(
                            np.asarray(npz["ue_aodt_ids"]), expected_aodt_ues
                        )
                        and bool(scalar(np.asarray(npz["full_antenna_pair"]))) is False
                    )
                except (TypeError, ValueError):
                    identity_ok = False
                if not identity_ok:
                    identity_mismatch_count += 1
                    record(
                        errors,
                        "IDENTITY_MISMATCH",
                        "Filename, slot/RU scalars, UE arrays, or AODT IDs disagree.",
                        relative_path,
                    )

                magnitude = np.abs(values)
                path_mask = np.any(
                    magnitude > 0.0, axis=tuple(range(1, values.ndim - 1))
                )
                path_counts = np.sum(path_mask, axis=1)
                link_path_counts.extend(float(value) for value in path_counts)
                logical_link_rows += int(path_mask.shape[0])
                active_here = int(np.count_nonzero(path_mask))
                active_path_count += active_here
                zero_path_link_count += int(np.count_nonzero(path_counts == 0))

                nonzero_magnitudes = magnitude[magnitude > 0.0]
                if nonzero_magnitudes.size:
                    nonzero_abs_h_min = min(
                        nonzero_abs_h_min, float(np.min(nonzero_magnitudes))
                    )
                    nonzero_abs_h_max = max(
                        nonzero_abs_h_max, float(np.max(nonzero_magnitudes))
                    )

                link_power = np.sum(
                    np.square(magnitude, dtype=np.float64),
                    axis=tuple(range(1, magnitude.ndim)),
                )
                link_total_powers.extend(float(value) for value in link_power)
                total_file_power = float(np.sum(link_power, dtype=np.float64))
                temporal_signatures[filename_ru].add((active_here, total_file_power))

                delay_view = delays.reshape(expected_ue_count, -1, delays.shape[-1])
                aod_view = aod.reshape(
                    expected_ue_count, -1, aod.shape[-2], aod.shape[-1]
                )
                aoa_view = aoa.reshape(
                    expected_ue_count, -1, aoa.shape[-2], aoa.shape[-1]
                )
                if (
                    delay_view.shape[1] != 1
                    or aod_view.shape[1] != 1
                    or aoa_view.shape[1] != 1
                ):
                    schema_mismatch_count += 1
                    record(
                        errors,
                        "UNEXPECTED_ANTENNA_LAYOUT",
                        "08B expected singleton non-UE/non-tap dimensions for full_antenna_pair=false.",
                        relative_path,
                    )
                    continue

                active_delays = delay_view[:, 0, :][path_mask]
                active_aod = aod_view[:, 0, :, :][path_mask]
                active_aoa = aoa_view[:, 0, :, :][path_mask]
                if active_delays.size:
                    active_delay_min = min(active_delay_min, float(np.min(active_delays)))
                    active_delay_max = max(active_delay_max, float(np.max(active_delays)))
                    active_delay_sum += float(np.sum(active_delays, dtype=np.float64))
                    active_delay_observations += int(active_delays.size)
                    delay_range_violation_count += int(
                        np.count_nonzero((active_delays < 0.0) | (active_delays >= 0.1))
                    )

                for prefix, active_angles in (("aod", active_aod), ("aoa", active_aoa)):
                    if not active_angles.size:
                        continue
                    azimuth = active_angles[:, 0]
                    zenith = active_angles[:, 1]
                    active_angle_min[f"{prefix}_azimuth"] = min(
                        active_angle_min[f"{prefix}_azimuth"], float(np.min(azimuth))
                    )
                    active_angle_max[f"{prefix}_azimuth"] = max(
                        active_angle_max[f"{prefix}_azimuth"], float(np.max(azimuth))
                    )
                    active_angle_min[f"{prefix}_zenith"] = min(
                        active_angle_min[f"{prefix}_zenith"], float(np.min(zenith))
                    )
                    active_angle_max[f"{prefix}_zenith"] = max(
                        active_angle_max[f"{prefix}_zenith"], float(np.max(zenith))
                    )
                    angle_range_violation_count += int(
                        np.count_nonzero(
                            (azimuth < 0.0)
                            | (azimuth > 2.0 * math.pi + 1e-5)
                            | (zenith < 0.0)
                            | (zenith > math.pi + 1e-5)
                        )
                    )
        except (OSError, ValueError, TypeError) as exc:
            unreadable_count += 1
            record(errors, "UNREADABLE_NPZ", str(exc), relative_path)

    if duplicate_file_pairs:
        record(
            errors,
            "DUPLICATE_SLOT_RU",
            f"{duplicate_file_pairs} duplicate slot/RU file identities were found.",
        )
    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - expected_pairs)
    if missing_pairs:
        record(
            errors,
            "MISSING_SLOT_RU",
            f"{len(missing_pairs)} slot/RU pairs are missing; first={missing_pairs[0]}",
        )
    if unexpected_pairs:
        record(
            errors,
            "UNEXPECTED_SLOT_RU",
            f"{len(unexpected_pairs)} unexpected slot/RU pairs exist; first={unexpected_pairs[0]}",
        )
    for field in PAYLOAD_FIELDS:
        if nan_counts[field]:
            record(errors, "NAN_VALUES", f"{field} contains {nan_counts[field]} NaN scalars.")
        if inf_counts[field]:
            record(errors, "INF_VALUES", f"{field} contains {inf_counts[field]} Inf scalars.")
    if active_path_count == 0:
        record(
            errors,
            "NO_USABLE_RF_CHANNEL",
            "No non-zero AODT CIR path was found; downstream processing is forbidden.",
        )
    if zero_path_link_count:
        record(
            errors,
            "LINKS_WITHOUT_PATHS",
            f"{zero_path_link_count} logical RU/UE/slot links have no non-zero CIR path.",
        )
    if delay_range_violation_count:
        record(
            errors,
            "ACTIVE_DELAY_RANGE",
            f"{delay_range_violation_count} active delays are outside [0, 0.1) seconds.",
        )
    if angle_range_violation_count:
        record(
            errors,
            "ACTIVE_ANGLE_RANGE",
            f"{angle_range_violation_count} active AoA/AoD samples violate azimuth/zenith ranges.",
        )

    first_ru = np.asarray(positions.get("first", {}).get("ru_positions", []), dtype=float)
    last_ru = np.asarray(positions.get("last", {}).get("ru_positions", []), dtype=float)
    first_ue = np.asarray(positions.get("first", {}).get("ue_positions", []), dtype=float)
    last_ue = np.asarray(positions.get("last", {}).get("ue_positions", []), dtype=float)
    coordinate_shapes_ok = (
        first_ru.shape == (expected_ru_count, 3)
        and last_ru.shape == (expected_ru_count, 3)
        and first_ue.shape == (expected_ue_count, 3)
        and last_ue.shape == (expected_ue_count, 3)
    )
    coordinates_finite = bool(
        coordinate_shapes_ok
        and np.all(np.isfinite(first_ru))
        and np.all(np.isfinite(last_ru))
        and np.all(np.isfinite(first_ue))
        and np.all(np.isfinite(last_ue))
    )
    ru_static_delta = (
        float(np.max(np.abs(first_ru - last_ru))) if coordinate_shapes_ok else None
    )
    ue_static_delta = (
        float(np.max(np.abs(first_ue - last_ue))) if coordinate_shapes_ok else None
    )
    coordinate_consistency = bool(
        coordinate_shapes_ok
        and coordinates_finite
        and ru_static_delta == 0.0
        and ue_static_delta == 0.0
        and positions.get("validation", {}).get("status") == "pass"
        and topology.get("validation", {}).get("coordinates_preserved") is True
    )
    if not coordinate_consistency:
        record(
            errors,
            "COORDINATE_INCONSISTENCY",
            "RU/UE position registry is non-finite, mobile, malformed, or disagrees with topology validation.",
        )

    temporal_unique_by_ru = {
        str(ru): len(signatures)
        for ru, signatures in sorted(temporal_signatures.items())
    }
    temporally_nonconstant_ru_count = sum(
        count > 1 for count in temporal_unique_by_ru.values()
    )
    spatially_nonconstant = bool(
        link_total_powers
        and np.ptp(np.asarray(link_total_powers, dtype=np.float64)) > 0.0
        and len(set(int(value) for value in link_path_counts)) > 1
    )
    placeholder_rejected = bool(
        active_path_count > 0
        and zero_path_link_count == 0
        and spatially_nonconstant
        and temporally_nonconstant_ru_count > 0
    )
    if not placeholder_rejected:
        record(
            warnings,
            "PLACEHOLDER_NOT_FULLY_REJECTED",
            "The raw tensors are usable, but diversity checks could not independently reject a constant placeholder.",
        )

    expected_logical_rows = len(expected_pairs) * expected_ue_count
    if logical_link_rows != expected_logical_rows:
        record(
            errors,
            "LOGICAL_ROW_COUNT",
            f"Observed {logical_link_rows} logical links, expected {expected_logical_rows}.",
        )

    status = "fail" if errors else ("pass_with_warnings" if warnings else "pass")
    elapsed = time.perf_counter() - started

    finite_min = lambda value: None if math.isinf(value) else float(value)
    schema_report = {
        "schema_version": "aodt-static-raw-schema-report-v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "source_run": run_dir.as_posix(),
        "source_format": "NumPy compressed NPZ",
        "logical_row_definition": "one (slot, RU, UE) CIR link record",
        "column_names_applicable": False,
        "array_names": list(PAYLOAD_FIELDS + IDENTITY_FIELDS),
        "file_count": len(actual_files),
        "logical_link_row_count": logical_link_rows,
        "expected_logical_link_row_count": expected_logical_rows,
        "tensor_schema": {
            field: {
                "observed_shapes": [list(shape) for shape in sorted(shapes[field])],
                "observed_dtypes": sorted(dtypes[field]),
                "required_dtype": EXPECTED_DTYPES[field],
                "unit": UNITS[field],
            }
            for field in PAYLOAD_FIELDS
        },
        "identity_schema": {
            "batch": "int64 scalar; batch 0",
            "slot": "int64 scalar; 0..255",
            "ru_tensor_index": "int64 scalar; 0..7",
            "ru_scenario_id": "int64 scalar; equals tensor index",
            "ru_aodt_id": "int64 scalar; tensor index + 1",
            "ue_tensor_indices": "int64[50]; 0..49",
            "ue_scenario_ids": "int64[50]; 0..49",
            "ue_aodt_ids": "int64[50]; 1..50",
            "full_antenna_pair": "bool scalar; false",
        },
        "declared_07b_schema_version": declared_schema.get("schema_version"),
        "client_layout_contract": {
            "values": "[rx][sample][rx_h][rx_v][rx_p][tx_h][tx_v][tx_p][tap]",
            "delays": "[rx][sample][rx_h][rx_v][tx_h][tx_v][tap]",
            "angles": "[rx][sample][rx_h][rx_v][tx_h][tx_v][tap][azimuth, zenith]",
        },
    }

    rf_inventory = {
        "schema_version": "aodt-static-rf-field-inventory-v1",
        "generated_at_utc": utc_now(),
        "status": status,
        "direct_aodt_fields": {
            "values": {
                "present": bool(shapes["values"]),
                "semantic": "complex multipath CIR coefficient",
                "unit": UNITS["values"],
            },
            "delays": {
                "present": bool(shapes["delays"]),
                "semantic": "multipath propagation delay",
                "unit": UNITS["delays"],
            },
            "angles_of_departure": {
                "present": bool(shapes["angles_of_departure"]),
                "semantic": "per-path departure azimuth and zenith",
                "unit": UNITS["angles_of_departure"],
            },
            "angles_of_arrival": {
                "present": bool(shapes["angles_of_arrival"]),
                "semantic": "per-path arrival azimuth and zenith",
                "unit": UNITS["angles_of_arrival"],
            },
            "multipath_information": {
                "present": active_path_count > 0,
                "semantic": "non-zero CIR taps with aligned delay, AoD, and AoA",
            },
        },
        "not_present_in_aodt_raw": {
            field: {
                "present": False,
                "fabricated": False,
                "downstream_rule": rule,
            }
            for field, rule in {
                "path_gain_db": "May be derived only with a documented CIR power normalization.",
                "path_loss_db": "Must not be treated as the sign-flipped value of an undocumented gain.",
                "received_power_dbm": "Requires transmit power and antenna/link-budget definitions.",
                "rsrp_dbm": "Requires an NR reference-signal resource definition; not present in CIR raw.",
                "snr_db": "Requires a documented signal-power and receiver-noise model.",
                "sinr_db": "Requires serving-RU selection, interference accounting, and receiver noise.",
                "serving_ru": "Requires an explicit deterministic association rule.",
                "path_type": "No direct AODT path-type label was returned.",
                "los_state": "No direct AODT LOS/NLOS label was returned.",
            }.items()
        },
        "claim_boundary": (
            "08B validates direct AODT CIR tensors only. It does not label linear "
            "coefficients as dB/dBm and does not fabricate SINR, RSRP, path loss, "
            "serving RU, path type, or LOS state."
        ),
    }

    quality_report = {
        "schema_version": "aodt-static-raw-quality-report-v1",
        "generated_at_utc": utc_now(),
        "stage": "08B",
        "status": status,
        "source_run": run_dir.as_posix(),
        "elapsed_seconds": elapsed,
        "coverage": {
            "slots_expected": len(expected_slots),
            "slots_observed": len({slot for slot, _ in observed_pairs}),
            "slot_min": min(expected_slots) if expected_slots else None,
            "slot_max": max(expected_slots) if expected_slots else None,
            "rus_expected": len(expected_rus),
            "rus_observed": len({ru for _, ru in observed_pairs}),
            "ues_per_slot_ru": expected_ue_count,
            "raw_npz_expected": len(expected_relpaths),
            "raw_npz_observed": len(actual_files),
            "logical_link_rows_expected": expected_logical_rows,
            "logical_link_rows_observed": logical_link_rows,
            "missing_slot_ru_count": len(missing_pairs),
            "unexpected_slot_ru_count": len(unexpected_pairs),
            "duplicate_slot_ru_count": duplicate_file_pairs,
        },
        "integrity": {
            "sha256_files_checked": len(actual_relpaths & expected_relpaths),
            "sha256_mismatch_count": hash_mismatch_count,
            "zero_byte_file_count": zero_byte_count,
            "unreadable_npz_count": unreadable_count,
        },
        "schema_and_identity": {
            "schema_mismatch_count": schema_mismatch_count,
            "identity_mismatch_count": identity_mismatch_count,
            "ue_ids_expected": [0, max(expected_ue_count - 1, 0)],
            "ue_aodt_ids_expected": [1, expected_ue_count],
            "ru_ids_expected": [0, max(expected_ru_count - 1, 0)],
            "ru_aodt_ids_expected": [1, expected_ru_count],
        },
        "coordinates": {
            "status": "pass" if coordinate_consistency else "fail",
            "ru_shape": list(first_ru.shape),
            "ue_shape": list(first_ue.shape),
            "all_finite": coordinates_finite,
            "ru_first_last_max_abs_delta_m": ru_static_delta,
            "ue_first_last_max_abs_delta_m": ue_static_delta,
            "mobility_contract": "fixed UE; no UAV; no GPX",
        },
        "finite_values": {
            "nan_scalar_counts": nan_counts,
            "inf_scalar_counts": inf_counts,
        },
        "rf_quality": {
            "active_path_count": active_path_count,
            "zero_path_link_count": zero_path_link_count,
            "paths_per_link": quantiles(link_path_counts),
            "link_total_linear_coefficient_power": quantiles(link_total_powers),
            "nonzero_abs_h": {
                "min": finite_min(nonzero_abs_h_min),
                "max": finite_min(nonzero_abs_h_max),
            },
            "active_delay_seconds": {
                "min": finite_min(active_delay_min),
                "mean": (
                    active_delay_sum / active_delay_observations
                    if active_delay_observations
                    else None
                ),
                "max": finite_min(active_delay_max),
                "range_violation_count": delay_range_violation_count,
            },
            "active_angle_radians": {
                key: {
                    "min": finite_min(active_angle_min[key]),
                    "max": finite_min(active_angle_max[key]),
                }
                for key in active_angle_min
            },
            "angle_range_violation_count": angle_range_violation_count,
        },
        "placeholder_checks": {
            "spatially_nonconstant": spatially_nonconstant,
            "temporal_signature_count_by_ru": temporal_unique_by_ru,
            "temporally_nonconstant_ru_count": temporally_nonconstant_ru_count,
            "constant_placeholder_rejected": placeholder_rejected,
        },
        "errors": errors,
        "warnings": warnings,
    }

    result = {
        "status": status,
        "generated_at_utc": quality_report["generated_at_utc"],
        "elapsed_seconds": elapsed,
        "source_run": run_dir,
        "manifest": manifest,
        "schema_report": schema_report,
        "rf_inventory": rf_inventory,
        "quality_report": quality_report,
        "stage_status": {
            "stage": "08B",
            "status": status,
            "generated_at_utc": quality_report["generated_at_utc"],
            "source_07b": stage_07b.as_posix(),
            "raw_npz_count": len(actual_files),
            "logical_link_row_count": logical_link_rows,
            "active_path_count": active_path_count,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "elapsed_seconds": elapsed,
        },
    }
    return result


def markdown_report(result: dict[str, Any]) -> str:
    quality = result["quality_report"]
    coverage = quality["coverage"]
    integrity = quality["integrity"]
    rf = quality["rf_quality"]
    coordinates = quality["coordinates"]
    placeholder = quality["placeholder_checks"]
    lines = [
        "# 08B AODT raw RF/channel validation report",
        "",
        f"- Status: **{result['status']}**",
        f"- Source run: `{result['source_run'].as_posix()}`",
        f"- Validation time: {result['elapsed_seconds']:.3f} seconds",
        "",
        "## Coverage and integrity",
        "",
        f"- Raw NPZ files: {coverage['raw_npz_observed']} / {coverage['raw_npz_expected']}",
        f"- Slots: {coverage['slots_observed']} ({coverage['slot_min']} through {coverage['slot_max']})",
        f"- RUs: {coverage['rus_observed']}; UEs per slot/RU: {coverage['ues_per_slot_ru']}",
        f"- Logical link rows: {coverage['logical_link_rows_observed']} / {coverage['logical_link_rows_expected']}",
        f"- Missing slot/RU pairs: {coverage['missing_slot_ru_count']}",
        f"- Duplicate slot/RU pairs: {coverage['duplicate_slot_ru_count']}",
        f"- SHA-256 mismatches: {integrity['sha256_mismatch_count']}",
        f"- Empty or unreadable files: {integrity['zero_byte_file_count'] + integrity['unreadable_npz_count']}",
        "",
        "## Channel quality",
        "",
        f"- Active multipath taps: {rf['active_path_count']}",
        f"- Links without a path: {rf['zero_path_link_count']}",
        f"- Paths/link: min={rf['paths_per_link']['min']}, median={rf['paths_per_link']['median']}, max={rf['paths_per_link']['max']}",
        f"- Active delay range: {rf['active_delay_seconds']['min']} to {rf['active_delay_seconds']['max']} seconds",
        f"- AoA/AoD range violations: {rf['angle_range_violation_count']}",
        f"- NaN counts: {quality['finite_values']['nan_scalar_counts']}",
        f"- Inf counts: {quality['finite_values']['inf_scalar_counts']}",
        "",
        "## Static topology and placeholder checks",
        "",
        f"- Coordinate status: {coordinates['status']}",
        f"- RU first/last maximum delta: {coordinates['ru_first_last_max_abs_delta_m']} m",
        f"- UE first/last maximum delta: {coordinates['ue_first_last_max_abs_delta_m']} m",
        f"- Constant placeholder rejected: {placeholder['constant_placeholder_rejected']}",
        f"- RUs with more than one temporal signature: {placeholder['temporally_nonconstant_ru_count']}",
        "",
        "## RF field boundary",
        "",
        "- Direct AODT measurements: complex CIR coefficients, delays, AoD, and AoA.",
        "- The raw tensors do not directly contain path gain/loss in dB, received power/RSRP in dBm, SNR/SINR, serving RU, path type, or LOS state.",
        "- No missing RF field was filled with a constant, random value, or AODT output.",
        "",
    ]
    if quality["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(
            f"- `{item['code']}`: {item['message']}" for item in quality["errors"]
        )
        lines.append("")
    if quality["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(
            f"- `{item['code']}`: {item['message']}" for item in quality["warnings"]
        )
        lines.append("")
    return "\n".join(lines)


def write_outputs(run_dir: Path, result: dict[str, Any], command: list[str]) -> Path:
    stage_dir = run_dir / STAGE_NAME
    if stage_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing 08B result: {stage_dir}"
        )
    stage_dir.mkdir(parents=False)
    json_dump(stage_dir / "aodt_raw_schema_report.json", result["schema_report"])
    json_dump(stage_dir / "aodt_rf_field_inventory.json", result["rf_inventory"])
    json_dump(stage_dir / "aodt_raw_quality_report.json", result["quality_report"])
    text_write(
        stage_dir / "aodt_raw_validation_report.md", markdown_report(result)
    )
    json_dump(stage_dir / "stage_status.json", result["stage_status"])

    manifest = result["manifest"]
    manifest.setdefault("stages", {})["08B"] = result["stage_status"] | {
        "schema_report": (
            stage_dir / "aodt_raw_schema_report.json"
        ).as_posix(),
        "rf_field_inventory": (
            stage_dir / "aodt_rf_field_inventory.json"
        ).as_posix(),
        "quality_report": (
            stage_dir / "aodt_raw_quality_report.json"
        ).as_posix(),
        "validation_report": (
            stage_dir / "aodt_raw_validation_report.md"
        ).as_posix(),
        "command": command,
    }
    manifest["overall_status"] = (
        "pass_through_08B"
        if result["status"] in {"pass", "pass_with_warnings"}
        else "failed_at_08B"
    )
    manifest["updated_at_utc"] = result["generated_at_utc"]
    manifest["finished_at_utc"] = result["generated_at_utc"]
    json_dump(run_dir / "run_manifest.json", manifest)
    return stage_dir


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Validate a completed AODT static 07B raw CIR collection."
    )
    result.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="AODT experiment repository root",
    )
    result.add_argument(
        "--run-dir",
        type=Path,
        help="Completed AODT run. Defaults to the newest passing 07B run.",
    )
    result.add_argument(
        "--no-write",
        action="store_true",
        help="Run the complete validator without creating 08B outputs.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_07b_run(repo_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not args.no_write and (run_dir / STAGE_NAME).exists():
        raise FileExistsError(
            f"Refusing to overwrite existing stage: {run_dir / STAGE_NAME}"
        )

    print(f"[08B] validating {run_dir}", flush=True)
    result = validate(run_dir)
    print(
        "[08B] "
        f"status={result['status']} "
        f"files={result['quality_report']['coverage']['raw_npz_observed']} "
        f"links={result['quality_report']['coverage']['logical_link_rows_observed']} "
        f"paths={result['quality_report']['rf_quality']['active_path_count']} "
        f"elapsed={result['elapsed_seconds']:.3f}s",
        flush=True,
    )
    if args.no_write:
        print("[08B] no-write validation complete", flush=True)
    else:
        stage_dir = write_outputs(run_dir, result, list(sys.argv))
        print(f"[08B] outputs={stage_dir}", flush=True)
        print(f"OVERALL_STATUS={result['manifest']['overall_status']}", flush=True)
    return 0 if result["status"] in {"pass", "pass_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
