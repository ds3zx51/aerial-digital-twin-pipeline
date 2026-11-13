#!/usr/bin/env python3
"""Validate the AODT BU trajectory, GPX, manifest, and scenario."""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
import math
from pathlib import Path
from typing import Any

try:
    from .generate_shared_multi_uav_gpx import (
        CANONICAL_REL, CSV_FIELDS, CSV_REL, DESIGN_REL, GPX_RELS, MANIFEST_REL,
        PREVIEW_REL, SCENARIO_REL, TRAJECTORY_ID, VALIDATION_REL, load_json,
        resolve_inputs, sha256, validate_design_config, write_json,
    )
    from .shared_geodetic import wgs84_to_scene_enu
    from .shared_gpx import read_gpx
except ImportError:
    from generate_shared_multi_uav_gpx import (
        CANONICAL_REL, CSV_FIELDS, CSV_REL, DESIGN_REL, GPX_RELS, MANIFEST_REL,
        PREVIEW_REL, SCENARIO_REL, TRAJECTORY_ID, VALIDATION_REL, load_json,
        resolve_inputs, sha256, validate_design_config, write_json,
    )
    from shared_geodetic import wgs84_to_scene_enu
    from shared_gpx import read_gpx


def validate_bu_scenario(scenario: dict[str, Any], errors: list[str]) -> None:
    if scenario.get("branch_id") != "BU" or scenario.get("backend") != "aodt":
        errors.append("BU scenario branch_id/backend is invalid")
    if scenario.get("experiment_mode") != "multi_uav_uplink_interference":
        errors.append("BU experiment_mode is invalid")
    if scenario.get("trajectory_id") != TRAJECTORY_ID:
        errors.append("BU scenario trajectory_id is invalid")
    if scenario.get("ue_counts") != {"ground": 50, "aerial": 4, "total": 54}:
        errors.append("BU scenario must define 50 ground + 4 aerial = 54 UE")
    if len(scenario.get("ue", [])) != 50 or len(scenario.get("bs", [])) != 8:
        errors.append("BU scenario must preserve 50 ground UE and 8 RU")
    aerial = scenario.get("aerial_ues")
    if not isinstance(aerial, list) or len(aerial) != 4:
        errors.append("BU scenario must define four aerial_ues")
        return
    required = {
        "ue_id", "ue_type", "network_role", "analysis_role", "mobility_type",
        "trajectory_start_reference_ru_id", "gpx_file", "export_to_raw_telemetry",
        "include_in_ground_impact_view",
    }
    for index, node in enumerate(aerial, start=1):
        if not required <= set(node):
            errors.append(f"aerial UE {index} is missing required fields")
        if node.get("ue_id") != f"uav_{index:02d}" or node.get("ue_type") != "aerial":
            errors.append(f"aerial UE {index} identity/type is invalid")
        if node.get("network_role") != "regular_ue" or node.get("mobility_type") != "gpx":
            errors.append(f"aerial UE {index} must be a regular GPX UE")
        if node.get("analysis_role") != "uplink_interference_source":
            errors.append(f"aerial UE {index} analysis_role is invalid")
        if node.get("export_to_raw_telemetry") is not True:
            errors.append(f"aerial UE {index} must export future raw telemetry")
        if node.get("include_in_ground_impact_view") is not False:
            errors.append(f"aerial UE {index} must stay outside the ground-impact view")
        if "serving_ru_id" in node:
            errors.append(f"aerial UE {index} must not preassign a serving RU")
    phase_execution = scenario.get("phase_execution", {})
    for key in ("aodt_executed", "scheduler_executed", "advtabddpm_executed"):
        if phase_execution.get(key) is not False:
            errors.append(f"BU scenario must record {key}=false")


def _manifest_hashes(root: Path, manifest: dict[str, Any], errors: list[str]) -> bool:
    entries = [
        (manifest.get("design_config_path"), manifest.get("design_config_sha256")),
        (manifest.get("canonical_enu_path"), manifest.get("canonical_enu_sha256")),
        (manifest.get("resolved_csv_path"), manifest.get("resolved_csv_sha256")),
        (manifest.get("preview_path"), manifest.get("preview_sha256")),
        (manifest.get("bu_scenario_path"), manifest.get("bu_scenario_sha256")),
    ]
    for item in manifest.get("gpx_files", []):
        if isinstance(item, dict):
            entries.append((item.get("path"), item.get("sha256")))
    passed = True
    for path_text, expected in entries:
        path = root / str(path_text)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"manifest SHA256 mismatch: {path_text}")
            passed = False
    return passed


def validate_artifact_root(artifact_root: Path, reference_project_root: Path) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    reference_project_root = reference_project_root.resolve()
    errors: list[str] = []
    try:
        design = load_json(artifact_root / DESIGN_REL)
        validate_design_config(design)
        inputs = resolve_inputs(reference_project_root, artifact_root / DESIGN_REL)
        canonical = load_json(artifact_root / CANONICAL_REL)
        manifest = load_json(artifact_root / MANIFEST_REL)
        scenario = load_json(artifact_root / SCENARIO_REL)
    except Exception as exc:
        return {"passed": False, "errors": [str(exc)]}
    hash_passed = _manifest_hashes(artifact_root, manifest, errors)
    if manifest.get("source_scenario_sha256") != sha256(inputs["scenario_path"]):
        errors.append("base static AODT scenario SHA256 differs from manifest")
    if manifest.get("geodetic_metadata_sha256") != sha256(inputs["conversion_path"]):
        errors.append("geodetic metadata SHA256 differs from manifest")
    if canonical.get("trajectory_id") != TRAJECTORY_ID or canonical.get("trajectory_type") != "synthetic_multi_uav":
        errors.append("canonical trajectory identity/type is invalid")
    if canonical.get("coordinate_frame") != "ENU":
        errors.append("canonical trajectory coordinate_frame must be ENU")
    if canonical.get("sample_count") != 256 or not math.isclose(float(canonical.get("sample_interval_s", -1)), inputs["sample_interval_s"]):
        errors.append("canonical trajectory timing does not match A")
    routes = canonical.get("routes")
    if not isinstance(routes, list) or len(routes) != 4:
        errors.append("canonical trajectory must contain four routes")
        routes = []
    reference_ids = [route.get("trajectory_start_reference_ru_id") for route in routes]
    if reference_ids != design["trajectory_start_reference_ru_ids"] or len(set(reference_ids)) != 4:
        errors.append("trajectory_start_reference_ru_ids are not the four fixed design values")
    georef = inputs["georef"]
    bounds = inputs["scene_bounds"]
    margin = float(design["scene_edge_margin_m"])
    expected_times = []
    if routes:
        expected_times = [point.get("timestamp") for point in routes[0].get("points", [])]
    unique_counts: dict[str, int] = {}
    distances: dict[str, float] = {}
    route_speeds: dict[str, float] = {}
    maximum_round_trip_error = 0.0
    all_inside = True
    edge_margin_passed = True
    timestamps_monotonic = True
    timestamps_synchronized = True
    no_nan = True
    no_duplicate_ids = True
    no_missing_ids = True
    altitude_passed = True
    gpx_passed = True

    for route_index, route in enumerate(routes):
        ue_id = str(route.get("ue_id"))
        points = route.get("points")
        if not isinstance(points, list) or len(points) != 256:
            errors.append(f"{ue_id} must contain 256 canonical points")
            continue
        sample_ids = [point.get("sample_id") for point in points]
        if len(set(sample_ids)) != len(sample_ids):
            no_duplicate_ids = False
            errors.append(f"{ue_id} contains duplicate sample IDs")
        if sample_ids != list(range(256)):
            no_missing_ids = False
            errors.append(f"{ue_id} sample IDs must be exactly 0..255")
        times = [point.get("timestamp") for point in points]
        if times != expected_times:
            timestamps_synchronized = False
            errors.append(f"{ue_id} timestamps are not synchronized")
        parsed_gpx = None
        try:
            parsed_gpx = read_gpx(artifact_root / GPX_RELS[route_index])
        except Exception as exc:
            gpx_passed = False
            errors.append(str(exc))
        if parsed_gpx and len(parsed_gpx["points"]) != 256:
            gpx_passed = False
            errors.append(f"{ue_id} GPX must contain 256 track points")
        previous_time = None
        positions: list[tuple[float, float, float]] = []
        for sample_id, point in enumerate(points):
            try:
                current_time = parsed_gpx["points"][sample_id]["datetime"] if parsed_gpx else None
                values = tuple(float(point[key]) for key in ("east_m", "north_m", "up_m"))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError
                positions.append(values)
                if current_time is not None and previous_time is not None:
                    if current_time <= previous_time:
                        timestamps_monotonic = False
                    if current_time - previous_time != timedelta(seconds=inputs["sample_interval_s"]):
                        errors.append(f"{ue_id} GPX interval differs from A at sample {sample_id}")
                previous_time = current_time
                inside = all(bounds[axis][0] <= value <= bounds[axis][1] for axis, value in zip(("x", "y", "z"), values))
                all_inside = all_inside and inside
                coordinate_epsilon_m = 1e-6
                edge_margin_passed = edge_margin_passed and (
                    bounds["x"][0] + margin - coordinate_epsilon_m <= values[0] <= bounds["x"][1] - margin + coordinate_epsilon_m
                    and bounds["y"][0] + margin - coordinate_epsilon_m <= values[1] <= bounds["y"][1] - margin + coordinate_epsilon_m
                )
                latitude = float(point["latitude_deg"])
                longitude = float(point["longitude_deg"])
                elevation = float(point["gpx_ele_m"])
                round_position = wgs84_to_scene_enu(latitude, longitude, elevation, georef)
                maximum_round_trip_error = max(maximum_round_trip_error, math.dist(values, round_position))
                expected_altitude = georef.altitude_m + float(point["scene_ground_offset_m"]) + 60.0
                altitude_passed = altitude_passed and math.isclose(elevation, expected_altitude, abs_tol=1e-6)
                if parsed_gpx:
                    gpx_point = parsed_gpx["points"][sample_id]
                    if not (
                        math.isclose(gpx_point["latitude_deg"], latitude, abs_tol=1e-9)
                        and math.isclose(gpx_point["longitude_deg"], longitude, abs_tol=1e-9)
                        and math.isclose(gpx_point["gpx_ele_m"], elevation, abs_tol=1e-6)
                        and gpx_point["timestamp"] == point["timestamp"]
                    ):
                        gpx_passed = False
                        errors.append(f"{ue_id} GPX differs from canonical sample {sample_id}")
            except (KeyError, TypeError, ValueError, IndexError):
                no_nan = False
                errors.append(f"{ue_id} canonical point {sample_id} is invalid")
        unique_counts[ue_id] = len(set(positions))
        distances[ue_id] = sum(math.dist(left, right) for left, right in zip(positions, positions[1:]))
        route_speeds[ue_id] = float(route.get("average_speed_mps", math.inf))
    if not timestamps_monotonic:
        errors.append("GPX timestamps must be strictly monotonic")
    if not all_inside:
        errors.append("one or more trajectory points are outside scene bounds")
    if not edge_margin_passed:
        errors.append("one or more trajectory points violate the scene edge margin")
    if not altitude_passed:
        errors.append("GPX elevation does not resolve 60 m AGL from the stored ground offset")
    if maximum_round_trip_error > float(design["coordinate_round_trip_tolerance_m"]):
        errors.append("coordinate round-trip tolerance exceeded")

    minimum_separation = math.inf
    if len(routes) == 4 and all(len(route.get("points", [])) == 256 for route in routes):
        for sample_id in range(256):
            positions = [
                (float(route["points"][sample_id]["east_m"]), float(route["points"][sample_id]["north_m"]))
                for route in routes
            ]
            for left_index, left in enumerate(positions):
                for right in positions[left_index + 1:]:
                    minimum_separation = min(minimum_separation, math.dist(left, right))
    separation_passed = minimum_separation >= float(design["minimum_uav_separation_m"])
    speed_passed = bool(route_speeds) and max(route_speeds.values()) <= float(design["maximum_uav_speed_mps"])
    if not separation_passed:
        errors.append("minimum horizontal UAV separation failed")
    if not speed_passed:
        errors.append("maximum UAV speed failed")
    if any(value <= 1 for value in unique_counts.values()):
        errors.append("each UAV must have more than one unique position")
    if any(value <= 0.0 for value in distances.values()):
        errors.append("each UAV route distance must be positive")

    try:
        with (artifact_root / CSV_REL).open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != 1024:
            errors.append("resolved CSV must contain 1024 trajectory rows")
        if list(csv_rows[0]) != CSV_FIELDS:
            errors.append("resolved CSV columns are invalid")
    except (OSError, IndexError) as exc:
        csv_rows = []
        errors.append(f"cannot validate resolved CSV: {exc}")

    validate_bu_scenario(scenario, errors)
    aerial = scenario.get("aerial_ues", [])
    execution = scenario.get("phase_execution", {})
    report = {
        "trajectory_generation_status": "succeeded" if not errors else "failed",
        "trajectory_id": canonical.get("trajectory_id"),
        "trajectory_type": canonical.get("trajectory_type"),
        "coordinate_frame": canonical.get("coordinate_frame"),
        "ground_ue_count": 50,
        "aerial_ue_count": len(aerial),
        "total_ue_count": 50 + len(aerial),
        "gpx_file_count": len(GPX_RELS),
        "sample_count_per_uav": canonical.get("sample_count"),
        "total_resolved_rows": len(csv_rows),
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_synchronized": timestamps_synchronized,
        "sample_interval_matches_A": math.isclose(float(canonical.get("sample_interval_s", -1)), inputs["sample_interval_s"]),
        "altitude_reference": canonical.get("altitude_reference"),
        "altitude_agl_m": canonical.get("altitude_agl_m"),
        "altitude_resolution_passed": altitude_passed,
        "all_points_inside_scene": all_inside,
        "scene_edge_margin_passed": edge_margin_passed,
        "minimum_uav_separation_m_observed": minimum_separation,
        "minimum_uav_separation_passed": separation_passed,
        "maximum_uav_speed_mps_observed": max(route_speeds.values()) if route_speeds else None,
        "maximum_uav_speed_passed": speed_passed,
        "unique_position_count_per_uav": unique_counts,
        "unique_position_count_passed": bool(unique_counts) and all(value > 1 for value in unique_counts.values()),
        "total_distance_per_uav_m": distances,
        "total_distance_passed": bool(distances) and all(value > 0.0 for value in distances.values()),
        "maximum_coordinate_round_trip_error_m": maximum_round_trip_error,
        "coordinate_round_trip_passed": maximum_round_trip_error <= float(design["coordinate_round_trip_tolerance_m"]),
        "no_nan_coordinates": no_nan,
        "no_duplicate_sample_ids": no_duplicate_ids,
        "no_missing_sample_ids": no_missing_ids,
        "gpx_schema_validation_passed": gpx_passed,
        "gpx_hash_manifest_created": (artifact_root / MANIFEST_REL).is_file(),
        "hash_validation_passed": hash_passed,
        "deterministic_hash_check_passed": manifest.get("deterministic_hash_check_passed") is True,
        "aerial_ue_network_role": "regular_ue" if aerial and all(node.get("network_role") == "regular_ue" for node in aerial) else "invalid",
        "aerial_ue_export_to_raw_telemetry": bool(aerial) and all(node.get("export_to_raw_telemetry") is True for node in aerial),
        "aerial_ue_include_in_ground_impact_view": any(node.get("include_in_ground_impact_view") is True for node in aerial),
        "aodt_executed": execution.get("aodt_executed"),
        "scheduler_executed": execution.get("scheduler_executed"),
        "advtabddpm_executed": execution.get("advtabddpm_executed"),
        "passed": not errors,
        "errors": errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = validate_artifact_root(args.artifact_root, args.reference_project_root)
    if args.write_report:
        write_json(args.artifact_root / VALIDATION_REL, report)
    if not report["passed"]:
        for error in report["errors"]:
            print(f"ERROR={error}")
        return 1
    print("STATUS=bu_gpx_validation_succeeded")
    print("BRANCH_ID=BU")
    print("UAV_COUNT=4")
    print("TOTAL_TRAJECTORY_ROWS=1024")
    print("TIMESTAMP_SYNC_PASSED=true")
    print("ALTITUDE_VALIDATION_PASSED=true")
    print("SCENE_BOUNDS_PASSED=true")
    print("SEPARATION_VALIDATION_PASSED=true")
    print("SPEED_VALIDATION_PASSED=true")
    print("HASH_VALIDATION_PASSED=true")
    print("AODT_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
