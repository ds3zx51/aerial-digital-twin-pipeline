#!/usr/bin/env python3
"""Build and optionally publish the deterministic AODT four-UAV trajectory.

This module is local-only. It does not execute AODT, a scheduler, a
classifier, or AdvTabDDPM.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    from .shared_geodetic import (
        CONVERTER_VERSION,
        SceneGeoreference,
        georeference_from_metadata,
        scene_enu_to_wgs84,
        wgs84_to_scene_enu,
    )
    from .shared_gpx import format_gpx_time, write_gpx
except ImportError:  # Direct script execution.
    from shared_geodetic import (
        CONVERTER_VERSION,
        SceneGeoreference,
        georeference_from_metadata,
        scene_enu_to_wgs84,
        wgs84_to_scene_enu,
    )
    from shared_gpx import format_gpx_time, write_gpx


GENERATOR_VERSION = "shared-multi-uav-gpx-1.0.0"
TRAJECTORY_ID = "tokyo_u4_boundary_60m_256"
DESIGN_REL = Path("configs/trajectories/shared/tokyo_u4_boundary_60m_256_design.json")
CANONICAL_REL = Path("configs/trajectories/shared/tokyo_u4_boundary_60m_256_enu.json")
GPX_DIR_REL = Path("configs/gpx/shared/tokyo_u4_boundary_60m_256")
MANIFEST_REL = GPX_DIR_REL / "trajectory_manifest.json"
SCENARIO_REL = Path("configs/scenarios/bu/tokyo_BU_multi_uav.json")
CSV_REL = Path("artifacts/trajectory/tokyo_u4_boundary_60m_256_resolved.csv")
VALIDATION_REL = Path("artifacts/trajectory/tokyo_u4_boundary_60m_256_validation.json")
PREVIEW_REL = Path("artifacts/trajectory/tokyo_u4_boundary_60m_256_preview.json")
GPX_RELS = [GPX_DIR_REL / f"uav_{index:02d}_boundary_60m_256.gpx" for index in range(1, 5)]
PAYLOAD_RELS = [DESIGN_REL, CANONICAL_REL, *GPX_RELS, MANIFEST_REL, SCENARIO_REL, CSV_REL, PREVIEW_REL]
PUBLISH_RELS = [*PAYLOAD_RELS, VALIDATION_REL]


CSV_FIELDS = [
    "trajectory_id", "sample_id", "timestamp", "ue_id", "ue_type",
    "trajectory_start_reference_ru_id", "latitude_deg", "longitude_deg",
    "gpx_ele_m", "altitude_reference", "altitude_agl_m", "absolute_altitude_m",
    "east_m", "north_m", "up_m", "local_x_m", "local_y_m", "local_z_m",
    "velocity_mps", "inside_scene", "nearest_ru_id",
    "distance_to_boundary_region_m", "source_gpx_file",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("start_time_utc must be a fixed UTC timestamp ending in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    return parsed.astimezone(timezone.utc)


def validate_design_config(design: dict[str, Any]) -> None:
    required = {
        "trajectory_id", "trajectory_type", "coordinate_frame", "anchor_id",
        "source_scenario_config", "geodetic_anchor_source", "ru_coordinates_source",
        "scene_bounds_source", "sample_count", "sample_interval_source",
        "start_time_utc", "altitude_reference", "altitude_agl_m", "terrain_model",
        "scene_ground_reference_source", "boundary_method", "interpolation_method",
        "maximum_uav_speed_mps", "minimum_uav_separation_m", "scene_edge_margin_m",
        "coordinate_round_trip_tolerance_m", "trajectory_start_reference_ru_ids",
        "target_offsets_m",
    }
    missing = sorted(required - set(design))
    if missing:
        raise ValueError(f"trajectory design config is missing: {', '.join(missing)}")
    expected = {
        "trajectory_id": TRAJECTORY_ID,
        "trajectory_type": "synthetic_multi_uav",
        "coordinate_frame": "ENU",
        "altitude_reference": "AGL",
        "boundary_method": "geometric_approximation",
        "interpolation_method": "smoothstep",
        "terrain_model": "reference_ru_flat_ground_approximation",
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if int(design["sample_count"]) != 256:
        raise ValueError("sample_count must be 256")
    if not math.isclose(float(design["altitude_agl_m"]), 60.0):
        raise ValueError("altitude_agl_m must be 60.0")
    for key in ("maximum_uav_speed_mps", "minimum_uav_separation_m", "scene_edge_margin_m", "coordinate_round_trip_tolerance_m"):
        if not math.isfinite(float(design[key])) or float(design[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    reference_ids = design["trajectory_start_reference_ru_ids"]
    offsets = design["target_offsets_m"]
    if not isinstance(reference_ids, list) or len(reference_ids) != 4 or len(set(reference_ids)) != 4:
        raise ValueError("exactly four distinct trajectory_start_reference_ru_ids are required")
    if not isinstance(offsets, list) or len(offsets) != 4 or any(not isinstance(item, list) or len(item) != 2 for item in offsets):
        raise ValueError("target_offsets_m must contain four XY offsets")
    if len({tuple(float(value) for value in item) for item in offsets}) != 4:
        raise ValueError("target_offsets_m must be distinct")
    _parse_utc(design["start_time_utc"])


def _scene_bounds(conversion: dict[str, Any]) -> dict[str, list[float]]:
    ground = conversion.get("ground")
    geometry = conversion.get("geometry")
    if not isinstance(ground, dict) or not isinstance(geometry, dict):
        raise ValueError("scene bounds require both ground and geometry metadata")
    try:
        ground_min = [float(value) for value in ground["min_m"]]
        ground_max = [float(value) for value in ground["max_m"]]
        geometry_min = [float(value) for value in geometry["min_m"]]
        geometry_max = [float(value) for value in geometry["max_m"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scene bounds metadata is incomplete") from exc
    if not all(len(values) == 3 for values in (ground_min, ground_max, geometry_min, geometry_max)):
        raise ValueError("scene bounds must contain XYZ triples")
    bounds = {
        "x": [ground_min[0], ground_max[0]],
        "y": [ground_min[1], ground_max[1]],
        "z": [min(ground_min[2], geometry_min[2]), max(ground_max[2], geometry_max[2])],
    }
    if any(not math.isfinite(value) for limits in bounds.values() for value in limits):
        raise ValueError("scene bounds must be finite")
    if any(limits[0] >= limits[1] for limits in bounds.values()):
        raise ValueError("scene bounds must be ordered")
    return bounds


def resolve_inputs(project_root: Path, design_path: Path | None = None) -> dict[str, Any]:
    design_path = design_path or project_root / DESIGN_REL
    design = load_json(design_path)
    validate_design_config(design)
    source_scenario = project_root / str(design["source_scenario_config"])
    conversion_path = project_root / str(design["geodetic_anchor_source"]).split("#", 1)[0]
    if not source_scenario.is_file():
        raise ValueError(f"base static AODT scenario is missing: {source_scenario}")
    if not conversion_path.is_file():
        raise ValueError(f"geodetic metadata is missing: {conversion_path}")
    scenario = load_json(source_scenario)
    conversion = load_json(conversion_path)
    meta = scenario.get("meta")
    ru_nodes = scenario.get("bs")
    ground_ues = scenario.get("ue")
    if not isinstance(meta, dict) or not isinstance(ru_nodes, list) or not isinstance(ground_ues, list):
        raise ValueError("base static AODT scenario is missing meta, bs, or ue")
    if len(ru_nodes) != 8 or len(ground_ues) != 50:
        raise ValueError("base static AODT scenario must contain 8 RU and 50 ground UE")
    if int(meta.get("time_steps", -1)) != int(design["sample_count"]):
        raise ValueError("trajectory sample_count does not match A meta.time_steps")
    try:
        sample_interval_s = float(meta["slot_ms"]) / 1000.0
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("base static AODT sample timing meta.slot_ms is missing") from exc
    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("base static AODT meta.slot_ms must be positive")
    ru_by_id: dict[int, dict[str, Any]] = {}
    for node in ru_nodes:
        try:
            ru_id = int(node["id"])
            for key in ("x", "y", "z", "height_agl_m"):
                if not math.isfinite(float(node[key])):
                    raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("RU coordinates and height_agl_m must be present and finite") from exc
        if ru_id in ru_by_id:
            raise ValueError("RU IDs must be unique")
        ru_by_id[ru_id] = node
    missing_ru = [int(value) for value in design["trajectory_start_reference_ru_ids"] if int(value) not in ru_by_id]
    if missing_ru:
        raise ValueError(f"trajectory references missing RU coordinates: {missing_ru}")
    georef = georeference_from_metadata(conversion, str(design["anchor_id"]))
    bounds = _scene_bounds(conversion)
    return {
        "design": design,
        "design_path": design_path,
        "scenario": scenario,
        "scenario_path": source_scenario,
        "conversion": conversion,
        "conversion_path": conversion_path,
        "sample_interval_s": sample_interval_s,
        "ru_by_id": ru_by_id,
        "georef": georef,
        "scene_bounds": bounds,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _inside(position: tuple[float, float, float], bounds: dict[str, list[float]]) -> bool:
    return all(bounds[axis][0] <= value <= bounds[axis][1] for axis, value in zip(("x", "y", "z"), position))


def _nearest_ru_id(east_m: float, north_m: float, ru_by_id: dict[int, dict[str, Any]]) -> int:
    return min(ru_by_id, key=lambda ru_id: (math.hypot(east_m - float(ru_by_id[ru_id]["x"]), north_m - float(ru_by_id[ru_id]["y"])), ru_id))


def _resolved_scenario(source: dict[str, Any], canonical: dict[str, Any], source_sha: str) -> dict[str, Any]:
    resolved = copy.deepcopy(source)
    meta = resolved["meta"]
    meta.update({
        "branch_id": "BU",
        "backend": "aodt",
        "experiment_mode": "multi_uav_uplink_interference",
        "phase": "BU trajectory input only",
        "num_ground_ue": 50,
        "num_aerial_ue": 4,
        "num_ue": 54,
        "aerial_channel_implemented": False,
        "aerial_scheduler_implemented": False,
        "aerial_interference_implemented": False,
    })
    aerial_ues = []
    for index, route in enumerate(canonical["routes"], start=1):
        ue_id = f"uav_{index:02d}"
        aerial_ues.append({
            "ue_id": ue_id,
            "ue_type": "aerial",
            "network_role": "regular_ue",
            "analysis_role": "uplink_interference_source",
            "mobility_type": "gpx",
            "trajectory_start_reference_ru_id": route["trajectory_start_reference_ru_id"],
            "gpx_file": GPX_RELS[index - 1].as_posix(),
            "export_to_raw_telemetry": True,
            "include_in_ground_impact_view": False,
            "initial_position_enu_m": route["start_enu_m"],
        })
    resolved.update({
        "branch_id": "BU",
        "backend": "aodt",
        "experiment_mode": "multi_uav_uplink_interference",
        "source_ground_scenario": "configs/scenario_layout_tokyo_docomo.json",
        "source_ground_scenario_sha256": source_sha,
        "trajectory_id": TRAJECTORY_ID,
        "trajectory_source": CANONICAL_REL.as_posix(),
        "trajectory_manifest": MANIFEST_REL.as_posix(),
        "ue_counts": {"ground": 50, "aerial": 4, "total": 54},
        "aerial_ues": aerial_ues,
        "phase_execution": {
            "aodt_executed": False,
            "scheduler_executed": False,
            "advtabddpm_executed": False,
        },
    })
    resolved["experiment_definition"]["bu_phase"] = "trajectory_input_only"
    resolved["experiment_definition"]["aerial_ue_model"] = "regular_network_ue_not_external_jammer"
    return resolved


def build_payload(project_root: Path, output_root: Path, design_path: Path | None = None) -> dict[str, Any]:
    inputs = resolve_inputs(project_root, design_path)
    design = inputs["design"]
    ru_by_id = inputs["ru_by_id"]
    georef: SceneGeoreference = inputs["georef"]
    bounds = inputs["scene_bounds"]
    margin = float(design["scene_edge_margin_m"])
    inset = {axis: [bounds[axis][0] + margin, bounds[axis][1] - margin] for axis in ("x", "y")}
    if any(limits[0] >= limits[1] for limits in inset.values()):
        raise ValueError("scene_edge_margin_m leaves no valid horizontal region")
    reference_ids = [int(value) for value in design["trajectory_start_reference_ru_ids"]]
    references = [ru_by_id[value] for value in reference_ids]
    centroid_x = _clamp(sum(float(node["x"]) for node in references) / 4.0, *inset["x"])
    centroid_y = _clamp(sum(float(node["y"]) for node in references) / 4.0, *inset["y"])
    sample_count = int(design["sample_count"])
    sample_interval_s = float(inputs["sample_interval_s"])
    duration_s = (sample_count - 1) * sample_interval_s
    start_time = _parse_utc(str(design["start_time_utc"]))
    altitude_agl_m = float(design["altitude_agl_m"])
    routes: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    max_round_trip_error_m = 0.0

    for route_index, (reference_id, offset) in enumerate(zip(reference_ids, design["target_offsets_m"]), start=1):
        ru = ru_by_id[reference_id]
        start_x = _clamp(float(ru["x"]), *inset["x"])
        start_y = _clamp(float(ru["y"]), *inset["y"])
        target_x = _clamp(centroid_x + float(offset[0]), *inset["x"])
        target_y = _clamp(centroid_y + float(offset[1]), *inset["y"])
        scene_ground_offset_m = float(ru["z"]) - float(ru["height_agl_m"])
        local_z = scene_ground_offset_m + altitude_agl_m
        if not _inside((start_x, start_y, local_z), bounds) or not _inside((target_x, target_y, local_z), bounds):
            raise ValueError(f"route {route_index} altitude or endpoint is outside scene bounds")
        distance_m = math.hypot(target_x - start_x, target_y - start_y)
        speed_mps = distance_m / duration_s
        if distance_m <= 0.0:
            raise ValueError(f"route {route_index} has zero distance")
        if speed_mps > float(design["maximum_uav_speed_mps"]):
            raise ValueError(f"route {route_index} exceeds maximum_uav_speed_mps")
        ue_id = f"uav_{route_index:02d}"
        source_gpx = GPX_RELS[route_index - 1].as_posix()
        points: list[dict[str, Any]] = []
        for sample_id in range(sample_count):
            fraction = sample_id / (sample_count - 1)
            smooth = 3.0 * fraction**2 - 2.0 * fraction**3
            east = start_x + (target_x - start_x) * smooth
            north = start_y + (target_y - start_y) * smooth
            timestamp = format_gpx_time(start_time + timedelta(seconds=sample_id * sample_interval_s))
            latitude, longitude, absolute_altitude = scene_enu_to_wgs84(east, north, local_z, georef)
            round_east, round_north, round_up = wgs84_to_scene_enu(latitude, longitude, absolute_altitude, georef)
            round_trip_error = math.dist((east, north, local_z), (round_east, round_north, round_up))
            max_round_trip_error_m = max(max_round_trip_error_m, round_trip_error)
            point = {
                "sample_id": sample_id,
                "timestamp": timestamp,
                "east_m": round(east, 9),
                "north_m": round(north, 9),
                "up_m": round(local_z, 9),
                "local_x_m": round(east, 9),
                "local_y_m": round(north, 9),
                "local_z_m": round(local_z, 9),
                "scene_ground_offset_m": round(scene_ground_offset_m, 9),
                "altitude_agl_m": altitude_agl_m,
                "absolute_altitude_m": round(absolute_altitude, 9),
                "latitude_deg": round(latitude, 12),
                "longitude_deg": round(longitude, 12),
                "gpx_ele_m": round(absolute_altitude, 9),
            }
            points.append(point)
            resolved_rows.append({
                "trajectory_id": TRAJECTORY_ID,
                "sample_id": sample_id,
                "timestamp": timestamp,
                "ue_id": ue_id,
                "ue_type": "aerial",
                "trajectory_start_reference_ru_id": reference_id,
                "latitude_deg": f"{latitude:.12f}",
                "longitude_deg": f"{longitude:.12f}",
                "gpx_ele_m": f"{absolute_altitude:.9f}",
                "altitude_reference": "AGL",
                "altitude_agl_m": f"{altitude_agl_m:.3f}",
                "absolute_altitude_m": f"{absolute_altitude:.9f}",
                "east_m": f"{east:.9f}",
                "north_m": f"{north:.9f}",
                "up_m": f"{local_z:.9f}",
                "local_x_m": f"{east:.9f}",
                "local_y_m": f"{north:.9f}",
                "local_z_m": f"{local_z:.9f}",
                "velocity_mps": f"{speed_mps:.9f}",
                "inside_scene": "true",
                "nearest_ru_id": _nearest_ru_id(east, north, ru_by_id),
                "distance_to_boundary_region_m": f"{math.hypot(target_x - east, target_y - north):.9f}",
                "source_gpx_file": source_gpx,
            })
        routes.append({
            "ue_id": ue_id,
            "trajectory_start_reference_ru_id": reference_id,
            "start_reference_ru_enu_m": [float(ru["x"]), float(ru["y"]), float(ru["z"])],
            "start_enu_m": [round(start_x, 9), round(start_y, 9), round(local_z, 9)],
            "target_enu_m": [round(target_x, 9), round(target_y, 9), round(local_z, 9)],
            "scene_ground_offset_m": round(scene_ground_offset_m, 9),
            "route_distance_m": round(distance_m, 9),
            "duration_s": round(duration_s, 9),
            "average_speed_mps": round(speed_mps, 9),
            "serving_ru_id": None,
            "serving_ru_assignment_status": "deferred_to_BU_channel_association",
            "points": points,
        })

    if max_round_trip_error_m > float(design["coordinate_round_trip_tolerance_m"]):
        raise ValueError(f"coordinate round-trip error {max_round_trip_error_m} m exceeds tolerance")
    canonical = {
        "trajectory_id": TRAJECTORY_ID,
        "trajectory_type": "synthetic_multi_uav",
        "coordinate_frame": "ENU",
        "coordinate_convention": {"x": "East", "y": "North", "z": "Up"},
        "anchor_id": design["anchor_id"],
        "geodetic_anchor": {
            "latitude_deg": georef.latitude_deg,
            "longitude_deg": georef.longitude_deg,
            "altitude_m": georef.altitude_m,
            "vertical_datum": "ellipsoid",
            "crs": f"EPSG:{32600 + georef.utm_zone if georef.northern_hemisphere else 32700 + georef.utm_zone}",
            "anchor_easting_m": georef.anchor_easting_m,
            "anchor_northing_m": georef.anchor_northing_m,
        },
        "scene_bounds_m": bounds,
        "scene_edge_margin_m": margin,
        "boundary_region_centroid_enu_m": [round(centroid_x, 9), round(centroid_y, 9)],
        "sample_count": sample_count,
        "sample_interval_s": sample_interval_s,
        "sample_interval_source": design["sample_interval_source"],
        "start_time_utc": design["start_time_utc"],
        "altitude_reference": "AGL",
        "altitude_agl_m": altitude_agl_m,
        "terrain_model": design["terrain_model"],
        "boundary_method": design["boundary_method"],
        "interpolation_method": design["interpolation_method"],
        "minimum_uav_separation_m": float(design["minimum_uav_separation_m"]),
        "maximum_uav_speed_mps": float(design["maximum_uav_speed_mps"]),
        "coordinate_round_trip_tolerance_m": float(design["coordinate_round_trip_tolerance_m"]),
        "maximum_coordinate_round_trip_error_m": max_round_trip_error_m,
        "routes": routes,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / DESIGN_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inputs["design_path"], output_root / DESIGN_REL)
    write_json(output_root / CANONICAL_REL, canonical)
    for route, path in zip(routes, GPX_RELS):
        write_gpx(output_root / path, TRAJECTORY_ID, route["ue_id"], route["points"])
    (output_root / CSV_REL).parent.mkdir(parents=True, exist_ok=True)
    with (output_root / CSV_REL).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(resolved_rows)
    preview = {
        "trajectory_id": TRAJECTORY_ID,
        "boundary_region_centroid_enu_m": canonical["boundary_region_centroid_enu_m"],
        "scene_bounds_m": bounds,
        "routes": [{
            key: route[key] for key in (
                "ue_id", "trajectory_start_reference_ru_id", "start_enu_m", "target_enu_m",
                "scene_ground_offset_m", "route_distance_m", "duration_s", "average_speed_mps",
            )
        } for route in routes],
    }
    write_json(output_root / PREVIEW_REL, preview)
    source_sha = sha256(inputs["scenario_path"])
    scenario = _resolved_scenario(inputs["scenario"], canonical, source_sha)
    write_json(output_root / SCENARIO_REL, scenario)
    manifest = {
        "trajectory_id": TRAJECTORY_ID,
        "generation_status": "succeeded",
        "generator_version": GENERATOR_VERSION,
        "coordinate_converter_version": CONVERTER_VERSION,
        "anchor_id": design["anchor_id"],
        "sample_count": sample_count,
        "sample_interval_s": sample_interval_s,
        "altitude_reference": "AGL",
        "design_config_path": DESIGN_REL.as_posix(),
        "design_config_sha256": sha256(output_root / DESIGN_REL),
        "canonical_enu_path": CANONICAL_REL.as_posix(),
        "canonical_enu_sha256": sha256(output_root / CANONICAL_REL),
        "gpx_files": [{"path": path.as_posix(), "sha256": sha256(output_root / path)} for path in GPX_RELS],
        "resolved_csv_path": CSV_REL.as_posix(),
        "resolved_csv_sha256": sha256(output_root / CSV_REL),
        "preview_path": PREVIEW_REL.as_posix(),
        "preview_sha256": sha256(output_root / PREVIEW_REL),
        "bu_scenario_path": SCENARIO_REL.as_posix(),
        "bu_scenario_sha256": sha256(output_root / SCENARIO_REL),
        "source_scenario_path": str(design["source_scenario_config"]),
        "source_scenario_sha256": source_sha,
        "geodetic_metadata_path": str(design["geodetic_anchor_source"]).split("#", 1)[0],
        "geodetic_metadata_sha256": sha256(inputs["conversion_path"]),
        "deterministic_hash_check_passed": True,
    }
    write_json(output_root / MANIFEST_REL, manifest)
    return {"canonical": canonical, "manifest": manifest, "resolved_rows": resolved_rows, "scenario": scenario}


def compare_payloads(left_root: Path, right_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in PAYLOAD_RELS:
        left = left_root / relative
        right = right_root / relative
        if not left.is_file() or not right.is_file():
            raise ValueError(f"deterministic build is missing {relative.as_posix()}")
        left_hash = sha256(left)
        right_hash = sha256(right)
        if left_hash != right_hash:
            raise ValueError(f"deterministic build mismatch: {relative.as_posix()}")
        hashes[relative.as_posix()] = left_hash
    return hashes


def publish_payload(staging_root: Path, project_root: Path) -> dict[str, list[str]]:
    created: list[str] = []
    matched: list[str] = []
    for relative in PUBLISH_RELS:
        source = staging_root / relative
        destination = project_root / relative
        if not source.is_file():
            raise ValueError(f"staging payload is missing: {relative.as_posix()}")
        if destination.exists():
            if not destination.is_file() or sha256(source) != sha256(destination):
                raise FileExistsError(f"published artifact differs; refusing overwrite: {destination}")
            matched.append(relative.as_posix())
    for relative in PUBLISH_RELS:
        source = staging_root / relative
        destination = project_root / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"atomic publish temporary path already exists: {temporary}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        created.append(relative.as_posix())
    return {"created": created, "matched": matched}


def run_build(project_root: Path, staging_root: Path, publish: bool) -> dict[str, Any]:
    if staging_root.exists() and any(staging_root.iterdir()):
        raise FileExistsError(f"staging directory must be empty: {staging_root}")
    staging_root.mkdir(parents=True, exist_ok=True)
    build_payload(project_root, staging_root)
    with tempfile.TemporaryDirectory(prefix="aodt_bu_determinism_") as second_name:
        second_root = Path(second_name)
        build_payload(project_root, second_root)
        deterministic_hashes = compare_payloads(staging_root, second_root)
    try:
        from .validate_shared_multi_uav_gpx import validate_artifact_root
    except ImportError:
        from validate_shared_multi_uav_gpx import validate_artifact_root
    validation = validate_artifact_root(staging_root, project_root)
    if not validation["passed"]:
        raise ValueError("BU trajectory validation failed: " + "; ".join(validation["errors"]))
    write_json(staging_root / VALIDATION_REL, validation)
    publication = publish_payload(staging_root, project_root) if publish else {"created": [], "matched": []}
    return {
        "trajectory_id": TRAJECTORY_ID,
        "staging_root": str(staging_root),
        "deterministic_hashes": deterministic_hashes,
        "validation": validation,
        "publication": publication,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.staging_root:
        result = run_build(project_root, args.staging_root.resolve(), args.publish)
    else:
        with tempfile.TemporaryDirectory(prefix="aodt_bu_trajectory_") as staging_name:
            result = run_build(project_root, Path(staging_name), args.publish)
    gates = result["validation"]
    print("STATUS=bu_trajectory_generation_succeeded")
    print("BRANCH_ID=BU")
    print(f"TRAJECTORY_ID={TRAJECTORY_ID}")
    print("GROUND_UE_COUNT=50")
    print("AERIAL_UE_COUNT=4")
    print("TOTAL_UE_COUNT=54")
    print("GPX_FILE_COUNT=4")
    print("SAMPLES_PER_UAV=256")
    print("TOTAL_TRAJECTORY_ROWS=1024")
    print(f"GPX_VALIDATION_PASSED={str(gates['passed']).lower()}")
    print("AODT_EXECUTED=false")
    print("SCHEDULER_EXECUTED=false")
    print("ADVTABDDPM_EXECUTED=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR={exc}")
        raise SystemExit(1)
