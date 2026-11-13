"""Offline-capable AODT GPX/UAV stages 00BU through 05BU.

The BU branch extends the passing fixed-UE B branch with four deterministic GPX
tracks. It preserves the 50 terrestrial UEs and
adds four regular network-attached aerial UEs with AODT IDs 51 through 54.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


BU_STAGES = (
    "00BU",
    "01BU",
    "02BU",
    "03BU",
    "04BU",
    "05BU",
    "06BU",
    "07BU",
)
BU_STAGE_DIRS = {
    "00BU": "00BU_inspection",
    "01BU": "01BU_gpx_inputs",
    "02BU": "02BU_scene_materials",
    "03BU": "03BU_topology",
    "04BU": "04BU_config",
    "05BU": "05BU_preflight",
    "06BU": "06BU_execution",
    "07BU": "07BU_raw_collection",
}
STATIC_TO_BU = {
    "00B": "00BU",
    "01B": "01BU",
    "02B": "02BU",
    "03B": "03BU",
    "04B": "04BU",
    "05B": "05BU",
    "06B": "06BU",
    "07B": "07BU",
}
BASELINE_EVIDENCE_FILES = (
    "aodt_capability_inventory.json",
    "static_topology_inventory.json",
)
ASSET_KEYS = (
    "du",
    "ru",
    "ue",
    "materials",
    "scatterers",
    "vegetation_materials",
    "vegetation_assets",
    "scenario",
    "panels",
)
GPX_NAMESPACE = "http://www.topografix.com/GPX/1/1"
EXPECTED_RU_COUNT = 8
EXPECTED_GROUND_UE_COUNT = 50
EXPECTED_UAV_COUNT = 4
EXPECTED_UE_COUNT = EXPECTED_GROUND_UE_COUNT + EXPECTED_UAV_COUNT
EXPECTED_SAMPLE_COUNT = 256
EXPECTED_SAMPLE_INTERVAL_S = 0.25
# AODT duration/interval mode includes both t=0 and t=duration.  The GPX
# tracks contain 256 timestamps from 0.00 through 63.75 seconds, so the
# timeline endpoint must be (sample_count - 1) * interval.  A 64.00-second
# endpoint would create 257 samples for waypoint/static UEs.
EXPECTED_DURATION_S = (EXPECTED_SAMPLE_COUNT - 1) * EXPECTED_SAMPLE_INTERVAL_S
AODT_MAX_RAY_INTERACTIONS = 5


class PipelineError(RuntimeError):
    """Fail-closed BU pipeline error."""


@dataclass(frozen=True)
class UAVPipelineContext:
    repo_root: Path
    run_dir: Path
    scenario_path: Path
    scene_archive: Path
    aodt_client_root: Path
    asset_config: Path
    worker_scene_url: str
    worker_gpx_prefix: str
    endpoint: str
    offline: bool
    ack_worker_assets_resolved: bool
    ack_worker_gpx_resolved: bool
    slots_spec: str
    full_antenna_pair: bool
    position_tolerance_m: float
    command: str

    def stage_dir(self, stage: str) -> Path:
        normalized = STATIC_TO_BU.get(stage, stage)
        return self.run_dir / BU_STAGE_DIRS[normalized]

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, Path):
                result[key] = path_text(value)
        return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def markdown_write(path: Path, title: str, lines: Iterable[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", *lines, ""]), encoding="utf-8")


def csv_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def create_stage_dir(ctx: UAVPipelineContext, stage: str) -> Path:
    stage_dir = ctx.stage_dir(stage)
    try:
        stage_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise PipelineError(
            f"Refusing to overwrite existing stage directory: {stage_dir}"
        ) from exc
    return stage_dir


def create_unique_run_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = results_root / f"tokyo_aodt_uav_{stamp}_{suffix:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise PipelineError("Unable to allocate a unique BU run directory.")


def update_run_manifest(
    ctx: UAVPipelineContext,
    stage: str | None = None,
    status: dict[str, Any] | None = None,
    overall_status: str | None = None,
) -> None:
    path = ctx.run_dir / "run_manifest.json"
    if path.exists():
        manifest = json_load(path)
    else:
        manifest = {
            "schema_version": "aodt-uav-gpx-run-v1",
            "created_at_utc": utc_now(),
            "run_directory": path_text(ctx.run_dir),
            "scope": {
                "ru": EXPECTED_RU_COUNT,
                "ue": EXPECTED_UE_COUNT,
                "ground_ue": EXPECTED_GROUND_UE_COUNT,
                "uav": EXPECTED_UAV_COUNT,
                "ue_mobility": "50 fixed terrestrial UE + 4 GPX aerial UE",
                "gpx": EXPECTED_UAV_COUNT,
                "temporal_mode": "duration_interval",
                "time_steps": EXPECTED_SAMPLE_COUNT,
                "interval_s": EXPECTED_SAMPLE_INTERVAL_S,
                "transport": "official AODT DigitalTwinClient direct gRPC",
            },
            "inputs": ctx.public_dict(),
            "stages": {},
            "overall_status": "in_progress",
        }
    if stage and status is not None:
        manifest["stages"][stage] = status
    if overall_status:
        manifest["overall_status"] = overall_status
        manifest["finished_at_utc"] = utc_now()
    manifest["updated_at_utc"] = utc_now()
    json_dump(path, manifest)


def _latest_passing_static_08b(repo_root: Path) -> Path:
    root = repo_root / "results" / "aodt_static"
    candidates: list[tuple[float, Path]] = []
    if root.is_dir():
        for run_dir in root.iterdir():
            manifest_path = run_dir / "run_manifest.json"
            if not run_dir.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = json_load(manifest_path)
            except (OSError, ValueError):
                continue
            if manifest.get("stages", {}).get("08B", {}).get("status") == "pass":
                candidates.append((manifest_path.stat().st_mtime, run_dir))
    if not candidates:
        raise PipelineError("No passing fixed-UE 08B baseline was found.")
    return max(candidates, key=lambda item: item[0])[1]


def _trajectory_paths(repo_root: Path, scenario_path: Path) -> dict[str, Path]:
    scenario = json_load(scenario_path)
    manifest_rel = scenario.get("trajectory_manifest")
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise PipelineError("BU scenario has no trajectory_manifest.")
    manifest_path = repo_root / manifest_rel
    manifest = json_load(manifest_path)
    result = {
        "scenario": scenario_path,
        "manifest": manifest_path,
        "canonical": repo_root / manifest["canonical_enu_path"],
        "design": repo_root / manifest["design_config_path"],
        "source_scenario": repo_root / manifest["source_scenario_path"],
        "geodetic_metadata": repo_root / manifest["geodetic_metadata_path"],
    }
    for index, item in enumerate(manifest.get("gpx_files", []), start=1):
        result[f"gpx_{index:02d}"] = repo_root / item["path"]
    return result


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PipelineError(f"Invalid GPX timestamp: {value}") from exc


def read_gpx(path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise PipelineError(f"Invalid GPX XML: {path}") from exc
    q = lambda name: f"{{{GPX_NAMESPACE}}}{name}"
    if root.tag != q("gpx") or root.attrib.get("version") != "1.1":
        raise PipelineError(f"GPX must use GPX 1.1 namespace: {path}")
    points = root.findall(f"{q('trk')}/{q('trkseg')}/{q('trkpt')}")
    parsed: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        ele = point.find(q("ele"))
        time_node = point.find(q("time"))
        if ele is None or time_node is None or not time_node.text:
            raise PipelineError(f"GPX point {index} lacks elevation or timestamp: {path}")
        parsed.append(
            {
                "latitude_deg": float(point.attrib["lat"]),
                "longitude_deg": float(point.attrib["lon"]),
                "elevation_m": float(ele.text),
                "timestamp": time_node.text,
                "datetime": _parse_timestamp(time_node.text),
            }
        )
    if len(parsed) != EXPECTED_SAMPLE_COUNT:
        raise PipelineError(
            f"{path.name} has {len(parsed)} points; expected {EXPECTED_SAMPLE_COUNT}."
        )
    return parsed


def worker_gpx_paths(ctx: UAVPipelineContext) -> list[str]:
    prefix = ctx.worker_gpx_prefix.strip().strip("/").replace("\\", "/")
    if not prefix:
        raise PipelineError("worker_gpx_prefix must not be empty.")
    return [
        f"{prefix}/uav_{index:02d}_boundary_60m_256.gpx"
        for index in range(1, EXPECTED_UAV_COUNT + 1)
    ]


def validate_shared_gpx(ctx: UAVPipelineContext) -> dict[str, Any]:
    paths = _trajectory_paths(ctx.repo_root, ctx.scenario_path)
    missing = [path_text(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PipelineError(f"Missing BU trajectory inputs: {missing}")
    scenario = json_load(paths["scenario"])
    manifest = json_load(paths["manifest"])
    canonical = json_load(paths["canonical"])
    errors: list[str] = []

    if scenario.get("branch_id") != "BU" or scenario.get("backend") != "aodt":
        errors.append("Scenario must be the AODT BU branch.")
    if len(scenario.get("bs", [])) != EXPECTED_RU_COUNT:
        errors.append("BU scenario must preserve 8 RUs.")
    if len(scenario.get("ue", [])) != EXPECTED_GROUND_UE_COUNT:
        errors.append("BU scenario must preserve 50 ground UEs.")
    if len(scenario.get("aerial_ues", [])) != EXPECTED_UAV_COUNT:
        errors.append("BU scenario must contain four aerial UEs.")
    if manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT:
        errors.append("Trajectory manifest sample_count must be 256.")
    if float(manifest.get("sample_interval_s", -1)) != EXPECTED_SAMPLE_INTERVAL_S:
        errors.append("Trajectory manifest sample_interval_s must be 0.25.")
    if manifest.get("trajectory_id") != "tokyo_u4_boundary_60m_256":
        errors.append("Unexpected trajectory_id.")
    if canonical.get("trajectory_id") != manifest.get("trajectory_id"):
        errors.append("Canonical trajectory ID differs from manifest.")
    routes = canonical.get("routes", [])
    if len(routes) != EXPECTED_UAV_COUNT:
        errors.append("Canonical trajectory must contain four routes.")

    hash_bindings = [
        ("design", paths["design"], manifest.get("design_config_sha256")),
        ("canonical", paths["canonical"], manifest.get("canonical_enu_sha256")),
        (
            "source_scenario",
            paths["source_scenario"],
            manifest.get("source_scenario_sha256"),
        ),
        (
            "geodetic_metadata",
            paths["geodetic_metadata"],
            manifest.get("geodetic_metadata_sha256"),
        ),
        ("bu_scenario", paths["scenario"], manifest.get("bu_scenario_sha256")),
    ]
    parsed_tracks: list[list[dict[str, Any]]] = []
    gpx_inventory: list[dict[str, Any]] = []
    for index, item in enumerate(manifest.get("gpx_files", []), start=1):
        if index > EXPECTED_UAV_COUNT:
            errors.append("Trajectory manifest contains more than four GPX files.")
            break
        local_path = paths[f"gpx_{index:02d}"]
        digest = sha256_file(local_path)
        if digest != item.get("sha256"):
            errors.append(f"GPX {index} SHA-256 differs from the trajectory manifest.")
        parsed = read_gpx(local_path)
        parsed_tracks.append(parsed)
        gpx_inventory.append(
            {
                "route_index": index - 1,
                "ue_id": f"uav_{index:02d}",
                "local_path": path_text(local_path),
                "worker_path": worker_gpx_paths(ctx)[index - 1],
                "sha256": digest,
                "point_count": len(parsed),
                "first_timestamp": parsed[0]["timestamp"],
                "last_timestamp": parsed[-1]["timestamp"],
            }
        )
    if len(parsed_tracks) != EXPECTED_UAV_COUNT:
        errors.append("Trajectory manifest must bind exactly four GPX files.")
    for label, path, declared in hash_bindings:
        actual = sha256_file(path)
        if actual != declared:
            errors.append(f"{label} SHA-256 differs from the trajectory manifest.")

    max_coordinate_error = 0.0
    reference_timestamps: list[str] | None = None
    for route_index, (track, route) in enumerate(zip(parsed_tracks, routes)):
        points = route.get("points", [])
        if len(points) != EXPECTED_SAMPLE_COUNT:
            errors.append(f"Canonical route {route_index} does not have 256 points.")
            continue
        timestamps = [point["timestamp"] for point in track]
        if reference_timestamps is None:
            reference_timestamps = timestamps
        elif timestamps != reference_timestamps:
            errors.append("GPX tracks do not share an identical timestamp vector.")
        intervals = [
            (track[index]["datetime"] - track[index - 1]["datetime"]).total_seconds()
            for index in range(1, len(track))
        ]
        if any(abs(value - EXPECTED_SAMPLE_INTERVAL_S) > 1e-9 for value in intervals):
            errors.append(f"GPX route {route_index} has a non-0.25-second interval.")
        for sample_id, (actual, expected) in enumerate(zip(track, points)):
            differences = (
                abs(actual["latitude_deg"] - float(expected["latitude_deg"])),
                abs(actual["longitude_deg"] - float(expected["longitude_deg"])),
                abs(actual["elevation_m"] - float(expected["gpx_ele_m"])),
            )
            max_coordinate_error = max(max_coordinate_error, *differences)
            # GPX stores latitude/longitude to ten decimal places while the
            # canonical JSON retains twelve, so allow only the declared text
            # serialization loss (well below a millimetre on the ground).
            if differences[0] > 1e-10 or differences[1] > 1e-10 or differences[2] > 1e-5:
                errors.append(
                    f"GPX route {route_index} differs from canonical sample {sample_id}."
                )
                break

    return {
        "schema_version": "aodt-uav-shared-gpx-validation-v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "trajectory_id": manifest.get("trajectory_id"),
        "manifest_path": path_text(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "sample_interval_s": EXPECTED_SAMPLE_INTERVAL_S,
        "duration_s": EXPECTED_DURATION_S,
        "gpx_file_count": len(parsed_tracks),
        "gpx_inventory": gpx_inventory,
        "maximum_gpx_vs_canonical_coordinate_error": max_coordinate_error,
        "hash_bindings": [
            {
                "name": label,
                "path": path_text(path),
                "declared_sha256": declared,
                "actual_sha256": sha256_file(path),
            }
            for label, path, declared in hash_bindings
        ],
        "trajectory_hashes_verified": not errors,
    }


def stage_00bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "00BU")
    source_dir = ctx.repo_root / "evidence" / "baseline"
    missing = [name for name in BASELINE_EVIDENCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise PipelineError(f"00B baseline evidence is incomplete: {missing}")
    copied = []
    for name in BASELINE_EVIDENCE_FILES:
        source = source_dir / name
        target = stage_dir / name
        shutil.copy2(source, target)
        copied.append({"name": name, "sha256": sha256_file(target)})
    baseline = _latest_passing_static_08b(ctx.repo_root)
    baseline_manifest = json_load(baseline / "run_manifest.json")
    validation = validate_shared_gpx(ctx)
    if validation["status"] != "pass":
        raise PipelineError("00BU failed to lock the deterministic GPX inputs.")
    lineage = {
        "schema_version": "aodt-uav-baseline-lineage-v1",
        "generated_at_utc": utc_now(),
        "fixed_ue_baseline_run": path_text(baseline),
        "fixed_ue_baseline_manifest_sha256": sha256_file(
            baseline / "run_manifest.json"
        ),
        "fixed_ue_08b_status": baseline_manifest["stages"]["08B"]["status"],
        "trajectory_id": validation["trajectory_id"],
        "trajectory_manifest_sha256": validation["manifest_sha256"],
        "trajectory_hashes_verified": True,
        "extension": "50 fixed ground UE + 4 regular GPX aerial UE",
    }
    json_dump(stage_dir / "baseline_lineage.json", lineage)
    status = {
        "stage": "00BU",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "fixed_08b_baseline": path_text(baseline),
        "trajectory_id": validation["trajectory_id"],
        "gpx_file_count": validation["gpx_file_count"],
        "copied_inspection_files": copied,
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def stage_01bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "01BU")
    validation = validate_shared_gpx(ctx)
    json_dump(stage_dir / "shared_gpx_validation.json", validation)
    if validation["status"] != "pass":
        raise PipelineError("01BU deterministic GPX validation failed.")
    scenario = json_load(ctx.scenario_path)
    topology = {
        "schema_version": "aodt-uav-topology-input-v1",
        "generated_at_utc": utc_now(),
        "status": "pass",
        "counts": {
            "ru": len(scenario["bs"]),
            "ground_ue": len(scenario["ue"]),
            "aerial_ue": len(scenario["aerial_ues"]),
            "total_ue": len(scenario["ue"]) + len(scenario["aerial_ues"]),
        },
        "identity": {
            "ground_scenario_ids": [0, 49],
            "ground_aodt_ids": [1, 50],
            "uav_scenario_ids": [50, 53],
            "uav_aodt_ids": [51, 54],
            "tensor_indices": [0, 53],
        },
        "trajectory_id": validation["trajectory_id"],
        "trajectory_hashes_verified": validation["trajectory_hashes_verified"],
    }
    json_dump(stage_dir / "topology_input_validation.json", topology)
    status = {
        "stage": "01BU",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "trajectory_id": validation["trajectory_id"],
        "gpx_manifest_sha256": validation["manifest_sha256"],
        "gpx_file_count": validation["gpx_file_count"],
        "sample_count_per_uav": validation["sample_count"],
        "sample_interval_s": validation["sample_interval_s"],
        "trajectory_hashes_verified": True,
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def stage_02bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    try:
        from static_core import stage_02b
    except ImportError as exc:
        raise PipelineError(f"Unable to import common B scene stage: {exc}") from exc
    status = stage_02b(ctx)
    status["stage"] = "02BU"
    status["shared_with_fixed_b"] = True
    status["mobility_effect"] = "none; scene and material inputs are unchanged"
    json_dump(ctx.stage_dir("02BU") / "stage_status.json", status)
    return status


def _mapping_rows(ctx: UAVPipelineContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario = json_load(ctx.scenario_path)
    trajectory = json_load(
        _trajectory_paths(ctx.repo_root, ctx.scenario_path)["canonical"]
    )
    ru_rows = []
    for tensor_index, node in enumerate(scenario["bs"]):
        ru_rows.append(
            {
                "node_type": "RU",
                "scenario_id": int(node["id"]),
                "display_id": f"RU{int(node['id']):02d}",
                "aodt_id": int(node["id"]) + 1,
                "tensor_api_index": tensor_index,
                "mobility": "static",
                "trajectory_id": "",
                "gpx_local_path": "",
                "gpx_worker_path": "",
                "x_m": float(node["x"]),
                "y_m": float(node["y"]),
                "z_m": float(node["z"]),
            }
        )
    ue_rows = []
    for tensor_index, node in enumerate(scenario["ue"]):
        ue_rows.append(
            {
                "node_type": "UE",
                "scenario_id": int(node["id"]),
                "display_id": f"UE{int(node['id']):02d}",
                "aodt_id": int(node["id"]) + 1,
                "tensor_api_index": tensor_index,
                "mobility": "static_ground",
                "trajectory_id": "",
                "gpx_local_path": "",
                "gpx_worker_path": "",
                "x_m": float(node["x"]),
                "y_m": float(node["y"]),
                "z_m": float(node["z"]),
            }
        )
    worker_paths = worker_gpx_paths(ctx)
    for route_index, (aerial, route) in enumerate(
        zip(scenario["aerial_ues"], trajectory["routes"])
    ):
        scenario_id = EXPECTED_GROUND_UE_COUNT + route_index
        initial = route["points"][0]
        ue_rows.append(
            {
                "node_type": "UE",
                "scenario_id": scenario_id,
                "display_id": aerial["ue_id"],
                "aodt_id": scenario_id + 1,
                "tensor_api_index": scenario_id,
                "mobility": "gpx_aerial",
                "trajectory_id": trajectory["trajectory_id"],
                "gpx_local_path": aerial["gpx_file"],
                "gpx_worker_path": worker_paths[route_index],
                "x_m": float(initial["local_x_m"]),
                "y_m": float(initial["local_y_m"]),
                "z_m": float(initial["local_z_m"]),
            }
        )
    return ru_rows, ue_rows


def stage_03bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "03BU")
    ru_rows, ue_rows = _mapping_rows(ctx)
    fields = list(ru_rows[0])
    csv_write(stage_dir / "ru_mapping.csv", fields, ru_rows)
    csv_write(stage_dir / "ue_mapping.csv", fields, ue_rows)
    mapping = {
        "schema_version": "aodt-uav-node-mapping-v1",
        "generated_at_utc": utc_now(),
        "status": "pass",
        "mapping_rule": "aodt_id = scenario_id + 1",
        "tensor_rule": "tensor index = scenario_id",
        "ru": ru_rows,
        "ue": ue_rows,
        "validation": {
            "ru_count": len(ru_rows),
            "ground_ue_count": sum(row["mobility"] == "static_ground" for row in ue_rows),
            "uav_count": sum(row["mobility"] == "gpx_aerial" for row in ue_rows),
            "total_ue_count": len(ue_rows),
            "aodt_ids_positive_unique": len({row["aodt_id"] for row in ue_rows})
            == EXPECTED_UE_COUNT,
            "uav_aodt_ids": [
                row["aodt_id"] for row in ue_rows if row["mobility"] == "gpx_aerial"
            ],
        },
    }
    if mapping["validation"] != {
        "ru_count": 8,
        "ground_ue_count": 50,
        "uav_count": 4,
        "total_ue_count": 54,
        "aodt_ids_positive_unique": True,
        "uav_aodt_ids": [51, 52, 53, 54],
    }:
        raise PipelineError("03BU deterministic node mapping validation failed.")
    json_dump(stage_dir / "aodt_node_mapping.json", mapping)
    markdown_write(
        stage_dir / "topology_mapping_report.md",
        "03BU AODT GPX/UAV topology mapping",
        [
            "- RU IDs remain AODT 1-8 and tensor indices 0-7.",
            "- The 50 ground UEs remain AODT 1-50 and tensor indices 0-49.",
            "- UAVs use AODT IDs 51-54 and tensor indices 50-53.",
            "- Each UAV binds one distinct file from the AODT shared GPX manifest.",
        ],
    )
    status = {
        "stage": "03BU",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "ru_count": 8,
        "ground_ue_count": 50,
        "uav_count": 4,
        "total_ue_count": 54,
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def configure_aodt_imports(aodt_client_root: Path) -> dict[str, Path]:
    paths = {
        "client_binding_dir": aodt_client_root / "client" / "build" / "Release",
        "config_binding_dir": aodt_client_root
        / "client"
        / "build"
        / "config"
        / "Release",
    }
    for path in paths.values():
        if not path.is_dir():
            raise PipelineError(f"AODT binding directory is missing: {path}")
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(value)
            except OSError:
                pass
    return paths


def _find_add_list(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    try:
        value = config["sim"][key]["add"]
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"AODT config is missing sim.{key}.add") from exc
    if not isinstance(value, list):
        raise PipelineError(f"AODT config sim.{key}.add is not a list.")
    return value


def _updates_by_id(config: dict[str, Any], section: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for update in config["sim"][section].get("update", []):
        for node_id in update.get("ids", []):
            if isinstance(node_id, int):
                result[node_id] = update.get("attributes", {})
    return result


def validate_uav_config(
    config: dict[str, Any],
    scenario: dict[str, Any],
    expected_worker_paths: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    ru_add = _find_add_list(config, "RUs")
    ue_add = _find_add_list(config, "UEs")
    ru_ids = sorted(int(item["id"]) for item in ru_add)
    ue_ids = sorted(int(item["id"]) for item in ue_add)
    if ru_ids != list(range(1, 9)):
        errors.append(f"Unexpected AODT RU IDs: {ru_ids}")
    if ue_ids != list(range(1, 55)):
        errors.append(f"Unexpected AODT UE IDs: {ue_ids}")
    by_id = {int(item["id"]): item for item in ue_add}
    for node_id in range(1, 51):
        item = by_id.get(node_id, {})
        if len(item.get("waypoints", [])) != 1 or "gpx" in item:
            errors.append(f"Ground UE {node_id} is not one-waypoint static.")
    actual_gpx_paths = []
    for offset, node_id in enumerate(range(51, 55)):
        item = by_id.get(node_id, {})
        gpx = item.get("gpx", {})
        actual_gpx_paths.append(gpx.get("src"))
        if "waypoints" in item:
            errors.append(f"UAV UE {node_id} unexpectedly contains waypoints.")
        if gpx.get("src") != expected_worker_paths[offset]:
            errors.append(f"UAV UE {node_id} does not bind the expected GPX path.")
        if gpx.get("use_pathfinding") is not False:
            errors.append(f"UAV UE {node_id} must disable terrestrial pathfinding.")
    scenario_updates = config["sim"]["Scenario"].get("update", [])
    attrs = scenario_updates[0].get("attributes", {}) if scenario_updates else {}
    if int(attrs.get("sim_simulation_mode", -1)) != 0:
        errors.append("BU must use duration/interval temporal mode.")
    if abs(float(attrs.get("sim_duration", math.nan)) - EXPECTED_DURATION_S) > 1e-12:
        errors.append(
            "BU duration must be 63.75 seconds to yield 256 inclusive samples."
        )
    if (
        abs(float(attrs.get("sim_interval", math.nan)) - EXPECTED_SAMPLE_INTERVAL_S)
        > 1e-12
    ):
        errors.append("BU interval must be 0.25 seconds.")
    if int(attrs.get("sim_slots_per_batch", 0)) != 0:
        errors.append("BU config must not use a nonzero slot/symbol timeline.")
    meta = scenario["meta"]
    radio = scenario["experiment_definition"]["radio_resource_model"]
    expected_freq_mhz = float(meta["freq_hz"]) / 1e6
    for node_id, update in _updates_by_id(config, "RUs").items():
        if float(update.get("aerial_gnb_carrier_freq", math.nan)) != expected_freq_mhz:
            errors.append(f"RU {node_id} frequency differs.")
    for node_id in range(1, 55):
        update = _updates_by_id(config, "UEs").get(node_id, {})
        if float(update.get("aerial_ue_radiated_power", math.nan)) != float(
            radio["ue_tx_power_dbm"]
        ):
            errors.append(f"UE {node_id} radiated power differs.")
        if update.get("aerial_ue_manual") is not True:
            errors.append(f"UE {node_id} manual flag is not true.")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "ru_ids": ru_ids,
        "ue_ids": ue_ids,
        "ground_ue_count": 50,
        "uav_count": 4,
        "gpx_worker_paths": actual_gpx_paths,
        "distinct_gpx_paths": len(set(actual_gpx_paths)) == 4,
        "temporal_mode": "duration_interval",
        "duration_s": attrs.get("sim_duration"),
        "interval_s": attrs.get("sim_interval"),
    }


def stage_04bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "04BU")
    configure_aodt_imports(ctx.aodt_client_root)
    try:
        from _config import (  # type: ignore
            AntennaElement,
            DBTable,
            DiffusionModel,
            GPXSource,
            Nodes,
            Panel,
            Position,
            S3Config,
            SimConfig,
            SimMode,
        )
    except ImportError as exc:
        raise PipelineError(f"Unable to import official AODT config binding: {exc}") from exc
    scenario = json_load(ctx.scenario_path)
    shared = validate_shared_gpx(ctx)
    if shared["status"] != "pass":
        raise PipelineError("04BU received invalid shared GPX inputs.")
    meta = scenario["meta"]
    radio = scenario["experiment_definition"]["radio_resource_model"]
    freq_mhz = float(meta["freq_hz"]) / 1e6
    scs_khz = 15.0 * (2 ** int(radio["numerology"]))
    interactions = min(int(meta["max_depth"]), AODT_MAX_RAY_INTERACTIONS)
    sim_id = "tokyo_uav_8ru_54ue_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    config = SimConfig(ctx.worker_scene_url, SimMode.EM, str(ctx.asset_config.resolve()))
    config.set_simulation_id(sim_id)
    config.set_db(
        db_host="clickhouse",
        db_port=9000,
        db_author="aodt-uav-gpx-pipeline",
        db_notes="8 RU, 50 fixed ground UE, 4 deterministic GPX aerial UE",
    )
    config.set_s3_config(
        S3Config(
            bucket="aerial-data",
            provider="minio",
            endpoint_url="http://minio:9000",
            access_key="${AODT_S3_ACCESS_KEY}",
            secret_key="${AODT_S3_SECRET_KEY}",
        )
    )
    config.set_num_batches(1)
    config.set_timeline(
        duration=EXPECTED_DURATION_S, interval=EXPECTED_SAMPLE_INTERVAL_S
    )
    config.set_seed(int(meta["rng_seed"]))
    # AODT 1.5 exposes TELEMETRY only in RAN mode. GPX requires EM
    # duration/interval mode, so positions are captured through the official
    # get_ue_positions API in 06BU instead.
    for table in (DBTable.CIRS, DBTable.CFRS, DBTable.RAYPATHS):
        config.add_tables_to_db(table)
    config.set_ray_tracing_model(
        DiffusionModel.DIRECTIONAL, interactions, 500, 500
    )
    ru_panel = Panel.create_panel(
        antenna_elements=[AntennaElement.ThreeGPP38901],
        frequency_mhz=freq_mhz,
        vertical_spacing=0.5,
        vertical_num=1,
        horizontal_spacing=0.5,
        horizontal_num=2,
        dual_polarized=True,
        roll_first=0,
        roll_second=90,
    )
    config.set_default_panel_ru(ru_panel)
    ue_panel = Panel.create_panel(
        antenna_elements=[AntennaElement.InfinitesimalDipole],
        frequency_mhz=freq_mhz,
        vertical_spacing=0.5,
        vertical_num=1,
        horizontal_spacing=0.5,
        horizontal_num=1,
        dual_polarized=False,
        roll_first=0,
        roll_second=90,
    )
    config.set_default_panel_ue(ue_panel)
    du = Nodes.create_du(du_id=1, frequency_mhz=freq_mhz, scs_khz=scs_khz)
    du.set_position(Position.cartesian(0.0, 0.0, 0.0))
    du.set_max_channel_bandwidth(float(meta["bandwidth_hz"]) / 1e6)
    config.add_du(du)
    for node in scenario["bs"]:
        ru = Nodes.create_ru(
            ru_id=int(node["id"]) + 1,
            frequency_mhz=freq_mhz,
            radiated_power_dbm=float(meta["tx_power_dbm"]),
            du_id=1,
        )
        ru.set_position(
            Position.cartesian(float(node["x"]), float(node["y"]), float(node["z"]))
        )
        config.add_ru(ru)
    for node in scenario["ue"]:
        ue = Nodes.create_ue(
            ue_id=int(node["id"]) + 1,
            radiated_power_dbm=float(radio["ue_tx_power_dbm"]),
        )
        ue.add_waypoint(
            Position.cartesian(float(node["x"]), float(node["y"]), float(node["z"]))
        )
        ue.set_manual(True)
        config.add_ue(ue)
    for route_index, worker_path in enumerate(worker_gpx_paths(ctx)):
        ue = Nodes.create_ue(
            ue_id=51 + route_index,
            radiated_power_dbm=float(radio["ue_tx_power_dbm"]),
        )
        gpx = GPXSource()
        gpx.src = worker_path
        gpx.usePathfinding = False
        ue.set_gpx_source(gpx)
        ue.set_manual(True)
        config.add_ue(ue)
    config.enable_vegetation()
    config.set_bldg_exterior_attr(
        activate_rf=True,
        activate_diffraction=bool(meta["diffraction"]),
        activate_diffusion=False,
        activate_transmission=True,
        diffuse_surface_element_area=None,
        building_ids=[],
    )
    config.set_bldg_interior_attr(
        activate_rf=True,
        activate_diffraction=bool(meta["diffraction"]),
        activate_transmission=True,
        building_ids=[],
    )
    config_dict = config.to_dict()
    generated_validation = validate_uav_config(
        config_dict, scenario, worker_gpx_paths(ctx)
    )
    if generated_validation["status"] != "pass":
        raise PipelineError(
            "04BU generated config failed: "
            + "; ".join(generated_validation["errors"])
        )
    yaml_path = stage_dir / "resolved_aodt_uav_config.yml"
    yaml_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    roundtrip = SimConfig.from_yaml_file(str(yaml_path.resolve())).to_dict()
    roundtrip_validation = validate_uav_config(
        roundtrip, scenario, worker_gpx_paths(ctx)
    )
    if roundtrip_validation["status"] != "pass":
        raise PipelineError(
            "04BU loader round-trip failed: "
            + "; ".join(roundtrip_validation["errors"])
        )
    gpx_binding = {
        "schema_version": "aodt-uav-gpx-binding-v1",
        "generated_at_utc": utc_now(),
        "trajectory_id": shared["trajectory_id"],
        "aodt_manifest_path": shared["manifest_path"],
        "aodt_manifest_sha256": shared["manifest_sha256"],
        "trajectory_hashes_verified": True,
        "worker_upload_required": True,
        "use_pathfinding": False,
        "files": shared["gpx_inventory"],
    }
    json_dump(stage_dir / "gpx_binding_manifest.json", gpx_binding)
    json_dump(
        stage_dir / "config_validation.json",
        {
            "generated_config": generated_validation,
            "aodt_loader_roundtrip": roundtrip_validation,
        },
    )
    json_dump(
        stage_dir / "resolved_output_config.json",
        {
            "schema_version": "aodt-uav-resolved-output-v1",
            "generated_at_utc": utc_now(),
            "tables_requested": ["CIRS", "CFRS", "RAYPATHS"],
            "position_capture": (
                "DigitalTwinClient.get_ue_positions for all 256 TimeStepIndex "
                "values; AODT TELEMETRY is unavailable in EM mode"
            ),
            "raw_collection": {
                "format": "compressed NPZ",
                "time_steps": ctx.slots_spec,
                "temporal_index_type": "TimeStepIndex",
                "ru_indices": list(range(8)),
                "ue_indices_per_ru": [list(range(54)) for _ in range(8)],
                "full_antenna_pair": ctx.full_antenna_pair,
            },
            "derived_fields_excluded": [
                "path_loss_db",
                "received_power_dbm",
                "snr_db",
                "sinr_db",
                "cqi",
                "serving_ru",
            ],
        },
    )
    markdown_write(
        stage_dir / "config_resolution_report.md",
        "04BU AODT 1.5 GPX/UAV config resolution",
        [
            f"- Simulation ID: `{sim_id}`",
            "- Topology: 8 RU, 50 fixed ground UE, and 4 GPX aerial UE.",
            "- UAVs 51-54 each bind a distinct file from the AODT trajectory manifest.",
            "- Timeline: 0 through 63.75 seconds at 0.25-second intervals, yielding 256 indices.",
            "- UAV pathfinding is disabled so the aerial tracks are not road-projected.",
            "- S3 credentials remain runtime-only placeholders.",
        ],
    )
    status = {
        "stage": "04BU",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "yaml": path_text(yaml_path),
        "yaml_sha256": sha256_file(yaml_path),
        "aodt_loader_roundtrip": "pass",
        "ru_count": 8,
        "ue_count": 54,
        "uav_count": 4,
        "time_steps": 256,
        "sample_interval_s": 0.25,
        "trajectory_hashes_verified": True,
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    endpoint = endpoint.strip()
    if not endpoint or endpoint.count(":") != 1:
        raise ValueError("Endpoint must use host:port format.")
    host, port_text = endpoint.rsplit(":", 1)
    if not host or not port_text.isdigit():
        raise ValueError("Endpoint must use host:port format.")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError("Endpoint port is outside 1..65535.")
    return host, port


def aodt_member_value(client: Any, name: str) -> Any:
    value = getattr(client, name)
    return value() if callable(value) else value


def stage_05bu(ctx: UAVPipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "05BU")
    imported = False
    loader_roundtrip = False
    import_error = None
    client_api: list[str] = []
    paths: dict[str, Path] = {}
    try:
        paths = configure_aodt_imports(ctx.aodt_client_root)
        import dt_client  # type: ignore
        from _config import SimConfig  # type: ignore

        imported = True
        loader_roundtrip = bool(
            SimConfig.from_yaml_file(
                str(
                    (ctx.stage_dir("04BU") / "resolved_aodt_uav_config.yml").resolve()
                )
            ).to_dict()
        )
        client_api = [
            name
            for name in (
                "start",
                "get_status",
                "get_ru_positions",
                "get_ue_positions",
                "run_full_simulation",
                "allocate_cirs_memory",
                "get_cirs",
                "to_numpy_all_cir",
                "deallocate_cirs_memory",
            )
            if hasattr(dt_client.DigitalTwinClient, name)
        ]
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
    assets = yaml.safe_load(ctx.asset_config.read_text(encoding="utf-8"))
    asset_keys_valid = isinstance(assets, dict) and all(
        isinstance(assets.get(key), str) and assets[key] for key in ASSET_KEYS
    )
    shared = validate_shared_gpx(ctx)
    local_gpx_hashes_valid = shared["status"] == "pass"
    endpoint_valid = False
    tcp_reachable = False
    grpc_connected = False
    transport_mode = None
    online_error = None
    if ctx.endpoint:
        try:
            host, port = parse_endpoint(ctx.endpoint)
            endpoint_valid = True
            if not ctx.offline:
                with socket.create_connection((host, port), timeout=5):
                    tcp_reachable = True
                if imported:
                    client = dt_client.DigitalTwinClient(ctx.endpoint, force=False)
                    grpc_connected = bool(aodt_member_value(client, "is_connected"))
                    transport_mode = aodt_member_value(client, "transport_mode")
        except Exception as exc:
            online_error = f"{type(exc).__name__}: {exc}"
    required_api = {
        "start",
        "get_status",
        "get_ru_positions",
        "get_ue_positions",
        "run_full_simulation",
        "allocate_cirs_memory",
        "get_cirs",
        "to_numpy_all_cir",
        "deallocate_cirs_memory",
    }
    checks = {
        "official_bindings_import": imported,
        "aodt_yaml_loader_roundtrip": loader_roundtrip,
        "required_client_api_complete": required_api.issubset(client_api),
        "asset_config_keys_valid": asset_keys_valid,
        "local_shared_gpx_hashes_valid": local_gpx_hashes_valid,
        "worker_assets_operator_confirmed": ctx.ack_worker_assets_resolved,
        "worker_gpx_operator_confirmed": ctx.ack_worker_gpx_resolved,
        "s3_access_key_env_present": bool(os.environ.get("AODT_S3_ACCESS_KEY")),
        "s3_secret_key_env_present": bool(os.environ.get("AODT_S3_SECRET_KEY")),
        "endpoint_present_and_valid": endpoint_valid,
        "tcp_reachable": tcp_reachable,
        "grpc_connected": grpc_connected,
        "transport_is_remote": transport_mode == "REMOTE",
    }
    if ctx.offline:
        preflight_status = "blocked_offline"
    elif all(checks.values()):
        preflight_status = "pass"
    else:
        preflight_status = "fail"
    environment = {
        "schema_version": "aodt-uav-environment-inventory-v1",
        "generated_at_utc": utc_now(),
        "python": {"executable": path_text(Path(sys.executable)), "version": sys.version},
        "aodt_client": {
            "root": path_text(ctx.aodt_client_root),
            "git_commit": _git_value(ctx.aodt_client_root, "rev-parse", "HEAD"),
            "git_branch": _git_value(
                ctx.aodt_client_root, "rev-parse", "--abbrev-ref", "HEAD"
            ),
            "bindings": {name: path_text(path) for name, path in paths.items()},
            "required_api_found": sorted(client_api),
            "import_error": import_error,
        },
        "worker_gpx": {
            "operator_confirmed_resolved": ctx.ack_worker_gpx_resolved,
            "trajectory_id": shared["trajectory_id"],
            "files": shared["gpx_inventory"],
            "trajectory_hashes_verified": shared["trajectory_hashes_verified"],
        },
        "runtime_secrets": {
            "AODT_S3_ACCESS_KEY_present": bool(os.environ.get("AODT_S3_ACCESS_KEY")),
            "AODT_S3_SECRET_KEY_present": bool(
                os.environ.get("AODT_S3_SECRET_KEY")
            ),
            "values_recorded": False,
        },
        "network": {
            "endpoint": ctx.endpoint or None,
            "endpoint_valid": endpoint_valid,
            "tcp_reachable": tcp_reachable,
            "grpc_connected": grpc_connected,
            "transport_mode": transport_mode,
            "error": online_error,
        },
        "secret_values_recorded": False,
    }
    status = {
        "schema_version": "aodt-uav-preflight-v1",
        "stage": "05BU",
        "generated_at_utc": utc_now(),
        "status": preflight_status,
        "offline_requested": ctx.offline,
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
        "execution_authorized": preflight_status == "pass",
        "transport_contract": "official DigitalTwinClient direct gRPC; no SSH",
    }
    json_dump(stage_dir / "environment_inventory.json", environment)
    json_dump(stage_dir / "preflight_status.json", status)
    markdown_write(
        stage_dir / "preflight_report.md",
        "05BU AODT GPX/UAV direct-gRPC preflight",
        [
            f"- Status: **{preflight_status}**",
            f"- Shared AODT GPX hash validation: {local_gpx_hashes_valid}.",
            f"- Worker GPX confirmation: {ctx.ack_worker_gpx_resolved}.",
            f"- Worker asset confirmation: {ctx.ack_worker_assets_resolved}.",
            f"- Direct gRPC REMOTE connection: {grpc_connected and transport_mode == 'REMOTE'}.",
            "- 06BU and 07BU are blocked unless every online gate passes.",
        ],
    )
    return status
