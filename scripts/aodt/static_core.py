"""Offline-capable AODT fixed-UE stages 00B through 05B.

The module is intentionally strict: it accepts only the locked 8-RU/50-UE
Tokyo scenario, creates positive AODT IDs deterministically, and never invents
RF outputs or backend object paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


STAGES = ("00B", "01B", "02B", "03B", "04B", "05B", "06B", "07B")
STAGE_DIRS = {
    "00B": "00B_inspection",
    "01B": "01B_inputs",
    "02B": "02B_scene_materials",
    "03B": "03B_topology",
    "04B": "04B_config",
    "05B": "05B_preflight",
    "06B": "06B_execution",
    "07B": "07B_raw_collection",
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
PROHIBITED_PATTERN = re.compile(r"(?i)(?:^|[^a-z0-9])(uav|drone|gpx|aerial)(?:$|[^a-z0-9])")
AODT_MAX_RAY_INTERACTIONS = 5


class PipelineError(RuntimeError):
    """Fail-closed pipeline error."""


def aodt_member_value(client: Any, name: str) -> Any:
    """Read an AODT binding member exposed as either a property or method."""
    value = getattr(client, name)
    return value() if callable(value) else value


@dataclass(frozen=True)
class PipelineContext:
    repo_root: Path
    run_dir: Path
    scenario_path: Path
    scene_archive: Path
    aodt_client_root: Path
    asset_config: Path
    worker_scene_url: str
    endpoint: str
    offline: bool
    ack_worker_assets_resolved: bool
    slots_spec: str
    full_antenna_pair: bool
    command: str

    def stage_dir(self, stage: str) -> Path:
        return self.run_dir / STAGE_DIRS[stage]

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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_write(path: Path, title: str, lines: Iterable[str]) -> None:
    body = [f"# {title}", "", *lines, ""]
    path.write_text("\n".join(body), encoding="utf-8")


def csv_write(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_unique_run_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        candidate = results_root / f"tokyo_aodt_static_{stamp}_{suffix:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise PipelineError("Unable to allocate a unique run directory.")


def create_stage_dir(ctx: PipelineContext, stage: str) -> Path:
    stage_dir = ctx.stage_dir(stage)
    try:
        stage_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise PipelineError(
            f"Refusing to overwrite existing stage directory: {stage_dir}"
        ) from exc
    return stage_dir


def recursive_strings(value: Any, trail: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{trail}.<key>", str(key)
            yield from recursive_strings(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from recursive_strings(item, f"{trail}[{index}]")
    elif isinstance(value, str):
        yield trail, value


def validate_locked_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if set(scenario) != {"meta", "bs", "ue", "experiment_definition"}:
        errors.append("Top-level keys must be meta, bs, ue, experiment_definition.")
    meta = scenario.get("meta", {})
    rus = scenario.get("bs", [])
    ues = scenario.get("ue", [])
    if len(rus) != 8:
        errors.append(f"Expected 8 RUs, found {len(rus)}.")
    if len(ues) != 50:
        errors.append(f"Expected 50 UEs, found {len(ues)}.")
    if meta.get("num_bs") != 8 or meta.get("num_ue") != 50:
        errors.append("meta.num_bs/num_ue do not match the locked 8/50 contract.")

    prohibited: list[dict[str, str]] = []
    for trail, text in recursive_strings(scenario):
        match = PROHIBITED_PATTERN.search(text)
        if match:
            prohibited.append({"path": trail, "token": match.group(1), "text": text})
    if prohibited:
        errors.append("Scenario contains prohibited UAV/drone/GPX/aerial content.")

    def check_nodes(nodes: list[dict[str, Any]], expected_ids: list[int], kind: str) -> None:
        ids = [node.get("id") for node in nodes]
        if ids != expected_ids:
            errors.append(f"{kind} IDs must be ordered and contiguous: {expected_ids}.")
        if len(set(ids)) != len(ids):
            errors.append(f"{kind} IDs are not unique.")
        for index, node in enumerate(nodes):
            for axis in ("x", "y", "z"):
                value = node.get(axis)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(f"{kind}[{index}].{axis} is not finite.")

    check_nodes(rus, list(range(8)), "RU")
    check_nodes(ues, list(range(50)), "UE")
    non_terrestrial = [
        ue.get("id")
        for ue in ues
        if str(ue.get("ue_type", "")).strip().lower() not in {"indoor", "outdoor"}
    ]
    if non_terrestrial:
        errors.append(f"Non-terrestrial UE types found: {non_terrestrial}.")

    if meta.get("time_steps") != 256:
        warnings.append(
            f"Locked reference uses 256 slots; input contains {meta.get('time_steps')}."
        )

    return {
        "schema_version": "aodt-static-scenario-validation-v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "ru": len(rus),
            "ue": len(ues),
            "static_ue": len(ues),
            "terrestrial_ue": len(ues) - len(non_terrestrial),
            "aerial_ue": len(non_terrestrial),
            "gpx_references": sum(
                1
                for _, text in recursive_strings(scenario)
                if re.search(r"(?i)\bgpx\b", text)
            ),
        },
        "prohibited_content": prohibited,
        "identity_contract": {
            "scenario_ids_zero_based": True,
            "aodt_ids_positive": True,
            "mapping_rule": "aodt_id = scenario_id + 1",
            "tensor_api_indices_zero_based": True,
        },
        "mobility_contract": {
            "mode": "fixed",
            "waypoints_per_ue_in_resolved_aodt_config": 1,
            "waypoint_speed_mps": 0.0,
            "procedural_ues": 0,
            "spawn_zones": 0,
            "urban_mobility": False,
        },
    }


def update_run_manifest(
    ctx: PipelineContext,
    stage: str | None = None,
    status: dict[str, Any] | None = None,
    overall_status: str | None = None,
) -> None:
    path = ctx.run_dir / "run_manifest.json"
    if path.exists():
        manifest = json_load(path)
    else:
        manifest = {
            "schema_version": "aodt-static-run-v1",
            "created_at_utc": utc_now(),
            "run_directory": path_text(ctx.run_dir),
            "scope": {
                "ru": 8,
                "ue": 50,
                "ue_mobility": "fixed",
                "uav": 0,
                "gpx": 0,
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


def stage_00b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "00B")
    source_dir = ctx.repo_root / "evidence" / "baseline"
    missing = [name for name in BASELINE_EVIDENCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise PipelineError(f"00B baseline evidence is incomplete: {missing}")
    files = []
    for name in BASELINE_EVIDENCE_FILES:
        source = source_dir / name
        target = stage_dir / name
        shutil.copy2(source, target)
        files.append(
            {
                "name": name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    capability = json_load(stage_dir / "aodt_capability_inventory.json")
    topology = json_load(stage_dir / "static_topology_inventory.json")
    checks = {
        "aodt_version": capability.get("aodt_version_lock", {}).get("version"),
        "scenario_sha256": topology.get("scenario_source_sha256"),
        "counts": topology.get("counts"),
        "fixed_ue_contract": (
            topology.get("counts", {}).get("ru") == 8
            and topology.get("counts", {}).get("ue") == 50
            and topology.get("counts", {}).get("uav") == 0
            and topology.get("counts", {}).get("gpx_references") == 0
        ),
    }
    status = {
        "stage": "00B",
        "status": "pass" if checks["fixed_ue_contract"] else "fail",
        "generated_at_utc": utc_now(),
        "source_directory": path_text(source_dir),
        "copied_files": files,
        "checks": checks,
        "note": "Versioned baseline evidence copied without modification.",
    }
    json_dump(stage_dir / "stage_status.json", status)
    if status["status"] != "pass":
        raise PipelineError("00B fixed topology inspection gate failed.")
    return status


def stage_01b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "01B")
    scenario = json_load(ctx.scenario_path)
    validation = validate_locked_scenario(scenario)
    if validation["status"] != "pass":
        json_dump(stage_dir / "scenario_contract_validation.json", validation)
        raise PipelineError(
            "01B scenario contract failed: " + "; ".join(validation["errors"])
        )

    capability_path = ctx.stage_dir("00B") / "aodt_capability_inventory.json"
    capability = json_load(capability_path)
    geometry_bounds = capability["tokyo_scene"]["geometry_inventory"][
        "ground_bounds_m"
    ]
    xy_min = geometry_bounds["min"][:2]
    xy_max = geometry_bounds["max"][:2]
    outside = []
    for kind, nodes in (("RU", scenario["bs"]), ("UE", scenario["ue"])):
        for node in nodes:
            if not (
                xy_min[0] <= float(node["x"]) <= xy_max[0]
                and xy_min[1] <= float(node["y"]) <= xy_max[1]
            ):
                outside.append({"type": kind, "id": node["id"]})

    topology_validation = {
        "schema_version": "aodt-static-topology-validation-v1",
        "generated_at_utc": utc_now(),
        "status": "pass" if not outside else "fail",
        "counts": validation["counts"],
        "coordinate_frame": "Tokyo scene-local Cartesian",
        "units": "metres",
        "axis_order": ["x", "y", "z"],
        "z_semantics": scenario["meta"].get("z_coordinate_semantics"),
        "ground_xy_bounds_m": {"min": xy_min, "max": xy_max},
        "nodes_outside_ground_xy_bounds": outside,
        "all_ids_explicit_and_unique": True,
        "all_ue_positions_explicit_and_static": True,
        "aodt_id_mapping_rule": "aodt_id = scenario_id + 1",
    }
    if outside:
        validation["status"] = "fail"
        validation["errors"].append(f"Nodes outside scene XY bounds: {outside}")

    input_manifest = {
        "schema_version": "aodt-static-input-manifest-v1",
        "generated_at_utc": utc_now(),
        "scenario": {
            "path": path_text(ctx.scenario_path),
            "bytes": ctx.scenario_path.stat().st_size,
            "sha256": sha256_file(ctx.scenario_path),
            "version": scenario["meta"].get("version"),
        },
        "scene_archive": {
            "path": path_text(ctx.scene_archive),
            "bytes": ctx.scene_archive.stat().st_size,
            "sha256": sha256_file(ctx.scene_archive),
            "primary_stage": "tokyo/sim/master.usd",
        },
        "aodt_client": {
            "root": path_text(ctx.aodt_client_root),
            "expected_version": "1.5.0",
            "transport": "direct gRPC",
        },
        "asset_config": {
            "path": path_text(ctx.asset_config),
            "sha256": sha256_file(ctx.asset_config),
            "resolution_scope": "worker-side paths",
        },
        "worker_scene_url": ctx.worker_scene_url,
        "secret_values_recorded": False,
    }
    json_dump(stage_dir / "aodt_static_input_manifest.json", input_manifest)
    json_dump(stage_dir / "scenario_contract_validation.json", validation)
    json_dump(stage_dir / "topology_validation.json", topology_validation)
    status = {
        "stage": "01B",
        "status": "pass"
        if validation["status"] == "pass"
        and topology_validation["status"] == "pass"
        else "fail",
        "generated_at_utc": utc_now(),
        "scenario_sha256": input_manifest["scenario"]["sha256"],
        "scene_archive_sha256": input_manifest["scene_archive"]["sha256"],
    }
    json_dump(stage_dir / "stage_status.json", status)
    if status["status"] != "pass":
        raise PipelineError("01B topology gate failed.")
    return status


def _zip_json(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    with archive.open(name) as handle:
        return json.loads(handle.read().decode("utf-8"))


def stage_02b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "02B")
    capability = json_load(
        ctx.stage_dir("00B") / "aodt_capability_inventory.json"
    )
    scene_evidence = capability["tokyo_scene"]
    required_entries = (
        "tokyo/sim/master.usd",
        "tokyo/sim/master_tiles/tile_0_0.usd",
        "tokyo/sim/master_metadata.json",
        "tokyo/sim/master_input_manifest.json",
        "tokyo/sim/vegetation.geojson",
    )
    with zipfile.ZipFile(ctx.scene_archive, "r") as archive:
        names = set(archive.namelist())
        missing = [name for name in required_entries if name not in names]
        if missing:
            raise PipelineError(f"02B Tokyo archive is missing entries: {missing}")
        metadata = _zip_json(archive, "tokyo/sim/master_metadata.json")
        input_manifest = _zip_json(
            archive, "tokyo/sim/master_input_manifest.json"
        )
        vegetation = _zip_json(archive, "tokyo/sim/vegetation.geojson")
        entries = []
        for name in required_entries:
            info = archive.getinfo(name)
            entries.append(
                {
                    "path": name,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )

    geometry = scene_evidence["geometry_inventory"]
    material = scene_evidence["usd_material_inventory"]
    scene_inventory = {
        "schema_version": "aodt-static-scene-inventory-v1",
        "generated_at_utc": utc_now(),
        "status": "pass_with_material_limitations",
        "archive": {
            "path": path_text(ctx.scene_archive),
            "sha256": sha256_file(ctx.scene_archive),
            "bytes": ctx.scene_archive.stat().st_size,
        },
        "entries": entries,
        "metadata": metadata,
        "input_parameters": input_manifest.get("params", {}),
        "vegetation": {
            "path": "tokyo/sim/vegetation.geojson",
            "geojson_type": vegetation.get("type"),
            "feature_count": len(vegetation.get("features", [])),
            "source": input_manifest.get("params", {}).get("vegetation_source"),
        },
        "worker_scene_url": ctx.worker_scene_url,
    }
    geometry_inventory = {
        "schema_version": "aodt-static-geometry-inventory-v1",
        "generated_at_utc": utc_now(),
        "status": "pass",
        "source_evidence": "00B_inspection/aodt_capability_inventory.json",
        **geometry,
    }
    material_inventory = {
        "schema_version": "aodt-static-material-inventory-v1",
        "generated_at_utc": utc_now(),
        "status": "partial_unresolved_worker_binding",
        "source_evidence": "00B_inspection/aodt_capability_inventory.json",
        **material,
        "worker_asset_config": path_text(ctx.asset_config),
        "operator_confirmation_required": True,
    }
    binding_rows = [
        {
            "stage_path": "tokyo/sim/master_tiles/tile_0_0.usd",
            "prim_scope": "building meshes (aggregate evidence)",
            "mesh_count": geometry["building_meshes_total"],
            "material": "DefaultMaterial",
            "direct_binding_count": material["tile_stage_direct_binding_count"],
            "tile_stage_resolution": "resolved",
            "composed_master_resolution": "unresolved",
            "evidence_kind": "OpenUSD aggregate inspection",
        },
        {
            "stage_path": "tokyo/sim/master_tiles/tile_0_0.usd",
            "prim_scope": "ground mesh (aggregate evidence)",
            "mesh_count": geometry["ground_meshes"],
            "material": "",
            "direct_binding_count": 0,
            "tile_stage_resolution": "unbound",
            "composed_master_resolution": "unbound",
            "evidence_kind": "OpenUSD aggregate inspection",
        },
    ]
    material_validation = {
        "schema_version": "aodt-static-material-validation-v1",
        "generated_at_utc": utc_now(),
        "status": "blocked_pending_worker_asset_confirmation",
        "checks": {
            "tokyo_usd_present": True,
            "vegetation_geojson_present": True,
            "geometry_inventory_nonempty": geometry["mesh_count"] > 0,
            "material_definition_present": material["material_definition_count"] > 0,
            "composed_master_bindings_resolved": False,
            "semantic_material_tag_mapping_supplied": False,
            "worker_material_json_locally_resolved": False,
        },
        "execution_gate": (
            "05B requires --ack-worker-assets-resolved after verifying the exact "
            "AODT asset/material/vegetation bundle on the worker."
        ),
        "claim_boundary": material["scene_limitation"],
    }
    json_dump(stage_dir / "aodt_scene_inventory.json", scene_inventory)
    json_dump(stage_dir / "aodt_geometry_inventory.json", geometry_inventory)
    json_dump(stage_dir / "aodt_material_inventory.json", material_inventory)
    csv_write(
        stage_dir / "aodt_material_binding_inventory.csv",
        [
            "stage_path",
            "prim_scope",
            "mesh_count",
            "material",
            "direct_binding_count",
            "tile_stage_resolution",
            "composed_master_resolution",
            "evidence_kind",
        ],
        binding_rows,
    )
    json_dump(stage_dir / "aodt_material_validation.json", material_validation)
    markdown_write(
        stage_dir / "aodt_material_report.md",
        "02B AODT scene and material report",
        [
            f"- Scene archive SHA-256: `{scene_inventory['archive']['sha256']}`",
            f"- Geometry: {geometry['mesh_count']} meshes "
            f"({geometry['building_meshes_total']} building, "
            f"{geometry['ground_meshes']} ground).",
            f"- Vegetation GeoJSON features: "
            f"{scene_inventory['vegetation']['feature_count']}.",
            "- The tile contains aggregate DefaultMaterial binding evidence, "
            "but the composed master binding is unresolved.",
            "- MaterialTag values exist without a supplied semantic mapping; "
            "glass, concrete, metal, and vegetation RF effects are not claimed.",
            "- 05B must confirm the exact worker-side assets before execution.",
        ],
    )
    status = {
        "stage": "02B",
        "status": "pass_with_online_gate",
        "generated_at_utc": utc_now(),
        "geometry_inventory_status": "pass",
        "material_validation_status": material_validation["status"],
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def _node_rows(
    nodes: list[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    rows = []
    for tensor_index, node in enumerate(nodes):
        source_id = int(node["id"])
        display_id = f"{kind}{source_id:02d}"
        rows.append(
            {
                "node_type": kind,
                "scenario_id": source_id,
                "display_id": display_id,
                "aodt_id": source_id + 1,
                "tensor_api_index": tensor_index,
                "x_m": float(node["x"]),
                "y_m": float(node["y"]),
                "z_m": float(node["z"]),
                "mobility": "static",
                "config_reference": f"sim.{kind}s.add[id={source_id + 1}]",
                "backend_object_path": "",
                "backend_path_status": "pending_server_registry",
                "ue_type": node.get("ue_type", "") if kind == "UE" else "",
            }
        )
    return rows


def stage_03b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "03B")
    scenario = json_load(ctx.scenario_path)
    validation = validate_locked_scenario(scenario)
    if validation["status"] != "pass":
        raise PipelineError("03B received an invalid locked scenario.")
    ru_rows = _node_rows(scenario["bs"], "RU")
    ue_rows = _node_rows(scenario["ue"], "UE")
    fields = [
        "node_type",
        "scenario_id",
        "display_id",
        "aodt_id",
        "tensor_api_index",
        "x_m",
        "y_m",
        "z_m",
        "mobility",
        "config_reference",
        "backend_object_path",
        "backend_path_status",
        "ue_type",
    ]
    csv_write(stage_dir / "ru_mapping.csv", fields, ru_rows)
    csv_write(stage_dir / "ue_mapping.csv", fields, ue_rows)
    csv_write(stage_dir / "static_topology_snapshot.csv", fields, [*ru_rows, *ue_rows])

    mapping = {
        "schema_version": "aodt-static-node-mapping-v1",
        "generated_at_utc": utc_now(),
        "status": "config_ids_locked_backend_paths_pending",
        "mapping_rule": "aodt_id = scenario_id + 1",
        "index_rule": "DigitalTwinClient tensor API index = scenario_id",
        "ru": ru_rows,
        "ue": ue_rows,
        "validation": {
            "ru_count": len(ru_rows),
            "ue_count": len(ue_rows),
            "aodt_ids_positive": all(row["aodt_id"] > 0 for row in [*ru_rows, *ue_rows]),
            "aodt_ru_ids_unique": len({row["aodt_id"] for row in ru_rows}) == 8,
            "aodt_ue_ids_unique": len({row["aodt_id"] for row in ue_rows}) == 50,
            "coordinates_preserved": True,
            "backend_object_paths_invented": False,
        },
    }
    json_dump(stage_dir / "aodt_node_mapping.json", mapping)
    markdown_write(
        stage_dir / "topology_mapping_report.md",
        "03B fixed topology mapping",
        [
            "- Scenario RU IDs 0–7 map to AODT IDs 1–8.",
            "- Scenario UE IDs 0–49 map to AODT IDs 1–50.",
            "- DigitalTwinClient tensor indices remain zero-based and match "
            "the ordered scenario IDs.",
            "- Every UE has exactly one fixed Cartesian position.",
            "- Backend USD/object paths are intentionally blank until a real "
            "worker registry provides evidence.",
        ],
    )
    status = {
        "stage": "03B",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "ru_count": len(ru_rows),
        "ue_count": len(ue_rows),
        "backend_object_paths": "pending_worker_evidence",
    }
    json_dump(stage_dir / "stage_status.json", status)
    return status


def configure_aodt_imports(aodt_client_root: Path) -> dict[str, Path]:
    paths = {
        "client_binding_dir": aodt_client_root / "client" / "build" / "Release",
        "config_binding_dir": (
            aodt_client_root / "client" / "build" / "config" / "Release"
        ),
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


def _config_validation(
    config_dict: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    ru_add = _find_add_list(config_dict, "RUs")
    ue_add = _find_add_list(config_dict, "UEs")
    ru_ids = sorted(int(item["id"]) for item in ru_add)
    ue_ids = sorted(int(item["id"]) for item in ue_add)
    prohibited_hits = []
    allowed_schema_keys = []
    for path, text in recursive_strings(config_dict):
        if not PROHIBITED_PATTERN.search(text):
            continue
        if path.endswith(".<key>") and text.startswith("aerial_"):
            allowed_schema_keys.append({"path": path, "key": text})
            continue
        if path == "$.db.s3_config.bucket" and text == "aerial-data":
            allowed_schema_keys.append(
                {
                    "path": path,
                    "key": text,
                    "reason": "official worker object-storage bucket name",
                }
            )
            continue
        prohibited_hits.append({"path": path, "text": text})
    waypoint_counts = {
        int(item["id"]): len(item.get("waypoints", [])) for item in ue_add
    }
    speeds = [
        float(waypoint.get("speed", 0.0))
        for item in ue_add
        for waypoint in item.get("waypoints", [])
    ]
    errors = []
    if ru_ids != list(range(1, 9)):
        errors.append(f"Unexpected AODT RU IDs: {ru_ids}")
    if ue_ids != list(range(1, 51)):
        errors.append(f"Unexpected AODT UE IDs: {ue_ids}")
    if set(waypoint_counts.values()) != {1}:
        errors.append(f"Each UE must have one waypoint: {waypoint_counts}")
    if any(speed != 0.0 for speed in speeds):
        errors.append("All UE waypoint speeds must be zero.")
    if prohibited_hits:
        errors.append(
            "Resolved AODT config contains prohibited content: "
            f"{prohibited_hits[:10]}"
        )
    ue_scenario_positions = {
        int(node["id"]) + 1: (float(node["x"]), float(node["y"]), float(node["z"]))
        for node in scenario["ue"]
    }
    ru_scenario_positions = {
        int(node["id"]) + 1: (float(node["x"]), float(node["y"]), float(node["z"]))
        for node in scenario["bs"]
    }
    mismatches = []
    for item in ue_add:
        pos = item["waypoints"][0]["pos"]
        actual = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        expected = ue_scenario_positions[int(item["id"])]
        if any(abs(a - b) > 1e-9 for a, b in zip(actual, expected)):
            mismatches.append(
                {
                    "node_type": "UE",
                    "aodt_id": item["id"],
                    "actual": actual,
                    "expected": expected,
                }
            )
    for item in ru_add:
        pos = item["position"]["pos"]
        actual = (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        expected = ru_scenario_positions[int(item["id"])]
        if any(abs(a - b) > 1e-9 for a, b in zip(actual, expected)):
            mismatches.append(
                {
                    "node_type": "RU",
                    "aodt_id": item["id"],
                    "actual": actual,
                    "expected": expected,
                }
            )
    if mismatches:
        errors.append("RU or UE coordinates changed during AODT config generation.")

    def updates_by_id(section_name: str) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for update in config_dict["sim"][section_name].get("update", []):
            for node_id in update.get("ids", []):
                if isinstance(node_id, int):
                    result[node_id] = update.get("attributes", {})
        return result

    meta = scenario["meta"]
    radio_model = scenario["experiment_definition"]["radio_resource_model"]
    expected_freq_mhz = float(meta["freq_hz"]) / 1e6
    attribute_errors = []
    ru_updates = updates_by_id("RUs")
    ue_updates = updates_by_id("UEs")
    for node_id in range(1, 9):
        attrs = ru_updates.get(node_id, {})
        if float(attrs.get("aerial_gnb_carrier_freq", math.nan)) != expected_freq_mhz:
            attribute_errors.append(f"RU {node_id} frequency")
        if float(attrs.get("aerial_gnb_radiated_power", math.nan)) != float(
            meta["tx_power_dbm"]
        ):
            attribute_errors.append(f"RU {node_id} radiated power")
        if int(attrs.get("aerial_gnb_du_id", -1)) != 1:
            attribute_errors.append(f"RU {node_id} DU mapping")
    for node_id in range(1, 51):
        attrs = ue_updates.get(node_id, {})
        if float(attrs.get("aerial_ue_radiated_power", math.nan)) != float(
            radio_model["ue_tx_power_dbm"]
        ):
            attribute_errors.append(f"UE {node_id} radiated power")
        if attrs.get("aerial_ue_manual") is not True:
            attribute_errors.append(f"UE {node_id} manual/static flag")
    scenario_updates = config_dict["sim"]["Scenario"].get("update", [])
    scenario_attrs = (
        scenario_updates[0].get("attributes", {}) if scenario_updates else {}
    )
    if int(scenario_attrs.get("sim_slots_per_batch", -1)) != int(meta["time_steps"]):
        attribute_errors.append("scenario slot count")
    if int(scenario_attrs.get("sim_seed", -1)) != int(meta["rng_seed"]):
        attribute_errors.append("scenario seed")
    expected_interactions = min(
        int(meta["max_depth"]), AODT_MAX_RAY_INTERACTIONS
    )
    if int(scenario_attrs.get("sim_em_interactions", -1)) != expected_interactions:
        attribute_errors.append(
            "scenario EM interactions "
            f"(expected AODT-safe {expected_interactions})"
        )
    if attribute_errors:
        errors.append(f"Resolved radio attributes differ: {attribute_errors[:20]}")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "ru_ids": ru_ids,
        "ue_ids": ue_ids,
        "waypoints_per_ue": waypoint_counts,
        "all_waypoint_speeds_zero": all(speed == 0.0 for speed in speeds),
        "prohibited_content": prohibited_hits,
        "allowed_official_schema_keys": allowed_schema_keys,
        "allowed_schema_key_note": (
            "AODT 1.5 internally prefixes radio attributes with aerial_. "
            "This does not create an aerial UE; all node values remain fixed "
            "Cartesian terrestrial positions and no mobility API is configured."
        ),
        "coordinate_mismatches": mismatches,
        "radio_attribute_errors": attribute_errors,
        "radio_attributes_preserved": not attribute_errors,
        "procedural_ues_configured": False,
        "spawn_zones_configured": False,
        "urban_mobility_configured": False,
    }


def stage_04b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "04B")
    configure_aodt_imports(ctx.aodt_client_root)
    try:
        from _config import (  # type: ignore
            AntennaElement,
            DBTable,
            DiffusionModel,
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
    validation = validate_locked_scenario(scenario)
    if validation["status"] != "pass":
        raise PipelineError("04B received an invalid scenario.")
    meta = scenario["meta"]
    radio_model = scenario["experiment_definition"]["radio_resource_model"]
    freq_mhz = float(meta["freq_hz"]) / 1e6
    scs_khz = 15.0 * (2 ** int(radio_model["numerology"]))
    source_ray_interactions = int(meta["max_depth"])
    aodt_ray_interactions = min(
        source_ray_interactions, AODT_MAX_RAY_INTERACTIONS
    )
    sim_id = (
        "tokyo_fixed_8ru_50ue_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    config = SimConfig(
        ctx.worker_scene_url, SimMode.EM, str(ctx.asset_config.resolve())
    )
    config.set_simulation_id(sim_id)
    config.set_db(
        db_host="clickhouse",
        db_port=9000,
        db_author="aodt-static-pipeline",
        db_notes="8 RU, 50 fixed terrestrial UE; raw collection over official gRPC",
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
        slots_per_batch=int(meta["time_steps"]), realizations_per_slot=1
    )
    config.set_seed(int(meta["rng_seed"]))
    config.add_tables_to_db(DBTable.CIRS)
    config.add_tables_to_db(DBTable.CFRS)
    config.add_tables_to_db(DBTable.RAYPATHS)
    config.set_ray_tracing_model(
        DiffusionModel.DIRECTIONAL,
        aodt_ray_interactions,
        500,
        500,
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
            radiated_power_dbm=float(radio_model["ue_tx_power_dbm"]),
        )
        ue.add_waypoint(
            Position.cartesian(float(node["x"]), float(node["y"]), float(node["z"]))
        )
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
    config_validation = _config_validation(config_dict, scenario)
    if config_validation["status"] != "pass":
        raise PipelineError(
            "04B generated config failed validation: "
            + "; ".join(config_validation["errors"])
        )

    yaml_path = stage_dir / "resolved_aodt_static_config.yml"
    yaml_path.write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    loaded = SimConfig.from_yaml_file(str(yaml_path.resolve())).to_dict()
    roundtrip_validation = _config_validation(loaded, scenario)
    if roundtrip_validation["status"] != "pass":
        raise PipelineError(
            "04B AODT loader round-trip failed: "
            + "; ".join(roundtrip_validation["errors"])
        )

    scene_config = {
        "schema_version": "aodt-static-resolved-scene-v1",
        "generated_at_utc": utc_now(),
        "worker_scene_url": ctx.worker_scene_url,
        "source_archive": path_text(ctx.scene_archive),
        "source_archive_sha256": sha256_file(ctx.scene_archive),
        "primary_usd": "tokyo/sim/master.usd",
        "vegetation": {
            "active": True,
            "derived_worker_path": f"{ctx.worker_scene_url.rstrip('/')}/sim/vegetation.geojson",
        },
        "materials": {
            "asset_config": path_text(ctx.asset_config),
            "worker_resolution_confirmed": ctx.ack_worker_assets_resolved,
            "composed_master_binding_status": "unresolved_offline",
        },
    }
    radio_config = {
        "schema_version": "aodt-static-resolved-radio-v1",
        "generated_at_utc": utc_now(),
        "source": {
            "frequency_hz": meta["freq_hz"],
            "bandwidth_hz": meta["bandwidth_hz"],
            "ru_tx_power_dbm": meta["tx_power_dbm"],
            "ue_tx_power_dbm": radio_model["ue_tx_power_dbm"],
            "numerology": radio_model["numerology"],
            "subcarrier_spacing_khz": scs_khz,
            "seed": meta["rng_seed"],
            "slots": meta["time_steps"],
        },
        "aodt_backend": {
            "mode": "EM",
            "ray_interactions": aodt_ray_interactions,
            "source_requested_max_depth": source_ray_interactions,
            "interaction_mapping": (
                "min(source_requested_max_depth, "
                f"AODT_MAX_RAY_INTERACTIONS={AODT_MAX_RAY_INTERACTIONS})"
            ),
            "diffusion_model_parameter": "DIRECTIONAL",
            "building_diffusion_active": False,
            "building_diffraction_active": bool(meta["diffraction"]),
            "building_transmission_active": True,
            "ru_panel": "ThreeGPP38901 1x2 dual-polarized",
            "ue_panel": "InfinitesimalDipole 1x1 single-polarized",
        },
        "standards_boundary": (
            "ThreeGPP38901 is an AODT antenna element name. This stage does "
            "not claim that the AODT YAML or raw tensors are 3GPP conformance evidence."
        ),
    }
    output_config = {
        "schema_version": "aodt-static-resolved-output-v1",
        "generated_at_utc": utc_now(),
        "transport": "official dt_client.DigitalTwinClient direct gRPC",
        "tables_requested": ["CIRS", "CFRS", "RAYPATHS"],
        "parquet_export": False,
        "s3_credentials_embedded": False,
        "s3_credential_placeholders": [
            "${AODT_S3_ACCESS_KEY}",
            "${AODT_S3_SECRET_KEY}",
        ],
        "runtime_secret_injection": "06B memory only; never persisted or logged",
        "raw_collection": {
            "format": "compressed NPZ",
            "fields": [
                "values",
                "delays",
                "angles_of_departure",
                "angles_of_arrival",
            ],
            "slots": ctx.slots_spec,
            "ru_indices": list(range(8)),
            "ue_indices_per_ru": [list(range(50)) for _ in range(8)],
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
    }
    json_dump(stage_dir / "resolved_scene_config.json", scene_config)
    json_dump(stage_dir / "resolved_radio_config.json", radio_config)
    json_dump(stage_dir / "resolved_output_config.json", output_config)
    json_dump(
        stage_dir / "config_validation.json",
        {
            "generated_config": config_validation,
            "aodt_loader_roundtrip": roundtrip_validation,
        },
    )
    markdown_write(
        stage_dir / "config_resolution_report.md",
        "04B AODT 1.5 config resolution",
        [
            f"- Simulation ID: `{sim_id}`",
            f"- Official AODT loader round-trip: {roundtrip_validation['status']}.",
            "- RU IDs 1–8 and UE IDs 1–50 are explicit; tensor indices remain 0-based.",
            "- Each UE has exactly one waypoint with speed 0.0 m/s.",
            f"- Requested max_depth={source_ray_interactions} maps to "
            f"AODT sim_em_interactions={aodt_ray_interactions}; "
            f"AODT 1.5 Worker maximum is {AODT_MAX_RAY_INTERACTIONS}.",
            "- Vegetation is enabled using the scene-derived worker GeoJSON path.",
            "- Parquet export is disabled; AODT-required S3 credentials remain "
            "environment placeholders and raw arrays are returned by direct gRPC.",
            "- Worker-side material assets remain an explicit 05B online gate.",
        ],
    )
    status = {
        "stage": "04B",
        "status": "pass",
        "generated_at_utc": utc_now(),
        "yaml": path_text(yaml_path),
        "yaml_sha256": sha256_file(yaml_path),
        "aodt_loader_roundtrip": "pass",
        "ru_count": 8,
        "ue_count": 50,
        "slots": int(meta["time_steps"]),
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


def stage_05b(ctx: PipelineContext) -> dict[str, Any]:
    stage_dir = create_stage_dir(ctx, "05B")
    paths: dict[str, Path] = {}
    import_error = None
    imported = False
    loader_roundtrip = False
    client_api = []
    try:
        paths = configure_aodt_imports(ctx.aodt_client_root)
        import dt_client  # type: ignore
        from _config import SimConfig  # type: ignore

        imported = True
        loader_roundtrip = bool(
            SimConfig.from_yaml_file(
                str(
                    (
                        ctx.stage_dir("04B")
                        / "resolved_aodt_static_config.yml"
                    ).resolve()
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
    except Exception as exc:  # binding failures must be recorded in 05B
        import_error = f"{type(exc).__name__}: {exc}"

    assets = yaml.safe_load(ctx.asset_config.read_text(encoding="utf-8"))
    asset_keys_valid = isinstance(assets, dict) and all(
        isinstance(assets.get(key), str) and assets[key] for key in ASSET_KEYS
    )
    asset_refs = {
        key: {
            "worker_path": assets.get(key) if isinstance(assets, dict) else None,
            "exists_on_windows_client": bool(
                isinstance(assets, dict)
                and isinstance(assets.get(key), str)
                and (ctx.aodt_client_root / assets[key]).exists()
            ),
        }
        for key in ASSET_KEYS
    }

    endpoint_valid = False
    tcp_reachable = False
    grpc_connected = False
    transport_mode = None
    client_has_gpu = None
    server_status = None
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
                    grpc_connected = bool(
                        aodt_member_value(client, "is_connected")
                    )
                    transport_mode = aodt_member_value(
                        client, "transport_mode"
                    )
                    client_has_gpu = bool(
                        aodt_member_value(client, "has_gpu")
                    )
                    server_status = client.get_status()
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
        "worker_assets_operator_confirmed": ctx.ack_worker_assets_resolved,
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

    binding_files = {}
    for name, path in paths.items():
        binding_files[name] = path_text(path)
    for label, relative in (
        ("dt_client_binding", Path("client/build/Release/dt_client.cp312-win_amd64.pyd")),
        (
            "config_binding",
            Path("client/build/config/Release/_config.cp312-win_amd64.pyd"),
        ),
    ):
        path = ctx.aodt_client_root / relative
        binding_files[label] = {
            "path": path_text(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    environment = {
        "schema_version": "aodt-static-environment-inventory-v1",
        "generated_at_utc": utc_now(),
        "python": {
            "executable": path_text(Path(sys.executable)),
            "version": sys.version,
        },
        "aodt_client": {
            "root": path_text(ctx.aodt_client_root),
            "git_commit": _git_value(ctx.aodt_client_root, "rev-parse", "HEAD"),
            "git_branch": _git_value(
                ctx.aodt_client_root, "rev-parse", "--abbrev-ref", "HEAD"
            ),
            "git_status_porcelain": _git_value(
                ctx.aodt_client_root, "status", "--porcelain"
            ),
            "bindings": binding_files,
            "required_api_found": sorted(client_api),
            "import_error": import_error,
        },
        "worker_assets": {
            "config": path_text(ctx.asset_config),
            "references": asset_refs,
            "operator_confirmed_resolved": ctx.ack_worker_assets_resolved,
            "note": "Paths are expected to resolve on the Azure worker.",
        },
        "runtime_secrets": {
            "AODT_S3_ACCESS_KEY_present": bool(
                os.environ.get("AODT_S3_ACCESS_KEY")
            ),
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
            "client_has_gpu": client_has_gpu,
            "client_gpu_note": (
                "Informational only. The Windows REMOTE client is allowed to "
                "be CPU-only; this API does not prove the Azure worker GPU model."
            ),
            "server_status_before_scenario_start": server_status,
            "error": online_error,
        },
        "secret_values_recorded": False,
    }
    status = {
        "schema_version": "aodt-static-preflight-v1",
        "stage": "05B",
        "generated_at_utc": utc_now(),
        "status": preflight_status,
        "offline_requested": ctx.offline,
        "checks": checks,
        "blockers": [
            name for name, passed in checks.items() if not passed
        ],
        "execution_authorized": preflight_status == "pass",
        "transport_contract": "official DigitalTwinClient direct gRPC; no SSH",
    }
    json_dump(stage_dir / "environment_inventory.json", environment)
    json_dump(stage_dir / "preflight_status.json", status)
    markdown_write(
        stage_dir / "preflight_report.md",
        "05B AODT direct-gRPC preflight",
        [
            f"- Status: **{preflight_status}**",
            f"- Official client binding import: {imported}.",
            f"- AODT YAML loader round-trip: {loader_roundtrip}.",
            f"- Worker asset confirmation: {ctx.ack_worker_assets_resolved}.",
            f"- Endpoint configured: {endpoint_valid}.",
            f"- Direct gRPC connected in REMOTE mode: "
            f"{grpc_connected and transport_mode == 'REMOTE'}.",
            f"- Windows client reports local GPU capability: {client_has_gpu} "
            "(informational, not a gate).",
            "- 06B and 07B are prohibited unless every online gate passes.",
        ],
    )
    return status
