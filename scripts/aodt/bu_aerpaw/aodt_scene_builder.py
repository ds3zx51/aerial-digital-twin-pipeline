"""Deterministic AODT OSM smoke-scene builder for AERPAW Lake Wheeler.

The formal paper-reconstruction scene cannot be built from OSMTask alone because the
satellite-guided supplementary foliage layer is not available through that API. Formal mode
therefore fails closed and expects an externally prepared, fully inventoried scene bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def add_bindings(client_root: Path) -> None:
    for path in reversed((client_root / "build" / "Release", client_root / "build" / "config" / "Release")):
        sys.path.insert(0, str(path))


def task_spec(args: argparse.Namespace) -> dict[str, Any]:
    west, south, east, north = args.bbox
    if not (west < east and south < north):
        raise ValueError("--bbox order is west south east north and must have positive extent")
    return {
        "schema_version": "1.0.0",
        "profile": "osm_only_smoke",
        "formal_eligible": False,
        "paper_reference": "DOI 10.1109/OJVT.2025.3585537",
        "bbox_order": "west,south,east,north",
        "bbox_epsg4326": args.bbox,
        "output_folder_key": args.output_folder_key,
        "include_elevation": True,
        "ground_source": "terrarium",
        "vegetation_source": "procedural",
        "vegetation_density_trees_per_hectare": args.vegetation_density,
        "vegetation_scale_min": 1.0,
        "vegetation_scale_max": 1.0,
        "rough": True,
        "disable_interiors": True,
        "terrain_clip_margin_m": 200.0,
        "random_seed_recorded": 2025,
        "limitations": [
            "OSMTask procedural foliage is not the paper's satellite-guided supplementary foliage reconstruction",
            "AODT OSMTask does not expose ingestion of the required manual foliage GeoJSON",
            "This profile can never satisfy BU07 formal scene gate",
        ],
    }


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def execute(spec: dict[str, Any], client_root: Path, server_address: str) -> dict[str, Any]:
    add_bindings(client_root)
    import dt_client
    from _config import S3Config

    terraform = dt_client.TerraformConfig(terraform=True)
    task = dt_client.OSMTask(
        output_folder_key=spec["output_folder_key"],
        coords=tuple(spec["bbox_epsg4326"]),
        include_elevation=True,
        ground_source="terrarium",
        vegetation_source="procedural",
        vegetation_density=spec["vegetation_density_trees_per_hectare"],
        vegetation_scale_min=1.0,
        vegetation_scale_max=1.0,
        cesium3dtiles_b3dm=None,
        cesium3dtiles_draco=None,
        cesium3dtiles_gzip=None,
        cesium3dtiles_chunk_size=None,
        cesium3dtiles_veg_instanced=False,
        rough=True,
        disable_interiors=True,
        terrain_clip_margin=200.0,
        terraform_config=terraform,
    )
    s3 = S3Config(
        bucket=_required_env("AODT_S3_BUCKET"),
        provider=os.environ.get("AODT_S3_PROVIDER", "minio"),
        endpoint_url=_required_env("AODT_S3_ENDPOINT"),
        access_key=_required_env("AODT_S3_ACCESS_KEY"),
        secret_key=_required_env("AODT_S3_SECRET_KEY"),
        region=os.environ.get("AODT_S3_REGION", "us-east-1"),
    )
    client = dt_client.DigitalTwinClient(server_address)
    result = dict(client.prepare_map(task, s3))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "success": bool(result.get("success")),
        "result": result,
        "credentials_serialized": False,
        "formal_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AERPAW Lake Wheeler OSM-only AODT smoke scene")
    parser.add_argument("--profile", choices=["osm_only_smoke", "paper_reconstruction"], default="osm_only_smoke")
    bbox_group = parser.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    bbox_group.add_argument("--geospatial-contract", type=Path, help="BU03 geospatial_contract.json containing scene_bbox_wgs84")
    parser.add_argument("--output-folder-key", default="aerpaw/lake_wheeler_osm_smoke")
    parser.add_argument("--vegetation-density", type=float, default=100.0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--server-address", default=os.environ.get("DT_SERVER_ADDRESS", ""))
    parser.add_argument("--aodt-client-root", type=Path, default=Path("C:/aerial-omniverse-digital-twin/client"))
    args = parser.parse_args()
    if args.geospatial_contract:
        contract = json.loads(args.geospatial_contract.read_text(encoding="utf-8"))
        args.bbox = [float(value) for value in contract["scene_bbox_wgs84"]["values"]]
    if args.profile == "paper_reconstruction":
        print("BLOCKED: paper_reconstruction requires external formal scene bundle and supplementary foliage GeoJSON", file=sys.stderr)
        return 2
    spec = task_spec(args)
    output: dict[str, Any] = {"task": spec, "submitted": False}
    if args.execute:
        if not args.server_address:
            print("BLOCKED: --execute requires --server-address or DT_SERVER_ADDRESS", file=sys.stderr)
            return 2
        output.update({"submitted": True, "execution": execute(spec, args.aodt_client_root, args.server_address)})
    write_json(args.manifest, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
