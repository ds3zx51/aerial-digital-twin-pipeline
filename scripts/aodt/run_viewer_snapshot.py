#!/usr/bin/env python3
"""Prepare or execute a one-slot AODT raypath export for the native Viewer.

The scientific 00B-08B run intentionally persisted raw CIR arrays over gRPC.
That data does not contain the interaction-point geometry required by the AODT
Viewer.  This companion workflow reuses the exact resolved topology and radio
configuration, changes only the timeline to one slot, and enables a full
raypaths Parquet/Iceberg export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ACCESS_PLACEHOLDER = "${AODT_S3_ACCESS_KEY}"
SECRET_PLACEHOLDER = "${AODT_S3_SECRET_KEY}"


class ViewerSnapshotError(RuntimeError):
    """Terminal error for the Viewer snapshot workflow."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def path_text(path: Path) -> str:
    return path.resolve().as_posix()


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _latest_base_run(repo_root: Path) -> Path:
    root = repo_root / "results" / "aodt_static"
    candidates: list[Path] = []
    if root.is_dir():
        for candidate in root.iterdir():
            if (
                candidate.is_dir()
                and (
                    candidate
                    / "04B_config"
                    / "resolved_aodt_static_config.yml"
                ).is_file()
                and (
                    candidate / "08B_raw_validation" / "stage_status.json"
                ).is_file()
            ):
                try:
                    status = json.loads(
                        (
                            candidate
                            / "08B_raw_validation"
                            / "stage_status.json"
                        ).read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if status.get("status") == "pass":
                    candidates.append(candidate)
    if not candidates:
        raise ViewerSnapshotError(
            f"No passing 08B AODT run was found under {root}."
        )
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _create_run_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"tokyo_aodt_viewer_snapshot_{stamp}"
    for index in range(1000):
        candidate = results_root / f"{stem}_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise ViewerSnapshotError("Unable to allocate a unique snapshot run directory.")


def _scenario_attributes(config: dict[str, Any]) -> dict[str, Any]:
    try:
        updates = config["sim"]["Scenario"]["update"]
        attributes = updates[0]["attributes"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ViewerSnapshotError(
            "Base YAML has no sim.Scenario.update[0].attributes."
        ) from exc
    if not isinstance(attributes, dict):
        raise ViewerSnapshotError("Scenario attributes are not a mapping.")
    return attributes


def _prepare_config(base: dict[str, Any], sim_id: str) -> dict[str, Any]:
    # Round-trip through YAML to obtain a deep copy containing only plain data.
    config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    if not isinstance(config, dict):
        raise ViewerSnapshotError("Base YAML root is not a mapping.")

    db = config.setdefault("db", {})
    if not isinstance(db, dict):
        raise ViewerSnapshotError("Base YAML db section is not a mapping.")
    db["sim_id"] = sim_id
    db["db_author"] = "aodt-static-pipeline"
    db["db_notes"] = (
        "Viewer snapshot: 8 RU, 50 fixed terrestrial UE, one slot, "
        "full raypaths exported to Parquet/Iceberg"
    )
    db["opt_in_tables"] = ["raypaths"]
    db["opt_in_tables_options"] = {"raypaths": "full"}
    db["s3_config"] = {
        "access_key": ACCESS_PLACEHOLDER,
        "bucket": "aerial-data",
        "endpoint_url": "http://minio:9000",
        "provider": "minio",
        "region": "us-east-1",
        "secret_key": SECRET_PLACEHOLDER,
    }
    db["parquet_export"] = {
        "compression": "zstd",
        "iceberg": {
            "catalog_name": "default",
            "catalog_type": "rest",
            "catalog_uri": "http://nessie:19120/iceberg",
            "nessie_ref": "main",
        },
        "max_workers": 2,
        "s3_configs": [
            {
                "access_key": ACCESS_PLACEHOLDER,
                "bucket": "aerial-data",
                "endpoint_url": "http://minio:9000",
                "nodes": ["node1"],
                "provider": "minio",
                "region": "us-east-1",
                "secret_key": SECRET_PLACEHOLDER,
                "use_ssl": False,
            }
        ],
        "timesteps_per_file": 1,
        "verify_exports": True,
    }

    attributes = _scenario_attributes(config)
    attributes["sim_batches"] = 1
    attributes["sim_slots_per_batch"] = 1
    attributes["sim_samples_per_slot"] = 1
    return config


def _node_ids(config: dict[str, Any], section: str) -> list[int]:
    try:
        records = config["sim"][section]["add"]
    except (KeyError, TypeError) as exc:
        raise ViewerSnapshotError(f"Base YAML has no sim.{section}.add.") from exc
    if not isinstance(records, list):
        raise ViewerSnapshotError(f"sim.{section}.add is not a list.")
    return sorted(int(record["id"]) for record in records)


def _validate_prepared(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    db = config.get("db", {})
    if _node_ids(config, "RUs") != list(range(1, 9)):
        errors.append("RU IDs are not exactly 1..8.")
    if _node_ids(config, "UEs") != list(range(1, 51)):
        errors.append("UE IDs are not exactly 1..50.")

    ue_add = config["sim"]["UEs"]["add"]
    waypoint_counts = [len(item.get("waypoints", [])) for item in ue_add]
    speeds = [
        float(waypoint.get("speed", 0.0))
        for item in ue_add
        for waypoint in item.get("waypoints", [])
    ]
    if set(waypoint_counts) != {1}:
        errors.append("Every fixed UE must have exactly one waypoint.")
    if any(speed != 0.0 for speed in speeds):
        errors.append("Every UE waypoint speed must be zero.")

    attributes = _scenario_attributes(config)
    if int(attributes.get("sim_batches", -1)) != 1:
        errors.append("sim_batches is not 1.")
    if int(attributes.get("sim_slots_per_batch", -1)) != 1:
        errors.append("sim_slots_per_batch is not 1.")
    if int(attributes.get("sim_samples_per_slot", -1)) != 1:
        errors.append("sim_samples_per_slot is not 1.")
    if db.get("opt_in_tables") != ["raypaths"]:
        errors.append("Only the raypaths table must be selected.")
    if db.get("opt_in_tables_options", {}).get("raypaths") != "full":
        errors.append("raypaths export is not full.")

    parquet = db.get("parquet_export", {})
    if int(parquet.get("timesteps_per_file", -1)) != 1:
        errors.append("Parquet timesteps_per_file is not 1.")
    if parquet.get("compression") != "zstd":
        errors.append("Parquet compression is not zstd.")
    iceberg = parquet.get("iceberg", {})
    if iceberg.get("catalog_uri") != "http://nessie:19120/iceberg":
        errors.append("Nessie/Iceberg catalog URI differs from the worker stack.")
    if parquet.get("verify_exports") is not True:
        errors.append("Parquet export verification is disabled.")

    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if ACCESS_PLACEHOLDER not in rendered or SECRET_PLACEHOLDER not in rendered:
        errors.append("Credential placeholders are missing.")
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "ru_count": len(_node_ids(config, "RUs")),
        "ue_count": len(_node_ids(config, "UEs")),
        "fixed_ue_waypoints": set(waypoint_counts) == {1},
        "all_ue_speeds_zero": all(speed == 0.0 for speed in speeds),
        "slots": attributes.get("sim_slots_per_batch"),
        "tables": db.get("opt_in_tables"),
        "raypaths_option": db.get("opt_in_tables_options", {}).get("raypaths"),
        "parquet_export": bool(parquet),
        "iceberg_catalog_uri": iceberg.get("catalog_uri"),
        "credentials_persisted": False,
    }


def _configure_aodt_imports(aodt_root: Path) -> None:
    paths = [
        aodt_root / "client" / "build" / "Release",
        aodt_root / "client" / "build" / "config" / "Release",
    ]
    for path in paths:
        if not path.is_dir():
            raise ViewerSnapshotError(f"AODT binding directory is missing: {path}")
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(value)
            except OSError:
                pass


def _official_roundtrip(yaml_path: Path, aodt_root: Path) -> dict[str, Any]:
    _configure_aodt_imports(aodt_root)
    try:
        from _config import SimConfig  # type: ignore
    except ImportError as exc:
        raise ViewerSnapshotError(
            f"Unable to import the official AODT config binding: {exc}"
        ) from exc
    loaded = SimConfig.from_yaml_file(str(yaml_path.resolve())).to_dict()
    validation = _validate_prepared(loaded)
    if validation["status"] != "pass":
        raise ViewerSnapshotError(
            "Official SimConfig round-trip failed: "
            + "; ".join(validation["errors"])
        )
    return validation


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _wait_for_automatic_export(
    server_log_path: Path, timeout_seconds: float = 10.0
) -> dict[str, Any] | None:
    """Read the Parquet export result emitted by run_full_simulation().

    AODT 1.5 automatically flushes Parquet when parquet_export is configured,
    registers the Iceberg table, and then clears the ClickHouse result tables.
    Calling ExportResults after that valid automatic flush returns zero rows.
    """

    deadline = time.monotonic() + timeout_seconds
    pattern = re.compile(
        r"Parquet export complete:\s*(\d+)/(\d+) files,\s*(\d+) rows"
    )
    while time.monotonic() < deadline:
        try:
            text = server_log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        matches = list(pattern.finditer(text))
        if matches:
            match = matches[-1]
            files_exported = int(match.group(1))
            files_expected = int(match.group(2))
            total_rows = int(match.group(3))
            raypaths_table = bool(
                re.search(r"Created Iceberg table .+\.raypaths", text)
            )
            raypaths_object = re.search(
                r"Added (s3://\S+/raypaths/data/\S+\.parquet) "
                r"to Iceberg table",
                text,
            )
            return {
                "mode": "automatic_during_run_full_simulation",
                "files_exported": files_exported,
                "files_expected": files_expected,
                "total_rows": total_rows,
                "iceberg_raypaths_table_registered": raypaths_table,
                "raypaths_object": (
                    raypaths_object.group(1) if raypaths_object else None
                ),
            }
        time.sleep(0.25)
    return None


def _execute(
    yaml_path: Path,
    run_dir: Path,
    endpoint: str,
    aodt_root: Path,
) -> dict[str, Any]:
    if not endpoint:
        raise ViewerSnapshotError(
            "--endpoint or DT_SERVER_ADDRESS is required with --execute."
        )
    access_key = os.environ.get("AODT_S3_ACCESS_KEY", "")
    secret_key = os.environ.get("AODT_S3_SECRET_KEY", "")
    if not access_key or not secret_key:
        raise ViewerSnapshotError(
            "AODT_S3_ACCESS_KEY and AODT_S3_SECRET_KEY are required with --execute."
        )

    _configure_aodt_imports(aodt_root)
    try:
        import dt_client  # type: ignore
    except ImportError as exc:
        raise ViewerSnapshotError(
            f"Unable to import the official AODT client binding: {exc}"
        ) from exc

    yaml_content = yaml_path.read_text(encoding="utf-8")
    yaml_content = yaml_content.replace(ACCESS_PLACEHOLDER, access_key)
    yaml_content = yaml_content.replace(SECRET_PLACEHOLDER, secret_key)
    if "${AODT_S3_" in yaml_content:
        raise ViewerSnapshotError("An unresolved S3 credential placeholder remains.")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        with stdout_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    client = None
    streaming = False
    started = time.monotonic()
    try:
        log(f"Connecting to {endpoint} with official direct gRPC.")
        client = dt_client.DigitalTwinClient(endpoint, force=False)
        if not bool(client.is_connected):
            raise ViewerSnapshotError("DigitalTwinClient reports is_connected=false.")
        if str(client.transport_mode) != "REMOTE":
            raise ViewerSnapshotError(
                f"Expected REMOTE transport, got {client.transport_mode}."
            )

        server_log_path = run_dir / "server.log"
        streaming = bool(
            client.start_server_log_streaming(
                str(server_log_path.resolve()), "INFO"
            )
        )
        log("Submitting one-slot full-raypaths AODT YAML.")
        if not client.start(yaml_content):
            raise ViewerSnapshotError("DigitalTwinClient.start returned false.")
        server_status = _plain(client.get_status())
        expected = {
            "scenario_loaded": True,
            "num_rus": 8,
            "num_ues": 50,
            "total_batches": 1,
            "is_slot_symbol_mode": True,
            "num_slots_or_timesteps_per_batch": 1,
        }
        mismatches = {
            key: {"actual": server_status.get(key), "expected": value}
            for key, value in expected.items()
            if server_status.get(key) != value
        }
        if mismatches:
            raise ViewerSnapshotError(
                "Server status gate failed: " + json.dumps(mismatches)
            )
        json_write(run_dir / "server_status_after_start.json", server_status)

        log("Running the full 1x1 server-side AODT simulation.")
        simulation_result = _plain(client.run_full_simulation())
        if int(simulation_result.get("time_steps_completed", -1)) != 1:
            raise ViewerSnapshotError(
                "One-slot simulation did not complete exactly one time step."
            )

        export_result = (
            _wait_for_automatic_export(server_log_path) if streaming else None
        )
        if export_result is None:
            log(
                "No automatic export evidence was observed; requesting an "
                "on-demand raypaths export."
            )
            export_result = _plain(client.export_results(["raypaths"]))
            export_result["mode"] = "on_demand_export_results_rpc"
        if int(export_result.get("files_exported", 0)) < 1:
            raise ViewerSnapshotError("AODT exported zero raypath files.")
        if int(export_result.get("total_rows", 0)) < 1:
            raise ViewerSnapshotError("AODT exported zero raypath rows.")
        if (
            export_result.get("mode") == "automatic_during_run_full_simulation"
            and not export_result.get("iceberg_raypaths_table_registered")
        ):
            raise ViewerSnapshotError(
                "Parquet file was exported but the Iceberg raypaths table "
                "registration was not observed."
            )

        status = {
            "schema_version": "aodt-viewer-snapshot-execution-v1",
            "status": "pass",
            "generated_at_utc": utc_now(),
            "endpoint": endpoint,
            "transport": "official dt_client.DigitalTwinClient direct gRPC",
            "server_status": server_status,
            "simulation_result": simulation_result,
            "export_result": export_result,
            "elapsed_seconds": time.monotonic() - started,
            "credentials_persisted": False,
        }
        json_write(run_dir / "execution_status.json", status)
        log(
            "Viewer snapshot export passed: "
            f"{export_result['files_exported']} files, "
            f"{export_result['total_rows']} rows "
            f"({export_result['mode']})."
        )
        return status
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        status = {
            "schema_version": "aodt-viewer-snapshot-execution-v1",
            "status": "fail",
            "generated_at_utc": utc_now(),
            "endpoint": endpoint,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            "credentials_persisted": False,
        }
        json_write(run_dir / "execution_status.json", status)
        raise
    finally:
        if client is not None and streaming:
            try:
                client.stop_server_log_streaming()
            except Exception:
                pass


def _write_viewer_guide(run_dir: Path, sim_id: str) -> None:
    text = f"""# AODT native Viewer snapshot

Simulation/database ID: `{sim_id}`

This one-slot run preserves the passing 00B-08B topology and radio settings:
8 RUs, 50 fixed terrestrial UEs, no UAV/GPX/mobility.  It exports only
`raypaths` with the `full` option so the native Viewer can draw interaction
points as received-power-colored 3D polylines.

After the live export passes:

1. Make MinIO port 9000 and Nessie port 19120 reachable from the Viewer host
   using a restricted connection method.
2. Start the local Viewer with `npm.cmd run dev` from the `viewer` directory
   inside your `AODT_CLIENT_ROOT` checkout.
3. Load the generated YAML or configure the catalog as
   `http://localhost:19120/iceberg` and MinIO as `http://localhost:9000`
   when local forwarding is used.
4. Select database `{sim_id}`, load the raypaths, select time index 0, and
   filter the RU selector to AODT RU ID 8 (scenario RU7).
5. Set the desired dynamic range/camera and capture the paper figure.

The Viewer image is a native AODT raypath visualization in dBm.  It is not a
dense ground-plane RadioMap and should not be labeled RSRP/SINR unless those
metrics are computed separately.
"""
    (run_dir / "VIEWER_STEPS.md").write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    repo = _default_repo()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or execute a one-slot AODT full-raypaths export for "
            "the native Viewer."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--base-run-dir", type=Path)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo / "results" / "aodt_viewer_snapshot",
    )
    parser.add_argument(
        "--aodt-client-root",
        type=Path,
        default=(
            Path(os.environ["AODT_CLIENT_ROOT"])
            if os.environ.get("AODT_CLIENT_ROOT")
            else None
        ),
        help="AODT checkout; defaults to AODT_CLIENT_ROOT.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("DT_SERVER_ADDRESS", ""),
        help="AODT worker host:port. Required only with --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Contact the AODT worker, run one slot, and export raypaths. "
            "Without this flag the command is offline and cannot incur Azure cost."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir: Path | None = None
    try:
        repo_root = args.repo_root.resolve()
        base_run = (
            args.base_run_dir.resolve()
            if args.base_run_dir
            else _latest_base_run(repo_root)
        )
        base_yaml = (
            base_run / "04B_config" / "resolved_aodt_static_config.yml"
        )
        if not base_yaml.is_file():
            raise ViewerSnapshotError(f"Base AODT YAML does not exist: {base_yaml}")
        if args.aodt_client_root is None or not args.aodt_client_root.is_dir():
            raise ViewerSnapshotError(
                f"AODT client repository does not exist: {args.aodt_client_root}"
            )

        run_dir = _create_run_dir(args.results_root.resolve())
        sim_id = (
            "tokyo_fixed_8ru_50ue_viewer_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        base_config = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
        config = _prepare_config(base_config, sim_id)
        validation = _validate_prepared(config)
        if validation["status"] != "pass":
            raise ViewerSnapshotError(
                "Prepared Viewer config failed validation: "
                + "; ".join(validation["errors"])
            )

        yaml_path = run_dir / "resolved_aodt_viewer_snapshot.yml"
        yaml_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        official_validation = _official_roundtrip(
            yaml_path, args.aodt_client_root.resolve()
        )
        _write_viewer_guide(run_dir, sim_id)
        manifest = {
            "schema_version": "aodt-viewer-snapshot-manifest-v1",
            "status": "prepared",
            "generated_at_utc": utc_now(),
            "base_run_dir": path_text(base_run),
            "base_yaml": path_text(base_yaml),
            "base_yaml_sha256": sha256_file(base_yaml),
            "snapshot_yaml": path_text(yaml_path),
            "snapshot_yaml_sha256": sha256_file(yaml_path),
            "sim_id": sim_id,
            "execution_requested": bool(args.execute),
            "transport": "official dt_client.DigitalTwinClient direct gRPC",
            "scope": {
                "rus": 8,
                "ues": 50,
                "ue_mobility": "fixed",
                "slots": 1,
                "tables": ["raypaths"],
                "raypaths_option": "full",
            },
            "validation": validation,
            "official_simconfig_roundtrip": official_validation,
            "credential_policy": (
                "YAML stores placeholders only; live secrets are substituted "
                "in memory and are never written or logged."
            ),
            "cost_boundary": (
                "Preparation is offline. Azure/A100 cost begins only after the "
                "user starts the worker VM and invokes this script with --execute."
            ),
        }
        json_write(run_dir / "snapshot_manifest.json", manifest)

        if args.execute:
            execution = _execute(
                yaml_path,
                run_dir,
                args.endpoint.strip(),
                args.aodt_client_root.resolve(),
            )
            manifest["status"] = execution["status"]
            manifest["execution"] = execution
            json_write(run_dir / "snapshot_manifest.json", manifest)

        print(f"RUN_DIR={path_text(run_dir)}", flush=True)
        print(f"SIM_ID={sim_id}", flush=True)
        print(
            "OVERALL_STATUS="
            + ("pass" if args.execute else "prepared_offline_no_azure_cost"),
            flush=True,
        )
        return 0
    except Exception as exc:
        if run_dir is not None:
            try:
                json_write(
                    run_dir / "failure.json",
                    {
                        "status": "fail",
                        "generated_at_utc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                print(f"RUN_DIR={path_text(run_dir)}", file=sys.stderr)
            except Exception:
                pass
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
