"""Online AODT execution and raw collection for stages 06BU and 07BU."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from uav_core import (
    EXPECTED_GROUND_UE_COUNT,
    EXPECTED_RU_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_UE_COUNT,
    EXPECTED_UAV_COUNT,
    PipelineError,
    UAVPipelineContext,
    _trajectory_paths,
    aodt_member_value,
    configure_aodt_imports,
    create_stage_dir,
    json_dump,
    json_load,
    markdown_write,
    path_text,
    sha256_file,
    utc_now,
)


PAYLOAD_FIELDS = (
    "values",
    "delays",
    "angles_of_departure",
    "angles_of_arrival",
)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_log(path: Path, line: str) -> None:
    timestamped = f"{utc_now()} {line}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")
    print(timestamped, flush=True)


def _validate_server_status(status: dict[str, Any]) -> None:
    errors = []
    if not status.get("scenario_loaded"):
        errors.append("scenario_loaded is false")
    if int(status.get("num_rus", -1)) != EXPECTED_RU_COUNT:
        errors.append(f"num_rus={status.get('num_rus')} (expected 8)")
    if int(status.get("num_ues", -1)) != EXPECTED_UE_COUNT:
        errors.append(f"num_ues={status.get('num_ues')} (expected 54)")
    if status.get("is_slot_symbol_mode"):
        errors.append("worker is in slot/symbol mode; BU requires duration/interval")
    if int(status.get("num_slots_or_timesteps_per_batch", -1)) != EXPECTED_SAMPLE_COUNT:
        errors.append(
            "num_slots_or_timesteps_per_batch="
            f"{status.get('num_slots_or_timesteps_per_batch')} (expected 256)"
        )
    if errors:
        raise PipelineError("AODT BU server status gate failed: " + "; ".join(errors))


def _all_positions(
    client: Any, time_step_index_type: Any
) -> tuple[list[list[list[float]]], list[list[float]]]:
    ru_positions = _plain(client.get_ru_positions())
    ue_positions = []
    for time_step in range(EXPECTED_SAMPLE_COUNT):
        ue_positions.append(
            _plain(
                client.get_ue_positions(
                    batch_index=0,
                    temporal_index=time_step_index_type(time_step),
                )
            )
        )
    return ue_positions, ru_positions


def _validate_positions(
    ctx: UAVPipelineContext,
    ue_positions: list[list[list[float]]],
    ru_positions: list[list[float]],
) -> dict[str, Any]:
    scenario = json_load(ctx.scenario_path)
    canonical = json_load(
        _trajectory_paths(ctx.repo_root, ctx.scenario_path)["canonical"]
    )
    actual = np.asarray(ue_positions, dtype=np.float64)
    ru_actual = np.asarray(ru_positions, dtype=np.float64)
    if actual.shape != (EXPECTED_SAMPLE_COUNT, EXPECTED_UE_COUNT, 3):
        raise PipelineError(f"Unexpected UE position tensor shape: {actual.shape}")
    if ru_actual.shape != (EXPECTED_RU_COUNT, 3):
        raise PipelineError(f"Unexpected RU position tensor shape: {ru_actual.shape}")
    expected_ground = np.asarray(
        [[node["x"], node["y"], node["z"]] for node in scenario["ue"]],
        dtype=np.float64,
    )
    expected_ru = np.asarray(
        [[node["x"], node["y"], node["z"]] for node in scenario["bs"]],
        dtype=np.float64,
    )
    expected_uav = np.asarray(
        [
            [
                [
                    point["local_x_m"],
                    point["local_y_m"],
                    point["local_z_m"],
                ]
                for point in route["points"]
            ]
            for route in canonical["routes"]
        ],
        dtype=np.float64,
    ).transpose(1, 0, 2)
    conversion = json_load(
        ctx.repo_root
        / "configs"
        / "geospatial"
        / "tokyo_aodt_scene_contract.json"
    )
    meters_per_stage_unit = float(conversion["meters_per_unit"])
    expected_ground_aodt = expected_ground * meters_per_stage_unit
    expected_ru_aodt = expected_ru * meters_per_stage_unit
    # The passing fixed-B branch uses the default 2.5 m RU asset.  AODT
    # reports the asset centre, 1.25 m above the configured Cartesian base.
    expected_ru_aodt[:, 2] += 1.25
    finite = bool(np.isfinite(actual).all() and np.isfinite(ru_actual).all())
    ru_error = float(np.max(np.abs(ru_actual - expected_ru_aodt)))
    ground_error = float(
        np.max(
            np.abs(
                actual[:, :EXPECTED_GROUND_UE_COUNT, :]
                - expected_ground_aodt
            )
        )
    )
    uav_absolute_error = np.abs(
        actual[:, EXPECTED_GROUND_UE_COUNT:, :] - expected_uav
    )
    uav_error = float(np.max(uav_absolute_error))
    uav_xy_absolute_error = uav_absolute_error[:, :, :2]
    uav_xy_error = float(np.max(uav_xy_absolute_error))
    per_uav_error = [
        float(np.max(uav_absolute_error[:, index, :]))
        for index in range(EXPECTED_UAV_COUNT)
    ]
    per_uav_xy_error = [
        float(np.max(uav_xy_absolute_error[:, index, :]))
        for index in range(EXPECTED_UAV_COUNT)
    ]
    movement = np.linalg.norm(
        actual[-1, EXPECTED_GROUND_UE_COUNT:, :]
        - actual[0, EXPECTED_GROUND_UE_COUNT:, :],
        axis=1,
    )
    static_ground_delta = float(
        np.max(
            np.abs(
                actual[:, :EXPECTED_GROUND_UE_COUNT, :]
                - actual[0:1, :EXPECTED_GROUND_UE_COUNT, :]
            )
        )
    )
    tolerance = ctx.position_tolerance_m
    checks = {
        "all_positions_finite": finite,
        "ru_positions_match_fixed_b_aodt_coordinates": ru_error <= tolerance,
        "ground_ue_positions_match_fixed_b_aodt_coordinates": (
            ground_error <= tolerance
        ),
        "ground_ues_are_static": static_ground_delta <= tolerance,
        "uav_native_gpx_xy_matches_canonical_enu": uav_xy_error <= tolerance,
        "every_uav_moves": bool(np.all(movement > tolerance)),
    }
    return {
        "schema_version": "aodt-uav-position-validation-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tolerance_m": tolerance,
        "ru_max_abs_error_m": ru_error,
        "ground_ue_max_abs_error_m": ground_error,
        "ground_ue_max_temporal_delta_m": static_ground_delta,
        "uav_max_abs_error_m": uav_error,
        "uav_max_abs_error_m_by_route": per_uav_error,
        "uav_xy_max_abs_error_m": uav_xy_error,
        "uav_xy_max_abs_error_m_by_route": per_uav_xy_error,
        "gpx_elevation_preserved_by_aodt_native": False,
        "altitude_limitation": (
            "AODT 1.5 native GPX preserves the GPX latitude/longitude timeline "
            "but projects Z to the scene surface; GPX <ele> is not preserved."
        ),
        "uav_first_last_displacement_m": movement.tolist(),
        "actual_shape": list(actual.shape),
        "canonical_shape": list(expected_uav.shape),
        "trajectory_runtime_verified": checks[
            "uav_native_gpx_xy_matches_canonical_enu"
        ],
    }


def _probe_uav_cirs(
    client: Any,
    time_step_index_type: Any,
    probe_dir: Path,
) -> dict[str, Any]:
    probe_dir.mkdir()
    records = []
    for time_step in (0, EXPECTED_SAMPLE_COUNT - 1):
        allocation = None
        try:
            allocation = client.allocate_cirs_memory(
                [0],
                [[EXPECTED_GROUND_UE_COUNT]],
                is_full_antenna_pair=False,
                num_time_steps=1,
            )
            client.get_cirs(
                allocation,
                batch_index=0,
                temporal_index=time_step_index_type(time_step),
            )
            arrays = {
                name: np.asarray(
                    client.to_numpy(allocation, time_step, 0, name)
                )
                for name in PAYLOAD_FIELDS
            }
            if any(array.size == 0 for array in arrays.values()):
                raise PipelineError(
                    f"06BU UAV raw CIR probe is empty at time step {time_step}."
                )
            output = probe_dir / f"cir_timestep_{time_step:04d}_ru_00_uav_00.npz"
            np.savez_compressed(
                output,
                **arrays,
                batch=np.asarray(0, dtype=np.int64),
                timestep=np.asarray(time_step, dtype=np.int64),
                ru_tensor_index=np.asarray(0, dtype=np.int64),
                ru_scenario_id=np.asarray(0, dtype=np.int64),
                ru_aodt_id=np.asarray(1, dtype=np.int64),
                ue_tensor_indices=np.asarray([50], dtype=np.int64),
                ue_scenario_ids=np.asarray([50], dtype=np.int64),
                ue_aodt_ids=np.asarray([51], dtype=np.int64),
            )
            records.append(
                {
                    "time_step": time_step,
                    "path": path_text(output),
                    "sha256": sha256_file(output),
                    "arrays": {
                        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                        for name, array in arrays.items()
                    },
                }
            )
        finally:
            if allocation is not None:
                client.deallocate_cirs_memory(allocation)
    return {"status": "pass", "ue_tensor_index": 50, "probes": records}


def stage_06bu(
    ctx: UAVPipelineContext, keep_session_for_07bu: bool
) -> tuple[dict[str, Any], Any | None]:
    if ctx.offline:
        raise PipelineError("06BU cannot run in offline mode.")
    preflight = json_load(ctx.stage_dir("05BU") / "preflight_status.json")
    if preflight.get("status") != "pass" or not preflight.get(
        "execution_authorized"
    ):
        raise PipelineError("06BU is blocked because 05BU did not pass.")
    stage_dir = create_stage_dir(ctx, "06BU")
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    (stage_dir / "command.txt").write_text(ctx.command + "\n", encoding="utf-8")
    yaml_path = ctx.stage_dir("04BU") / "resolved_aodt_uav_config.yml"
    execution_manifest = {
        "schema_version": "aodt-uav-execution-manifest-v1",
        "generated_at_utc": utc_now(),
        "endpoint": ctx.endpoint,
        "transport": "official dt_client.DigitalTwinClient direct gRPC",
        "yaml": path_text(yaml_path),
        "yaml_sha256": sha256_file(yaml_path),
        "expected": {
            "ru": 8,
            "ue": 54,
            "ground_ue": 50,
            "uav": 4,
            "time_steps": 256,
            "interval_s": 0.25,
            "temporal_mode": "duration_interval",
            "temporal_index_type": "TimeStepIndex",
        },
        "fallback_policy": "none; failure is terminal",
    }
    json_dump(stage_dir / "execution_manifest.json", execution_manifest)
    configure_aodt_imports(ctx.aodt_client_root)
    import dt_client  # type: ignore
    from dt_client import TimeStepIndex  # type: ignore

    client = None
    log_streaming = False
    started_at = time.monotonic()
    status_doc: dict[str, Any] = {
        "schema_version": "aodt-uav-execution-status-v1",
        "stage": "06BU",
        "generated_at_utc": utc_now(),
        "status": "running",
    }
    try:
        _append_log(stdout_path, f"Connecting to {ctx.endpoint} via direct gRPC.")
        client = dt_client.DigitalTwinClient(ctx.endpoint, force=False)
        if not bool(aodt_member_value(client, "is_connected")):
            raise PipelineError("DigitalTwinClient reports is_connected=false.")
        if aodt_member_value(client, "transport_mode") != "REMOTE":
            raise PipelineError("DigitalTwinClient transport is not REMOTE.")
        log_streaming = bool(
            client.start_server_log_streaming(
                str((stage_dir / "server.log").resolve()), "INFO"
            )
        )
        yaml_content = yaml_path.read_text(encoding="utf-8")
        access_key = os.environ.get("AODT_S3_ACCESS_KEY", "")
        secret_key = os.environ.get("AODT_S3_SECRET_KEY", "")
        if not access_key or not secret_key:
            raise PipelineError(
                "AODT_S3_ACCESS_KEY and AODT_S3_SECRET_KEY are required at runtime."
            )
        yaml_content = yaml_content.replace(
            "${AODT_S3_ACCESS_KEY}", access_key
        ).replace("${AODT_S3_SECRET_KEY}", secret_key)
        if "${AODT_S3_" in yaml_content:
            raise PipelineError("An unresolved AODT S3 placeholder remains.")
        _append_log(stdout_path, "Submitting the 54-UE AODT GPX YAML.")
        if not client.start(yaml_content):
            raise PipelineError("DigitalTwinClient.start returned false.")
        server_status = _plain(client.get_status())
        _validate_server_status(server_status)
        json_dump(stage_dir / "server_status_after_start.json", server_status)
        _append_log(
            stdout_path,
            "Running the complete 1x256 duration/interval GPX simulation.",
        )
        simulation_result = _plain(client.run_full_simulation())
        if int(simulation_result.get("time_steps_completed", -1)) != 256:
            raise PipelineError(
                "Full simulation completed "
                f"{simulation_result.get('time_steps_completed')} time steps, expected 256."
            )
        _append_log(stdout_path, "Reading all 256 UE position snapshots.")
        ue_positions, ru_positions = _all_positions(client, TimeStepIndex)
        position_validation = _validate_positions(ctx, ue_positions, ru_positions)
        position_registry = {
            "schema_version": "aodt-uav-position-registry-v1",
            "generated_at_utc": utc_now(),
            "ru_positions": ru_positions,
            "ue_positions_by_time_step": ue_positions,
            "validation": position_validation,
            "identity": {
                "ru_tensor_index_to_aodt_id": {
                    str(index): index + 1 for index in range(8)
                },
                "ue_tensor_index_to_aodt_id": {
                    str(index): index + 1 for index in range(54)
                },
                "uav_tensor_indices": [50, 51, 52, 53],
                "uav_aodt_ids": [51, 52, 53, 54],
            },
        }
        json_dump(stage_dir / "position_registry.json", position_registry)
        if position_validation["status"] != "pass":
            raise PipelineError(
                "AODT GPX positions do not match the AODT canonical ENU tracks: "
                f"{position_validation}"
            )
        _append_log(stdout_path, "Retrieving first/last CIR probes for UAV 51.")
        probe = _probe_uav_cirs(
            client, TimeStepIndex, stage_dir / "raw_uav_probe"
        )
        status_doc.update(
            {
                "status": "pass",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_at,
                "server_status": server_status,
                "simulation_result": simulation_result,
                "position_validation": position_validation,
                "raw_uav_probe": probe,
                "server_log_streaming": log_streaming,
            }
        )
        json_dump(stage_dir / "execution_status.json", status_doc)
        _append_log(stdout_path, "06BU execution and GPX runtime validation passed.")
        if not keep_session_for_07bu and log_streaming:
            client.stop_server_log_streaming()
            log_streaming = False
        return status_doc, client if keep_session_for_07bu else None
    except Exception as exc:
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        status_doc.update(
            {
                "status": "fail",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_at,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        json_dump(stage_dir / "execution_status.json", status_doc)
        if client is not None and log_streaming:
            try:
                client.stop_server_log_streaming()
            except Exception:
                pass
        raise PipelineError(f"06BU execution failed: {exc}") from exc


def parse_time_steps(spec: str, total: int) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(total))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise PipelineError(f"Invalid time-step range: {part}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else total
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            if step <= 0:
                raise PipelineError("Time-step range step must be positive.")
            selected.update(range(start, stop, step))
        else:
            selected.add(int(part))
    result = sorted(selected)
    invalid = [value for value in result if value < 0 or value >= total]
    if not result or invalid:
        raise PipelineError(f"Invalid selected time steps: {invalid or result}")
    return result


def _nested(data: dict[Any, Any], temporal: int, ru: int) -> np.ndarray:
    temporal_data = data.get(temporal, data.get(str(temporal)))
    if temporal_data is None:
        raise PipelineError(f"Raw tensor has no temporal key {temporal}.")
    array = temporal_data.get(ru, temporal_data.get(str(ru)))
    if array is None:
        raise PipelineError(f"Raw tensor has no RU key {ru} at {temporal}.")
    return np.asarray(array)


def _schema_record(
    schemas: dict[str, list[dict[str, Any]]], name: str, array: np.ndarray
) -> None:
    unit = {
        "values": "linear complex channel coefficient",
        "delays": "seconds",
        "angles_of_departure": "radians as returned by AODT",
        "angles_of_arrival": "radians as returned by AODT",
    }[name]
    record = {"shape": list(array.shape), "dtype": str(array.dtype), "unit": unit}
    if record not in schemas[name]:
        schemas[name].append(record)


def stage_07bu(ctx: UAVPipelineContext, client: Any) -> dict[str, Any]:
    if ctx.offline:
        raise PipelineError("07BU cannot run in offline mode.")
    execution = json_load(ctx.stage_dir("06BU") / "execution_status.json")
    if execution.get("status") != "pass":
        raise PipelineError("07BU is blocked because 06BU did not pass.")
    if client is None:
        raise PipelineError("07BU requires the live 06BU DigitalTwinClient session.")
    stage_dir = create_stage_dir(ctx, "07BU")
    raw_dir = stage_dir / "raw"
    raw_dir.mkdir()
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    started_at = time.monotonic()
    allocation = None
    inventory: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    schemas: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PAYLOAD_FIELDS
    }
    configure_aodt_imports(ctx.aodt_client_root)
    from dt_client import TimeStepIndex  # type: ignore

    status = _plain(client.get_status())
    _validate_server_status(status)
    selected = parse_time_steps(
        ctx.slots_spec, int(status["num_slots_or_timesteps_per_batch"])
    )
    ru_indices = list(range(8))
    ue_indices = list(range(54))
    ue_indices_per_ru = [ue_indices[:] for _ in ru_indices]
    progress_path = stage_dir / "collection_progress.json"
    progress = {
        "schema_version": "aodt-uav-collection-progress-v1",
        "started_at_utc": utc_now(),
        "selected_time_steps": selected,
        "completed_time_steps": [],
        "expected_npz_files": len(selected) * len(ru_indices),
        "completed_npz_files": 0,
        "status": "running",
    }
    json_dump(progress_path, progress)
    try:
        for ordinal, time_step in enumerate(selected, start=1):
            _append_log(
                stdout_path,
                f"Collecting time step {time_step} ({ordinal}/{len(selected)}), "
                "8 RUs x 54 UEs.",
            )
            allocation = client.allocate_cirs_memory(
                ru_indices,
                ue_indices_per_ru,
                is_full_antenna_pair=ctx.full_antenna_pair,
                num_time_steps=1,
            )
            client.get_cirs(
                allocation,
                batch_index=0,
                temporal_index=TimeStepIndex(time_step),
            )
            all_cir = client.to_numpy_all_cir(allocation)
            for ru_index in ru_indices:
                arrays = {
                    name: _nested(all_cir[name], time_step, ru_index)
                    for name in PAYLOAD_FIELDS
                }
                if any(array.size == 0 for array in arrays.values()):
                    raise PipelineError(
                        f"Empty raw tensor at time step {time_step}, RU {ru_index}."
                    )
                if not (
                    np.isfinite(arrays["values"].real).any()
                    and np.isfinite(arrays["values"].imag).any()
                    and np.isfinite(arrays["delays"]).any()
                ):
                    raise PipelineError(
                        f"No finite CIR/delay at time step {time_step}, RU {ru_index}."
                    )
                for name, array in arrays.items():
                    _schema_record(schemas, name, array)
                output = (
                    raw_dir
                    / f"cir_timestep_{time_step:04d}_ru_{ru_index:02d}.npz"
                )
                np.savez_compressed(
                    output,
                    **arrays,
                    batch=np.asarray(0, dtype=np.int64),
                    timestep=np.asarray(time_step, dtype=np.int64),
                    ru_tensor_index=np.asarray(ru_index, dtype=np.int64),
                    ru_scenario_id=np.asarray(ru_index, dtype=np.int64),
                    ru_aodt_id=np.asarray(ru_index + 1, dtype=np.int64),
                    ue_tensor_indices=np.asarray(ue_indices, dtype=np.int64),
                    ue_scenario_ids=np.asarray(ue_indices, dtype=np.int64),
                    ue_aodt_ids=np.asarray(
                        [index + 1 for index in ue_indices], dtype=np.int64
                    ),
                    full_antenna_pair=np.asarray(
                        ctx.full_antenna_pair, dtype=np.bool_
                    ),
                )
                relative = output.relative_to(stage_dir).as_posix()
                digest = sha256_file(output)
                hashes[relative] = digest
                inventory.append(
                    {
                        "relative_path": relative,
                        "bytes": output.stat().st_size,
                        "sha256": digest,
                        "batch": 0,
                        "timestep": time_step,
                        "ru_tensor_index": ru_index,
                        "ru_scenario_id": ru_index,
                        "ru_aodt_id": ru_index + 1,
                        "ue_tensor_indices": [0, 53],
                        "ue_count": 54,
                        "uav_tensor_indices": [50, 51, 52, 53],
                        "fields": {
                            name: {
                                "shape": list(array.shape),
                                "dtype": str(array.dtype),
                            }
                            for name, array in arrays.items()
                        },
                    }
                )
            client.deallocate_cirs_memory(allocation)
            allocation = None
            progress["completed_time_steps"].append(time_step)
            progress["completed_npz_files"] = len(inventory)
            json_dump(progress_path, progress)
        expected = len(selected) * 8
        if len(inventory) != expected:
            raise PipelineError(
                f"07BU collected {len(inventory)} NPZ files, expected {expected}."
            )
        raw_inventory = {
            "schema_version": "aodt-uav-raw-artifact-inventory-v1",
            "generated_at_utc": utc_now(),
            "status": "pass",
            "source": {
                "endpoint": ctx.endpoint,
                "transport": "official DigitalTwinClient direct gRPC",
                "yaml_sha256": sha256_file(
                    ctx.stage_dir("04BU") / "resolved_aodt_uav_config.yml"
                ),
            },
            "selection": {
                "time_steps": selected,
                "ru_tensor_indices": ru_indices,
                "ue_tensor_indices_per_ru": ue_indices,
                "uav_tensor_indices": [50, 51, 52, 53],
                "full_antenna_pair": ctx.full_antenna_pair,
            },
            "expected_file_count": expected,
            "actual_file_count": len(inventory),
            "total_bytes": sum(item["bytes"] for item in inventory),
            "artifacts": inventory,
        }
        schema_inventory = {
            "schema_version": "aodt-uav-raw-schema-inventory-v1",
            "generated_at_utc": utc_now(),
            "format": "NumPy compressed NPZ",
            "arrays": schemas,
            "identity_arrays": {
                "batch": "int64 scalar",
                "timestep": "int64 scalar",
                "ru_tensor_index": "int64 scalar, zero-based",
                "ru_scenario_id": "int64 scalar, zero-based",
                "ru_aodt_id": "int64 scalar, positive",
                "ue_tensor_indices": "int64[54], 0..53",
                "ue_scenario_ids": "int64[54], 0..53",
                "ue_aodt_ids": "int64[54], 1..54",
                "full_antenna_pair": "bool scalar",
            },
            "claim_boundary": (
                "Tensor dimensions are preserved exactly as returned by AODT. "
                "No serving-RU, SINR, CQI, or scheduler field is inferred."
            ),
        }
        json_dump(stage_dir / "raw_artifact_inventory.json", raw_inventory)
        json_dump(
            stage_dir / "raw_file_hashes.json",
            {
                "schema_version": "aodt-uav-raw-hashes-v1",
                "algorithm": "sha256",
                "files": hashes,
            },
        )
        json_dump(stage_dir / "raw_schema_inventory.json", schema_inventory)
        json_dump(
            stage_dir / "selected_raw_inputs.json",
            {
                "schema_version": "aodt-uav-selected-raw-inputs-v1",
                "generated_at_utc": utc_now(),
                "selection_rule": (
                    "Every selected duration/interval time step, all 8 RUs, "
                    "and all 54 UEs including UAV tensor indices 50-53."
                ),
                "selected_count": len(inventory),
                "selected_files": [
                    item["relative_path"] for item in inventory
                ],
                "identity_companions": [
                    "../03BU_topology/aodt_node_mapping.json",
                    "../04BU_config/gpx_binding_manifest.json",
                    "../06BU_execution/position_registry.json",
                ],
                "fallback_used": False,
            },
        )
        markdown_write(
            stage_dir / "raw_collection_report.md",
            "07BU AODT GPX/UAV raw collection report",
            [
                f"- Selected time steps: {len(selected)}.",
                f"- Raw NPZ files: {len(inventory)} (8 RUs per time step).",
                f"- Total raw bytes: {raw_inventory['total_bytes']}.",
                "- Every file contains all 54 UEs; UAV tensor indices are 50-53.",
                "- No AODT RF values or derived network fields were substituted.",
            ],
        )
        progress.update(
            {
                "status": "pass",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_at,
            }
        )
        json_dump(progress_path, progress)
        status_doc = {
            "stage": "07BU",
            "status": "pass",
            "generated_at_utc": utc_now(),
            "selected_time_step_count": len(selected),
            "raw_npz_count": len(inventory),
            "raw_total_bytes": raw_inventory["total_bytes"],
            "elapsed_seconds": time.monotonic() - started_at,
        }
        json_dump(stage_dir / "stage_status.json", status_doc)
        _append_log(stdout_path, "07BU raw collection passed.")
        return status_doc
    except Exception as exc:
        if allocation is not None:
            try:
                client.deallocate_cirs_memory(allocation)
            except Exception:
                pass
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        progress.update(
            {
                "status": "fail",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_at,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        json_dump(progress_path, progress)
        json_dump(
            stage_dir / "stage_status.json",
            {
                "stage": "07BU",
                "status": "fail",
                "generated_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise PipelineError(f"07BU raw collection failed: {exc}") from exc
    finally:
        try:
            client.stop_server_log_streaming()
        except Exception:
            pass
