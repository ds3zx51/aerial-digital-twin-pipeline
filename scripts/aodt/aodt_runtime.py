"""Online AODT 1.5 execution and raw collection stages (06B and 07B)."""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from static_core import (
    PipelineContext,
    PipelineError,
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
    if int(status.get("num_rus", -1)) != 8:
        errors.append(f"num_rus={status.get('num_rus')} (expected 8)")
    if int(status.get("num_ues", -1)) != 50:
        errors.append(f"num_ues={status.get('num_ues')} (expected 50)")
    if not status.get("is_slot_symbol_mode"):
        errors.append("worker is not in slot/symbol mode")
    if int(status.get("num_slots_or_timesteps_per_batch", -1)) != 256:
        errors.append(
            "num_slots_or_timesteps_per_batch="
            f"{status.get('num_slots_or_timesteps_per_batch')} (expected 256)"
        )
    if errors:
        raise PipelineError("AODT server status gate failed: " + "; ".join(errors))


def _positions(client: Any, slot_index_type: Any, slot: int) -> dict[str, Any]:
    return {
        "slot": slot,
        "ru_positions": _plain(client.get_ru_positions()),
        "ue_positions": _plain(
            client.get_ue_positions(
                batch_index=0, temporal_index=slot_index_type(slot)
            )
        ),
    }


def _static_position_validation(
    first: dict[str, Any], last: dict[str, Any], tolerance_m: float = 1e-6
) -> dict[str, Any]:
    first_ru = np.asarray(first["ru_positions"], dtype=float)
    last_ru = np.asarray(last["ru_positions"], dtype=float)
    first_ue = np.asarray(first["ue_positions"], dtype=float)
    last_ue = np.asarray(last["ue_positions"], dtype=float)
    if first_ru.shape != (8, 3) or last_ru.shape != (8, 3):
        raise PipelineError(
            f"Unexpected RU position shape: {first_ru.shape}, {last_ru.shape}"
        )
    if first_ue.shape != (50, 3) or last_ue.shape != (50, 3):
        raise PipelineError(
            f"Unexpected UE position shape: {first_ue.shape}, {last_ue.shape}"
        )
    ru_max = float(np.max(np.abs(first_ru - last_ru)))
    ue_max = float(np.max(np.abs(first_ue - last_ue)))
    return {
        "status": "pass"
        if ru_max <= tolerance_m and ue_max <= tolerance_m
        else "fail",
        "tolerance_m": tolerance_m,
        "ru_max_abs_delta_m": ru_max,
        "ue_max_abs_delta_m": ue_max,
        "all_positions_finite": bool(
            np.isfinite(first_ru).all()
            and np.isfinite(last_ru).all()
            and np.isfinite(first_ue).all()
            and np.isfinite(last_ue).all()
        ),
    }


def _probe_cir(client: Any, slot_index_type: Any, probe_dir: Path) -> dict[str, Any]:
    allocation = None
    try:
        allocation = client.allocate_cirs_memory(
            [0], [[0]], is_full_antenna_pair=False, num_time_steps=1
        )
        client.get_cirs(
            allocation, batch_index=0, temporal_index=slot_index_type(0)
        )
        arrays = {
            name: np.asarray(client.to_numpy(allocation, 0, 0, name))
            for name in (
                "values",
                "delays",
                "angles_of_departure",
                "angles_of_arrival",
            )
        }
        if any(array.size == 0 for array in arrays.values()):
            raise PipelineError("06B raw CIR probe contains an empty array.")
        if not (
            np.isfinite(arrays["values"].real).any()
            and np.isfinite(arrays["values"].imag).any()
            and np.isfinite(arrays["delays"]).any()
        ):
            raise PipelineError("06B raw CIR probe contains no finite CIR/delay sample.")
        probe_dir.mkdir()
        output = probe_dir / "cir_slot_0000_ru_00_ue_00_probe.npz"
        np.savez_compressed(
            output,
            **arrays,
            slot=np.asarray(0, dtype=np.int64),
            batch=np.asarray(0, dtype=np.int64),
            ru_tensor_index=np.asarray(0, dtype=np.int64),
            ru_scenario_id=np.asarray(0, dtype=np.int64),
            ru_aodt_id=np.asarray(1, dtype=np.int64),
            ue_tensor_indices=np.asarray([0], dtype=np.int64),
            ue_scenario_ids=np.asarray([0], dtype=np.int64),
            ue_aodt_ids=np.asarray([1], dtype=np.int64),
        )
        return {
            "status": "pass",
            "path": path_text(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in arrays.items()
            },
        }
    finally:
        if allocation is not None:
            client.deallocate_cirs_memory(allocation)


def stage_06b(
    ctx: PipelineContext, keep_session_for_07b: bool
) -> tuple[dict[str, Any], Any | None]:
    if ctx.offline:
        raise PipelineError("06B cannot run in offline mode.")
    preflight = json_load(ctx.stage_dir("05B") / "preflight_status.json")
    if preflight.get("status") != "pass" or not preflight.get(
        "execution_authorized"
    ):
        raise PipelineError("06B is blocked because 05B did not pass.")

    stage_dir = create_stage_dir(ctx, "06B")
    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    (stage_dir / "command.txt").write_text(ctx.command + "\n", encoding="utf-8")
    execution_manifest = {
        "schema_version": "aodt-static-execution-manifest-v1",
        "generated_at_utc": utc_now(),
        "endpoint": ctx.endpoint,
        "transport": "official dt_client.DigitalTwinClient direct gRPC",
        "yaml": path_text(
            ctx.stage_dir("04B") / "resolved_aodt_static_config.yml"
        ),
        "yaml_sha256": sha256_file(
            ctx.stage_dir("04B") / "resolved_aodt_static_config.yml"
        ),
        "expected": {
            "ru": 8,
            "ue": 50,
            "slots": 256,
            "batches": 1,
            "ue_mobility": "fixed",
        },
        "fallback_policy": "none; failure is terminal",
    }
    json_dump(stage_dir / "execution_manifest.json", execution_manifest)

    configure_aodt_imports(ctx.aodt_client_root)
    import dt_client  # type: ignore
    from dt_client import SlotIndex  # type: ignore

    client = None
    log_streaming = False
    started_at = time.monotonic()
    status_doc: dict[str, Any] = {
        "schema_version": "aodt-static-execution-status-v1",
        "stage": "06B",
        "generated_at_utc": utc_now(),
        "status": "running",
    }
    try:
        _append_log(stdout_path, f"Connecting to {ctx.endpoint} via direct gRPC.")
        client = dt_client.DigitalTwinClient(ctx.endpoint, force=False)
        if not bool(aodt_member_value(client, "is_connected")):
            raise PipelineError("DigitalTwinClient reports is_connected=false.")
        transport_mode = aodt_member_value(client, "transport_mode")
        if transport_mode != "REMOTE":
            raise PipelineError(
                f"Expected REMOTE transport, got {transport_mode}."
            )
        server_log = stage_dir / "server.log"
        log_streaming = bool(
            client.start_server_log_streaming(str(server_log.resolve()), "INFO")
        )
        yaml_content = (
            ctx.stage_dir("04B") / "resolved_aodt_static_config.yml"
        ).read_text(encoding="utf-8")
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
        _append_log(stdout_path, "Submitting resolved AODT YAML.")
        if not client.start(yaml_content):
            raise PipelineError("DigitalTwinClient.start returned false.")
        server_status = _plain(client.get_status())
        _validate_server_status(server_status)
        json_dump(stage_dir / "server_status_after_start.json", server_status)

        first_positions = _positions(client, SlotIndex, 0)
        _append_log(stdout_path, "Running the full 1x256 server-side simulation.")
        simulation_result = _plain(client.run_full_simulation())
        if int(simulation_result.get("time_steps_completed", -1)) != 256:
            raise PipelineError(
                "Full simulation completed "
                f"{simulation_result.get('time_steps_completed')} time steps, expected 256."
            )
        last_positions = _positions(client, SlotIndex, 255)
        position_validation = _static_position_validation(
            first_positions, last_positions
        )
        if position_validation["status"] != "pass":
            raise PipelineError("UE/RU positions changed in the fixed topology run.")
        json_dump(
            stage_dir / "position_registry.json",
            {
                "first": first_positions,
                "last": last_positions,
                "validation": position_validation,
                "identity": {
                    "ru_tensor_index_to_aodt_id": {
                        str(index): index + 1 for index in range(8)
                    },
                    "ue_tensor_index_to_aodt_id": {
                        str(index): index + 1 for index in range(50)
                    },
                },
            },
        )
        _append_log(stdout_path, "Retrieving a one-RU/one-UE raw CIR probe.")
        probe = _probe_cir(client, SlotIndex, stage_dir / "raw_probe")
        status_doc.update(
            {
                "status": "pass",
                "finished_at_utc": utc_now(),
                "elapsed_seconds": time.monotonic() - started_at,
                "server_status": server_status,
                "simulation_result": simulation_result,
                "position_validation": position_validation,
                "raw_probe": probe,
                "server_log_streaming": log_streaming,
            }
        )
        json_dump(stage_dir / "execution_status.json", status_doc)
        _append_log(stdout_path, "06B execution and raw probe passed.")
        if not keep_session_for_07b and log_streaming:
            client.stop_server_log_streaming()
            log_streaming = False
        return status_doc, client if keep_session_for_07b else None
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
        raise PipelineError(f"06B execution failed: {exc}") from exc


def parse_slots(spec: str, total_slots: int) -> list[int]:
    normalized = spec.strip().lower()
    if normalized == "all":
        return list(range(total_slots))
    selected: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) not in (2, 3):
                raise PipelineError(f"Invalid slot range: {part}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if pieces[1] else total_slots
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            if step <= 0:
                raise PipelineError("Slot range step must be positive.")
            selected.update(range(start, stop, step))
        else:
            selected.add(int(part))
    result = sorted(selected)
    if not result:
        raise PipelineError("No slots were selected.")
    invalid = [slot for slot in result if slot < 0 or slot >= total_slots]
    if invalid:
        raise PipelineError(f"Slot indices outside 0..{total_slots - 1}: {invalid}")
    return result


def _nested(data: dict[Any, Any], temporal: int, ru: int) -> np.ndarray:
    temporal_data = data.get(temporal, data.get(str(temporal)))
    if temporal_data is None:
        raise PipelineError(f"Raw tensor has no temporal key {temporal}.")
    array = temporal_data.get(ru, temporal_data.get(str(ru)))
    if array is None:
        raise PipelineError(f"Raw tensor has no RU key {ru} at slot {temporal}.")
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


def stage_07b(ctx: PipelineContext, client: Any) -> dict[str, Any]:
    if ctx.offline:
        raise PipelineError("07B cannot run in offline mode.")
    execution = json_load(ctx.stage_dir("06B") / "execution_status.json")
    if execution.get("status") != "pass":
        raise PipelineError("07B is blocked because 06B did not pass.")
    if client is None:
        raise PipelineError("07B requires the live 06B DigitalTwinClient session.")

    stage_dir = create_stage_dir(ctx, "07B")
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
        "values": [],
        "delays": [],
        "angles_of_departure": [],
        "angles_of_arrival": [],
    }

    configure_aodt_imports(ctx.aodt_client_root)
    from dt_client import SlotIndex  # type: ignore

    status = _plain(client.get_status())
    _validate_server_status(status)
    selected_slots = parse_slots(
        ctx.slots_spec, int(status["num_slots_or_timesteps_per_batch"])
    )
    ru_indices = list(range(8))
    ue_indices = list(range(50))
    ue_indices_per_ru = [ue_indices[:] for _ in ru_indices]
    progress_path = stage_dir / "collection_progress.json"
    progress = {
        "schema_version": "aodt-static-collection-progress-v1",
        "started_at_utc": utc_now(),
        "selected_slots": selected_slots,
        "completed_slots": [],
        "expected_npz_files": len(selected_slots) * len(ru_indices),
        "completed_npz_files": 0,
        "status": "running",
    }
    json_dump(progress_path, progress)

    try:
        for ordinal, slot in enumerate(selected_slots, start=1):
            _append_log(
                stdout_path,
                f"Collecting slot {slot} ({ordinal}/{len(selected_slots)}), "
                "8 RUs x 50 UEs.",
            )
            allocation = client.allocate_cirs_memory(
                ru_indices,
                ue_indices_per_ru,
                is_full_antenna_pair=ctx.full_antenna_pair,
                num_time_steps=1,
            )
            client.get_cirs(
                allocation, batch_index=0, temporal_index=SlotIndex(slot)
            )
            all_cir = client.to_numpy_all_cir(allocation)
            for ru_index in ru_indices:
                arrays = {
                    name: _nested(all_cir[name], slot, ru_index)
                    for name in schemas
                }
                if any(array.size == 0 for array in arrays.values()):
                    raise PipelineError(
                        f"Empty raw tensor at slot {slot}, RU {ru_index}."
                    )
                if not (
                    np.isfinite(arrays["values"].real).any()
                    and np.isfinite(arrays["values"].imag).any()
                    and np.isfinite(arrays["delays"]).any()
                ):
                    raise PipelineError(
                        f"No finite CIR/delay data at slot {slot}, RU {ru_index}."
                    )
                for name, array in arrays.items():
                    _schema_record(schemas, name, array)
                output = (
                    raw_dir
                    / f"cir_slot_{slot:04d}_ru_{ru_index:02d}.npz"
                )
                np.savez_compressed(
                    output,
                    **arrays,
                    batch=np.asarray(0, dtype=np.int64),
                    slot=np.asarray(slot, dtype=np.int64),
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
                        "slot": slot,
                        "ru_tensor_index": ru_index,
                        "ru_scenario_id": ru_index,
                        "ru_aodt_id": ru_index + 1,
                        "ue_tensor_indices": [0, 49],
                        "ue_count": 50,
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
            progress["completed_slots"].append(slot)
            progress["completed_npz_files"] = len(inventory)
            json_dump(progress_path, progress)

        expected = len(selected_slots) * 8
        if len(inventory) != expected:
            raise PipelineError(
                f"07B collected {len(inventory)} NPZ files, expected {expected}."
            )
        raw_inventory = {
            "schema_version": "aodt-static-raw-artifact-inventory-v1",
            "generated_at_utc": utc_now(),
            "status": "pass",
            "source": {
                "endpoint": ctx.endpoint,
                "transport": "official DigitalTwinClient direct gRPC",
                "yaml_sha256": sha256_file(
                    ctx.stage_dir("04B") / "resolved_aodt_static_config.yml"
                ),
            },
            "selection": {
                "slots": selected_slots,
                "ru_tensor_indices": ru_indices,
                "ue_tensor_indices_per_ru": ue_indices,
                "full_antenna_pair": ctx.full_antenna_pair,
            },
            "expected_file_count": expected,
            "actual_file_count": len(inventory),
            "total_bytes": sum(item["bytes"] for item in inventory),
            "artifacts": inventory,
        }
        schema_inventory = {
            "schema_version": "aodt-static-raw-schema-inventory-v1",
            "generated_at_utc": utc_now(),
            "format": "NumPy compressed NPZ",
            "arrays": schemas,
            "identity_arrays": {
                "batch": "int64 scalar",
                "slot": "int64 scalar",
                "ru_tensor_index": "int64 scalar, zero-based",
                "ru_scenario_id": "int64 scalar, zero-based",
                "ru_aodt_id": "int64 scalar, positive",
                "ue_tensor_indices": "int64[50], zero-based allocation order",
                "ue_scenario_ids": "int64[50], zero-based",
                "ue_aodt_ids": "int64[50], positive",
                "full_antenna_pair": "bool scalar",
            },
            "claim_boundary": (
                "Tensor dimensions are preserved exactly as returned by AODT. "
                "No SINR/CQI/serving-RU field is inferred in 07B."
            ),
        }
        selected_inputs = {
            "schema_version": "aodt-static-selected-raw-inputs-v1",
            "generated_at_utc": utc_now(),
            "selection_rule": (
                "All nonempty direct-gRPC NPZ files for every explicitly selected "
                "slot and all 8 RU x 50 UE allocations."
            ),
            "selected_count": len(inventory),
            "selected_files": [item["relative_path"] for item in inventory],
            "identity_companions": [
                "../03B_topology/aodt_node_mapping.json",
                "../04B_config/resolved_aodt_static_config.yml",
                "../06B_execution/position_registry.json",
            ],
            "representative_csv_used": False,
            "fallback_used": False,
        }
        json_dump(stage_dir / "raw_artifact_inventory.json", raw_inventory)
        json_dump(
            stage_dir / "raw_file_hashes.json",
            {
                "schema_version": "aodt-static-raw-hashes-v1",
                "algorithm": "sha256",
                "files": hashes,
            },
        )
        json_dump(stage_dir / "raw_schema_inventory.json", schema_inventory)
        json_dump(stage_dir / "selected_raw_inputs.json", selected_inputs)
        markdown_write(
            stage_dir / "raw_collection_report.md",
            "07B AODT raw collection report",
            [
                f"- Selected slots: {len(selected_slots)} "
                f"({selected_slots[0]} through {selected_slots[-1]}).",
                f"- Raw NPZ files: {len(inventory)} (8 RUs per slot).",
                f"- Total raw bytes: {raw_inventory['total_bytes']}.",
                "- Every file contains the complete AODT-returned values, delays, "
                "AoD, and AoA arrays for one RU and all 50 allocated UEs.",
                "- Scenario IDs, positive AODT IDs, and zero-based tensor indices "
                "are stored together; no creation-order inference is required.",
                "- No RF fallback or derived SINR/CQI/serving-RU field was created.",
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
            "stage": "07B",
            "status": "pass",
            "generated_at_utc": utc_now(),
            "selected_slot_count": len(selected_slots),
            "raw_npz_count": len(inventory),
            "raw_total_bytes": raw_inventory["total_bytes"],
            "elapsed_seconds": time.monotonic() - started_at,
        }
        json_dump(stage_dir / "stage_status.json", status_doc)
        _append_log(stdout_path, "07B raw collection passed.")
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
                "stage": "07B",
                "status": "fail",
                "generated_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise PipelineError(f"07B raw collection failed: {exc}") from exc
    finally:
        try:
            client.stop_server_log_streaming()
        except Exception:
            pass
