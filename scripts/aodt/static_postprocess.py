#!/usr/bin/env python3
"""Build AODT static 09B trace and legacy 10B AI-contract artifacts.

The source is the complete 07B AODT CIR collection.  This program does not
call the AODT worker and does not change AdvTabDDPM model code.  It creates a
an internal serving/candidate trace, then calls the legacy AODT contract
adapter for the deterministic queue and scheduler portion of the legacy
AI-facing interface. Physical equivalence to Colosseum is not claimed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


RAW_FILE_RE = re.compile(r"cir_slot_(\d{4})_ru_(\d{2})\.npz$")
CHANNEL_COLUMNS = [
    "slot_id", "ue_id", "bs_id", "distance_m", "path_gain_db",
    "path_loss_db", "rssi_dbm", "sinr_db", "delay_s", "path_type",
    "los_state", "valid",
]
CANONICAL_COLUMNS = [
    "time_index", "slot_id", "simulation_time_s", "ue_id", "ru_id",
    "serving_bs_id", "ue_x", "ue_y", "ue_z", "bs_x", "bs_y", "bs_z",
    "distance_2d_m", "distance_3d_m", "path_gain_db", "path_loss_db",
    "received_power_dbm", "snr_db", "sinr_db",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dbm_to_mw(value: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(value, dtype=np.float64) / 10.0)


def mw_to_dbm(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Channel/noise power must be finite and positive.")
    return 10.0 * np.log10(arr)


def load_topology(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = run_dir / "03B_topology" / "static_topology_snapshot.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rus: list[dict[str, Any]] = []
    ues: list[dict[str, Any]] = []
    for row in rows:
        item = {
            "id": int(row["scenario_id"]),
            "x": float(row["x_m"]),
            "y": float(row["y_m"]),
            "z": float(row["z_m"]),
        }
        if row["node_type"] == "RU":
            rus.append(item)
        elif row["node_type"] == "UE":
            item["ue_type"] = row["ue_type"].lower()
            ues.append(item)
    rus.sort(key=lambda value: value["id"])
    ues.sort(key=lambda value: value["id"])
    if [item["id"] for item in rus] != list(range(8)):
        raise ValueError("Expected AODT RU IDs 0..7 in 03B topology snapshot.")
    if [item["id"] for item in ues] != list(range(50)):
        raise ValueError("Expected AODT UE IDs 0..49 in 03B topology snapshot.")
    return rus, ues


def power_and_delay(path: Path, ue_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a legacy squared-magnitude CIR power proxy and weighted delay.

    Absolute CIR normalization has not been independently verified.
    """
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["values"])
        delays = np.asarray(data["delays"], dtype=np.float64)
    if values.shape[0] != ue_count or delays.shape[0] != ue_count:
        raise ValueError(f"{path.name}: unexpected UE dimension.")
    if values.ndim < 2 or delays.ndim < 2 or values.shape[-1] != delays.shape[-1]:
        raise ValueError(f"{path.name}: incompatible CIR value/delay dimensions.")
    power_by_tap = np.sum(np.abs(values) ** 2, axis=tuple(range(1, values.ndim - 1)), dtype=np.float64)
    delay_by_tap = np.mean(delays.reshape(ue_count, -1, delays.shape[-1]), axis=1)
    total_power = np.sum(power_by_tap, axis=1)
    if np.any(~np.isfinite(total_power)) or np.any(total_power <= 0.0):
        raise ValueError(f"{path.name}: non-finite or empty CIR power.")
    mean_delay = np.sum(power_by_tap * delay_by_tap, axis=1) / total_power
    if np.any(~np.isfinite(mean_delay)) or np.any(mean_delay < 0.0):
        raise ValueError(f"{path.name}: invalid power-weighted delay.")
    return total_power, mean_delay


def load_cir_tensor(run_dir: Path, slots: int, ru_count: int, ue_count: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    raw_dir = run_dir / "07B_raw_collection" / "raw"
    files = sorted(raw_dir.glob("cir_slot_*_ru_*.npz"))
    expected = slots * ru_count
    if len(files) != expected:
        raise ValueError(f"Expected {expected} 07B CIR NPZ files, found {len(files)}.")
    power = np.full((slots, ru_count, ue_count), np.nan, dtype=np.float64)
    delay = np.full_like(power, np.nan)
    seen: set[tuple[int, int]] = set()
    for index, path in enumerate(files, start=1):
        match = RAW_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected 07B filename: {path.name}")
        slot, ru = int(match.group(1)), int(match.group(2))
        if not 0 <= slot < slots or not 0 <= ru < ru_count or (slot, ru) in seen:
            raise ValueError(f"Invalid or duplicate slot/RU identity in {path.name}")
        seen.add((slot, ru))
        power[slot, ru], delay[slot, ru] = power_and_delay(path, ue_count)
        if index % 256 == 0:
            print(f"Loaded AODT CIR files: {index}/{expected}")
    if len(seen) != expected or not np.all(np.isfinite(power)) or not np.all(np.isfinite(delay)):
        raise ValueError("07B CIR grid is incomplete or contains non-finite values.")
    return power, delay, files


def import_shared_adapter(repo_root: Path):
    path = repo_root / "scripts" / "adapter" / "legacy_aodt_contract_adapter.py"
    spec = importlib.util.spec_from_file_location("legacy_aodt_contract_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def aodt_provenance(shared: Any, seed: int) -> dict[str, Any]:
    value = shared.provenance(seed)
    value["schema_version"] = "aodt-colosseum-provenance-v1"
    value["aodt_direct_inputs"] = [
        {
            "field": "07B raw NPZ values",
            "meaning": "AODT complex CIR coefficients; aggregate power is sum(|CIR|^2).",
        },
        {
            "field": "07B raw NPZ delays",
            "meaning": "AODT multipath delays; reported canonical delay is power-weighted.",
        },
    ]
    return value


def build_stages(repo_root: Path, run_dir: Path, stop_after_09b: bool = False) -> dict[str, Any]:
    stage09 = run_dir / "09B_canonical_trace"
    stage10 = run_dir / "10B_colosseum_adapter"
    if stage09.exists() or (not stop_after_09b and stage10.exists()):
        raise FileExistsError("09B/10B directories already exist; non-overwrite policy prevents reuse.")
    scenario_path = repo_root / "configs" / "scenario_layout_tokyo_docomo.json"
    radio_path = run_dir / "04B_config" / "resolved_radio_config.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    radio = json.loads(radio_path.read_text(encoding="utf-8"))["source"]
    slots = int(radio["slots"])
    ru_tx_dbm = float(radio["ru_tx_power_dbm"])
    bandwidth_hz = float(radio["bandwidth_hz"])
    noise_figure_db = float(scenario["meta"]["noise_figure_db"])
    slot_ms = int(scenario["meta"]["slot_ms"])
    timestamp0_ms = int(scenario["meta"]["timestamp0_ms"])
    seed = int(scenario["meta"]["rng_seed"])
    rus, ues = load_topology(run_dir)
    power_w, delay_s, raw_files = load_cir_tensor(run_dir, slots, len(rus), len(ues))
    rx_dbm = mw_to_dbm(power_w) + 30.0
    gain_db = rx_dbm - ru_tx_dbm
    rx_mw = dbm_to_mw(rx_dbm)
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    noise_mw = float(dbm_to_mw(noise_dbm))
    serving_ru = np.argmax(rx_mw, axis=1)
    desired_mw = np.take_along_axis(rx_mw, serving_ru[:, None, :], axis=1)[:, 0, :]
    interference_mw = np.sum(rx_mw, axis=1) - desired_mw
    sinr_db = mw_to_dbm(desired_mw / (interference_mw + noise_mw))
    snr_db = mw_to_dbm(desired_mw / noise_mw)
    ru_xyz = np.asarray([[item["x"], item["y"], item["z"]] for item in rus], dtype=np.float64)
    ue_xyz = np.asarray([[item["x"], item["y"], item["z"]] for item in ues], dtype=np.float64)
    stage09.mkdir(parents=True, exist_ok=False)
    candidate_path = stage09 / "aodt_static_candidate_trace.csv"
    serving_path = stage09 / "aodt_static_serving_trace.csv"
    canonical_path = stage09 / "aodt_static_trace_normalized.csv"
    candidates: list[dict[str, str]] = []
    serving: list[dict[str, str]] = []
    canonical: list[dict[str, str]] = []
    for slot in range(slots):
        for ue_id in range(len(ues)):
            selected_ru = int(serving_ru[slot, ue_id])
            for ru_id in range(len(rus)):
                delta = ue_xyz[ue_id] - ru_xyz[ru_id]
                distance_2d = float(np.linalg.norm(delta[:2]))
                distance_3d = float(np.linalg.norm(delta))
                candidate = {
                    "slot_id": str(slot), "ue_id": str(ue_id), "bs_id": str(ru_id),
                    "distance_m": repr(distance_3d), "path_gain_db": repr(float(gain_db[slot, ru_id, ue_id])),
                    "path_loss_db": repr(float(-gain_db[slot, ru_id, ue_id])),
                    "rssi_dbm": repr(float(rx_dbm[slot, ru_id, ue_id])),
                    "sinr_db": repr(float(10.0 * math.log10(rx_mw[slot, ru_id, ue_id] / (np.sum(rx_mw[slot, :, ue_id]) - rx_mw[slot, ru_id, ue_id] + noise_mw)))),
                    "delay_s": repr(float(delay_s[slot, ru_id, ue_id])),
                    "path_type": "unknown", "los_state": "unknown", "valid": "1",
                }
                candidates.append(candidate)
                if ru_id == selected_ru:
                    serving.append(candidate)
                    canonical.append({
                        "time_index": str(slot), "slot_id": str(slot),
                        "simulation_time_s": repr(slot * slot_ms / 1000.0),
                        "ue_id": str(ue_id), "ru_id": str(ru_id), "serving_bs_id": str(selected_ru),
                        "ue_x": repr(float(ue_xyz[ue_id, 0])), "ue_y": repr(float(ue_xyz[ue_id, 1])), "ue_z": repr(float(ue_xyz[ue_id, 2])),
                        "bs_x": repr(float(ru_xyz[ru_id, 0])), "bs_y": repr(float(ru_xyz[ru_id, 1])), "bs_z": repr(float(ru_xyz[ru_id, 2])),
                        "distance_2d_m": repr(distance_2d), "distance_3d_m": repr(distance_3d),
                        "path_gain_db": candidate["path_gain_db"], "path_loss_db": candidate["path_loss_db"],
                        "received_power_dbm": candidate["rssi_dbm"], "snr_db": repr(float(snr_db[slot, ue_id])), "sinr_db": repr(float(sinr_db[slot, ue_id])),
                    })
    for path, rows, fields in ((candidate_path, candidates, CHANNEL_COLUMNS), (serving_path, serving, CHANNEL_COLUMNS), (canonical_path, canonical, CANONICAL_COLUMNS)):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    validation = {
        "status": "pass",
        "candidate_rows": len(candidates), "expected_candidate_rows": slots * len(rus) * len(ues),
        "serving_rows": len(serving), "expected_serving_rows": slots * len(ues),
        "canonical_rows": len(canonical), "unique_slots": len({row["slot_id"] for row in canonical}),
        "unique_ues": len({row["ue_id"] for row in canonical}),
        "unique_rus": len({row["bs_id"] for row in candidates}),
        "all_coordinates_static": True,
        "all_channel_values_finite": bool(np.all(np.isfinite(power_w)) and np.all(np.isfinite(sinr_db))),
        "all_candidates_valid": True,
        "serving_rule": "strongest received-power RU independently per UE and slot",
    }
    if validation["candidate_rows"] != validation["expected_candidate_rows"] or validation["serving_rows"] != validation["expected_serving_rows"] or not validation["all_channel_values_finite"]:
        raise RuntimeError(f"09B validation failed: {validation}")
    lineage = {
        "schema_version": "aodt-static-field-lineage-v1",
        "source_run": str(run_dir),
        "raw_source": "07B raw AODT CIR NPZ files",
        "channel_power": "Legacy proxy assumes P_rx,dBm = 10*log10(sum(abs(CIR)^2)) + 30; configured RU Tx power is not added a second time.",
        "absolute_cir_normalization_verified": False,
        "effective_path_gain": "G_eff,dB = P_rx,dBm - configured_RU_tx_power_dBm",
        "effective_path_loss": "L_eff,dB = -G_eff,dB; link-budget-derived quantity, not a native AODT field.",
        "serving_bs_id": validation["serving_rule"],
        "sinr_db": "Legacy EM downlink proxy: 10*log10(S_mW/(sum(I_mW)+N_mW)); all non-serving 7 RUs are cochannel interferers.",
        "snr_db": "10*log10(S_mW/N_mW)",
        "noise": {"thermal_density_dbm_hz": -174.0, "bandwidth_hz": bandwidth_hz, "noise_figure_db": noise_figure_db, "noise_power_dbm": noise_dbm},
        "delay_s": "Power-weighted mean of AODT raw per-tap delays.",
        "native_vs_derived": {"native": ["complex CIR", "per-tap delay"], "derived": ["received_power_dbm", "path_gain_db", "path_loss_db", "serving_bs_id", "snr_db", "legacy_em_proxy_sinr_db"]},
        "claim_boundary": "Derived AODT wideband link metrics under the documented full-reuse model; the SINR-named field is a legacy EM proxy, not native AODT RAN SINR, 3GPP per-RE, post-equalization, beam-scheduled, or throughput telemetry.",
    }
    write_json(stage09 / "aodt_field_lineage.json", lineage)
    write_json(stage09 / "canonical_trace_validation.json", validation)
    metadata = {
        "schema_version": "aodt-static-canonical-trace-v1", "created_at_utc": utc_now(),
        "source_raw_file_count": len(raw_files), "source_raw_sha256_sample": {path.name: sha256(path) for path in raw_files[:2] + raw_files[-2:]},
        "rows": {"candidate": len(candidates), "serving": len(serving), "canonical": len(canonical)},
        "radio": {"ru_tx_power_dbm": ru_tx_dbm, "bandwidth_hz": bandwidth_hz, "noise_figure_db": noise_figure_db, "noise_power_dbm": noise_dbm},
        "artifacts": {"candidate_trace": candidate_path.name, "serving_trace": serving_path.name, "canonical_trace": canonical_path.name},
    }
    write_json(stage09 / "aodt_static_trace_metadata.json", metadata)

    if stop_after_09b:
        return {"09B": validation, "stage09": str(stage09)}

    shared, shared_path = import_shared_adapter(repo_root)
    normalized_serving, normalized_candidates = shared.normalize_06_traces(serving, candidates, scenario)
    outputs, model_report = shared.build_metrics(normalized_serving, normalized_candidates, scenario, seed)
    stage10.mkdir(parents=True, exist_ok=False)
    metrics_dir = stage10 / "metrics_csv"
    shared.write_metrics(metrics_dir, outputs, scenario)
    metrics_paths = sorted(metrics_dir.glob("*_metrics.csv"))
    aggregate_path = stage10 / "aodt_static_colosseum_compatible.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as out_stream:
        writer: csv.DictWriter | None = None
        for metrics_path in metrics_paths:
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as in_stream:
                reader = csv.DictReader(in_stream)
                if reader.fieldnames != shared.COLOSSEUM_COLUMNS:
                    raise ValueError(f"AI-contract field order mismatch: {metrics_path.name}")
                if writer is None:
                    writer = csv.DictWriter(out_stream, fieldnames=shared.COLOSSEUM_COLUMNS)
                    writer.writeheader()
                writer.writerows(reader)
    with aggregate_path.open("r", encoding="utf-8-sig", newline="") as stream:
        aggregate_rows = list(csv.DictReader(stream))
    label_counts: dict[str, int] = {}
    for row in aggregate_rows:
        label_counts[row["traffic_type"]] = label_counts.get(row["traffic_type"], 0) + 1
    adapter_report = {
        "status": "pass", "channel_source": "aodt", "backend": "AODT 1.5", "seed": seed,
        "shared_adapter": str(shared_path), "shared_adapter_sha256": sha256(shared_path),
        "shared_adapter_functions": ["normalize_06_traces", "build_metrics", "write_metrics"],
        "raw_metric_files": len(metrics_paths), "rows": len(aggregate_rows), "columns": len(shared.COLOSSEUM_COLUMNS),
        "label_counts": label_counts, "queue_scheduler_report": model_report,
        "semantic_qualifications": {
            "ul_sinr": {
                "semantic_role": "aodt_em_proxy_ul_sinr_db",
                "semantic_equivalence_to_colosseum": "NOT_YET_VERIFIED_SEMANTIC_EQUIVALENT",
                "physical_distribution_comparison_allowed": False,
                "assumptions": ["all other UEs cochannel-active", "no PRB assignment or activity mask"],
            }
        },
        "claim_boundary": "AI_CONTRACT_COMPATIBLE_ONLY. Channel columns originate from AODT-derived canonical links. The historical ul_sinr field is an EM proxy. MAC, queue, CQI/MCS, PRB, and traffic fields are deterministic downstream model values, not native AODT RAN telemetry.",
    }
    schema_report = {
        "status": "pass", "expected_columns": shared.COLOSSEUM_COLUMNS,
        "metrics_file_count": len(metrics_paths), "expected_metrics_file_count": len(ues),
        "aggregate_rows": len(aggregate_rows), "expected_aggregate_rows": slots * len(ues),
        "columns_match": True, "all_values_finite": all(all(value != "" for value in row.values()) for row in aggregate_rows),
        "label_counts": label_counts,
    }
    if len(metrics_paths) != len(ues) or len(aggregate_rows) != slots * len(ues) or not schema_report["all_values_finite"]:
        raise RuntimeError(f"10B schema contract failed: {schema_report}")
    write_json(stage10 / "aodt_static_dataset_metadata.json", adapter_report)
    write_json(stage10 / "aodt_adapter_lineage.json", aodt_provenance(shared, seed))
    write_json(stage10 / "schema_compatibility_report.json", schema_report)
    return {"09B": validation, "10B": schema_report, "stage09": str(stage09), "stage10": str(stage10)}


def update_run_manifest(run_dir: Path, results: dict[str, Any]) -> None:
    path = run_dir / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("stages", {})["09B"] = {"stage": "09B", "status": "pass", **results["09B"], "directory": results["stage09"]}
    manifest.setdefault("stages", {})["10B"] = {"stage": "10B", "status": "pass", **results["10B"], "directory": results["stage10"]}
    manifest["overall_status"] = "pass_through_10B"
    manifest["updated_at_utc"] = utc_now()
    write_json(path, manifest)



