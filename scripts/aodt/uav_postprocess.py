#!/usr/bin/env python3
"""Build formal UAV channel traces (09BU) and gate the AI contract (10BU).

This is an offline post-processing command.  It never contacts AODT.  Zero-CIR
links remain explicit invalid/no-coverage observations.  An AI-ready dataset is
materialized only when every UE/time pair has a finite serving link.
"""

from __future__ import annotations

import copy
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


RAW_RE = re.compile(r"cir_timestep_(\d{4})_ru_(\d{2})\.npz")
TRACE_COLUMNS = [
    "time_index", "simulation_time_s", "ru_tensor_index", "ue_tensor_index",
    "ue_class", "link_valid", "received_power_dbm", "power_weighted_delay_s",
]
ADAPTER_COLUMNS = [
    "slot_id", "ue_id", "bs_id", "distance_m", "path_gain_db",
    "path_loss_db", "rssi_dbm", "sinr_db", "delay_s", "path_type",
    "los_state", "valid",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def power_delay(values: np.ndarray, delays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen squared-magnitude CIR proxy and weighted delay.

    Absolute CIR normalization is not independently verified.  This function
    intentionally permits zero-power links so that they can be represented as
    invalid/no-coverage rather than imputed.
    """
    if values.ndim < 2 or delays.ndim < 2 or values.shape[0] != delays.shape[0]:
        raise ValueError("CIR values/delays have incompatible UE axes")
    if values.shape[-1] != delays.shape[-1]:
        raise ValueError("CIR values/delays have incompatible tap axes")
    ue_count = values.shape[0]
    per_tap = np.sum(
        np.abs(values) ** 2,
        axis=tuple(range(1, values.ndim - 1)),
        dtype=np.float64,
    )
    delay_per_tap = np.mean(delays.reshape(ue_count, -1, delays.shape[-1]), axis=1)
    total = np.sum(per_tap, axis=1)
    weighted = np.full(ue_count, np.nan, dtype=np.float64)
    valid = total > 0.0
    weighted[valid] = np.sum(per_tap[valid] * delay_per_tap[valid], axis=1) / total[valid]
    if np.any(~np.isfinite(total)) or np.any(total < 0.0):
        raise ValueError("CIR power proxy contains an invalid value")
    if np.any(valid & (~np.isfinite(weighted) | (weighted < 0.0))):
        raise ValueError("power-weighted delay contains an invalid value")
    return total, weighted


def load_grid(raw_dir: Path, slots: int, rus: int, ues: int) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    paths = sorted(raw_dir.glob("cir_timestep_*_ru_*.npz"))
    if len(paths) != slots * rus:
        raise ValueError(f"expected {slots * rus} raw NPZ files, found {len(paths)}")
    power = np.full((slots, rus, ues), np.nan, dtype=np.float64)
    delay = np.full_like(power, np.nan)
    seen: set[tuple[int, int]] = set()
    for index, path in enumerate(paths, 1):
        match = RAW_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected raw filename: {path.name}")
        slot, ru = int(match.group(1)), int(match.group(2))
        if not (0 <= slot < slots and 0 <= ru < rus) or (slot, ru) in seen:
            raise ValueError(f"invalid or duplicate raw identity: {path.name}")
        with np.load(path, allow_pickle=False) as payload:
            if int(np.asarray(payload["timestep"]).item()) != slot:
                raise ValueError(f"timestep metadata differs in {path.name}")
            if int(np.asarray(payload["ru_tensor_index"]).item()) != ru:
                raise ValueError(f"RU metadata differs in {path.name}")
            values = np.asarray(payload["values"])
            delays = np.asarray(payload["delays"], dtype=np.float64)
        if values.shape[0] != ues or delays.shape[0] != ues:
            raise ValueError(f"unexpected UE dimension in {path.name}")
        power[slot, ru], delay[slot, ru] = power_delay(values, delays)
        seen.add((slot, ru))
        if index % 256 == 0:
            print(f"Loaded AODT UAV CIR files: {index}/{len(paths)}")
    if len(seen) != slots * rus or np.any(~np.isfinite(power)):
        raise ValueError("raw UAV CIR grid is incomplete")
    return power, delay, paths


def load_positions(run_dir: Path, slots: int, rus: int, ues: int) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "06BU_execution" / "position_registry.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    ru_xyz = np.asarray(value["ru_positions"], dtype=np.float64)
    ue_xyz = np.asarray(value["ue_positions_by_time_step"], dtype=np.float64)
    if ru_xyz.shape != (rus, 3) or ue_xyz.shape != (slots, ues, 3):
        raise ValueError(f"unexpected position shapes: RU={ru_xyz.shape}, UE={ue_xyz.shape}")
    if not np.all(np.isfinite(ru_xyz)) or not np.all(np.isfinite(ue_xyz)):
        raise ValueError("position registry contains non-finite values")
    return ru_xyz, ue_xyz


def augment_ai_scenario(source: dict[str, Any], ue_xyz: np.ndarray) -> dict[str, Any]:
    """Add deterministic AI-contract metadata for the four aerial UE records."""
    scenario = copy.deepcopy(source)
    ground = list(scenario["ue"])
    if len(ground) != 50 or [int(row["id"]) for row in ground] != list(range(50)):
        raise ValueError("BU scenario must define ground UE IDs 0..49")
    profiles = [("eMBB", 0), ("mMTC", 1), ("URLLC", 2), ("eMBB", 0)]
    for offset, (traffic, slice_id) in enumerate(profiles):
        ue_id = 50 + offset
        x, y, z = (float(value) for value in ue_xyz[0, ue_id])
        ground.append({
            "id": ue_id,
            "x": x, "y": y, "z": z,
            "ue_type": "aerial",
            "imsi": f"1010123456{ue_id:03d}",
            "rnti": 70 + ue_id,
            "traffic_type": traffic,
            "slice_id": slice_id,
            "scheduling_policy": slice_id,
            "colosseum_source_bs_id": 0,
            "ai_contract_metadata_source": "deterministic BU round-robin profile; not native AODT RAN telemetry",
        })
    scenario["ue"] = ground
    return scenario


def import_adapter(repo_root: Path):
    path = repo_root / "scripts" / "adapter" / "legacy_aodt_contract_adapter.py"
    spec = importlib.util.spec_from_file_location("legacy_aodt_contract_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def derive_rows(
    power: np.ndarray,
    delay: np.ndarray,
    ru_xyz: np.ndarray,
    ue_xyz: np.ndarray,
    tx_power_dbm: float,
    bandwidth_hz: float,
    noise_figure_db: float,
    slot_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    slots, rus, ues = power.shape
    rx_dbm = np.full_like(power, np.nan)
    valid = power > 0.0
    rx_dbm[valid] = 10.0 * np.log10(power[valid]) + 30.0
    rx_mw = np.zeros_like(power)
    rx_mw[valid] = 10.0 ** (rx_dbm[valid] / 10.0)
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    noise_mw = 10.0 ** (noise_dbm / 10.0)
    candidate_rows: list[dict[str, Any]] = []
    serving_rows: list[dict[str, Any]] = []
    adapter_candidates: list[dict[str, Any]] = []
    adapter_serving: list[dict[str, Any]] = []
    for slot in range(slots):
        for ue in range(ues):
            valid_rus = np.flatnonzero(valid[slot, :, ue])
            serving_ru = int(valid_rus[np.argmax(power[slot, valid_rus, ue])]) if valid_rus.size else None
            total_rx = float(np.sum(rx_mw[slot, :, ue]))
            for ru in range(rus):
                is_valid = bool(valid[slot, ru, ue])
                ue_class = "ground" if ue < 50 else "uav"
                historical = {
                    "time_index": slot,
                    "simulation_time_s": slot * slot_ms / 1000.0,
                    "ru_tensor_index": ru,
                    "ue_tensor_index": ue,
                    "ue_class": ue_class,
                    "link_valid": 1 if is_valid else 0,
                    "received_power_dbm": repr(float(rx_dbm[slot, ru, ue])) if is_valid else "",
                    "power_weighted_delay_s": repr(float(delay[slot, ru, ue])) if is_valid else "",
                }
                candidate_rows.append(historical)
                distance = float(np.linalg.norm(ue_xyz[slot, ue] - ru_xyz[ru]))
                if is_valid:
                    gain_db = float(rx_dbm[slot, ru, ue] - tx_power_dbm)
                    signal = float(rx_mw[slot, ru, ue])
                    sinr = 10.0 * math.log10(signal / (total_rx - signal + noise_mw))
                    adapter = {
                        "slot_id": slot, "ue_id": ue, "bs_id": ru,
                        "distance_m": repr(distance), "path_gain_db": repr(gain_db),
                        "path_loss_db": repr(-gain_db),
                        "rssi_dbm": repr(float(rx_dbm[slot, ru, ue])),
                        "sinr_db": repr(sinr), "delay_s": repr(float(delay[slot, ru, ue])),
                        "path_type": "unknown", "los_state": "unknown", "valid": 1,
                    }
                else:
                    adapter = {
                        "slot_id": slot, "ue_id": ue, "bs_id": ru,
                        "distance_m": repr(distance), "path_gain_db": "", "path_loss_db": "",
                        "rssi_dbm": "", "sinr_db": "", "delay_s": "",
                        "path_type": "unknown", "los_state": "unknown", "valid": 0,
                    }
                adapter_candidates.append(adapter)
                if serving_ru == ru:
                    serving_rows.append({
                        "time_index": slot,
                        "simulation_time_s": slot * slot_ms / 1000.0,
                        "ue_tensor_index": ue,
                        "ue_class": ue_class,
                        "serving_ru_tensor_index": ru,
                        "received_power_dbm": historical["received_power_dbm"],
                        "power_weighted_delay_s": historical["power_weighted_delay_s"],
                    })
                    adapter_serving.append(adapter)
    return candidate_rows, serving_rows, adapter_candidates, adapter_serving


def coverage_summary(serving_rows: list[dict[str, Any]], slots: int, ues: int) -> dict[str, Any]:
    counts = {ue: 0 for ue in range(ues)}
    for row in serving_rows:
        counts[int(row["ue_tensor_index"])] += 1
    ground = sum(counts[index] for index in range(min(50, ues))) / (slots * min(50, ues))
    uav_count = max(0, ues - 50)
    uav = (
        sum(counts[index] for index in range(50, ues)) / (slots * uav_count)
        if uav_count else 1.0
    )
    return {
        "aggregate": len(serving_rows) / (slots * ues),
        "ground": ground,
        "uav": uav,
        "uav_by_tensor_index": {str(index): counts[index] / slots for index in range(50, ues)},
        "per_ue_serving_rows": {str(key): value for key, value in counts.items()},
    }


def blocked_tail(run_dir: Path, admission: dict[str, Any], status_09: dict[str, Any]) -> None:
    generated = utc_now()
    stage10 = run_dir / "10BU_colosseum_adapter"
    stage11 = run_dir / "11BU_advtabddpm_prepare"
    stage12 = run_dir / "12BU_advtabddpm_execution"
    stage13 = run_dir / "13BU_finalization"
    for path in (stage10, stage11, stage12, stage13):
        path.mkdir(exist_ok=False)
    status10 = {"stage": "10BU", "status": "fail_admission", "generated_at_utc": generated, **admission}
    status11 = {"stage": "11BU", "status": "blocked_by_10BU", "generated_at_utc": generated,
                "remote_upload_attempted": False, "advtabddpm_project_prepared": False,
                "reason": "10BU formal 54-UE dataset admission did not pass."}
    status12 = {"stage": "12BU", "status": "blocked_by_11BU", "generated_at_utc": generated,
                "training_attempted": False, "evaluation_attempted": False,
                "reason": "No admitted 54-UE dataset or prepared AdvTabDDPM project exists."}
    status13 = {
        "stage": "13BU", "status": "completed_with_gate_failure", "generated_at_utc": generated,
        "formal_pipeline_pass": False, "last_successful_processing_stage": "09BU",
        "blocking_stage": "10BU", "stages": {"09BU": status_09["status"], "10BU": "fail_admission",
        "11BU": "blocked_by_10BU", "12BU": "blocked_by_11BU", "13BU": "completed_with_gate_failure"},
        "remote_compute_used": False,
        "claim_boundary": "No AdvTabDDPM training/evaluation result is claimed for this BU run.",
    }
    write_json(stage10 / "dataset_admission_report.json", status10)
    write_json(stage10 / "stage_status.json", status10)
    write_json(stage11 / "stage_status.json", status11)
    write_json(stage12 / "stage_status.json", status12)
    write_json(stage13 / "stage_status.json", status13)
    coverage = admission["observed_serving_coverage"]
    report = (
        "# 13BU UAV pipeline completion report\n\n"
        "Overall: **completed with formal dataset-admission failure**.\n\n"
        f"- Serving rows: {status_09['serving_rows']} / {status_09['expected_serving_rows']}\n"
        f"- Aggregate serving coverage: {coverage['aggregate']:.6f}\n"
        f"- Ground serving coverage: {coverage['ground']:.6f}\n"
        f"- UAV serving coverage: {coverage['uav']:.6f}\n\n"
        "Zero-power links remain explicit invalid/no-coverage observations. No RF floor, "
        "serving fallback, outage imputation, or fabricated SINR/CQI value was introduced.\n"
    )
    (stage13 / "pipeline_completion_report.md").write_text(report, encoding="utf-8")


def build_09bu(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    if (run_dir / "09BU_channel_trace").exists():
        raise FileExistsError("09BU_channel_trace already exists; refusing to overwrite historical results")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = manifest["scope"]
    slots, rus, ues = int(scope["time_steps"]), int(scope["ru"]), int(scope["ue"])
    if (slots, rus, ues) != (256, 8, 54):
        raise ValueError(f"unsupported BU scope: slots={slots}, RUs={rus}, UEs={ues}")
    raw_dir = run_dir / "07BU_raw_collection" / "raw"
    power, delay, raw_paths = load_grid(raw_dir, slots, rus, ues)
    ru_xyz, ue_xyz = load_positions(run_dir, slots, rus, ues)
    source_scenario = json.loads((repo_root / "configs/scenarios/bu/tokyo_BU_multi_uav.json").read_text(encoding="utf-8"))
    scenario = augment_ai_scenario(source_scenario, ue_xyz)
    meta = scenario["meta"]
    rows = derive_rows(power, delay, ru_xyz, ue_xyz, float(meta["tx_power_dbm"]),
                       float(meta["bandwidth_hz"]), float(meta["noise_figure_db"]), int(meta["slot_ms"]))
    candidate, serving, adapter_candidate, adapter_serving = rows
    coverage = coverage_summary(serving, slots, ues)
    stage09 = run_dir / "09BU_channel_trace"
    stage09.mkdir()
    candidate_path = stage09 / "aodt_uav_candidate_trace.csv"
    serving_path = stage09 / "aodt_uav_serving_trace.csv"
    write_csv(candidate_path, candidate, TRACE_COLUMNS)
    serving_columns = ["time_index", "simulation_time_s", "ue_tensor_index", "ue_class",
                       "serving_ru_tensor_index", "received_power_dbm", "power_weighted_delay_s"]
    write_csv(serving_path, serving, serving_columns)
    write_csv(stage09 / "aodt_uav_adapter_candidate_trace.csv", adapter_candidate, ADAPTER_COLUMNS)
    write_csv(stage09 / "aodt_uav_adapter_serving_trace.csv", adapter_serving, ADAPTER_COLUMNS)
    status09 = {
        "stage": "09BU", "status": "pass" if len(serving) == slots * ues else "pass_with_no_path_links",
        "generated_at_utc": utc_now(),
        "source_08bu_status": manifest.get("stages", {}).get("08BU", {}).get("status", "unknown"),
        "candidate_rows": len(candidate), "valid_candidate_links": int(np.count_nonzero(power > 0.0)),
        "invalid_candidate_links": int(np.count_nonzero(power == 0.0)),
        "serving_rows": len(serving), "expected_serving_rows": slots * ues,
        "serving_coverage": coverage,
        "no_path_policy": "Zero-power links are retained with link_valid=0 and blank RF fields. No RF floor, serving fallback, external RF fallback, or fabricated SINR/CQI is used.",
    }
    trace_manifest = {**status09,
        "candidate_trace": candidate_path.name, "candidate_trace_sha256": sha256(candidate_path),
        "serving_trace": serving_path.name, "serving_trace_sha256": sha256(serving_path),
        "source_raw_file_count": len(raw_paths),
        "semantic_qualification": {
            "received_power_dbm": "legacy squared-magnitude CIR proxy; absolute normalization not independently verified",
            "derived_sinr": "aodt_em_proxy; not native RAN telemetry; physical cross-backend comparison disallowed",
        },
    }
    write_json(stage09 / "stage_status.json", status09)
    write_json(stage09 / "channel_trace_manifest.json", trace_manifest)
    manifest.setdefault("stages", {})["09BU"] = status09
    manifest["overall_status"] = "pass_through_09BU"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return status09


def read_csv(path: Path, expected: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f"column order mismatch: {path}")
        return list(reader)


def build_10bu(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    stage10 = run_dir / "10BU_colosseum_adapter"
    if stage10.exists():
        raise FileExistsError("10BU_colosseum_adapter already exists; refusing to overwrite historical results")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status09_path = run_dir / "09BU_channel_trace" / "stage_status.json"
    if not status09_path.is_file():
        raise FileNotFoundError("09BU must be completed before 10BU")
    status09 = json.loads(status09_path.read_text(encoding="utf-8"))
    scope = manifest["scope"]
    slots, rus, ues = int(scope["time_steps"]), int(scope["ru"]), int(scope["ue"])
    if (slots, rus, ues) != (256, 8, 54):
        raise ValueError(f"unsupported BU scope: slots={slots}, RUs={rus}, UEs={ues}")
    coverage = status09["serving_coverage"]
    admission = {
        "formal_54ue_dataset_admitted": int(status09["serving_rows"]) == slots * ues,
        "required_serving_coverage": 1.0,
        "observed_serving_coverage": coverage,
        "metrics_csv_generated": False,
        "fallback_used": False,
        "rf_floor_imputation_used": False,
        "outage_imputation_used": False,
    }
    if not admission["formal_54ue_dataset_admitted"]:
        admission["error"] = {"code": "INSUFFICIENT_VALID_SERVING_COVERAGE",
            "message": "The frozen AI contract requires one finite serving observation for every UE/time pair."}
        stage10.mkdir()
        status10 = {"stage": "10BU", "status": "fail_admission", "generated_at_utc": utc_now(), **admission}
        write_json(stage10 / "dataset_admission_report.json", status10)
        write_json(stage10 / "stage_status.json", status10)
        manifest.setdefault("stages", {})["10BU"] = status10
        manifest["overall_status"] = "blocked_at_10BU"
        manifest["formal_pipeline_pass"] = False
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        return status10

    stage09 = run_dir / "09BU_channel_trace"
    adapter_candidate = read_csv(stage09 / "aodt_uav_adapter_candidate_trace.csv", ADAPTER_COLUMNS)
    adapter_serving = read_csv(stage09 / "aodt_uav_adapter_serving_trace.csv", ADAPTER_COLUMNS)
    positions = json.loads((run_dir / "06BU_execution" / "position_registry.json").read_text(encoding="utf-8"))
    ue_xyz = np.asarray(positions["ue_positions_by_time_step"], dtype=np.float64)
    source_scenario = json.loads((repo_root / "configs/scenarios/bu/tokyo_BU_multi_uav.json").read_text(encoding="utf-8"))
    scenario = augment_ai_scenario(source_scenario, ue_xyz)
    meta = scenario["meta"]
    adapter, adapter_path = import_adapter(repo_root)
    normalized_serving, normalized_candidates = adapter.normalize_06_traces(adapter_serving, adapter_candidate, scenario)
    outputs, model_report = adapter.build_metrics(normalized_serving, normalized_candidates, scenario, int(meta["rng_seed"]))
    stage10.mkdir()
    metrics_dir = stage10 / "metrics_csv"
    adapter.write_metrics(metrics_dir, outputs, scenario)
    metrics_paths = sorted(metrics_dir.glob("*_metrics.csv"))
    aggregate = stage10 / "aodt_uav_colosseum_compatible.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=adapter.COLOSSEUM_COLUMNS)
        writer.writeheader()
        for path in metrics_paths:
            with path.open("r", newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != adapter.COLOSSEUM_COLUMNS:
                    raise ValueError(f"AI-contract field order mismatch: {path.name}")
                writer.writerows(reader)
    admission.update({"metrics_csv_generated": True, "metrics_csv_count": len(metrics_paths),
                      "aggregate_rows": slots * ues, "adapter_sha256": sha256(adapter_path),
                      "queue_scheduler_report": model_report,
                      "semantic_equivalence": "AI_CONTRACT_COMPATIBLE_ONLY"})
    status10 = {"stage": "10BU", "status": "pass", "generated_at_utc": utc_now(), **admission}
    write_json(stage10 / "dataset_admission_report.json", status10)
    write_json(stage10 / "stage_status.json", status10)
    write_json(stage10 / "dataset_provenance.json", adapter.provenance(int(meta["rng_seed"])))
    manifest["stages"]["10BU"] = status10
    manifest["overall_status"] = "pass_through_10BU"
    manifest["formal_pipeline_pass"] = None
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return status10
