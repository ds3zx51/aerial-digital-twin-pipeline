#!/usr/bin/env python3
"""Materialize static stage 10B from an existing, validated 09B trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path, expected: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f"column order mismatch: {path}")
        return list(reader)


def import_adapter(repo_root: Path):
    path = repo_root / "scripts" / "adapter" / "legacy_aodt_contract_adapter.py"
    spec = importlib.util.spec_from_file_location("legacy_aodt_contract_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def build(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    stage09 = run_dir / "09B_canonical_trace"
    stage10 = run_dir / "10B_colosseum_adapter"
    if not stage09.is_dir():
        raise FileNotFoundError("09B_canonical_trace must exist before 10B")
    if stage10.exists():
        raise FileExistsError("10B_colosseum_adapter already exists; refusing overwrite")
    scenario = json.loads((repo_root / "configs/scenario_layout_tokyo_docomo.json").read_text(encoding="utf-8"))
    adapter, adapter_path = import_adapter(repo_root)
    channel_columns = ["slot_id", "ue_id", "bs_id", "distance_m", "path_gain_db",
                       "path_loss_db", "rssi_dbm", "sinr_db", "delay_s", "path_type",
                       "los_state", "valid"]
    serving = read_csv(stage09 / "aodt_static_serving_trace.csv", channel_columns)
    candidates = read_csv(stage09 / "aodt_static_candidate_trace.csv", channel_columns)
    normalized_serving, normalized_candidates = adapter.normalize_06_traces(serving, candidates, scenario)
    seed = int(scenario["meta"]["rng_seed"])
    outputs, model_report = adapter.build_metrics(normalized_serving, normalized_candidates, scenario, seed)
    stage10.mkdir()
    metrics_dir = stage10 / "metrics_csv"
    adapter.write_metrics(metrics_dir, outputs, scenario)
    metrics_paths = sorted(metrics_dir.glob("*_metrics.csv"))
    aggregate = stage10 / "aodt_static_colosseum_compatible.csv"
    row_count = 0
    labels: dict[str, int] = {}
    with aggregate.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=adapter.COLOSSEUM_COLUMNS)
        writer.writeheader()
        for path in metrics_paths:
            with path.open("r", newline="", encoding="utf-8-sig") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != adapter.COLOSSEUM_COLUMNS:
                    raise ValueError(f"AI-contract field order mismatch: {path.name}")
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
                    labels[row["traffic_type"]] = labels.get(row["traffic_type"], 0) + 1
    expected_rows = int(scenario["meta"]["num_ue"]) * int(scenario["meta"]["time_steps"])
    if len(metrics_paths) != int(scenario["meta"]["num_ue"]) or row_count != expected_rows:
        raise RuntimeError("10B output row/file count differs from the frozen contract")
    status = {
        "stage": "10B", "status": "pass", "generated_at_utc": utc_now(),
        "metrics_file_count": len(metrics_paths), "aggregate_rows": row_count,
        "columns": len(adapter.COLOSSEUM_COLUMNS), "label_counts": labels,
        "adapter_sha256": sha256(adapter_path), "queue_scheduler_report": model_report,
        "semantic_equivalence": "AI_CONTRACT_COMPATIBLE_ONLY",
        "physical_distribution_comparison_allowed_for_ul_sinr": False,
    }
    write_json(stage10 / "stage_status.json", status)
    write_json(stage10 / "dataset_provenance.json", adapter.provenance(seed))
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("stages", {})["10B"] = status
    manifest["overall_status"] = "pass_through_10B"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.repo_root.resolve(), args.run_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
