#!/usr/bin/env python3
"""Validate the replaced AODT dataset and create AdvTabDDPM classifier pickles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_COLUMNS = [
    "traffic_type", "Timestamp", "num_ues", "slice_prb", "power_multiplier",
    "dl_mcs", "dl_n_samples", "dl_buffer [bytes]", "tx_brate downlink [Mbps]",
    "tx_pkts downlink", "tx_errors downlink (%)", "dl_cqi", "ul_mcs",
    "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
    "rx_pkts uplink", "rx_errors uplink (%)", "ul_rssi", "ul_sinr", "phr",
    "sum_requested_prbs", "sum_granted_prbs", "dl_pmi", "dl_ri", "ul_n",
    "ul_turbo_iters", "IMSI", "RNTI", "slicing_enabled", "slice_id",
    "scheduling_policy",
]
DEFAULT_MIN_FORMAL_SERVING_COVERAGE = 0.99


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(root: Path, prefix: str, split: str) -> np.ndarray:
    path = root / f"{prefix}_{split}.npy"
    if not path.is_file():
        raise RuntimeError(f"missing array: {path}")
    return np.load(path, allow_pickle=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source-result-name", required=True)
    parser.add_argument("--package-sha256", default="unknown")
    parser.add_argument("--expected-ue-count", type=int, choices=(50, 54), default=50)
    parser.add_argument(
        "--minimum-formal-serving-coverage",
        type=float,
        default=DEFAULT_MIN_FORMAL_SERVING_COVERAGE,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0.0 < args.minimum_formal_serving_coverage <= 1.0:
        raise RuntimeError("minimum formal serving coverage must be in (0, 1]")

    project = args.project.resolve()
    csv_root = project / "data" / "colosseum"
    compact = project / "data" / "colosseum_22"
    pickle_root = csv_root / "pickle_to_csv"
    config = project / "exp" / "colosseum" / "ddpm_tune_best" / "config.toml"

    csv_files = sorted(csv_root.glob("*_metrics.csv"))
    if len(csv_files) != args.expected_ue_count:
        raise RuntimeError(
            f"expected {args.expected_ue_count} metrics CSV files, found {len(csv_files)}"
        )

    total_csv_rows = 0
    csv_label_counts: Counter[int] = Counter()
    for path in csv_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPECTED_COLUMNS:
                raise RuntimeError(f"schema mismatch: {path.name}")
            rows = list(reader)
        if not 1 <= len(rows) <= 256:
            raise RuntimeError(f"{path.name}: expected 1..256 real rows, found {len(rows)}")
        total_csv_rows += len(rows)
        csv_label_counts.update(int(row["traffic_type"]) for row in rows)

    expected_rows = args.expected_ue_count * 256
    serving_coverage = total_csv_rows / float(expected_rows)
    if (
        serving_coverage < args.minimum_formal_serving_coverage
        or total_csv_rows > expected_rows
    ):
        raise RuntimeError(
            f"formal serving coverage {serving_coverage:.6f} is outside "
            f"[{args.minimum_formal_serving_coverage:.6f}, 1.0]"
        )
    if set(csv_label_counts) != {0, 1, 2}:
        raise RuntimeError(f"expected labels 0,1,2; found {sorted(csv_label_counts)}")

    split_report: dict[str, dict[str, object]] = {}
    split_total = 0
    for split in ("train", "val", "test"):
        x_num = load_array(compact, "X_num", split)
        x_cat = load_array(compact, "X_cat", split)
        labels = load_array(compact, "y", split)
        rows = int(labels.shape[0]) if labels.ndim == 1 else -1
        if rows < 1 or x_num.shape != (rows, 20):
            raise RuntimeError(f"{split}: incompatible X_num/y shapes {x_num.shape}/{labels.shape}")
        if x_cat.shape != (rows, 3):
            raise RuntimeError(f"{split}: X_cat shape {x_cat.shape}, expected {(rows, 3)}")
        label_counts = Counter(int(value) for value in labels.tolist())
        if set(label_counts) != {0, 1, 2}:
            raise RuntimeError(f"{split}: all three labels are required")
        split_report[split] = {
            "rows": rows,
            "x_num_shape": list(x_num.shape),
            "x_cat_shape": list(x_cat.shape),
            "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        }
        split_total += rows
    if split_total != total_csv_rows:
        raise RuntimeError(f"split rows {split_total} do not conserve CSV rows {total_csv_rows}")

    config_text = config.read_text(encoding="utf-8")
    required_config = (
        'real_data_path = "data/colosseum_22"',
        "num_numerical_features = 20",
        "num_classes = 3",
        "d_in = 41",
    )
    missing_config = [item for item in required_config if item not in config_text]
    if missing_config:
        raise RuntimeError(f"config contract mismatch: {missing_config}")

    sys.path.insert(0, str(project))
    from classifier_data_preprocess import (  # pylint: disable=import-error,import-outside-toplevel
        gen_slice_dataset_from_numpy,
    )

    dataset, normalizer = gen_slice_dataset_from_numpy(str(compact), slice_len=1)
    pickle_root.mkdir(parents=True, exist_ok=True)
    dataset_path = pickle_root / "colosseum_dataset_globalnorm.pkl"
    normalizer_path = pickle_root / "colosseum_global_cols_maxmin.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle)
    with normalizer_path.open("wb") as handle:
        pickle.dump(normalizer, handle)

    payload = {
        "status": "formal_dataset_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "aodt_rt_tokyo",
        "source_09a_result_name": args.source_result_name,
        "base_application": "AdvTabDDPM_V2_aodt",
        "base_package_sha256": args.package_sha256,
        "application_code_modified": False,
        "config_modified": False,
        "replacement_scope": "dataset_only",
        "formal_eligible": True,
        "csv_files": len(csv_files),
        "expected_ue_count": args.expected_ue_count,
        "expected_rows": expected_rows,
        "csv_rows": total_csv_rows,
        "serving_coverage": serving_coverage,
        "minimum_formal_serving_coverage": args.minimum_formal_serving_coverage,
        "csv_label_counts": {str(k): v for k, v in sorted(csv_label_counts.items())},
        "feature_contract": {"numerical": 20, "categorical": 3, "classes": 3, "model_d_in": 41},
        "split_method": "original_AdvTabDDPM_seeded_row_split",
        "split_seed": 42,
        "split_report": split_report,
        "dataset_pickle": str(dataset_path.relative_to(project)),
        "normalizer_pickle": str(normalizer_path.relative_to(project)),
        "hashes": {
            "dataset_pickle_sha256": sha256(dataset_path),
            "normalizer_pickle_sha256": sha256(normalizer_path),
            "config_sha256": sha256(config),
        },
        "claim_boundary": (
            "This dataset supports the AODT-to-AdvTabDDPM experiment. "
            "Stage-A radio-fidelity claims remain supported by the separate AODT/AODT link metrics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
