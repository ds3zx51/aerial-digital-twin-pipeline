#!/usr/bin/env python3
"""Audit deterministic target proxies without changing the compatibility track."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def mutual_information(xs: list[str], ys: list[str]) -> float:
    n = len(xs); px, py, joint = Counter(xs), Counter(ys), Counter(zip(xs, ys)); value = 0.0
    for (x, y), count in joint.items():
        pxy = count/n; value += pxy*math.log(pxy/((px[x]/n)*(py[y]/n)), 2)
    return value


def entropy(values: list[str]) -> float:
    n = len(values); return -sum((c/n)*math.log(c/n, 2) for c in Counter(values).values())


def relation(labels: list[str], values: list[str]) -> dict:
    mapping = defaultdict(set)
    reverse = defaultdict(set)
    for label, value in zip(labels, values): mapping[value].add(label); reverse[label].add(value)
    one_to_one = all(len(x) == 1 for x in mapping.values()) and all(len(x) == 1 for x in reverse.values())
    mi = mutual_information(values, labels); h = entropy(labels)
    return {"mutual_information_bits": mi, "normalized_mutual_information_to_label": mi/h if h else 0.0, "one_to_one_mapping": one_to_one, "value_to_labels": {k: sorted(v) for k, v in mapping.items()}}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--csv-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=[]; filename_labels=[]
    for path in sorted(a.csv_dir.glob("*_metrics.csv")):
        m=re.search(r"_(eMBB|mMTC|URLLC)_",path.name); inferred=m.group(1) if m else "unknown"
        with path.open(newline="",encoding="utf-8-sig") as f:
            current=list(csv.DictReader(f)); rows.extend(current); filename_labels.extend([inferred]*len(current))
    if not rows: raise RuntimeError("no metrics CSV files found for leakage audit")
    labels=[r["traffic_type"] for r in rows]
    fields={
        "slice_id":[r["slice_id"] for r in rows],
        "scheduling_policy":[r["scheduling_policy"] for r in rows],
        "IMSI":[r["IMSI"] for r in rows],
        "RNTI":[r["RNTI"] for r in rows],
        "filename_traffic_type":filename_labels,
    }
    audits={name:relation(labels,values) for name,values in fields.items()}
    categorical=["RNTI","slice_id","scheduling_policy"]
    deterministic=[name for name,x in audits.items() if x["one_to_one_mapping"] or x["normalized_mutual_information_to_label"] >= 0.999999]
    result={
        "status":"audit_complete", "compatibility_track":"compatibility/reproduction track",
        "target_column":"traffic_type", "categorical_columns":categorical,
        "imsi_rnti_assignment_rule":"copied from Colosseum topology evidence by IMSI offset; deterministic",
        "field_audits":audits, "deterministic_label_proxies":deterministic,
        "label_leakage_risk":bool(deterministic),
        "risk_level":"high" if deterministic else "low",
        "pipeline_gate":"warning_only",
        "warning":"Leakage risk is reported but does not stop the compatibility experiment. A leakage-controlled study requires a separately reviewed feature contract.",
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result));return 0


if __name__=="__main__":raise SystemExit(main())
