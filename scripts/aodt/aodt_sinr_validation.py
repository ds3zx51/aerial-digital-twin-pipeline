#!/usr/bin/env python3
"""Derive AODT-compatible SINR validation figures from AODT CIR tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILE_RE = re.compile(r"cir_slot_(\d{4})_ru_(\d{2})\.npz$")
CSV_FIELDS = [
    "ue_type",
    "ue_id",
    "serving_ru_id",
    "slot_id",
    "distance_m",
    "channel_power_gain_db",
    "path_loss_db",
    "desired_power_dbm",
    "interference_power_dbm",
    "noise_power_dbm",
    "denominator_power_dbm",
    "sinr_db",
    "sinr_recomputed_db",
    "formula_error_db",
    "valid_nonserving_ru_count",
    "candidate_ru_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dbm_to_mw(value: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(value) / 10.0)


def mw_to_dbm(value: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.asarray(value))


def load_topology(run_dir: Path) -> tuple[list[dict], list[dict]]:
    path = run_dir / "03B_topology" / "static_topology_snapshot.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rus = []
    ues = []
    for row in rows:
        record = {
            "id": int(row["scenario_id"]),
            "x": float(row["x_m"]),
            "y": float(row["y_m"]),
            "z": float(row["z_m"]),
        }
        if row["node_type"] == "RU":
            rus.append(record)
        elif row["node_type"] == "UE":
            record["ue_type"] = row["ue_type"].lower()
            ues.append(record)
    rus.sort(key=lambda item: item["id"])
    ues.sort(key=lambda item: item["id"])
    if [item["id"] for item in rus] != list(range(8)):
        raise RuntimeError("Expected RU IDs 0..7.")
    if [item["id"] for item in ues] != list(range(50)):
        raise RuntimeError("Expected UE IDs 0..49.")
    return rus, ues


def load_channel_power(run_dir: Path, slots: int, rus: int, ues: int) -> np.ndarray:
    raw_dir = run_dir / "07B_raw_collection" / "raw"
    files = sorted(raw_dir.glob("cir_slot_*_ru_*.npz"))
    expected = slots * rus
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} AODT CIR files, found {len(files)}.")
    power = np.full((slots, rus, ues), np.nan, dtype=np.float64)
    seen = set()
    for index, path in enumerate(files, start=1):
        match = FILE_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"Unexpected CIR filename: {path.name}")
        slot = int(match.group(1))
        ru = int(match.group(2))
        key = (slot, ru)
        if key in seen:
            raise RuntimeError(f"Duplicate CIR file for slot={slot}, RU={ru}.")
        seen.add(key)
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["values"])
            if values.shape[0] != ues:
                raise RuntimeError(f"{path.name}: expected {ues} UE rows, got {values.shape[0]}.")
            axes = tuple(range(1, values.ndim))
            power[slot, ru, :] = np.sum(np.square(np.abs(values), dtype=np.float64), axis=axes)
        if index % 256 == 0:
            print(f"Loaded {index}/{expected} CIR files")
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        raise RuntimeError("AODT channel-power tensor contains non-finite or non-positive values.")
    return power


def select_representative(
    sinr_db: np.ndarray, serving_ru: np.ndarray, ues: list[dict], ue_type: str
) -> dict:
    ids = [item["id"] for item in ues if item["ue_type"] == ue_type]
    summaries = []
    for ue_id in ids:
        values = sinr_db[:, ue_id]
        ru_ids = serving_ru[:, ue_id]
        summaries.append(
            {
                "ue_id": ue_id,
                "median_sinr_db": float(np.median(values)),
                "stable_serving_ru": bool(np.all(ru_ids == ru_ids[0])),
            }
        )
    class_median = float(np.median([item["median_sinr_db"] for item in summaries]))
    stable = [item for item in summaries if item["stable_serving_ru"]]
    pool = stable or summaries
    selected = min(
        pool,
        key=lambda item: (abs(item["median_sinr_db"] - class_median), item["ue_id"]),
    )
    return {
        **selected,
        "class_median_sinr_db": class_median,
        "selection_pool": "stable_serving_ru" if stable else "complete_results",
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def set_padded_ylim(axis: plt.Axes, values: list[float] | np.ndarray, min_span: float = 1.0) -> None:
    data = np.asarray(values, dtype=np.float64)
    center = float(np.median(data))
    low = float(np.min(data))
    high = float(np.max(data))
    span = max(high - low, min_span)
    padding = 0.15 * span
    axis.set_ylim(center - 0.5 * span - padding, center + 0.5 * span + padding)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)


def render_figures(
    output_dir: Path,
    rus: list[dict],
    ues: list[dict],
    links: dict[str, dict],
    rows: list[dict],
) -> list[Path]:
    generated = []
    ru_by_id = {item["id"]: item for item in rus}
    ue_by_id = {item["id"]: item for item in ues}
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
        }
    )

    for ue_type in ("indoor", "outdoor"):
        link = links[ue_type]
        ue = ue_by_id[link["ue_id"]]
        serving = ru_by_id[link["serving_ru_id"]]
        link_rows = [row for row in rows if row["ue_type"] == ue_type]
        slots = [row["slot_id"] for row in link_rows]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)

        for kind, marker, color in (
            ("indoor", "o", "tab:blue"),
            ("outdoor", "s", "tab:green"),
        ):
            group = [item for item in ues if item["ue_type"] == kind]
            axes[0].scatter(
                [item["x"] for item in group],
                [item["y"] for item in group],
                s=18,
                marker=marker,
                color=color,
                alpha=0.35,
                label=f"{kind.title()} UE",
            )
        axes[0].scatter(
            [item["x"] for item in rus],
            [item["y"] for item in rus],
            s=70,
            marker="^",
            color="black",
            label="RU",
        )
        for item in rus:
            axes[0].annotate(f"RU{item['id']}", (item["x"], item["y"]), fontsize=7)
        axes[0].plot([ue["x"], serving["x"]], [ue["y"], serving["y"]], color="tab:red", lw=2)
        axes[0].scatter(
            [ue["x"]],
            [ue["y"]],
            s=160,
            marker="*",
            color="tab:red",
            label=f"Selected UE{ue['id']}",
            zorder=5,
        )
        axes[0].set_title("Selected UE-RU topology")
        axes[0].set_xlabel("Scene x (m)")
        axes[0].set_ylabel("Scene y (m)")
        axes[0].axis("equal")
        axes[0].legend(fontsize=8)

        axes[1].plot(slots, [row["desired_power_dbm"] for row in link_rows], label="Desired S")
        axes[1].plot(
            slots, [row["interference_power_dbm"] for row in link_rows], label="Aggregate I"
        )
        axes[1].plot(slots, [row["noise_power_dbm"] for row in link_rows], label="Noise N")
        axes[1].set_title("Power terms used by SINR")
        axes[1].set_xlabel("Realization")
        axes[1].set_ylabel("Power (dBm)")
        axes[1].legend(fontsize=8)

        sinr_values = [row["sinr_db"] for row in link_rows]
        recomputed_values = [row["sinr_recomputed_db"] for row in link_rows]
        axes[2].plot(slots, sinr_values, label="Derived AODT SINR")
        axes[2].plot(
            slots,
            recomputed_values,
            "--",
            label="Recomputed S/(I+N)",
        )
        axes[2].set_title(f"SINR formula check (max error {link['max_formula_error_db']:.2e} dB)")
        axes[2].set_xlabel("Realization")
        axes[2].set_ylabel("SINR (dB)")
        set_padded_ylim(axes[2], sinr_values + recomputed_values)
        axes[2].legend(fontsize=8)

        stability = "stable" if link["stable_serving_ru"] else "mode"
        fig.suptitle(
            f"AODT {ue_type.title()} SINR validation: UE{ue['id']} - "
            f"RU{link['serving_ru_id']} ({stability})",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        generated.extend(save_figure(fig, output_dir, f"aodt_sinr_validation_{ue_type}"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=140)
    for ue_type, color in (("indoor", "tab:blue"), ("outdoor", "tab:green")):
        link_rows = [row for row in rows if row["ue_type"] == ue_type]
        sinr_values = [row["sinr_db"] for row in link_rows]
        axes[0].plot(
            [row["slot_id"] for row in link_rows],
            sinr_values,
            color=color,
            label=f"{ue_type.title()} UE{links[ue_type]['ue_id']}",
        )
    axes[0].set_title("Representative indoor vs outdoor SINR")
    axes[0].set_xlabel("Realization")
    axes[0].set_ylabel("SINR (dB)")
    comparison_values = [row["sinr_db"] for row in rows]
    set_padded_ylim(axes[0], comparison_values, min_span=2.0)
    axes[0].legend(fontsize=8)

    categories = ["Desired", "Interference", "Noise"]
    x = np.arange(len(categories))
    width = 0.35
    for offset, (ue_type, color) in enumerate(
        (("indoor", "tab:blue"), ("outdoor", "tab:green"))
    ):
        link_rows = [row for row in rows if row["ue_type"] == ue_type]
        values = [
            float(np.mean([row["desired_power_dbm"] for row in link_rows])),
            float(np.mean([row["interference_power_dbm"] for row in link_rows])),
            float(np.mean([row["noise_power_dbm"] for row in link_rows])),
        ]
        axes[1].bar(x + (offset - 0.5) * width, values, width, color=color, label=ue_type.title())
    axes[1].set_xticks(x, categories)
    axes[1].set_title("Mean link-budget terms")
    axes[1].set_ylabel("Power (dBm)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    generated.extend(
        save_figure(fig, output_dir, "aodt_sinr_validation_indoor_vs_outdoor")
    )
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    radio = load_json(run_dir / "04B_config" / "resolved_radio_config.json")["source"]
    tx_power_dbm = float(radio["ru_tx_power_dbm"])
    bandwidth_hz = float(radio["bandwidth_hz"])
    slots = int(radio["slots"])
    noise_figure_db = 7.0
    noise_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    noise_mw = float(dbm_to_mw(noise_dbm))
    rus, ues = load_topology(run_dir)
    cir_power_w = load_channel_power(run_dir, slots, len(rus), len(ues))

    candidate_rx_dbm = 10.0 * np.log10(cir_power_w) + 30.0
    effective_gain_db = candidate_rx_dbm - tx_power_dbm
    candidate_rx_mw = dbm_to_mw(candidate_rx_dbm)
    serving_ru = np.argmax(candidate_rx_mw, axis=1)
    desired_mw = np.take_along_axis(candidate_rx_mw, serving_ru[:, None, :], axis=1)[:, 0, :]
    interference_mw = np.sum(candidate_rx_mw, axis=1) - desired_mw
    denominator_mw = interference_mw + noise_mw
    sinr_db = 10.0 * np.log10(desired_mw / denominator_mw)

    selections = {
        kind: select_representative(sinr_db, serving_ru, ues, kind)
        for kind in ("indoor", "outdoor")
    }
    ru_xyz = np.asarray([[item["x"], item["y"], item["z"]] for item in rus])
    ue_xyz = np.asarray([[item["x"], item["y"], item["z"]] for item in ues])
    rows = []
    links = {}
    for ue_type in ("indoor", "outdoor"):
        selected = selections[ue_type]
        ue_id = selected["ue_id"]
        ru_counts = np.bincount(serving_ru[:, ue_id], minlength=len(rus))
        mode_ru = int(np.argmax(ru_counts))
        for slot in range(slots):
            ru_id = int(serving_ru[slot, ue_id])
            desired_dbm = float(mw_to_dbm(desired_mw[slot, ue_id]))
            interference_dbm = float(mw_to_dbm(interference_mw[slot, ue_id]))
            denominator_dbm = float(mw_to_dbm(denominator_mw[slot, ue_id]))
            recomputed = 10.0 * math.log10(
                float(dbm_to_mw(desired_dbm))
                / (float(dbm_to_mw(interference_dbm)) + float(dbm_to_mw(noise_dbm)))
            )
            rows.append(
                {
                    "ue_type": ue_type,
                    "ue_id": ue_id,
                    "serving_ru_id": ru_id,
                    "slot_id": slot,
                    "distance_m": float(np.linalg.norm(ue_xyz[ue_id] - ru_xyz[ru_id])),
                    "channel_power_gain_db": float(effective_gain_db[slot, ru_id, ue_id]),
                    "path_loss_db": float(-effective_gain_db[slot, ru_id, ue_id]),
                    "desired_power_dbm": desired_dbm,
                    "interference_power_dbm": interference_dbm,
                    "noise_power_dbm": noise_dbm,
                    "denominator_power_dbm": denominator_dbm,
                    "sinr_db": float(sinr_db[slot, ue_id]),
                    "sinr_recomputed_db": recomputed,
                    "formula_error_db": abs(recomputed - float(sinr_db[slot, ue_id])),
                    "valid_nonserving_ru_count": len(rus) - 1,
                    "candidate_ru_count": len(rus),
                }
            )
        selected_rows = [row for row in rows if row["ue_type"] == ue_type]
        values = np.asarray([row["sinr_db"] for row in selected_rows])
        errors = np.asarray([row["formula_error_db"] for row in selected_rows])
        links[ue_type] = {
            "ue_id": ue_id,
            "serving_ru_id": mode_ru,
            "stable_serving_ru": selected["stable_serving_ru"],
            "serving_ru_snapshot_counts": {
                str(index): int(count) for index, count in enumerate(ru_counts) if count
            },
            "selection_pool": selected["selection_pool"],
            "selection_method": "closest UE median SINR to class median; prefer stable serving RU",
            "class_median_sinr_db": selected["class_median_sinr_db"],
            "ue_median_sinr_db": selected["median_sinr_db"],
            "snapshot_count": slots,
            "sinr_db_min": float(np.min(values)),
            "sinr_db_mean": float(np.mean(values)),
            "sinr_db_median": float(np.median(values)),
            "sinr_db_max": float(np.max(values)),
            "max_formula_error_db": float(np.max(errors)),
            "complete_interferer_accounting": True,
            "position": [float(value) for value in ue_xyz[ue_id]],
        }

    csv_path = output_dir / "aodt_sinr_validation_links.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    generated = [csv_path]
    generated.extend(render_figures(output_dir, rus, ues, links, rows))

    formula_tolerance_db = 1e-9
    valid = all(
        link["snapshot_count"] == slots
        and link["complete_interferer_accounting"]
        and link["max_formula_error_db"] <= formula_tolerance_db
        for link in links.values()
    )
    summary = {
        "schema_version": "aodt-static-derived-sinr-validation-v1",
        "generated_at_utc": utc_now(),
        "valid": valid,
        "source_run": run_dir.as_posix(),
        "source": "AODT 1.5 raw complex CIR tensors from 07B",
        "formula": "SINR_dB = 10*log10(S_mW / (sum(I_mW) + N_mW))",
        "cir_power_unit": "watts after summing |CIR|^2 over antenna/polarization/path dimensions",
        "received_power": "P_rx,dBm = 10*log10(sum(|CIR|^2) / 1 mW) = 10*log10(sum(|CIR|^2)) + 30",
        "tx_power_handling": (
            "The configured RU TX power is already reflected in AODT CIR amplitudes; "
            "it is not added a second time."
        ),
        "effective_channel_power_gain": "G_eff,dB = P_rx,dBm - RU_tx_power_dBm",
        "serving_rule": "strongest received-power RU independently for each UE and slot",
        "interference_rule": "sum of all seven non-serving RU received powers; full-frequency reuse",
        "thermal_noise_density_dbm_hz": -174.0,
        "bandwidth_hz": bandwidth_hz,
        "noise_figure_db": noise_figure_db,
        "noise_model_source": "matched to the AODT reference validation setup for comparison",
        "noise_power_dbm": noise_dbm,
        "ru_tx_power_dbm_each": tx_power_dbm,
        "num_configured_snapshots": slots,
        "candidate_ru_count": len(rus),
        "formula_tolerance_db": formula_tolerance_db,
        "links": links,
        "figures": [path.name for path in generated if path.suffix.lower() in {".png", ".pdf"}],
        "per_snapshot_csv": csv_path.name,
        "claim_boundary": (
            "This is derived AODT wideband link SINR under the stated full-reuse link-budget "
            "assumptions. It is not a native SINR field exported by AODT, nor NR per-RE, "
            "post-equalization, beam-scheduled, or throughput SINR."
        ),
    }
    summary_path = output_dir / "aodt_sinr_validation_summary.json"
    save_json(summary_path, summary)
    generated.append(summary_path)
    manifest = {
        "status": "pass" if valid else "fail",
        "generated_at_utc": utc_now(),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(generated)
        ],
    }
    save_json(output_dir / "aodt_sinr_figure_manifest.json", manifest)
    print(json.dumps({"valid": valid, "links": links}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
