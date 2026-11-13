from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


FILE_PATTERN = re.compile(r"^cir_slot_(\d{4})_ru_(\d{2})\.npz$")
INDOOR_COLOR = "#0072B2"
OUTDOOR_COLOR = "#D55E00"
RU_COLOR = "#202020"
GRID_COLOR = "#D8D8D8"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def latest_08b_run(repo_root: Path) -> Path:
    root = repo_root / "results" / "aodt_static"
    candidates: list[tuple[float, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"AODT results root not found: {root}")
    for run_dir in root.iterdir():
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json_load(manifest_path)
        if manifest.get("stages", {}).get("08B", {}).get("status") in {
            "pass",
            "pass_with_warnings",
        }:
            candidates.append((manifest_path.stat().st_mtime, run_dir))
    if not candidates:
        raise FileNotFoundError("No passing AODT 08B run was found.")
    return max(candidates, key=lambda item: item[0])[1]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return [png, pdf]


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.sort(values[np.isfinite(values)])
    if finite.size == 0:
        return finite, finite
    probability = np.arange(1, finite.size + 1, dtype=float) / finite.size
    return finite, probability


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def aggregate(run_dir: Path) -> dict[str, Any]:
    raw_dir = run_dir / "07B_raw_collection" / "raw"
    scenario_path = Path(
        json_load(run_dir / "run_manifest.json")["inputs"]["scenario_path"]
    )
    scenario = json_load(scenario_path)
    positions = json_load(run_dir / "06B_execution" / "position_registry.json")
    raw_files = sorted(raw_dir.glob("*.npz"))
    if len(raw_files) != 2048:
        raise RuntimeError(f"Expected 2048 raw NPZ files, found {len(raw_files)}.")

    ue_types = np.asarray(
        [str(item["ue_type"]).lower() for item in scenario["ue"]], dtype="U16"
    )
    if np.count_nonzero(ue_types == "indoor") != 40:
        raise RuntimeError("Scenario does not contain the required 40 indoor UEs.")
    if np.count_nonzero(ue_types == "outdoor") != 10:
        raise RuntimeError("Scenario does not contain the required 10 outdoor UEs.")

    n_links = len(raw_files) * len(ue_types)
    slot_ids = np.empty(n_links, dtype=np.int16)
    ru_ids = np.empty(n_links, dtype=np.int8)
    ue_ids = np.empty(n_links, dtype=np.int8)
    type_codes = np.empty(n_links, dtype=np.int8)
    path_counts = np.empty(n_links, dtype=np.int16)
    coefficient_power_db = np.empty(n_links, dtype=np.float64)
    rms_delay_spread_us = np.empty(n_links, dtype=np.float64)
    mean_excess_delay_us = np.empty(n_links, dtype=np.float64)

    azimuth_edges = np.linspace(0.0, 2.0 * math.pi, 73)
    zenith_edges = np.linspace(0.0, math.pi, 37)
    angular_histograms = {
        kind: {
            category: {
                "aod_azimuth": np.zeros(72, dtype=np.float64),
                "aod_zenith": np.zeros(36, dtype=np.float64),
                "aoa_azimuth": np.zeros(72, dtype=np.float64),
                "aoa_zenith": np.zeros(36, dtype=np.float64),
            }
            for category in ("indoor", "outdoor")
        }
        for kind in ("power_weighted",)
    }

    cursor = 0
    slot0_ru7: dict[str, np.ndarray] | None = None
    for file_path in raw_files:
        match = FILE_PATTERN.match(file_path.name)
        if match is None:
            raise RuntimeError(f"Invalid raw filename: {file_path.name}")
        slot = int(match.group(1))
        ru = int(match.group(2))
        with np.load(file_path, allow_pickle=False) as data:
            values = np.asarray(data["values"])
            delays = np.asarray(data["delays"]).reshape(len(ue_types), -1, 500)[:, 0, :]
            aod = np.asarray(data["angles_of_departure"]).reshape(
                len(ue_types), -1, 500, 2
            )[:, 0, :, :]
            aoa = np.asarray(data["angles_of_arrival"]).reshape(
                len(ue_types), -1, 500, 2
            )[:, 0, :, :]

            magnitude_squared = np.square(np.abs(values), dtype=np.float64)
            tap_power = np.sum(
                magnitude_squared,
                axis=tuple(range(1, magnitude_squared.ndim - 1)),
            )
            active = tap_power > 0.0
            total_power = np.sum(tap_power, axis=1)
            if np.any(total_power <= 0.0):
                raise RuntimeError(f"Zero-power logical link in {file_path.name}")
            normalized_weights = tap_power / total_power[:, None]
            mean_delay = np.sum(normalized_weights * delays, axis=1)
            variance = np.sum(
                normalized_weights * np.square(delays - mean_delay[:, None]),
                axis=1,
            )
            rms_delay = np.sqrt(np.maximum(variance, 0.0))

            sl = slice(cursor, cursor + len(ue_types))
            slot_ids[sl] = slot
            ru_ids[sl] = ru
            ue_ids[sl] = np.arange(len(ue_types), dtype=np.int8)
            type_codes[sl] = (ue_types == "outdoor").astype(np.int8)
            path_counts[sl] = np.sum(active, axis=1)
            coefficient_power_db[sl] = 10.0 * np.log10(total_power)
            rms_delay_spread_us[sl] = rms_delay * 1e6
            mean_excess_delay_us[sl] = mean_delay * 1e6
            cursor += len(ue_types)

            for category in ("indoor", "outdoor"):
                ue_mask = ue_types == category
                selected_power = tap_power[ue_mask]
                selected_aod = aod[ue_mask]
                selected_aoa = aoa[ue_mask]
                selected_active = active[ue_mask]
                weights = selected_power[selected_active]
                target = angular_histograms["power_weighted"][category]
                target["aod_azimuth"] += np.histogram(
                    selected_aod[:, :, 0][selected_active],
                    bins=azimuth_edges,
                    weights=weights,
                )[0]
                target["aod_zenith"] += np.histogram(
                    selected_aod[:, :, 1][selected_active],
                    bins=zenith_edges,
                    weights=weights,
                )[0]
                target["aoa_azimuth"] += np.histogram(
                    selected_aoa[:, :, 0][selected_active],
                    bins=azimuth_edges,
                    weights=weights,
                )[0]
                target["aoa_zenith"] += np.histogram(
                    selected_aoa[:, :, 1][selected_active],
                    bins=zenith_edges,
                    weights=weights,
                )[0]

            if slot == 0 and ru == 7:
                slot0_ru7 = {
                    "tap_power": tap_power.copy(),
                    "delays": delays.copy(),
                    "total_power": total_power.copy(),
                }

    if cursor != n_links or slot0_ru7 is None:
        raise RuntimeError("Raw aggregation did not cover the expected link grid.")

    return {
        "scenario": scenario,
        "positions": positions,
        "ue_types": ue_types,
        "slot_ids": slot_ids,
        "ru_ids": ru_ids,
        "ue_ids": ue_ids,
        "type_codes": type_codes,
        "path_counts": path_counts,
        "coefficient_power_db": coefficient_power_db,
        "rms_delay_spread_us": rms_delay_spread_us,
        "mean_excess_delay_us": mean_excess_delay_us,
        "angular_histograms": angular_histograms,
        "azimuth_edges": azimuth_edges,
        "zenith_edges": zenith_edges,
        "slot0_ru7": slot0_ru7,
    }


def plot_layout(data: dict[str, Any], output_dir: Path) -> list[Path]:
    scenario = data["scenario"]
    ru = np.asarray(
        [[item["x"], item["y"], item["z"]] for item in scenario["bs"]],
        dtype=float,
    )
    ue = np.asarray(
        [[item["x"], item["y"], item["z"]] for item in scenario["ue"]],
        dtype=float,
    )
    ue_types = data["ue_types"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for category, color, marker in (
        ("indoor", INDOOR_COLOR, "o"),
        ("outdoor", OUTDOOR_COLOR, "s"),
    ):
        mask = ue_types == category
        ax.scatter(
            ue[mask, 0],
            ue[mask, 1],
            s=34,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=f"{category.title()} UE (n={int(np.count_nonzero(mask))})",
            zorder=3,
        )
    ax.scatter(
        ru[:, 0],
        ru[:, 1],
        s=105,
        marker="^",
        facecolor=RU_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="RU (n=8)",
        zorder=4,
    )
    for index, (x_value, y_value) in enumerate(ru[:, :2]):
        ax.annotate(
            f"RU{index}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    ax.set_title("AODT Tokyo fixed topology")
    ax.set_xlabel("AODT scene x-coordinate (m)")
    ax.set_ylabel("AODT scene y-coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="best")
    return save_figure(fig, output_dir, "aodt_scene_device_layout_xy")


def plot_ecdf_metric(
    values: np.ndarray,
    type_codes: np.ndarray,
    xlabel: str,
    title: str,
    stem: str,
    output_dir: Path,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.8, 4.5), constrained_layout=True)
    for code, label, color, linestyle in (
        (0, "Indoor UE", INDOOR_COLOR, "-"),
        (1, "Outdoor UE", OUTDOOR_COLOR, "--"),
    ):
        x_value, y_value = ecdf(values[type_codes == code])
        ax.step(
            x_value,
            y_value,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{label} (n={len(x_value):,})",
        )
        if x_value.size:
            median = float(np.median(x_value))
            ax.plot(
                [median],
                [0.5],
                marker="o" if code == 0 else "s",
                color=color,
                markersize=5,
            )
            ax.annotate(
                f"{median:.2f}",
                (median, 0.5),
                xytext=(5, 4 if code == 0 else -12),
                textcoords="offset points",
                color=color,
                fontsize=8.5,
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    return save_figure(fig, output_dir, stem)


def plot_pdp(data: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, int]]:
    ue_types = data["ue_types"]
    sample = data["slot0_ru7"]
    tap_power = sample["tap_power"]
    delays = sample["delays"]
    total_power_db = 10.0 * np.log10(sample["total_power"])
    selected: dict[str, int] = {}
    for category in ("indoor", "outdoor"):
        candidates = np.flatnonzero(ue_types == category)
        median = np.median(total_power_db[candidates])
        selected[category] = int(
            candidates[np.argmin(np.abs(total_power_db[candidates] - median))]
        )

    fig, axes = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=False, constrained_layout=True
    )
    for ax, category, color in zip(
        axes, ("indoor", "outdoor"), (INDOOR_COLOR, OUTDOOR_COLOR)
    ):
        ue = selected[category]
        active = tap_power[ue] > 0.0
        x_value = delays[ue, active] * 1e6
        y_value = 10.0 * np.log10(tap_power[ue, active] / np.max(tap_power[ue, active]))
        visible = y_value >= -60.0
        ax.vlines(
            x_value[visible],
            -60.0,
            y_value[visible],
            color=color,
            linewidth=0.75,
            alpha=0.75,
        )
        ax.scatter(
            x_value[visible],
            y_value[visible],
            s=12,
            marker="o" if category == "indoor" else "s",
            color=color,
            edgecolor="none",
            zorder=3,
        )
        ax.set_title(
            f"RU7 → {category} UE{ue}, slot 0 "
            f"({int(np.count_nonzero(active))} active paths)"
        )
        ax.set_xlabel("Propagation delay (μs)")
        ax.set_ylabel("Normalized tap power (dB)")
        ax.set_ylim(-61.0, 2.0)
    return (
        save_figure(fig, output_dir, "aodt_ru7_representative_pdp"),
        selected,
    )


def plot_angles(data: dict[str, Any], output_dir: Path) -> list[Path]:
    hist = data["angular_histograms"]["power_weighted"]
    az_centers = np.rad2deg(
        0.5 * (data["azimuth_edges"][:-1] + data["azimuth_edges"][1:])
    )
    ze_centers = np.rad2deg(
        0.5 * (data["zenith_edges"][:-1] + data["zenith_edges"][1:])
    )
    panels = (
        ("aod_azimuth", az_centers, "AoD azimuth (degrees)"),
        ("aod_zenith", ze_centers, "AoD zenith (degrees)"),
        ("aoa_azimuth", az_centers, "AoA azimuth (degrees)"),
        ("aoa_zenith", ze_centers, "AoA zenith (degrees)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    for ax, (field, centers, label) in zip(axes.ravel(), panels):
        for category, color, linestyle in (
            ("indoor", INDOOR_COLOR, "-"),
            ("outdoor", OUTDOOR_COLOR, "--"),
        ):
            values = hist[category][field]
            probability = values / np.sum(values)
            ax.plot(
                centers,
                probability,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=f"{category.title()} UE",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Power-weighted probability")
        ax.set_ylim(bottom=0.0)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("AODT power-weighted multipath angle distributions", fontsize=11)
    return save_figure(fig, output_dir, "aodt_power_weighted_angular_distributions")


def make_overview(png_files: list[Path], output_dir: Path) -> Path:
    thumbnails: list[Image.Image] = []
    target = (900, 600)
    for path in png_files:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(target, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", target, "white")
            left = (target[0] - image.width) // 2
            top = (target[1] - image.height) // 2
            canvas.paste(image, (left, top))
            thumbnails.append(ImageOps.expand(canvas, border=2, fill="#D8D8D8"))
    columns = 2
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * 904, rows * 604), "white")
    for index, image in enumerate(thumbnails):
        x_value = (index % columns) * 904
        y_value = (index // columns) * 604
        sheet.paste(image, (x_value, y_value))
    path = output_dir / "aodt_08b_figure_overview.png"
    sheet.save(path, format="PNG", optimize=True)
    return path


def create_figures(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    configure_style()
    data = aggregate(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    generated: list[Path] = []
    generated.extend(plot_layout(data, output_dir))
    generated.extend(
        plot_ecdf_metric(
            data["coefficient_power_db"],
            data["type_codes"],
            "Aggregated CIR coefficient power, 10 log₁₀(Σ|h|²) (dB re 1)",
            "AODT link-level CIR coefficient power",
            "aodt_link_coefficient_power_ecdf",
            output_dir,
        )
    )
    generated.extend(
        plot_ecdf_metric(
            data["path_counts"].astype(float),
            data["type_codes"],
            "Active multipath count per link",
            "AODT multipath richness",
            "aodt_path_count_ecdf",
            output_dir,
        )
    )
    generated.extend(
        plot_ecdf_metric(
            data["rms_delay_spread_us"],
            data["type_codes"],
            "RMS delay spread (μs)",
            "AODT RMS delay spread",
            "aodt_rms_delay_spread_ecdf",
            output_dir,
        )
    )
    pdp_files, selected_ues = plot_pdp(data, output_dir)
    generated.extend(pdp_files)
    generated.extend(plot_angles(data, output_dir))

    png_files = [
        path
        for path in generated
        if path.suffix.lower() == ".png"
    ]
    overview = make_overview(png_files, output_dir)
    generated.append(overview)

    type_codes = data["type_codes"]
    summaries = {}
    for code, category in ((0, "indoor"), (1, "outdoor")):
        mask = type_codes == code
        summaries[category] = {
            "logical_link_count": int(np.count_nonzero(mask)),
            "coefficient_power_db_re_1": stats(data["coefficient_power_db"][mask]),
            "path_count": stats(data["path_counts"][mask]),
            "rms_delay_spread_us": stats(data["rms_delay_spread_us"][mask]),
            "mean_excess_delay_us": stats(data["mean_excess_delay_us"][mask]),
        }

    return {
        "schema_version": "aodt-static-08b-figure-manifest-v1",
        "generated_at_utc": utc_now(),
        "status": "pass",
        "source_run": run_dir.as_posix(),
        "source_stage": (run_dir / "07B_raw_collection").as_posix(),
        "figure_directory": output_dir.as_posix(),
        "figures": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(generated)
        ],
        "representative_pdp_selection": {
            "rule": "UE closest to the median RU7 slot-0 aggregate CIR coefficient power within each UE type",
            "ru": 7,
            "slot": 0,
            "indoor_ue": selected_ues["indoor"],
            "outdoor_ue": selected_ues["outdoor"],
        },
        "summaries": summaries,
        "claim_boundary": (
            "10log10(sum(|h|^2)) is reported in dB relative to a unit coefficient. "
            "It is not received power, RSRP, path gain/loss, SNR, or SINR. PDP is "
            "normalized per representative link. No AODT value or random RF "
            "placeholder is used."
        ),
    }


def update_run_manifest(
    run_dir: Path, figure_manifest: dict[str, Any], command: list[str]
) -> None:
    path = run_dir / "run_manifest.json"
    manifest = json_load(path)
    stage = manifest.setdefault("stages", {}).setdefault("08B", {})
    stage["figures"] = {
        "status": "pass",
        "generated_at_utc": figure_manifest["generated_at_utc"],
        "directory": figure_manifest["figure_directory"],
        "manifest": (
            Path(figure_manifest["figure_directory"]) / "figure_manifest.json"
        ).as_posix(),
        "png_count": sum(
            item["name"].endswith(".png") for item in figure_manifest["figures"]
        ),
        "pdf_count": sum(
            item["name"].endswith(".pdf") for item in figure_manifest["figures"]
        ),
        "command": command,
    }
    manifest["updated_at_utc"] = figure_manifest["generated_at_utc"]
    json_dump(path, manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate paper-ready AODT channel figures from a passing 08B run."
    )
    result.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    result.add_argument("--run-dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="Do not update the run manifest; useful for a preview render.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_08b_run(repo_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    quality = json_load(
        run_dir / "08B_raw_validation" / "aodt_raw_quality_report.json"
    )
    if quality.get("status") not in {"pass", "pass_with_warnings"}:
        raise RuntimeError("08B raw validation is not passing.")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "08B_raw_validation" / "figures"
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite figure directory: {output_dir}")

    print(f"[08B figures] aggregating {run_dir}", flush=True)
    figure_manifest = create_figures(run_dir, output_dir)
    json_dump(output_dir / "figure_manifest.json", figure_manifest)
    if not args.no_manifest_update:
        update_run_manifest(run_dir, figure_manifest, list(sys.argv))
    print(
        f"[08B figures] status=pass files={len(figure_manifest['figures']) + 1} "
        f"output={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
