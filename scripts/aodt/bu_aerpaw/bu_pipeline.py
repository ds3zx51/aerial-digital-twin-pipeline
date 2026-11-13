from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import io
import json
import math
import os
import shutil
import statistics
import sys
import tarfile
import zipfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aodt_native_builder import build_execution_spec
from bu_common import (
    STAGES,
    StageBlocked,
    StageContext,
    hash_tree,
    load_upstream,
    read_csv,
    read_json,
    require_dir,
    require_file,
    run_external,
    run_stage,
    sha256_file,
    verify_declared_hashes,
    write_csv,
    write_json,
)


TOWERS = ["LW1", "LW2", "LW3", "LW4", "LW5"]
MEASUREMENT_FIELDS = [
    "tower_id", "sample_id", "timestamp_s", "longitude_deg", "latitude_deg",
    "raw_altitude_m", "usrp1_rss_dbm", "usrp2_rss_dbm", "source_mat",
]
TRAJECTORY_FIELDS = [
    "sample_id", "timestamp_s", "longitude_deg", "latitude_deg", "raw_altitude_m",
    "simulation_height_agl_m", "terrain_elevation_m", "simulation_altitude_m",
    "scene_x_m", "scene_y_m", "scene_z_m", "altitude_was_clamped",
]
ALLOWED_PROVENANCE = {
    "AODT_NATIVE", "AODT_DERIVED", "MEASUREMENT_SUPPORTED", "SCENARIO_METADATA",
    "PROJECT_MODEL", "PROJECT_LABEL",
}


def _input_map(args: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in args.get("input", []) or []:
        if "=" not in item:
            raise StageBlocked("invalid_input_mapping", f"Use --input BUxx=PATH, received: {item}")
        key, value = item.split("=", 1)
        key = key.upper()
        if key in result:
            raise StageBlocked("duplicate_input_mapping", f"Duplicate upstream mapping: {key}")
        result[key] = value
    return result


def _upstream(ctx: StageContext, stage: str) -> tuple[Path, dict[str, Any]]:
    mapping = _input_map(ctx.args)
    path, manifest = load_upstream(mapping.get(stage), stage)
    verify_declared_hashes(path)
    return path, manifest


def _source_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in sorted(archive.infolist(), key=lambda x: x.filename):
                if item.is_dir():
                    continue
                with archive.open(item) as handle:
                    digest = __import__("hashlib").sha256(handle.read()).hexdigest()
                rows.append({"relative_path": item.filename, "size_bytes": item.file_size, "sha256": digest, "container": str(path)})
    elif path.is_dir():
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            rows.append({
                "relative_path": item.relative_to(path).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
                "container": str(path),
            })
    else:
        raise StageBlocked("unsupported_source", f"Source must be a directory or ZIP: {path}")
    return rows


def _raw_mat_name(entries: Iterable[str], tower: str) -> str:
    suffix = f"GPS_RSS_data_40m_fs20MHz_{tower}.mat"
    matches = [name for name in entries if name.endswith(suffix) and not Path(name).name.startswith("Converted_")]
    if len(matches) != 1:
        raise StageBlocked("raw_mat_ambiguous", f"Expected exactly one raw MAT for {tower}; found {matches}")
    return matches[0]


def _load_mat(source: Path, relative_name: str) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise StageBlocked("missing_dependency", "BU02 requires scipy; install requirements.txt") from exc
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            return loadmat(io.BytesIO(archive.read(relative_name)))
    return loadmat(source / Path(relative_name))


def _flat(values: Any) -> list[float]:
    import numpy as np

    return [float(value) for value in np.asarray(values).reshape(-1)]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clamp_height(raw_altitude_m: float, minimum_m: float = 0.5) -> tuple[float, bool]:
    value = float(raw_altitude_m)
    return (max(value, minimum_m), value < minimum_m)


def geodetic_transformer(origin_lat: float, origin_lon: float):
    # Dependency-free WGS84 ECEF <-> local ENU transform. This avoids opaque
    # coordinate scaling and makes the exact axis convention auditable.
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat0 = math.radians(origin_lat)
    lon0 = math.radians(origin_lon)

    def to_ecef(lon_deg: float, lat_deg: float, height_m: float = 0.0) -> tuple[float, float, float]:
        lat, lon = math.radians(lat_deg), math.radians(lon_deg)
        radius = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        return (
            (radius + height_m) * math.cos(lat) * math.cos(lon),
            (radius + height_m) * math.cos(lat) * math.sin(lon),
            (radius * (1.0 - e2) + height_m) * math.sin(lat),
        )

    def from_ecef(x: float, y: float, z: float) -> tuple[float, float, float]:
        lon = math.atan2(y, x)
        p = math.hypot(x, y)
        lat = math.atan2(z, p * (1.0 - e2))
        height = 0.0
        for _ in range(12):
            radius = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
            height = p / max(math.cos(lat), 1e-15) - radius
            updated = math.atan2(z, p * (1.0 - e2 * radius / (radius + height)))
            if abs(updated - lat) < 1e-14:
                lat = updated
                break
            lat = updated
        return math.degrees(lon), math.degrees(lat), height

    x0, y0, z0 = to_ecef(origin_lon, origin_lat)

    class Forward:
        @staticmethod
        def transform(lon_deg: float, lat_deg: float) -> tuple[float, float]:
            x, y, z = to_ecef(lon_deg, lat_deg)
            dx, dy, dz = x - x0, y - y0, z - z0
            east = -math.sin(lon0) * dx + math.cos(lon0) * dy
            north = -math.sin(lat0) * math.cos(lon0) * dx - math.sin(lat0) * math.sin(lon0) * dy + math.cos(lat0) * dz
            return east, north

    class Inverse:
        @staticmethod
        def transform(east: float, north: float) -> tuple[float, float]:
            # Horizontal contract uses zero local Up for the 2D inverse test.
            dx = -math.sin(lon0) * east - math.sin(lat0) * math.cos(lon0) * north
            dy = math.cos(lon0) * east - math.sin(lat0) * math.sin(lon0) * north
            dz = math.cos(lat0) * north
            lon_deg, lat_deg, _ = from_ecef(x0 + dx, y0 + dy, z0 + dz)
            return lon_deg, lat_deg

    local_definition = f"WGS84 local ENU; origin lat={origin_lat}, lon={origin_lon}, h=0; x=east,y=north,z=up"
    return local_definition, Forward(), Inverse()


def build_geospatial_contract(contract: dict[str, Any]) -> dict[str, Any]:
    origin = contract["tower_coordinates"]["LW1"]
    local, forward, inverse = geodetic_transformer(origin["latitude_deg"], origin["longitude_deg"])
    x, y = forward.transform(origin["longitude_deg"], origin["latitude_deg"])
    lon2, lat2 = inverse.transform(x, y)
    return {
        "schema_version": "1.0.0",
        "geodetic_crs": "EPSG:4326",
        "local_crs_definition": local,
        "origin": {"latitude_deg": origin["latitude_deg"], "longitude_deg": origin["longitude_deg"]},
        "axis_order": {"x": "east", "y": "north", "z": "up"},
        "linear_unit": "metre",
        "vertical_datum": "UNRESOLVED_FOR_FORMAL_SCENE",
        "scene_transform": "x=east,y=north,z=up relative to origin; formal scene manifest must confirm",
        "roundtrip_error_deg": max(abs(lon2 - origin["longitude_deg"]), abs(lat2 - origin["latitude_deg"])),
        "formal_scene_alignment_status": "missing_verified_scene_manifest",
    }


def validate_foliage_geojson(path: Path) -> dict[str, Any]:
    data = read_json(require_file(path, "missing_foliage_layer"))
    if data.get("type") != "FeatureCollection":
        raise StageBlocked("invalid_foliage_layer", "Foliage layer must be a GeoJSON FeatureCollection")
    features = data.get("features")
    if not isinstance(features, list) or not features:
        raise StageBlocked("empty_foliage_layer", "Formal foliage GeoJSON is empty/template-only")
    return data


def validate_scene_manifest(path: Path) -> dict[str, Any]:
    data = read_json(require_file(path, "missing_scene_manifest"))
    required = {"scene_url", "scene_sha256", "geodetic_crs", "origin", "axis_order", "vertical_datum", "terrain_samples_csv", "tree_inventory_csv", "random_seed"}
    missing = sorted(required - set(data))
    if missing:
        raise StageBlocked("incomplete_scene_manifest", f"Scene manifest missing: {missing}")
    terrain = (path.parent / data["terrain_samples_csv"]).resolve()
    require_file(terrain, "missing_terrain_samples")
    tree_inventory = (path.parent / data["tree_inventory_csv"]).resolve()
    require_file(tree_inventory, "missing_tree_inventory")
    return data


def write_gpx(path: Path, trajectory: list[dict[str, Any]]) -> None:
    import xml.etree.ElementTree as ET

    root = ET.Element("gpx", version="1.1", creator="AERPAW-BU05", xmlns="http://www.topografix.com/GPX/1/1")
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "desc").text = "Audit-only GPX; AODT execution uses explicit 3D waypoints"
    track = ET.SubElement(root, "trk")
    ET.SubElement(track, "name").text = "Paper-referenced Lake Wheeler coverage route"
    segment = ET.SubElement(track, "trkseg")
    from datetime import datetime, timezone

    for row in trajectory:
        point = ET.SubElement(segment, "trkpt", lat=str(row["latitude_deg"]), lon=str(row["longitude_deg"]))
        ET.SubElement(point, "ele").text = str(row["simulation_altitude_m"])
        ET.SubElement(point, "time").text = datetime.fromtimestamp(float(row["timestamp_s"]), timezone.utc).isoformat().replace("+00:00", "Z")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def cir_to_rss_dbm(cir_values: Iterable[complex], semantics: str, tx_power_dbm: float | None = None) -> float | None:
    values = list(cir_values)
    if not values:
        return None
    if semantics != "cir_amplitude_sqrt_w_includes_tx_power":
        raise StageBlocked("unsupported_power_semantics", f"Unsupported CIR semantics: {semantics}")
    power_w = sum(abs(value) ** 2 for value in values)
    if power_w <= 0:
        return None
    # AODT CIR already contains configured RU power. tx_power_dbm is deliberately not added again.
    return 10.0 * math.log10(power_w) + 30.0


def received_power_w_to_dbm(power_w: float | None) -> float | None:
    if power_w is None or power_w <= 0:
        return None
    return 10.0 * math.log10(power_w) + 30.0


def nearest_index(sorted_values: list[float], target: float, tolerance: float) -> int | None:
    if not sorted_values:
        return None
    position = bisect.bisect_left(sorted_values, target)
    candidates = [idx for idx in (position - 1, position) if 0 <= idx < len(sorted_values)]
    best = min(candidates, key=lambda idx: abs(sorted_values[idx] - target))
    return best if abs(sorted_values[best] - target) <= tolerance else None


def outer_align(measurements: list[dict[str, Any]], simulated: list[dict[str, Any]], tolerance_s: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used_sim: set[int] = set()
    by_tower: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(simulated):
        by_tower.setdefault(str(row["tower_id"]), []).append((index, row))
    for values in by_tower.values():
        values.sort(key=lambda item: float(item[1]["timestamp_s"]))
    for measurement in measurements:
        tower = str(measurement["tower_id"])
        candidates = by_tower.get(tower, [])
        timestamps = [float(item[1]["timestamp_s"]) for item in candidates]
        relative = nearest_index(timestamps, float(measurement["timestamp_s"]), tolerance_s)
        sim_index, sim_row = candidates[relative] if relative is not None else (-1, {})
        if sim_index >= 0 and sim_index in used_sim:
            sim_index, sim_row = -1, {}
        if sim_index >= 0:
            used_sim.add(sim_index)
        result.append({
            **measurement,
            "aodt_timestamp_s": sim_row.get("timestamp_s"),
            "aodt_rss_dbm": sim_row.get("aodt_rss_dbm"),
            "aodt_path_status": sim_row.get("path_status"),
            "alignment_status": "matched" if sim_index >= 0 else "measurement_only",
            "time_error_s": abs(float(measurement["timestamp_s"]) - float(sim_row["timestamp_s"])) if sim_row else None,
        })
    for index, row in enumerate(simulated):
        if index not in used_sim:
            result.append({
                "tower_id": row["tower_id"], "sample_id": None, "timestamp_s": None,
                "usrp1_rss_dbm": None, "usrp2_rss_dbm": None,
                "aodt_timestamp_s": row["timestamp_s"], "aodt_rss_dbm": row.get("aodt_rss_dbm"),
                "aodt_path_status": row.get("path_status"), "alignment_status": "aodt_only", "time_error_s": None,
            })
    return result


def contiguous_splits(count: int) -> dict[str, range]:
    if count < 5:
        raise StageBlocked("insufficient_samples", "At least five matched samples are required")
    calibration_end = max(1, int(count * 0.2))
    validation_end = max(calibration_end + 1, int(count * 0.4))
    return {
        "calibration": range(0, calibration_end),
        "validation": range(calibration_end, validation_end),
        "test": range(validation_end, count),
    }


def best_offset(measured: list[float], simulated: list[float], minimum: float = -50, maximum: float = 50, step: float = 0.1) -> float:
    if len(measured) != len(simulated) or not measured:
        raise StageBlocked("invalid_calibration_arrays", "Calibration arrays must be non-empty and equal length")
    candidates = int(round((maximum - minimum) / step)) + 1
    return min(
        (minimum + i * step for i in range(candidates)),
        key=lambda offset: math.sqrt(sum((m - (s + offset)) ** 2 for m, s in zip(measured, simulated)) / len(measured)),
    )


def regression_metrics(measured: list[float], simulated: list[float]) -> dict[str, float]:
    errors = [s - m for m, s in zip(measured, simulated)]
    return {
        "count": len(errors),
        "mae_db": sum(abs(value) for value in errors) / len(errors),
        "rmse_db": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "bias_db": sum(errors) / len(errors),
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for cursor in range(index, end):
            ranks[order[cursor]] = rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left, mean_right = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right))
    return numerator / denominator if denominator else None


def extended_metrics(measured: list[float], simulated: list[float], noise_floor_dbm: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = regression_metrics(measured, simulated)
    result["pearson_r"] = _correlation(measured, simulated)
    result["spearman_r"] = _correlation(_average_ranks(measured), _average_ranks(simulated))
    result.update({"outage_tp": None, "outage_fp": None, "outage_fn": None, "outage_tn": None, "outage_precision": None, "outage_recall": None, "outage_f1": None})
    if noise_floor_dbm is not None:
        truth = [value <= noise_floor_dbm for value in measured]
        predicted = [value <= noise_floor_dbm for value in simulated]
        tp = sum(t and p for t, p in zip(truth, predicted))
        fp = sum((not t) and p for t, p in zip(truth, predicted))
        fn = sum(t and (not p) for t, p in zip(truth, predicted))
        tn = sum((not t) and (not p) for t, p in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        result.update({"outage_tp": tp, "outage_fp": fp, "outage_fn": fn, "outage_tn": tn, "outage_precision": precision, "outage_recall": recall, "outage_f1": f1})
    return result


def _write_bu13_plots(result_dir: Path, rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise StageBlocked("missing_dependency", "BU13 plotting requires matplotlib") from exc

    def series(tower: str, channel: str) -> list[dict[str, Any]]:
        return sorted((row for row in rows if row["tower_id"] == tower and row["channel"] == channel), key=lambda row: float(row["elapsed_s"]))

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
    for axis, tower in zip(axes, TOWERS):
        one, two = series(tower, "USRP1"), series(tower, "USRP2")
        if one:
            axis.plot([float(r["elapsed_s"]) for r in one], [float(r["aodt_rss_raw_dbm"]) for r in one], label="AODT raw", lw=1.2)
            axis.plot([float(r["elapsed_s"]) for r in one], [float(r["measured_rss_dbm"]) for r in one], label="USRP1 raw", lw=0.8)
        if two:
            axis.plot([float(r["elapsed_s"]) for r in two], [float(r["measured_rss_dbm"]) for r in two], label="USRP2 raw", lw=0.8)
        no_path = [r for r in all_rows if r["tower_id"] == tower and r.get("aodt_rss_dbm") in (None, "")]
        if no_path:
            bottom = axis.get_ylim()[0]
            axis.scatter([float(r["elapsed_s"]) for r in no_path], [bottom] * len(no_path), marker="x", s=10, label="Z / no path")
        labels_seen: set[str] = set()
        for row in one:
            label = str(row.get("waypoint_label") or "")
            if label and label not in labels_seen:
                axis.axvline(float(row["elapsed_s"]), color="0.6", ls="--", lw=0.6)
                axis.text(float(row["elapsed_s"]), axis.get_ylim()[1], label, va="top", fontsize=7)
                labels_seen.add(label)
        axis.set_ylabel(f"{tower}\nRSS (dBm)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("BU13 View A: raw AODT vs dual-USRP RSS")
    fig.savefig(result_dir / "fig_bu13_aodt_vs_usrp_timeseries.png", dpi=180)
    fig.savefig(result_dir / "fig_bu13_raw.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
    for axis, tower in zip(axes, TOWERS):
        for channel, color in (("USRP1", "tab:orange"), ("USRP2", "tab:green")):
            values = series(tower, channel)
            if not values:
                continue
            x = [float(r["elapsed_s"]) for r in values]
            axis.plot(x, [float(r["measured_rss_dbm"]) for r in values], color=color, lw=0.8, label=f"{channel} raw")
            axis.plot(x, [float(r["aodt_rss_paper_style_dbm"]) for r in values], lw=0.8, ls="--", label=f"AODT paper-style for {channel}")
        axis.set_ylabel(f"{tower}\nRSS (dBm)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=7)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("BU13 View B: paper_style_calibration (descriptive, not blind)")
    fig.savefig(result_dir / "fig_bu13_calibrated.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for field, label in (("measured_rss_dbm", "USRP raw"), ("aodt_rss_raw_dbm", "AODT raw"), ("aodt_rss_heldout_dbm", "AODT held-out calibrated")):
        values = sorted(float(row[field]) for row in rows if row.get(field) not in (None, ""))
        if values:
            axis.plot(values, [(index + 1) / len(values) for index in range(len(values))], label=label)
    axis.set_xlabel("RSS (dBm)")
    axis.set_ylabel("Empirical CDF")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(result_dir / "fig_bu13_cdf.png", dpi=180)
    plt.close(fig)


COLOSSEUM_COLUMNS = [
    "traffic_type", "Timestamp", "num_ues", "slice_prb", "power_multiplier", "dl_mcs",
    "dl_n_samples", "dl_buffer [bytes]", "tx_brate downlink [Mbps]", "tx_pkts downlink",
    "tx_errors downlink (%)", "dl_cqi", "ul_mcs", "ul_n_samples", "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]", "rx_pkts uplink", "rx_errors uplink (%)", "ul_rssi", "ul_sinr",
    "phr", "sum_requested_prbs", "sum_granted_prbs", "dl_pmi", "dl_ri", "ul_n",
    "ul_turbo_iters", "IMSI", "RNTI", "slicing_enabled", "slice_id", "scheduling_policy",
]


def field_provenance_rows() -> list[dict[str, str]]:
    rows = []
    for column in COLOSSEUM_COLUMNS:
        if column == "traffic_type":
            source = "PROJECT_LABEL"
        elif column == "ul_rssi":
            source = "AODT_DERIVED"
        elif column == "Timestamp":
            source = "MEASUREMENT_SUPPORTED"
        elif column in {"num_ues", "IMSI", "RNTI", "slicing_enabled", "slice_id", "scheduling_policy"}:
            source = "SCENARIO_METADATA"
        else:
            source = "PROJECT_MODEL"
        rows.append({"column_name": column, "source_class": source, "formal_statement": "not AODT-native" if source in {"PROJECT_MODEL", "PROJECT_LABEL"} else source.lower()})
    return rows


def require_pinned_run(selected_run_id: str, supplied_run_id: str | None, stage: str) -> None:
    if not supplied_run_id or supplied_run_id != selected_run_id:
        raise StageBlocked(
            f"unpinned_{stage.lower()}",
            f"Selected {stage} run_id {selected_run_id!r} must exactly match the explicit pin; latest-run discovery is forbidden",
        )


def stage_bu00(ctx: StageContext) -> dict[str, Any]:
    patterns = ["*BU*.py", "*BU*.bat", "*BU*.cmd", "*BU*.sh", "*B*.py"]
    files: dict[str, str] = {}
    for base in (ctx.repo_root / "scripts", ctx.repo_root / "configs"):
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path.is_file():
                    files[path.relative_to(ctx.repo_root).as_posix()] = sha256_file(path)
    inventory = {"files": files, "count": len(files), "mutation_policy": "preserve"}
    write_json(ctx.result_dir / "legacy_inventory.json", inventory)
    write_json(ctx.result_dir / "legacy_manifest.json", inventory)
    write_json(ctx.result_dir / "repository_state.json", {"repo_root": str(ctx.repo_root), "git_repository_detected": (ctx.repo_root / ".git").exists(), "historical_results_mutated": False})
    write_json(ctx.result_dir / "stage_map.json", STAGES)
    (ctx.result_dir / "file_sha256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())), encoding="utf-8")
    return {"formal_eligible": False, "inventory_count": len(files), "message": "Legacy branch inventoried; no files modified."}


def stage_bu01(ctx: StageContext) -> dict[str, Any]:
    sources = ctx.args.get("source") or []
    if not sources:
        raise StageBlocked("source_required", "At least one explicit --source is required")
    registries = []
    raw_towers: set[str] = set()
    for source_text in sources:
        source = Path(source_text).resolve()
        inventory = _source_inventory(source)
        for row in inventory:
            for tower in TOWERS:
                name = row["relative_path"]
                if name.endswith(f"GPS_RSS_data_40m_fs20MHz_{tower}.mat") and not Path(name).name.startswith("Converted_"):
                    raw_towers.add(tower)
        registries.append({
            "source_path": str(source),
            "source_kind": "zip" if source.is_file() else "directory",
            "container_sha256": sha256_file(source) if source.is_file() else None,
            "files": inventory,
        })
    write_json(ctx.result_dir / "source_registry.json", {"sources": registries, "raw_measurement_towers": sorted(raw_towers)})
    write_json(ctx.result_dir / "source_manifest.json", {"sources": registries, "raw_measurement_towers": sorted(raw_towers), "internet_required": False})
    inventory_rows = [row for registry in registries for row in registry["files"]]
    write_csv(ctx.result_dir / "source_inventory.csv", inventory_rows, ["container", "relative_path", "size_bytes", "sha256"])
    (ctx.result_dir / "source_sha256.txt").write_text("".join(f"{row['sha256']}  {row['container']}::{row['relative_path']}\n" for row in inventory_rows), encoding="utf-8")
    (ctx.result_dir / "SOURCE_README.md").write_text("# Registered AERPAW sources\n\nSources are user-supplied local artifacts. Raw LW1-LW5 MAT files are the primary evidence; converted/calibrated and RT files are reference-only. No substitute measurements were synthesized.\n", encoding="utf-8")
    eligible = raw_towers == set(TOWERS)
    if not eligible:
        raise StageBlocked("incomplete_raw_measurements", f"Raw tower MAT set is incomplete: {sorted(raw_towers)}")
    return {"formal_eligible": True, "registered_sources": len(registries), "raw_towers": sorted(raw_towers), "source_artifacts": sources}


def stage_bu02(ctx: StageContext) -> dict[str, Any]:
    upstream, manifest = _upstream(ctx, "BU01")
    registry = read_json(require_file(upstream / "source_registry.json"))
    candidates = registry["sources"]
    selected = next((item for item in candidates if set(registry["raw_measurement_towers"]) == set(TOWERS)), None)
    if selected is None:
        raise StageBlocked("raw_source_unavailable", "BU01 did not register a complete raw Dataset21 source")
    source = Path(selected["source_path"])
    entries = [item["relative_path"] for item in selected["files"]]
    measurement_rows: list[dict[str, Any]] = []
    raw_gps_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    mat_carrier_values: dict[str, float] = {}
    for tower in TOWERS:
        name = _raw_mat_name(entries, tower)
        data = _load_mat(source, name)
        required = ["RSS1", "RSS2", "timestamp_meas", "mX", "mY", "mZ"]
        missing = [key for key in required if key not in data]
        if missing:
            raise StageBlocked("raw_mat_schema_mismatch", f"{tower} missing {missing}")
        arrays = {key: _flat(data[key]) for key in required}
        lengths = {key: len(value) for key, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise StageBlocked("raw_mat_length_mismatch", f"{tower}: {lengths}")
        counts[tower] = len(arrays["RSS1"])
        for index in range(counts[tower]):
            row = {
                "tower_id": tower, "sample_id": index, "timestamp_s": arrays["timestamp_meas"][index],
                "longitude_deg": arrays["mX"][index], "latitude_deg": arrays["mY"][index],
                "raw_altitude_m": arrays["mZ"][index], "usrp1_rss_dbm": arrays["RSS1"][index],
                "usrp2_rss_dbm": arrays["RSS2"][index], "source_mat": name,
            }
            if not all(_finite(row[key]) for key in MEASUREMENT_FIELDS[2:8]):
                raise StageBlocked("nonfinite_measurement", f"Non-finite row in {tower} index {index}")
            measurement_rows.append(row)
        if tower == "LW1":
            gps_arrays = {key: _flat(data[key]) for key in ("GPSx", "GPSy", "GPSz", "timestamp_gps")}
            for index in range(len(gps_arrays["GPSx"])):
                raw_gps_rows.append({
                    "gps_sample_id": index, "timestamp_s": gps_arrays["timestamp_gps"][index],
                    "longitude_deg": gps_arrays["GPSx"][index], "latitude_deg": gps_arrays["GPSy"][index],
                    "raw_altitude_m": gps_arrays["GPSz"][index],
                })
        if "carrier_freq" in data:
            mat_carrier_values[tower] = _flat(data["carrier_freq"])[0]
    expected = ctx.contract["measurement"]["raw_lengths_expected"]
    if counts != expected:
        raise StageBlocked("raw_count_mismatch", f"Observed {counts}, expected {expected}")
    write_csv(ctx.result_dir / "measurements_raw_normalized.csv", measurement_rows, MEASUREMENT_FIELDS)
    write_csv(ctx.result_dir / "gps_raw_917.csv", raw_gps_rows, ["gps_sample_id", "timestamp_s", "longitude_deg", "latitude_deg", "raw_altitude_m"])
    trajectory = [row for row in measurement_rows if row["tower_id"] == "LW1"]
    write_csv(ctx.result_dir / "trajectory_measurement_clock.csv", trajectory, MEASUREMENT_FIELDS)
    first_time = min(row["timestamp_s"] for row in measurement_rows)
    waypoint_maps = ctx.contract["measurement"]["waypoint_indices_matlab_1_based"]
    long_rows: list[dict[str, Any]] = []
    for row in measurement_rows:
        waypoint = next((label for label, matlab_index in waypoint_maps[row["tower_id"]].items() if row["sample_id"] == matlab_index - 1), "")
        for channel, field in ((1, "usrp1_rss_dbm"), (2, "usrp2_rss_dbm")):
            long_rows.append({
                "sample_id": row["sample_id"], "source_row_id": row["sample_id"], "source_timestamp": row["timestamp_s"],
                "elapsed_s": row["timestamp_s"] - first_time, "tower_id": row["tower_id"], "usrp_channel": channel,
                "usrp_rss_dbm_raw": row[field], "latitude_deg": row["latitude_deg"], "longitude_deg": row["longitude_deg"],
                "raw_altitude_m": row["raw_altitude_m"], "waypoint_label": waypoint, "measurement_valid": True,
                "noise_floor_candidate": None,
            })
    long_fields = ["sample_id", "source_row_id", "source_timestamp", "elapsed_s", "tower_id", "usrp_channel", "usrp_rss_dbm_raw", "latitude_deg", "longitude_deg", "raw_altitude_m", "waypoint_label", "measurement_valid", "noise_floor_candidate"]
    write_csv(ctx.result_dir / "reference_measurements_raw.csv", long_rows, long_fields)
    write_csv(ctx.result_dir / "reference_trajectory_raw.csv", trajectory, MEASUREMENT_FIELDS)
    tower_rows = [{"tower_id": tower, **ctx.contract["tower_coordinates"][tower], "height_agl_m": ctx.contract["scene"]["tower_height_agl_m"]} for tower in TOWERS]
    write_csv(ctx.result_dir / "reference_towers.csv", tower_rows, ["tower_id", "latitude_deg", "longitude_deg", "height_agl_m"])
    write_json(ctx.result_dir / "measurement_quality.json", {"finite": True, "rows_dropped": 0, "calibration_applied": False, "usrp_channels_preserved_separately": True, "wide_rows": len(measurement_rows), "long_channel_rows": len(long_rows)})
    write_json(ctx.result_dir / "normalization_manifest.json", {
        "tower_counts": counts, "raw_gps_count": len(raw_gps_rows), "calibration_applied": False,
        "rows_dropped": 0, "mat_carrier_metadata_hz": mat_carrier_values,
        "formal_coverage_frequency_hz": ctx.contract["formal_profile"]["carrier_frequency_hz"],
        "carrier_metadata_disposition": "recorded_only; final-paper coverage override controls BU06",
    })
    return {"formal_eligible": True, "source_run_ids": [manifest["run_id"]], "measurement_rows": len(measurement_rows), "trajectory_rows": len(trajectory)}


def stage_bu03(ctx: StageContext) -> dict[str, Any]:
    upstream, manifest = _upstream(ctx, "BU02")
    geo = build_geospatial_contract(ctx.contract)
    trajectory = read_csv(require_file(upstream / "reference_trajectory_raw.csv"))
    origin = ctx.contract["tower_coordinates"]["LW1"]
    _, forward, inverse = geodetic_transformer(float(origin["latitude_deg"]), float(origin["longitude_deg"]))
    projected = [forward.transform(float(row["longitude_deg"]), float(row["latitude_deg"])) for row in trajectory]
    projected.extend(
        forward.transform(float(ctx.contract["tower_coordinates"][tower]["longitude_deg"]), float(ctx.contract["tower_coordinates"][tower]["latitude_deg"]))
        for tower in TOWERS
    )
    observed_width = max(x for x, _ in projected) - min(x for x, _ in projected)
    observed_height = max(y for _, y in projected) - min(y for _, y in projected)
    dimensions = ctx.contract["scene"]["dimensions_m"]
    width, height = float(dimensions["east_west"]), float(dimensions["north_south"])
    if observed_width > width or observed_height > height:
        raise StageBlocked("paper_scene_extent_too_small", f"Observed route/towers {observed_width:.2f} x {observed_height:.2f} m exceed paper scene {width} x {height} m")
    center_x = (min(x for x, _ in projected) + max(x for x, _ in projected)) / 2.0
    center_y = (min(y for _, y in projected) + max(y for _, y in projected)) / 2.0
    corners = [inverse.transform(center_x + dx * width / 2.0, center_y + dy * height / 2.0) for dx in (-1, 1) for dy in (-1, 1)]
    scene_bbox = {
        "order": "west,south,east,north",
        "values": [min(lon for lon, _ in corners), min(lat for _, lat in corners), max(lon for lon, _ in corners), max(lat for _, lat in corners)],
        "source": "paper 1080x2130 m dimensions centered on combined raw trajectory and LW1-LW5 ENU extent",
        "osm_source": "OpenStreetMap contributors",
        "profile_scope": "osm_only_smoke until formal foliage/terrain bundle is supplied",
        "observed_content_extent_m": {"east_west": observed_width, "north_south": observed_height},
    }
    geo.update({
        "tower_ids": TOWERS,
        "tower_coordinates": {tower: ctx.contract["tower_coordinates"][tower] for tower in TOWERS},
        "tower_height_agl_m": ctx.contract["scene"]["tower_height_agl_m"],
        "trajectory_bounds_wgs84": {
            "min_latitude_deg": min(float(row["latitude_deg"]) for row in trajectory), "max_latitude_deg": max(float(row["latitude_deg"]) for row in trajectory),
            "min_longitude_deg": min(float(row["longitude_deg"]) for row in trajectory), "max_longitude_deg": max(float(row["longitude_deg"]) for row in trajectory),
        },
        "target_scene_dimensions_m": ctx.contract["scene"]["dimensions_m"],
        "scene_bbox_wgs84": scene_bbox,
        "terrain_source": "formal AODT scene manifest required", "building_source": "formal AODT scene manifest required",
        "vegetation_source": "OSM forest plus separately supplied satellite-guided supplement required",
        "field_provenance": "parameter contract plus raw Dataset21 trajectory; scene-specific transforms remain unresolved until BU04 formal registration",
    })
    if geo["roundtrip_error_deg"] > 1e-9:
        raise StageBlocked("coordinate_roundtrip_failed", f"Round-trip error {geo['roundtrip_error_deg']}")
    write_json(ctx.result_dir / "geospatial_contract.json", geo)
    return {"formal_eligible": False, "source_run_ids": [manifest["run_id"]], "roundtrip_error_deg": geo["roundtrip_error_deg"], "blocker": "verified formal scene transform and vertical datum not yet supplied"}


def stage_bu04(ctx: StageContext) -> dict[str, Any]:
    profile = ctx.args.get("scene_profile") or "paper_reconstruction"
    if profile == "osm_only_smoke":
        bu03, m03 = _upstream(ctx, "BU03")
        geo = read_json(require_file(bu03 / "geospatial_contract.json"))
        request = {
            "profile": profile, "formal_eligible": False, "builder": str((Path(__file__).parent / "aodt_scene_builder.py").resolve()),
            "include_elevation": True, "ground_source": "terrarium", "vegetation_source": "procedural",
            "dimensions_m": ctx.contract["scene"]["dimensions_m"],
            "bbox_wgs84": geo["scene_bbox_wgs84"],
            "osm_attribution": "OpenStreetMap contributors",
            "warning": "OSM/procedural foliage is not the paper's satellite-guided supplemental foliage layer",
        }
        write_json(ctx.result_dir / "scene_build_request.json", request)
        write_csv(ctx.result_dir / "tree_inventory.csv", [], ["tree_id", "longitude_deg", "latitude_deg", "height_m", "diameter_m", "source_feature_id"])
        write_json(ctx.result_dir / "scene_manifest.json", {"profile": profile, "formal_eligible": False, "scene_url": None, "status": "request_only_not_built"})
        write_json(ctx.result_dir / "scene_sources.json", {"terrain": "terrarium candidate", "buildings": "OSM candidate", "vegetation": ["OSM/procedural candidate"], "supplementary_foliage": None})
        write_json(ctx.result_dir / "vegetation_manifest.json", {"random_seed": 2025, "tree_height_m": 20.0, "tree_max_diameter_m": 10.0, "formal_supplement_present": False})
        write_json(ctx.result_dir / "scene_reconstruction_scope.json", {"exact_author_scene": False, "publication_formal": False, "scope": "engineering smoke request only"})
        return {"formal_eligible": False, "formal_profile": profile, "source_run_ids": [m03["run_id"]], "message": "Lake Wheeler OSM smoke-scene request created from measured route/tower AOI; not executed."}
    if profile != "paper_reconstruction":
        raise StageBlocked("unknown_scene_profile", profile)
    bundle_text = ctx.args.get("scene_bundle")
    foliage_text = ctx.args.get("foliage_geojson")
    if not bundle_text:
        raise StageBlocked("blocked_missing_formal_scene_bundle", "paper_reconstruction requires --scene-bundle")
    if not foliage_text:
        raise StageBlocked("blocked_missing_formal_foliage_layer", "paper_reconstruction requires --foliage-geojson")
    bundle = require_dir(Path(bundle_text).resolve(), "missing_formal_scene_bundle")
    scene_manifest_path = require_file(bundle / "scene_manifest.json", "missing_scene_manifest")
    scene_manifest = validate_scene_manifest(scene_manifest_path)
    foliage = Path(foliage_text).resolve()
    geojson = validate_foliage_geojson(foliage)
    write_json(ctx.result_dir / "formal_scene_registration.json", {
        "profile": profile, "bundle": str(bundle), "scene_manifest": str(scene_manifest_path),
        "scene_manifest_sha256": sha256_file(scene_manifest_path), "foliage_geojson": str(foliage),
        "foliage_sha256": sha256_file(foliage), "foliage_feature_count": len(geojson["features"]),
        "scene_url": scene_manifest["scene_url"],
    })
    write_json(ctx.result_dir / "scene_manifest.json", scene_manifest)
    write_json(ctx.result_dir / "scene_sources.json", {"scene_bundle": str(bundle), "foliage_geojson": str(foliage), "vegetation_sources": ["OSM forest", "supplied manual/satellite-guided supplement"]})
    write_json(ctx.result_dir / "vegetation_manifest.json", {"random_seed": 2025, "tree_height_m": 20.0, "tree_max_diameter_m": 10.0, "foliage_feature_count": len(geojson["features"]), "formal_supplement_present": True})
    write_json(ctx.result_dir / "scene_reconstruction_scope.json", {"exact_author_scene": False, "publication_formal_profile": True, "statement": "public/reference reconstruction, not the exact original author scene"})
    tree_source = (scene_manifest_path.parent / scene_manifest["tree_inventory_csv"]).resolve()
    tree_rows = read_csv(tree_source)
    required_tree_fields = ["tree_id", "longitude_deg", "latitude_deg", "height_m", "diameter_m", "source_feature_id"]
    if not tree_rows or any(field not in tree_rows[0] for field in required_tree_fields):
        raise StageBlocked("invalid_tree_inventory", "Formal scene tree inventory is empty or missing required fields")
    write_csv(ctx.result_dir / "tree_inventory.csv", tree_rows, required_tree_fields)
    return {"formal_eligible": True, "formal_profile": profile, "scene_url": scene_manifest["scene_url"], "source_artifacts": [str(bundle), str(foliage)]}


def _terrain_lookup(manifest_path: Path) -> dict[int, float]:
    manifest = validate_scene_manifest(manifest_path)
    rows = read_csv((manifest_path.parent / manifest["terrain_samples_csv"]).resolve())
    result: dict[int, float] = {}
    for row in rows:
        if row.get("entity_type") == "uav":
            result[int(row["sample_id"])] = float(row["terrain_elevation_m"])
    return result


def stage_bu05(ctx: StageContext) -> dict[str, Any]:
    bu02, m02 = _upstream(ctx, "BU02")
    bu04, m04 = _upstream(ctx, "BU04")
    measurements = read_csv(require_file(bu02 / "trajectory_measurement_clock.csv"))
    formal = bool(m04.get("formal_eligible"))
    if formal:
        registration = read_json(require_file(bu04 / "formal_scene_registration.json"))
        terrain = _terrain_lookup(Path(registration["scene_manifest"]))
        if len(terrain) != len(measurements):
            raise StageBlocked("incomplete_terrain_samples", f"Terrain samples {len(terrain)} != trajectory {len(measurements)}")
        geo_manifest = read_json(Path(registration["scene_manifest"]))
        origin = geo_manifest["origin"]
    else:
        terrain = {int(row["sample_id"]): 112.0 for row in measurements}
        origin = ctx.contract["tower_coordinates"]["LW1"]
    _, forward, _ = geodetic_transformer(float(origin["latitude_deg"]), float(origin["longitude_deg"]))
    rows: list[dict[str, Any]] = []
    minimum = float(ctx.contract["trajectory"]["minimum_simulation_height_m"])
    for source in measurements:
        sample_id = int(source["sample_id"])
        height, clamped = clamp_height(float(source["raw_altitude_m"]), minimum)
        x, y = forward.transform(float(source["longitude_deg"]), float(source["latitude_deg"]))
        ground = terrain[sample_id]
        rows.append({
            "sample_id": sample_id, "timestamp_s": float(source["timestamp_s"]),
            "longitude_deg": float(source["longitude_deg"]), "latitude_deg": float(source["latitude_deg"]),
            "raw_altitude_m": float(source["raw_altitude_m"]), "simulation_height_agl_m": height,
            "terrain_elevation_m": ground, "simulation_altitude_m": ground + height,
            "scene_x_m": x, "scene_y_m": y, "scene_z_m": ground + height,
            "altitude_was_clamped": str(clamped).lower(),
        })
    if len(rows) != int(ctx.contract["trajectory"]["formal_sample_count"]):
        raise StageBlocked("trajectory_count_mismatch", f"Expected 3530, received {len(rows)}")
    write_csv(ctx.result_dir / "trajectory_3d.csv", rows, TRAJECTORY_FIELDS)
    write_gpx(ctx.result_dir / "trajectory_audit_only.gpx", rows)
    write_json(ctx.result_dir / "trajectory_manifest.json", {
        "sample_count": len(rows), "raw_altitude_preserved": True, "minimum_height_m": minimum,
        "clamped_sample_count": sum(row["altitude_was_clamped"] == "true" for row in rows),
        "gpx_role": "audit_only", "aodt_execution": "explicit georef 3D waypoints",
        "terrain_source": "formal scene manifest" if formal else "public-candidate anchor 112 m; smoke only",
    })
    shutil.copy2(ctx.result_dir / "trajectory_3d.csv", ctx.result_dir / "aerpaw_coverage_trajectory.csv")
    shutil.copy2(ctx.result_dir / "trajectory_audit_only.gpx", ctx.result_dir / "aerpaw_coverage_reference.gpx")
    write_json(ctx.result_dir / "aerpaw_coverage_waypoints_3d.json", {"coordinate_type": "WGS84 georef with explicit altitude", "waypoints": rows})
    shutil.copy2(ctx.result_dir / "trajectory_manifest.json", ctx.result_dir / "trajectory_conversion_manifest.json")
    write_csv(ctx.result_dir / "trajectory_validation.csv", rows, TRAJECTORY_FIELDS)
    return {"formal_eligible": formal, "source_run_ids": [m02["run_id"], m04["run_id"]], "trajectory_rows": len(rows)}


def stage_bu06(ctx: StageContext) -> dict[str, Any]:
    bu04, m04 = _upstream(ctx, "BU04")
    bu05, m05 = _upstream(ctx, "BU05")
    rows = read_csv(require_file(bu05 / "trajectory_3d.csv"))
    if m04.get("formal_eligible"):
        registration = read_json(require_file(bu04 / "formal_scene_registration.json"))
        scene_url = registration["scene_url"]
    else:
        scene_url = "UNRESOLVED_SMOKE_SCENE_URL"
    spec = build_execution_spec(ctx.contract, rows, scene_url)
    if spec["frequency_mhz"] != 3300.0 or spec["array_mode"] != "SISO":
        raise StageBlocked("coverage_profile_drift", "BU06 must remain 3.3 GHz SISO")
    write_json(ctx.result_dir / "aodt_execution_spec.json", spec)
    write_json(ctx.result_dir / "config_manifest.json", {
        "builder": str((Path(__file__).parent / "aodt_native_builder.py").resolve()),
        "native_api": "AODT 1.5.1 _config", "mode": "EM", "waypoint_count": len(rows),
        "arrival_time_api_available": False, "runtime_position_readback_required": True,
        "physical_link_direction": "UAV_TO_TOWER", "aodt_mapping": "TOWER_RU_TO_UAV_UE",
        "reciprocity_assumption": True, "tx_power_dbm": 44.0,
        "rss_power_rule": "CIR includes configured RU power; BU11 must not add 44 dBm again",
    })
    write_json(ctx.result_dir / "aerpaw_lw_coverage_em.yaml", spec)
    shutil.copy2(ctx.result_dir / "config_manifest.json", ctx.result_dir / "aerpaw_lw_coverage_em_manifest.json")
    return {"formal_eligible": bool(m04.get("formal_eligible") and m05.get("formal_eligible")), "source_run_ids": [m04["run_id"], m05["run_id"]], "waypoints": len(rows)}


def _scan_forbidden_text(paths: Iterable[Path]) -> list[str]:
    forbidden = ("tokyo", "synthetic_foliage", "placeholder_scene", "unresolved_smoke")
    hits: list[str] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".txt", ".gpx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in forbidden:
            if token in text:
                hits.append(f"{path}:{token}")
    return hits


def formal_gate_checks(
    manifests: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    trajectory_count: int,
    scene_registration: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        {"check": "BU01 complete raw five-tower source", "passed": manifests["BU01"].get("formal_eligible") is True},
        {"check": "BU02 normalized without calibration", "passed": manifests["BU02"].get("formal_eligible") is True},
        {"check": "BU04 verified formal scene and foliage", "passed": manifests["BU04"].get("formal_eligible") is True},
        {"check": "BU05 formal 3D trajectory", "passed": manifests["BU05"].get("formal_eligible") is True and trajectory_count == 3530},
        {"check": "BU06 formal native AODT config", "passed": manifests["BU06"].get("formal_eligible") is True},
        {"check": "coverage frequency is 3.3 GHz", "passed": spec.get("frequency_mhz") == 3300.0},
        {"check": "coverage array is SISO", "passed": spec.get("array_mode") == "SISO"},
        {"check": "five towers and one UAV", "passed": len(spec.get("towers", [])) == 5 and spec.get("uav", {}).get("ue_id") == 1},
        {"check": "44 dBm configured once", "passed": spec.get("tx_power_dbm") == 44.0},
        {"check": "formal foliage is nonempty", "passed": int(scene_registration.get("foliage_feature_count", 0)) > 0},
        {"check": "runtime position readback required", "passed": spec.get("timing_contract", {}).get("runtime_position_readback_required") is True},
    ]
    return checks


def stage_bu07(ctx: StageContext) -> dict[str, Any]:
    runs: dict[str, tuple[Path, dict[str, Any]]] = {stage: _upstream(ctx, stage) for stage in ("BU01", "BU02", "BU03", "BU04", "BU05", "BU06")}
    bu04 = runs["BU04"][0]
    bu05 = runs["BU05"][0]
    bu06 = runs["BU06"][0]
    if not runs["BU04"][1].get("formal_eligible"):
        raise StageBlocked("formal_scene_gate_failed", "BU04 is smoke/blocked; exact scene plus nonempty supplemental foliage are required")
    registration = read_json(require_file(bu04 / "formal_scene_registration.json"))
    spec = read_json(require_file(bu06 / "aodt_execution_spec.json"))
    trajectory = read_csv(require_file(bu05 / "trajectory_3d.csv"))
    checks = formal_gate_checks({stage: value[1] for stage, value in runs.items()}, spec, len(trajectory), registration)
    forbidden_hits = _scan_forbidden_text([bu06 / "aodt_execution_spec.json", bu05 / "trajectory_3d.csv", bu04 / "formal_scene_registration.json"])
    checks.append({"check": "no Tokyo/synthetic/placeholder inputs", "passed": not forbidden_hits, "details": forbidden_hits})
    write_csv(ctx.result_dir / "formal_gate.csv", checks, ["check", "passed", "details"])
    write_json(ctx.result_dir / "formal_gate.json", {"checks": checks, "passed": all(item["passed"] for item in checks)})
    failed = [item["check"] for item in checks if not item["passed"]]
    if failed:
        raise StageBlocked("formal_gate_failed", f"Failed checks: {failed}")
    return {"formal_eligible": True, "source_run_ids": [value[1]["run_id"] for value in runs.values()], "checks_passed": len(checks)}


def stage_bu08(ctx: StageContext) -> dict[str, Any]:
    runs = {stage: _upstream(ctx, stage) for stage in ("BU01", "BU02", "BU03", "BU04", "BU05", "BU06", "BU07")}
    bu01, m01 = runs["BU01"]
    bu02, m02 = runs["BU02"]
    bu03, m03 = runs["BU03"]
    bu04, m04 = runs["BU04"]
    bu05, m05 = runs["BU05"]
    bu06, m06 = runs["BU06"]
    bu07, m07 = runs["BU07"]
    if not m07.get("formal_eligible"):
        raise StageBlocked("formal_gate_required", "BU08 requires a passed BU07 run")
    files = [
        bu01 / "source_registry.json",
        bu02 / "normalization_manifest.json",
        bu03 / "geospatial_contract.json",
        bu04 / "formal_scene_registration.json",
        bu05 / "trajectory_3d.csv",
        bu05 / "trajectory_audit_only.gpx",
        bu05 / "trajectory_manifest.json",
        bu06 / "aodt_execution_spec.json",
        bu06 / "config_manifest.json",
        bu07 / "formal_gate.json",
    ]
    bundle_dir = ctx.result_dir / "bundle"
    bundle_dir.mkdir()
    inventory = []
    for source in files:
        require_file(source)
        destination = bundle_dir / source.name
        shutil.copy2(source, destination)
        inventory.append({"name": destination.name, "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size})
    shutil.copy2(ctx.contract_path, bundle_dir / "simulation_parameter_contract.json")
    inventory.append({"name": "simulation_parameter_contract.json", "sha256": sha256_file(bundle_dir / "simulation_parameter_contract.json"), "size_bytes": (bundle_dir / "simulation_parameter_contract.json").stat().st_size})
    tower_rows = [
        {"tower_id": tower, **ctx.contract["tower_coordinates"][tower], "height_agl_m": ctx.contract["scene"]["tower_height_agl_m"]}
        for tower in TOWERS
    ]
    write_csv(bundle_dir / "tower_table.csv", tower_rows, ["tower_id", "latitude_deg", "longitude_deg", "height_agl_m"])
    inventory.append({"name": "tower_table.csv", "sha256": sha256_file(bundle_dir / "tower_table.csv"), "size_bytes": (bundle_dir / "tower_table.csv").stat().st_size})
    write_json(bundle_dir / "bundle_manifest.json", {"schema_version": "1.0.0", "files": inventory, "formal_gate_run_id": m07["run_id"]})
    bundle_run_id = f"aerpaw_lw_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}"
    archive = ctx.result_dir / f"{bundle_run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(bundle_dir, arcname="aerpaw_bu_bundle", recursive=True)
    if ctx.args.get("execute"):
        destination_text = ctx.args.get("upload_destination")
        if not destination_text:
            raise StageBlocked("upload_destination_required", "--execute requires --upload-destination")
        destination = Path(destination_text).resolve() / archive.name
        if destination.exists():
            raise StageBlocked("destination_exists", f"Refusing overwrite: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination)
    return {
        "formal_eligible": True, "source_run_ids": [value[1]["run_id"] for value in runs.values()],
        "archive": str(archive), "uploaded": bool(ctx.args.get("execute")), "BU_RUN_ID": bundle_run_id,
    }


def _require_execute_command(ctx: StageContext, label: str) -> list[str]:
    if not ctx.args.get("execute"):
        raise StageBlocked("execution_not_requested", f"{label} was not started; pass --execute plus an explicit --command")
    command = ctx.args.get("command") or []
    if not command:
        raise StageBlocked("explicit_command_required", f"{label} requires an explicit --command")
    return command


def stage_bu09(ctx: StageContext) -> dict[str, Any]:
    bu08, m08 = _upstream(ctx, "BU08")
    if not m08.get("formal_eligible"):
        raise StageBlocked("formal_bundle_required", "BU09 requires formal BU08")
    command = _require_execute_command(ctx, "AODT EM engineering smoke test")
    code = run_external(command, ctx.repo_root, ctx.result_dir / "remote_stdout.log", ctx.result_dir / "remote_stderr.log")
    if code:
        raise StageBlocked("aodt_smoke_command_failed", f"AODT smoke command exited {code}")
    evidence_text = ctx.args.get("smoke_evidence")
    if not evidence_text:
        raise StageBlocked("smoke_evidence_required", "BU09 command completed but --smoke-evidence was not supplied")
    evidence_path = require_file(Path(evidence_text).resolve())
    evidence = read_json(evidence_path)
    required = ["scenario_loads", "five_tower_links", "uav_position_query", "altitude_3d_preserved", "channel_data_returned", "sample_ids_mapped", "native_exported", "no_tokyo_reference"]
    failed = [key for key in required if evidence.get(key) is not True]
    write_json(ctx.result_dir / "smoke_evidence.json", evidence)
    if failed:
        raise StageBlocked("smoke_postcondition_failed", f"Failed smoke checks: {failed}")
    return {"formal_eligible": True, "source_run_ids": [m08["run_id"]], "command_exit_code": code, "publication_evidence": False}


def stage_bu10(ctx: StageContext) -> dict[str, Any]:
    bu08, m08 = _upstream(ctx, "BU08")
    bu09, m09 = _upstream(ctx, "BU09")
    if not (m08.get("formal_eligible") and m09.get("formal_eligible")):
        raise StageBlocked("prepared_formal_scene_required", "BU10 requires formal BU08 and successful BU09")
    command = _require_execute_command(ctx, "Full AODT EM measurement replay")
    code = run_external(command, ctx.repo_root, ctx.result_dir / "aodt_stdout.log", ctx.result_dir / "aodt_stderr.log")
    if code:
        raise StageBlocked("aodt_command_failed", f"AODT command exited {code}")
    native_text = ctx.args.get("native_output")
    if not native_text:
        raise StageBlocked("native_output_required", "BU10 requires --native-output after the full replay")
    native = Path(native_text).resolve()
    if not native.exists():
        raise StageBlocked("native_output_missing", str(native))
    inventory = hash_tree(native)
    write_json(ctx.result_dir / "native_output_registry.json", {"path": str(native), "files": inventory, "file_count": len(inventory)})
    return {
        "formal_eligible": True, "source_run_ids": [m08["run_id"], m09["run_id"]], "command_exit_code": code,
        "native_output": str(native), "postcondition": "BU11 still requires explicit power semantics and position readback verification",
    }


def parse_native_rows(native_csv: Path, semantics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv(native_csv)
    required_contract = {"power_semantics", "timestamp_semantics", "tower_mapping", "position_readback_verified"}
    missing = sorted(required_contract - set(semantics))
    if missing:
        raise StageBlocked("incomplete_native_schema_contract", f"Missing semantic declarations: {missing}")
    if semantics["position_readback_verified"] is not True:
        raise StageBlocked("position_readback_unverified", "AODT trajectory/timeline readback has not been verified")
    power_semantics = semantics["power_semantics"]
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float, int], list[complex]] = {}
    if power_semantics == "cir_amplitude_sqrt_w_includes_tx_power":
        for row in rows:
            for field in ("tower_id", "timestamp_s", "cir_real", "cir_imag"):
                if field not in row:
                    raise StageBlocked("native_schema_mismatch", f"CIR CSV missing {field}")
            key = (str(row["tower_id"]), float(row["timestamp_s"]), int(row.get("sample_id") or -1))
            if str(row.get("path_exists", "true")).lower() not in {"false", "0", "no"}:
                grouped.setdefault(key, []).append(complex(float(row["cir_real"]), float(row["cir_imag"])))
            else:
                grouped.setdefault(key, [])
        for (tower, timestamp, sample_id), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
            output.append({"tower_id": tower, "sample_id": sample_id, "timestamp_s": timestamp, "aodt_rss_dbm": cir_to_rss_dbm(values, power_semantics), "path_status": "path" if values else "no_path"})
    elif power_semantics in {"received_power_w", "received_power_dbm"}:
        field = power_semantics
        for row in rows:
            if field not in row:
                raise StageBlocked("native_schema_mismatch", f"CSV missing {field}")
            raw = row[field]
            value = None if raw in (None, "", "null", "None") else float(raw)
            rss = received_power_w_to_dbm(value) if power_semantics == "received_power_w" else value
            output.append({"tower_id": row["tower_id"], "sample_id": int(row.get("sample_id") or -1), "timestamp_s": float(row["timestamp_s"]), "aodt_rss_dbm": rss, "path_status": "path" if rss is not None else "no_path"})
    else:
        raise StageBlocked("unsupported_power_semantics", str(power_semantics))
    return output


def stage_bu11(ctx: StageContext) -> dict[str, Any]:
    bu10, m10 = _upstream(ctx, "BU10")
    native_text = ctx.args.get("native_output")
    contract_text = ctx.args.get("semantic_contract")
    if not native_text or not contract_text:
        raise StageBlocked("native_output_contract_required", "BU11 requires --native-output and --semantic-contract")
    native_path = require_file(Path(native_text).resolve())
    semantics_path = require_file(Path(contract_text).resolve())
    semantics = read_json(semantics_path)
    rows = parse_native_rows(native_path, semantics)
    write_csv(ctx.result_dir / "aodt_rss_native.csv", rows, ["tower_id", "sample_id", "timestamp_s", "aodt_rss_dbm", "path_status"])
    write_json(ctx.result_dir / "rss_semantics.json", {
        **semantics, "tx_power_added_by_bu11_db": 0.0, "no_path_representation": None,
        "synthetic_floor_applied": False, "row_count": len(rows), "no_path_count": sum(row["aodt_rss_dbm"] is None for row in rows),
    })
    audit_lines = [
        "# RSS semantic audit", "", f"- Native source: `{native_path}`", f"- Power semantics: `{semantics['power_semantics']}`",
        f"- Endpoint convention: `{semantics.get('endpoint_convention', 'declared in tower_mapping')}`",
        "- Antenna effects: must be declared by the supplied semantic contract.",
        "- Transmit power addition in BU11: `0 dB` (AODT CIR contract already includes configured RU power).",
        "- True no-path values remain JSON/CSV null; no synthetic floor is used.",
    ]
    (ctx.result_dir / "rss_semantic_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    shutil.copy2(ctx.result_dir / "aodt_rss_native.csv", ctx.result_dir / "aodt_rss_raw.csv")
    shutil.copy2(ctx.result_dir / "rss_semantics.json", ctx.result_dir / "rss_provenance.json")
    return {"formal_eligible": True, "source_run_ids": [m10["run_id"]], "rows": len(rows), "source_artifacts": [str(native_path), str(semantics_path)]}


def stage_bu12(ctx: StageContext) -> dict[str, Any]:
    bu02, m02 = _upstream(ctx, "BU02")
    bu11, m11 = _upstream(ctx, "BU11")
    measurements = read_csv(require_file(bu02 / "measurements_raw_normalized.csv"))
    simulated = read_csv(require_file(bu11 / "aodt_rss_native.csv"))
    tolerance = float(ctx.args.get("tolerance_s") or 0.06)
    aligned = outer_align(measurements, simulated, tolerance)
    fields = list(dict.fromkeys(MEASUREMENT_FIELDS + ["aodt_timestamp_s", "aodt_rss_dbm", "aodt_path_status", "alignment_status", "time_error_s"]))
    write_csv(ctx.result_dir / "aligned_outer.csv", aligned, fields)
    counts = {status: sum(row["alignment_status"] == status for row in aligned) for status in ("matched", "measurement_only", "aodt_only")}
    first_time = min(float(row["timestamp_s"]) for row in measurements)
    noise_floor = ctx.args.get("noise_floor_dbm")
    waypoint_maps = ctx.contract["measurement"]["waypoint_indices_matlab_1_based"]
    matched_rows: list[dict[str, Any]] = []
    unmatched_usrp: list[dict[str, Any]] = []
    unmatched_aodt: list[dict[str, Any]] = []
    for row in aligned:
        if row["alignment_status"] == "aodt_only":
            unmatched_aodt.append({**row, "reason_code": "no_usrp_within_tolerance"})
            continue
        waypoint = next((label for label, matlab_index in waypoint_maps[row["tower_id"]].items() if int(row["sample_id"]) == int(matlab_index) - 1), "")
        for channel, source_field in (("USRP1", "usrp1_rss_dbm"), ("USRP2", "usrp2_rss_dbm")):
            canonical = {
                "sample_id": row["sample_id"], "elapsed_s": float(row["timestamp_s"]) - first_time,
                "tower_id": row["tower_id"], "usrp_channel": channel, "waypoint_label": waypoint,
                "latitude_deg": row.get("latitude_deg"), "longitude_deg": row.get("longitude_deg"),
                "height_agl_raw_m": row.get("raw_altitude_m"),
                "height_agl_sim_m": max(float(row["raw_altitude_m"]), 0.5),
                "aodt_rss_dbm": row.get("aodt_rss_dbm"), "usrp_rss_dbm_raw": row.get(source_field),
                "aodt_path_status": row.get("aodt_path_status") or ("path" if row.get("aodt_rss_dbm") not in (None, "") else "no_path"),
                "noise_floor_candidate": noise_floor, "usrp_timestamp_s": row["timestamp_s"],
                "aodt_timestamp_s": row.get("aodt_timestamp_s"), "time_error_s": row.get("time_error_s"),
            }
            if row["alignment_status"] == "matched":
                matched_rows.append(canonical)
            else:
                unmatched_usrp.append({**canonical, "reason_code": "no_aodt_within_tolerance"})
    matched_fields = ["sample_id", "elapsed_s", "tower_id", "usrp_channel", "waypoint_label", "latitude_deg", "longitude_deg", "height_agl_raw_m", "height_agl_sim_m", "aodt_rss_dbm", "usrp_rss_dbm_raw", "aodt_path_status", "noise_floor_candidate", "usrp_timestamp_s", "aodt_timestamp_s", "time_error_s"]
    write_csv(ctx.result_dir / "matched_raw.csv", matched_rows, matched_fields)
    write_csv(ctx.result_dir / "unmatched_usrp.csv", unmatched_usrp, matched_fields + ["reason_code"])
    write_csv(ctx.result_dir / "unmatched_aodt.csv", unmatched_aodt, fields + ["reason_code"])
    report = {"join": "outer_nearest_by_tower", "tolerance_s": tolerance, "counts": counts, "matched_channel_rows": len(matched_rows), "unmatched_usrp_channel_rows": len(unmatched_usrp), "unmatched_aodt_rows": len(unmatched_aodt), "unmatched_rows_preserved": True, "noise_floor_dbm": noise_floor}
    write_json(ctx.result_dir / "alignment_manifest.json", report)
    write_json(ctx.result_dir / "alignment_report.json", report)
    return {"formal_eligible": True, "source_run_ids": [m02["run_id"], m11["run_id"]], "counts": counts}


def stage_bu13(ctx: StageContext) -> dict[str, Any]:
    bu12, m12 = _upstream(ctx, "BU12")
    all_rows = read_csv(require_file(bu12 / "matched_raw.csv"))
    rows = [row for row in all_rows if row.get("aodt_rss_dbm") not in (None, "") and row.get("usrp_rss_dbm_raw") not in (None, "")]
    output_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for tower in TOWERS:
        for channel in ("USRP1", "USRP2"):
            tower_rows = sorted((row for row in rows if row["tower_id"] == tower and row["usrp_channel"] == channel), key=lambda row: float(row["usrp_timestamp_s"]))
            splits = contiguous_splits(len(tower_rows))
            paper_offset = best_offset(
                [float(row["usrp_rss_dbm_raw"]) for row in tower_rows],
                [float(row["aodt_rss_dbm"]) for row in tower_rows],
                step=float(ctx.contract["measurement"]["calibration_search_db"]["step"]),
            )
            calibration = list(splits["calibration"])
            offset = best_offset(
                [float(tower_rows[i]["usrp_rss_dbm_raw"]) for i in calibration],
                [float(tower_rows[i]["aodt_rss_dbm"]) for i in calibration],
                step=float(ctx.contract["measurement"]["calibration_search_db"]["step"]),
            )
            for split_name, indices in splits.items():
                measured = [float(tower_rows[i]["usrp_rss_dbm_raw"]) for i in indices]
                simulated = [float(tower_rows[i]["aodt_rss_dbm"]) + offset for i in indices]
                noise_values = [row.get("noise_floor_candidate") for row in tower_rows]
                noise_floor = next((float(value) for value in noise_values if value not in (None, "")), None)
                metrics = extended_metrics(measured, simulated, noise_floor)
                no_path_count = sum(row["tower_id"] == tower and row["usrp_channel"] == channel and row.get("aodt_rss_dbm") in (None, "") for row in all_rows)
                metric_rows.append({"tower_id": tower, "channel": channel, "split": split_name, "paper_style_offset_db": paper_offset, "heldout_offset_db": offset, "no_path_count": no_path_count, **metrics})
                for index, measured_value, simulated_value in zip(indices, measured, simulated):
                    row = tower_rows[index]
                    output_rows.append({
                        "tower_id": tower, "channel": channel, "sample_id": row["sample_id"], "elapsed_s": row["elapsed_s"], "waypoint_label": row["waypoint_label"],
                        "split": split_name, "measured_rss_dbm": measured_value, "aodt_rss_raw_dbm": row["aodt_rss_dbm"],
                        "paper_style_offset_db": paper_offset, "aodt_rss_paper_style_dbm": float(row["aodt_rss_dbm"]) + paper_offset,
                        "heldout_offset_db": offset, "aodt_rss_heldout_dbm": simulated_value,
                    })
    comparison_fields = ["tower_id", "channel", "sample_id", "elapsed_s", "waypoint_label", "split", "measured_rss_dbm", "aodt_rss_raw_dbm", "paper_style_offset_db", "aodt_rss_paper_style_dbm", "heldout_offset_db", "aodt_rss_heldout_dbm"]
    metric_fields = ["tower_id", "channel", "split", "paper_style_offset_db", "heldout_offset_db", "count", "no_path_count", "mae_db", "rmse_db", "bias_db", "pearson_r", "spearman_r", "outage_tp", "outage_fp", "outage_fn", "outage_tn", "outage_precision", "outage_recall", "outage_f1"]
    write_csv(ctx.result_dir / "comparison_rows.csv", output_rows, comparison_fields)
    write_csv(ctx.result_dir / "metrics.csv", metric_rows, metric_fields)
    write_csv(ctx.result_dir / "per_tower_metrics.csv", metric_rows, metric_fields)
    split_manifest = {"heldout_method": "contiguous per tower/channel", "calibration_fraction": 0.2, "validation_fraction": 0.2, "test_fraction": 0.6, "calibration_uses_only_calibration_split": True, "paper_style_calibration": "global additive offset over all valid rows; descriptive and not blind", "random_split_used": False}
    write_json(ctx.result_dir / "split_manifest.json", split_manifest)
    write_json(ctx.result_dir / "calibration_manifest.json", split_manifest)
    overall = {"weighting": "sample-weighted summary is not silently substituted; per-tower/channel metrics are primary", "methodological_complete": True, "metric_row_count": len(metric_rows), "agreement_threshold": None}
    write_json(ctx.result_dir / "overall_metrics.json", overall)
    write_json(ctx.result_dir / "physical_validation_manifest.json", {"claim_boundary": "scenario-specific RSS / propagation / coverage consistency", "methodological_completeness": True, "strength_of_measured_agreement": "reported numerically; no arbitrary pass threshold", "no_path_preserved": True})
    claims = """# Validation claims\n\nBU13 evaluates scenario-specific RSS, propagation, and coverage consistency for the supplied Lake Wheeler scene and measured route.\n\nIt does **not** validate complete RAN realism, scheduling, SINR, CQI/MCS/PRB/BLER, throughput, or full network equivalence. Paper-style additive calibration is descriptive and is not an independent blind validation. Held-out results use contiguous time splits. Methodological completeness is separate from the numerical strength of agreement; poor agreement remains a result and is not a reason to retune silently.\n"""
    (ctx.result_dir / "VALIDATION_CLAIMS.md").write_text(claims, encoding="utf-8")
    _write_bu13_plots(ctx.result_dir, output_rows, all_rows)
    return {"formal_eligible": True, "source_run_ids": [m12["run_id"]], "comparison_rows": len(output_rows)}


def _load_adapter_columns(adapter_path: Path) -> list[str]:
    spec = importlib.util.spec_from_file_location("legacy_aodt_contract_adapter_audit", adapter_path)
    if spec is None or spec.loader is None:
        raise StageBlocked("adapter_import_failed", str(adapter_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.COLOSSEUM_COLUMNS)


def stage_bu14(ctx: StageContext) -> dict[str, Any]:
    bu13, m13 = _upstream(ctx, "BU13")
    require_pinned_run(m13["run_id"], ctx.args.get("pinned_bu13_run_id"), "BU13")
    adapter = ctx.repo_root / "scripts" / "adapter" / "legacy_aodt_contract_adapter.py"
    require_file(adapter, "legacy_adapter_missing")
    actual = _load_adapter_columns(adapter)
    if actual != COLOSSEUM_COLUMNS:
        raise StageBlocked("legacy_schema_drift", "Legacy adapter no longer exposes the pinned 32-column schema")
    provenance = field_provenance_rows()
    if len(provenance) != 32 or any(row["source_class"] not in ALLOWED_PROVENANCE for row in provenance):
        raise StageBlocked("field_provenance_incomplete", "Every legacy field must have exactly one allowed source class")
    write_csv(ctx.result_dir / "field_provenance.csv", provenance, ["column_name", "source_class", "formal_statement"])
    write_json(ctx.result_dir / "adapter_contract.json", {
        "legacy_adapter": str(adapter), "legacy_adapter_sha256": sha256_file(adapter),
        "pinned_bu13_run_id": m13["run_id"], "columns": COLOSSEUM_COLUMNS,
        "warning": "Only ul_rssi is AODT-derived; network/KPM fields are project-model output, not AODT-native telemetry.",
    })
    if not ctx.args.get("execute"):
        raise StageBlocked("adapter_execution_not_requested", "Provenance was audited, but no 32-column dataset was generated; pass --execute with an explicit adapter command")
    command = ctx.args.get("command") or []
    if not command:
        raise StageBlocked("explicit_adapter_command_required", "BU14 requires an explicit adapter command")
    code = run_external(command, ctx.repo_root, ctx.result_dir / "adapter_stdout.log", ctx.result_dir / "adapter_stderr.log")
    if code:
        raise StageBlocked("adapter_command_failed", f"Adapter exited {code}")
    output_text = ctx.args.get("adapter_output")
    if not output_text:
        raise StageBlocked("adapter_output_required", "Declare --adapter-output for schema verification")
    output = require_file(Path(output_text).resolve())
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    if columns != COLOSSEUM_COLUMNS:
        raise StageBlocked("adapter_output_schema_mismatch", f"Expected exact 32-column order, received {columns}")
    shutil.copy2(output, ctx.result_dir / "legacy_32_column.csv")
    return {"formal_eligible": True, "source_run_ids": [m13["run_id"]], "columns": 32}


def stage_bu15(ctx: StageContext) -> dict[str, Any]:
    bu14, m14 = _upstream(ctx, "BU14")
    require_pinned_run(m14["run_id"], ctx.args.get("pinned_bu14_run_id"), "BU14")
    dataset = require_file(bu14 / "legacy_32_column.csv", "verified_legacy_dataset_required")
    core = require_dir(ctx.repo_root / "ai_server" / "AdvTabDDPM", "advtabddpm_core_missing")
    before = hash_tree(core, ignore=("__pycache__",))
    write_json(ctx.result_dir / "advtabddpm_core_before.json", before)
    command = _require_execute_command(ctx, "Frozen AdvTabDDPM consumer")
    code = run_external(command, ctx.repo_root, ctx.result_dir / "consumer_stdout.log", ctx.result_dir / "consumer_stderr.log")
    after = hash_tree(core, ignore=("__pycache__",))
    write_json(ctx.result_dir / "advtabddpm_core_after.json", after)
    if before != after:
        raise StageBlocked("frozen_consumer_modified", "AdvTabDDPM core changed during BU15; result is invalid")
    if code:
        raise StageBlocked("consumer_command_failed", f"Frozen consumer exited {code}")
    return {"formal_eligible": True, "source_run_ids": [m14["run_id"]], "dataset_sha256": sha256_file(dataset), "core_unchanged": True}


IMPLEMENTATIONS = {
    "BU00": stage_bu00, "BU01": stage_bu01, "BU02": stage_bu02, "BU03": stage_bu03,
    "BU04": stage_bu04, "BU05": stage_bu05, "BU06": stage_bu06, "BU07": stage_bu07,
    "BU08": stage_bu08, "BU09": stage_bu09, "BU10": stage_bu10, "BU11": stage_bu11,
    "BU12": stage_bu12, "BU13": stage_bu13, "BU14": stage_bu14, "BU15": stage_bu15,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed AERPAW Lake Wheeler BU00-BU15 pipeline")
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--input", action="append", default=[], help="Explicit upstream mapping, e.g. BU02=C:\\path\\run")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--scene-profile", choices=["paper_reconstruction", "osm_only_smoke"], default="paper_reconstruction")
    parser.add_argument("--scene-bundle")
    parser.add_argument("--foliage-geojson")
    parser.add_argument("--native-output")
    parser.add_argument("--semantic-contract")
    parser.add_argument("--smoke-evidence")
    parser.add_argument("--tolerance-s", type=float, default=0.06)
    parser.add_argument("--noise-floor-dbm", type=float)
    parser.add_argument("--pinned-bu13-run-id")
    parser.add_argument("--pinned-bu14-run-id")
    parser.add_argument("--adapter-output")
    parser.add_argument("--upload-destination")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        print(f"Repository root is missing: {repo_root}", file=sys.stderr)
        return 2
    results_root = (args.results_root or (repo_root / "results" / "bu_aerpaw")).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["repo_root"] = str(repo_root)
    payload["results_root"] = str(results_root)
    code, result_dir = run_stage(args.stage, repo_root, results_root, payload, IMPLEMENTATIONS[args.stage])
    print(json.dumps({"stage": args.stage, "exit_code": code, "result_dir": str(result_dir)}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
