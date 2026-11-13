"""AODT 1.5.1 native configuration builder for the paper-controlled coverage profile."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def build_execution_spec(
    contract: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    scene_url: str,
) -> dict[str, Any]:
    profile = contract["formal_profile"]
    towers = contract["tower_coordinates"]
    waypoints = []
    previous: dict[str, Any] | None = None
    for row in trajectory_rows:
        current = {
            "sample_id": int(row["sample_id"]),
            "timestamp_s": float(row["timestamp_s"]),
            "latitude_deg": float(row["latitude_deg"]),
            "longitude_deg": float(row["longitude_deg"]),
            "altitude_m": float(row["simulation_altitude_m"]),
            "speed_mps": 0.0,
        }
        if previous is not None:
            dt = current["timestamp_s"] - float(previous["timestamp_s"])
            distance = math.sqrt(
                (float(row["scene_x_m"]) - float(previous["scene_x_m"])) ** 2
                + (float(row["scene_y_m"]) - float(previous["scene_y_m"])) ** 2
                + (float(row["scene_z_m"]) - float(previous["scene_z_m"])) ** 2
            )
            current["speed_mps"] = distance / dt if dt > 0 else 0.0
        waypoints.append(current)
        previous = {
            "timestamp_s": current["timestamp_s"],
            "scene_x_m": float(row["scene_x_m"]),
            "scene_y_m": float(row["scene_y_m"]),
            "scene_z_m": float(row["scene_z_m"]),
        }
    return {
        "schema_version": "1.0.0",
        "mode": "EM",
        "scene_url": scene_url,
        "frequency_mhz": float(profile["carrier_frequency_hz"]) / 1e6,
        "tx_power_dbm": float(profile["transmit_power_dbm"]),
        "array_mode": "SISO",
        "max_reflections": int(contract["scene"]["max_reflections"]),
        "scattering_enabled": bool(contract["scene"]["scattering_enabled"]),
        "diffraction_enabled": bool(contract["scene"]["diffraction_enabled"]),
        "physical_link_direction": profile["physical_link_direction"],
        "aodt_endpoint_mapping": profile["aodt_endpoint_mapping"],
        "reciprocity_assumption": True,
        "database_tables": ["CIRS", "CFRS", "RAYPATHS"],
        "towers": [
            {
                "tower_id": tower_id,
                "ru_id": index,
                "latitude_deg": float(towers[tower_id]["latitude_deg"]),
                "longitude_deg": float(towers[tower_id]["longitude_deg"]),
                "height_agl_m": float(contract["scene"]["tower_height_agl_m"]),
            }
            for index, tower_id in enumerate(profile["tower_ids"], 1)
        ],
        "uav": {"ue_id": 1, "waypoints": waypoints},
        "timing_contract": {
            "source": "measured LW1 timestamp_s",
            "aodt_arrival_time_supported": False,
            "speed_derived_from_measured_delta_t": True,
            "runtime_position_readback_required": True,
        },
    }


def build_native_dict(spec: dict[str, Any], asset_config: Path, client_root: Path) -> dict[str, Any]:
    """Build through the supported AODT 1.5.1 Python binding; never hand-write runtime YAML."""
    for path in reversed((client_root / "build" / "Release", client_root / "build" / "config" / "Release")):
        sys.path.insert(0, str(path))
    from _config import AntennaElement, DBTable, DiffusionModel, Nodes, Panel, Position, SimConfig, SimMode

    cfg = SimConfig(spec["scene_url"], SimMode.EM, str(asset_config))
    cfg.set_simulation_id("aerpaw_lake_wheeler_coverage")
    cfg.set_num_batches(1)
    timestamps = [w["timestamp_s"] for w in spec["uav"]["waypoints"]]
    duration = max(timestamps[-1] - timestamps[0], 0.1) if timestamps else 0.1
    cfg.set_timeline(duration=duration, interval=0.1)
    cfg.set_seed(2025)
    for table in (DBTable.CIRS, DBTable.CFRS, DBTable.RAYPATHS):
        cfg.add_tables_to_db(table)
    cfg.add_table_option("raypaths", "full")
    cfg.set_ray_tracing_model(DiffusionModel.DIRECTIONAL, 5, 500, 500)
    panel = Panel.create_panel(
        antenna_elements=[AntennaElement.Isotropic],
        frequency_mhz=spec["frequency_mhz"],
        vertical_num=1,
        horizontal_num=1,
        dual_polarized=False,
    )
    cfg.set_default_panel_ru(panel)
    cfg.set_default_panel_ue(panel)
    for tower in spec["towers"]:
        ru_id = int(tower["ru_id"])
        du = Nodes.create_du(du_id=ru_id, frequency_mhz=spec["frequency_mhz"], scs_khz=30)
        du.set_position(Position.cartesian(0.0, 0.0, 100.0 + ru_id))
        du.set_num_antennas(1)
        cfg.add_du(du)
        ru = Nodes.create_ru(
            ru_id=ru_id,
            frequency_mhz=spec["frequency_mhz"],
            radiated_power_dbm=spec["tx_power_dbm"],
            du_id=du.id(),
        )
        ru.set_position(Position.georef(tower["latitude_deg"], tower["longitude_deg"]))
        ru.set_height(tower["height_agl_m"])
        cfg.add_ru(ru)
    ue = Nodes.create_ue(ue_id=1, radiated_power_dbm=spec["tx_power_dbm"])
    for waypoint in spec["uav"]["waypoints"]:
        ue.add_waypoint(
            Position.georef(waypoint["latitude_deg"], waypoint["longitude_deg"], alt=waypoint["altitude_m"]),
            speed=waypoint["speed_mps"],
        )
    cfg.add_ue(ue)
    cfg.set_bldg_exterior_attr(
        activate_rf=True,
        activate_diffraction=False,
        activate_diffusion=False,
        activate_transmission=True,
        diffuse_surface_element_area=None,
        building_ids=[],
    )
    return cfg.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--asset-config", type=Path, required=True)
    parser.add_argument("--aodt-client-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    native = build_native_dict(spec, args.asset_config, args.aodt_client_root)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(native, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
