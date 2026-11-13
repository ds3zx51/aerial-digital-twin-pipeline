from __future__ import annotations

import os
import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


STAGING_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = STAGING_ROOT / "scripts" / "aodt"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

REPO = Path(os.environ.get("AODT_REPO_ROOT", STAGING_ROOT))
if not (REPO / "configs").is_dir():
    REPO = STAGING_ROOT
REPO_MODULE_DIR = REPO / "scripts" / "aodt"
if str(REPO_MODULE_DIR) not in sys.path:
    sys.path.insert(1, str(REPO_MODULE_DIR))

from uav_core import (  # noqa: E402
    UAVPipelineContext,
    _trajectory_paths,
    json_load,
    stage_04bu,
    validate_shared_gpx,
    worker_gpx_paths,
)
from uav_runtime import _validate_positions, parse_time_steps  # noqa: E402
from run_uav_08BU import validate as validate_08bu  # noqa: E402


class AODTUAVPipelineTests(unittest.TestCase):
    def make_context(self, run_dir: Path, tolerance: float = 2.0) -> UAVPipelineContext:
        return UAVPipelineContext(
            repo_root=REPO,
            run_dir=run_dir,
            scenario_path=REPO
            / "configs"
            / "scenarios"
            / "bu"
            / "tokyo_BU_multi_uav.json",
            scene_archive=Path(
                os.environ.get("AODT_SCENE_ARCHIVE", "__missing_scene__.zip")
            ),
            aodt_client_root=Path(
                os.environ.get("AODT_CLIENT_ROOT", "__missing_aodt_client__")
            ),
            asset_config=REPO / "configs" / "aodt" / "aodt_static_assets.yml",
            worker_scene_url="test_data/maps/tokyo",
            worker_gpx_prefix="test_data/gpx/shared/tokyo_u4_boundary_60m_256",
            endpoint="",
            offline=True,
            ack_worker_assets_resolved=False,
            ack_worker_gpx_resolved=False,
            slots_spec="all",
            full_antenna_pair=False,
            position_tolerance_m=tolerance,
            command="test",
        )

    def test_shared_gpx_is_exact_aodt_manifest_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.make_context(Path(temp))
            report = validate_shared_gpx(ctx)
        self.assertEqual(report["status"], "pass", report["errors"])
        self.assertTrue(report["trajectory_hashes_verified"])
        self.assertEqual(report["trajectory_id"], "tokyo_u4_boundary_60m_256")
        self.assertEqual(report["gpx_file_count"], 4)
        self.assertEqual(report["sample_count"], 256)
        self.assertEqual(report["sample_interval_s"], 0.25)
        self.assertEqual(len({item["sha256"] for item in report["gpx_inventory"]}), 4)

    def test_04bu_config_has_50_static_and_4_distinct_gpx_ues(self) -> None:
        if not os.environ.get("AODT_CLIENT_ROOT"):
            self.skipTest("AODT_CLIENT_ROOT is required for config-binding integration")
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            ctx = self.make_context(run_dir)
            status = stage_04bu(ctx)
            config = yaml.safe_load(
                (ctx.stage_dir("04BU") / "resolved_aodt_uav_config.yml").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(status["status"], "pass")
        ues = config["sim"]["UEs"]["add"]
        self.assertEqual([item["id"] for item in ues], list(range(1, 55)))
        self.assertTrue(all(len(item.get("waypoints", [])) == 1 for item in ues[:50]))
        self.assertTrue(all("gpx" not in item for item in ues[:50]))
        self.assertTrue(all("waypoints" not in item for item in ues[50:]))
        self.assertEqual(
            [item["gpx"]["src"] for item in ues[50:]], worker_gpx_paths(ctx)
        )
        self.assertTrue(
            all(item["gpx"]["use_pathfinding"] is False for item in ues[50:])
        )
        attrs = config["sim"]["Scenario"]["update"][0]["attributes"]
        self.assertEqual(attrs["sim_simulation_mode"], 0)
        self.assertEqual(attrs["sim_duration"], 63.75)
        self.assertEqual(attrs["sim_interval"], 0.25)

    def test_runtime_position_gate_accepts_only_canonical_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ctx = self.make_context(Path(temp), tolerance=0.01)
            scenario = json_load(ctx.scenario_path)
            canonical = json_load(
                _trajectory_paths(REPO, ctx.scenario_path)["canonical"]
            )
            ground = [
                [
                    float(node["x"]) * 0.01,
                    float(node["y"]) * 0.01,
                    float(node["z"]) * 0.01,
                ]
                for node in scenario["ue"]
            ]
            positions = []
            for sample_id in range(256):
                uavs = [
                    [
                        float(route["points"][sample_id]["local_x_m"]),
                        float(route["points"][sample_id]["local_y_m"]),
                        float(route["points"][sample_id]["local_z_m"]),
                    ]
                    for route in canonical["routes"]
                ]
                positions.append([*ground, *uavs])
            rus = [
                [
                    float(node["x"]) * 0.01,
                    float(node["y"]) * 0.01,
                    float(node["z"]) * 0.01 + 1.25,
                ]
                for node in scenario["bs"]
            ]
            passing = _validate_positions(ctx, positions, rus)
            positions[100][50][0] += 1.0
            failing = _validate_positions(ctx, positions, rus)
        self.assertEqual(passing["status"], "pass")
        self.assertTrue(passing["trajectory_runtime_verified"])
        self.assertFalse(passing["gpx_elevation_preserved_by_aodt_native"])
        self.assertEqual(failing["status"], "fail")
        self.assertFalse(failing["trajectory_runtime_verified"])

    def test_duration_interval_time_step_selection(self) -> None:
        self.assertEqual(parse_time_steps("all", 256), list(range(256)))
        self.assertEqual(parse_time_steps("0,5,255", 256), [0, 5, 255])
        self.assertEqual(parse_time_steps("0:10:2", 256), [0, 2, 4, 6, 8])
        with self.assertRaises(Exception):
            parse_time_steps("256", 256)

    def test_08bu_validator_accepts_complete_mobile_fixture(self) -> None:
        def dump(path: Path, value: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            raw_root = run / "07BU_raw_collection"
            raw = raw_root / "raw"
            raw.mkdir(parents=True)
            dump(
                run / "run_manifest.json",
                {
                    "scope": {
                        "ru": 8,
                        "ue": 54,
                        "ground_ue": 50,
                        "uav": 4,
                        "gpx": 4,
                        "time_steps": 256,
                    },
                    "stages": {"07BU": {"status": "pass"}},
                },
            )
            dump(
                run / "06BU_execution" / "position_registry.json",
                {
                    "validation": {
                        "status": "pass",
                        "trajectory_runtime_verified": True,
                    }
                },
            )
            dump(
                run / "03BU_topology" / "aodt_node_mapping.json",
                {
                    "validation": {
                        "ru_count": 8,
                        "ground_ue_count": 50,
                        "uav_count": 4,
                        "total_ue_count": 54,
                        "uav_aodt_ids": [51, 52, 53, 54],
                    }
                },
            )
            dump(
                run / "04BU_config" / "gpx_binding_manifest.json",
                {
                    "trajectory_id": "tokyo_u4_boundary_60m_256",
                    "trajectory_hashes_verified": True,
                },
            )
            dump(raw_root / "raw_schema_inventory.json", {"arrays": {}})
            hashes: dict[str, str] = {}
            values = np.ones(
                (54, 1, 1, 1, 1, 1, 1, 2, 1), dtype=np.complex64
            )
            delays = np.zeros((54, 1, 1, 1, 1, 1, 1), dtype=np.float32)
            angles = np.zeros((54, 1, 1, 1, 1, 1, 1, 2), dtype=np.float32)
            ue_indices = np.arange(54, dtype=np.int64)
            ue_aodt_ids = np.arange(1, 55, dtype=np.int64)
            for time_step in range(256):
                for uav_offset in range(4):
                    values[50 + uav_offset, ...] = np.complex64(
                        1.0 + (time_step + uav_offset) / 1000.0
                    )
                for ru in range(8):
                    output = raw / f"cir_timestep_{time_step:04d}_ru_{ru:02d}.npz"
                    np.savez_compressed(
                        output,
                        values=values,
                        delays=delays,
                        angles_of_departure=angles,
                        angles_of_arrival=angles,
                        batch=np.asarray(0, dtype=np.int64),
                        timestep=np.asarray(time_step, dtype=np.int64),
                        ru_tensor_index=np.asarray(ru, dtype=np.int64),
                        ru_scenario_id=np.asarray(ru, dtype=np.int64),
                        ru_aodt_id=np.asarray(ru + 1, dtype=np.int64),
                        ue_tensor_indices=ue_indices,
                        ue_scenario_ids=ue_indices,
                        ue_aodt_ids=ue_aodt_ids,
                        full_antenna_pair=np.asarray(False, dtype=np.bool_),
                    )
                    relative = output.relative_to(raw_root).as_posix()
                    hashes[relative] = digest(output)
            dump(
                raw_root / "raw_file_hashes.json",
                {"algorithm": "sha256", "files": hashes},
            )
            dump(
                raw_root / "raw_artifact_inventory.json",
                {
                    "selection": {
                        "time_steps": list(range(256)),
                        "ru_tensor_indices": list(range(8)),
                    }
                },
            )
            result = validate_08bu(run)
        self.assertEqual(
            result["stage_status"]["status"],
            "pass",
            result["quality_report"]["errors"],
        )
        self.assertEqual(result["stage_status"]["raw_npz_count"], 2048)
        self.assertEqual(result["stage_status"]["logical_link_row_count"], 110592)
        self.assertEqual(result["stage_status"]["uav_logical_link_row_count"], 8192)


if __name__ == "__main__":
    unittest.main()
