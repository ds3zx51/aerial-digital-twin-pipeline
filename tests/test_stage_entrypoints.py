from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_uav_postprocessor():
    path = REPO / "scripts" / "aodt" / "uav_postprocess.py"
    spec = importlib.util.spec_from_file_location("run_uav_09bu_10bu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageEntrypointTests(unittest.TestCase):
    def test_exactly_fourteen_entrypoints_per_track(self) -> None:
        static = sorted((REPO / "scripts" / "remote" / "static").glob("*.bat"))
        uav = sorted((REPO / "scripts" / "remote" / "uav").glob("*.bat"))
        self.assertEqual([path.stem for path in static], [
            "00B_inspect_inputs", "01B_prepare_inputs", "02B_validate_scene_materials",
            "03B_build_topology", "04B_build_config", "05B_run_preflight",
            "06B_run_aodt", "07B_collect_raw", "08B_validate_raw",
            "09B_build_channel_trace", "10B_build_ai_contract",
            "11B_prepare_advtabddpm", "12B_run_advtabddpm", "13B_download_results",
        ])
        self.assertEqual([path.stem for path in uav], [
            "00BU_inspect_inputs", "01BU_validate_gpx", "02BU_validate_scene_materials",
            "03BU_build_topology", "04BU_build_config", "05BU_run_preflight",
            "06BU_run_aodt", "07BU_collect_raw", "08BU_validate_raw",
            "09BU_build_channel_trace", "10BU_build_ai_contract",
            "11BU_prepare_advtabddpm", "12BU_run_advtabddpm", "13BU_download_results",
        ])

    def test_uav_zero_power_remains_invalid(self) -> None:
        module = load_uav_postprocessor()
        values = np.zeros((2, 1, 1, 1, 1, 1, 1, 2, 3), dtype=np.complex64)
        delays = np.zeros((2, 1, 1, 1, 1, 1, 3), dtype=np.float32)
        values[0, ..., 0, 0] = 2.0 + 0.0j
        power, weighted = module.power_delay(values, delays)
        self.assertEqual(float(power[0]), 4.0)
        self.assertEqual(float(power[1]), 0.0)
        self.assertTrue(np.isnan(weighted[1]))

    def test_aerial_contract_profiles_are_deterministic_metadata(self) -> None:
        module = load_uav_postprocessor()
        source = json.loads((REPO / "configs/scenarios/bu/tokyo_BU_multi_uav.json").read_text(encoding="utf-8"))
        positions = np.zeros((1, 54, 3), dtype=np.float64)
        scenario = module.augment_ai_scenario(source, positions)
        self.assertEqual([row["id"] for row in scenario["ue"]], list(range(54)))
        self.assertEqual([row["traffic_type"] for row in scenario["ue"][50:]], ["eMBB", "mMTC", "URLLC", "eMBB"])
        self.assertTrue(all("not native AODT" in row["ai_contract_metadata_source"] for row in scenario["ue"][50:]))

    def test_bundled_ai_server_payload_and_checkpoint_hashes(self) -> None:
        ai = REPO / "ai_server"
        self.assertTrue((ai / "AdvTabDDPM" / "AdvTabDDPM.py").is_file())
        self.assertTrue((ai / "deploy" / "aodt_advtabddpm_remote.sh").is_file())
        self.assertTrue((ai / "deploy" / "materialize_aodt_dataset.py").is_file())
        with (ai / "model_manifest.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        for row in rows:
            path = ai / Path(row["path"])
            self.assertTrue(path.is_file(), row["path"])
            self.assertEqual(path.stat().st_size, int(row["bytes"]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])

    def test_server_addresses_are_configuration_only(self) -> None:
        example = (REPO / "scripts" / "remote" / "remote_settings.example.bat").read_text(encoding="utf-8")
        self.assertIn('set "REMOTE_HOST=CHANGE_ME_AI_SERVER_IP"', example)
        self.assertIn('set "REMOTE_PORT=22"', example)
        self.assertFalse((REPO / "scripts" / "remote" / "remote_settings.bat").exists())
        for track in ("static", "uav"):
            for launcher in (REPO / "scripts" / "remote" / track).glob("*.bat"):
                text = launcher.read_text(encoding="utf-8")
                self.assertNotIn("CHANGE_ME_AI_SERVER_IP", text)


if __name__ == "__main__":
    unittest.main()
