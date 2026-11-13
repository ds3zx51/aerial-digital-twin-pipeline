import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
AODT_SCRIPT_DIR = REPO / "scripts" / "aodt"
if str(AODT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AODT_SCRIPT_DIR))

from aodt_runtime import parse_slots, stage_06b, stage_07b
from static_core import (
    PipelineContext,
    _node_rows,
    aodt_member_value,
    stage_00b,
    stage_01b,
    stage_02b,
    stage_03b,
    stage_04b,
    stage_05b,
    validate_locked_scenario,
)


SCENARIO = REPO / "configs" / "scenario_layout_tokyo_docomo.json"
SCENE_ARCHIVE = Path(os.environ.get("AODT_SCENE_ARCHIVE", "__missing_scene__.zip"))
AODT_CLIENT = Path(os.environ.get("AODT_CLIENT_ROOT", "__missing_aodt_client__"))
ASSET_CONFIG = REPO / "configs" / "aodt" / "aodt_static_assets.yml"


def make_context(run_dir, *, offline=True, slots="all"):
    return PipelineContext(
        repo_root=REPO,
        run_dir=Path(run_dir),
        scenario_path=SCENARIO,
        scene_archive=SCENE_ARCHIVE,
        aodt_client_root=AODT_CLIENT,
        asset_config=ASSET_CONFIG,
        worker_scene_url="test_data/maps/tokyo",
        endpoint="" if offline else "127.0.0.1:50051",
        offline=offline,
        ack_worker_assets_resolved=not offline,
        slots_spec=slots,
        full_antenna_pair=False,
        command="unit-test",
    )


class FakeSlotIndex:
    def __init__(self, value):
        self.value = value


class FakeAllocation:
    def __init__(self):
        self.temporal_indices = []


class FakeDigitalTwinClient:
    instances = []

    def __init__(self, endpoint, force=False):
        self.endpoint = endpoint
        self.force = force
        self.log_started = False
        self.stopped = False
        self.started_yaml = ""
        FakeDigitalTwinClient.instances.append(self)

    def is_connected(self):
        return True

    def transport_mode(self):
        return "REMOTE"

    def start_server_log_streaming(self, path, level):
        self.log_started = True
        return True

    def stop_server_log_streaming(self):
        self.stopped = True

    def start(self, yaml_content):
        if "${AODT_S3_" in yaml_content:
            return False
        self.started_yaml = yaml_content
        return True

    def get_status(self):
        return {
            "scenario_loaded": True,
            "num_rus": 8,
            "num_ues": 50,
            "total_batches": 1,
            "is_slot_symbol_mode": True,
            "num_slots_or_timesteps_per_batch": 256,
        }

    def get_ru_positions(self):
        return [[float(index), 0.0, 10.0] for index in range(8)]

    def get_ue_positions(self, batch_index, temporal_index):
        return [[float(index), 1.0, 1.5] for index in range(50)]

    def run_full_simulation(self):
        return {"time_steps_completed": 256, "total_time_seconds": 1.0}

    def allocate_cirs_memory(
        self,
        ru_indices,
        ue_indices_per_ru,
        is_full_antenna_pair=False,
        num_time_steps=None,
    ):
        return FakeAllocation()

    def get_cirs(self, allocation, batch_index, temporal_index):
        allocation.temporal_indices = [temporal_index.value]

    def to_numpy(self, allocation, temporal_idx, ru_idx, name):
        if name == "values":
            return np.ones((1, 1, 2), dtype=np.complex64)
        if name == "delays":
            return np.ones((1, 2), dtype=np.float32)
        return np.ones((1, 2, 2), dtype=np.float32)

    def to_numpy_all_cir(self, allocation):
        slot = allocation.temporal_indices[0]
        result = {}
        for name in (
            "values",
            "delays",
            "angles_of_departure",
            "angles_of_arrival",
        ):
            result[name] = {slot: {}}
            for ru in range(8):
                if name == "values":
                    array = np.ones((50, 2), dtype=np.complex64)
                elif name == "delays":
                    array = np.ones((50, 2), dtype=np.float32)
                else:
                    array = np.ones((50, 2, 2), dtype=np.float32)
                result[name][slot][ru] = array
        return result

    def deallocate_cirs_memory(self, allocation):
        return None


def fake_dt_module():
    module = types.ModuleType("dt_client")
    module.DigitalTwinClient = FakeDigitalTwinClient
    module.SlotIndex = FakeSlotIndex
    return module


class TestAODTStaticOfflineContracts(unittest.TestCase):
    def test_locked_scenario_and_positive_id_mapping(self):
        scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
        validation = validate_locked_scenario(scenario)
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["counts"]["ru"], 8)
        self.assertEqual(validation["counts"]["ue"], 50)
        self.assertEqual(validation["counts"]["aerial_ue"], 0)
        ru_rows = _node_rows(scenario["bs"], "RU")
        ue_rows = _node_rows(scenario["ue"], "UE")
        self.assertEqual([row["aodt_id"] for row in ru_rows], list(range(1, 9)))
        self.assertEqual([row["aodt_id"] for row in ue_rows], list(range(1, 51)))
        self.assertEqual(
            [row["tensor_api_index"] for row in ue_rows], list(range(50))
        )

    def test_uav_or_gpx_content_is_rejected(self):
        scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
        bad = copy.deepcopy(scenario)
        bad["ue"][0]["mobility_source"] = "route.gpx"
        bad["ue"][0]["ue_type"] = "UAV"
        validation = validate_locked_scenario(bad)
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(validation["prohibited_content"])

    def test_00b_through_05b_offline_artifacts_and_loader_roundtrip(self):
        if not SCENE_ARCHIVE.is_file():
            self.skipTest(f"Tokyo scene fixture is missing: {SCENE_ARCHIVE}")
        if not AODT_CLIENT.is_dir():
            self.skipTest(f"AODT client checkout is missing: {AODT_CLIENT}")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            ctx = make_context(run_dir)
            self.assertEqual(stage_00b(ctx)["status"], "pass")
            self.assertEqual(stage_01b(ctx)["status"], "pass")
            self.assertEqual(stage_02b(ctx)["status"], "pass_with_online_gate")
            self.assertEqual(stage_03b(ctx)["status"], "pass")
            self.assertEqual(stage_04b(ctx)["status"], "pass")
            preflight = stage_05b(ctx)
            self.assertEqual(preflight["status"], "blocked_offline")
            self.assertFalse(preflight["execution_authorized"])

            required = {
                "01B_inputs": {
                    "aodt_static_input_manifest.json",
                    "scenario_contract_validation.json",
                    "topology_validation.json",
                },
                "02B_scene_materials": {
                    "aodt_scene_inventory.json",
                    "aodt_geometry_inventory.json",
                    "aodt_material_inventory.json",
                    "aodt_material_binding_inventory.csv",
                    "aodt_material_validation.json",
                    "aodt_material_report.md",
                },
                "03B_topology": {
                    "ru_mapping.csv",
                    "ue_mapping.csv",
                    "aodt_node_mapping.json",
                    "static_topology_snapshot.csv",
                    "topology_mapping_report.md",
                },
                "04B_config": {
                    "resolved_aodt_static_config.yml",
                    "resolved_scene_config.json",
                    "resolved_radio_config.json",
                    "resolved_output_config.json",
                    "config_resolution_report.md",
                },
                "05B_preflight": {
                    "preflight_status.json",
                    "environment_inventory.json",
                    "preflight_report.md",
                },
            }
            for directory, names in required.items():
                actual = {
                    path.name
                    for path in (run_dir / directory).iterdir()
                    if path.is_file()
                }
                self.assertTrue(names.issubset(actual), (directory, names - actual))

            config = yaml.safe_load(
                (run_dir / "04B_config" / "resolved_aodt_static_config.yml").read_text(
                    encoding="utf-8"
                )
            )
            scenario_attributes = config["sim"]["Scenario"]["update"][0]["attributes"]
            self.assertEqual(scenario_attributes["sim_em_interactions"], 5)
            radio = json.loads(
                (run_dir / "04B_config" / "resolved_radio_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(radio["aodt_backend"]["ray_interactions"], 5)
            self.assertEqual(
                radio["aodt_backend"]["source_requested_max_depth"], 6
            )
            self.assertEqual(
                sorted(item["id"] for item in config["sim"]["RUs"]["add"]),
                list(range(1, 9)),
            )
            self.assertEqual(
                sorted(item["id"] for item in config["sim"]["UEs"]["add"]),
                list(range(1, 51)),
            )
            self.assertTrue(
                all(len(item["waypoints"]) == 1 for item in config["sim"]["UEs"]["add"])
            )
            self.assertIn(
                "${AODT_S3_SECRET_KEY}", json.dumps(config, ensure_ascii=False)
            )

    def test_slot_parser(self):
        self.assertEqual(parse_slots("all", 4), [0, 1, 2, 3])
        self.assertEqual(parse_slots("0,5,9", 10), [0, 5, 9])
        self.assertEqual(parse_slots("1:8:3", 10), [1, 4, 7])
        with self.assertRaises(Exception):
            parse_slots("0,10", 10)

    def test_aodt_binding_members_support_properties_and_methods(self):
        property_client = types.SimpleNamespace(
            is_connected=True,
            transport_mode="REMOTE",
            has_gpu=False,
        )
        method_client = types.SimpleNamespace(
            is_connected=lambda: True,
            transport_mode=lambda: "REMOTE",
            has_gpu=lambda: False,
        )
        for client in (property_client, method_client):
            self.assertIs(aodt_member_value(client, "is_connected"), True)
            self.assertEqual(
                aodt_member_value(client, "transport_mode"), "REMOTE"
            )
            self.assertIs(aodt_member_value(client, "has_gpu"), False)


class TestAODTStaticRuntimeWithFakeClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not AODT_CLIENT.is_dir():
            raise unittest.SkipTest(f"AODT client checkout is missing: {AODT_CLIENT}")

    def test_06b_and_07b_collect_complete_selected_raw_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            ctx = make_context(run_dir, offline=False, slots="0,2")
            config_dir = run_dir / "04B_config"
            config_dir.mkdir()
            (config_dir / "resolved_aodt_static_config.yml").write_text(
                "db:\n"
                "  s3_config:\n"
                "    access_key: ${AODT_S3_ACCESS_KEY}\n"
                "    secret_key: ${AODT_S3_SECRET_KEY}\n",
                encoding="utf-8",
            )
            preflight_dir = run_dir / "05B_preflight"
            preflight_dir.mkdir()
            (preflight_dir / "preflight_status.json").write_text(
                json.dumps({"status": "pass", "execution_authorized": True}),
                encoding="utf-8",
            )

            FakeDigitalTwinClient.instances = []
            with mock.patch.dict(
                sys.modules, {"dt_client": fake_dt_module()}
            ), mock.patch.dict(
                os.environ,
                {
                    "AODT_S3_ACCESS_KEY": "test-access",
                    "AODT_S3_SECRET_KEY": "test-secret",
                },
                clear=False,
            ):
                execution, client = stage_06b(ctx, keep_session_for_07b=True)
                self.assertEqual(execution["status"], "pass")
                raw = stage_07b(ctx, client)

            self.assertEqual(raw["status"], "pass")
            self.assertEqual(raw["selected_slot_count"], 2)
            self.assertEqual(raw["raw_npz_count"], 16)
            raw_dir = run_dir / "07B_raw_collection"
            inventory = json.loads(
                (raw_dir / "raw_artifact_inventory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(inventory["expected_file_count"], 16)
            self.assertEqual(inventory["actual_file_count"], 16)
            selected = json.loads(
                (raw_dir / "selected_raw_inputs.json").read_text(encoding="utf-8")
            )
            self.assertFalse(selected["fallback_used"])
            self.assertFalse(selected["representative_csv_used"])
            first = raw_dir / selected["selected_files"][0]
            with np.load(first) as data:
                self.assertEqual(data["ue_aodt_ids"].tolist(), list(range(1, 51)))
                self.assertIn("angles_of_departure", data.files)
                self.assertIn("angles_of_arrival", data.files)


if __name__ == "__main__":
    unittest.main()
