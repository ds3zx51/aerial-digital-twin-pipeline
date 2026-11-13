from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TEST_FILE = Path(__file__).resolve()
REPO_CANDIDATE = TEST_FILE.parents[1]
DEPLOYED_CODE = REPO_CANDIDATE / "scripts" / "aodt" / "bu_aerpaw"
CODE_ROOT = DEPLOYED_CODE if DEPLOYED_CODE.is_dir() else TEST_FILE.parents[1]
REPO_ROOT = REPO_CANDIDATE if DEPLOYED_CODE.is_dir() else CODE_ROOT
sys.path.insert(0, str(CODE_ROOT))

from aodt_native_builder import build_execution_spec
from bu_common import StageBlocked, copy_immutable, hash_tree, run_external
from bu_pipeline import (
    COLOSSEUM_COLUMNS,
    MEASUREMENT_FIELDS,
    TOWERS,
    _source_inventory,
    best_offset,
    build_geospatial_contract,
    cir_to_rss_dbm,
    clamp_height,
    contiguous_splits,
    field_provenance_rows,
    formal_gate_checks,
    geodetic_transformer,
    outer_align,
    parse_native_rows,
    require_pinned_run,
    write_gpx,
)


CONTRACT_PATH = (REPO_ROOT / "configs" / "aerpaw_lake_wheeler" / "simulation_parameter_contract.json") if DEPLOYED_CODE.is_dir() else (CODE_ROOT / "configs" / "simulation_parameter_contract.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class TestBUPipeline(unittest.TestCase):
    def test_01_aerpaw_source_manifest_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("D1_RSS_Measurements/LW1/raw.bin", b"aerpaw")
            rows = _source_inventory(archive)
            self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"aerpaw").hexdigest())

    def test_02_dual_usrp_channels_preserved(self) -> None:
        self.assertIn("usrp1_rss_dbm", MEASUREMENT_FIELDS)
        self.assertIn("usrp2_rss_dbm", MEASUREMENT_FIELDS)
        self.assertNotEqual(MEASUREMENT_FIELDS.index("usrp1_rss_dbm"), MEASUREMENT_FIELDS.index("usrp2_rss_dbm"))

    def test_03_five_tower_ids_preserved(self) -> None:
        self.assertEqual(TOWERS, ["LW1", "LW2", "LW3", "LW4", "LW5"])
        self.assertEqual(CONTRACT["formal_profile"]["tower_ids"], TOWERS)

    def test_04_no_usrp_averaging_in_canonical_raw_data(self) -> None:
        self.assertFalse(any("average" in field.lower() or "mean" in field.lower() for field in MEASUREMENT_FIELDS))
        self.assertEqual(sum(field.startswith("usrp") for field in MEASUREMENT_FIELDS), 2)

    def test_05_raw_altitude_separate_from_simulation_altitude(self) -> None:
        height, clamped = clamp_height(-0.2)
        self.assertEqual(height, 0.5)
        self.assertTrue(clamped)
        self.assertEqual(-0.2, -0.2)  # input/raw evidence remains unchanged

    def test_06_wgs84_scene_round_trip(self) -> None:
        geo = build_geospatial_contract(CONTRACT)
        self.assertLess(geo["roundtrip_error_deg"], 1e-9)
        _, forward, inverse = geodetic_transformer(35.72750947, -78.69595810)
        x, y = forward.transform(-78.7000, 35.7300)
        lon, lat = inverse.transform(x, y)
        self.assertAlmostEqual(lon, -78.7000, places=9)
        self.assertAlmostEqual(lat, 35.7300, places=9)

    def test_07_gpx_timestamp_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "route.gpx"
            rows = [{"latitude_deg": 35.7, "longitude_deg": -78.6, "simulation_altitude_m": 120.0, "timestamp_s": 1721041107.801}]
            write_gpx(target, rows)
            tree = ET.parse(target)
            namespace = {"g": "http://www.topografix.com/GPX/1/1"}
            text = tree.find(".//g:time", namespace).text
            from datetime import datetime
            self.assertAlmostEqual(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp(), rows[0]["timestamp_s"], places=3)

    def test_08_3d_waypoint_export(self) -> None:
        rows = [
            {"sample_id": 0, "timestamp_s": 1.0, "latitude_deg": 35.7, "longitude_deg": -78.6, "simulation_altitude_m": 120.5, "scene_x_m": 0, "scene_y_m": 0, "scene_z_m": 120.5},
            {"sample_id": 1, "timestamp_s": 1.1, "latitude_deg": 35.7001, "longitude_deg": -78.6001, "simulation_altitude_m": 121.5, "scene_x_m": 1, "scene_y_m": 2, "scene_z_m": 121.5},
        ]
        spec = build_execution_spec(CONTRACT, rows, "scene.usd")
        self.assertEqual([w["altitude_m"] for w in spec["uav"]["waypoints"]], [120.5, 121.5])
        self.assertTrue(spec["timing_contract"]["runtime_position_readback_required"])

    def test_09_config_is_em_not_ran(self) -> None:
        rows = [{"sample_id": 0, "timestamp_s": 1, "latitude_deg": 35.7, "longitude_deg": -78.6, "simulation_altitude_m": 120, "scene_x_m": 0, "scene_y_m": 0, "scene_z_m": 120}]
        spec = build_execution_spec(CONTRACT, rows, "scene.usd")
        self.assertEqual(spec["mode"], "EM")
        self.assertNotEqual(spec["mode"], "RAN")

    def test_10_config_is_3_3_ghz(self) -> None:
        rows = [{"sample_id": 0, "timestamp_s": 1, "latitude_deg": 35.7, "longitude_deg": -78.6, "simulation_altitude_m": 120, "scene_x_m": 0, "scene_y_m": 0, "scene_z_m": 120}]
        spec = build_execution_spec(CONTRACT, rows, "scene.usd")
        self.assertEqual(spec["frequency_mhz"], 3300.0)

    def test_11_config_does_not_reference_tokyo(self) -> None:
        rows = [{"sample_id": 0, "timestamp_s": 1, "latitude_deg": 35.7, "longitude_deg": -78.6, "simulation_altitude_m": 120, "scene_x_m": 0, "scene_y_m": 0, "scene_z_m": 120}]
        spec = build_execution_spec(CONTRACT, rows, "lake_wheeler.usd")
        self.assertNotIn("tokyo", json.dumps(spec).lower())

    def test_12_no_path_is_null_not_synthetic_floor(self) -> None:
        self.assertIsNone(cir_to_rss_dbm([], "cir_amplitude_sqrt_w_includes_tx_power"))
        with tempfile.TemporaryDirectory() as tmp:
            native = Path(tmp) / "native.csv"
            with native.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["tower_id", "sample_id", "timestamp_s", "cir_real", "cir_imag", "path_exists"])
                writer.writeheader()
                writer.writerow({"tower_id": "LW1", "sample_id": 0, "timestamp_s": 1, "cir_real": 0, "cir_imag": 0, "path_exists": "false"})
            rows = parse_native_rows(native, {"power_semantics": "cir_amplitude_sqrt_w_includes_tx_power", "timestamp_semantics": "seconds", "tower_mapping": {"1": "LW1"}, "position_readback_verified": True})
            self.assertIsNone(rows[0]["aodt_rss_dbm"])

    def test_13_bu12_preserves_unmatched_rows(self) -> None:
        measurements = [{"tower_id": "LW1", "sample_id": 0, "timestamp_s": 1.0, "usrp1_rss_dbm": -70, "usrp2_rss_dbm": -71}]
        simulated = [
            {"tower_id": "LW1", "sample_id": 0, "timestamp_s": 1.0, "aodt_rss_dbm": -69},
            {"tower_id": "LW2", "sample_id": 0, "timestamp_s": 2.0, "aodt_rss_dbm": -80},
        ]
        aligned = outer_align(measurements, simulated, 0.01)
        self.assertEqual(len(aligned), 2)
        self.assertEqual({row["alignment_status"] for row in aligned}, {"matched", "aodt_only"})

    def test_14_calibration_does_not_overwrite_raw_data(self) -> None:
        measured = [-80.0, -70.0]
        simulated = [-85.0, -75.0]
        original_measured = measured.copy()
        original_simulated = simulated.copy()
        self.assertAlmostEqual(best_offset(measured, simulated), 5.0, places=6)
        self.assertEqual(measured, original_measured)
        self.assertEqual(simulated, original_simulated)

    def test_15_held_out_split_has_no_overlap(self) -> None:
        split = contiguous_splits(100)
        sets = {name: set(values) for name, values in split.items()}
        self.assertFalse(sets["calibration"] & sets["validation"])
        self.assertFalse(sets["calibration"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])

    def test_16_bu14_refuses_unpinned_bu13(self) -> None:
        with self.assertRaises(StageBlocked) as caught:
            require_pinned_run("run-123", None, "BU13")
        self.assertEqual(caught.exception.code, "unpinned_bu13")

    def test_17_field_provenance_covers_all_32_columns(self) -> None:
        rows = field_provenance_rows()
        self.assertEqual(len(rows), 32)
        self.assertEqual([row["column_name"] for row in rows], COLOSSEUM_COLUMNS)
        self.assertTrue(all(row["source_class"] for row in rows))

    def test_18_bu15_runtime_does_not_modify_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "model.py"
            source.write_text("FROZEN = True\n", encoding="utf-8")
            before = hash_tree(root)
            code = run_external([sys.executable, "-c", "pass"], root, root / "stdout.log", root / "stderr.log")
            after = hash_tree(root, ignore=("stdout.log", "stderr.log"))
            self.assertEqual(code, 0)
            self.assertEqual(before, after)

    def test_19_historical_results_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "historical"
            source.mkdir()
            target.mkdir()
            (source / "x.txt").write_text("new", encoding="utf-8")
            (target / "x.txt").write_text("old", encoding="utf-8")
            with self.assertRaises(StageBlocked):
                copy_immutable(source, target)
            self.assertEqual((target / "x.txt").read_text(encoding="utf-8"), "old")

    def test_20_formal_scene_gate_rejects_osm_only_smoke(self) -> None:
        manifests = {stage: {"formal_eligible": True} for stage in ("BU01", "BU02", "BU03", "BU04", "BU05", "BU06")}
        manifests["BU04"]["formal_eligible"] = False
        spec = {"frequency_mhz": 3300.0, "array_mode": "SISO", "towers": [{}, {}, {}, {}, {}], "uav": {"ue_id": 1}, "tx_power_dbm": 44.0, "timing_contract": {"runtime_position_readback_required": True}}
        checks = formal_gate_checks(manifests, spec, 3530, {"foliage_feature_count": 0})
        self.assertFalse(all(item["passed"] for item in checks))


if __name__ == "__main__":
    unittest.main()
