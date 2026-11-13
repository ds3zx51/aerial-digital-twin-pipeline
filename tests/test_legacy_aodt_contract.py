from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts" / "adapter" / "legacy_aodt_contract_adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("legacy_aodt_contract_adapter", ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyAODTContractTests(unittest.TestCase):
    def test_frozen_contract_and_proxy_qualification(self) -> None:
        adapter = load_adapter()
        self.assertEqual(len(adapter.COLOSSEUM_COLUMNS), 32)
        self.assertEqual(adapter.COLOSSEUM_COLUMNS[0], "traffic_type")
        self.assertEqual(adapter.COLOSSEUM_COLUMNS[19], "ul_sinr")
        self.assertEqual(adapter.COLOSSEUM_COLUMNS[-1], "scheduling_policy")
        qualification = adapter.provenance(2023)["semantic_qualifications"]["ul_sinr"]
        self.assertEqual(qualification["semantic_role"], "aodt_em_proxy_ul_sinr_db")
        self.assertEqual(
            qualification["semantic_equivalence_to_colosseum"],
            "NOT_YET_VERIFIED_SEMANTIC_EQUIVALENT",
        )
        self.assertFalse(qualification["physical_distribution_comparison_allowed"])

    def test_adapter_is_library_only(self) -> None:
        adapter = load_adapter()
        self.assertFalse(hasattr(adapter, "main"))
        self.assertFalse(hasattr(adapter, "convert"))


if __name__ == "__main__":
    unittest.main()
