#!/usr/bin/env python3
"""Standalone BU scenario schema validation entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate_shared_multi_uav_gpx import validate_bu_scenario
except ImportError:
    from validate_shared_multi_uav_gpx import validate_bu_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.scenario.read_text(encoding="utf-8"))
        errors: list[str] = []
        validate_bu_scenario(payload, errors)
    except Exception as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f"ERROR={error}")
        return 1
    print("STATUS=bu_scenario_validation_succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
