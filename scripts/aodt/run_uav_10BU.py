#!/usr/bin/env python3
"""Run only UAV stage 10BU from an existing 09BU channel trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uav_postprocess import build_10bu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_10bu(args.repo_root.resolve(), args.run_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
