#!/usr/bin/env python3
"""Run only static stage 09B from a completed 08B/07B result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from static_postprocess import build_stages, utc_now, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    result = build_stages(args.repo_root.resolve(), run_dir, stop_after_09b=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("stages", {})["09B"] = {
        "stage": "09B", "status": "pass", **result["09B"], "directory": result["stage09"]
    }
    manifest["overall_status"] = "pass_through_09B"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps(result["09B"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
