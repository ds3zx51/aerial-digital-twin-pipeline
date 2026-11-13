#!/usr/bin/env python3
"""Run the AODT 1.5 fixed-UE 00B–07B pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from static_core import (
    STAGES,
    PipelineContext,
    PipelineError,
    create_unique_run_dir,
    path_text,
    stage_00b,
    stage_01b,
    stage_02b,
    stage_03b,
    stage_04b,
    stage_05b,
    update_run_manifest,
)


def _default_repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _parser() -> argparse.ArgumentParser:
    repo = _default_repo()
    parser = argparse.ArgumentParser(
        description=(
            "Build and run AODT 1.5 stages 00B–07B for the locked Tokyo "
            "8-RU/50-fixed-UE scenario."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=repo / "configs" / "scenario_layout_tokyo_docomo.json",
    )
    parser.add_argument(
        "--scene-archive",
        type=Path,
        default=_path_from_env("AODT_SCENE_ARCHIVE"),
        help="AODT OpenUSD scene archive; defaults to AODT_SCENE_ARCHIVE.",
    )
    parser.add_argument(
        "--aodt-client-root",
        type=Path,
        default=_path_from_env("AODT_CLIENT_ROOT"),
        help="Open-source AODT checkout; defaults to AODT_CLIENT_ROOT.",
    )
    parser.add_argument(
        "--asset-config",
        type=Path,
        default=repo / "configs" / "aodt" / "aodt_static_assets.yml",
    )
    parser.add_argument(
        "--worker-scene-url",
        default="test_data/maps/tokyo",
        help="Scene prefix as resolved on the Azure AODT worker.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("DT_SERVER_ADDRESS", ""),
        help="AODT worker host:port. Defaults to DT_SERVER_ADDRESS.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo / "results" / "aodt_static",
    )
    parser.add_argument(
        "--end-stage",
        choices=STAGES,
        default="07B",
        help="Stop after this stage. Execution always begins at frozen 00B.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Build through 05B without contacting an Azure worker.",
    )
    parser.add_argument(
        "--ack-worker-assets-resolved",
        action="store_true",
        help=(
            "Assert that the exact scene/material/vegetation asset references "
            "in configs/aodt/aodt_static_assets.yml resolve on the worker."
        ),
    )
    parser.add_argument(
        "--slots",
        default="all",
        help="07B slots: all, comma list (0,5), or Python-style range (0:256).",
    )
    parser.add_argument(
        "--full-antenna-pair",
        action="store_true",
        help="Collect full antenna-pair tensors instead of the AODT compact mode.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    required_files = {
        "scenario": args.scenario,
        "scene archive": args.scene_archive,
        "asset config": args.asset_config,
    }
    for label, path in required_files.items():
        if path is None or not path.is_file():
            raise PipelineError(f"{label} file does not exist: {path}")
    if args.aodt_client_root is None or not args.aodt_client_root.is_dir():
        raise PipelineError(
            f"AODT client repository does not exist: {args.aodt_client_root}"
        )
    if args.offline and STAGES.index(args.end_stage) > STAGES.index("05B"):
        raise PipelineError("Offline mode may not proceed beyond 05B.")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    run_dir: Path | None = None
    try:
        _validate_args(args)
        run_dir = create_unique_run_dir(args.results_root.resolve())
        command = " ".join(
            shlex.quote(item)
            for item in [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
        )
        ctx = PipelineContext(
            repo_root=args.repo_root.resolve(),
            run_dir=run_dir,
            scenario_path=args.scenario.resolve(),
            scene_archive=args.scene_archive.resolve(),
            aodt_client_root=args.aodt_client_root.resolve(),
            asset_config=args.asset_config.resolve(),
            worker_scene_url=args.worker_scene_url,
            endpoint=args.endpoint.strip(),
            offline=bool(args.offline),
            ack_worker_assets_resolved=bool(args.ack_worker_assets_resolved),
            slots_spec=args.slots,
            full_antenna_pair=bool(args.full_antenna_pair),
            command=command,
        )
        update_run_manifest(ctx)
        handlers = {
            "00B": stage_00b,
            "01B": stage_01b,
            "02B": stage_02b,
            "03B": stage_03b,
            "04B": stage_04b,
            "05B": stage_05b,
        }
        end_index = STAGES.index(args.end_stage)
        preflight = None
        for stage in STAGES[: min(end_index + 1, 6)]:
            print(f"[{stage}] starting", flush=True)
            result = handlers[stage](ctx)
            update_run_manifest(ctx, stage, result)
            print(f"[{stage}] {result['status']}", flush=True)
            if stage == "05B":
                preflight = result

        client = None
        if end_index >= STAGES.index("06B"):
            if preflight is None or preflight.get("status") != "pass":
                raise PipelineError(
                    "Online 05B preflight did not pass; 06B/07B were not started."
                )
            from aodt_runtime import stage_06b

            print("[06B] starting full AODT simulation", flush=True)
            execution, client = stage_06b(
                ctx, keep_session_for_07b=end_index >= STAGES.index("07B")
            )
            update_run_manifest(ctx, "06B", execution)
            print(f"[06B] {execution['status']}", flush=True)

        if end_index >= STAGES.index("07B"):
            from aodt_runtime import stage_07b

            print("[07B] starting complete raw collection", flush=True)
            raw = stage_07b(ctx, client)
            update_run_manifest(ctx, "07B", raw)
            print(f"[07B] {raw['status']}", flush=True)

        if args.end_stage == "05B" and preflight and preflight["status"] == "blocked_offline":
            overall = "ready_through_04B_blocked_at_05B_offline"
        else:
            overall = f"pass_through_{args.end_stage}"
        update_run_manifest(ctx, overall_status=overall)
        print(f"RUN_DIR={path_text(run_dir)}", flush=True)
        print(f"OVERALL_STATUS={overall}", flush=True)
        return 0
    except Exception as exc:
        if run_dir is not None:
            try:
                # Rebuild a minimal context only if argument validation reached run creation.
                if "ctx" in locals():
                    update_run_manifest(
                        ctx, overall_status=f"fail: {type(exc).__name__}: {exc}"
                    )
            except Exception:
                pass
            print(f"RUN_DIR={path_text(run_dir)}", file=sys.stderr)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
