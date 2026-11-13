#!/usr/bin/env python3
"""Run AODT stages 00BU through 07BU for deterministic GPX tracks."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


def _bootstrap_repo(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path)
    known, _ = parser.parse_known_args(argv)
    if known.repo_root:
        repo = known.repo_root.resolve()
    else:
        repo = Path(__file__).resolve().parents[2]
    module_dir = repo / "scripts" / "aodt"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    return repo


BOOTSTRAP_REPO = _bootstrap_repo(sys.argv[1:])

from uav_core import (  # noqa: E402
    BU_STAGES,
    PipelineError,
    UAVPipelineContext,
    create_unique_run_dir,
    path_text,
    stage_00bu,
    stage_01bu,
    stage_02bu,
    stage_03bu,
    stage_04bu,
    stage_05bu,
    update_run_manifest,
)


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _parser() -> argparse.ArgumentParser:
    repo = BOOTSTRAP_REPO
    parser = argparse.ArgumentParser(
        description=(
            "Run AODT 1.5 00BU-07BU for 8 RU, 50 fixed ground UE, and "
            "four UAVs driven by the repository's deterministic GPX files."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=repo / "configs" / "scenarios" / "bu" / "tokyo_BU_multi_uav.json",
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
    parser.add_argument("--worker-scene-url", default="test_data/maps/tokyo")
    parser.add_argument(
        "--worker-gpx-prefix",
        default="test_data/gpx/shared/tokyo_u4_boundary_60m_256",
        help="Worker S3 key prefix containing the four hash-matched GPX files.",
    )
    parser.add_argument(
        "--endpoint", default=os.environ.get("DT_SERVER_ADDRESS", "")
    )
    parser.add_argument(
        "--results-root", type=Path, default=repo / "results" / "aodt_uav"
    )
    parser.add_argument("--end-stage", choices=BU_STAGES, default="07BU")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ack-worker-assets-resolved", action="store_true")
    parser.add_argument("--ack-worker-gpx-resolved", action="store_true")
    parser.add_argument(
        "--slots",
        default="all",
        help="07BU time steps: all, comma list, or Python-style range.",
    )
    parser.add_argument("--full-antenna-pair", action="store_true")
    parser.add_argument("--position-tolerance-m", type=float, default=2.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for label, path in {
        "scenario": args.scenario,
        "scene archive": args.scene_archive,
        "asset config": args.asset_config,
    }.items():
        if path is None or not path.is_file():
            raise PipelineError(f"{label} file does not exist: {path}")
    if args.aodt_client_root is None or not args.aodt_client_root.is_dir():
        raise PipelineError(
            f"AODT client repository does not exist: {args.aodt_client_root}"
        )
    if args.position_tolerance_m <= 0:
        raise PipelineError("position-tolerance-m must be positive.")
    if args.offline and BU_STAGES.index(args.end_stage) > BU_STAGES.index("05BU"):
        raise PipelineError("Offline mode may not proceed beyond 05BU.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    run_dir: Path | None = None
    try:
        _validate_args(args)
        run_dir = create_unique_run_dir(args.results_root.resolve())
        command = " ".join(
            shlex.quote(item)
            for item in [
                str(Path(sys.executable).resolve()),
                str(Path(__file__).resolve()),
                *argv,
            ]
        )
        ctx = UAVPipelineContext(
            repo_root=args.repo_root.resolve(),
            run_dir=run_dir,
            scenario_path=args.scenario.resolve(),
            scene_archive=args.scene_archive.resolve(),
            aodt_client_root=args.aodt_client_root.resolve(),
            asset_config=args.asset_config.resolve(),
            worker_scene_url=args.worker_scene_url,
            worker_gpx_prefix=args.worker_gpx_prefix,
            endpoint=args.endpoint.strip(),
            offline=bool(args.offline),
            ack_worker_assets_resolved=bool(args.ack_worker_assets_resolved),
            ack_worker_gpx_resolved=bool(args.ack_worker_gpx_resolved),
            slots_spec=args.slots,
            full_antenna_pair=bool(args.full_antenna_pair),
            position_tolerance_m=float(args.position_tolerance_m),
            command=command,
        )
        update_run_manifest(ctx)
        handlers = {
            "00BU": stage_00bu,
            "01BU": stage_01bu,
            "02BU": stage_02bu,
            "03BU": stage_03bu,
            "04BU": stage_04bu,
            "05BU": stage_05bu,
        }
        end_index = BU_STAGES.index(args.end_stage)
        preflight = None
        for stage in BU_STAGES[: min(end_index + 1, 6)]:
            print(f"[{stage}] starting", flush=True)
            result = handlers[stage](ctx)
            update_run_manifest(ctx, stage, result)
            print(f"[{stage}] {result['status']}", flush=True)
            if stage == "05BU":
                preflight = result
        client = None
        if end_index >= BU_STAGES.index("06BU"):
            if preflight is None or preflight.get("status") != "pass":
                raise PipelineError(
                    "Online 05BU preflight did not pass; 06BU/07BU were not started."
                )
            from uav_runtime import stage_06bu

            print("[06BU] starting full AODT GPX simulation", flush=True)
            execution, client = stage_06bu(
                ctx, keep_session_for_07bu=end_index >= BU_STAGES.index("07BU")
            )
            update_run_manifest(ctx, "06BU", execution)
            print(f"[06BU] {execution['status']}", flush=True)
        if end_index >= BU_STAGES.index("07BU"):
            from uav_runtime import stage_07bu

            print("[07BU] starting complete raw collection", flush=True)
            raw = stage_07bu(ctx, client)
            update_run_manifest(ctx, "07BU", raw)
            print(f"[07BU] {raw['status']}", flush=True)
        if args.end_stage == "05BU" and preflight:
            if preflight["status"] == "blocked_offline":
                overall = "ready_through_04BU_blocked_at_05BU_offline"
            elif preflight["status"] != "pass":
                overall = "failed_at_05BU"
                update_run_manifest(ctx, overall_status=overall)
                print(f"RUN_DIR={path_text(run_dir)}", flush=True)
                print(f"OVERALL_STATUS={overall}", flush=True)
                return 2
            else:
                overall = "pass_through_05BU"
        else:
            overall = f"pass_through_{args.end_stage}"
        update_run_manifest(ctx, overall_status=overall)
        print(f"RUN_DIR={path_text(run_dir)}", flush=True)
        print(f"OVERALL_STATUS={overall}", flush=True)
        return 0
    except Exception as exc:
        if run_dir is not None and "ctx" in locals():
            try:
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
