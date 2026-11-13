from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


PIPELINE_VERSION = "1.0.0"
STAGES = {
    "BU00": "legacy_audit",
    "BU01": "source_registry",
    "BU02": "measurement_normalization",
    "BU03": "geospatial_contract",
    "BU04": "scene_contract",
    "BU05": "trajectory_builder",
    "BU06": "aodt_config_builder",
    "BU07": "formal_gate",
    "BU08": "immutable_bundle",
    "BU09": "aodt_em_smoke",
    "BU10": "full_aodt_em_replay",
    "BU11": "rss_extraction",
    "BU12": "measurement_alignment",
    "BU13": "physical_validation",
    "BU14": "advtabddpm_adapter",
    "BU15": "frozen_consumer",
}


class StageBlocked(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_slug(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in value).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, *, ignore: Iterable[str] = ()) -> dict[str, str]:
    ignored = set(ignore)
    result: dict[str, str] = {}
    if not root.exists():
        return result
    if root.is_file():
        return {root.name: sha256_file(root)}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if any(rel == item or rel.startswith(item.rstrip("/") + "/") for item in ignored):
            continue
        result[rel] = sha256_file(path)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path, code: str = "missing_input") -> Path:
    if not path.is_file():
        raise StageBlocked(code, f"Required file is missing: {path}")
    return path


def require_dir(path: Path, code: str = "missing_input") -> Path:
    if not path.is_dir():
        raise StageBlocked(code, f"Required directory is missing: {path}")
    return path


def load_upstream(run_dir: str | None, expected_stage: str | None = None) -> tuple[Path, dict[str, Any]]:
    if not run_dir:
        raise StageBlocked("explicit_upstream_required", "An explicit upstream run directory is required; latest-run discovery is forbidden.")
    path = require_dir(Path(run_dir).resolve())
    manifest_path = require_file(path / "run_manifest.json", "invalid_upstream")
    manifest = read_json(manifest_path)
    if expected_stage and manifest.get("stage") != expected_stage:
        raise StageBlocked(
            "wrong_upstream_stage",
            f"Expected {expected_stage}, received {manifest.get('stage')} at {path}",
        )
    if manifest.get("status") not in {"success", "blocked"}:
        raise StageBlocked("invalid_upstream_status", f"Upstream status is not consumable: {manifest.get('status')}")
    return path, manifest


def verify_declared_hashes(run_dir: Path) -> None:
    manifest_path = require_file(run_dir / "output_sha256.txt", "missing_upstream_hashes")
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        path = run_dir / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise StageBlocked("upstream_hash_mismatch", f"Hash mismatch: {path}")


def _write_hash_list(path: Path, entries: dict[str, str]) -> None:
    path.write_text("".join(f"{digest}  {rel}\n" for rel, digest in sorted(entries.items())), encoding="utf-8")


@dataclass
class StageContext:
    stage: str
    repo_root: Path
    result_dir: Path
    run_id: str
    args: dict[str, Any]

    @property
    def contract_path(self) -> Path:
        return self.repo_root / "configs" / "aerpaw_lake_wheeler" / "simulation_parameter_contract.json"

    @property
    def contract(self) -> dict[str, Any]:
        return read_json(self.contract_path)

    def block(self, code: str, message: str) -> None:
        raise StageBlocked(code, message)


def create_result_dir(results_root: Path, stage: str) -> tuple[Path, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stage.lower()}_{timestamp}_{uuid.uuid4().hex[:8]}"
    result = results_root / f"{stage.lower()}_{STAGES[stage]}_{timestamp}_{uuid.uuid4().hex[:8]}"
    result.mkdir(parents=True, exist_ok=False)
    return result, run_id


def run_stage(
    stage: str,
    repo_root: Path,
    results_root: Path,
    args: dict[str, Any],
    implementation: Callable[[StageContext], dict[str, Any]],
) -> tuple[int, Path]:
    result_dir, run_id = create_result_dir(results_root, stage)
    started = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage,
        "stage_name": STAGES[stage],
        "run_id": run_id,
        "started_at_utc": started,
        "status": "running",
        "repo_root": str(repo_root),
        "arguments": args,
        "parameter_contract_sha256": sha256_file(repo_root / "configs" / "aerpaw_lake_wheeler" / "simulation_parameter_contract.json"),
        "source_run_ids": [],
    }
    write_json(result_dir / "run_manifest.json", manifest)
    stdout_path = result_dir / "stdout.log"
    stderr_path = result_dir / "stderr.log"
    code = 0
    summary: dict[str, Any]
    ctx = StageContext(stage, repo_root, result_dir, run_id, args)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        with redirect_stdout(out), redirect_stderr(err):
            try:
                summary = implementation(ctx)
                manifest["status"] = "success"
                manifest["formal_eligible"] = bool(summary.get("formal_eligible", False))
            except StageBlocked as exc:
                code = 2
                summary = {"status": "blocked", "blocker_code": exc.code, "message": str(exc), "formal_eligible": False}
                manifest.update({"status": "blocked", "blocker_code": exc.code, "formal_eligible": False})
                print(f"BLOCKED [{exc.code}]: {exc}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover - defensive process boundary
                code = 1
                summary = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc), "formal_eligible": False}
                manifest.update({"status": "failed", "error_type": type(exc).__name__, "formal_eligible": False})
                traceback.print_exc(file=sys.stderr)
    manifest["completed_at_utc"] = utc_now()
    manifest["exit_code"] = code
    for key in ("source_run_ids", "source_artifacts", "formal_profile"):
        if key in summary:
            manifest[key] = summary[key]
    write_json(result_dir / "summary.json", summary)
    write_json(result_dir / "run_manifest.json", manifest)

    inputs: dict[str, str] = {"parameter_contract.json": manifest["parameter_contract_sha256"]}
    for value in args.values():
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, str):
                path = Path(candidate)
                if path.is_file():
                    inputs[str(path.resolve())] = sha256_file(path)
    _write_hash_list(result_dir / "input_sha256.txt", inputs)

    excluded = {"output_sha256.txt"}
    outputs = {
        path.relative_to(result_dir).as_posix(): sha256_file(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    _write_hash_list(result_dir / "output_sha256.txt", outputs)
    return code, result_dir


def copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        raise StageBlocked("destination_exists", f"Refusing to overwrite: {destination}")
    shutil.copytree(source, destination)


def run_external(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    import subprocess

    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=err, check=False)
    return int(completed.returncode)
