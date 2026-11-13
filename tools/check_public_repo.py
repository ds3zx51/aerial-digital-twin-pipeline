#!/usr/bin/env python3
"""Fail when a tracked-source candidate contains common release hazards."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bat", ".cff", ".csv", ".gpx", ".json", ".md", ".mjs", ".py",
    ".sh", ".toml", ".txt", ".yaml", ".yml",
}
IGNORED_TOP_LEVEL = {".git", ".venv", "assets", "logs", "packages", "results"}
PROHIBITED = {
    "removed_backend_name": re.compile(r"sionna", re.IGNORECASE),
    "private_ipv4": re.compile(r"\b140\.115\.152\.169\b"),
    "private_user_path": re.compile(r"(?i)(?:C:\\Users\\Edword|D:\\AODT|/home/ipc)"),
    "demo_password": re.compile(r"aerial_123456"),
}


def candidate_files() -> list[Path]:
    result: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in IGNORED_TOP_LEVEL:
            continue
        result.append(path)
    return sorted(result)


def main() -> int:
    errors: list[str] = []
    for path in candidate_files():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for label, pattern in PROHIBITED.items():
            if pattern.search(text):
                errors.append(f"{label}: {path.relative_to(ROOT).as_posix()}")

    forbidden_paths = (
        ROOT / "scripts" / "remote" / "remote_settings.bat",
        ROOT / "AGENTS.md",
    )
    errors.extend(
        f"forbidden_path: {path.relative_to(ROOT).as_posix()}"
        for path in forbidden_paths
        if path.exists()
    )

    if errors:
        print("PUBLIC_REPO_CHECK=FAIL")
        for error in errors:
            print(error)
        return 1
    print(f"PUBLIC_REPO_CHECK=PASS files={len(candidate_files())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
