"""Deterministic result documents and evaluation provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import scipy


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state(root: Path) -> dict[str, object]:
    def run(*args: str) -> str | None:
        done = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
        return done.stdout.strip() if done.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment_provenance(lock_path: Path, trackeval: dict[str, object]) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "dependency_lock_sha256": sha256_file(lock_path),
        "trackeval": trackeval,
    }


def write_result(output: Path, result: dict[str, object]) -> None:
    """Write collision-safe deterministic JSON and a concise Markdown summary."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_bytes(canonical_bytes(result))
    metrics = result.get("metrics", {})
    lines = [
        "# Evaluation result",
        "",
        f"- Result family: `{result['result_family']}`",
        f"- Metric scale: `{result.get('metric_scale', 'n/a')}`",
    ]
    if isinstance(metrics, dict):
        lines.extend(f"- {name}: {metrics[name]}" for name in sorted(metrics))
    fixtures = result.get("synthetic_fixture_metrics")
    if isinstance(fixtures, dict):
        lines.extend(("", "## Synthetic fixtures"))
        for name in sorted(fixtures):
            lines.extend(("", f"### `{name}`"))
            fixture = fixtures[name]
            if isinstance(fixture, dict):
                for key in sorted(fixture):
                    lines.append(f"- {key}: `{json.dumps(fixture[key], sort_keys=True)}`")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
