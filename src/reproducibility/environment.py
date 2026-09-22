"""Centralized, machine-readable environment capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


class EnvironmentOutputError(ValueError):
    """Raised when environment metadata would be written into protected raw data."""


CORE_PACKAGES = (
    "Pillow",
    "PyYAML",
    "matplotlib",
    "numpy",
    "opencv-python",
    "pandas",
    "psutil",
    "pytest",
    "ruff",
    "torch",
    "torchvision",
    "tqdm",
)


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return resolved.relative_to(repo_root).as_posix()
    return str(resolved)


def hash_configs(config_paths: Sequence[str | Path], repo_root: str | Path) -> str:
    """Hash config names and bytes in deterministic path order."""

    repo = Path(repo_root).resolve()
    normalized = sorted(
        ((Path(path).resolve(), _display_path(Path(path), repo)) for path in config_paths),
        key=lambda item: item[1],
    )
    digest = hashlib.sha256()
    for path, display in normalized:
        digest.update(display.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in CORE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _accelerator_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_available": False,
        "cuda_runtime": None,
        "cuda_toolkit": None,
        "cudnn_version": None,
        "gpus": [],
        "nvidia_driver": None,
    }
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_runtime"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                result["gpus"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "vram_bytes": properties.total_memory,
                    }
                )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
            if rows:
                parsed = []
                drivers = set()
                for index, row in enumerate(rows):
                    name, memory_mib, driver = (item.strip() for item in row.split(",", maxsplit=2))
                    parsed.append(
                        {"index": index, "name": name, "vram_bytes": int(memory_mib) * 1024**2}
                    )
                    drivers.add(driver)
                if not result["gpus"]:
                    result["gpus"] = parsed
                result["nvidia_driver"] = sorted(drivers)
    nvcc = shutil.which("nvcc")
    if nvcc:
        completed = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            match = re.search(r"release\s+([0-9.]+)", completed.stdout)
            if match:
                result["cuda_toolkit"] = match.group(1)
    if result["cuda_toolkit"] is None:
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            result["cuda_toolkit"] = Path(cuda_path).name.removeprefix("v") or None
    return result


def capture_environment(
    repo_root: str | Path,
    *,
    config_paths: Sequence[str | Path],
    seed: int,
    command: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Capture the software, hardware, config, and Git state for a run."""

    repo = Path(repo_root).resolve()
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    timestamp = instant.isoformat().replace("+00:00", "Z")
    config_files = sorted(_display_path(Path(path), repo) for path in config_paths)
    config_hash = hash_configs(config_paths, repo)
    git = _git_metadata(repo)
    short_sha = git["commit"][:12]
    identifier_timestamp = instant.strftime("%Y%m%dT%H%M%SZ")

    return {
        "accelerator": _accelerator_metadata(),
        "command": list(command if command is not None else [sys.executable, *sys.argv]),
        "config_files": config_files,
        "config_hash_sha256": config_hash,
        "cpu": {
            "logical_cores": os.cpu_count(),
            "model": platform.processor() or None,
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "experiment_id": f"{identifier_timestamp}_{short_sha}_{config_hash[:12]}",
        "git": git,
        "packages": _package_versions(),
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "schema_version": 1,
        "seed": seed,
        "system": {
            "architecture": platform.machine(),
            "os": platform.platform(),
        },
        "timestamp_utc": timestamp,
    }


def write_environment(
    metadata: dict[str, Any],
    output_path: str | Path,
    *,
    protected_roots: Sequence[str | Path] = (),
    overwrite: bool = False,
) -> Path:
    """Atomically write environment metadata as sorted JSON."""

    destination = Path(output_path).resolve()
    for protected_root in protected_roots:
        resolved_root = Path(protected_root).resolve()
        if destination == resolved_root or destination.is_relative_to(resolved_root):
            raise EnvironmentOutputError(
                f"Environment destination is inside protected raw-data path: {destination}"
            )
    if destination.exists() and not overwrite:
        raise EnvironmentOutputError(
            f"Environment destination already exists: {destination}. Use --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            json.dump(metadata, target, indent=2, sort_keys=True, ensure_ascii=False)
            target.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination
