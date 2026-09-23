"""Research-integrity checks used by M1 dataset verification."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitSafetyError(RuntimeError):
    """Raised when raw dataset content is not safely excluded from Git."""


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def verify_git_safety(repo_root: str | Path, dataset_root: str | Path) -> None:
    """Require local raw data to be ignored and absent from the Git index."""

    repo = Path(repo_root).resolve()
    root = Path(dataset_root).resolve()
    configured_raw_root = (repo / "data" / "ua_detrac").resolve()
    targets = [configured_raw_root]
    if root.is_relative_to(repo) and root not in targets:
        targets.append(root)

    for target in targets:
        relative = target.relative_to(repo).as_posix()
        ignore_target = target if target.exists() else target / ".git-safety-probe"
        ignore_relative = ignore_target.relative_to(repo).as_posix()
        ignored = _git(repo, "check-ignore", "--quiet", "--no-index", "--", ignore_relative)
        if ignored.returncode != 0:
            raise GitSafetyError(f"Raw dataset path is not ignored by Git: {relative}")
        tracked = _git(repo, "ls-files", "--cached", "--", relative)
        if tracked.returncode != 0:
            raise GitSafetyError(f"Unable to inspect Git index for raw dataset path: {relative}")
        tracked_paths = [line for line in tracked.stdout.splitlines() if line.strip()]
        if tracked_paths:
            raise GitSafetyError(
                "Raw dataset files are tracked or staged: " + ", ".join(tracked_paths[:5])
            )
