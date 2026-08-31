"""What code produced a result.

A result is only reproducible if we can say which commit produced it, so the
runner reads HEAD and records whether the working tree was clean at the time.
A dirty tree does not block a run — iterating would be miserable — but it is
recorded and it changes where the result is filed, so a dirty run can never be
mistaken for the commit it was based on.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class NotAGitRepository(Exception):
    """Results are keyed by commit hash, so a run needs a repository."""


@dataclass(frozen=True)
class Provenance:
    commit: str
    working_tree_dirty: bool


def describe_head(repo_root: Path | str) -> Provenance:
    """Read HEAD and the working tree state of the repository at `repo_root`."""
    repo_root = Path(repo_root)
    commit = _git(repo_root, "rev-parse", "HEAD")
    # --porcelain lists untracked files too: an untracked module can change a
    # result just as easily as an edited one.
    status = _git(repo_root, "status", "--porcelain")
    return Provenance(commit=commit, working_tree_dirty=bool(status))


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as error:
        raise NotAGitRepository(
            f"could not read git state at {repo_root}: {error}"
        ) from error
    return completed.stdout.strip()
