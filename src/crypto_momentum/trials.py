"""The trials log: one JSON object per run, appended, never rewritten.

Git tracks this file. It is the count of configurations tried that
`docs/agents/quant-research.md` requires alongside every quoted result, so it is
only useful if it is complete — which means it is written by the runner rather
than by a person, and this module offers no way to remove or amend a line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRIALS_FILENAME = "trials.jsonl"


class UnrecordableTrial(Exception):
    """A trial could not be serialised. Nothing was written."""


def append_trial(path: Path | str, trial: dict[str, Any]) -> None:
    """Append one trial to the log, creating it if this is the first run.

    The line is serialised before the file is opened, so a trial carrying an
    unserialisable value fails without truncating or half-writing the log.
    """
    try:
        line = json.dumps(trial, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise UnrecordableTrial(f"trial is not JSON-serialisable: {error}") from error

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", encoding="utf-8") as log:
        log.write(line + "\n")


def read_trials(path: Path | str) -> list[dict[str, Any]]:
    """Every trial ever recorded, oldest first. Empty when the log does not exist."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as log:
        return [json.loads(line) for line in log if line.strip()]
