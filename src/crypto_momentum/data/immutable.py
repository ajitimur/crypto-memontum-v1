"""The mechanism `data/raw/` is built on: write once, then never again.

`docs/agents/quant-research.md` makes the raw layer append-only. Two things
enforce it — the store refuses to write over an existing file, and every stored
file is left read-only — and both live here so the Binance archive and the
CoinMarketCap panel cannot drift apart on the discipline they share.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_READ_ONLY = 0o444


class RawArtifactAlreadyStored(Exception):
    """Something already in `data/raw/` was fetched again. Raw data is append-only."""


class RawArtifactMissing(Exception):
    """Raw data was read before it was fetched."""


def manifest_path(path: Path) -> Path:
    """The JSON sidecar recording where `path` came from."""
    return path.with_suffix(path.suffix + ".manifest.json")


def write_immutable(path: Path, payload: bytes, manifest: dict[str, Any]) -> Path:
    """Store `payload` at `path` with its manifest sidecar, both read-only.

    Raises `RawArtifactAlreadyStored` rather than overwriting. Callers that can
    say something more specific about the collision should check first and raise
    their own error; this is the backstop underneath them.
    """
    if path.exists():
        raise RawArtifactAlreadyStored(
            f"{path} is already stored. Raw data is append-only; delete it "
            "deliberately or investigate why it was fetched twice."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    sidecar = manifest_path(path)
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(path, _READ_ONLY)
    os.chmod(sidecar, _READ_ONLY)
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the manifest sidecar beside `path`."""
    sidecar = manifest_path(path)
    if not sidecar.exists():
        raise RawArtifactMissing(f"no manifest beside {path}")
    return json.loads(sidecar.read_text())


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
