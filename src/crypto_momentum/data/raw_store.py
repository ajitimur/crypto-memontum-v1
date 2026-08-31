"""`data/raw/` — exactly as fetched from the venue, append-only.

Re-fetching a window that is already stored is a bug report, not an overwrite
(`docs/agents/quant-research.md`). Two things enforce that: the store refuses to
write over an existing file, and every stored file is left read-only.

Each file gets a JSON manifest sidecar recording the five things the protocol
asks us to record per raw source — venue, symbol convention, bar close
convention, timezone, and the exact window fetched — plus the verified digest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from crypto_momentum.data.binance_archive import ArchiveFile

VENUE = "binance-spot"
SYMBOL_CONVENTION = "base+quote concatenated, uppercase"
BAR_CLOSE_CONVENTION = "index is bar open_time; bar covers open_time to next open"
TIMEZONE = "UTC"

_PARTITION = "binance/spot/monthly/klines"
_READ_ONLY = 0o444


class RawWindowAlreadyStored(Exception):
    """A window already in `data/raw/` was fetched again. Raw data is append-only."""


class RawWindowMissing(Exception):
    """A window was read before it was fetched."""


class RawStore:
    """The append-only raw archive on disk."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, archive_file: ArchiveFile) -> Path:
        return (
            self.root
            / _PARTITION
            / archive_file.symbol
            / archive_file.interval
            / archive_file.filename
        )

    def has(self, archive_file: ArchiveFile) -> bool:
        return self.path_for(archive_file).exists()

    def write(
        self,
        archive_file: ArchiveFile,
        payload: bytes,
        *,
        sha256: str,
        fetched_at_utc: str,
    ) -> Path:
        """Store `payload` for `archive_file`. Raises if the window is already stored.

        `fetched_at_utc` is passed in rather than read from the clock, so a
        caller can make a stored manifest reproducible.
        """
        path = self.path_for(archive_file)
        if path.exists():
            raise RawWindowAlreadyStored(
                f"{archive_file.filename} is already in {path.parent}. "
                "Raw data is append-only; delete it deliberately or investigate "
                "why the window was fetched twice."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self._manifest_path(path).write_text(
            json.dumps(
                {
                    "venue": VENUE,
                    "symbol": archive_file.symbol,
                    "symbol_convention": SYMBOL_CONVENTION,
                    "interval": archive_file.interval,
                    "month": archive_file.month,
                    "timezone": TIMEZONE,
                    "bar_close_convention": BAR_CLOSE_CONVENTION,
                    "url": archive_file.url,
                    "sha256": sha256,
                    "fetched_at_utc": fetched_at_utc,
                    "bytes": len(payload),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.chmod(path, _READ_ONLY)
        os.chmod(self._manifest_path(path), _READ_ONLY)
        return path

    def read(self, archive_file: ArchiveFile) -> bytes:
        path = self.path_for(archive_file)
        if not path.exists():
            raise RawWindowMissing(
                f"{archive_file.filename} has not been fetched into {path.parent}"
            )
        return path.read_bytes()

    def manifest(self, archive_file: ArchiveFile) -> dict[str, Any]:
        path = self._manifest_path(self.path_for(archive_file))
        if not path.exists():
            raise RawWindowMissing(f"no manifest for {archive_file.filename}")
        return json.loads(path.read_text())

    @staticmethod
    def _manifest_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".manifest.json")
