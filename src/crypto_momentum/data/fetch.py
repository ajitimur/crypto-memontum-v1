"""Downloading from the archive. The only module in the data layer that opens a socket.

`open_url` is injected so the adapter and every test above it run against
recorded fixture bytes.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Callable

from crypto_momentum.data.binance_archive import ArchiveFile, verify_sha256

UrlOpener = Callable[[str], bytes]

_TIMEOUT_SECONDS = 60


class ArchiveUnavailable(Exception):
    """The archive did not serve a file — a missing month, or a network failure."""


def fetch_archive_file(
    archive_file: ArchiveFile, *, open_url: UrlOpener | None = None
) -> tuple[bytes, str]:
    """Download a partition and its checksum, verify, and return `(payload, digest)`.

    A checksum mismatch raises rather than returning: a corrupted download must
    never reach `data/raw/`.
    """
    opener = open_url or download
    try:
        payload = opener(archive_file.url)
        checksum_text = opener(archive_file.checksum_url).decode("utf-8")
    except ArchiveUnavailable as error:
        raise ArchiveUnavailable(
            f"{archive_file.symbol} {archive_file.interval} {archive_file.month}: {error}"
        ) from error
    digest = verify_sha256(payload, checksum_text, archive_file.filename)
    return payload, digest


def download(url: str) -> bytes:
    """Fetch `url` over HTTPS. Any failure surfaces as `ArchiveUnavailable`."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as error:
        raise ArchiveUnavailable(f"could not fetch {url}: {error}") from error
