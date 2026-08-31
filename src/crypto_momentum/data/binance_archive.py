"""Adapter for the public `data.binance.vision` archive.

Everything here is a pure function of bytes, so it is tested against recorded
fixture files and never against the live archive. Downloading lives in
`crypto_momentum.data.fetch`.

Recorded source conventions (`docs/agents/quant-research.md` asks for these):

- **Venue**: Binance spot. Per ADR-0007 Tokocrypto's USDT book is this book, so
  this is also the venue we would trade.
- **Symbol convention**: concatenated base+quote, uppercase, no separator —
  `BTCUSDT`.
- **Bar close convention**: a kline's `open_time` opens the bar and its
  `close_time` is one time-unit before the next bar's open. We index on
  `open_time`, so a `1d` bar stamped `2021-01-01T00:00:00Z` covers that whole
  UTC day and is complete only after `23:59:59.999Z`.
- **Timezone**: UTC throughout. The archive carries no local time.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from typing import NamedTuple

import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# The 12 columns Binance publishes, in order. Named here so a silent reshuffle
# upstream shows up as a parse failure rather than as prices in a volume column.
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

# Binance stamped the archive in milliseconds until 2025 and in microseconds
# after. Anything at or above this many ticks since the epoch cannot be a
# millisecond timestamp within the archive's lifetime (it would be year 5138).
_MICROSECOND_THRESHOLD = 100_000_000_000_000


class ChecksumMismatch(Exception):
    """A downloaded file did not match its published SHA256. The run stops."""


class MalformedArchiveFile(Exception):
    """Bytes that are not a readable Binance kline zip."""


class ArchiveFile(NamedTuple):
    """One addressable monthly partition of the archive."""

    symbol: str
    interval: str
    month: str
    filename: str
    url: str
    checksum_url: str


def monthly_klines_file(symbol: str, interval: str, month: str) -> ArchiveFile:
    """Address the monthly kline partition for `symbol` at `interval` in `month`."""
    filename = f"{symbol}-{interval}-{month}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    return ArchiveFile(
        symbol=symbol,
        interval=interval,
        month=month,
        filename=filename,
        url=url,
        checksum_url=url + ".CHECKSUM",
    )


def verify_sha256(payload: bytes, checksum_text: str, filename: str) -> str:
    """Check `payload` against the archive's published `.CHECKSUM` text.

    Returns the verified digest. Raises `ChecksumMismatch` if the digest differs
    or if the checksum file names a different archive file — pairing a payload
    with the wrong checksum would otherwise pass silently.
    """
    expected_digest, named_file = _parse_checksum_line(checksum_text, filename)
    if named_file != filename:
        raise ChecksumMismatch(
            f"checksum file names {named_file!r}, not {filename!r}"
        )
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        raise ChecksumMismatch(
            f"{filename} failed SHA256 verification: "
            f"published {expected_digest}, downloaded {actual_digest}"
        )
    return actual_digest


def parse_klines_zip(payload: bytes) -> pd.DataFrame:
    """Parse a monthly kline zip into a frame of bars.

    One row is one bar. The index is `ts_utc`, the bar's UTC open time; columns
    are open, high, low, close and volume as floats.
    """
    csv_bytes = _single_member(payload)
    try:
        raw = pd.read_csv(
            io.BytesIO(csv_bytes),
            header=None,
            names=list(KLINE_COLUMNS),
            comment=None,
        )
    except (ValueError, pd.errors.ParserError) as error:
        raise MalformedArchiveFile(f"could not parse kline CSV: {error}") from error

    if raw.shape[1] != len(KLINE_COLUMNS):
        raise MalformedArchiveFile(
            f"expected {len(KLINE_COLUMNS)} kline columns, got {raw.shape[1]}"
        )
    # Files published from 2025 onward carry a header row; older ones do not.
    if raw["open_time"].iloc[0] == "open_time":
        raw = raw.iloc[1:]
    try:
        open_time = pd.to_numeric(raw["open_time"])
        bars = raw[BAR_COLUMNS].astype(float)
    except (ValueError, TypeError) as error:
        raise MalformedArchiveFile(f"kline columns are not numeric: {error}") from error

    bars.index = _to_utc(open_time)
    bars.index.name = "ts_utc"
    return bars.sort_index()


def _to_utc(open_time: pd.Series) -> pd.DatetimeIndex:
    unit = "us" if open_time.max() >= _MICROSECOND_THRESHOLD else "ms"
    return pd.DatetimeIndex(pd.to_datetime(open_time, unit=unit, utc=True))


def _single_member(payload: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise MalformedArchiveFile(
                    f"expected exactly one file in the zip, got {members}"
                )
            return archive.read(members[0])
    except zipfile.BadZipFile as error:
        raise MalformedArchiveFile(f"not a zip archive: {error}") from error


def _parse_checksum_line(checksum_text: str, filename: str) -> tuple[str, str]:
    parts = checksum_text.split()
    if len(parts) != 2:
        raise ChecksumMismatch(
            f"unreadable checksum file for {filename}: {checksum_text!r}"
        )
    digest, named_file = parts
    return digest.lower(), named_file
