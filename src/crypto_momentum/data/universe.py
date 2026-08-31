"""The point-in-time Universe, reconstructed from archive file coverage.

This is the module the survivorship invariant in `CLAUDE.md` lives or dies on.
Symbols are enumerated from the `data.binance.vision` bucket listing, so an
asset that traded in 2021 and delisted in 2022 is still here, tradeable on the
dates it traded and untradeable after. `SRMUSDT` is the standing witness: the
archive publishes it from 2020-08 to 2022-11 and the panel says so.

What is deliberately *not* used: `exchangeInfo`, and Binance's own
`fetch-all-trading-pairs.sh` which wraps it. Both enumerate what is listed
today, which is survivorship-biased and (per ADR-0008) documented nowhere in
Binance's README.

Coverage granularity is the monthly archive partition, because that is what the
listing publishes. A symbol delisted mid-month has a partial final partition, so
the panel would otherwise call it tradeable for a few days past its last bar;
`bar_span_by_symbol` narrows both ends to real bars once they have been built.

Nothing here fetches a data file. Listing pages carry no published checksum —
they are bucket metadata, not data — but a month is only counted as coverage
when its `.zip` *and* its `.CHECKSUM` are both published, so a partition we
could never verify never enters the Universe. `fetch_covered_month` is the one
door to the bytes themselves, and it goes through the verifying fetch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from crypto_momentum.data.archive_listing import (
    KLINES_PREFIX,
    LISTING_ENDPOINT,
    MAX_KEYS_PER_PAGE,
    walk_listing,
)
from crypto_momentum.data.binance_archive import monthly_klines_file
from crypto_momentum.data.fetch import UrlOpener, fetch_archive_file

# The first day `data.binance.vision` publishes anything. Earlier dates are not
# "nothing traded" — they are outside the archive, and the two must not be
# confused when a window starts before it (Han et al.'s sample starts
# 2017-01-01, five months before the archive does).
ARCHIVE_FLOOR = pd.Timestamp("2017-08-17T00:00:00Z")

CHECKSUM_SUFFIX = ".CHECKSUM"
_MONTH_FORMAT = "%Y-%m"


class SymbolNotCovered(Exception):
    """A month was requested that the archive does not publish for this symbol."""


class UniverseError(Exception):
    """A Universe could not be built from the coverage it was given."""


@dataclass(frozen=True)
class SymbolCoverage:
    """Which monthly partitions the archive publishes for one symbol.

    `months` is sorted `YYYY-MM`, and holds only months whose zip and published
    SHA256 are both present. It may have holes: a symbol halted for a month and
    relisted has two runs, and treating that as one span would claim it was
    tradeable while it was not.
    """

    symbol: str
    interval: str
    months: tuple[str, ...]

    @property
    def first_month(self) -> str:
        return self.months[0]

    @property
    def last_month(self) -> str:
        return self.months[-1]

    @property
    def first_covered_date(self) -> pd.Timestamp:
        """The earliest date this symbol can be tradeable on, floor included."""
        return max(_month_start(self.first_month), ARCHIVE_FLOOR)

    @property
    def last_covered_date(self) -> pd.Timestamp:
        """The last date the final partition can carry a bar for.

        Month-resolution: the real last bar may be earlier if the symbol was
        delisted mid-month. See `bar_span_by_symbol` on `build_universe_panel`.
        """
        return _month_end(self.last_month)

    def covers(self, month: str) -> bool:
        return month in self.months


@dataclass(frozen=True)
class UniversePanel:
    """Which assets were tradeable on which dates, as it stood at the time.

    One row of `tradeable` is one UTC date; one column is one venue symbol; the
    value is True when the archive publishes a verifiable bar-carrying partition
    covering that date for that symbol. `metadata` records what produced it —
    the archive floor above all, so a run cannot quietly mistake "before the
    archive" for "nothing traded".
    """

    tradeable: pd.DataFrame
    metadata: dict

    def tradeable_on(self, date: pd.Timestamp | str) -> tuple[str, ...]:
        """The Universe as of one rebalance date."""
        timestamp = pd.Timestamp(date)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        if timestamp not in self.tradeable.index:
            raise UniverseError(
                f"{timestamp.date()} is outside the panel, which runs "
                f"{self.tradeable.index[0].date()} to {self.tradeable.index[-1].date()}"
            )
        row = self.tradeable.loc[timestamp]
        return tuple(row.index[row.to_numpy()])


def symbols_in_archive(
    *,
    quote_asset: str | None = None,
    prefix: str = KLINES_PREFIX,
    open_url: UrlOpener | None = None,
    max_keys: int = MAX_KEYS_PER_PAGE,
) -> tuple[str, ...]:
    """Every symbol the archive has ever published, in bucket key order.

    Delisted symbols are included — that is the entire point. `quote_asset`
    keeps only pairs quoted in it, so a USDT-quoted Universe does not silently
    mix in the BTC and BNB books.
    """
    listing = walk_listing(prefix, open_url=open_url, max_keys=max_keys)
    symbols = tuple(_symbol_of(common_prefix) for common_prefix in listing.prefixes)
    if quote_asset is None:
        return symbols
    return tuple(
        symbol
        for symbol in symbols
        if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset)
    )


def coverage_from_keys(symbol: str, interval: str, keys: Iterable[str]) -> SymbolCoverage:
    """Read one symbol's coverage out of its listing keys.

    A month counts only when both `SYMBOL-INTERVAL-YYYY-MM.zip` and its
    `.CHECKSUM` are published: an unverifiable partition is not data we would
    ever load, so it is not coverage either.
    """
    partition = re.compile(
        rf"^(?:.*/)?{re.escape(symbol)}-{re.escape(interval)}-(\d{{4}}-\d{{2}})\.zip$"
    )
    zipped: set[str] = set()
    checksummed: set[str] = set()
    for key in keys:
        if key.endswith(CHECKSUM_SUFFIX):
            match = partition.match(key[: -len(CHECKSUM_SUFFIX)])
            if match:
                checksummed.add(match.group(1))
            continue
        match = partition.match(key)
        if match:
            zipped.add(match.group(1))
    return SymbolCoverage(
        symbol=symbol, interval=interval, months=tuple(sorted(zipped & checksummed))
    )


def coverage_for_symbol(
    symbol: str,
    interval: str,
    *,
    prefix: str = KLINES_PREFIX,
    open_url: UrlOpener | None = None,
    max_keys: int = MAX_KEYS_PER_PAGE,
) -> SymbolCoverage:
    """List one symbol's partitions and read its coverage from them."""
    listing = walk_listing(
        f"{prefix}{symbol}/{interval}/", open_url=open_url, max_keys=max_keys
    )
    return coverage_from_keys(symbol, interval, listing.keys)


def build_universe_panel(
    coverages: Iterable[SymbolCoverage],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    bar_span_by_symbol: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    archive_floor: pd.Timestamp = ARCHIVE_FLOOR,
) -> UniversePanel:
    """Turn per-symbol archive coverage into a dated tradeable flag per asset.

    The panel spans `start` to `end` as asked, even where that reaches below
    `archive_floor`: those dates are present and False, and the count of them is
    in the metadata, so a window starting before the archive is visible rather
    than silently trimmed.

    `bar_span_by_symbol` maps a symbol to the `(first, last)` UTC timestamps of
    its actual bars. Where given, it narrows that symbol to real bars — a symbol
    delisted mid-month is untradeable from its last bar onward rather than to
    the end of its final partition.
    """
    coverages = list(coverages)
    _reject_duplicates(coverages)
    intervals = {coverage.interval for coverage in coverages}
    if len(intervals) > 1:
        raise UniverseError(f"a panel takes one interval, got {sorted(intervals)}")

    dates = pd.date_range(
        _as_utc(start), _as_utc(end), freq="D", tz="UTC", name="ts_utc"
    )
    if len(dates) == 0:
        raise UniverseError(f"the window {start} to {end} contains no dates")
    months_of_date = pd.Index(dates.strftime(_MONTH_FORMAT))
    above_floor = dates >= archive_floor
    spans = dict(bar_span_by_symbol or {})

    flags = {}
    per_symbol_metadata = {}
    for coverage in coverages:
        tradeable = months_of_date.isin(coverage.months) & above_floor
        last_tradeable = coverage.last_covered_date
        span = spans.get(coverage.symbol)
        if span is not None:
            first_bar, last_bar = (_as_utc(edge) for edge in span)
            tradeable &= (dates >= first_bar.normalize()) & (dates <= last_bar.normalize())
            last_tradeable = min(last_tradeable, last_bar.normalize())
        flags[coverage.symbol] = tradeable
        per_symbol_metadata[coverage.symbol] = {
            "first_month": coverage.first_month,
            "last_month": coverage.last_month,
            "n_months": len(coverage.months),
            "first_tradeable_date_utc": _iso_date(
                max(coverage.first_covered_date, archive_floor)
            ),
            "last_tradeable_date_utc": _iso_date(last_tradeable),
            "narrowed_to_bars": span is not None,
        }

    tradeable = pd.DataFrame(flags, index=dates, dtype=bool)
    tradeable.columns.name = "symbol"
    metadata = {
        "source": "data.binance.vision bucket listing",
        "listing_endpoint": LISTING_ENDPOINT,
        "listing_prefix": KLINES_PREFIX,
        "enumerated_from": (
            "archive bucket listing — not exchangeInfo and not "
            "fetch-all-trading-pairs.sh, neither of which is survivorship-free "
            "(ADR-0008)"
        ),
        "interval": intervals.pop() if intervals else None,
        "archive_floor_utc": _iso_date(archive_floor),
        "start_utc": _iso_date(dates[0]),
        "end_utc": _iso_date(dates[-1]),
        "n_dates": len(dates),
        "n_dates_before_archive_floor": int((~above_floor).sum()),
        "n_symbols": len(coverages),
        "n_symbols_delisted_within_window": sum(
            1
            for entry in per_symbol_metadata.values()
            if entry["last_tradeable_date_utc"] < _iso_date(dates[-1])
        ),
        "coverage_resolution": (
            "monthly archive partition, narrowed to real bars where a bar span "
            "was supplied"
        ),
        "coverage_requires_published_checksum": True,
        "symbols": per_symbol_metadata,
    }
    return UniversePanel(tradeable=tradeable, metadata=metadata)


def fetch_covered_month(
    coverage: SymbolCoverage, month: str, *, open_url: UrlOpener | None = None
) -> tuple[bytes, str]:
    """Fetch one partition the coverage says exists, verified against its SHA256.

    Refusing an uncovered month up front turns a survivorship mistake — asking a
    delisted symbol for a month after it stopped trading — into an error here
    rather than a 404 halfway through a panel build.
    """
    if not coverage.covers(month):
        raise SymbolNotCovered(
            f"the archive publishes no {coverage.interval} partition for "
            f"{coverage.symbol} in {month}; coverage runs {coverage.first_month} "
            f"to {coverage.last_month}"
        )
    return fetch_archive_file(
        monthly_klines_file(coverage.symbol, coverage.interval, month),
        open_url=open_url,
    )


def _reject_duplicates(coverages: list[SymbolCoverage]) -> None:
    seen: set[str] = set()
    for coverage in coverages:
        if not coverage.months:
            raise UniverseError(
                f"{coverage.symbol} has no covered months, so it cannot be in a Universe"
            )
        if coverage.symbol in seen:
            raise UniverseError(f"{coverage.symbol} appears twice in the coverage given")
        seen.add(coverage.symbol)


def _symbol_of(common_prefix: str) -> str:
    return common_prefix.rstrip("/").rsplit("/", 1)[-1]


def _month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01T00:00:00Z")


def _month_end(month: str) -> pd.Timestamp:
    return _month_start(month) + pd.offsets.MonthEnd(0)


def _as_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _iso_date(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
