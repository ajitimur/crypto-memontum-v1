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

Coverage comes from two partition kinds, because the archive publishes two. A
month that is over is republished as one monthly partition; the running month
exists only as one file per day. Reading the monthly partitions alone would end
every symbol's coverage at the last completed month, which marks every asset
still trading today as untradeable for weeks and reports it delisted — so
`coverage_for_symbol` reads the daily partitions for the tail as well, and that
end of a symbol's life is exact to the day.

The settled history is still month-resolution, so both ends of a symbol's life
inside it over-claim by up to a partial month: a symbol listed on the 11th has a
partition covering that whole month, as does one delisted on the 28th. Pass
`bar_span_by_symbol` — built with `bar_span_from_bars` once the months are
downloaded — to narrow a symbol to its real bars. Whether that happened is in the
panel metadata rather than assumed, because an un-narrowed panel can select an
asset a few days before it traded.

This panel is what *existed*, not what we would consider holding: Stablecoins,
Wrapped Assets and the liquidity floor are policy and belong to issue #12,
which layers them on top. A caller must not mistake this for a Universe that
has been through those exclusions.

Nothing here fetches a data file. Listing pages carry no published checksum —
they are bucket metadata, not data — but a month is only counted as coverage
when its `.zip` *and* its `.CHECKSUM` are both published, so a partition we
could never verify never enters the Universe. `fetch_covered_month` is the one
door to the bytes themselves, and it goes through the verifying fetch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from crypto_momentum.data.archive_listing import (
    DAILY_KLINES_PREFIX,
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
_DAY_FORMAT = "%Y-%m-%d"


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
    # The archive rolls a month up only once it is over, so the current month's
    # days exist only as daily partitions. Without them the newest weeks look
    # like a gap, and every still-trading asset reads as delisted at the tail.
    daily_only_dates: tuple[str, ...] = ()

    @property
    def first_month(self) -> str:
        return self.months[0]

    @property
    def last_month(self) -> str:
        return self.months[-1]

    @property
    def first_covered_date(self) -> pd.Timestamp:
        """The first date the opening partition can carry a bar for.

        Coverage only — the archive floor is applied by `build_universe_panel`,
        which owns it, so there is one place a floor can be changed.
        """
        return _month_start(self.first_month)

    @property
    def last_covered_date(self) -> pd.Timestamp:
        """The last date any partition can carry a bar for.

        Exact where it comes from a daily partition. Month-resolution otherwise:
        the real last bar may be earlier if the symbol was delisted mid-month.
        See `bar_span_by_symbol` on `build_universe_panel`.
        """
        if self.daily_only_dates:
            return _as_utc(self.daily_only_dates[-1])
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
    return SymbolCoverage(
        symbol=symbol,
        interval=interval,
        months=_verified_partitions(keys, partition),
    )


def daily_dates_from_keys(symbol: str, interval: str, keys: Iterable[str]) -> tuple[str, ...]:
    """Read exact dates out of a symbol's *daily* partition keys.

    Same rule as the monthly partitions: a day counts only when its zip and its
    published SHA256 are both there.
    """
    return _verified_partitions(
        keys,
        re.compile(
            rf"^(?:.*/)?{re.escape(symbol)}-{re.escape(interval)}-"
            rf"(\d{{4}}-\d{{2}}-\d{{2}})\.zip$"
        ),
    )


def coverage_for_symbol(
    symbol: str,
    interval: str,
    *,
    prefix: str = KLINES_PREFIX,
    daily_prefix: str = DAILY_KLINES_PREFIX,
    open_url: UrlOpener | None = None,
    max_keys: int = MAX_KEYS_PER_PAGE,
) -> SymbolCoverage:
    """List one symbol's partitions and read its coverage from them.

    Two requests. The monthly partitions carry the settled history; the days
    after the last rolled-up month are then read from the daily partitions,
    because the archive publishes the running month only that way. Without the
    second request a panel ending near today marks every still-trading asset
    untradeable at the tail and reports it delisted.
    """
    monthly = walk_listing(
        f"{prefix}{symbol}/{interval}/", open_url=open_url, max_keys=max_keys
    )
    coverage = coverage_from_keys(symbol, interval, monthly.keys)
    if not coverage.months:
        return coverage
    daily = walk_listing(
        f"{daily_prefix}{symbol}/{interval}/",
        # Exclusive lower bound: every day of the last rolled-up month sorts
        # below `-99`, and the first day of the next month sorts above it.
        start_after=(
            f"{daily_prefix}{symbol}/{interval}/"
            f"{symbol}-{interval}-{coverage.last_month}-99"
        ),
        open_url=open_url,
        max_keys=max_keys,
    )
    return replace(
        coverage,
        daily_only_dates=daily_dates_from_keys(symbol, interval, daily.keys),
    )


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
    days_of_date = pd.Index(dates.strftime(_DAY_FORMAT))
    above_floor = dates >= archive_floor
    spans = dict(bar_span_by_symbol or {})

    flags = {}
    per_symbol_metadata = {}
    last_flagged_by_symbol: dict[str, pd.Timestamp | None] = {}
    for coverage in coverages:
        tradeable = (
            months_of_date.isin(coverage.months)
            | days_of_date.isin(coverage.daily_only_dates)
        ) & above_floor
        span = spans.get(coverage.symbol)
        if span is not None:
            first_bar, last_bar = (_as_utc(edge) for edge in span)
            tradeable &= (dates >= first_bar.normalize()) & (dates <= last_bar.normalize())
        flags[coverage.symbol] = tradeable
        first_flagged, last_flagged = _flagged_edges(dates, tradeable)
        last_flagged_by_symbol[coverage.symbol] = last_flagged
        per_symbol_metadata[coverage.symbol] = {
            "first_month": coverage.first_month,
            "last_month": coverage.last_month,
            "n_months": len(coverage.months),
            # Read off the flags rather than recomputed alongside them, so the
            # metadata cannot drift from the panel it describes. Both are
            # clipped to the window, and are None for a symbol the window misses.
            "first_tradeable_ts_utc": _iso_timestamp(first_flagged),
            "last_tradeable_ts_utc": _iso_timestamp(last_flagged),
            "narrowed_to_bars": span is not None,
        }

    # A symbol counts as delisted only if it stopped before the archive itself
    # ran out. Measuring against the window end alone would call every live
    # asset delisted whenever the window reaches today, because today's bar is
    # not published until the day closes. Any symbol still running gives us that
    # frontier: its daily partitions end where the archive ends.
    archive_frontier = max(
        (
            coverage.last_covered_date
            for coverage in coverages
            if coverage.daily_only_dates
        ),
        default=dates[-1],
    )
    delisting_reference = min(dates[-1], archive_frontier)

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
        "archive_floor_ts_utc": _iso_timestamp(archive_floor),
        "start_ts_utc": _iso_timestamp(dates[0]),
        "end_ts_utc": _iso_timestamp(dates[-1]),
        "n_dates": len(dates),
        "n_dates_before_archive_floor": int((~above_floor).sum()),
        "n_symbols": len(coverages),
        "delisting_reference_ts_utc": _iso_timestamp(delisting_reference),
        "n_symbols_delisted_within_window": sum(
            1
            for last_flagged in last_flagged_by_symbol.values()
            if last_flagged is not None and last_flagged < delisting_reference
        ),
        "n_symbols_narrowed_to_bars": sum(
            1 for entry in per_symbol_metadata.values() if entry["narrowed_to_bars"]
        ),
        "coverage_resolution": (
            "monthly archive partition, narrowed to real bars where a bar span "
            "was supplied"
        ),
        "coverage_requires_published_checksum": True,
        "symbols": per_symbol_metadata,
    }
    return UniversePanel(tradeable=tradeable, metadata=metadata)


def bar_span_from_bars(bars: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The `(first, last)` bar timestamps of a built bar series.

    The bridge from derived bars back to the panel: what the listing can only
    say to the month, the bars say to the day.
    """
    if bars.empty:
        raise UniverseError("a bar span needs at least one bar")
    return bars.index[0], bars.index[-1]


def build_archive_universe(
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    interval: str = "1d",
    quote_asset: str | None = None,
    prefix: str = KLINES_PREFIX,
    search_prefix: str | None = None,
    open_url: UrlOpener | None = None,
    max_keys: int = MAX_KEYS_PER_PAGE,
    archive_floor: pd.Timestamp = ARCHIVE_FLOOR,
) -> UniversePanel:
    """Enumerate the archive and build the whole point-in-time Universe from it.

    The composed path: list every symbol the bucket has ever held, read each
    one's coverage from its own partitions, and turn the lot into a dated
    tradeable flag. A symbol the archive lists but publishes no verifiable
    partition for at `interval` is dropped rather than carried as an empty
    column — it never traded on this interval, so it was never tradeable.

    `prefix` is the klines directory each symbol hangs off. `search_prefix`
    narrows *which* symbols are enumerated (`.../klines/SR` finds only the SRM
    pairs) and defaults to enumerating the whole directory; the two are separate
    because one addresses a symbol and the other filters the list of them.

    One listing request per symbol, so this is a slow call against the live
    bucket and a cheap one against recorded pages.
    """
    coverages = []
    for symbol in symbols_in_archive(
        quote_asset=quote_asset,
        prefix=search_prefix or prefix,
        open_url=open_url,
        max_keys=max_keys,
    ):
        coverage = coverage_for_symbol(
            symbol, interval, prefix=prefix, open_url=open_url, max_keys=max_keys
        )
        if coverage.months:
            coverages.append(coverage)
    return build_universe_panel(
        coverages, start=start, end=end, archive_floor=archive_floor
    )


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


def _verified_partitions(keys: Iterable[str], partition: re.Pattern[str]) -> tuple[str, ...]:
    """The partition labels that have both a zip and a published checksum, sorted."""
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
    return tuple(sorted(zipped & checksummed))


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


def _flagged_edges(
    dates: pd.DatetimeIndex, tradeable: np.ndarray
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """The first and last date a symbol is flagged tradeable, or `(None, None)`."""
    flagged = np.flatnonzero(tradeable)
    if flagged.size == 0:
        return None, None
    return dates[flagged[0]], dates[flagged[-1]]


def _iso_timestamp(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
