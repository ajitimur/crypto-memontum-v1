"""Prices and a point-in-time Universe from the CoinMarketCap panel.

The Faithful Run's data layer. ADR-0008 buys the panel for market caps *and* for
this: the Binance archive floor of 2017-08-17 sits inside Han, Kang and Ryu's
2017-01-01 sample, so an archive-priced run cannot cover their window and cannot
be the run that eliminates vendor differences as an explanation for disagreement.

Two things this module does that the archive path does not have to.

**It refuses a panel too coarse to mark daily.** ADR-0008's one-time pull runs at
`--interval 7d`, because weekly is the grain CoinMarketCap published across the
whole panel's life. A weekly panel cannot support ADR-0001's daily marking, and
it cannot represent a three-day holding period at all — the Grid has four cells
under a week. Rather than resample a weekly price into daily bars, which would
invent five prices out of every seven and hide a liquidation inside the week it
happened, `panel_bars` refuses and names what would have to change.

**It builds the Universe from the panel rather than from archive coverage.** An
asset is in the Universe on a snapshot when the panel prices it on that
snapshot, which is exact per date — the archive can only speak to the month and
be narrowed to real bars afterwards. `build_universe_panel`'s archive floor also
has no meaning here: the panel starts in 2013.

The panel is survivorship-free by construction (`cmc_panel.assert_survivorship_free`),
so a Universe built from it retains the assets that later died, which is the
whole reason ADR-0008 pays for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from crypto_momentum.data.binance_archive import BAR_COLUMNS
from crypto_momentum.data.cmc_panel import PANEL_START, SOURCE, VENDOR
from crypto_momentum.data.market_caps import base_asset_of
from crypto_momentum.data.symbol_map import SymbolMap, vendor_symbol_map
from crypto_momentum.data.universe import UniversePanel

# The panel's own floor: CoinMarketCap's first historical listings snapshot. It
# is below Han et al.'s 2017-01-01 start, which is the point of using it.
PANEL_FLOOR = pd.Timestamp(PANEL_START, tz="UTC")

# ADR-0001 marks every position daily, so the panel has to carry a price for
# every day. One day, exactly — not "at most one day", because a panel whose
# snapshots are irregular would mark some holding periods and not others.
REQUIRED_SNAPSHOT_SPACING = pd.Timedelta(days=1)


class PanelGrainTooCoarse(Exception):
    """The stored panel's snapshots are further apart than daily marking needs."""


class NoPanelPrices(Exception):
    """No requested symbol has a price in the panel over the window."""


def panel_bars(
    panel: pd.DataFrame,
    symbols: Iterable[str],
    *,
    repo_root: Path | str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    quote_asset: str = "USDT",
    symbol_map: SymbolMap | None = None,
) -> dict[str, pd.DataFrame]:
    """The panel's prices as one bar frame per venue symbol.

    The frames carry `binance_archive.BAR_COLUMNS`, so everything downstream of
    the data layer reads a Faithful Run exactly as it reads a Venue Run and no
    simulator code branches on where the prices came from.

    A snapshot is a price *as of* 00:00 UTC on its date and not a bar, so open,
    high, low and close are all that one price. That is a real narrowing —
    intraday range is not available from this vendor at this grain — and it
    means the Faithful Run's maximum drawdown is measured close-to-close. It is
    also what Han, Kang and Ryu had: their table is built from the same daily
    CoinMarketCap series.

    Volume is the panel's `volume_24h_usd`, already in dollars. The archive's
    volume column is in base units and `dollar_volume_from_bars` multiplies it
    by the close; a caller applying a liquidity floor to a Faithful Run would
    therefore be applying it to dollars times price. The floor is left to
    `universe.liquidity_floor_usd` and the Faithful Run configs do not set one —
    see `configs/gate-faithful.toml`.

    A symbol the panel never prices inside the window is left out of the result
    rather than carried as an empty column, matching `load_cross_section`.
    """
    requested = list(symbols)
    first = _as_utc(start)
    last = _as_utc(end)
    # Emptiness before grain, because they are different faults with different
    # answers: a window the panel never covered is not a panel pulled too
    # coarsely, and telling the caller to re-pull at a daily interval would send
    # them after the wrong thing.
    snapshots = pd.DatetimeIndex(sorted(set(panel.index)))
    if not len(snapshots[(snapshots >= first) & (snapshots <= last)]):
        raise NoPanelPrices(
            f"the {VENDOR} panel has no snapshot between {first.date()} and "
            f"{last.date()}; it runs {snapshots[0].date()} to {snapshots[-1].date()}"
        )
    assert_daily_grain(panel, start=first, end=last)

    priced = _priced_rows(
        panel, requested, repo_root=repo_root, quote_asset=quote_asset,
        symbol_map=symbol_map,
    )
    within = priced[(priced["ts_utc"] >= first) & (priced["ts_utc"] <= last)]
    if within.empty:
        raise NoPanelPrices(
            f"the {VENDOR} panel prices none of {', '.join(requested)} between "
            f"{first.date()} and {last.date()}"
        )

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in requested:
        rows = within[within["symbol"] == symbol]
        rows = rows[rows["price_usd"] > 0.0]
        if rows.empty:
            continue
        index = pd.DatetimeIndex(rows["ts_utc"], name="ts_utc")
        price = rows["price_usd"].to_numpy(dtype=float)
        bars = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": rows["volume_24h_usd"].to_numpy(dtype=float),
            },
            index=index,
        )
        bars_by_symbol[symbol] = bars.loc[:, BAR_COLUMNS].sort_index()
    if not bars_by_symbol:
        raise NoPanelPrices(
            f"every {VENDOR} price for {', '.join(requested)} between "
            f"{first.date()} and {last.date()} is zero or missing"
        )
    return bars_by_symbol


def assert_daily_grain(
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> None:
    """Raise unless the panel carries a snapshot every day across the window.

    The stored panel is ADR-0008's one-time pull, which runs at `--interval 7d`.
    A weekly panel is refused here rather than resampled: filling six days in
    seven would invent prices the vendor never published, and ADR-0001 exists
    precisely because what happens *inside* a holding period is the thing that
    liquidates a portfolio. Five of the Grid's 21 cells hold for a week or less
    and would be unrepresentable regardless.

    Judged over the run's own window and not over the whole panel. The panel
    reaches back to 2013 and its early years are thin — CoinMarketCap published
    less often then, and a dead asset contributes rows only up to the snapshot it
    died on — so a whole-panel check would refuse a panel that is daily
    everywhere a given run looks. A run is entitled to the grain of its own
    window and to nothing more.
    """
    snapshots = pd.DatetimeIndex(sorted(set(panel.index)))
    if start is not None:
        snapshots = snapshots[snapshots >= _as_utc(start)]
    if end is not None:
        snapshots = snapshots[snapshots <= _as_utc(end)]
    window = (
        ""
        if start is None and end is None
        else f" between {_as_utc(start).date()} and {_as_utc(end).date()}"
    )
    if len(snapshots) < 2:
        raise PanelGrainTooCoarse(
            f"the {VENDOR} panel has {len(snapshots)} snapshot(s){window}, which "
            "is not a series anything can be marked against"
        )
    coarsest = snapshots.to_series().diff().iloc[1:].max()
    if coarsest > REQUIRED_SNAPSHOT_SPACING:
        raise PanelGrainTooCoarse(
            f"the stored {VENDOR} panel steps {coarsest.days} days between "
            f"snapshots at its widest{window}, and the Faithful Run marks daily "
            "(ADR-0001). Nothing here resamples it: filling the days between "
            "would invent prices the vendor never published, and what happens "
            "inside a holding period is exactly what ADR-0001 is for. Re-pull "
            "the panel at a daily interval — `Rscript scripts/pull_cmc_panel.R "
            "--interval daily ...`, into a fresh raw path, per ADR-0008's "
            "one-pull discipline — or run the gate's Venue Run only."
        )


def panel_universe(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> UniversePanel:
    """The point-in-time Universe the panel itself states, date by date.

    An asset is tradeable on a date when the panel prices it on that date. No
    month-resolution coverage and no narrowing step, because the vendor's own
    grain is the grain of the answer — which is the one respect in which the
    Faithful Run's Universe is sharper than the Venue Run's.

    Built as a *point-in-time* Universe in CONTEXT.md's sense: the panel is
    survivorship-free, so an asset that stopped being listed simply stops being
    priced, and its earlier dates stay True.
    """
    dates = pd.date_range(_as_utc(start), _as_utc(end), freq="D", tz="UTC", name="ts_utc")
    symbols = list(bars_by_symbol)
    flags = {
        symbol: dates.isin(bars_by_symbol[symbol].index.normalize())
        for symbol in symbols
    }
    tradeable = pd.DataFrame(flags, index=dates, dtype=bool)
    tradeable.columns.name = "symbol"

    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        column = tradeable[symbol]
        flagged = column.index[column.to_numpy()]
        per_symbol[symbol] = {
            "first_tradeable_ts_utc": _iso(flagged[0]) if len(flagged) else None,
            "last_tradeable_ts_utc": _iso(flagged[-1]) if len(flagged) else None,
            "n_dates_priced": int(column.sum()),
        }

    last_priced = [
        entry["last_tradeable_ts_utc"]
        for entry in per_symbol.values()
        if entry["last_tradeable_ts_utc"] is not None
    ]
    frontier = max(last_priced) if last_priced else None
    metadata = {
        "source": f"{VENDOR} panel ({SOURCE})",
        "enumerated_from": (
            "the panel's own rows — survivorship-free by ADR-0008, so an asset "
            "that died is priced up to the snapshot it died on and no further"
        ),
        "interval": "1d",
        "panel_floor_ts_utc": _iso(PANEL_FLOOR),
        "start_ts_utc": _iso(dates[0]),
        "end_ts_utc": _iso(dates[-1]),
        "n_dates": len(dates),
        "n_dates_before_panel_floor": int((dates < PANEL_FLOOR).sum()),
        "n_symbols": len(symbols),
        "delisting_reference_ts_utc": frontier,
        "n_symbols_delisted_within_window": sum(
            1
            for entry in per_symbol.values()
            if entry["last_tradeable_ts_utc"] is not None
            and frontier is not None
            and entry["last_tradeable_ts_utc"] < frontier
        ),
        "coverage_resolution": "vendor snapshot, one per UTC date",
        "coverage_requires_published_checksum": False,
        "symbols": per_symbol,
    }
    return UniversePanel(tradeable=tradeable, metadata=metadata)


def _priced_rows(
    panel: pd.DataFrame,
    symbols: list[str],
    *,
    repo_root: Path | str,
    quote_asset: str,
    symbol_map: SymbolMap | None,
) -> pd.DataFrame:
    """The panel's rows tagged with the venue symbol each one was, on its date.

    The same dated join `market_caps.market_cap_panel` makes, and for the same
    reason: a CoinMarketCap id is permanent and a Binance base is a name the
    venue reuses, so LUNA in 2021 and LUNA in 2023 are two assets under one
    ticker and joining on the ticker would merge them.
    """
    bases = {symbol: base_asset_of(symbol, quote_asset=quote_asset) for symbol in symbols}
    if symbol_map is None:
        symbol_map = vendor_symbol_map(panel, bases.values(), repo_root=repo_root)
    base_to_symbol = {base: symbol for symbol, base in bases.items()}

    rows = panel.reset_index()[
        ["ts_utc", "cmc_id", "price_usd", "volume_24h_usd"]
    ].copy()
    rows["symbol"] = pd.Series(pd.NA, index=rows.index, dtype=object)
    for link in symbol_map.links:
        symbol = base_to_symbol.get(link.binance_base)
        if symbol is None:
            continue
        within = (rows["cmc_id"] == link.cmc_id) & (
            rows["ts_utc"] >= pd.Timestamp(link.valid_from, tz="UTC")
        )
        if link.valid_until is not None:
            within &= rows["ts_utc"] < pd.Timestamp(link.valid_until, tz="UTC")
        rows.loc[within, "symbol"] = symbol
    return rows.dropna(subset=["symbol", "price_usd"])


def _as_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp


def _iso(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
