"""Market capitalisations, keyed by the venue symbol the simulator trades.

The CoinMarketCap panel is keyed by a permanent numeric id; the Universe is keyed
by a Binance pair. Value weighting needs the two joined, and the join is dated,
because a Binance base is a name the venue reuses while a CoinMarketCap id is
not — LUNA in 2021 and LUNA in 2023 are two different assets under one ticker.

`crypto_momentum.data.symbol_map` answers which id a base referred to on a date.
This module is the narrow thing built on top of it: the same panel, pivoted onto
venue symbols, so the strategy never has to know a vendor id exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from crypto_momentum.data.symbol_map import SymbolMap, vendor_symbol_map


class UnmappableSymbol(Exception):
    """A venue symbol whose base cannot be split off its quote asset."""


def base_asset_of(symbol: str, *, quote_asset: str) -> str:
    """The base half of a venue pair — `BTC` out of `BTCUSDT`.

    A pair that is not quoted in `quote_asset` is refused rather than guessed at:
    the Universe is built from one quote book, and a symbol from another book
    arriving here means two books have been mixed somewhere upstream.
    """
    if not symbol.endswith(quote_asset) or len(symbol) <= len(quote_asset):
        raise UnmappableSymbol(
            f"{symbol} is not a {quote_asset}-quoted pair, so it has no base asset "
            "to match against the vendor panel"
        )
    return symbol[: -len(quote_asset)]


def market_cap_panel(
    panel: pd.DataFrame,
    symbols: Iterable[str],
    *,
    repo_root: Path | str,
    quote_asset: str = "USDT",
    symbol_map: SymbolMap | None = None,
) -> pd.DataFrame:
    """The panel's capitalisations as one column per venue symbol.

    `panel` is a CoinMarketCap panel as `parse_panel_csv` returns it. The result
    is one row per snapshot date and one column per requested symbol, in the
    order requested, holding `market_cap_usd`.

    A symbol with no vendor match on a snapshot is NaN on that snapshot, never
    zero. Unweightable is not the same as worthless, and a zero would put a real
    asset in the portfolio at no weight rather than keeping it out of it.

    The mapping is applied per snapshot, at that snapshot's own date, so an id
    that changed ticker mid-panel contributes to whichever venue symbol it
    actually was at the time.
    """
    requested = list(symbols)
    bases = {symbol: base_asset_of(symbol, quote_asset=quote_asset) for symbol in requested}
    if symbol_map is None:
        symbol_map = vendor_symbol_map(panel, bases.values(), repo_root=repo_root)

    snapshots = pd.DatetimeIndex(sorted(set(panel.index)), name="ts_utc")
    base_to_symbol = {base: symbol for symbol, base in bases.items()}

    rows = panel.reset_index()[["ts_utc", "cmc_id", "market_cap_usd"]].copy()
    rows["symbol"] = pd.Series(pd.NA, index=rows.index, dtype=object)
    # One pass per link rather than per panel row: the panel is every asset on
    # every weekly snapshot since 2013, and a row-by-row join over it is minutes
    # of work to produce a frame of a few columns.
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

    matched = rows.dropna(subset=["symbol"])
    # `_assert_no_overlap` in the symbol map guarantees one link per id per date,
    # so no snapshot can carry two rows for one venue symbol and this pivot
    # cannot silently pick a winner between them.
    caps = matched.pivot(index="ts_utc", columns="symbol", values="market_cap_usd")
    caps = caps.reindex(index=snapshots, columns=requested).astype(float)
    caps.columns.name = "symbol"
    return caps
