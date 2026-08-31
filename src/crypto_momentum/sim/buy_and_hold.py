"""Buy-and-hold on one symbol, marked daily, net of costs.

The walking skeleton's strategy. Depth arrives with the cross-sectional signal;
what this establishes is the shape every later strategy fills in: a signal formed
on a Decision Bar, a fill at the next bar's open, a daily mark through the hold,
and costs charged inside the path rather than deducted from the answer.

The mark itself, the Liquidation trigger and the exit for an asset that stops
trading live in `marking`; the reading of the resulting path lives in `report`.
Both are strategy-agnostic, and this module is what a strategy is: how positions
are formed, and nothing else.
"""

from __future__ import annotations

import pandas as pd

from crypto_momentum.sim.marking import halted_at, mark_daily
from crypto_momentum.sim.report import (
    HALTED,
    WINDOW_END,
    RunResult,
    summarise,
)

BPS = 1e-4


class NotEnoughBars(Exception):
    """A window with no tradeable bar after the Decision Bar has nowhere to fill."""


def simulate_buy_and_hold(bars: pd.DataFrame, *, cost_bps_per_side: float) -> RunResult:
    """Hold one unit of `bars`' symbol from the fill bar to the last bar's close.

    `bars` is a frame of daily bars: one row per UTC day, indexed on the bar's
    open time (`ts_utc`), with open/high/low/close/volume columns. The first row
    is the Decision Bar and is never traded on — the fill is at the *next* bar's
    open, so no information from the Decision Bar's own session is used.

    Every day held is marked, per ADR-0001, and the hold ends early if the asset
    halts: the exit is then its last tradeable price, not a later print
    nobody could have sold into.

    `cost_bps_per_side` is charged on the buy and again on the sell, per ADR-0007.
    """
    if len(bars) < 2:
        raise NotEnoughBars(
            f"need a Decision Bar plus at least one bar to fill on, got {len(bars)}"
        )

    held = _hold_before_halt(bars.iloc[1:])
    exit_reason = WINDOW_END if len(held) == len(bars) - 1 else HALTED
    cost = cost_bps_per_side * BPS
    entry_price = float(held["open"].iloc[0])
    if not entry_price > 0.0:
        raise NotEnoughBars(
            f"the fill bar at {held.index[0].date()} has no price to buy at "
            f"(open {entry_price})"
        )
    closes = held["close"].astype(float)

    # The first mark is measured from the fill price, every later one from the
    # previous close, so the path compounds day by day rather than jumping from
    # boundary to boundary.
    previous_close = closes.shift(1)
    previous_close.iloc[0] = entry_price
    # A close of zero is the position reaching nothing. The mark on it is -100%
    # and the series ends there, so it is never a base to divide by.
    mark_returns = closes / previous_close.where(previous_close > 0.0) - 1.0
    path = mark_daily(mark_returns, entry_cost=cost, exit_cost=cost)

    return summarise(
        path,
        decision_ts_utc=bars.index[0],
        entry_price=entry_price,
        exit_price=float(closes.loc[path.equity_net.index[-1]]),
        cost_bps_per_side=cost_bps_per_side,
        exit_reason=exit_reason,
    )


def _hold_before_halt(held: pd.DataFrame) -> pd.DataFrame:
    """The held bars up to, but not including, the bar the asset halted on."""
    halt_ts = halted_at(held)
    if halt_ts is None:
        return held
    tradeable = held.loc[held.index < halt_ts]
    if tradeable.empty:
        raise NotEnoughBars(
            f"the asset halted at {halt_ts.date()}, the first bar after the "
            "Decision Bar, so the position has no price to fill at"
        )
    return tradeable
