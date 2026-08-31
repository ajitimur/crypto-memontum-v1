"""Daily marking, the Liquidation trigger, and the exit for an asset that halts.

ADR-0001: the simulator's native granularity is the daily mark, and a weekly
rebalance is a decision-frequency choice layered on top of it. Two things live
inside the holding period that a boundary-to-boundary return cannot show:

- A **Liquidation** — cumulative loss breaching 100%. It is terminal. Positions
  close, the series ends at the breach, and no later gain restores it.
- A **Halt** — a bar the asset could not have been traded on. The position exits
  at its last tradeable price, not at one nobody could have sold into.

Under v1's long-only unlevered spot (ADR-0004) the Liquidation trigger is inert:
a spot holding cannot lose more than it cost. It is built and tested anyway,
because it goes live the moment leverage or a short leg appears and retrofitting
path accounting into working code is worse than carrying it unused.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Equity at or below this is a breach of a 100% cumulative loss. Exactly zero
# counts: there is nothing left to earn a return on, so the series is over.
LIQUIDATION_EQUITY = 0.0


class NothingToMark(Exception):
    """A holding period with no marks in it. There is no path to account for."""


@dataclass(frozen=True)
class MarkedPath:
    """A daily-marked net equity curve and how it ended.

    `equity_net` starts at the first mark and is already net of the entry cost;
    equity is 1.0 at the Decision Bar by construction. When it was liquidated the
    curve ends at `liquidation_ts_utc` with a final value of exactly 0.0 and no
    exit cost, because a wiped-out position is not sold — it is gone.
    """

    equity_net: pd.Series
    liquidation_ts_utc: pd.Timestamp | None

    @property
    def liquidated(self) -> bool:
        """Whether the path ended in a Liquidation. One fact, one place it lives."""
        return self.liquidation_ts_utc is not None


def mark_daily(
    mark_returns: pd.Series, *, entry_cost: float = 0.0, exit_cost: float = 0.0
) -> MarkedPath:
    """Compound `mark_returns` into a net equity curve, ending it at a breach.

    `mark_returns` is one simple return per day held, indexed on the bar's UTC
    open time; the first is measured from the fill price. Costs are fractions,
    not basis points, and are charged inside the path: the entry cost at the
    fill, so every subsequent mark is already net of it, and the exit cost on
    the final mark, where the position is actually closed.
    """
    if len(mark_returns) == 0:
        raise NothingToMark("a holding period needs at least one mark to account for")

    equity = (1.0 - entry_cost) * (1.0 + mark_returns.astype(float)).cumprod()
    equity.name = "equity_net"

    breached = equity <= LIQUIDATION_EQUITY
    if breached.any():
        # Compounding through a breach is meaningless — the position closed at
        # zero — so the curve is cut there rather than continued.
        breach_ts = breached.idxmax()
        equity = equity.loc[:breach_ts].copy()
        equity.iloc[-1] = 0.0
        return MarkedPath(equity_net=equity, liquidation_ts_utc=breach_ts)

    equity = equity.copy()
    equity.iloc[-1] = equity.iloc[-1] * (1.0 - exit_cost)
    return MarkedPath(equity_net=equity, liquidation_ts_utc=None)


def halted_at(bars: pd.DataFrame) -> pd.Timestamp | None:
    """The first bar `bars`' asset could not have been traded on, or `None`.

    A bar with no volume or no price is a bar no order could have filled at: the
    venue pulled the pair, the asset unwound, or trading simply ceased. The exit
    is the last tradeable price before the Halt, so a resumption afterwards is
    deliberately ignored — waiting for one means knowing in advance it comes.

    A price of zero printed on real volume is not a Halt. It is the asset trading
    at nothing, which is the one path by which v1's long-only spot holding reaches
    a 100% loss; it is left in the series so the mark can liquidate on it rather
    than being booked as an exit at the last price that happened to be positive.
    """
    close = bars["close"].astype(float)
    tradeable = close.notna()
    if "volume" in bars.columns:
        volume = bars["volume"].astype(float)
        tradeable &= volume.notna() & (volume > 0.0)
    if tradeable.all():
        return None
    return (~tradeable).idxmax()
