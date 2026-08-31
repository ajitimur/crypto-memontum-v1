"""Buy-and-hold on one symbol, marked daily, net of costs.

The walking skeleton's strategy. Depth arrives with the cross-sectional signal;
what this establishes is the shape every later strategy fills in: a signal formed
on a Decision Bar, a fill at the next bar's open, a daily mark through the hold,
and costs charged inside the path rather than deducted from the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Crypto trades every calendar day, so a year is 365 marks and not 252.
MARKS_PER_YEAR = 365

BPS = 1e-4

# An annualised volatility below this is floating-point noise on a flat path,
# not risk. Dividing by it would report a Sharpe in the trillions.
FLAT_PATH_VOL = 1e-12


class NotEnoughBars(Exception):
    """A window with no bar after the Decision Bar has nowhere to fill."""


@dataclass(frozen=True)
class RunResult:
    """One simulated run of one configuration.

    `equity_net` is the daily-marked net equity curve, indexed on `ts_utc` and
    starting at the fill bar; equity is 1.0 at the Decision Bar by construction.
    Every return here is net of `cost_bps_per_side` charged on both legs, except
    the two fields named `gross`.
    """

    decision_ts_utc: pd.Timestamp
    entry_ts_utc: pd.Timestamp
    exit_ts_utc: pd.Timestamp
    entry_price: float
    exit_price: float
    n_marks: int
    cost_bps_per_side: float
    equity_net: pd.Series
    gross_return: float
    net_return: float
    ann_return_gross: float
    ann_return_net: float
    ann_vol_net: float
    sharpe_net: float | None
    mean_log_return_daily_net: float
    max_drawdown: float
    max_drawdown_peak_ts_utc: pd.Timestamp | None
    max_drawdown_trough_ts_utc: pd.Timestamp | None
    cost_drag_annualised: float


def simulate_buy_and_hold(bars: pd.DataFrame, *, cost_bps_per_side: float) -> RunResult:
    """Hold one unit of `bars`' symbol from the fill bar to the last bar's close.

    `bars` is a frame of daily bars: one row per UTC day, indexed on the bar's
    open time (`ts_utc`), with open/high/low/close/volume columns. The first row
    is the Decision Bar and is never traded on — the fill is at the *next* bar's
    open, so no information from the Decision Bar's own session is used.

    `cost_bps_per_side` is charged on the buy and again on the sell, per ADR-0007.
    """
    if len(bars) < 2:
        raise NotEnoughBars(
            f"need a Decision Bar plus at least one bar to fill on, got {len(bars)}"
        )

    held = bars.iloc[1:]
    cost = cost_bps_per_side * BPS
    entry_price = float(bars["open"].iloc[1])
    exit_price = float(bars["close"].iloc[-1])

    # Equity is 1.0 at the Decision Bar. The buy-side cost is paid at the fill,
    # so every subsequent daily mark is already net of it; the sell-side cost
    # lands on the final mark, where the position is actually closed.
    price_relative = held["close"].astype(float) / entry_price
    equity_net = (1.0 - cost) * price_relative
    equity_net = equity_net.copy()
    equity_net.iloc[-1] = equity_net.iloc[-1] * (1.0 - cost)
    equity_net.name = "equity_net"

    gross_return = exit_price / entry_price - 1.0
    net_return = float(equity_net.iloc[-1]) - 1.0
    n_marks = len(equity_net)
    years = n_marks / MARKS_PER_YEAR

    daily_net = _daily_returns(equity_net)
    ann_vol_net = float(daily_net.std(ddof=1) * math.sqrt(MARKS_PER_YEAR)) if n_marks > 1 else 0.0
    ann_return_net = _annualise(net_return, years)
    ann_return_gross = _annualise(gross_return, years)
    peak_ts, trough_ts, max_drawdown = _max_drawdown(equity_net)

    return RunResult(
        decision_ts_utc=bars.index[0],
        entry_ts_utc=held.index[0],
        exit_ts_utc=held.index[-1],
        entry_price=entry_price,
        exit_price=exit_price,
        n_marks=n_marks,
        cost_bps_per_side=cost_bps_per_side,
        equity_net=equity_net,
        gross_return=gross_return,
        net_return=net_return,
        ann_return_gross=ann_return_gross,
        ann_return_net=ann_return_net,
        ann_vol_net=ann_vol_net,
        sharpe_net=_sharpe(daily_net, ann_vol_net),
        mean_log_return_daily_net=float(np.log1p(daily_net).mean()),
        max_drawdown=max_drawdown,
        max_drawdown_peak_ts_utc=peak_ts,
        max_drawdown_trough_ts_utc=trough_ts,
        cost_drag_annualised=ann_return_gross - ann_return_net,
    )


def _daily_returns(equity_net: pd.Series) -> pd.Series:
    """Simple daily returns, with equity 1.0 at the Decision Bar as the first base."""
    previous = equity_net.shift(1)
    previous.iloc[0] = 1.0
    return equity_net / previous - 1.0


def _annualise(total_return: float, years: float) -> float:
    """Geometric annualisation. A total loss annualises to -100%, not to a NaN."""
    growth = 1.0 + total_return
    if growth <= 0.0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def _sharpe(daily_net: pd.Series, ann_vol_net: float) -> float | None:
    """Annualised Sharpe at a zero risk-free rate.

    `None` when the path has no dispersion — a ratio with a vanishing denominator
    is not a large Sharpe, it is an undefined one, and reporting it as a number
    invites a comparison that means nothing.
    """
    if ann_vol_net < FLAT_PATH_VOL or not math.isfinite(ann_vol_net):
        return None
    return float(daily_net.mean() * MARKS_PER_YEAR / ann_vol_net)


def _max_drawdown(
    equity_net: pd.Series,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, float]:
    """Worst peak-to-trough fall in the daily-marked net curve, with its dates."""
    running_peak = equity_net.cummax()
    drawdown = equity_net / running_peak - 1.0
    trough_ts = drawdown.idxmin()
    worst = float(drawdown.min())
    if worst == 0.0:
        return None, None, 0.0
    peak_value = float(running_peak.loc[trough_ts])
    up_to_trough = equity_net.loc[:trough_ts]
    peak_ts = up_to_trough[up_to_trough == peak_value].index[0]
    return peak_ts, trough_ts, worst
