"""A marked path, read as the numbers `docs/agents/quant-research.md` asks for.

Strategy-agnostic on purpose: every strategy ends at a daily-marked equity curve,
and the reading of that curve — annualised return, vol, Sharpe, the profitability
bar of ADR-0002, drawdown, Cost Drag, and whether the run ended in a Liquidation —
is the same reading whatever formed the positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_momentum.sim.marking import MarkedPath

# Crypto trades every calendar day, so a year is 365 marks and not 252.
MARKS_PER_YEAR = 365

# An annualised volatility below this is floating-point noise on a flat path,
# not risk. Dividing by it would report a Sharpe in the trillions.
FLAT_PATH_VOL = 1e-12

# How a holding period ended.
WINDOW_END = "window_end"
STOPPED_TRADING = "stopped_trading"
LIQUIDATED = "liquidated"


@dataclass(frozen=True)
class RunResult:
    """One simulated run of one configuration.

    Both annualised figures extrapolate the window, so a short window produces
    large ones; `cost_drag_as_fraction_of_gross` is the scale-free reading and is
    the one `docs/agents/quant-research.md` puts a ceiling on.

    `equity_net` is the daily-marked net equity curve, indexed on `ts_utc` and
    starting at the fill bar; equity is 1.0 at the Decision Bar by construction.
    Every return here is net of `cost_bps_per_side` charged on both legs, except
    the two fields named `gross`.

    `exit_reason` says why the series ends where it does: the window ran out, the
    asset stopped trading, or the run was liquidated. On a liquidation the series
    ends at `liquidation_ts_utc` and nothing after it exists to report; the two
    `gross` fields then describe the *asset's* path, not a position that survived
    to earn it, and Cost Drag is `None` because the comparison it measures — the
    same run without costs — is a different run once the position is wiped out.
    """

    decision_ts_utc: pd.Timestamp
    entry_ts_utc: pd.Timestamp
    exit_ts_utc: pd.Timestamp
    exit_reason: str
    entry_price: float
    exit_price: float
    n_marks: int
    cost_bps_per_side: float
    equity_net: pd.Series
    liquidation_ts_utc: pd.Timestamp | None
    gross_return: float
    net_return: float
    ann_return_gross: float
    ann_return_net: float
    ann_vol_net: float
    sharpe_net: float | None
    mean_log_return_daily_net: float | None
    max_drawdown: float
    max_drawdown_peak_ts_utc: pd.Timestamp | None
    max_drawdown_trough_ts_utc: pd.Timestamp | None
    cost_drag_annualised: float | None
    cost_drag_as_fraction_of_gross: float | None

    @property
    def liquidated(self) -> bool:
        """Whether the run ended in a Liquidation."""
        return self.liquidation_ts_utc is not None

    @property
    def liquidation_dates(self) -> list[pd.Timestamp]:
        """The run's Liquidation events, as the reporting block wants them.

        A list rather than a flag because the block reports count and dates
        across a Grid of runs, and an empty list is the explicit "none".
        """
        if self.liquidation_ts_utc is None:
            return []
        return [self.liquidation_ts_utc]


def summarise(
    path: MarkedPath,
    *,
    decision_ts_utc: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    cost_bps_per_side: float,
    exit_reason: str = WINDOW_END,
) -> RunResult:
    """Read a marked path into the reporting block.

    `exit_reason` is why the *unliquidated* series ends; a liquidation overrides
    it, because that is the terminal fact about the run.
    """
    equity_net = path.equity_net
    gross_return = exit_price / entry_price - 1.0
    net_return = float(equity_net.iloc[-1]) - 1.0
    n_marks = len(equity_net)
    years = n_marks / MARKS_PER_YEAR

    daily_net = _daily_returns(equity_net)
    ann_vol_net = (
        float(daily_net.std(ddof=1) * math.sqrt(MARKS_PER_YEAR)) if n_marks > 1 else 0.0
    )
    ann_return_net = _annualise(net_return, years)
    ann_return_gross = _annualise(gross_return, years)
    peak_ts, trough_ts, max_drawdown = _max_drawdown(equity_net)
    cost_drag_annualised = (
        None if path.liquidated else ann_return_gross - ann_return_net
    )

    return RunResult(
        decision_ts_utc=decision_ts_utc,
        entry_ts_utc=equity_net.index[0],
        exit_ts_utc=equity_net.index[-1],
        exit_reason=LIQUIDATED if path.liquidated else exit_reason,
        liquidation_ts_utc=path.liquidation_ts_utc,
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
        mean_log_return_daily_net=_mean_log_return(daily_net),
        max_drawdown=max_drawdown,
        max_drawdown_peak_ts_utc=peak_ts,
        max_drawdown_trough_ts_utc=trough_ts,
        cost_drag_annualised=cost_drag_annualised,
        cost_drag_as_fraction_of_gross=_fraction_of_gross(
            cost_drag_annualised, ann_return_gross
        ),
    )


def _fraction_of_gross(
    cost_drag_annualised: float | None, ann_return_gross: float
) -> float | None:
    """Cost Drag as a share of gross annualised return.

    `docs/agents/quant-research.md` caps this at one third. `None` when gross
    return is not positive, because a share of a loss is not a meaningful ratio,
    and `None` when there is no Cost Drag to take a share of.
    """
    if cost_drag_annualised is None or ann_return_gross <= 0.0:
        return None
    return cost_drag_annualised / ann_return_gross


def _daily_returns(equity_net: pd.Series) -> pd.Series:
    """Simple daily returns, with equity 1.0 at the Decision Bar as the first base."""
    previous = equity_net.shift(1)
    previous.iloc[0] = 1.0
    return equity_net / previous - 1.0


def _mean_log_return(daily_net: pd.Series) -> float | None:
    """ADR-0002's profitability bar, per daily mark.

    `None` on a liquidated path: a mark that takes equity to zero has a log
    return of negative infinity, and an average that includes it is not a number
    the bar can be applied to. The liquidation itself is the reported result.
    """
    if (daily_net <= -1.0).any():
        return None
    return float(np.log1p(daily_net).mean())


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
