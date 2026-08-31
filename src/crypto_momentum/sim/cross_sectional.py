"""Cross-sectional momentum: rank the Universe, hold the top quintile by value.

The first strategy with any depth in it, and the point where the three layers
meet: the data adapter's bars, the Universe policy's dated tradeable panel, and
the daily mark. What this module owns is only how positions are formed — the
mark, the Liquidation trigger and the halt exit live in `marking`, and the
reading of the resulting path lives in `report`.

The mechanics, and why each one is what it is:

- **Signal.** The cross-section's past return over a configurable lookback,
  measured from the bar before the Decision Bar. Nothing timestamped at or after
  the Decision Bar is read, which is the invariant `past_return` exists to make
  testable on a frame whose last row *is* the Decision Bar.
- **Selection.** The top quintile of the assets eligible on that date. Rounded
  up, so a quintile is never empty; refused below `MIN_UNIVERSE` names, because
  a fifth of four assets is one asset and calling that a cross-section is a
  fiction the number would not survive.
- **Weighting.** Value weights from the CoinMarketCap panel, read from the last
  snapshot strictly before the Decision Bar. A stale snapshot is not a weight:
  past `DEFAULT_CAP_STALENESS_DAYS` the asset is unweightable and drops out.
- **Rebalance.** Every `holding_days`, weekly by default, per the paper's
  convention. One portfolio, not the paper's five overlapping sub-portfolios:
  per the issue, the distributed scheme is a deployment device for spreading
  execution, and replicating it here would change what is being measured.
- **Fill.** At the next bar's open, never on the Decision Bar's own session.
- **Long-only, unlevered, no Trend Gate.** ADR-0004. Weights are non-negative
  and sum to at most one; what is not invested is cash, earning nothing. There
  is deliberately no risk overlay: one the paper did not have would invalidate
  the comparison the Replication Gate is for.

Costs are charged inside the path — on both legs of every rebalance, per
ADR-0007 — and the same walk is run a second time with costs switched off. That
costless walk is what the reporting block calls gross, so Cost Drag is the
measured gap between two identical runs rather than a haircut applied afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from crypto_momentum.costs import BPS, TURNOVER_CEILING_WEEKLY, weekly_turnover
from crypto_momentum.sim.marking import MarkedPath, is_tradeable_bar, mark_daily
from crypto_momentum.sim.report import WINDOW_END, RunResult, summarise

# A fifth. The quintile the literature ranks on, and the default the Grid varies
# nothing else against.
DEFAULT_QUANTILE = 0.2

# Weekly, per the paper's rebalance convention.
DEFAULT_HOLDING_DAYS = 7

# Below five names a quintile is one name, and a one-name "cross-section" is a
# single bet wearing a portfolio's clothes. Such a date holds cash instead.
MIN_UNIVERSE = 5

# The CoinMarketCap panel is a weekly snapshot, so two snapshots is the widest
# gap a live asset should ever show. Beyond it the asset is not being tracked and
# its last known capitalisation is not a weight we would size a position on.
DEFAULT_CAP_STALENESS_DAYS = 14


class NotEnoughHistory(Exception):
    """The window cannot support one formation and one holding period."""


class SelectionError(Exception):
    """A cross-section could not be ranked or weighted. Nothing is guessed at."""


class TurnoverBudgetBreached(Exception):
    """The walk traded more than the config budgeted, so it produces no result.

    The run-time half of ADR-0007's ceiling. The config loader refuses a budget
    above 25% weekly before the run starts; this refuses a run that turns out to
    breach the budget it declared. It carries the realised figure, because a
    researcher's next move is to widen the holding period or add a no-trade band
    until the number fits, and they need to know by how much it missed.
    """

    def __init__(self, realised: float, budget: float, *, holding_days: int) -> None:
        self.realised_weekly_turnover = realised
        self.budget = budget
        super().__init__(
            f"the walk turned over {realised:.1%} of the book a week against a "
            f"budget of {budget:.1%} (measured over a {holding_days}-day holding "
            "period and put on a weekly footing). ADR-0007 budgets turnover "
            "rather than observing it, so this run is refused rather than filed "
            "as a result that breaches the ceiling."
        )


@dataclass(frozen=True)
class Selection:
    """What one rebalance decided, and what it cost to get there.

    `weights` are the *target* weights formed on `decision_ts_utc`; what was
    actually filled is `symbols` minus `unfilled`, because an asset selected on
    the Decision Bar can still be untradeable on the bar the order would fill on,
    and that is an execution failure rather than something to see coming.

    `turnover` is one-way Rebalance Turnover — the share of the portfolio bought
    at this rebalance — measured against the drifted weights it replaced. It is
    the figure ADR-0007 puts a 25% weekly ceiling on.
    """

    decision_ts_utc: pd.Timestamp
    entry_ts_utc: pd.Timestamp
    symbols: tuple[str, ...]
    weights: dict[str, float]
    n_eligible: int
    unfilled: tuple[str, ...]
    turnover: float

    @property
    def held_cash(self) -> bool:
        """Whether this rebalance took no position at all."""
        return not self.symbols


@dataclass(frozen=True)
class HaltExit:
    """A position closed mid-holding-period because its asset stopped trading.

    `exit_ts_utc` is the bar the halt was visible on; `exit_price` is the last
    close before it, which is the last price an order could have been filled at.
    """

    symbol: str
    exit_ts_utc: pd.Timestamp
    exit_price: float


@dataclass(frozen=True)
class CrossSectionalRun:
    """One cross-sectional run: the marked path, and what formed it.

    `result` is the same `RunResult` every strategy in this repo ends at. The
    rest is what a cross-sectional run has to say that a single hold does not —
    what was held, how often it changed, and what changed hands to do it.
    """

    result: RunResult
    selections: tuple[Selection, ...]
    halt_exits: tuple[HaltExit, ...]
    exposure_gross: pd.Series
    lookback_days: int
    holding_days: int
    quantile: float
    min_universe: int
    max_cap_staleness_days: int
    max_weekly_rebalance_turnover: float

    @property
    def n_rebalances(self) -> int:
        return len(self.selections)

    @property
    def n_rebalances_held_cash(self) -> int:
        return sum(1 for selection in self.selections if selection.held_cash)

    @property
    def rebalance_turnovers(self) -> tuple[float, ...]:
        """One-way turnover at each rebalance that *replaced* a book.

        The first fill is excluded. It buys the opening position from cash, so it
        turns over 100% by construction whatever the signal did, and it happens
        once however long the run is. Averaging it in would make a short window
        look like it churned more than a long one running the identical strategy,
        and would put every configuration over the ceiling on its first week.

        Its cost is charged all the same — the walk pays for that first buy like
        any other. What is excluded here is only its contribution to the ongoing
        rate the ceiling budgets.
        """
        return tuple(selection.turnover for selection in self.selections[1:])

    @property
    def mean_rebalance_turnover(self) -> float:
        return _mean(self.rebalance_turnovers)

    @property
    def max_rebalance_turnover(self) -> float:
        return max(self.rebalance_turnovers, default=0.0)

    @property
    def weekly_rebalance_turnover(self) -> float:
        """Mean Rebalance Turnover on the weekly footing ADR-0007's ceiling uses.

        A per-rebalance figure is not comparable to a weekly ceiling unless the
        rebalance happens to be weekly, and the whole point of the ceiling is to
        make longer holding periods look as attractive as they are.
        """
        return weekly_turnover(
            self.mean_rebalance_turnover, holding_days=self.holding_days
        )

    @property
    def mean_n_positions(self) -> float:
        return _mean(len(selection.symbols) for selection in self.selections)

    @property
    def mean_gross_exposure(self) -> float:
        return float(self.exposure_gross.mean()) if len(self.exposure_gross) else 0.0

    @property
    def max_gross_exposure(self) -> float:
        return float(self.exposure_gross.max()) if len(self.exposure_gross) else 0.0

    def to_metadata(self) -> dict[str, Any]:
        """What the result file records about how the positions were formed.

        Net and gross exposure are the same number under ADR-0004: long-only
        unlevered spot has no short leg for the two to differ over. Both are
        reported anyway, so the day a short leg appears the reporting block does
        not silently start comparing two different quantities.
        """
        return {
            "strategy": "cross_sectional",
            "lookback_days": self.lookback_days,
            "holding_days": self.holding_days,
            "quantile": self.quantile,
            # Why a date can hold cash, recorded alongside how often one did —
            # otherwise a run of cash weeks is a result nobody can account for.
            "min_universe": self.min_universe,
            "max_cap_staleness_days": self.max_cap_staleness_days,
            "long_only": True,
            "levered": False,
            "trend_gate": False,
            "n_rebalances": self.n_rebalances,
            "n_rebalances_held_cash": self.n_rebalances_held_cash,
            "mean_n_positions": self.mean_n_positions,
            "mean_rebalance_turnover": self.mean_rebalance_turnover,
            "max_rebalance_turnover": self.max_rebalance_turnover,
            # The three together are the reporting protocol's "Rebalance
            # Turnover, against the ceiling": what was traded, on the footing the
            # ceiling is stated on, and what it was held to.
            "weekly_rebalance_turnover": self.weekly_rebalance_turnover,
            "max_weekly_rebalance_turnover": self.max_weekly_rebalance_turnover,
            "turnover_ceiling_weekly": TURNOVER_CEILING_WEEKLY,
            "mean_gross_exposure": self.mean_gross_exposure,
            "mean_net_exposure": self.mean_gross_exposure,
            "max_gross_exposure": self.max_gross_exposure,
            # `entry_price` and `exit_price` on the result are equity levels for a
            # portfolio, not prices: 1.0 at the Decision Bar, and where the same
            # walk without costs ended up.
            "equity_basis": (
                "entry_price and exit_price are portfolio equity levels, not "
                "asset prices"
            ),
            "n_halt_exits": len(self.halt_exits),
            "halt_exits": [
                {
                    "symbol": exit.symbol,
                    "exit_ts_utc": exit.exit_ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "exit_price": exit.exit_price,
                }
                for exit in self.halt_exits
            ],
        }


def past_return(
    closes: pd.DataFrame, *, decision_ts: pd.Timestamp, lookback_days: int
) -> pd.Series:
    """The cross-section's return over `lookback_days`, formed on `decision_ts`.

    `closes` is one row per UTC date and one column per venue symbol. Only rows
    timestamped strictly before `decision_ts` are read: the return runs from the
    close `lookback_days` before the last such bar to that bar's own close. The
    Decision Bar's session is still open when the signal is formed, so nothing in
    it — not the close, not the volume — can enter here.

    A symbol without a price at both ends is NaN rather than zero: unranked is
    not the same as ranked last, and a zero would put it in the bottom quintile
    of a cross-section it was never in.
    """
    if lookback_days < 1:
        raise SelectionError(f"lookback_days must be at least 1, got {lookback_days}")
    before = closes.loc[closes.index < decision_ts]
    if before.empty:
        raise NotEnoughHistory(
            f"no bars before the Decision Bar at {pd.Timestamp(decision_ts).date()}"
        )
    formation_ts = before.index[-1]
    start_ts = formation_ts - pd.Timedelta(days=lookback_days)
    if start_ts not in before.index:
        raise NotEnoughHistory(
            f"a {lookback_days}-day lookback formed at "
            f"{pd.Timestamp(decision_ts).date()} reaches back to {start_ts.date()}, "
            f"which is before the frame starts at {before.index[0].date()}"
        )

    start = before.loc[start_ts].astype(float)
    end = before.loc[formation_ts].astype(float)
    signal = end / start.where(start > 0.0) - 1.0
    signal.name = "past_return"
    return signal


def select_top_quantile(signal: pd.Series, *, quantile: float = DEFAULT_QUANTILE) -> tuple[str, ...]:
    """The top `quantile` of the ranked cross-section, best first.

    Rounded up, so a cross-section that does not divide evenly still yields a
    quintile rather than one short of it. Unranked symbols — NaN — are not in the
    cross-section at all, so they neither appear in the result nor count towards
    its size.

    Ties break on the symbol. An arbitrary but fixed order is what makes a run
    reproducible; pandas' own ordering on a tie is not something to rely on.
    """
    if not 0.0 < quantile <= 1.0:
        raise SelectionError(f"quantile must be in (0, 1], got {quantile}")
    ranked = signal.dropna()
    if ranked.empty:
        return ()
    n_selected = math.ceil(len(ranked) * quantile)
    ordered = sorted(ranked.items(), key=lambda item: (-item[1], item[0]))
    return tuple(symbol for symbol, _ in ordered[:n_selected])


def value_weights(market_caps: pd.Series) -> pd.Series:
    """Market-cap shares of the selected assets, summing to one.

    Long-only by construction: a negative capitalisation is not a thing, and a
    cross-section with nothing to weight by is refused rather than divided by
    zero and reported as an equal-weighted portfolio nobody asked for.
    """
    caps = market_caps.astype(float)
    if (caps < 0.0).any():
        raise SelectionError("a market capitalisation cannot be negative")
    total = float(caps.sum())
    if not total > 0.0:
        raise SelectionError(
            "the selected assets have no market capitalisation to weight by"
        )
    weights = caps / total
    weights.name = "weight"
    return weights


def market_caps_before(
    market_caps: pd.DataFrame,
    *,
    decision_ts: pd.Timestamp,
    max_staleness_days: int = DEFAULT_CAP_STALENESS_DAYS,
) -> pd.Series:
    """Each asset's capitalisation as last published strictly before `decision_ts`.

    The panel is a weekly snapshot, so the value in force on a Wednesday was
    published on the preceding snapshot date and carried forward. A snapshot
    timestamped *at* the Decision Bar is not carried forward — it is timestamped
    at the bar we are forming on, and the invariant is strictly-before.

    Past `max_staleness_days` the last known value is not carried at all. An
    asset the vendor stopped tracking has no capitalisation we would size a
    position on, and a months-old figure quietly weighting a position is exactly
    the kind of error that shows up only as an unexplained edge.
    """
    decision_ts = pd.Timestamp(decision_ts)
    columns = market_caps.columns
    before = market_caps.loc[market_caps.index < decision_ts]
    if before.empty:
        return pd.Series(float("nan"), index=columns, name="market_cap_usd")

    carried = before.ffill().iloc[-1].astype(float)
    horizon = decision_ts - pd.Timedelta(days=max_staleness_days)
    for symbol in columns:
        last_seen = before[symbol].last_valid_index()
        if last_seen is None or last_seen < horizon:
            carried[symbol] = float("nan")
    carried.name = "market_cap_usd"
    return carried


def simulate_cross_sectional(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    tradeable: pd.DataFrame,
    market_caps: pd.DataFrame,
    lookback_days: int,
    cost_bps_per_side: float,
    holding_days: int = DEFAULT_HOLDING_DAYS,
    quantile: float = DEFAULT_QUANTILE,
    min_universe: int = MIN_UNIVERSE,
    max_cap_staleness_days: int = DEFAULT_CAP_STALENESS_DAYS,
    max_weekly_rebalance_turnover: float = TURNOVER_CEILING_WEEKLY,
) -> CrossSectionalRun:
    """Run the cross-sectional strategy over the window the bars span.

    `bars_by_symbol` maps a venue symbol to its daily bars, indexed on the bar's
    UTC open time with open/high/low/close/volume columns. `tradeable` is a
    Universe panel after policy — one row per UTC date, one column per symbol —
    and is a gate, never a source of prices: an asset it marks tradeable on a
    date its bars do not cover is still not held. `market_caps` is the
    CoinMarketCap panel's capitalisations, one row per snapshot and one column
    per venue symbol.

    The walk runs twice, identically, once with `cost_bps_per_side` charged on
    both legs of every trade and once with it switched off. The costless walk is
    the gross path the reporting block compares against.

    `max_weekly_rebalance_turnover` is the budget the run declared. What the walk
    actually trades is only knowable once it has been walked, so the budget is
    checked at the end and a breach raises `TurnoverBudgetBreached` instead of
    returning a run. ADR-0007 budgets turnover rather than observing it: a
    breaching configuration is not a result with a footnote, it is a
    configuration we do not trade.
    """
    if holding_days < 1:
        raise SelectionError(f"holding_days must be at least 1, got {holding_days}")
    if min_universe < 1:
        raise SelectionError(f"min_universe must be at least 1, got {min_universe}")
    # The budget must be a budget; whether it is a *permissible* budget is the
    # config loader's question, not this one. ADR-0007's 25% ceiling is a policy
    # about which configurations we are willing to run, and duplicating it here
    # would put the policy in two places and leave the core unable to walk the
    # high-turnover regime the ADR is an argument about.
    if max_weekly_rebalance_turnover <= 0.0:
        raise SelectionError(
            "max_weekly_rebalance_turnover must be above 0, got "
            f"{max_weekly_rebalance_turnover}"
        )

    prices = _AlignedPrices.of(bars_by_symbol)
    gate = _aligned_gate(tradeable, prices.index, prices.columns)
    decision_bars = _decision_bars(
        prices.index, lookback_days=lookback_days, holding_days=holding_days
    )

    plan = _plan_rebalances(
        prices,
        gate=gate,
        market_caps=market_caps,
        decision_bars=decision_bars,
        lookback_days=lookback_days,
        quantile=quantile,
        min_universe=min_universe,
        max_cap_staleness_days=max_cap_staleness_days,
    )
    charged = _walk(prices, plan, cost_per_side=cost_bps_per_side * BPS)
    gross = _walk(prices, plan, cost_per_side=0.0)

    path = _marked(charged.equity)
    result = summarise(
        path,
        decision_ts_utc=decision_bars[0],
        # A portfolio has no entry price, so the two are equity levels: the
        # Decision Bar's 1.0, and where the same walk without costs ended up.
        entry_price=1.0,
        exit_price=float(gross.equity.iloc[-1]),
        cost_bps_per_side=cost_bps_per_side,
        exit_reason=WINDOW_END,
    )
    run = CrossSectionalRun(
        result=result,
        selections=tuple(charged.selections),
        halt_exits=tuple(charged.halt_exits),
        exposure_gross=charged.exposure,
        lookback_days=lookback_days,
        holding_days=holding_days,
        quantile=quantile,
        min_universe=min_universe,
        max_cap_staleness_days=max_cap_staleness_days,
        max_weekly_rebalance_turnover=max_weekly_rebalance_turnover,
    )
    if run.weekly_rebalance_turnover > max_weekly_rebalance_turnover:
        raise TurnoverBudgetBreached(
            run.weekly_rebalance_turnover,
            max_weekly_rebalance_turnover,
            holding_days=holding_days,
        )
    return run


@dataclass(frozen=True)
class _AlignedPrices:
    """Every symbol's bars on one contiguous daily index.

    Contiguous on purpose: the rebalance schedule and the lookback are measured
    in days, and a frame with a hole in its index would quietly turn a 14-day
    lookback into a 14-*row* one spanning however long the hole is.
    """

    opens: pd.DataFrame
    closes: pd.DataFrame
    tradeable_bar: pd.DataFrame

    @classmethod
    def of(cls, bars_by_symbol: Mapping[str, pd.DataFrame]) -> "_AlignedPrices":
        if not bars_by_symbol:
            raise NotEnoughHistory("a cross-section needs at least one symbol's bars")
        spans = [bars.index for bars in bars_by_symbol.values() if len(bars)]
        if not spans:
            raise NotEnoughHistory("every symbol's bars are empty")
        index = pd.date_range(
            min(span[0] for span in spans),
            max(span[-1] for span in spans),
            freq="D",
            tz="UTC",
            name="ts_utc",
        )
        opens, closes, tradeable = {}, {}, {}
        for symbol, bars in bars_by_symbol.items():
            aligned = bars.reindex(index)
            opens[symbol] = aligned["open"].astype(float)
            closes[symbol] = aligned["close"].astype(float)
            tradeable[symbol] = is_tradeable_bar(aligned)
        return cls(
            opens=_framed(opens, index),
            closes=_framed(closes, index),
            tradeable_bar=_framed(tradeable, index).astype(bool),
        )

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def columns(self) -> pd.Index:
        return self.closes.columns


@dataclass(frozen=True)
class _PlannedRebalance:
    """What one Decision Bar decided, before any of it is filled or paid for.

    Separate from `Selection`, which is what the rebalance turned out to be: this
    is formed from the signal alone, and the same plan is walked twice — once
    with costs and once without — so the two walks cannot select differently.
    """

    decision_ts: pd.Timestamp
    selected: tuple[str, ...]
    weights: pd.Series
    n_eligible: int


@dataclass(frozen=True)
class _Walk:
    """One pass of the path: the equity it produced and what happened along it."""

    equity: pd.Series
    exposure: pd.Series
    selections: list[Selection]
    halt_exits: list[HaltExit]


def _decision_bars(
    index: pd.DatetimeIndex, *, lookback_days: int, holding_days: int
) -> list[pd.Timestamp]:
    """Every Decision Bar in the window, `holding_days` apart.

    The first is the earliest date with a full lookback behind it — the formation
    bar is the day before, and the lookback runs `lookback_days` back from there.
    The last is the latest with a bar after it to fill on.
    """
    first = index[0] + pd.Timedelta(days=lookback_days + 1)
    step = pd.Timedelta(days=holding_days)
    last_fillable = index[-1] - pd.Timedelta(days=1)
    bars = []
    decision_ts = first
    while decision_ts <= last_fillable:
        bars.append(decision_ts)
        decision_ts = decision_ts + step
    if not bars:
        raise NotEnoughHistory(
            f"a {lookback_days}-day lookback and a fill on the next bar need at "
            f"least {lookback_days + 2} days; the window runs "
            f"{index[0].date()} to {index[-1].date()}"
        )
    return bars


def _plan_rebalances(
    prices: _AlignedPrices,
    *,
    gate: pd.DataFrame,
    market_caps: pd.DataFrame,
    decision_bars: list[pd.Timestamp],
    lookback_days: int,
    quantile: float,
    min_universe: int,
    max_cap_staleness_days: int,
) -> list[_PlannedRebalance]:
    """Form every rebalance's target weights, once, before any of them is filled.

    Selection depends on nothing downstream of it — not on what is currently
    held, not on what the trades cost — so it is computed here and reused by both
    the charged and the costless walk. Two walks that could select differently
    would make Cost Drag a comparison of two strategies rather than of two cost
    assumptions.
    """
    plan = []
    for decision_ts in decision_bars:
        signal = past_return(
            prices.closes, decision_ts=decision_ts, lookback_days=lookback_days
        )
        caps = market_caps_before(
            _aligned_caps(market_caps, prices.columns),
            decision_ts=decision_ts,
            max_staleness_days=max_cap_staleness_days,
        )
        # The formation bar is the last one the signal could see, so it is also
        # the last one on which we know the asset was actually trading.
        formation_ts = decision_ts - pd.Timedelta(days=1)
        eligible = (
            gate.loc[decision_ts]
            & prices.tradeable_bar.loc[formation_ts]
            & signal.notna()
            & caps.notna()
            & (caps > 0.0)
        )
        n_eligible = int(eligible.sum())
        if n_eligible < min_universe:
            plan.append(
                _PlannedRebalance(decision_ts, (), pd.Series(dtype=float), n_eligible)
            )
            continue
        selected = select_top_quantile(signal.where(eligible), quantile=quantile)
        plan.append(
            _PlannedRebalance(
                decision_ts, selected, value_weights(caps[list(selected)]), n_eligible
            )
        )
    return plan


def _walk(
    prices: _AlignedPrices,
    plan: list[_PlannedRebalance],
    *,
    cost_per_side: float,
) -> _Walk:
    """Walk the plan day by day, holding units of each asset and marking nightly.

    `cost_per_side` is a fraction, not basis points, and is charged on each side
    of every trade. Zero walks the same plan for free, which is the gross path.

    Units rather than weights, because a value-weighted portfolio drifts between
    rebalances and the drift is the position: re-imposing the target weights on
    every mark would be a daily rebalance nobody asked for and nobody paid for.
    """
    index = prices.index
    first_fill = plan[0].decision_ts + pd.Timedelta(days=1)
    marks = index[index >= first_fill]

    fills = {
        planned.decision_ts + pd.Timedelta(days=1): planned for planned in plan
    }

    units: dict[str, float] = {}
    last_tradeable_close: dict[str, float] = {}
    cash = 1.0
    equity, exposure = {}, {}
    selections: list[Selection] = []
    halt_exits: list[HaltExit] = []

    for position, today in enumerate(marks):
        opens_today = prices.opens.loc[today]
        closes_today = prices.closes.loc[today]
        # Two different questions, and conflating them is how the fill decision
        # ends up reading the fill session's own volume. `has_open` is what is
        # knowable at the open, when orders go in. `traded_all_day` is only
        # knowable once the session is over, and is what says the asset halted.
        has_open = opens_today.notna() & (opens_today > 0.0)
        traded_all_day = prices.tradeable_bar.loc[today]

        def exit_position(symbol: str, cash: float) -> float:
            exit_price = last_tradeable_close[symbol]
            proceeds = units.pop(symbol) * exit_price
            halt_exits.append(HaltExit(symbol, today, exit_price))
            return cash + proceeds - cost_per_side * proceeds

        # Morning. A holding with no opening price is one no order can reach
        # today, so it leaves the book at the last price it could have been sold
        # at — before the rebalance, which would otherwise value it at a NaN.
        for symbol in [held for held in units if not has_open[held]]:
            cash = exit_position(symbol, cash)

        planned = fills.get(today)
        if planned is not None:
            rebalanced = _rebalance(
                prices,
                planned,
                today=today,
                units=units,
                cash=cash,
                cost_per_side=cost_per_side,
                has_open=has_open,
            )
            units, cash = rebalanced.units, rebalanced.cash
            selections.append(rebalanced.recorded)
            # Whatever is on the book was traded at today's open, so that is now
            # the last price we know an order of ours filled at.
            for symbol in units:
                last_tradeable_close[symbol] = float(opens_today[symbol])

        # Evening. The session is over, so now we know which assets it produced
        # no tradeable bar for. One bought at this morning's open and gone dark
        # by the close exits at what we paid for it, which is the only price
        # anyone actually transacted at.
        for symbol in [held for held in units if not traded_all_day[held]]:
            cash = exit_position(symbol, cash)

        for symbol in units:
            last_tradeable_close[symbol] = float(closes_today[symbol])
        invested = sum(held * closes_today[symbol] for symbol, held in units.items())

        # Exposure is recorded before the book is closed: the position was held
        # through this session, and averaging in the flat day the window happens
        # to end on would understate every run by one day.
        value = cash + invested
        equity[today] = value
        exposure[today] = invested / value if value > 0.0 else 0.0

        if position == len(marks) - 1:
            # The window ends, so the book is closed at the last close and the
            # sell-side cost lands on the final mark — the same convention the
            # single-asset hold uses.
            cash += invested - cost_per_side * invested
            units = {}
            equity[today] = cash

    return _Walk(
        equity=pd.Series(equity, name="equity_net").rename_axis("ts_utc"),
        exposure=pd.Series(exposure, name="exposure_gross").rename_axis("ts_utc"),
        selections=selections,
        halt_exits=halt_exits,
    )


@dataclass(frozen=True)
class _Rebalanced:
    """The book after one rebalance, and the record of what it did."""

    units: dict[str, float]
    cash: float
    recorded: Selection


def _rebalance(
    prices: _AlignedPrices,
    planned: _PlannedRebalance,
    *,
    today: pd.Timestamp,
    units: dict[str, float],
    cash: float,
    cost_per_side: float,
    has_open: pd.Series,
) -> _Rebalanced:
    """Trade the book to the target weights at today's open, and pay for it.

    `has_open` is the only thing this may read about today, and it is what an
    order sees at the open: a price to fill against. Whether the session goes on
    to trade at all is not knowable yet, so it cannot gate a fill — a selected
    asset that halts later today is bought here and exits this evening at what
    was paid for it.

    A selected asset with no opening price is not bought: its weight stays in
    cash and it is recorded as unfilled. Waiting for it to reopen, or
    renormalising the rest to cover for it, would both spend information the
    Decision Bar did not have.
    """
    decision_ts, selected = planned.decision_ts, planned.selected
    opens_today = prices.opens.loc[today]

    value = cash + sum(held * opens_today[symbol] for symbol, held in units.items())
    if not value > 0.0:
        # Nothing left to trade with. The mark that follows records the fact.
        return _Rebalanced(
            units={},
            cash=value,
            recorded=Selection(
                decision_ts_utc=decision_ts,
                entry_ts_utc=today,
                symbols=(),
                weights={},
                n_eligible=planned.n_eligible,
                unfilled=selected,
                turnover=0.0,
            ),
        )

    fillable = [symbol for symbol in selected if has_open[symbol]]
    unfilled = tuple(symbol for symbol in selected if symbol not in fillable)

    held_weights = {
        symbol: held * opens_today[symbol] / value for symbol, held in units.items()
    }
    target_weights = {symbol: float(planned.weights[symbol]) for symbol in fillable}

    changes = [
        target_weights.get(symbol, 0.0) - held_weights.get(symbol, 0.0)
        for symbol in set(target_weights) | set(held_weights)
    ]
    # Both legs pay, per ADR-0007: `traded` counts the sells and the buys
    # separately, and each is charged the per-side cost. `bought` is the buy side
    # alone, which is one-way Rebalance Turnover as ADR-0007 measures it.
    traded = sum(abs(change) for change in changes)
    bought = sum(change for change in changes if change > 0.0)
    net_value = value - cost_per_side * traded * value

    new_units = {
        symbol: target_weights[symbol] * net_value / float(opens_today[symbol])
        for symbol in fillable
        if target_weights[symbol] > 0.0
    }
    invested = sum(target_weights[symbol] for symbol in fillable) * net_value
    return _Rebalanced(
        units=new_units,
        cash=net_value - invested,
        recorded=Selection(
            decision_ts_utc=decision_ts,
            entry_ts_utc=today,
            symbols=tuple(fillable),
            weights=target_weights,
            n_eligible=planned.n_eligible,
            unfilled=unfilled,
            turnover=bought,
        ),
    )


def _marked(equity: pd.Series) -> MarkedPath:
    """Hand the walked equity to the daily mark, which owns the Liquidation rule.

    The costs are already inside `equity` — charged trade by trade, where they
    were actually paid — so nothing more is charged here. What `mark_daily` adds
    is the one thing the walk must not decide for itself: whether the path
    breached a 100% cumulative loss, and where it therefore ends.
    """
    breached = equity <= 0.0
    if breached.any():
        # Cut at the breach before dividing, rather than filling a NaN afterwards:
        # a portfolio worth nothing is not a base to measure the next day against,
        # and a fill would turn an arithmetic accident into a silent liquidation.
        equity = equity.loc[: breached.idxmax()]
    previous = equity.shift(1)
    previous.iloc[0] = 1.0
    return mark_daily(equity / previous - 1.0)


def _aligned_gate(
    tradeable: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index
) -> pd.DataFrame:
    """The Universe panel on the price frame's dates and symbols, absent as False.

    A date or symbol the panel does not carry is untradeable rather than an
    error: the panel is a gate that can only ever remove, and silence from it is
    not permission.
    """
    return tradeable.reindex(index=index, columns=columns).fillna(False).astype(bool)


def _aligned_caps(market_caps: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    """The capitalisation panel on the price frame's symbols, absent as NaN."""
    return market_caps.reindex(columns=columns)


def _framed(columns: dict[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame(columns, index=index)
    frame.columns.name = "symbol"
    return frame


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return float(sum(collected) / len(collected))
