"""What a run is read against: BTC buy-and-hold, and the cap-weighted market.

A strategy's own numbers do not say whether it is worth deploying. ADR-0005
names what does: BTC buy-and-hold over the same window, because holding Bitcoin
needs no research, no rebalancing and almost no cost, so it is the honest
alternative use of the money. The cap-weighted market portfolio is logged
alongside it as the secondary reference the literature quotes — it decides
nothing, but a run that beats BTC while trailing the market is a different
finding from one that beats both.

Both benchmarks are simulated the same way the strategy is: marked daily, net of
the same per-side cost, over the same window. A benchmark computed on a
different basis than the run it judges is not a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from crypto_momentum.sim.buy_and_hold import simulate_buy_and_hold
from crypto_momentum.sim.cross_sectional import (
    DEFAULT_CAP_STALENESS_DAYS,
    CrossSectionalRun,
    simulate_cross_sectional,
)
from crypto_momentum.sim.report import RunResult

# The deployment hurdle is Bitcoin, on the venue we would actually trade it on.
BENCHMARK_SYMBOL = "BTCUSDT"

# The whole eligible cross-section, not a quintile of it: the market portfolio
# is what the Universe offered on the date, weighted by capitalisation.
WHOLE_UNIVERSE = 1.0

# A market portfolio holds whatever the Universe had. The strategy's own
# `min_universe` floor exists to stop a quintile of four names being called a
# cross-section; applying it here would have the reference hold cash on the
# dates the strategy did, which is the opposite of a reference.
MARKET_MIN_UNIVERSE = 1


def btc_buy_and_hold(bars: pd.DataFrame, *, cost_bps_per_side: float) -> RunResult:
    """Bitcoin bought at the fill bar and held to the end of the window.

    `bars` is BTC's daily bars over the run's own window, first row the Decision
    Bar, so the hurdle is filled and marked exactly as the strategy is. The same
    per-side cost is charged, per ADR-0007: a costless benchmark against a costed
    strategy flatters neither honestly.
    """
    return simulate_buy_and_hold(bars, cost_bps_per_side=cost_bps_per_side)


def cap_weighted_market(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    tradeable: pd.DataFrame,
    market_caps: pd.DataFrame,
    lookback_days: int,
    cost_bps_per_side: float,
    holding_days: int,
    max_cap_staleness_days: int = DEFAULT_CAP_STALENESS_DAYS,
) -> CrossSectionalRun:
    """The whole point-in-time Universe, weighted by CoinMarketCap capitalisation.

    The same simulator the strategy runs through, with the selection widened
    from the top quintile to everything eligible. That is what makes it a
    reference: same Universe as of each rebalance date, same vendor caps, same
    rebalance cadence, same daily mark, same costs — the one difference is that
    nothing is selected on the signal.

    `lookback_days` does not form a position here, but it does two things, and
    the second is a deliberate choice rather than a side effect. It sets where
    the first Decision Bar falls, so the two paths cover the same dates. And
    because eligibility runs through the same ranking the strategy uses, an
    asset without `lookback_days` of price history is out of the market
    portfolio on that date, exactly as it is out of the strategy's cross-section.

    That is narrower than "everything the Universe listed", and it is the right
    reference anyway: a newly listed asset the strategy could not have ranked is
    not something the strategy failed to hold. The alternative — a market
    portfolio holding names the strategy was structurally unable to select —
    would attribute the gap between them to the signal, which is the one thing
    the comparison is supposed to isolate.
    """
    return simulate_cross_sectional(
        bars_by_symbol,
        tradeable=tradeable,
        market_caps=market_caps,
        lookback_days=lookback_days,
        holding_days=holding_days,
        quantile=WHOLE_UNIVERSE,
        min_universe=MARKET_MIN_UNIVERSE,
        max_cap_staleness_days=max_cap_staleness_days,
        cost_bps_per_side=cost_bps_per_side,
    )


@dataclass(frozen=True)
class Hurdle:
    """ADR-0005's three conditions, and whether all of them are met.

    Each condition is `None` when it could not be decided — no BTC path over the
    window, or no Sharpe on either side of the comparison. An undecided condition
    never counts as met: `clears` is true only when all three are true, so a
    missing benchmark fails the hurdle rather than quietly passing it.
    """

    sharpe_above_btc: bool | None
    drawdown_no_worse_than_btc: bool | None
    clears_profitability_bar: bool

    @property
    def clears(self) -> bool:
        return (
            self.sharpe_above_btc is True
            and self.drawdown_no_worse_than_btc is True
            and self.clears_profitability_bar
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "adr": "ADR-0005",
            "sharpe_above_btc": self.sharpe_above_btc,
            "drawdown_no_worse_than_btc": self.drawdown_no_worse_than_btc,
            "clears_profitability_bar": self.clears_profitability_bar,
            "clears": self.clears,
        }


def deployment_hurdle(result: RunResult, *, btc: RunResult | None) -> Hurdle:
    """Read a run against BTC buy-and-hold, on ADR-0005's three conditions.

    Drawdowns are negative, so "no worse than BTC's" is `>=`: a fall of 40%
    against BTC's 60% clears, and one of 80% does not. ADR-0005 expects this to
    be the binding condition, not the Sharpe.
    """
    return Hurdle(
        sharpe_above_btc=_above(result.sharpe_net, btc.sharpe_net if btc else None),
        drawdown_no_worse_than_btc=(
            None if btc is None else result.max_drawdown >= btc.max_drawdown
        ),
        clears_profitability_bar=result.clears_profitability_bar,
    )


def _above(value: float | None, reference: float | None) -> bool | None:
    """Whether `value` beats `reference`, or `None` when either is undefined."""
    if value is None or reference is None:
        return None
    return value > reference
