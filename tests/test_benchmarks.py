"""The two reference portfolios every run is read against.

BTC buy-and-hold is the deployment hurdle of ADR-0005 — the honest alternative
use of the money. The cap-weighted market portfolio is the secondary reference
the literature quotes, built from the same point-in-time Universe and the same
CoinMarketCap panel the strategy weights with, so the comparison is like for like.

The fixtures are hand-built: every number asserted here was worked out by hand.
"""

import pandas as pd
import pytest

from crypto_momentum.sim.benchmarks import (
    BENCHMARK_SYMBOL,
    btc_buy_and_hold,
    cap_weighted_market,
    deployment_hurdle,
)
from crypto_momentum.sim.cross_sectional import simulate_cross_sectional
from crypto_momentum.sim.marking import mark_daily
from crypto_momentum.sim.report import summarise

DAILY_RATE = {
    "BTCUSDT": 0.010,
    "ETHUSDT": 0.008,
    "BNBUSDT": 0.006,
    "ADAUSDT": 0.004,
    "XRPUSDT": 0.002,
    "DOGEUSDT": 0.000,
}
# The reverse of the return ranking, so a cap-weighted book cannot be mistaken
# for an equal-weighted one.
MARKET_CAP = {
    "BTCUSDT": 1e9,
    "ETHUSDT": 2e9,
    "BNBUSDT": 3e9,
    "ADAUSDT": 4e9,
    "XRPUSDT": 5e9,
    "DOGEUSDT": 6e9,
}


def dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC", name="ts_utc")


def closes_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One steady ramp per symbol, so a ranking is fixed and readable by hand."""
    built = pd.DataFrame(
        {
            symbol: [100.0 * (1.0 + rate) ** step for step in range(len(index))]
            for symbol, rate in DAILY_RATE.items()
        },
        index=index,
    )
    built.columns.name = "symbol"
    return built


def bars_from_closes(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily bars whose open is the previous close, so a fill price is checkable."""
    bars = {}
    for symbol in closes.columns:
        close = closes[symbol]
        opens = close.shift(1)
        opens.iloc[0] = close.iloc[0]
        bars[symbol] = pd.DataFrame(
            {
                "open": opens,
                "high": close,
                "low": close,
                "close": close,
                "volume": pd.Series(1_000.0, index=close.index),
            }
        )
    return bars


def all_tradeable(closes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=closes.index, columns=closes.columns)


def flat_caps(closes: pd.DataFrame) -> pd.DataFrame:
    """Weekly capitalisation snapshots that do not move, so weights are hand-checkable."""
    snapshots = closes.index[::7]
    return pd.DataFrame(
        {symbol: [cap] * len(snapshots) for symbol, cap in MARKET_CAP.items()},
        index=snapshots,
    )


@pytest.fixture
def market_inputs():
    closes = closes_frame(dates("2021-01-01", 60))
    return {
        "bars_by_symbol": bars_from_closes(closes),
        "tradeable": all_tradeable(closes),
        "market_caps": flat_caps(closes),
    }


def market_run(market_inputs, **overrides):
    arguments = {
        "tradeable": market_inputs["tradeable"],
        "market_caps": market_inputs["market_caps"],
        "lookback_days": 14,
        "holding_days": 7,
        "cost_bps_per_side": 0.0,
        **overrides,
    }
    return cap_weighted_market(market_inputs["bars_by_symbol"], **arguments)


def strategy_run(market_inputs):
    return simulate_cross_sectional(
        market_inputs["bars_by_symbol"],
        tradeable=market_inputs["tradeable"],
        market_caps=market_inputs["market_caps"],
        lookback_days=14,
        holding_days=7,
        quantile=0.2,
        cost_bps_per_side=0.0,
    )


def test_the_market_portfolio_holds_the_whole_eligible_universe(market_inputs):
    """Not a quintile: the market is everything the Universe offered that date."""
    market = market_run(market_inputs)

    assert set(market.selections[0].symbols) == set(DAILY_RATE)
    assert len(strategy_run(market_inputs).selections[0].symbols) == 2


def test_the_market_portfolio_is_weighted_by_capitalisation(market_inputs):
    """Caps of 1 to 6 billion sum to 21, so BTC carries 1/21 and DOGE 6/21."""
    market = market_run(market_inputs)

    weights = market.selections[0].weights
    assert weights["BTCUSDT"] == pytest.approx(1.0 / 21.0)
    assert weights["DOGEUSDT"] == pytest.approx(6.0 / 21.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_an_asset_the_universe_did_not_offer_that_date_is_not_in_the_market(
    market_inputs,
):
    """The market portfolio resolves as of the rebalance date, like everything else."""
    tradeable = market_inputs["tradeable"].copy()
    tradeable["DOGEUSDT"] = False

    market = market_run(market_inputs, tradeable=tradeable)

    weights = market.selections[0].weights
    assert "DOGEUSDT" not in market.selections[0].symbols
    # The remaining caps sum to 15 billion, so BTC's share rises to 1/15.
    assert weights["BTCUSDT"] == pytest.approx(1.0 / 15.0)


def test_the_market_portfolio_shares_the_strategy_s_rebalance_cadence(market_inputs):
    """Same lookback and holding period, so the two paths cover the same dates."""
    market = market_run(market_inputs)
    strategy = strategy_run(market_inputs)

    assert market.result.entry_ts_utc == strategy.result.entry_ts_utc
    assert market.result.exit_ts_utc == strategy.result.exit_ts_utc


def test_the_market_portfolio_is_net_of_the_same_cost_the_strategy_pays(market_inputs):
    charged = market_run(market_inputs, cost_bps_per_side=50.0)
    free = market_run(market_inputs)

    assert charged.result.net_return < free.result.net_return


def bars_from(rows) -> pd.DataFrame:
    index = pd.date_range(
        "2021-01-01", periods=len(rows), freq="D", tz="UTC", name="ts_utc"
    )
    return pd.DataFrame(
        [
            {"open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1.0}
            for o, c in rows
        ],
        index=index,
    )


# A Decision Bar, then three days compounding at exactly +10%.
BTC_BARS = bars_from([(100.0, 100.0), (100.0, 110.0), (110.0, 121.0), (121.0, 133.1)])


def test_the_hurdle_is_btc_bought_and_held_net_of_costs():
    assert BENCHMARK_SYMBOL == "BTCUSDT"

    hurdle = btc_buy_and_hold(BTC_BARS, cost_bps_per_side=50.0)

    assert hurdle.gross_return == pytest.approx(0.331)
    assert hurdle.net_return < 0.331


def result_from(returns):
    """A path built from daily marks, so its Sharpe and drawdown are ours to set."""
    index = pd.date_range(
        "2021-01-02", periods=len(returns), freq="D", tz="UTC", name="ts_utc"
    )
    return summarise(
        mark_daily(pd.Series(returns, index=index, dtype=float)),
        decision_ts_utc=pd.Timestamp("2021-01-01T00:00:00Z"),
        entry_price=1.0,
        exit_price=1.0,
        cost_bps_per_side=0.0,
    )


BTC_REFERENCE = result_from([0.01, -0.01] * 60)


def test_a_run_that_beats_btc_on_all_three_conditions_clears_the_hurdle():
    """ADR-0005: better Sharpe, t > 3.0, and a drawdown no worse than BTC's."""
    strong = result_from([0.01, 0.005] * 60)

    hurdle = deployment_hurdle(strong, btc=BTC_REFERENCE)

    assert hurdle.sharpe_above_btc is True
    assert hurdle.drawdown_no_worse_than_btc is True
    assert hurdle.clears_profitability_bar is True
    assert hurdle.clears is True


def test_a_run_that_draws_down_harder_than_btc_fails_the_hurdle():
    """The condition ADR-0005 expects to bind, asserted rather than assumed."""
    deep = result_from([0.05] * 40 + [-0.30] + [0.05] * 40)

    hurdle = deployment_hurdle(deep, btc=BTC_REFERENCE)

    assert hurdle.drawdown_no_worse_than_btc is False
    assert hurdle.clears is False


def test_a_run_that_cannot_be_judged_against_btc_does_not_clear_by_default():
    """No BTC path is not a pass. The hurdle says it was not decided, and fails."""
    hurdle = deployment_hurdle(result_from([0.01, 0.005] * 60), btc=None)

    assert hurdle.sharpe_above_btc is None
    assert hurdle.drawdown_no_worse_than_btc is None
    assert hurdle.clears is False


def test_the_hurdle_is_recorded_as_the_three_conditions_it_is():
    metadata = deployment_hurdle(
        result_from([0.01, 0.005] * 60), btc=BTC_REFERENCE
    ).to_metadata()

    assert metadata["adr"] == "ADR-0005"
    assert set(metadata) >= {
        "sharpe_above_btc",
        "drawdown_no_worse_than_btc",
        "clears_profitability_bar",
        "clears",
    }
