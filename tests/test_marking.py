"""Seam: the daily mark. Liquidation is terminal, and a halt exits at the last price.

ADR-0001. The trigger is inert under v1 long-only unlevered spot — a spot holding
cannot lose more than it cost — so the fixtures here carry the leveraged and short
paths that make it reachable, which is what the trigger exists for.
"""

import pandas as pd
import pytest

from crypto_momentum.sim.marking import NothingToMark, mark_daily, stops_trading_at


def marks(returns) -> pd.Series:
    """Per-mark simple returns on consecutive UTC days, as the simulator hands them over."""
    index = pd.date_range("2021-07-05", periods=len(returns), freq="D", tz="UTC", name="ts_utc")
    return pd.Series(returns, index=index, dtype=float)


SQUEEZE_PRICES = [100.0, 250.0, 90.0]


def short_position_through_a_squeeze() -> pd.Series:
    """Han et al.'s case: a short whose week ends up while its Tuesday wipes it out.

    Prices 100 -> 250 -> 90. The short's boundary return is +10%, so a simulator
    that only looks at period boundaries records a profitable week; the daily path
    is -150% on the first mark and the position is gone before the price falls back.
    """
    return marks(
        [
            -(SQUEEZE_PRICES[day] / SQUEEZE_PRICES[day - 1] - 1.0)
            for day in (1, 2)
        ]
    )


def test_the_position_is_marked_on_every_day_it_is_held():
    path = mark_daily(marks([0.10, 0.10, 0.10]))

    assert len(path.equity_net) == 3
    assert path.equity_net.tolist() == pytest.approx([1.10, 1.21, 1.331])
    assert not path.liquidated


def test_a_cumulative_loss_breaching_one_hundred_percent_is_a_liquidation_with_its_date():
    path = mark_daily(marks([-0.50, -1.40]))

    assert path.liquidated
    assert path.liquidation_ts_utc == pd.Timestamp("2021-07-06T00:00:00Z")
    assert path.equity_net.iloc[-1] == 0.0


def test_a_liquidated_run_never_continues_past_the_breach():
    """The wipeout is terminal: no later gain, however large, restores the series."""
    path = mark_daily(marks([-1.20, 50.0, 50.0]))

    assert path.liquidated
    assert path.equity_net.index[-1] == pd.Timestamp("2021-07-05T00:00:00Z")
    assert len(path.equity_net) == 1


def test_a_path_that_breaches_intra_period_liquidates_though_its_boundary_return_is_positive():
    """The fixture ADR-0001 asks for. Boundary-only accounting books a winning week."""
    daily = short_position_through_a_squeeze()
    # What a boundary-only simulator would book: the short's return from the
    # first price of the period to the last, with nothing in between.
    boundary_return = -(SQUEEZE_PRICES[-1] / SQUEEZE_PRICES[0] - 1.0)
    assert boundary_return == pytest.approx(0.10)

    path = mark_daily(daily)

    assert path.liquidated
    assert path.liquidation_ts_utc == pd.Timestamp("2021-07-05T00:00:00Z")
    assert path.equity_net.tolist() == [0.0]


def test_equity_reaching_exactly_zero_is_a_liquidation():
    """A 100% loss is a breach: there is nothing left to earn a return on."""
    path = mark_daily(marks([-1.0, 0.5]))

    assert path.liquidated
    assert path.equity_net.iloc[-1] == 0.0


def test_a_loss_short_of_one_hundred_percent_is_a_drawdown_and_not_a_liquidation():
    path = mark_daily(marks([-0.90, -0.90]))

    assert not path.liquidated
    assert path.liquidation_ts_utc is None
    assert path.equity_net.iloc[-1] == pytest.approx(0.01)


def test_the_entry_cost_is_paid_at_the_fill_so_every_mark_is_already_net_of_it():
    path = mark_daily(marks([0.10, 0.10]), entry_cost=0.01)

    assert path.equity_net.iloc[0] == pytest.approx(0.99 * 1.10)
    assert path.equity_net.iloc[-1] == pytest.approx(0.99 * 1.21 * 1.0)


def test_the_exit_cost_lands_on_the_final_mark_where_the_position_closes():
    path = mark_daily(marks([0.10, 0.10]), entry_cost=0.01, exit_cost=0.01)

    assert path.equity_net.iloc[-1] == pytest.approx(0.99 * 1.21 * 0.99)


def test_a_liquidation_charges_no_exit_cost_because_nothing_is_left_to_sell():
    path = mark_daily(marks([-1.50]), entry_cost=0.01, exit_cost=0.01)

    assert path.equity_net.iloc[-1] == 0.0


def test_marking_a_holding_period_with_no_marks_is_refused():
    with pytest.raises(NothingToMark):
        mark_daily(marks([]))


def test_marking_does_not_mutate_the_returns_it_was_given():
    given = marks([-1.50, 2.0])
    before = given.copy()

    mark_daily(given)

    pd.testing.assert_series_equal(given, before)


def bars_from(rows) -> pd.DataFrame:
    """Daily bars as `close, volume` pairs, one row per UTC day."""
    index = pd.date_range("2021-07-05", periods=len(rows), freq="D", tz="UTC", name="ts_utc")
    return pd.DataFrame(
        [
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
            }
            for close, volume in rows
        ],
        index=index,
    )


def test_an_asset_trading_through_the_window_never_stops():
    assert stops_trading_at(bars_from([(100.0, 5.0), (110.0, 5.0)])) is None


def test_an_asset_whose_bars_stop_printing_a_trade_stops_at_the_first_empty_bar():
    """A bar with no volume is a bar at which the asset could not have been sold."""
    bars = bars_from([(100.0, 5.0), (110.0, 5.0), (110.0, 0.0), (110.0, 0.0)])

    assert stops_trading_at(bars) == pd.Timestamp("2021-07-07T00:00:00Z")


def test_an_asset_whose_price_goes_missing_stops_there():
    bars = bars_from([(100.0, 5.0), (float("nan"), 5.0)])

    assert stops_trading_at(bars) == pd.Timestamp("2021-07-06T00:00:00Z")


def test_a_price_of_zero_printed_on_real_volume_is_a_trade_and_not_a_halt():
    """The asset traded at nothing. That is a 100% loss for the mark to liquidate
    on, not an exit at the last price that happened to be positive."""
    bars = bars_from([(100.0, 5.0), (0.0, 5.0)])

    assert stops_trading_at(bars) is None


def test_a_price_of_zero_with_nothing_trading_is_a_halt():
    bars = bars_from([(100.0, 5.0), (0.0, 0.0)])

    assert stops_trading_at(bars) == pd.Timestamp("2021-07-06T00:00:00Z")


def test_an_asset_that_halts_and_then_prints_again_stops_at_the_halt():
    """The exit is the last tradeable price, not a resumption we could not have
    waited for without knowing in advance that it would come."""
    bars = bars_from([(100.0, 5.0), (110.0, 0.0), (120.0, 5.0)])

    assert stops_trading_at(bars) == pd.Timestamp("2021-07-06T00:00:00Z")
