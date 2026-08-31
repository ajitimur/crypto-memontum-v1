"""How a marked path reads once it is reported — a liquidated one above all.

The v1 strategy is long-only unlevered spot, under which a Liquidation cannot
happen (ADR-0004), so this is where the reporting of one is exercised: at the
seam, on a path handed over directly.
"""

import math

import pandas as pd
import pytest

from crypto_momentum.sim.marking import mark_daily
from crypto_momentum.sim.report import (
    LIQUIDATED,
    WINDOW_END,
    newey_west_lag_count,
    newey_west_t_statistic,
    summarise,
)

DECISION_TS = pd.Timestamp("2021-07-04T00:00:00Z")


def marks(returns) -> pd.Series:
    index = pd.date_range("2021-07-05", periods=len(returns), freq="D", tz="UTC", name="ts_utc")
    return pd.Series(returns, index=index, dtype=float)


def liquidated_run():
    """A position wiped out on its second mark, reported as of the price it died at."""
    path = mark_daily(marks([-0.50, -1.40, 3.0]))
    return summarise(
        path,
        decision_ts_utc=DECISION_TS,
        entry_price=100.0,
        exit_price=0.0,
        cost_bps_per_side=10.0,
    )


def test_a_liquidated_run_reports_as_liquidated_with_its_date():
    result = liquidated_run()

    assert result.liquidated
    assert result.exit_reason == LIQUIDATED
    assert result.liquidation_ts_utc == pd.Timestamp("2021-07-06T00:00:00Z")


def test_a_liquidated_run_ends_at_the_breach_and_reports_no_return_past_it():
    result = liquidated_run()

    assert result.exit_ts_utc == pd.Timestamp("2021-07-06T00:00:00Z")
    assert result.n_marks == 2
    assert result.net_return == pytest.approx(-1.0)
    assert result.ann_return_net == pytest.approx(-1.0)
    assert result.max_drawdown == pytest.approx(-1.0)


def test_a_liquidated_run_has_no_profitability_number_to_report():
    """ADR-0002's bar is a mean log return, and a wipeout's log return is not finite.
    The liquidation is the result; a number here would invite comparing it to one."""
    result = liquidated_run()

    assert result.mean_log_return_daily_net is None


def test_a_run_that_survives_reports_the_reason_it_was_handed():
    path = mark_daily(marks([0.10, 0.10]))

    result = summarise(
        path,
        decision_ts_utc=DECISION_TS,
        entry_price=100.0,
        exit_price=121.0,
        cost_bps_per_side=0.0,
        exit_reason=WINDOW_END,
    )

    assert not result.liquidated
    assert result.liquidation_ts_utc is None
    assert result.exit_reason == WINDOW_END
    assert result.mean_log_return_daily_net == pytest.approx(math.log(1.10))


def summarised(returns, *, cost_bps_per_side: float = 0.0):
    """Report a path built straight from daily marks, so the numbers are hand-checkable."""
    return summarise(
        mark_daily(marks(returns)),
        decision_ts_utc=DECISION_TS,
        entry_price=1.0,
        exit_price=1.0,
        cost_bps_per_side=cost_bps_per_side,
    )


def test_the_newey_west_lag_count_follows_the_usual_rule():
    """floor(4 * (T/100) ** (2/9)) — 4 lags at a hundred observations, 2 at five."""
    assert newey_west_lag_count(100) == 4
    assert newey_west_lag_count(5) == 2
    assert newey_west_lag_count(0) == 0


def test_the_t_statistic_at_zero_lags_is_the_plain_one():
    """Hand-worked: mean 3, population variance 2, so t = 3 / sqrt(2/5)."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    t_statistic, lags = newey_west_t_statistic(series, lags=0)

    assert lags == 0
    assert t_statistic == pytest.approx(3.0 / math.sqrt(2.0 / 5.0))


def test_the_t_statistic_widens_the_error_for_autocorrelation():
    """One lag: gamma_1 = 0.8 at weight 0.5, so S = 2 + 2(0.5)(0.8) = 2.8."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    t_statistic, lags = newey_west_t_statistic(series, lags=1)

    assert lags == 1
    assert t_statistic == pytest.approx(3.0 / math.sqrt(2.8 / 5.0))


def test_a_series_with_no_dispersion_has_no_t_statistic():
    """A vanishing standard error is not an infinite t, it is an undefined one."""
    t_statistic, _ = newey_west_t_statistic(pd.Series([0.01, 0.01, 0.01]))

    assert t_statistic is None


def test_the_profitability_bar_is_read_off_the_log_return_t_statistic():
    """ADR-0002: t > 3.0 on the mean log return, and nothing else, decides."""
    result = summarised([0.01] * 60 + [0.02] * 60)

    assert result.mean_log_return_t_stat is not None
    assert result.clears_profitability_bar == (result.mean_log_return_t_stat > 3.0)


def test_a_liquidated_run_has_no_t_statistic_and_clears_no_bar():
    result = liquidated_run()

    assert result.mean_log_return_t_stat is None
    assert result.clears_profitability_bar is False


def test_a_positive_mean_return_can_sit_on_a_negative_mean_log_return():
    """The divergence ADR-0002 asks to be called out: +50% and -40% alternating
    averages to +5% a day arithmetically while compounding to a loss."""
    result = summarised([0.50, -0.40, 0.50, -0.40])

    assert result.mean_return_daily_net == pytest.approx(0.05)
    assert result.mean_log_return_daily_net == pytest.approx(
        (math.log(1.5) + math.log(0.6)) / 2.0
    )
    assert result.mean_log_return_daily_net < 0.0
    assert result.mean_return_sign_divergence is True


def test_a_run_whose_two_means_agree_reports_no_divergence():
    result = summarised([0.01, 0.02, 0.01, 0.02])

    assert result.mean_return_daily_net > 0.0
    assert result.mean_log_return_daily_net > 0.0
    assert result.mean_return_sign_divergence is False
