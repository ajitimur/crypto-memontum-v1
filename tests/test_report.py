"""How a marked path reads once it is reported — a liquidated one above all.

The v1 strategy is long-only unlevered spot, under which a Liquidation cannot
happen (ADR-0004), so this is where the reporting of one is exercised: at the
seam, on a path handed over directly.
"""

import math

import pandas as pd
import pytest

from crypto_momentum.sim.marking import mark_daily
from crypto_momentum.sim.report import LIQUIDATED, WINDOW_END, summarise

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
