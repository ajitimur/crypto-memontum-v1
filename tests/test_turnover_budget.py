"""Rebalance Turnover as a budget, on a fixture worked out entirely by hand.

Everything that could blur the arithmetic is held still. Both names' prices are
flat for the whole window, so nothing drifts between rebalances, no return is
earned, and the only thing that moves the equity is what it costs to trade. What
*does* move is the market-cap panel: the pair's value weights swap on each
snapshot, so the book has to be traded from one weighting to the other and back.

That makes both figures ADR-0007 turns on checkable without running anything —
the turnover at each rebalance, and the net equity the costs leave behind.
"""

import pandas as pd
import pytest

from crypto_momentum.costs import PAPER, TOKOCRYPTO, TURNOVER_CEILING_WEEKLY
from crypto_momentum.sim.cross_sectional import (
    SelectionError,
    TurnoverBudgetBreached,
    simulate_cross_sectional,
)

PAIR = ("XUSDT", "YUSDT")

# 2021-01-01 to 2021-01-19 inclusive. A one-day lookback puts the first Decision
# Bar on 01-03, and a weekly holding period runs them 01-03, 01-10 and 01-17,
# filling the next morning on 01-04, 01-11 and 01-18. The book closes on the last
# mark, 01-19.
N_DAYS = 19

# Flat for the whole window, and different from each other so that a unit count
# can never be mistaken for a weight.
FLAT_PRICE = {"XUSDT": 100.0, "YUSDT": 50.0}

# Three snapshots, each the last one strictly before one of the weekly Decision
# Bars, and each swapping which name carries three quarters of the pair's value.
WEEKLY_SNAPSHOTS = ("2021-01-01", "2021-01-09", "2021-01-16")
WEEKLY_CAPS = {"XUSDT": [3e9, 1e9, 3e9], "YUSDT": [1e9, 3e9, 1e9]}

# A fortnightly run only reaches the Decision Bars on 01-03 and 01-17, so it
# needs the swap to fall between those two. On the weekly grid above it would
# step straight over the 01-09 snapshot and find the weights it already held.
FORTNIGHTLY_SNAPSHOTS = ("2021-01-01", "2021-01-16")
FORTNIGHTLY_CAPS = {"XUSDT": [3e9, 1e9], "YUSDT": [1e9, 3e9]}


def market(
    *, snapshots=WEEKLY_SNAPSHOTS, caps_by_symbol=None
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """The pair's bars, an open Universe gate, and the swapping value panel."""
    index = pd.date_range("2021-01-01", periods=N_DAYS, freq="D", tz="UTC", name="ts_utc")
    bars = {
        symbol: pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1_000.0,
            },
            index=index,
        )
        for symbol, price in FLAT_PRICE.items()
    }
    tradeable = pd.DataFrame(True, index=index, columns=list(PAIR))
    caps = pd.DataFrame(
        WEEKLY_CAPS if caps_by_symbol is None else caps_by_symbol,
        index=pd.DatetimeIndex(snapshots, tz="UTC", name="ts_utc"),
    )
    return bars, tradeable, caps


def rotation(
    *,
    cost_bps_per_side: float,
    budget: float = 0.5,
    holding_days: int = 7,
    snapshots=WEEKLY_SNAPSHOTS,
    caps_by_symbol=None,
):
    bars, tradeable, caps = market(snapshots=snapshots, caps_by_symbol=caps_by_symbol)
    return simulate_cross_sectional(
        bars,
        tradeable=tradeable,
        market_caps=caps,
        lookback_days=1,
        holding_days=holding_days,
        # The whole pair is held, so the weights are the panel's own and each
        # rebalance is a reweighting rather than a change of names.
        quantile=1.0,
        min_universe=2,
        cost_bps_per_side=cost_bps_per_side,
        max_weekly_rebalance_turnover=budget,
    )


class TestTheHandComputedTurnover:
    """The fill on 01-04 buys 0.75 of X and 0.25 of Y from cash — one whole book.
    On 01-11 the target is 0.25/0.75, so 0.5 of the book is sold and 0.5 bought:
    one-way Rebalance Turnover of exactly 0.5. On 01-18 the weights swap back, so
    0.5 again."""

    def test_the_book_is_rebalanced_three_times_on_the_expected_bars(self):
        run = rotation(cost_bps_per_side=0.0)

        assert [
            selection.entry_ts_utc.strftime("%Y-%m-%d") for selection in run.selections
        ] == ["2021-01-04", "2021-01-11", "2021-01-18"]
        assert run.selections[0].weights == pytest.approx({"XUSDT": 0.75, "YUSDT": 0.25})
        assert run.selections[1].weights == pytest.approx({"XUSDT": 0.25, "YUSDT": 0.75})

    def test_each_rotation_turns_over_exactly_half_the_book(self):
        run = rotation(cost_bps_per_side=0.0)

        # The opening buy takes the whole book, by construction rather than by
        # signal, which is why it is recorded but not measured.
        assert run.selections[0].turnover == pytest.approx(1.0)
        assert run.rebalance_turnovers == pytest.approx((0.5, 0.5))
        assert run.mean_rebalance_turnover == pytest.approx(0.5)
        assert run.max_rebalance_turnover == pytest.approx(0.5)

    def test_a_weekly_rebalance_reports_its_turnover_unchanged(self):
        run = rotation(cost_bps_per_side=0.0)

        assert run.weekly_rebalance_turnover == pytest.approx(0.5)
        assert run.weekly_rebalance_turnover == 2 * TURNOVER_CEILING_WEEKLY

    def test_a_fortnightly_rotation_halves_onto_the_weekly_footing(self):
        # Decision Bars on 01-03 and 01-17 now, so one measured rebalance, and it
        # still swaps 0.75/0.25 for 0.25/0.75 — half the book. Half a book a
        # fortnight is a quarter of a book a week: the same trade, half the rate,
        # and the comparison the ceiling exists to get right.
        run = rotation(
            cost_bps_per_side=0.0,
            holding_days=14,
            budget=0.25,
            snapshots=FORTNIGHTLY_SNAPSHOTS,
            caps_by_symbol=FORTNIGHTLY_CAPS,
        )

        assert run.rebalance_turnovers == pytest.approx((0.5,))
        assert run.mean_rebalance_turnover == pytest.approx(0.5)
        assert run.weekly_rebalance_turnover == pytest.approx(0.25)


class TestTheHandComputedNetFigure:
    """Costs land on four occasions: the opening buy, the two rotations, and the
    closing sell. Each turns over one whole book's worth — the buy and the sell
    trade the lot in one direction, each rotation trades 0.5 out and 0.5 in — and
    each charges `c` a side on it. Prices never move, so the equity left at the
    end is the cost alone: `(1 - c) ** 4`."""

    def test_the_net_figure_is_the_cost_of_four_whole_books_changing_hands(self):
        c = TOKOCRYPTO.bps_per_side * 1e-4

        run = rotation(cost_bps_per_side=TOKOCRYPTO.bps_per_side)

        assert run.result.net_return == pytest.approx((1.0 - c) ** 4 - 1.0)
        assert run.result.equity_net.iloc[-1] == pytest.approx((1.0 - c) ** 4)

    def test_the_costless_walk_of_the_same_plan_is_exactly_flat(self):
        run = rotation(cost_bps_per_side=0.0)

        assert run.result.net_return == pytest.approx(0.0)
        assert run.result.equity_net.iloc[-1] == pytest.approx(1.0)

    def test_every_penny_of_the_shortfall_is_cost_drag(self):
        # Gross is the same walk without costs, and on flat prices it earns
        # nothing. So the net loss is Cost Drag exactly, and no part of it is a
        # haircut taken off a number the walk never produced.
        run = rotation(cost_bps_per_side=TOKOCRYPTO.bps_per_side)

        assert run.result.gross_return == pytest.approx(0.0)
        assert run.result.net_return < 0.0
        assert run.result.cost_drag_annualised > 0.0

    def test_the_two_models_price_the_same_trades_differently(self):
        # ADR-0007's whole argument on a fixture where nothing but cost moves.
        paper = rotation(cost_bps_per_side=PAPER.bps_per_side)
        tokocrypto = rotation(cost_bps_per_side=TOKOCRYPTO.bps_per_side)

        assert paper.result.net_return == pytest.approx(
            (1.0 - PAPER.bps_per_side * 1e-4) ** 4 - 1.0
        )
        assert tokocrypto.result.net_return < paper.result.net_return


class TestTheBudgetTheRunnerEnforces:
    """The run-time half of ADR-0007's ceiling: what the walk actually traded."""

    def test_a_walk_that_breaches_its_budget_produces_no_result(self):
        # 50% weekly against ADR-0007's 25% ceiling. The run is refused rather
        # than filed as a result with the breach noted in its metadata.
        with pytest.raises(TurnoverBudgetBreached) as breach:
            rotation(cost_bps_per_side=0.0, budget=TURNOVER_CEILING_WEEKLY)

        assert breach.value.realised_weekly_turnover == pytest.approx(0.5)
        assert breach.value.budget == pytest.approx(0.25)
        # The message carries the realised figure, because the next move is to
        # widen the holding period until it fits and that needs a starting point.
        assert "50.0%" in str(breach.value)

    def test_a_walk_inside_its_budget_is_returned(self):
        run = rotation(cost_bps_per_side=0.0, budget=0.5)

        assert run.weekly_rebalance_turnover <= run.max_weekly_rebalance_turnover

    def test_a_budget_of_nothing_is_refused(self):
        # Whether a budget is *permissible* is ADR-0007's question and the config
        # loader's to answer — the core deliberately does not re-decide it, or
        # the policy would live in two places and the core could never walk the
        # high-turnover regime the ADR is an argument about. What the core does
        # insist on is that a budget is a budget.
        with pytest.raises(SelectionError, match="must be above 0"):
            rotation(cost_bps_per_side=0.0, budget=0.0)

    def test_the_reported_turnover_reaches_the_result_metadata(self):
        metadata = rotation(cost_bps_per_side=0.0, budget=0.5).to_metadata()

        assert metadata["mean_rebalance_turnover"] == pytest.approx(0.5)
        assert metadata["weekly_rebalance_turnover"] == pytest.approx(0.5)
        assert metadata["max_weekly_rebalance_turnover"] == pytest.approx(0.5)
        assert metadata["turnover_ceiling_weekly"] == 0.25
