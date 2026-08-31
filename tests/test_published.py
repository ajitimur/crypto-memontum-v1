"""The transcribed Table 14, checked against the paper's own prose.

Like `test_grid.py`, these assert literals against a citation rather than a
computation. The paper states several of its own numbers in the surrounding
text, and those sentences are the independent check on the transcription: if a
column were shifted or a row mistyped, the counts and the named cells below
would stop agreeing with what Han, Kang and Ryu wrote about their own table.
"""

import pytest

from crypto_momentum.sim.grid import HAN_KANG_RYU_21, GridCell
from crypto_momentum.sim.published import (
    LEGS,
    LONG_ONLY,
    LONG_SHORT,
    PUBLISHED_COST_BPS_PER_SIDE,
    PUBLISHED_MARKET,
    SHORT_ONLY,
    PublishedTableError,
    published_leg,
    published_liquidation_count,
)


def cell(lookback, holding):
    return GridCell(lookback_days=lookback, holding_days=holding)


def by_cell(leg):
    return {entry.cell: entry for entry in published_leg(leg)}


def test_every_leg_covers_the_published_grid_exactly():
    for leg in LEGS:
        assert [entry.cell for entry in published_leg(leg)] == list(HAN_KANG_RYU_21)


def test_the_table_is_the_after_cost_panel():
    """Panel (b). Their 15bp is not our 40.44bp per side, so it is recorded."""
    assert PUBLISHED_COST_BPS_PER_SIDE == 15.0


def test_the_five_liquidated_long_short_cells_are_the_five_the_paper_names():
    """Section 5.2.1: "five portfolios, (3, 21), (3, 28), (5, 21), (5, 28), and
    (7, 28), are liquidated during the sample period"."""
    liquidated = {
        entry.cell for entry in published_leg(LONG_SHORT) if entry.liquidated
    }
    assert liquidated == {
        cell(3, 21), cell(3, 28), cell(5, 21), cell(5, 28), cell(7, 28)
    }
    assert published_liquidation_count(LONG_SHORT) == 5


def test_no_long_only_portfolio_is_liquidated():
    """Section 5.2.1: "all long-only portfolios earn a positive profit"."""
    assert published_liquidation_count(LONG_ONLY) == 0
    assert all(entry.cum_return_pct > 0.0 for entry in published_leg(LONG_ONLY))


def test_most_short_only_portfolios_are_liquidated():
    """Section 5.2.1: "most short-only portfolios are liquidated"."""
    assert published_liquidation_count(SHORT_ONLY) == 16


def test_the_best_after_cost_long_short_cell_is_the_one_the_paper_names():
    """Section 5.2.1: "The (1, 7) portfolio remains the best performer with a
    Sharpe ratio of 1.31"."""
    entries = published_leg(LONG_SHORT)
    best = max(entries, key=lambda entry: entry.sharpe)
    assert best.cell == cell(1, 7)
    assert best.sharpe == 1.31


def test_the_fourteen_seven_long_short_cell_is_adr_0003_s_anchor():
    """Section 5.2.1: "the (14, 7) portfolio yields the highest cumulative return
    of 101,218% (Sharpe ratio = 1.28)"."""
    entry = by_cell(LONG_SHORT)[cell(14, 7)]
    assert entry.sharpe == 1.28
    assert entry.cum_return_pct == pytest.approx(101218.3)
    assert entry.cum_return_pct == max(
        published.cum_return_pct for published in published_leg(LONG_SHORT)
    )


def test_the_after_cost_long_only_sharpes_sit_in_the_published_range():
    """Section 5.2.1 reports the gross long-only range as 0.95 to 1.62; after
    costs every cell is lower, and the best is (14, 5)."""
    sharpes = [entry.sharpe for entry in published_leg(LONG_ONLY)]
    assert min(sharpes) == 0.90
    assert max(sharpes) == 1.52
    best = max(published_leg(LONG_ONLY), key=lambda entry: entry.sharpe)
    assert best.cell == cell(14, 5)


def test_only_the_three_twenty_one_long_short_cell_loses_money_on_average():
    """Section 5.2.1: "All the long-short portfolios except for the (3, 21) pair
    yield a positive mean return during the sample period"."""
    negative = {
        entry.cell
        for entry in published_leg(LONG_SHORT)
        if entry.mean_return_pct < 0.0
    }
    assert negative == {cell(3, 21)}


def test_the_market_row_is_carried_beside_the_cells():
    assert PUBLISHED_MARKET.sharpe == 1.01
    assert PUBLISHED_MARKET.mean_return_pct == pytest.approx(78.86)


def test_an_unknown_leg_is_refused_rather_than_defaulted():
    with pytest.raises(PublishedTableError):
        published_leg("long_flat")


def test_every_transcribed_row_has_a_standard_deviation_above_zero():
    """A zero standard deviation would be a column shift, not a portfolio."""
    for leg in LEGS:
        assert all(entry.std_pct > 0.0 for entry in published_leg(leg))


def test_every_drawdown_is_a_positive_percentage_at_or_below_a_hundred():
    """Their MDD column is positive-signed; a negative one here would be a sign
    convention silently flipped between the paper and the transcription."""
    for leg in LEGS:
        assert all(
            0.0 < entry.max_drawdown_pct <= 100.0 for entry in published_leg(leg)
        )
