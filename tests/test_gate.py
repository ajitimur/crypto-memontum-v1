"""The Replication Gate: four fixed criteria, and an explicit pass or fail.

The tolerances are ADR-0003's and are asserted here as literals, for the same
reason `test_grid.py` asserts the 21 pairs as literals: a gate whose bar moved
after somebody saw a result is not a gate. If one of these numbers changes, the
ADR changes first and this test changes with it.
"""

import math

import pytest

from crypto_momentum.gate import (
    FAITHFUL,
    LIQUIDATION_TOLERANCE,
    MIN_SIGN_AGREEMENT,
    MIN_SPEARMAN,
    PUBLISHED_BEST_NET_SHARPE,
    SHARPE_TOLERANCE,
    VENUE,
    GateError,
    describe_gap,
    evaluate_gate,
    spearman_rank_correlation,
)
from crypto_momentum.results import RECORDED, REFUSED, GridCellRecord
from crypto_momentum.sim.grid import HAN_KANG_RYU_21
from crypto_momentum.sim.published import (
    LONG_ONLY,
    LONG_SHORT,
    published_leg,
    published_liquidation_count,
)


# --- the tolerances, as ADR-0003 fixes them -------------------------------


def test_the_tolerances_are_the_ones_the_adr_fixed_in_advance():
    """ADR-0003's amendment table, transcribed. Moving one moves the ADR first."""
    assert MIN_SPEARMAN == 0.70
    assert LIQUIDATION_TOLERANCE == 2
    assert MIN_SIGN_AGREEMENT == 18
    assert SHARPE_TOLERANCE == 0.15
    assert PUBLISHED_BEST_NET_SHARPE == 1.28


# --- Spearman rank correlation --------------------------------------------


def test_spearman_of_a_series_with_itself_is_one():
    values = [0.1, 0.9, 0.4, 1.2, -0.3]
    assert spearman_rank_correlation(values, values) == pytest.approx(1.0)


def test_spearman_of_a_reversed_series_is_minus_one():
    values = [0.1, 0.9, 0.4, 1.2, -0.3]
    assert spearman_rank_correlation(values, list(reversed(sorted(values)))) is not None
    ascending = sorted(values)
    assert spearman_rank_correlation(
        ascending, list(reversed(ascending))
    ) == pytest.approx(-1.0)


def test_spearman_reads_ranks_and_not_levels():
    """A monotone rescaling of one side leaves the rank correlation untouched."""
    ours = [1.0, 2.0, 3.0, 4.0]
    theirs = [10.0, 1000.0, 100000.0, 10000000.0]
    assert spearman_rank_correlation(ours, theirs) == pytest.approx(1.0)


def test_spearman_averages_the_ranks_of_ties():
    """Two cells with the same Sharpe share a rank rather than taking an order.

    Breaking a tie by position would make the correlation depend on the order the
    grid happens to be written in.
    """
    ours = [1.0, 1.0, 2.0]
    theirs = [5.0, 5.0, 9.0]
    assert spearman_rank_correlation(ours, theirs) == pytest.approx(1.0)


def test_spearman_of_a_constant_side_is_undefined_rather_than_zero():
    """A side with no dispersion has no ranking to correlate with."""
    assert spearman_rank_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_spearman_needs_at_least_two_points():
    assert spearman_rank_correlation([1.0], [2.0]) is None


def test_spearman_refuses_two_series_of_different_lengths():
    with pytest.raises(GateError):
        spearman_rank_correlation([1.0, 2.0], [1.0, 2.0, 3.0])


# --- cells to evaluate ----------------------------------------------------


def cell_record(cell, *, sharpe, t_stat, liquidations=0):
    """One recorded cell, carrying only what the gate reads off it."""
    return GridCellRecord(
        lookback_days=cell.lookback_days,
        holding_days=cell.holding_days,
        name=cell.name,
        config_name=f"gate-{cell.name}",
        outcome=RECORDED,
        metrics={
            "sharpe_net": sharpe,
            "mean_log_return_t_stat": t_stat,
            "liquidation_count": liquidations,
            "liquidation_dates": ["2020-03-12T00:00:00Z"] * liquidations,
        },
    )


def cells_tracking(leg, *, sharpe_offset=0.0, t_sign=1.0):
    """21 cells whose Sharpes rank exactly as the published leg's do.

    The published Sharpe plus a constant, so the rank correlation is 1.0 by
    construction and a test can move one thing at a time.
    """
    return tuple(
        cell_record(
            published.cell,
            sharpe=published.sharpe + sharpe_offset,
            t_stat=t_sign * abs(published.mean_return_pct) / 30.0,
        )
        for published in published_leg(leg)
    )


def test_a_grid_that_tracks_the_published_leg_passes_the_faithful_gate():
    # The long-only column, because ADR-0004 makes that the leg our simulator
    # can actually produce — see the two tests below. Offset so the best cell
    # lands on 1.28 exactly: the published long-only best is 1.52, at (14, 5).
    cells = cells_tracking(LONG_ONLY, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.52)
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_ONLY)
    assert verdict.passes, verdict.to_dict()
    assert verdict.spearman == pytest.approx(1.0)


# --- the ADR-0003 / ADR-0004 disagreement, asserted rather than described --


def test_a_long_only_grid_cannot_clear_the_long_short_liquidation_criterion():
    """ADR-0004: an unlevered long-only book cannot breach a 100% loss.

    So it liquidates nought of 21, and nought is five away from the long-short
    column's five — outside ADR-0003's ±2 however good the pipeline is. The gate
    says so in a warning rather than quietly failing on it.
    """
    cells = cells_tracking(LONG_SHORT, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.31)
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert verdict.criterion("liquidation_count").observed == 0
    assert verdict.criterion("liquidation_count").passed is False
    assert not verdict.passes
    assert any("ADR-0004" in warning for warning in verdict.warnings)


def test_matching_the_published_five_costs_the_sign_agreement_criterion():
    """The other half of the same bind, and why the leg is the real question.

    A cell that liquidates has no finite mean log return and so no t-statistic
    to take a sign from. Producing their five liquidations therefore leaves at
    most 16 of 21 cells with a readable sign, under ADR-0003's bar of 18.
    """
    cells = _with_liquidations(
        cells_tracking(LONG_SHORT, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.31),
        first=5,
    )
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert verdict.criterion("liquidation_count").passed
    assert verdict.criterion("t_statistic_sign_agreement").observed <= 16
    assert verdict.criterion("t_statistic_sign_agreement").passed is False


def test_the_faithful_gate_fails_when_the_shape_does_not_track():
    """Rank correlation is the criterion; a grid ranked backwards fails it."""
    published = published_leg(LONG_SHORT)
    cells = tuple(
        cell_record(
            entry.cell,
            sharpe=-entry.sharpe,
            t_stat=abs(entry.mean_return_pct) / 30.0,
        )
        for entry in published
    )
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert not verdict.passes
    assert verdict.spearman == pytest.approx(-1.0)
    assert verdict.criterion("spearman_rank_correlation").passed is False


def test_the_gate_states_a_verdict_for_every_criterion():
    cells = cells_tracking(LONG_SHORT)
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert [criterion.name for criterion in verdict.criteria] == [
        "spearman_rank_correlation",
        "liquidation_count",
        "t_statistic_sign_agreement",
        "best_net_sharpe",
    ]


# --- the liquidation criterion --------------------------------------------


def test_the_liquidation_count_may_sit_two_either_side_of_the_published_one():
    published_count = published_liquidation_count(LONG_SHORT)
    assert published_count == 5

    for ours in (3, 5, 7):
        cells = _with_liquidations(cells_tracking(LONG_SHORT), first=ours)
        verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
        assert verdict.criterion("liquidation_count").passed, ours


def _with_liquidations(cells, *, first):
    """The same 21 cells, with the first `first` of them liquidated.

    A liquidated cell carries no t-statistic, because a path that compounds to
    nothing has no finite mean log return — that is what `report.summarise`
    produces, and a fixture that kept one would be testing against a cell the
    simulator cannot emit.
    """
    marked = list(cells)
    for index in range(first):
        marked[index] = cell_record(
            HAN_KANG_RYU_21[index],
            sharpe=marked[index].metrics["sharpe_net"],
            t_stat=None,
            liquidations=1,
        )
    return tuple(marked)


def test_a_liquidation_count_three_off_the_published_one_fails():
    cells = _with_liquidations(cells_tracking(LONG_SHORT), first=8)
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert verdict.criterion("liquidation_count").passed is False
    assert not verdict.passes


def test_the_long_only_leg_publishes_no_liquidations_at_all():
    """ADR-0004 leaves an unlevered long-only book unable to liquidate.

    Their long-only column cannot either — every cell earns a positive
    cumulative return — so a long-only run compared against the long-only column
    meets this criterion at nought against nought.
    """
    assert published_liquidation_count(LONG_ONLY) == 0
    verdict = evaluate_gate(cells_tracking(LONG_ONLY), run=FAITHFUL, leg=LONG_ONLY)
    assert verdict.criterion("liquidation_count").observed == 0
    assert verdict.criterion("liquidation_count").passed


# --- the sign agreement criterion -----------------------------------------


def test_sign_agreement_counts_cells_whose_t_statistic_matches_the_published_sign():
    cells = cells_tracking(LONG_SHORT, t_sign=1.0)
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    # Their (3, 21) long-short mean return is negative; every other cell of the
    # leg is positive, so an all-positive grid agrees on 20 of 21.
    assert verdict.criterion("t_statistic_sign_agreement").observed == 20
    assert verdict.criterion("t_statistic_sign_agreement").passed


def test_sign_agreement_fails_when_more_than_three_cells_disagree():
    published = published_leg(LONG_SHORT)
    cells = tuple(
        cell_record(
            entry.cell,
            sharpe=entry.sharpe,
            t_stat=-1.0 if index < 4 else 1.0,
        )
        for index, entry in enumerate(published)
    )
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    assert verdict.criterion("t_statistic_sign_agreement").observed < MIN_SIGN_AGREEMENT
    assert verdict.criterion("t_statistic_sign_agreement").passed is False


def test_a_cell_with_no_t_statistic_cannot_agree_in_sign():
    """A liquidated cell has no finite mean log return, so it has no sign.

    Counted as disagreement rather than skipped: skipping would shrink the
    denominator and let a grid of three readable cells clear an 18-of-21 bar.
    """
    published = published_leg(LONG_SHORT)
    cells = tuple(
        cell_record(entry.cell, sharpe=entry.sharpe, t_stat=None if index < 5 else 1.0)
        for index, entry in enumerate(published)
    )
    verdict = evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)
    criterion = verdict.criterion("t_statistic_sign_agreement")
    assert criterion.observed <= 16
    assert criterion.passed is False


# --- the level criterion, and where it does not apply ---------------------


def test_the_faithful_run_needs_its_best_cell_within_the_sharpe_tolerance():
    inside = cells_tracking(LONG_SHORT, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.31)
    assert evaluate_gate(inside, run=FAITHFUL, leg=LONG_SHORT).criterion(
        "best_net_sharpe"
    ).passed

    outside = cells_tracking(
        LONG_SHORT, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.31 - 0.3
    )
    verdict = evaluate_gate(outside, run=FAITHFUL, leg=LONG_SHORT)
    assert verdict.criterion("best_net_sharpe").passed is False
    assert not verdict.passes


def test_the_venue_run_is_not_held_to_the_level_at_all():
    """ADR-0003: different prices and a truncated window make a level match
    meaningless, and demanding one would just invite fitting."""
    cells = cells_tracking(
        LONG_ONLY, sharpe_offset=PUBLISHED_BEST_NET_SHARPE - 1.52 - 0.9
    )
    verdict = evaluate_gate(cells, run=VENUE, leg=LONG_ONLY)
    level = verdict.criterion("best_net_sharpe")
    assert level.required is None
    assert level.passed is True
    assert verdict.passes
    assert level.observed == pytest.approx(PUBLISHED_BEST_NET_SHARPE - 0.9)


def test_the_venue_run_still_reports_the_level_it_reached():
    """Not required is not the same as not measured — the number is the gap."""
    cells = cells_tracking(LONG_SHORT)
    verdict = evaluate_gate(cells, run=VENUE, leg=LONG_SHORT)
    assert verdict.best_net_sharpe == pytest.approx(1.31)


# --- incomplete grids ------------------------------------------------------


def test_a_refused_cell_fails_the_gate_rather_than_shrinking_the_grid():
    """21 cells or no verdict. Correlating 20 against 21 is not the comparison
    ADR-0003 fixed a tolerance on."""
    cells = list(cells_tracking(LONG_SHORT))
    cells[3] = GridCellRecord(
        lookback_days=cells[3].lookback_days,
        holding_days=cells[3].holding_days,
        name=cells[3].name,
        config_name=cells[3].config_name,
        outcome=REFUSED,
        refused="turnover_budget_breached",
        refused_reason="traded more than the ceiling allows",
    )
    verdict = evaluate_gate(tuple(cells), run=FAITHFUL, leg=LONG_SHORT)
    assert not verdict.passes
    assert verdict.spearman is None
    assert verdict.n_cells_compared == 20
    assert "21" in verdict.criterion("spearman_rank_correlation").note


def test_a_grid_of_the_wrong_cells_is_refused_outright():
    cells = cells_tracking(LONG_SHORT)[:20]
    with pytest.raises(GateError):
        evaluate_gate(cells, run=FAITHFUL, leg=LONG_SHORT)


def test_an_unknown_run_kind_is_refused():
    with pytest.raises(GateError):
        evaluate_gate(cells_tracking(LONG_SHORT), run="whichever", leg=LONG_SHORT)


# --- the gap between the two runs, which is a result in its own right ------


def test_the_gap_reports_how_far_the_venue_run_sits_from_the_faithful_one():
    faithful = evaluate_gate(cells_tracking(LONG_SHORT), run=FAITHFUL, leg=LONG_SHORT)
    venue = evaluate_gate(
        cells_tracking(LONG_SHORT, sharpe_offset=-0.4), run=VENUE, leg=LONG_SHORT
    )
    gap = describe_gap(faithful, venue)

    assert gap.best_net_sharpe_gap == pytest.approx(-0.4)
    assert gap.mean_absolute_sharpe_gap == pytest.approx(0.4)
    # The two runs rank the cells identically here; only the level moved.
    assert gap.spearman_between_runs == pytest.approx(1.0)
    assert gap.per_cell_sharpe_gap["l14-h7"] == pytest.approx(-0.4)


def test_the_gap_survives_a_venue_cell_that_produced_no_sharpe():
    faithful = evaluate_gate(cells_tracking(LONG_SHORT), run=FAITHFUL, leg=LONG_SHORT)
    venue_cells = list(cells_tracking(LONG_SHORT))
    venue_cells[0] = cell_record(HAN_KANG_RYU_21[0], sharpe=None, t_stat=None)
    venue = evaluate_gate(tuple(venue_cells), run=VENUE, leg=LONG_SHORT)

    gap = describe_gap(faithful, venue)
    assert gap.per_cell_sharpe_gap["l1-h7"] is None
    assert gap.n_cells_compared == 20
    assert math.isfinite(gap.mean_absolute_sharpe_gap)


def test_the_gap_refuses_two_verdicts_of_the_same_kind():
    """A Faithful Run against a Faithful Run measures nothing about the venue."""
    one = evaluate_gate(cells_tracking(LONG_SHORT), run=FAITHFUL, leg=LONG_SHORT)
    with pytest.raises(GateError):
        describe_gap(one, one)


def test_the_gap_refuses_two_verdicts_read_against_different_legs():
    faithful = evaluate_gate(cells_tracking(LONG_SHORT), run=FAITHFUL, leg=LONG_SHORT)
    venue = evaluate_gate(cells_tracking(LONG_ONLY), run=VENUE, leg=LONG_ONLY)
    with pytest.raises(GateError):
        describe_gap(faithful, venue)


# --- serialisation ---------------------------------------------------------


def test_a_verdict_serialises_with_its_pass_or_fail_stated_outright():
    verdict = evaluate_gate(cells_tracking(LONG_SHORT), run=FAITHFUL, leg=LONG_SHORT)
    payload = verdict.to_dict()
    assert payload["run"] == FAITHFUL
    assert payload["leg"] == LONG_SHORT
    assert payload["passes"] is False or payload["passes"] is True
    assert payload["citation"]
    assert len(payload["criteria"]) == 4
    assert all("required" in criterion for criterion in payload["criteria"])
