"""The Replication Gate: a Grid read against the published one, pass or fail.

ADR-0003's amendment splits Step 1 into two runs and fixes four tolerances in
advance, so that no run produces a number to rationalise against. This module is
those tolerances and nothing else — it takes cells that have already been run and
says whether they clear the bar. It fetches nothing, simulates nothing, and holds
no default that a config could reach in and change.

The two runs and what each is for:

- **Faithful Run** — the paper's own vendor and window. Vendor differences are
  eliminated as an excuse, so this tests whether *our pipeline is correct*. It is
  held to all four criteria, the level included.
- **Venue Run** — the same configuration on the archive prices of the venue we
  would actually trade. Different prices and a truncated window make a level
  match meaningless, so the level is measured and reported but not required.
  Demanding one would just invite fitting.

`describe_gap` reports the distance between them, which ADR-0003 calls a result
in its own right: it measures how much of the published effect is an artefact of
cross-exchange aggregate pricing rather than something that could have been
traded on one venue. Nobody in the surveyed literature reports it.

**A known disagreement between two ADRs, stated rather than resolved here.**
ADR-0003 fixes these tolerances against Han et al.'s long-short leg — net Sharpe
1.28, their five liquidated portfolios. ADR-0004, accepted the same day, makes
this repo long-only spot and observes that an unlevered long-only book cannot
liquidate at all. So a long-only run compared against the long-short leg fails
the liquidation criterion structurally, at nought against five, whatever the
pipeline does. `leg` therefore selects which published column a verdict is read
against: it defaults to nothing and every caller states it, `evaluate_gate` warns
in the verdict when a long-only run is held to the long-short column, and the
choice is recorded in the result. It is not a knob on the strategy — the
published columns are all fixed facts about the same table — but resolving which
of the two ADRs governs is a decision for an ADR amendment, not for this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from crypto_momentum.results import GridCellRecord
from crypto_momentum.sim.grid import HAN_KANG_RYU_21, GridCell
from crypto_momentum.sim.published import (
    CITATION,
    LONG_SHORT,
    PUBLISHED_COST_BPS_PER_SIDE,
    PublishedCell,
    published_leg,
    published_liquidation_count,
)

# ADR-0003, amendment of 2026-08-30. Four numbers, fixed before any run, so that
# a result cannot be argued into clearing a bar that moved to meet it. 0.70 and
# ±0.15 are judgement calls and the ADR says so; what makes them useful is that
# they were written down first. Rank correlation of 0.70 across 21 cells is
# significant past p < 0.001, so it is a real bar rather than a formality.
MIN_SPEARMAN = 0.70
LIQUIDATION_TOLERANCE = 2
MIN_SIGN_AGREEMENT = 18
SHARPE_TOLERANCE = 0.15

# The level the Faithful Run is held to: Han, Kang and Ryu's (14, 7) long-short
# net Sharpe, which is the figure ADR-0003 names. Their *best* net long-short
# Sharpe is 1.31, at (1, 7); the ADR anchors on 1.28 and the anchor stays where
# it was put. The criterion applies it to our best cell, as issue #11 words it.
PUBLISHED_BEST_NET_SHARPE = 1.28

FAITHFUL = "faithful"
VENUE = "venue"
RUNS = (FAITHFUL, VENUE)

SPEARMAN = "spearman_rank_correlation"
LIQUIDATIONS = "liquidation_count"
SIGN_AGREEMENT = "t_statistic_sign_agreement"
BEST_NET_SHARPE = "best_net_sharpe"


class GateError(Exception):
    """The gate was asked for a verdict it cannot give, and gives none."""


@dataclass(frozen=True)
class Criterion:
    """One of the four, with what was measured and what it had to clear.

    `required` is `None` where the criterion is measured but not binding — the
    Venue Run's level — and `passed` is then `True`, because a criterion that
    does not apply cannot be failed. The distinction is kept in `required`
    rather than by dropping the criterion, so both runs report the same four
    lines and a reader can see the level the Venue Run reached beside the one
    the Faithful Run was held to.
    """

    name: str
    observed: float | int | None
    required: str | None
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.name,
            "observed": self.observed,
            "required": self.required,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class GateVerdict:
    """One run of the gate: the four criteria, and an explicit pass or fail.

    `passes` is a conjunction and is never a summary judgement. ADR-0003 expects
    this to be hard to clear — under both of Han et al.'s corrections none of
    their cross-sectional portfolios clears t > 3.0 on log returns — and a
    failure here is the pipeline working, not the project failing.
    """

    run: str
    leg: str
    criteria: tuple[Criterion, ...]
    n_cells_compared: int
    our_sharpes: tuple[tuple[str, float | None], ...]
    published_sharpes: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    def criterion(self, name: str) -> Criterion:
        for criterion in self.criteria:
            if criterion.name == name:
                return criterion
        raise GateError(f"the gate has no criterion named {name!r}")

    @property
    def spearman(self) -> float | None:
        observed = self.criterion(SPEARMAN).observed
        return None if observed is None else float(observed)

    @property
    def best_net_sharpe(self) -> float | None:
        observed = self.criterion(BEST_NET_SHARPE).observed
        return None if observed is None else float(observed)

    @property
    def sharpe_by_cell(self) -> dict[str, float | None]:
        return dict(self.our_sharpes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "leg": self.leg,
            "citation": CITATION,
            "published_cost_bps_per_side": PUBLISHED_COST_BPS_PER_SIDE,
            "passes": self.passes,
            "n_cells_compared": self.n_cells_compared,
            "n_cells_in_grid": len(HAN_KANG_RYU_21),
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "sharpe_net_by_cell": dict(self.our_sharpes),
            "published_sharpe_by_cell": dict(self.published_sharpes),
            "warnings": list(self.warnings),
        }


def evaluate_gate(
    cells: Sequence[GridCellRecord],
    *,
    run: str,
    leg: str = LONG_SHORT,
) -> GateVerdict:
    """Read a Grid's 21 cells against the published leg and return the verdict.

    `cells` must cover the published grid exactly. A grid of some other shape is
    refused rather than compared on what it has in common: ADR-0003 fixed a
    tolerance on the correlation across *these* 21 cells, and 0.70 across some
    other number of them is a different bar nobody agreed to.

    A cell that ran and produced nothing is a different matter. It stays in the
    grid, and the criteria fail on it — see the note each carries — because a
    refused cell is a finding about the configuration and dropping it would
    shrink the comparison to whatever happened to work.
    """
    if run not in RUNS:
        raise GateError(
            f"a gate run is one of {', '.join(RUNS)}, got {run!r} — the Faithful "
            "Run tests the pipeline and the Venue Run tests the prices, and "
            "which one this is decides whether the level is binding"
        )
    published = published_leg(leg)
    ours = _aligned_with_grid(cells)

    pairs = [
        (cell.metrics.get("sharpe_net"), entry.sharpe)
        for cell, entry in zip(ours, published, strict=True)
    ]
    comparable = [(mine, theirs) for mine, theirs in pairs if mine is not None]
    n_compared = len(comparable)

    criteria = (
        _spearman_criterion(comparable, n_compared),
        _liquidation_criterion(ours, leg),
        _sign_agreement_criterion(ours, published),
        _level_criterion(ours, run=run),
    )
    return GateVerdict(
        run=run,
        leg=leg,
        criteria=criteria,
        n_cells_compared=n_compared,
        our_sharpes=tuple(
            (cell.name, cell.metrics.get("sharpe_net")) for cell in ours
        ),
        published_sharpes=tuple(
            (entry.cell.name, entry.sharpe) for entry in published
        ),
        warnings=_warnings(ours, leg=leg),
    )


def _aligned_with_grid(
    cells: Sequence[GridCellRecord],
) -> tuple[GridCellRecord, ...]:
    """The given cells in the published grid's order, or a refusal.

    Positional pairing with the published table is only safe if both sides are
    the same 21 cells in the same order, so the alignment is done once here
    rather than assumed at four call sites.
    """
    by_cell: dict[GridCell, GridCellRecord] = {}
    for cell in cells:
        key = GridCell(
            lookback_days=cell.lookback_days, holding_days=cell.holding_days
        )
        if key in by_cell:
            raise GateError(
                f"cell {key.name} appears twice in the grid handed to the gate"
            )
        by_cell[key] = cell

    missing = [cell.name for cell in HAN_KANG_RYU_21 if cell not in by_cell]
    extra = sorted(cell.name for cell in set(by_cell) - set(HAN_KANG_RYU_21))
    if missing or extra:
        raise GateError(
            "the gate compares the whole published grid or none of it. Missing "
            f"{missing or 'nothing'}; not in the published grid {extra or 'nothing'}"
        )
    return tuple(by_cell[cell] for cell in HAN_KANG_RYU_21)


def _spearman_criterion(
    comparable: Sequence[tuple[float, float]], n_compared: int
) -> Criterion:
    """The shape criterion: do we rank the 21 cells as the paper ranks them.

    Rank rather than level, because the Venue Run's prices and window make a
    level comparison meaningless and the shape is what both runs share. It is
    also the harder thing to hit by accident: one cell can match by luck, an
    ordering of 21 cannot.
    """
    n_cells = len(HAN_KANG_RYU_21)
    if n_compared < n_cells:
        return Criterion(
            name=SPEARMAN,
            observed=None,
            required=f">= {MIN_SPEARMAN}",
            passed=False,
            note=(
                f"only {n_compared} of {n_cells} cells produced a net Sharpe, so "
                "there is no correlation across the published grid to take. The "
                "cells that produced nothing are the finding"
            ),
        )
    correlation = spearman_rank_correlation(
        [mine for mine, _ in comparable], [theirs for _, theirs in comparable]
    )
    if correlation is None:
        return Criterion(
            name=SPEARMAN,
            observed=None,
            required=f">= {MIN_SPEARMAN}",
            passed=False,
            note=(
                "one side of the comparison has no dispersion across the 21 "
                "cells, so it has no ranking to correlate with"
            ),
        )
    return Criterion(
        name=SPEARMAN,
        observed=correlation,
        required=f">= {MIN_SPEARMAN}",
        passed=correlation >= MIN_SPEARMAN,
        note=f"net Sharpe across {n_cells} cells, ranked against the published leg",
    )


def _liquidation_criterion(cells: Sequence[GridCellRecord], leg: str) -> Criterion:
    """Liquidations, ours against theirs, within ADR-0003's ±2.

    Counted as cells that liquidated rather than as liquidation events, which is
    the count the paper reports: five of their 21 long-short portfolios were
    wiped out during the sample.
    """
    theirs = published_liquidation_count(leg)
    ours = sum(1 for cell in cells if cell.liquidated)
    return Criterion(
        name=LIQUIDATIONS,
        observed=ours,
        required=f"{theirs} +/- {LIQUIDATION_TOLERANCE}",
        passed=abs(ours - theirs) <= LIQUIDATION_TOLERANCE,
        note=(
            f"{ours} of {len(cells)} cells liquidated against the {theirs} "
            f"published for the {leg} leg"
        ),
    )


def _sign_agreement_criterion(
    cells: Sequence[GridCellRecord], published: Sequence[PublishedCell]
) -> Criterion:
    """Sign agreement on the log-return t-statistics, 18 of 21 or better.

    Table 14 reports no t-statistic per cell — its columns are mean return,
    standard deviation, Sharpe, cumulative return and maximum drawdown — so the
    published side of this comparison is the sign of the annualised mean return,
    which is the sign a t-statistic on that mean would carry. Stated here
    because a reader who assumes a published t-statistic exists would be reading
    the criterion as stricter than it is.

    A cell with no t-statistic — a liquidated path has no finite mean log return
    to test — counts as disagreement rather than being dropped. Dropping it
    would shrink the denominator, and a grid of three readable cells would clear
    an 18-of-21 bar on three cells.
    """
    agreed = 0
    unreadable = 0
    for cell, entry in zip(cells, published, strict=True):
        t_statistic = cell.metrics.get("mean_log_return_t_stat")
        if t_statistic is None:
            unreadable += 1
            continue
        if (t_statistic > 0.0) == (entry.mean_return_pct > 0.0):
            agreed += 1
    n_cells = len(published)
    note = f"{agreed} of {n_cells} cells agree in sign with the published mean return"
    if unreadable:
        note += (
            f"; {unreadable} produced no t-statistic and so cannot agree — a "
            "path with no finite mean log return has no sign to compare"
        )
    return Criterion(
        name=SIGN_AGREEMENT,
        observed=agreed,
        required=f">= {MIN_SIGN_AGREEMENT} of {n_cells}",
        passed=agreed >= MIN_SIGN_AGREEMENT,
        note=note,
    )


def _level_criterion(cells: Sequence[GridCellRecord], *, run: str) -> Criterion:
    """The best cell's net Sharpe against 1.28 — binding on the Faithful Run only.

    Measured on both runs regardless, because the Venue Run's level is half of
    what `describe_gap` reports and an unmeasured number cannot be differenced.
    """
    sharpes = [
        cell.metrics["sharpe_net"]
        for cell in cells
        if cell.metrics.get("sharpe_net") is not None
    ]
    best = max(sharpes) if sharpes else None
    if run == VENUE:
        return Criterion(
            name=BEST_NET_SHARPE,
            observed=best,
            required=None,
            passed=True,
            note=(
                "not required on the Venue Run: ADR-0003 holds that different "
                "prices and a truncated window make a level match meaningless, "
                "and demanding one would invite fitting. Reported because the "
                "gap between the two runs is itself a result"
            ),
        )
    if best is None:
        return Criterion(
            name=BEST_NET_SHARPE,
            observed=None,
            required=(
                f"{PUBLISHED_BEST_NET_SHARPE} +/- {SHARPE_TOLERANCE}"
            ),
            passed=False,
            note="no cell produced a net Sharpe, so there is no level to compare",
        )
    return Criterion(
        name=BEST_NET_SHARPE,
        observed=best,
        required=f"{PUBLISHED_BEST_NET_SHARPE} +/- {SHARPE_TOLERANCE}",
        passed=abs(best - PUBLISHED_BEST_NET_SHARPE) <= SHARPE_TOLERANCE,
        note=(
            f"best of the {len(cells)} cells against Han, Kang and Ryu's "
            f"(14, 7) net Sharpe of {PUBLISHED_BEST_NET_SHARPE}"
        ),
    )


def _warnings(cells: Sequence[GridCellRecord], *, leg: str) -> tuple[str, ...]:
    """What a reader has to know to read the verdict, said in the verdict.

    Only the ADR disagreement so far. It is a warning rather than a refusal
    because the comparison ADR-0003 literally specifies is still the one it
    specifies, and a gate that refused to run it would be resolving an ADR
    conflict by fiat.
    """
    if leg != LONG_SHORT:
        return ()
    if any(cell.liquidated for cell in cells):
        return ()
    return (
        "this verdict reads a run against Han, Kang and Ryu's long-short leg, "
        f"whose published liquidation count is {published_liquidation_count(leg)}. "
        "ADR-0004 makes this repo long-only spot and notes that an unlevered "
        "long-only book cannot breach a 100% loss, so no such run can produce a "
        "liquidation and the liquidation criterion is unmeetable against this "
        "leg. Their long-only column liquidates nought of 21 and is the "
        "like-for-like comparison; which leg the gate should bind to is an ADR "
        "amendment, not a run-time choice",
    )


# --- the gap between the two runs -----------------------------------------


@dataclass(frozen=True)
class RunGap:
    """How far the Venue Run sits from the Faithful Run, cell by cell.

    ADR-0003 calls this a result in its own right rather than an error: it
    measures how much of the published effect is an artefact of cross-exchange
    aggregate pricing versus what could actually have been traded on one venue.
    Our survey found nobody reporting it, and both runs existing makes it nearly
    free.

    Every gap is signed venue minus faithful, so a negative number is the venue
    doing worse. `spearman_between_runs` is the two runs against each other
    rather than against the paper: it separates "the venue moved the level" from
    "the venue moved which cells win", which are different findings.
    """

    faithful: GateVerdict
    venue: GateVerdict
    leg: str
    per_cell_sharpe_gap: dict[str, float | None]
    n_cells_compared: int
    best_net_sharpe_gap: float | None
    mean_absolute_sharpe_gap: float
    spearman_between_runs: float | None
    liquidation_count_gap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "leg": self.leg,
            "signed": "venue minus faithful",
            "n_cells_compared": self.n_cells_compared,
            "best_net_sharpe_gap": self.best_net_sharpe_gap,
            "mean_absolute_sharpe_gap": self.mean_absolute_sharpe_gap,
            "spearman_between_runs": self.spearman_between_runs,
            "liquidation_count_gap": self.liquidation_count_gap,
            "sharpe_gap_by_cell": self.per_cell_sharpe_gap,
        }


def describe_gap(faithful: GateVerdict, venue: GateVerdict) -> RunGap:
    """The distance between the two runs, reported rather than explained away."""
    if faithful.run != FAITHFUL or venue.run != VENUE:
        raise GateError(
            "the gap is between one Faithful Run and one Venue Run, and it is "
            f"what separates vendor prices from venue prices; got {faithful.run} "
            f"and {venue.run}"
        )
    if faithful.leg != venue.leg:
        raise GateError(
            f"the two runs are read against different published legs "
            f"({faithful.leg} and {venue.leg}), so their difference is a "
            "difference of columns and not of prices"
        )

    ours = faithful.sharpe_by_cell
    theirs = venue.sharpe_by_cell
    per_cell: dict[str, float | None] = {}
    paired: list[tuple[float, float]] = []
    for cell in HAN_KANG_RYU_21:
        mine, venue_side = ours.get(cell.name), theirs.get(cell.name)
        if mine is None or venue_side is None:
            per_cell[cell.name] = None
            continue
        per_cell[cell.name] = venue_side - mine
        paired.append((mine, venue_side))

    gaps = [gap for gap in per_cell.values() if gap is not None]
    faithful_best = faithful.best_net_sharpe
    venue_best = venue.best_net_sharpe
    return RunGap(
        faithful=faithful,
        venue=venue,
        leg=faithful.leg,
        per_cell_sharpe_gap=per_cell,
        n_cells_compared=len(gaps),
        best_net_sharpe_gap=(
            None
            if faithful_best is None or venue_best is None
            else venue_best - faithful_best
        ),
        # Zero on an empty comparison, and `n_cells_compared` beside it says the
        # comparison was empty. A NaN here would propagate into a reported number.
        mean_absolute_sharpe_gap=(
            sum(abs(gap) for gap in gaps) / len(gaps) if gaps else 0.0
        ),
        spearman_between_runs=spearman_rank_correlation(
            [mine for mine, _ in paired], [venue_side for _, venue_side in paired]
        ),
        liquidation_count_gap=(
            int(venue.criterion(LIQUIDATIONS).observed or 0)
            - int(faithful.criterion(LIQUIDATIONS).observed or 0)
        ),
    )


# --- rank correlation ------------------------------------------------------


def spearman_rank_correlation(
    ours: Sequence[float], theirs: Sequence[float]
) -> float | None:
    """Spearman's rho: Pearson correlation of the two series' ranks.

    Ties take the average of the ranks they span, so two cells with the same
    Sharpe share a rank rather than being ordered by where they sit in the list.
    Breaking a tie by position would make the correlation depend on the order
    the grid happens to be written in, which is the one thing a correlation
    across a published grid must not depend on.

    `None` rather than a number when there is nothing to correlate: fewer than
    two points, or a side with no dispersion. A constant side has no ranking,
    and reporting 0.0 for it would read as "ranked independently" rather than
    "not ranked at all".
    """
    if len(ours) != len(theirs):
        raise GateError(
            f"a rank correlation needs two series of the same length, got "
            f"{len(ours)} and {len(theirs)}"
        )
    if len(ours) < 2:
        return None
    return _pearson(_average_ranks(ours), _average_ranks(theirs))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1 upward, with tied values sharing the average of their ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def _pearson(ours: Sequence[float], theirs: Sequence[float]) -> float | None:
    mean_ours = sum(ours) / len(ours)
    mean_theirs = sum(theirs) / len(theirs)
    ours_centred = [value - mean_ours for value in ours]
    theirs_centred = [value - mean_theirs for value in theirs]
    variance_ours = sum(value * value for value in ours_centred)
    variance_theirs = sum(value * value for value in theirs_centred)
    if variance_ours <= 0.0 or variance_theirs <= 0.0:
        return None
    covariance = sum(
        mine * yours for mine, yours in zip(ours_centred, theirs_centred, strict=True)
    )
    return covariance / (variance_ours * variance_theirs) ** 0.5
