"""The published table the Replication Gate is read against.

Han, Kang and Ryu (SSRN 4675565) Table 14, panel (b) — the 21 cells of
`grid.HAN_KANG_RYU_21` after their 15bp transaction cost, over 2017-01-01 to
2023-08-28. Transcribed here rather than kept in prose, because the gate's
verdict is a comparison against these numbers and a comparison against numbers
nobody can check is not a gate.

**Panel (b) only, and deliberately.** Panel (a) is gross, and `CLAUDE.md`'s Net
invariant is that a figure without its cost assumption is not a result. A gross
column sitting here would be available to compare a net run against, and the
comparison would flatter the run by exactly their 15bp.

**All three legs, and also deliberately.** ADR-0004 makes this repo long-only,
so `LONG_ONLY` is the column our simulator is like-for-like with; ADR-0003 fixes
the gate's tolerances against `LONG_SHORT`. The two disagree — see the `leg`
argument to `gate.evaluate_gate` and the module docstring above it — and that
disagreement is only visible if both columns are here to be read. `SHORT_ONLY` is carried because it is what makes the other two
add up, and because a leg quietly omitted is a leg nobody can check.
"""

from __future__ import annotations

from dataclasses import dataclass

from crypto_momentum.sim.grid import HAN_KANG_RYU_21, GridCell

CITATION = (
    "Han, Kang and Ryu (SSRN 4675565), Table 14 panel (b): cross-sectional "
    "momentum portfolios after a 15bp transaction cost, 2017-01-01 to "
    "2023-08-28"
)

# Their cost assumption, stated beside their numbers because ours is not the
# same one: ADR-0007 prices a Tokocrypto round trip at 80.88bp against their 30.
PUBLISHED_COST_BPS_PER_SIDE = 15.0

LONG_ONLY = "long_only"
SHORT_ONLY = "short_only"
LONG_SHORT = "long_short"
LEGS = (LONG_ONLY, SHORT_ONLY, LONG_SHORT)

# A cumulative return of exactly -100% is a portfolio that reached zero, which
# is CONTEXT.md's Liquidation. Their five liquidated long-short cells — (3, 21),
# (3, 28), (5, 21), (5, 28) and (7, 28), named in their section 5.2.1 — are
# exactly the long-short cells at this value, so the count the gate compares
# against is derived from the transcribed column rather than typed in beside it
# as a second number that could disagree with it.
LIQUIDATED_CUM_RETURN_PCT = -100.0


class PublishedTableError(Exception):
    """The transcribed table does not line up with the grid it stands for."""


@dataclass(frozen=True)
class PublishedCell:
    """One leg of one cell, as the paper's row reads left to right.

    Percentages are the paper's own: `mean_return_pct` and `std_pct` are
    annualised, `cum_return_pct` is the cumulative return over the whole sample
    and `max_drawdown_pct` is positive-signed, as their column is.
    """

    lookback_days: int
    holding_days: int
    leg: str
    mean_return_pct: float
    std_pct: float
    sharpe: float
    cum_return_pct: float
    max_drawdown_pct: float

    @property
    def cell(self) -> GridCell:
        return GridCell(
            lookback_days=self.lookback_days, holding_days=self.holding_days
        )

    @property
    def liquidated(self) -> bool:
        """Whether the published portfolio was wiped out during their sample."""
        return self.cum_return_pct <= LIQUIDATED_CUM_RETURN_PCT

    @property
    def log_return_is_positive(self) -> bool:
        """The sign a mean *log* return of this cell would carry.

        Read off the cumulative return and not off `mean_return_pct`, because
        those two disagree in sign on seven of the long-short cells and the
        disagreement is the whole point of ADR-0002. Cumulative return is the
        compounded quantity: sign(log(1 + cum)) is sign(cum), so a portfolio that
        ended below where it started has a negative mean log return whatever its
        arithmetic mean says.

        Their (5, 21) long-short cell is the case that makes this matter — mean
        return +194.01% against a cumulative -100.0%. Taking the sign off the
        mean would call that cell positive, which is exactly the fat-tailed path
        that tests significant while losing money, and exactly the reading the
        log-return bar exists to prevent.
        """
        return self.cum_return_pct > 0.0


# (j, k) -> (L row, S row, LS row), each (Mean, Std, Sharpe, Cum, MDD).
_TABLE_14B: tuple[
    tuple[
        tuple[int, int],
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
        tuple[float, float, float, float, float],
    ],
    ...,
] = (
    ((1, 7),   (105.03, 96.20, 1.09, 4830.9, 95.6),   (-54.43, 99.58, -0.55, -99.9, 100.0),     (82.42, 62.95, 1.31, 7227.1, 47.3)),
    ((1, 14),  (107.21, 95.85, 1.12, 5751.9, 95.5),   (-78.02, 108.92, -0.72, -100.0, 100.0),   (49.18, 61.55, 0.80, 687.6, 62.8)),
    ((1, 21),  (101.21, 94.41, 1.07, 4091.4, 95.0),   (-1357.04, 263.45, -5.15, -100.0, 100.0), (24.11, 104.39, 0.23, -87.2, 99.1)),
    ((3, 14),  (106.11, 95.02, 1.12, 5609.2, 95.6),   (-734.00, 198.59, -3.70, -100.0, 100.0),  (39.36, 76.10, 0.52, 90.9, 97.5)),
    ((3, 21),  (84.12, 93.32, 0.90, 1309.0, 95.7),    (-577.53, 361.17, -1.60, -100.0, 100.0),  (-351.57, 290.98, -1.21, -100.0, 100.0)),
    ((3, 28),  (95.79, 93.93, 1.02, 2913.1, 94.4),    (-658.42, 235.38, -2.80, -100.0, 100.0),  (3.52, 328.68, 0.01, -100.0, 100.0)),
    ((5, 14),  (119.89, 103.52, 1.16, 9696.5, 95.7),  (-62.30, 121.70, -0.51, -100.0, 100.0),   (108.66, 93.52, 1.16, 6359.1, 91.5)),
    ((5, 21),  (106.98, 101.44, 1.05, 4371.4, 94.1),  (38.12, 899.16, 0.04, -100.0, 100.0),     (194.01, 366.65, 0.53, -100.0, 100.0)),
    ((5, 28),  (104.48, 102.79, 1.02, 3452.8, 94.4),  (-698.21, 218.38, -3.20, -100.0, 100.0),  (94.18, 220.08, 0.43, -100.0, 100.0)),
    ((7, 14),  (135.35, 113.83, 1.19, 17485.2, 95.9), (-9.50, 117.64, -0.08, -99.7, 100.0),     (110.19, 101.45, 1.09, 7758.8, 89.9)),
    ((7, 21),  (117.74, 113.45, 1.04, 5288.1, 94.6),  (-997.99, 222.40, -4.49, -100.0, 100.0),  (72.74, 118.07, 0.62, 68.4, 97.7)),
    ((7, 28),  (117.47, 113.91, 1.03, 5073.3, 95.1),  (-636.56, 252.14, -2.52, -100.0, 100.0),  (126.56, 292.82, 0.43, -100.0, 100.0)),
    ((14, 5),  (164.99, 108.84, 1.52, 126039.6, 93.3), (36.81, 180.60, 0.20, -99.2, 99.9),      (181.03, 147.31, 1.23, 93788.6, 92.1)),
    ((14, 7),  (149.95, 105.74, 1.42, 55166.6, 93.9), (77.73, 231.28, 0.34, -99.2, 99.9),       (174.73, 136.93, 1.28, 101218.3, 86.5)),
    ((14, 14), (125.43, 103.35, 1.21, 12736.2, 94.6), (-534.08, 237.92, -2.24, -100.0, 100.0),  (146.57, 162.83, 0.90, 4553.6, 96.6)),
    ((21, 3),  (150.40, 109.26, 1.38, 45749.4, 95.2), (-41.68, 109.11, -0.38, -99.9, 100.0),    (104.40, 102.25, 1.02, 3304.4, 90.9)),
    ((21, 5),  (129.52, 106.14, 1.22, 13904.9, 95.5), (-48.24, 114.62, -0.42, -100.0, 100.0),   (103.39, 110.37, 0.94, 1870.3, 95.7)),
    ((21, 7),  (116.23, 104.41, 1.11, 6302.3, 95.3),  (-581.68, 257.71, -2.26, -100.0, 100.0),  (79.07, 112.00, 0.71, 147.3, 98.5)),
    ((28, 3),  (121.62, 108.25, 1.12, 6987.8, 94.3),  (-63.04, 107.82, -0.58, -100.0, 100.0),   (57.98, 98.06, 0.59, 87.4, 95.9)),
    ((28, 5),  (113.67, 105.60, 1.08, 4806.5, 94.3),  (-51.96, 115.73, -0.45, -100.0, 100.0),   (62.38, 97.77, 0.64, 163.5, 96.0)),
    ((28, 7),  (97.38, 104.08, 0.94, 1704.0, 94.4),   (-41.58, 122.00, -0.34, -100.0, 100.0),   (41.32, 97.29, 0.42, -37.7, 97.9)),
)

# Their market portfolio, from the same panel's last row. The gate does not test
# against it — ADR-0005's hurdle is what a run is read against, and it is built
# from *our* Universe — but it is the scale their Sharpes are large or small
# relative to, and a reader comparing 1.28 to nothing is comparing it to nothing.
PUBLISHED_MARKET = PublishedCell(
    lookback_days=0,
    holding_days=0,
    leg=LONG_SHORT,
    mean_return_pct=78.86,
    std_pct=77.80,
    sharpe=1.01,
    cum_return_pct=2333.3,
    max_drawdown_pct=89.1,
)


def _build() -> dict[str, dict[GridCell, PublishedCell]]:
    by_leg: dict[str, dict[GridCell, PublishedCell]] = {leg: {} for leg in LEGS}
    for (lookback, holding), *rows in _TABLE_14B:
        cell = GridCell(lookback_days=lookback, holding_days=holding)
        for leg, (mean, std, sharpe, cum, mdd) in zip(LEGS, rows, strict=True):
            by_leg[leg][cell] = PublishedCell(
                lookback_days=lookback,
                holding_days=holding,
                leg=leg,
                mean_return_pct=mean,
                std_pct=std,
                sharpe=sharpe,
                cum_return_pct=cum,
                max_drawdown_pct=mdd,
            )
    return by_leg


TABLE_14B: dict[str, dict[GridCell, PublishedCell]] = _build()


def published_leg(leg: str) -> tuple[PublishedCell, ...]:
    """One leg of the published table, in the grid's own published order.

    The order is `HAN_KANG_RYU_21`'s and never a ranking, for the reason
    `results.GridRecord` gives: a grid sorted best-first is read as a ranking,
    and reading a ranking is the mistake the Grid exists to prevent. Ordering it
    here rather than at each call site is also what lets the gate pair our cells
    with theirs positionally without either side re-deriving the order.
    """
    if leg not in TABLE_14B:
        raise PublishedTableError(
            f"unknown leg {leg!r} — the published legs are {', '.join(LEGS)}"
        )
    return tuple(TABLE_14B[leg][cell] for cell in HAN_KANG_RYU_21)


def published_liquidation_count(leg: str) -> int:
    """How many of the leg's 21 cells the paper reports as liquidated.

    Five for the long-short leg, which is the number ADR-0003 fixes the gate's
    ±2 tolerance around, and zero for the long-only leg, which is the leg
    ADR-0004 leaves our simulator able to produce.
    """
    return sum(1 for published in published_leg(leg) if published.liquidated)


def _assert_covers_grid() -> None:
    """That the transcription is the grid, exactly — no cell missing or extra.

    Checked at import rather than in a test, because a table that has drifted
    from the grid it stands for makes every gate verdict built on it wrong, and
    the failure should stop the process rather than wait for someone to run the
    suite.
    """
    transcribed = {
        GridCell(lookback_days=lookback, holding_days=holding)
        for (lookback, holding), *_ in _TABLE_14B
    }
    published_only = transcribed - set(HAN_KANG_RYU_21)
    grid_only = set(HAN_KANG_RYU_21) - transcribed
    if published_only or grid_only:
        raise PublishedTableError(
            "the transcribed Table 14 and han-kang-ryu-21 disagree about which "
            f"cells exist: only in the table {sorted(c.name for c in published_only)}, "
            f"only in the grid {sorted(c.name for c in grid_only)}"
        )
    if len(_TABLE_14B) != len(HAN_KANG_RYU_21):
        raise PublishedTableError(
            f"the transcribed Table 14 has {len(_TABLE_14B)} rows for a grid of "
            f"{len(HAN_KANG_RYU_21)} cells"
        )


_assert_covers_grid()
