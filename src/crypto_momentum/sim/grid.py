"""The Grid: the (lookback, holding) pairs a strategy is judged across.

A result is read on the shape of the grid, not on one cell. A single cell that
looks good is the thing this literature produces by accident — 21 pairs tested,
the best one quoted — so the runner evaluates all of them or none.

The pairs are a published fact, not a knob. Han, Kang and Ryu select three
holding periods for each look-back period up to 28 days, taking the ones with
the highest t-statistics in their Table 13, and name the resulting 21 pairs in
section 5.1. Those are the pairs here, in their order. A config may choose *a*
grid by name; it may not write its own list, for the same reason
`costs.model` is a name rather than a number of basis points: a grid nobody
published is not something the Replication Gate can be run against.
"""

from __future__ import annotations

from dataclasses import dataclass


class GridError(Exception):
    """A grid was named that does not exist, or a cell that cannot be run."""


@dataclass(frozen=True)
class GridCell:
    """One cell: how far back the signal looks, and how long the book is held.

    Both are days. `holding_days` is also the rebalance interval — the book is
    re-formed when the holding period ends — so a cell is a complete statement
    of the strategy's cadence.
    """

    lookback_days: int
    holding_days: int

    def __post_init__(self) -> None:
        for label, value in (
            ("lookback_days", self.lookback_days),
            ("holding_days", self.holding_days),
        ):
            if value < 1:
                raise GridError(
                    f"a grid cell's {label} is a number of days and must be at "
                    f"least 1, got {value}"
                )

    @property
    def name(self) -> str:
        """The cell's identifier, used as a config name suffix and a results path.

        Short and filename-safe, because it is appended to the grid config's own
        name to key the cell's result — see `config.RunConfig.cell_config`.
        """
        return f"l{self.lookback_days}-h{self.holding_days}"


def _cells(*pairs: tuple[int, int]) -> tuple[GridCell, ...]:
    cells = tuple(
        GridCell(lookback_days=lookback, holding_days=holding)
        for lookback, holding in pairs
    )
    if len(set(cells)) != len(cells):
        raise GridError("a grid lists the same pair twice")
    return cells


# Han, Kang and Ryu (SSRN 4675565) section 5.1, p.45. Their Table 14 reports
# every one of these, which is what makes the grid comparable cell by cell.
HAN_KANG_RYU_21 = _cells(
    (1, 7), (1, 14), (1, 21),
    (3, 14), (3, 21), (3, 28),
    (5, 14), (5, 21), (5, 28),
    (7, 14), (7, 21), (7, 28),
    (14, 5), (14, 7), (14, 14),
    (21, 3), (21, 5), (21, 7),
    (28, 3), (28, 5), (28, 7),
)

GRIDS: dict[str, tuple[GridCell, ...]] = {"han-kang-ryu-21": HAN_KANG_RYU_21}
GRID_NAMES = tuple(GRIDS)


def grid_named(name: str) -> tuple[GridCell, ...]:
    """The cells of a published grid. Raises `GridError` on an unknown name.

    Refused rather than defaulted: a config that misspells its grid should not
    silently run a different one and report it under the name it asked for.
    """
    try:
        return GRIDS[name]
    except KeyError:
        raise GridError(
            f"unknown grid {name!r} — the grids this repo can run are "
            f"{', '.join(GRID_NAMES)}"
        ) from None
