"""The Grid as a fixed published fact, checked against the paper it comes from.

These are the only tests in the suite that assert a list of literals against a
citation rather than a computation. That is the point: the 21 pairs are Han,
Kang and Ryu's choice, and a run that quietly grids over 20 or 22 of them is not
the Replication Gate whatever else it does.
"""

import pytest

from crypto_momentum.sim.grid import (
    GRID_NAMES,
    HAN_KANG_RYU_21,
    GridCell,
    GridError,
    grid_named,
)

# Section 5.1, p.45: "Using this criterion, we choose 21 pairs: (1, 7), (1, 14),
# (1, 21), (3, 14), (3, 21), (3, 28), (5, 14), (5, 21), (5, 28), (7, 14),
# (7, 21), (7, 28), (14, 5), (14, 7), (14, 14), (21, 3), (21, 5), (21, 7),
# (28, 3), (28, 5), and (28, 7)."
PUBLISHED_PAIRS = [
    (1, 7), (1, 14), (1, 21),
    (3, 14), (3, 21), (3, 28),
    (5, 14), (5, 21), (5, 28),
    (7, 14), (7, 21), (7, 28),
    (14, 5), (14, 7), (14, 14),
    (21, 3), (21, 5), (21, 7),
    (28, 3), (28, 5), (28, 7),
]


def test_the_grid_is_the_paper_s_own_twenty_one_pairs():
    assert [(cell.lookback_days, cell.holding_days) for cell in HAN_KANG_RYU_21] == (
        PUBLISHED_PAIRS
    )


def test_no_pair_appears_twice():
    """A repeated pair would run one cell twice and count it as two."""
    assert len(set(HAN_KANG_RYU_21)) == 21


def test_a_cell_names_itself_by_its_two_periods():
    """The name becomes a config name and a results path, so it is stable."""
    assert GridCell(lookback_days=14, holding_days=7).name == "l14-h7"


def test_every_cell_name_is_distinct():
    assert len({cell.name for cell in HAN_KANG_RYU_21}) == 21


def test_a_grid_is_reached_by_the_name_a_config_writes():
    assert grid_named("han-kang-ryu-21") is HAN_KANG_RYU_21
    assert "han-kang-ryu-21" in GRID_NAMES


def test_an_unknown_grid_name_is_refused_rather_than_defaulted():
    with pytest.raises(GridError) as refusal:
        grid_named("some-other-21")

    assert "han-kang-ryu-21" in str(refusal.value)


def test_a_cell_must_have_positive_periods():
    """The pure layer refuses a nonsense cell, so the config loader is not the
    only thing standing between a typo and a run."""
    with pytest.raises(GridError):
        GridCell(lookback_days=0, holding_days=7)
    with pytest.raises(GridError):
        GridCell(lookback_days=14, holding_days=-1)
