"""Seam 4: the simulation core. Pure, daily-marked, costs applied as it trades."""

import ast
from pathlib import Path

import pandas as pd
import pytest

import crypto_momentum.sim as sim_package
from crypto_momentum.sim.buy_and_hold import NotEnoughBars, simulate_buy_and_hold


def bars_from(rows) -> pd.DataFrame:
    """Build a daily bar frame. One row is one UTC day, indexed on its open time."""
    index = pd.date_range("2021-01-01", periods=len(rows), freq="D", tz="UTC", name="ts_utc")
    return pd.DataFrame(
        [
            {"open": o, "high": max(o, c), "low": min(o, c), "close": c, "volume": 1.0}
            for o, c in rows
        ],
        index=index,
    )


def doubling_over_one_year() -> pd.DataFrame:
    """A Decision Bar followed by 365 marks compounding from 100 to exactly 200.

    The fill is the open of the first mark, so the held path doubles over exactly
    one year and must annualise to +100%.
    """
    path = [100.0 * (2 ** (day / 365)) for day in range(366)]
    rows = [(100.0, 100.0)] + [(path[day - 1], path[day]) for day in range(1, 366)]
    return bars_from(rows)


# Decision bar, then three days compounding at exactly +10%.
COMPOUNDING = bars_from([(100.0, 100.0), (100.0, 110.0), (110.0, 121.0), (121.0, 133.1)])


def test_the_position_fills_at_the_bar_after_the_decision_bar():
    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=0.0)

    assert result.decision_ts_utc == pd.Timestamp("2021-01-01T00:00:00Z")
    assert result.entry_ts_utc == pd.Timestamp("2021-01-02T00:00:00Z")
    assert result.entry_price == pytest.approx(100.0)
    assert result.exit_ts_utc == pd.Timestamp("2021-01-04T00:00:00Z")
    assert result.exit_price == pytest.approx(133.1)


def test_the_decision_bar_never_enters_the_result():
    """The signal is formed on the decision bar, so no return may be earned on it."""
    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=0.0)

    assert result.decision_ts_utc not in result.equity_net.index
    assert result.n_marks == 3


def test_a_move_on_the_decision_bar_cannot_change_the_result():
    """Feed a frame whose first row is the decision bar and vary only that row."""
    baseline = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=0.0)
    tampered = COMPOUNDING.copy()
    tampered.iloc[0, :] = [1.0, 1.0, 1.0, 1.0, 1.0]

    result = simulate_buy_and_hold(tampered, cost_bps_per_side=0.0)

    assert result.net_return == pytest.approx(baseline.net_return)
    assert result.entry_price == pytest.approx(baseline.entry_price)


def test_the_position_is_marked_on_every_bar_not_only_at_the_boundary():
    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=0.0)

    assert list(result.equity_net.index) == list(COMPOUNDING.index[1:])
    assert result.equity_net.tolist() == pytest.approx([1.10, 1.21, 1.331])


def test_costs_are_charged_on_the_buy_as_well_as_the_sell():
    """ADR-0007: the Indonesian PPh is a per-leg tax, so entry pays too."""
    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=100.0)

    # 0.99 paid in, 1.331x on price, 0.99 paid out.
    assert result.net_return == pytest.approx(0.99 * 1.331 * 0.99 - 1)
    assert result.gross_return == pytest.approx(0.331)


def test_the_cost_is_inside_the_path_not_a_haircut_on_the_answer():
    """Every daily mark is already net of the entry cost."""
    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=100.0)

    assert result.equity_net.iloc[0] == pytest.approx(0.99 * 1.10)
    assert result.equity_net.iloc[-1] == pytest.approx(0.99 * 1.331 * 0.99)


def test_a_doubling_over_exactly_one_year_annualises_to_one_hundred_percent():
    bars = doubling_over_one_year()

    result = simulate_buy_and_hold(bars, cost_bps_per_side=0.0)

    assert result.n_marks == 365
    assert result.ann_return_net == pytest.approx(1.0)
    # A perfectly constant daily growth rate has no dispersion.
    assert result.ann_vol_net == pytest.approx(0.0, abs=1e-12)
    assert result.sharpe_net is None


def test_cost_drag_is_the_annualised_gap_between_gross_and_net():
    bars = doubling_over_one_year()

    result = simulate_buy_and_hold(bars, cost_bps_per_side=100.0)

    assert result.ann_return_gross == pytest.approx(1.0)
    assert result.cost_drag_annualised == pytest.approx(
        result.ann_return_gross - result.ann_return_net
    )
    assert result.cost_drag_annualised > 0


def test_max_drawdown_reports_its_peak_and_trough_dates():
    bars = bars_from(
        [
            (100.0, 100.0),  # decision bar
            (100.0, 100.0),
            (100.0, 80.0),
            (80.0, 90.0),
            (90.0, 120.0),
        ]
    )

    result = simulate_buy_and_hold(bars, cost_bps_per_side=0.0)

    assert result.max_drawdown == pytest.approx(-0.20)
    assert result.max_drawdown_peak_ts_utc == pd.Timestamp("2021-01-02T00:00:00Z")
    assert result.max_drawdown_trough_ts_utc == pd.Timestamp("2021-01-03T00:00:00Z")


def test_mean_log_return_is_reported_because_it_is_the_profitability_bar():
    """ADR-0002. Three days of exactly +10% net of nothing."""
    import math

    result = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=0.0)

    assert result.mean_log_return_daily_net == pytest.approx(math.log(1.10))


def test_a_window_with_no_bar_after_the_decision_bar_cannot_be_filled():
    with pytest.raises(NotEnoughBars):
        simulate_buy_and_hold(COMPOUNDING.iloc[:1], cost_bps_per_side=0.0)


def test_the_simulation_does_not_mutate_the_caller_s_frame():
    before = COMPOUNDING.copy()

    simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=25.0)

    pd.testing.assert_frame_equal(COMPOUNDING, before)


def test_the_same_bars_always_produce_the_same_result():
    first = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=40.44)
    second = simulate_buy_and_hold(COMPOUNDING, cost_bps_per_side=40.44)

    assert first.net_return == second.net_return
    assert first.max_drawdown == second.max_drawdown


ALLOWED_SIM_IMPORTS = {"__future__", "dataclasses", "math", "typing", "numpy", "pandas"}

# Reaching the outside world without an import: these are the ways in.
FORBIDDEN_SIM_CALLS = {"open", "__import__", "eval", "exec", "compile", "input"}


def test_the_simulation_core_reaches_for_no_network_filesystem_or_clock():
    """The invariant, asserted structurally: nothing under `sim/` may import a
    module that could read the outside world, so a result depends only on the
    frame and the parameters it was given."""
    for module_path in Path(sim_package.__file__).parent.glob("*.py"):
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported = {(node.module or "").split(".")[0]}
            else:
                continue
            forbidden = imported - ALLOWED_SIM_IMPORTS
            assert not forbidden, f"{module_path.name} imports {sorted(forbidden)}"


def test_the_simulation_core_reaches_the_outside_world_by_no_other_route():
    """An allowlist of imports is not enough on its own: `open` and
    `__import__` are builtins and need no import statement."""
    for module_path in Path(sim_package.__file__).parent.glob("*.py"):
        tree = ast.parse(module_path.read_text())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not called & FORBIDDEN_SIM_CALLS, (
            f"{module_path.name} calls {sorted(called & FORBIDDEN_SIM_CALLS)}"
        )
