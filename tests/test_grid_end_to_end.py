"""The whole Grid from one invocation, against the same synthetic venue.

What matters here is not any cell's number — the fixture's prices were chosen,
so no cell means anything on its own. It is that all 21 run, that each is filed
and counted separately, and that a cell which cannot produce a result does not
take the other twenty with it.
"""

import json

import pytest

from crypto_momentum.cli import EXIT_REFUSED, main
from crypto_momentum.config import ConfigError
from crypto_momentum.runner import run_config, run_grid
from crypto_momentum.sim.grid import HAN_KANG_RYU_21
from crypto_momentum.trials import read_trials
from synthetic_cross_section import RUN_AT, archive, workspace  # noqa: F401

GRID_TEXT = """
name = "xsec-grid-2021q1"

[data]
venue = "binance-spot"
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
interval = "1d"
start_month = "2021-01"
end_month = "2021-03"

[strategy]
kind = "cross_sectional"
grid = "han-kang-ryu-21"
quantile = 0.2

[universe]
bracket = "binance-full"

[costs]
model = "tokocrypto"
slippage_bps_per_side = 5.0
"""


@pytest.fixture
def config_path(workspace):
    path = workspace.repo_root / "configs" / "grid.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GRID_TEXT)
    return path


@pytest.fixture
def grid(workspace, config_path, archive):
    return run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)


def test_one_invocation_runs_every_cell_of_the_grid(grid):
    assert grid.grid == "han-kang-ryu-21"
    assert len(grid.cells) == 21
    assert [(cell.lookback_days, cell.holding_days) for cell in grid.cells] == [
        (cell.lookback_days, cell.holding_days) for cell in HAN_KANG_RYU_21
    ]


def test_the_grid_completes_without_manual_intervention(grid):
    """Every cell reaches an outcome. Some of them may be refusals — a cell that
    breaches its turnover budget or has too little history for its lookback is a
    finding, not an interruption — but none of them is left unattempted."""
    assert all(cell.outcome in ("recorded", "refused") for cell in grid.cells)
    assert grid.n_recorded + grid.n_refused == 21


def test_each_cell_is_filed_under_the_run_by_its_own_key(grid, workspace):
    directory = workspace.results_root / f"{grid.commit}-dirty" / "xsec-grid-2021q1"

    for cell in grid.cells:
        if cell.outcome != "recorded":
            continue
        path = directory / f"xsec-grid-2021q1-l{cell.lookback_days}-h{cell.holding_days}.json"
        assert path.exists(), f"{cell.name} was recorded but not filed"
        written = json.loads(path.read_text())
        assert written["config"]["lookback_days"] == cell.lookback_days
        assert written["config"]["holding_days"] == cell.holding_days
        # The cell knows it is one, and which grid of.
        assert written["grid"] == "han-kang-ryu-21"


def test_the_grid_writes_a_summary_beside_its_cells(grid, workspace):
    path = (
        workspace.results_root / f"{grid.commit}-dirty" / "xsec-grid-2021q1" / "grid.json"
    )

    assert path.exists()
    written = json.loads(path.read_text())
    assert written["grid"] == "han-kang-ryu-21"
    assert len(written["cells"]) == 21
    # The shape is the result, so the summary is what a reader opens first.
    assert written["n_recorded"] + written["n_refused"] == 21


def test_every_cell_is_appended_to_the_trials_log(grid, workspace):
    """All 21, including the ones we would rather forget. The count of
    configurations tried is only useful if nothing quietly stays out of it."""
    trials = read_trials(workspace.trials_path)

    assert len(trials) == 21
    assert {trial["config_name"] for trial in trials} == {
        f"xsec-grid-2021q1-{cell.name}" for cell in HAN_KANG_RYU_21
    }


def test_each_cell_counts_as_its_own_configuration_tried(grid, workspace):
    """21 cells is 21 configurations tried, not one. That is the multiple testing
    the reporting protocol asks to be counted, and it is the whole reason Han et
    al. hold their own results to t > 3.0."""
    trials = read_trials(workspace.trials_path)

    assert trials[-1]["configurations_tried"] == 21
    assert len({trial["configuration_fingerprint"] for trial in trials}) == 21


def test_a_refused_cell_does_not_stop_the_grid(workspace, config_path, archive):
    # A budget every cell breaches. The grid still runs all 21 and records each
    # refusal with the figure that caused it — a grid that stopped on the first
    # would leave twenty configurations uncounted.
    config_path.write_text(
        GRID_TEXT.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.0000001",
        )
    )

    grid = run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert grid.n_refused == 21
    assert grid.n_recorded == 0
    trials = read_trials(workspace.trials_path)
    assert len(trials) == 21
    assert all(trial["refused"] == "turnover_budget_breached" for trial in trials)
    # Nothing was produced, so nothing is filed but the summary of what was tried.
    assert [path.name for path in workspace.results_root.rglob("*.json")] == ["grid.json"]


def test_a_cell_the_window_is_too_short_for_is_recorded_rather_than_raised(
    workspace, config_path, archive
):
    """A 28-day lookback needs more window than a 1-day one. February 2021 is 28
    bars, so the long end of the grid cannot form a signal at all while the short
    end runs fine. That is a fact about those cells — reported beside the ones
    that ran, not raised over them."""
    config_path.write_text(
        GRID_TEXT.replace('start_month = "2021-01"', 'start_month = "2021-02"').replace(
            'end_month = "2021-03"', 'end_month = "2021-02"'
        )
    )

    grid = run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert len(grid.cells) == 21
    refused = [cell for cell in grid.cells if cell.outcome == "refused"]
    assert refused, "a one-month window cannot run the 28/28 end of the grid"
    assert all(cell.refused for cell in refused)
    assert len(read_trials(workspace.trials_path)) == 21


def test_the_cells_are_the_same_runs_a_hand_written_config_would_produce(
    workspace, config_path, archive
):
    """A cell differs from a config written for that pair in its name and nothing
    else. If it did differ, the Grid would not be evidence about the strategy."""
    grid = run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    by_hand = config_path.parent / "one-cell.toml"
    by_hand.write_text(
        GRID_TEXT.replace("xsec-grid-2021q1", "xsec-one-cell").replace(
            'grid = "han-kang-ryu-21"', "lookback_days = 14\nholding_days = 7"
        )
    )

    alone = run_config(by_hand, workspace, run_at_utc=RUN_AT, open_url=archive)

    cell = next(cell for cell in grid.cells if cell.name == "l14-h7")
    assert cell.outcome == "recorded"
    assert cell.metrics["net_return"] == pytest.approx(alone.metrics["net_return"])
    assert cell.metrics["sharpe_net"] == pytest.approx(alone.metrics["sharpe_net"])


def test_running_a_grid_config_as_a_single_run_is_refused_and_says_what_to_run(
    workspace, config_path, archive
):
    with pytest.raises(ConfigError, match="grid"):
        run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)


def test_running_a_single_config_as_a_grid_is_refused(workspace, config_path, archive):
    config_path.write_text(
        GRID_TEXT.replace('grid = "han-kang-ryu-21"', "lookback_days = 14\nholding_days = 7")
    )

    with pytest.raises(ConfigError, match="grid"):
        run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)


def test_the_archive_is_read_once_for_the_whole_grid(workspace, config_path, archive):
    """21 cells differ only in two knobs, so they share their bars, their
    point-in-time Universe and their vendor caps. Fetching per cell would be 21
    downloads of one window, and — worse — 21 chances for the Universe under one
    cell to differ from the Universe under another."""
    fetched: list[str] = []

    def counting(url: str) -> bytes:
        fetched.append(url)
        return archive(url)

    run_grid(config_path, workspace, run_at_utc=RUN_AT, open_url=counting)

    # Six symbols over three months, each a zip and a checksum, plus the
    # coverage listings. Every cell after the first reads what the first stored.
    assert len([url for url in fetched if url.endswith(".zip")]) == 18


def test_the_grid_summary_carries_the_shape_a_reader_judges_it_on(grid):
    """Not one cell's Sharpe: the whole column, with the liquidations and the
    refusals beside it, because the shape is what ADR-0003 compares."""
    recorded = [cell for cell in grid.cells if cell.outcome == "recorded"]

    for cell in recorded:
        assert "sharpe_net" in cell.metrics
        assert "mean_log_return_t_stat" in cell.metrics
        assert "liquidation_count" in cell.metrics
        assert cell.weekly_rebalance_turnover is not None


def test_the_cli_runs_a_grid_and_prints_a_row_for_every_cell(
    workspace, config_path, archive, monkeypatch, capsys
):
    """The one command of the acceptance criteria, through the entry point a
    researcher actually types."""
    monkeypatch.setattr(
        "crypto_momentum.cli.run_grid",
        lambda path, ws, **kwargs: run_grid(path, ws, **kwargs, open_url=archive),
    )

    exit_code = main(["--repo-root", str(workspace.repo_root), "grid", str(config_path)])

    assert exit_code == 0
    printed = capsys.readouterr()
    # The machine's copy stays alone on stdout.
    assert json.loads(printed.out)["n_cells"] == 21
    # And the researcher's table is a row per cell, in the published order.
    for cell in HAN_KANG_RYU_21:
        assert f"{cell.name:>10}" in printed.err
    assert "configurations tried: 21" in printed.err


def test_the_cli_refuses_a_grid_config_that_names_no_grid(workspace, config_path, capsys):
    config_path.write_text(
        GRID_TEXT.replace('grid = "han-kang-ryu-21"', "lookback_days = 14\nholding_days = 7")
    )

    exit_code = main(["--repo-root", str(workspace.repo_root), "grid", str(config_path)])

    assert exit_code == EXIT_REFUSED
    assert "momentum run" in capsys.readouterr().err
