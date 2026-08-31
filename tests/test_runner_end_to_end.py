"""The whole path: a TOML config in, a recorded result and a trials line out."""

import json
import subprocess

import pandas as pd
import pytest

from crypto_momentum.config import ConfigError, RunConfig
from crypto_momentum.data.binance_archive import ChecksumMismatch
from crypto_momentum.runner import Workspace, _btc_over_the_same_window, run_config
from crypto_momentum.sim.marking import mark_daily
from crypto_momentum.sim.report import summarise
from crypto_momentum.trials import read_trials

RUN_AT = "2026-08-31T09:00:00Z"

CONFIG_TEXT = """
name = "skeleton-btcusdt-2021jan-feb"

[data]
venue = "binance-spot"
symbol = "BTCUSDT"
interval = "1d"
start_month = "2021-01"
end_month = "2021-02"

[strategy]
kind = "buy_and_hold"

[costs]
fee_bps_per_side = 40.44
slippage_bps_per_side = 5.0
"""


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(args, cwd=root, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (root / "README.md").write_text("skeleton\n")
    run("git", "add", ".")
    run("git", "commit", "-qm", "first")
    return Workspace.under(root)


@pytest.fixture
def config_path(workspace):
    path = workspace.repo_root / "configs" / "skeleton.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEXT)
    return path


@pytest.fixture
def archive(recorded_archive_file):
    """An opener serving the recorded archive bytes. Never touches the network."""
    served = {}
    for month in ("2021-01", "2021-02"):
        filename, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", month)
        base = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/{filename}"
        served[base] = payload
        served[base + ".CHECKSUM"] = checksum_text.encode()

    def open_url(url: str) -> bytes:
        from crypto_momentum.data.fetch import ArchiveUnavailable

        if url not in served:
            raise ArchiveUnavailable(f"404 for {url}")
        return served[url]

    return open_url


def test_one_config_end_to_end_produces_a_recorded_result(workspace, config_path, archive):
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    result_path = workspace.results_root / f"{record.commit}-dirty" / (
        "skeleton-btcusdt-2021jan-feb.json"
    )
    assert result_path.exists()
    written = json.loads(result_path.read_text())
    assert written["config"]["name"] == "skeleton-btcusdt-2021jan-feb"
    assert written["run_at_utc"] == RUN_AT


def test_the_result_matches_the_prices_in_the_recorded_archive(
    workspace, config_path, archive
):
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    # Read off the recorded CSVs: the fill is the open of 2021-01-02, the exit
    # is the close of 2021-02-28, and the position is never entered on the
    # Decision Bar of 2021-01-01.
    assert record.metrics["entry_ts_utc"] == "2021-01-02T00:00:00Z"
    assert record.metrics["entry_price"] == pytest.approx(29331.70)
    assert record.metrics["exit_ts_utc"] == "2021-02-28T00:00:00Z"
    assert record.metrics["exit_price"] == pytest.approx(45135.66)
    assert record.metrics["gross_return"] == pytest.approx(45135.66 / 29331.70 - 1)
    assert record.metrics["n_marks"] == 58


def test_the_recorded_result_carries_the_liquidation_line_of_the_reporting_block(
    workspace, config_path, archive
):
    """ADR-0001. Long-only spot cannot liquidate, and the result says so explicitly."""
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert record.metrics["liquidation_count"] == 0
    assert record.metrics["liquidation_dates"] == []
    assert record.metrics["exit_reason"] == "window_end"


def test_the_run_is_marked_on_every_day_of_the_window_not_only_at_its_boundaries(
    workspace, config_path, archive
):
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    # 59 bars from 2021-01-01 to 2021-02-28, less the Decision Bar.
    assert record.window["n_bars"] == 59
    assert record.metrics["n_marks"] == record.window["n_bars"] - 1


def test_every_run_is_appended_to_the_trials_log(workspace, config_path, archive):
    run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    run_config(config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=archive)

    trials = read_trials(workspace.trials_path)

    assert len(trials) == 2
    assert [trial["run_at_utc"] for trial in trials] == [RUN_AT, "2026-09-01T09:00:00Z"]
    assert trials[0]["config_name"] == "skeleton-btcusdt-2021jan-feb"
    assert trials[0]["net_return"] == pytest.approx(trials[1]["net_return"])


def test_the_result_is_net_of_the_configured_cost_on_both_legs(
    workspace, config_path, archive
):
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    cost = (40.44 + 5.0) * 1e-4
    gross_growth = 45135.66 / 29331.70
    assert record.metrics["net_return"] == pytest.approx(
        gross_growth * (1 - cost) ** 2 - 1
    )
    assert record.metrics["cost_bps_per_side"] == pytest.approx(45.44)


def test_a_second_run_reuses_the_stored_raw_window_rather_than_refetching(
    workspace, config_path, archive
):
    run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    manifest_before = (
        workspace.raw_root
        / "binance/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2021-01.zip.manifest.json"
    ).read_text()

    def refuse(url: str) -> bytes:
        raise AssertionError(f"the archive was hit again for {url}")

    run_config(config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=refuse)

    assert (
        workspace.raw_root
        / "binance/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2021-01.zip.manifest.json"
    ).read_text() == manifest_before


def test_derived_data_can_be_deleted_and_the_next_run_rebuilds_it(
    workspace, config_path, archive
):
    first = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    import shutil

    shutil.rmtree(workspace.derived_root)

    def refuse(url: str) -> bytes:
        raise AssertionError("rebuilding derived data must not re-fetch raw data")

    second = run_config(
        config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=refuse
    )

    assert second.metrics["net_return"] == pytest.approx(first.metrics["net_return"])


def test_an_invalid_config_is_rejected_and_nothing_is_run(workspace, config_path, archive):
    config_path.write_text(CONFIG_TEXT.replace('kind = "buy_and_hold"', 'kind = "martingale"'))

    with pytest.raises(ConfigError):
        run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert read_trials(workspace.trials_path) == []
    assert not workspace.results_root.exists()
    assert not workspace.raw_root.exists()


def test_a_corrupted_download_stops_the_run_before_anything_is_stored(
    workspace, config_path, recorded_archive_file
):
    filename, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", "2021-01")
    corrupted = {
        f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/{filename}": (
            payload[:-1] + bytes([payload[-1] ^ 0xFF])
        ),
        f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/{filename}.CHECKSUM": (
            checksum_text.encode()
        ),
    }

    with pytest.raises(ChecksumMismatch):
        run_config(
            config_path, workspace, run_at_utc=RUN_AT, open_url=lambda url: corrupted[url]
        )

    assert not any(workspace.raw_root.rglob("*.zip"))
    assert read_trials(workspace.trials_path) == []


def test_the_config_is_fingerprinted_so_an_edited_config_is_a_different_trial(
    workspace, config_path, archive
):
    first = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    config_path.write_text(CONFIG_TEXT.replace("40.44", "10.0"))
    second = run_config(
        config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=archive
    )

    assert first.config_sha256 != second.config_sha256
    assert second.metrics["net_return"] > first.metrics["net_return"]


def test_the_recorded_result_carries_the_whole_reporting_block(
    workspace, config_path, archive
):
    """`docs/agents/quant-research.md` names what a result reports. All of it lands."""
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert set(record.metrics) >= {
        "ann_return_net",
        "ann_vol_net",
        "sharpe_net",
        "mean_return_daily_net",
        "mean_log_return_daily_net",
        "mean_log_return_t_stat",
        "newey_west_lags",
        "mean_return_sign_divergence",
        "clears_profitability_bar",
        "max_drawdown",
        "max_drawdown_peak_ts_utc",
        "max_drawdown_trough_ts_utc",
        "liquidation_count",
        "liquidation_dates",
        "cost_drag_annualised",
        "cost_drag_as_fraction_of_gross",
    }
    # 58 marks, so floor(4 * (58/100) ** (2/9)) lags.
    assert record.metrics["newey_west_lags"] == 3


def test_the_result_counts_the_configurations_tried_to_reach_it(
    workspace, config_path, archive
):
    """The count the protocol requires beside every quoted number, on the result
    itself rather than only in the log a reader would have to go and count.

    Running the same bytes twice is one configuration tried, not two: a count
    that rose on every re-run would read as a search that never happened."""
    first = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    second = run_config(
        config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=archive
    )

    assert first.configurations_tried == 1
    assert second.configurations_tried == 1
    assert second.trials_recorded == 2
    assert read_trials(workspace.trials_path)[-1]["configurations_tried"] == 1


def test_an_edited_config_is_another_configuration_tried(
    workspace, config_path, archive
):
    """The count is of configurations, so editing one and running it again
    raises it — that is the search the protocol asks to be declared."""
    run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)
    config_path.write_text(CONFIG_TEXT.replace("40.44", "10.0"))

    second = run_config(
        config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=archive
    )

    assert second.configurations_tried == 2


def test_btc_buy_and_hold_is_computed_on_every_run(workspace, config_path, archive):
    """ADR-0005: the hurdle is not a thing you opt into, it is on every result."""
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    btc = record.benchmarks["btc_buy_and_hold"]
    assert btc["symbol"] == "BTCUSDT"
    # The run is BTC itself here, so the hurdle is the run: it ties, and a tie
    # is not "above", which is what makes the comparison strict.
    assert btc["net_return"] == pytest.approx(record.metrics["net_return"])
    assert record.benchmarks["deployment_hurdle"]["sharpe_above_btc"] is False
    assert record.benchmarks["deployment_hurdle"]["clears"] is False


def test_a_single_asset_run_says_why_it_has_no_market_portfolio(
    workspace, config_path, archive
):
    """A cap-weighted market needs a Universe, and a one-symbol hold has none.
    The result says that rather than leaving the field silently absent."""
    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    market = record.benchmarks["cap_weighted_market"]
    assert market["computed"] is False
    assert market["reason"]


def test_the_hurdle_is_held_over_the_run_s_window_and_not_a_day_longer():
    """A run that ended early must not be read against a Bitcoin that kept
    compounding after it: ADR-0005 compares over the same window, both edges."""
    index = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC", name="ts_utc")
    bars = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [1.0] * 10,
        },
        index=index,
    )
    # A strategy that stopped on the fifth bar, whatever the archive published after.
    ended_early = summarise(
        mark_daily(pd.Series(0.0, index=index[1:5], dtype=float)),
        decision_ts_utc=index[0],
        entry_price=1.0,
        exit_price=1.0,
        cost_bps_per_side=0.0,
    )

    btc, reason = _btc_over_the_same_window(
        ended_early,
        config=RunConfig(
            name="held-over-the-same-window",
            venue="binance-spot",
            symbol="BTCUSDT",
            interval="1d",
            start_month="2021-01",
            end_month="2021-01",
            strategy_kind="buy_and_hold",
            fee_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
        ),
        # The bars are already loaded, so nothing is read from the workspace.
        workspace=None,
        bars_by_symbol={"BTCUSDT": bars},
        run_at_utc=RUN_AT,
        open_url=None,
    )

    assert reason is None
    assert btc.decision_ts_utc == ended_early.decision_ts_utc
    assert btc.exit_ts_utc == ended_early.exit_ts_utc
    assert btc.n_marks == ended_early.n_marks
