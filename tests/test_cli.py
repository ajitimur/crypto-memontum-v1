"""The CLI reports a refusal as a message and an exit code, not a traceback."""

from datetime import date
from pathlib import Path

from crypto_momentum.cli import EXIT_REFUSED, main
from crypto_momentum.data.cmc_panel import CmcPanelStore
from crypto_momentum.trials import append_trial

BAD_CONFIG = """
name = "broken"

[data]
venue = "binance-spot"
symbol = "BTCUSDT"
interval = "1d"
start_month = "not-a-month"
end_month = "2021-02"

[strategy]
kind = "buy_and_hold"

[costs]
fee_bps_per_side = 40.44
slippage_bps_per_side = 5.0
"""


def test_an_invalid_config_refuses_with_a_message_and_a_non_zero_exit(tmp_path, capsys):
    config = tmp_path / "broken.toml"
    config.write_text(BAD_CONFIG)

    exit_code = main(["--repo-root", str(tmp_path), "run", str(config)])

    assert exit_code == EXIT_REFUSED
    assert "start_month" in capsys.readouterr().err


def test_trials_reports_how_many_configurations_were_tried(tmp_path, capsys):
    append_trial(tmp_path / "trials.jsonl", {"config_name": "one", "net_return": 0.1})
    append_trial(tmp_path / "trials.jsonl", {"config_name": "two", "net_return": 0.2})

    exit_code = main(["--repo-root", str(tmp_path), "trials"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "2 configurations tried" in out
    assert "one" in out and "two" in out


def test_trials_on_a_repository_that_has_never_run_reports_zero(tmp_path, capsys):
    assert main(["--repo-root", str(tmp_path), "trials"]) == 0
    assert "0 configurations tried" in capsys.readouterr().out


def test_a_single_trial_is_reported_in_the_singular(tmp_path, capsys):
    append_trial(tmp_path / "trials.jsonl", {"config_name": "only", "net_return": 0.1})

    assert main(["--repo-root", str(tmp_path), "trials"]) == 0
    assert "1 configuration tried" in capsys.readouterr().out


def test_pulling_a_panel_that_is_already_stored_fetches_nothing(tmp_path, capsys):
    """Re-running the build must not re-call CoinMarketCap (ADR-0008)."""
    panel = (
        Path(__file__).parent / "fixtures" / "coinmarketcap" / "cmc-panel-sample.csv"
    ).read_bytes()
    CmcPanelStore(tmp_path / "data" / "raw").write(
        panel, pulled_at_utc="2026-08-31T00:00:00Z", window_start=date(2017, 1, 1)
    )

    exit_code = main(["--repo-root", str(tmp_path), "pull-cmc-panel"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "already at" in out
    assert "nothing was fetched" in out


def test_a_missing_r_toolchain_refuses_with_a_message_not_a_traceback(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    exit_code = main(["--repo-root", str(tmp_path), "pull-cmc-panel"])

    assert exit_code == EXIT_REFUSED
    assert "Rscript" in capsys.readouterr().err
