"""The CLI reports a refusal as a message and an exit code, not a traceback."""

from datetime import date
from pathlib import Path

from crypto_momentum.cli import (
    EXIT_REFUSED,
    _describe_divergence,
    _describe_hurdle,
    _describe_profitability,
    main,
)
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


def test_a_trial_that_survived_reports_liquidation_as_an_explicit_none(tmp_path, capsys):
    """ADR-0001: the reporting block says "none", it does not stay silent."""
    append_trial(tmp_path / "trials.jsonl", {"config_name": "one", "liquidation_dates": []})

    assert main(["--repo-root", str(tmp_path), "trials"]) == 0
    assert "liquidation=none" in capsys.readouterr().out


def test_a_liquidated_trial_reports_its_count_and_dates(tmp_path, capsys):
    append_trial(
        tmp_path / "trials.jsonl",
        {"config_name": "blown-up", "liquidation_dates": ["2021-07-07T00:00:00Z"]},
    )

    assert main(["--repo-root", str(tmp_path), "trials"]) == 0
    out = capsys.readouterr().out
    assert "liquidation=1 event" in out
    assert "2021-07-07T00:00:00Z" in out


def test_a_trial_logged_before_daily_marking_does_not_claim_it_survived(tmp_path, capsys):
    append_trial(tmp_path / "trials.jsonl", {"config_name": "older", "net_return": 0.1})

    assert main(["--repo-root", str(tmp_path), "trials"]) == 0
    assert "liquidation=not recorded" in capsys.readouterr().out


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


CLEARED = {
    "mean_log_return_daily_net": 0.002,
    "mean_log_return_t_stat": 4.2,
    "newey_west_lags": 3,
    "mean_return_daily_net": 0.0025,
    "mean_return_sign_divergence": False,
    "clears_profitability_bar": True,
}


def test_the_profitability_line_quotes_the_statistic_that_decides():
    """ADR-0002: the bar is t > 3.0 on the mean log return, and the line says so."""
    line = _describe_profitability(CLEARED)

    assert "t = 4.20" in line
    assert "3 lags" in line
    assert "clears" in line


def test_a_run_below_the_bar_is_not_described_as_having_cleared_it():
    line = _describe_profitability(
        {
            **CLEARED,
            "mean_log_return_t_stat": 2.5,
            "clears_profitability_bar": False,
        }
    )

    assert "below" in line
    assert "clears" not in line


def test_a_run_with_no_t_statistic_says_so_rather_than_quoting_nothing():
    line = _describe_profitability({**CLEARED, "mean_log_return_t_stat": None})

    assert "no t-statistic" in line


def test_diverging_means_are_called_out_on_their_own_line():
    """The divergence ADR-0002 asks for explicitly, not left to be inferred."""
    line = _describe_divergence(
        {
            **CLEARED,
            "mean_return_daily_net": 0.05,
            "mean_log_return_daily_net": -0.0525,
            "mean_return_sign_divergence": True,
        }
    )

    assert "disagree in sign" in line
    assert "0.05" in line and "-0.0525" in line


def test_means_that_agree_report_that_they_agree():
    assert "agree in sign" in _describe_divergence(CLEARED)


def test_a_liquidated_run_says_it_has_no_mean_log_return_to_compare():
    """The divergence line must not read as "they agree" on a path that has no
    mean log return at all — that is the loudest version of the diagnostic."""
    line = _describe_divergence(
        {
            **CLEARED,
            "mean_log_return_daily_net": None,
            "mean_return_sign_divergence": False,
        }
    )

    assert "liquidated" in line
    assert "agree in sign" not in line


def test_the_hurdle_line_names_the_condition_that_failed():
    line = _describe_hurdle(
        {
            "deployment_hurdle": {
                "sharpe_above_btc": True,
                "drawdown_no_worse_than_btc": False,
                "clears_profitability_bar": True,
                "clears": False,
            }
        }
    )

    assert "not cleared" in line
    assert "drawdown_no_worse_than_btc" in line
