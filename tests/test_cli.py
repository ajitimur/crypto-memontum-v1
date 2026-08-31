"""The CLI reports a refusal as a message and an exit code, not a traceback."""

from crypto_momentum.cli import EXIT_REFUSED, main
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
