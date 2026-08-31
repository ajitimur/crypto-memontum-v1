"""Seam 1: a TOML config file in, a validated RunConfig out."""

import pytest

from crypto_momentum.config import ConfigError, RunConfig, load_config

VALID = """
name = "skeleton-btcusdt-2021h1"

[data]
venue = "binance-spot"
symbol = "BTCUSDT"
interval = "1d"
start_month = "2021-01"
end_month = "2021-03"

[strategy]
kind = "buy_and_hold"

[costs]
fee_bps_per_side = 40.44
slippage_bps_per_side = 5.0
"""


def write(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_loads_a_valid_config(tmp_path):
    cfg = load_config(write(tmp_path, VALID))

    assert cfg == RunConfig(
        name="skeleton-btcusdt-2021h1",
        venue="binance-spot",
        symbol="BTCUSDT",
        interval="1d",
        start_month="2021-01",
        end_month="2021-03",
        strategy_kind="buy_and_hold",
        fee_bps_per_side=40.44,
        slippage_bps_per_side=5.0,
    )


def test_months_expand_to_the_exact_window_fetched(tmp_path):
    cfg = load_config(write(tmp_path, VALID))

    assert cfg.months() == ["2021-01", "2021-02", "2021-03"]


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    text = VALID.replace('kind = "buy_and_hold"', 'kind = "buy_and_hold"\nleverage = 3')

    with pytest.raises(ConfigError, match="unknown key.*leverage"):
        load_config(write(tmp_path, text))


def test_a_missing_section_names_the_section(tmp_path):
    text = VALID.split("[costs]")[0]

    with pytest.raises(ConfigError, match="costs"):
        load_config(write(tmp_path, text))


def test_a_malformed_month_names_the_field_and_the_expected_shape(tmp_path):
    text = VALID.replace('start_month = "2021-01"', 'start_month = "Jan 2021"')

    with pytest.raises(ConfigError, match="start_month.*YYYY-MM"):
        load_config(write(tmp_path, text))


def test_an_end_month_before_the_start_month_is_rejected(tmp_path):
    text = VALID.replace('end_month = "2021-03"', 'end_month = "2020-12"')

    with pytest.raises(ConfigError, match="end_month"):
        load_config(write(tmp_path, text))


def test_an_unsupported_strategy_is_rejected(tmp_path):
    text = VALID.replace('kind = "buy_and_hold"', 'kind = "cross_sectional_momentum"')

    with pytest.raises(ConfigError, match="buy_and_hold"):
        load_config(write(tmp_path, text))


def test_a_negative_cost_is_rejected(tmp_path):
    text = VALID.replace("fee_bps_per_side = 40.44", "fee_bps_per_side = -1.0")

    with pytest.raises(ConfigError, match="fee_bps_per_side"):
        load_config(write(tmp_path, text))


def test_a_wrong_type_is_rejected_rather_than_coerced(tmp_path):
    text = VALID.replace('symbol = "BTCUSDT"', "symbol = 12345")

    with pytest.raises(ConfigError, match="symbol"):
        load_config(write(tmp_path, text))


def test_the_loader_cannot_execute_code_from_a_config(tmp_path):
    """The config format is inert data. A payload that would execute under a
    pickle or a YAML tag loader is either a plain string or a syntax error here."""
    text = VALID.replace(
        'symbol = "BTCUSDT"',
        'symbol = "!!python/object/apply:os.system [\'touch pwned\']"',
    )

    with pytest.raises(ConfigError, match="symbol"):
        load_config(write(tmp_path, text))
    assert not (tmp_path / "pwned").exists()


def test_a_config_that_is_not_toml_at_all_is_rejected_as_malformed(tmp_path):
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write(tmp_path, "name: skeleton\ndata:\n  symbol: BTCUSDT\n"))


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(ConfigError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")
