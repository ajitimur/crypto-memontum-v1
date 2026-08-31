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


@pytest.mark.parametrize(
    "name", ["../../escape", "results/../../etc/passwd", "with space", "", ".."]
)
def test_a_name_that_could_escape_the_results_directory_is_rejected(tmp_path, name):
    """`name` is half the result key and becomes a path segment under results/."""
    text = VALID.replace('name = "skeleton-btcusdt-2021h1"', f'name = "{name}"')

    with pytest.raises(ConfigError, match="name"):
        load_config(write(tmp_path, text))


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(ConfigError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")


CROSS_SECTIONAL = """
name = "xsec-l14-h7"

[data]
venue = "binance-spot"
symbols = ["BTCUSDT", "ETHUSDT", "SRMUSDT"]
interval = "1d"
start_month = "2021-01"
end_month = "2021-03"

[strategy]
kind = "cross_sectional"
lookback_days = 14
holding_days = 7
quantile = 0.2

[universe]
bracket = "binance-full"
liquidity_floor_usd = 100000.0

[costs]
fee_bps_per_side = 40.44
slippage_bps_per_side = 5.0
"""


class TestCrossSectionalConfig:
    def test_loads_a_cross_section_of_symbols_and_its_knobs(self, tmp_path):
        cfg = load_config(write(tmp_path, CROSS_SECTIONAL))

        assert cfg.symbol is None
        assert cfg.symbols == ("BTCUSDT", "ETHUSDT", "SRMUSDT")
        assert cfg.universe_symbols == ("BTCUSDT", "ETHUSDT", "SRMUSDT")
        assert (cfg.lookback_days, cfg.holding_days, cfg.quantile) == (14, 7, 0.2)
        assert cfg.bracket == "binance-full"
        assert cfg.liquidity_floor_usd == 100000.0
        # Not stated in the config, so the policy layer's own default stands.
        assert cfg.liquidity_window_days == 30

    def test_a_single_asset_hold_still_reads_as_one_symbol(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))

        assert cfg.universe_symbols == ("BTCUSDT",)

    def test_a_cross_section_without_symbols_is_rejected(self, tmp_path):
        text = CROSS_SECTIONAL.replace(
            'symbols = ["BTCUSDT", "ETHUSDT", "SRMUSDT"]', 'symbol = "BTCUSDT"'
        )

        with pytest.raises(ConfigError, match="data.symbols"):
            load_config(write(tmp_path, text))

    def test_naming_both_shapes_at_once_is_rejected(self, tmp_path):
        text = CROSS_SECTIONAL.replace(
            "[strategy]", 'symbol = "BTCUSDT"\n\n[strategy]'
        )

        with pytest.raises(ConfigError, match="both"):
            load_config(write(tmp_path, text))

    def test_a_repeated_symbol_is_rejected_rather_than_weighted_twice(self, tmp_path):
        text = CROSS_SECTIONAL.replace(
            '["BTCUSDT", "ETHUSDT", "SRMUSDT"]', '["BTCUSDT", "ETHUSDT", "BTCUSDT"]'
        )

        with pytest.raises(ConfigError, match="more than once"):
            load_config(write(tmp_path, text))

    def test_a_knob_belonging_to_no_strategy_is_rejected(self, tmp_path):
        text = VALID.replace('kind = "buy_and_hold"', 'kind = "buy_and_hold"\nlookback_days = 14')

        with pytest.raises(ConfigError, match="lookback_days"):
            load_config(write(tmp_path, text))

    def test_a_cross_section_missing_a_knob_is_rejected(self, tmp_path):
        text = CROSS_SECTIONAL.replace("lookback_days = 14\n", "")

        with pytest.raises(ConfigError, match="strategy.lookback_days"):
            load_config(write(tmp_path, text))

    def test_a_quantile_outside_the_unit_interval_is_rejected(self, tmp_path):
        text = CROSS_SECTIONAL.replace("quantile = 0.2", "quantile = 1.4")

        with pytest.raises(ConfigError, match="quantile"):
            load_config(write(tmp_path, text))

    def test_an_unknown_bracket_is_rejected(self, tmp_path):
        text = CROSS_SECTIONAL.replace('bracket = "binance-full"', 'bracket = "kraken"')

        with pytest.raises(ConfigError, match="bracket"):
            load_config(write(tmp_path, text))

    def test_a_run_without_a_universe_section_takes_the_upper_bound(self, tmp_path):
        text = CROSS_SECTIONAL.replace(
            '[universe]\nbracket = "binance-full"\nliquidity_floor_usd = 100000.0\n\n', ""
        )

        cfg = load_config(write(tmp_path, text))

        assert cfg.bracket == "binance-full"
        # No floor asked for is no floor applied, and the result says which.
        assert cfg.liquidity_floor_usd is None
