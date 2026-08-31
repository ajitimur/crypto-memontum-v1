"""Seam 1: a TOML config file in, a validated RunConfig out."""

import pytest

from crypto_momentum.config import ConfigError, RunConfig, load_config
from crypto_momentum.costs import TOKOCRYPTO, TURNOVER_CEILING_WEEKLY

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
model = "tokocrypto"
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
        cost_model=TOKOCRYPTO,
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


def test_a_negative_slippage_assumption_is_rejected(tmp_path):
    text = VALID.replace("slippage_bps_per_side = 5.0", "slippage_bps_per_side = -1.0")

    with pytest.raises(ConfigError, match="slippage_bps_per_side"):
        load_config(write(tmp_path, text))


class TestTheCostModel:
    """A run names a venue's cost structure; it does not invent basis points."""

    def test_the_named_model_carries_its_components(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))

        assert cfg.cost_model is TOKOCRYPTO
        assert cfg.cost_model.tax_bps_per_side == pytest.approx(21.0)
        # The model's 40.44 plus the run's own 5bp slippage assumption.
        assert cfg.cost_bps_per_side == pytest.approx(45.44)

    def test_the_papers_model_is_selectable_for_a_replication(self, tmp_path):
        text = VALID.replace('model = "tokocrypto"', 'model = "paper"')

        cfg = load_config(write(tmp_path, text))

        assert cfg.cost_model.name == "paper"
        assert cfg.cost_bps_per_side == pytest.approx(20.0)

    def test_a_config_cannot_write_its_own_cost_in_basis_points(self, tmp_path):
        # The venue's cost structure is an ADR decision. A loose number here
        # would be a result quoted net of a venue nobody chose.
        text = VALID.replace('model = "tokocrypto"', "fee_bps_per_side = 12.0")

        with pytest.raises(ConfigError, match="unknown key.*fee_bps_per_side"):
            load_config(write(tmp_path, text))

    def test_an_unknown_venue_names_the_models_that_exist(self, tmp_path):
        text = VALID.replace('model = "tokocrypto"', 'model = "binance"')

        with pytest.raises(ConfigError, match="paper, tokocrypto"):
            load_config(write(tmp_path, text))

    def test_there_is_no_funding_knob_to_set(self, tmp_path):
        # v1 is unlevered long-only spot per ADR-0004 and holds no perpetual
        # position. A funding rate here would read as a modelled assumption.
        text = VALID.replace(
            'model = "tokocrypto"', 'model = "tokocrypto"\nfunding_rate_8h = 0.0001'
        )

        with pytest.raises(ConfigError, match="unknown key.*funding_rate_8h"):
            load_config(write(tmp_path, text))


class TestTheTurnoverBudget:
    """ADR-0007's ceiling, enforced at load time — before any bar is fetched."""

    def test_omitting_the_budget_takes_the_full_ceiling(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))

        assert cfg.turnover_budget_weekly == TURNOVER_CEILING_WEEKLY

    def test_a_tighter_budget_than_the_ceiling_is_allowed(self, tmp_path):
        text = VALID.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.1",
        )

        cfg = load_config(write(tmp_path, text))

        assert cfg.turnover_budget_weekly == pytest.approx(0.1)

    def test_a_budget_above_the_ceiling_is_rejected_before_the_run(self, tmp_path):
        # The literature's ~68% weekly turnover, declared honestly. ADR-0007
        # refuses it at the loader rather than executing and reporting it.
        text = VALID.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.68",
        )

        with pytest.raises(ConfigError, match="above the 25% weekly"):
            load_config(write(tmp_path, text))

    def test_the_ceiling_itself_is_exactly_on_the_line_and_allowed(self, tmp_path):
        text = VALID.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.25",
        )

        assert load_config(write(tmp_path, text)).turnover_budget_weekly == 0.25

    def test_a_budget_of_nothing_is_rejected(self, tmp_path):
        text = VALID.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.0",
        )

        with pytest.raises(ConfigError, match="must be above 0"):
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
model = "tokocrypto"
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


GRID = """
name = "xsec-grid-2021h1"

[data]
venue = "binance-spot"
symbols = ["BTCUSDT", "ETHUSDT", "SRMUSDT"]
interval = "1d"
start_month = "2021-01"
end_month = "2021-03"

[strategy]
kind = "cross_sectional"
grid = "han-kang-ryu-21"
quantile = 0.2

[costs]
model = "tokocrypto"
slippage_bps_per_side = 5.0
"""


class TestGridConfig:
    def test_a_grid_config_names_a_published_grid_and_no_single_cell(self, tmp_path):
        cfg = load_config(write(tmp_path, GRID))

        assert cfg.is_grid
        assert cfg.grid == "han-kang-ryu-21"
        # The two knobs the grid stands in for are unset, because no one cell is
        # the run: a config holding both a grid and a lookback would run one of
        # them and record the other.
        assert cfg.lookback_days is None
        assert cfg.holding_days is None

    def test_the_grid_expands_to_twenty_one_runnable_cells(self, tmp_path):
        cfg = load_config(write(tmp_path, GRID))

        cells = cfg.cell_configs()

        assert len(cells) == 21
        assert [(c.lookback_days, c.holding_days) for c in cells[:2]] == [(1, 7), (1, 14)]

    def test_a_cell_carries_the_whole_config_but_its_own_two_knobs(self, tmp_path):
        cfg = load_config(write(tmp_path, GRID))

        cell = cfg.cell_configs()[13]

        assert (cell.lookback_days, cell.holding_days) == (14, 7)
        assert cell.name == "xsec-grid-2021h1-l14-h7"
        assert cell.symbols == cfg.symbols
        assert cell.quantile == cfg.quantile
        assert cell.turnover_budget_weekly == cfg.turnover_budget_weekly
        # The cell is a run, not a grid: it cannot expand again.
        assert cell.grid is None
        assert not cell.is_grid

    def test_a_plain_cross_sectional_config_is_not_a_grid(self, tmp_path):
        cfg = load_config(write(tmp_path, CROSS_SECTIONAL))

        assert cfg.grid is None
        assert not cfg.is_grid
        assert cfg.cell_configs() == ()

    def test_naming_a_grid_and_a_lookback_at_once_is_rejected(self, tmp_path):
        text = GRID.replace(
            'grid = "han-kang-ryu-21"', 'grid = "han-kang-ryu-21"\nlookback_days = 14'
        )

        with pytest.raises(ConfigError, match="lookback_days"):
            load_config(write(tmp_path, text))

    def test_an_unknown_grid_names_the_grids_that_exist(self, tmp_path):
        text = GRID.replace("han-kang-ryu-21", "my-own-21")

        with pytest.raises(ConfigError, match="han-kang-ryu-21"):
            load_config(write(tmp_path, text))

    def test_a_grid_on_a_single_asset_hold_is_rejected(self, tmp_path):
        text = VALID.replace('kind = "buy_and_hold"', 'kind = "buy_and_hold"\ngrid = "han-kang-ryu-21"')

        with pytest.raises(ConfigError, match="grid"):
            load_config(write(tmp_path, text))

    def test_a_name_too_long_to_suffix_a_cell_onto_is_rejected_up_front(self, tmp_path):
        # The cell name becomes a path under results/, so the refusal belongs at
        # load time — not on the nineteenth cell of a grid already half run.
        text = GRID.replace('name = "xsec-grid-2021h1"', f'name = "{"g" * 78}"')

        with pytest.raises(ConfigError, match="cell"):
            load_config(write(tmp_path, text))
