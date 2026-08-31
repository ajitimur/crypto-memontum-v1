"""A cross-sectional config in, a recorded result out.

The synthetic archive and vendor panel this runs against live in
`synthetic_cross_section`, shared with the grid suite, which puts the same
wiring under a whole Grid.
"""

import json
from urllib.parse import parse_qs, urlparse

import pytest

from crypto_momentum.runner import run_config
from crypto_momentum.sim.cross_sectional import TurnoverBudgetBreached
from crypto_momentum.trials import read_trials
from synthetic_cross_section import RUN_AT, archive, workspace  # noqa: F401

CONFIG_TEXT = """
name = "xsec-l14-h7-2021q1"

[data]
venue = "binance-spot"
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
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

[costs]
model = "tokocrypto"
slippage_bps_per_side = 5.0
"""



@pytest.fixture
def config_path(workspace):
    path = workspace.repo_root / "configs" / "xsec.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEXT)
    return path


@pytest.fixture
def record(workspace, config_path, archive):
    return run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)


def test_a_cross_sectional_config_produces_a_recorded_result(record, workspace):
    path = workspace.results_root / f"{record.commit}-dirty" / "xsec-l14-h7-2021q1.json"

    assert path.exists()
    written = json.loads(path.read_text())
    assert written["config"]["strategy_kind"] == "cross_sectional"
    assert written["config"]["lookback_days"] == 14
    assert written["portfolio"]["strategy"] == "cross_sectional"


def test_the_run_holds_the_top_quintile_weekly(record):
    portfolio = record.portfolio

    # The window is 2021-01-01 to 2021-03-31. A 14-day lookback puts the first
    # Decision Bar on 01-16, and they run weekly to 03-27 — the last one with a
    # bar after it to fill on.
    assert portfolio["n_rebalances"] == 11
    # Six names at a fifth, rounded up.
    assert portfolio["mean_n_positions"] == pytest.approx(2.0)
    assert portfolio["n_rebalances_held_cash"] == 0


def test_the_portfolio_is_long_only_unlevered_and_ungated(record):
    portfolio = record.portfolio

    assert portfolio["long_only"] is True
    assert portfolio["levered"] is False
    assert portfolio["trend_gate"] is False
    assert portfolio["max_gross_exposure"] <= 1.0 + 1e-9


def test_the_run_is_marked_daily_and_reports_its_liquidation_line(record):
    # Every day from the first fill on 01-17 to the end of the window.
    assert record.metrics["n_marks"] == 74
    assert record.metrics["liquidation_count"] == 0
    assert record.metrics["liquidation_dates"] == []


def test_the_result_records_the_universe_it_was_drawn_from(record):
    universe = record.window["universe"]

    assert universe["n_symbols_in_universe"] == 6
    assert universe["exclusion_list"]["version"]
    assert universe["bracket"]["selected"] == "binance-full"
    # Nothing was asked for, so nothing was applied — and the result says which.
    assert universe["liquidity_floor"]["applied"] is False


def test_costs_are_charged_inside_the_path_not_deducted_from_it(record):
    assert record.metrics["cost_bps_per_side"] == pytest.approx(45.44)
    assert record.metrics["net_return"] < record.metrics["gross_return"]
    assert record.metrics["cost_drag_annualised"] > 0.0


def test_the_result_says_which_cost_world_it_was_priced_in(record):
    costs = record.costs

    # The components, not just the total: a net figure that cannot be read back
    # against ADR-0007's table is not a result under the Net invariant.
    assert costs["cost_model"] == "tokocrypto"
    assert costs["fee_bps_per_side"] == pytest.approx(15.0)
    assert costs["tax_bps_per_side"] == pytest.approx(21.0)
    assert costs["levy_bps_per_side"] == pytest.approx(4.44)
    assert costs["tax_charged_on_buys"] is True
    assert costs["slippage_bps_per_side"] == pytest.approx(5.0)
    assert costs["total_bps_per_side"] == pytest.approx(45.44)


def test_the_result_carries_no_funding_model(record):
    # ADR-0004: unlevered long-only spot holds no perpetual position, so there is
    # no funding leg. Stated in words, and deliberately not as a zero rate — a
    # rate would read as a funding model that happened to price at nothing.
    assert "funding_bps" not in record.costs
    assert record.costs["funding"].startswith("none")
    assert "perpetual" in record.costs["funding"]


def test_cost_drag_is_reported_annualised_and_against_gross(record):
    metrics = record.metrics

    # Both readings, per the reporting protocol. The annualised figure
    # extrapolates a 74-day window and so is large; the fraction is the
    # scale-free one the one-third ceiling applies to.
    assert metrics["cost_drag_annualised"] == pytest.approx(
        metrics["ann_return_gross"] - metrics["ann_return_net"]
    )
    assert metrics["cost_drag_as_fraction_of_gross"] == pytest.approx(
        metrics["cost_drag_annualised"] / metrics["ann_return_gross"]
    )


def test_rebalance_turnover_is_reported_against_the_ceiling(record):
    portfolio = record.portfolio

    # The rebalance is weekly, so the per-rebalance figure and the weekly one
    # coincide here — which is exactly why a fortnightly cell needs the second.
    assert portfolio["weekly_rebalance_turnover"] == pytest.approx(
        portfolio["mean_rebalance_turnover"]
    )
    assert portfolio["turnover_ceiling_weekly"] == 0.25
    assert portfolio["turnover_budget_weekly"] == 0.25
    # The run got as far as being recorded, so it stayed inside its budget.
    assert portfolio["weekly_rebalance_turnover"] <= 0.25
    # And it is far inside it, for a reason worth naming rather than pinning to a
    # number: six names at a fifth is a two-name book, and over Q1 2021 the same
    # two lead the ranking almost every week. What little turns over is drift in
    # their value weights, not the book changing hands. A real cross-section is
    # the case the ceiling exists for, and this fixture is not one.
    assert portfolio["weekly_rebalance_turnover"] < 0.01


def test_a_run_refused_on_turnover_is_still_counted_as_a_configuration_tried(
    workspace, config_path, archive
):
    # A budget of 0.1% weekly against a run that turns over more. The refusal is
    # the point — but the reporting protocol asks for the count of
    # configurations tried, and this one was tried. It is counted, with the
    # figure that refused it, and no result file is written because there is no
    # result to write.
    config_path.write_text(
        CONFIG_TEXT.replace(
            "slippage_bps_per_side = 5.0",
            "slippage_bps_per_side = 5.0\nturnover_budget_weekly = 0.001",
        )
    )

    with pytest.raises(TurnoverBudgetBreached):
        run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    trial = read_trials(workspace.trials_path)[0]
    assert trial["config_name"] == "xsec-l14-h7-2021q1"
    assert trial["refused"] == "turnover_budget_breached"
    assert trial["turnover_budget_weekly"] == 0.001
    assert trial["weekly_rebalance_turnover"] > 0.001
    # No metrics: nothing was produced, and an absent net_return is honest where
    # a zero would read as a run that broke even.
    assert "net_return" not in trial
    assert not any(workspace.results_root.rglob("*.json"))


def test_the_trials_log_carries_the_shape_of_the_portfolio(workspace, config_path, archive):
    run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    trial = read_trials(workspace.trials_path)[0]

    assert trial["n_symbols"] == 6
    assert trial["n_rebalances"] == 11
    assert trial["mean_rebalance_turnover"] > 0.0


def test_a_second_run_reuses_the_stored_raw_window(workspace, config_path, archive):
    run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    def listing_only(url: str) -> bytes:
        if "prefix" in parse_qs(urlparse(url).query):
            return archive(url)
        raise AssertionError(f"the archive was hit again for {url}")

    second = run_config(
        config_path, workspace, run_at_utc="2026-09-01T09:00:00Z", open_url=listing_only
    )

    assert second.metrics["net_return"] == pytest.approx(
        read_trials(workspace.trials_path)[0]["net_return"]
    )


def test_the_liquidity_floor_can_evict_the_whole_cross_section(
    workspace, config_path, archive
):
    """A floor above every pair's dollar volume leaves nothing to rank."""
    config_path.write_text(
        CONFIG_TEXT.replace(
            'bracket = "binance-full"',
            'bracket = "binance-full"\nliquidity_floor_usd = 1e18',
        )
    )

    record = run_config(config_path, workspace, run_at_utc=RUN_AT, open_url=archive)

    assert record.portfolio["n_rebalances_held_cash"] == record.portfolio["n_rebalances"]
    assert record.metrics["net_return"] == pytest.approx(0.0)
    assert record.window["universe"]["liquidity_floor"]["applied"] is True


def test_the_cap_weighted_market_portfolio_is_built_from_the_same_universe(record):
    """The secondary reference of ADR-0005: everything the Universe offered,
    weighted by the same vendor caps the strategy sizes positions with."""
    market = record.benchmarks["cap_weighted_market"]

    assert market["computed"] is True
    # All six names, against the strategy's quintile of two, on the same cadence.
    assert market["mean_n_positions"] == pytest.approx(6.0)
    assert market["n_rebalances"] == record.portfolio["n_rebalances"]
    assert market["n_marks"] == record.metrics["n_marks"]


def test_the_market_portfolio_is_reported_net_of_the_same_cost(record):
    market = record.benchmarks["cap_weighted_market"]

    assert market["cost_bps_per_side"] == pytest.approx(45.44)
    assert market["net_return"] < market["gross_return"]


def test_the_run_is_read_against_btc_over_its_own_window(record):
    """The hurdle is held over the strategy's window, not the config's: BTC's
    Decision Bar is the strategy's, so the two paths cover the same days."""
    btc = record.benchmarks["btc_buy_and_hold"]

    assert btc["symbol"] == "BTCUSDT"
    assert btc["decision_ts_utc"] == record.metrics["decision_ts_utc"]
    assert btc["n_marks"] == record.metrics["n_marks"]


def test_the_deployment_hurdle_is_recorded_on_its_three_conditions(record):
    hurdle = record.benchmarks["deployment_hurdle"]

    assert hurdle["adr"] == "ADR-0005"
    assert set(hurdle) >= {
        "sharpe_above_btc",
        "drawdown_no_worse_than_btc",
        "clears_profitability_bar",
        "clears",
    }
    # BTC is the fastest-compounding name here, so the quintile that holds it
    # cannot beat holding it alone once the strategy's turnover is paid for.
    assert hurdle["clears"] is False
