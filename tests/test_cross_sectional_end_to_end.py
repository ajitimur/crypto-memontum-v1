"""A cross-sectional config in, a recorded result out.

The archive and the vendor panel are both synthesised here rather than recorded,
because the point of this test is the wiring — coverage, bars, the point-in-time
Universe, policy, the vendor join, the simulator, the result file — and a fixture
whose prices we chose is the only kind that lets the wiring be checked against a
number worked out by hand. The recorded-bytes tests live one layer down, against
the adapters this exercises.
"""

import hashlib
import io
import json
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from crypto_momentum.data.cmc_panel import CmcPanelStore
from crypto_momentum.data.fetch import ArchiveUnavailable
from crypto_momentum.runner import Workspace, run_config
from crypto_momentum.sim.cross_sectional import TurnoverBudgetBreached
from crypto_momentum.trials import read_trials

RUN_AT = "2026-08-31T09:00:00Z"
MONTHS = ("2021-01", "2021-02", "2021-03")

# Six names, so a quintile is two and the cross-section clears MIN_UNIVERSE.
# Each compounds at its own steady rate, so the ranking never changes and the
# expected selection can be read straight off this table.
DAILY_RATE = {
    "BTCUSDT": 0.010,
    "ETHUSDT": 0.008,
    "BNBUSDT": 0.006,
    "ADAUSDT": 0.004,
    "XRPUSDT": 0.002,
    "DOGEUSDT": 0.000,
}
CMC_ID = {
    "BTCUSDT": (1, "BTC", "Bitcoin"),
    "ETHUSDT": (1027, "ETH", "Ethereum"),
    "BNBUSDT": (1839, "BNB", "BNB"),
    "ADAUSDT": (2010, "ADA", "Cardano"),
    "XRPUSDT": (52, "XRP", "XRP"),
    "DOGEUSDT": (74, "DOGE", "Dogecoin"),
}
# Deliberately the reverse of the return ranking, so a value-weighted portfolio
# cannot be mistaken for an equal-weighted or a signal-weighted one.
MARKET_CAP = {
    "BTCUSDT": 1e9,
    "ETHUSDT": 2e9,
    "BNBUSDT": 3e9,
    "ADAUSDT": 4e9,
    "XRPUSDT": 5e9,
    "DOGEUSDT": 6e9,
}

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

WINDOW = pd.date_range("2021-01-01", "2021-03-31", freq="D", tz="UTC")


def close_of(symbol: str, day: pd.Timestamp) -> float:
    """The synthetic close: a steady compounding ramp from 100 on the first day."""
    step = (day - WINDOW[0]).days
    return 100.0 * (1.0 + DAILY_RATE[symbol]) ** step


def klines_zip(symbol: str, month: str) -> bytes:
    """A monthly partition in the archive's own shape: a zip of a headerless CSV."""
    days = [day for day in WINDOW if day.strftime("%Y-%m") == month]
    rows = []
    for day in days:
        close = close_of(symbol, day)
        # The open is the previous close, so a fill price is checkable by hand.
        open_price = close / (1.0 + DAILY_RATE[symbol])
        open_ms = int(day.value // 1_000_000)
        close_ms = open_ms + 86_399_999
        rows.append(
            f"{open_ms},{open_price:.8f},{close:.8f},{open_price:.8f},{close:.8f},"
            f"1000.0,{close_ms},1000000.0,100,500.0,500000.0,0"
        )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"{symbol}-1d-{month}.csv", "\n".join(rows) + "\n")
    return payload.getvalue()


def listing_page(keys: list[str]) -> bytes:
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
    ).encode()


@pytest.fixture
def archive():
    """An opener serving the synthetic archive. Nothing here touches the network."""
    files: dict[str, bytes] = {}
    listings: dict[str, bytes] = {}
    for symbol in DAILY_RATE:
        keys = []
        for month in MONTHS:
            filename = f"{symbol}-1d-{month}.zip"
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{filename}"
            payload = klines_zip(symbol, month)
            files[url] = payload
            files[url + ".CHECKSUM"] = (
                f"{hashlib.sha256(payload).hexdigest()}  {filename}".encode()
            )
            key = f"data/spot/monthly/klines/{symbol}/1d/{filename}"
            keys.extend([key, key + ".CHECKSUM"])
        listings[f"data/spot/monthly/klines/{symbol}/1d/"] = listing_page(keys)
        # The window is long over, so the archive has rolled every month up and
        # there are no daily-only partitions at the tail.
        listings[f"data/spot/daily/klines/{symbol}/1d/"] = listing_page([])

    def open_url(url: str) -> bytes:
        query = parse_qs(urlparse(url).query)
        if "prefix" in query:
            prefix = query["prefix"][0]
            if prefix not in listings:
                raise ArchiveUnavailable(f"no listing for {prefix}")
            return listings[prefix]
        if url not in files:
            raise ArchiveUnavailable(f"404 for {url}")
        return files[url]

    return open_url


def panel_csv() -> bytes:
    """A weekly vendor panel covering the window, plus what the store demands of it.

    The first row reaches back to CoinMarketCap's own first snapshot so the
    stored manifest is not stamped with a window the payload does not cover, and
    the two known dead assets are present so the panel is not refused as
    survivorship-biased.
    """
    header = (
        "ts_utc,cmc_id,symbol,name,cmc_rank,price_usd,market_cap_usd,"
        "volume_24h_usd,circulating_supply"
    )
    rows = ["2013-04-28,1,BTC,Bitcoin,1,135.30,1500000000.0,0.0,11091000.0"]
    # The dead assets, listed once each and never again — which is exactly the
    # shape a survivorship-free panel has.
    rows.append("2018-01-07,827,BCC,BitConnect,207,29.10,268000000.0,3100000.0,9200000.0")
    rows.append("2020-08-16,6187,SRM,Serum,144,1.42,68000000.0,41000000.0,47900000.0")
    for snapshot in pd.date_range("2020-12-06", "2021-04-04", freq="7D"):
        for rank, (symbol, (cmc_id, ticker, name)) in enumerate(CMC_ID.items(), start=1):
            rows.append(
                f"{snapshot.date()},{cmc_id},{ticker},{name},{rank},1.0,"
                f"{MARKET_CAP[symbol]},1000000.0,1000000.0"
            )
    return ("\n".join([header, *rows]) + "\n").encode()


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(args, cwd=root, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (root / "README.md").write_text("cross-sectional\n")
    run("git", "add", ".")
    run("git", "commit", "-qm", "first")

    workspace = Workspace.under(root)
    # The policy artefacts and the vendor override table are committed data, so
    # the run reads the repo's own copies rather than a test-local invention.
    source = Path(__file__).parent.parent
    for relative in (
        "policy/exclusions-v1.toml",
        "policy/tokocrypto-listing-v1.toml",
        "configs/vendor-symbol-map.toml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / relative).read_bytes())

    CmcPanelStore(workspace.raw_root).write(
        panel_csv(), pulled_at_utc=RUN_AT, window_start=date(2013, 4, 28)
    )
    return workspace


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
