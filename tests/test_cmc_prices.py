"""The Faithful Run's data layer: prices and a Universe from the vendor panel."""

from pathlib import Path

import pandas as pd
import pytest

from crypto_momentum.data.binance_archive import BAR_COLUMNS
from crypto_momentum.data.cmc_panel import parse_panel_csv
from crypto_momentum.data.cmc_prices import (
    NoPanelPrices,
    PanelGrainTooCoarse,
    assert_daily_grain,
    panel_bars,
    panel_universe,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coinmarketcap"
REPO_ROOT = Path(__file__).parent.parent


def daily_panel_csv(rows):
    """A panel CSV in the vendor's own column order, from `(date, id, symbol,
    name, rank, price, cap, volume, supply)` tuples."""
    header = (
        "ts_utc,cmc_id,symbol,name,cmc_rank,price_usd,market_cap_usd,"
        "volume_24h_usd,circulating_supply"
    )
    lines = [",".join(str(field) for field in row) for row in rows]
    return ("\n".join([header, *lines]) + "\n").encode()


def daily_rows(*, symbol, cmc_id, start, days, price, volume=1_000_000.0):
    dates = pd.date_range(start, periods=days, freq="D")
    return [
        (
            date.strftime("%Y-%m-%d"),
            cmc_id,
            symbol,
            symbol,
            1,
            price + index,
            (price + index) * 1_000_000.0,
            volume,
            1_000_000.0,
        )
        for index, date in enumerate(dates)
    ]


@pytest.fixture
def daily_panel():
    """A daily two-asset panel. BTC runs the window; SRM stops halfway."""
    return parse_panel_csv(
        daily_panel_csv(
            [
                *daily_rows(
                    symbol="BTC", cmc_id=1, start="2021-01-01", days=10, price=100.0
                ),
                *daily_rows(
                    symbol="SRM", cmc_id=6187, start="2021-01-01", days=5, price=20.0
                ),
            ]
        )
    )


@pytest.fixture
def weekly_panel():
    """The stored panel's own grain: ADR-0008 pulls at `--interval 7d`."""
    return parse_panel_csv((FIXTURES / "cmc-panel-sample.csv").read_bytes())


class TestGrain:
    def test_a_daily_panel_is_accepted(self, daily_panel):
        assert_daily_grain(daily_panel) is None

    def test_the_weekly_panel_adr_0008_pulls_is_refused(self, weekly_panel):
        """Resampling it would invent six prices in seven, and ADR-0001 exists
        because what happens inside a holding period is what liquidates."""
        with pytest.raises(PanelGrainTooCoarse) as refusal:
            assert_daily_grain(weekly_panel)

        assert "daily" in str(refusal.value)

    def test_the_refusal_names_what_would_have_to_change(self, weekly_panel):
        with pytest.raises(PanelGrainTooCoarse) as refusal:
            assert_daily_grain(weekly_panel)

        assert "pull_cmc_panel.R" in str(refusal.value)

    def test_a_panel_of_one_snapshot_is_not_a_series(self):
        one = parse_panel_csv(
            daily_panel_csv(
                daily_rows(symbol="BTC", cmc_id=1, start="2021-01-01", days=1, price=100.0)
            )
        )
        with pytest.raises(PanelGrainTooCoarse):
            assert_daily_grain(one)


class TestPanelBars:
    def test_carries_the_columns_every_other_bar_frame_carries(self, daily_panel):
        """So nothing downstream of the data layer branches on the price source."""
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )

        assert list(bars["BTCUSDT"].columns) == BAR_COLUMNS

    def test_a_snapshot_price_is_the_whole_bar(self, daily_panel):
        """The vendor publishes a price as of 00:00 UTC, not a range."""
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )
        row = bars["BTCUSDT"].loc[pd.Timestamp("2021-01-01T00:00:00Z")]

        assert row["open"] == row["high"] == row["low"] == row["close"] == 100.0

    def test_volume_is_the_vendor_s_own_dollar_volume(self, daily_panel):
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )

        assert bars["BTCUSDT"]["volume"].iloc[0] == pytest.approx(1_000_000.0)

    def test_the_window_clips_the_series_at_both_ends(self, daily_panel):
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-03",
            end="2021-01-06",
        )

        assert bars["BTCUSDT"].index[0] == pd.Timestamp("2021-01-03T00:00:00Z")
        assert bars["BTCUSDT"].index[-1] == pd.Timestamp("2021-01-06T00:00:00Z")

    def test_an_asset_that_stops_being_priced_keeps_the_bars_it_had(self, daily_panel):
        """The panel is survivorship-free, so a death is a series that ends."""
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT", "SRMUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )

        assert len(bars["SRMUSDT"]) == 5
        assert bars["SRMUSDT"].index[-1] == pd.Timestamp("2021-01-05T00:00:00Z")

    def test_a_symbol_the_vendor_never_priced_is_left_out_not_carried_empty(
        self, daily_panel
    ):
        bars = panel_bars(
            daily_panel,
            ["BTCUSDT", "PEPEUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )

        assert set(bars) == {"BTCUSDT"}

    def test_a_window_the_panel_does_not_reach_is_refused(self, daily_panel):
        with pytest.raises(NoPanelPrices):
            panel_bars(
                daily_panel,
                ["BTCUSDT"],
                repo_root=REPO_ROOT,
                start="2022-01-01",
                end="2022-01-10",
            )

    def test_a_coarse_panel_is_refused_before_any_bar_is_built(self, weekly_panel):
        with pytest.raises(PanelGrainTooCoarse):
            panel_bars(
                weekly_panel,
                ["BTCUSDT"],
                repo_root=REPO_ROOT,
                start="2017-01-01",
                end="2018-02-01",
            )


class TestPanelUniverse:
    @pytest.fixture
    def bars(self, daily_panel):
        return panel_bars(
            daily_panel,
            ["BTCUSDT", "SRMUSDT"],
            repo_root=REPO_ROOT,
            start="2021-01-01",
            end="2021-01-10",
        )

    def test_an_asset_is_tradeable_exactly_on_the_dates_it_is_priced(self, bars):
        universe = panel_universe(bars, start="2021-01-01", end="2021-01-10")

        assert universe.tradeable_on("2021-01-05") == ("BTCUSDT", "SRMUSDT")
        assert universe.tradeable_on("2021-01-06") == ("BTCUSDT",)

    def test_the_dates_before_an_asset_died_stay_tradeable(self, bars):
        """A survivorship-biased panel would have dropped SRM's history too."""
        universe = panel_universe(bars, start="2021-01-01", end="2021-01-10")

        assert universe.tradeable["SRMUSDT"].iloc[:5].all()

    def test_the_window_is_spanned_as_asked_even_where_nothing_is_priced(self, bars):
        universe = panel_universe(bars, start="2020-12-28", end="2021-01-10")

        assert len(universe.tradeable) == 14
        assert not universe.tradeable.loc[pd.Timestamp("2020-12-28T00:00:00Z")].any()

    def test_the_metadata_names_the_vendor_rather_than_the_archive(self, bars):
        universe = panel_universe(bars, start="2021-01-01", end="2021-01-10")

        assert "coinmarketcap" in universe.metadata["source"]
        assert universe.metadata["coverage_resolution"] == (
            "vendor snapshot, one per UTC date"
        )

    def test_the_metadata_counts_the_asset_that_stopped_being_priced(self, bars):
        universe = panel_universe(bars, start="2021-01-01", end="2021-01-10")

        assert universe.metadata["n_symbols_delisted_within_window"] == 1

    def test_the_panel_floor_is_recorded_and_is_not_the_archive_s(self, bars):
        """2013-04-28, below Han et al.'s 2017-01-01 — which is the point."""
        universe = panel_universe(bars, start="2021-01-01", end="2021-01-10")

        assert universe.metadata["panel_floor_ts_utc"].startswith("2013-04-28")
