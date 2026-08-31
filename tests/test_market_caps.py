"""Market capitalisations pivoted from vendor ids onto venue symbols."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_momentum.data.cmc_panel import parse_panel_csv
from crypto_momentum.data.market_caps import (
    UnmappableSymbol,
    base_asset_of,
    market_cap_panel,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coinmarketcap"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def panel():
    return parse_panel_csv((FIXTURES / "cmc-panel-sample.csv").read_bytes())


class TestBaseAssetOf:
    def test_splits_the_base_off_the_quote(self):
        assert base_asset_of("BTCUSDT", quote_asset="USDT") == "BTC"

    def test_a_pair_from_another_quote_book_is_refused(self):
        with pytest.raises(UnmappableSymbol):
            base_asset_of("ETHBTC", quote_asset="USDT")

    def test_a_symbol_that_is_only_the_quote_is_refused(self):
        with pytest.raises(UnmappableSymbol):
            base_asset_of("USDT", quote_asset="USDT")


class TestMarketCapPanel:
    def test_is_one_column_per_requested_symbol_in_the_order_asked_for(self, panel):
        caps = market_cap_panel(panel, ["ETHUSDT", "BTCUSDT"], repo_root=REPO_ROOT)

        assert list(caps.columns) == ["ETHUSDT", "BTCUSDT"]

    def test_carries_the_capitalisation_of_the_matching_vendor_row(self, panel):
        caps = market_cap_panel(panel, ["BTCUSDT"], repo_root=REPO_ROOT)

        assert caps.loc[pd.Timestamp("2017-01-01T00:00:00Z"), "BTCUSDT"] == pytest.approx(
            15_491_000_000.0
        )

    def test_a_symbol_the_vendor_never_listed_is_nan_throughout(self, panel):
        caps = market_cap_panel(panel, ["PEPEUSDT"], repo_root=REPO_ROOT)

        assert caps["PEPEUSDT"].isna().all()

    def test_a_snapshot_before_an_asset_listed_is_nan_not_zero(self, panel):
        """Unweightable and worthless are different, and only one is a weight."""
        caps = market_cap_panel(panel, ["SRMUSDT"], repo_root=REPO_ROOT)

        assert np.isnan(caps.loc[pd.Timestamp("2017-01-01T00:00:00Z"), "SRMUSDT"])
        assert caps.loc[pd.Timestamp("2020-08-16T00:00:00Z"), "SRMUSDT"] == pytest.approx(
            68_000_000.0
        )

    def test_a_delisted_asset_keeps_the_capitalisations_it_had(self, panel):
        """Serum died in November 2022. The 2022 cross-section still contains it."""
        caps = market_cap_panel(panel, ["SRMUSDT"], repo_root=REPO_ROOT)

        assert caps.loc[pd.Timestamp("2022-11-13T00:00:00Z"), "SRMUSDT"] == pytest.approx(
            49_000_000.0
        )
        assert np.isnan(caps.loc[pd.Timestamp("2023-01-01T00:00:00Z"), "SRMUSDT"])

    def test_a_reused_ticker_is_split_at_the_venues_own_rename_date(self, panel):
        """LUNA was reassigned to Terra 2.0 while the original became LUNC.

        A mapping that collapsed the two would hand LUNAUSDT the original
        chain's collapsing capitalisation after the rename, and the 2022
        cross-section would weight a fresh listing by a dead one.
        """
        caps = market_cap_panel(panel, ["LUNAUSDT", "LUNCUSDT"], repo_root=REPO_ROOT)

        # The boundary is Binance's rename on 2022-05-31, not the vendor's own
        # snapshot grid: the 05-29 snapshot already calls id 4172 "LUNC", but
        # LUNCUSDT did not exist to trade until the venue said so two days later.
        before = pd.Timestamp("2022-05-29T00:00:00Z")
        after = pd.Timestamp("2022-11-13T00:00:00Z")

        assert caps.loc[before, "LUNAUSDT"] == pytest.approx(740_000_000.0)
        assert np.isnan(caps.loc[before, "LUNCUSDT"])
        # After the rename LUNAUSDT is Terra 2.0 and the original chain is LUNCUSDT.
        assert caps.loc[after, "LUNAUSDT"] == pytest.approx(270_000_000.0)
        assert caps.loc[after, "LUNCUSDT"] == pytest.approx(1_010_000_000.0)
