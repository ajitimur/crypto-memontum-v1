"""Policy turns "what existed" into "what we would consider holding".

Three gates, tested one at a time and then together: the dated exclusion list,
the trailing liquidity floor, and the Universe bracket. Coverage is hand-built
and the expected dates were worked out by hand from it.
"""

from pathlib import Path

import pandas as pd
import pytest

from crypto_momentum.data.universe import SymbolCoverage, build_universe_panel
from crypto_momentum.policy import (
    EXCLUSIONS_FILENAME,
    TOKOCRYPTO_LISTING_FILENAME,
    load_exclusion_list,
    load_venue_listing,
    policy_root,
)
from crypto_momentum.sim.universe_policy import (
    BINANCE_FULL,
    TOKOCRYPTO,
    ExclusionList,
    LiquidityFloor,
    PolicyError,
    VenueListing,
    apply_universe_policy,
    dollar_volume_from_bars,
    universe_bracket,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUSIONS_V1 = policy_root(REPO_ROOT) / EXCLUSIONS_FILENAME
TOKOCRYPTO_V1 = policy_root(REPO_ROOT) / TOKOCRYPTO_LISTING_FILENAME


def months_between(start, end):
    return tuple(
        stamp.strftime("%Y-%m") for stamp in pd.period_range(start, end, freq="M")
    )


def coverage(symbol, start, end, interval="1d"):
    return SymbolCoverage(
        symbol=symbol, interval=interval, months=months_between(start, end)
    )


def panel_of(*coverages, start="2021-01-01", end="2022-12-31"):
    return build_universe_panel(coverages, start=start, end=end)


def exclusion_list(entries, *, version="test-v1", as_of="2021-01-01"):
    return ExclusionList.from_document(
        {"version": version, "as_of": as_of, "entry": list(entries)}
    )


def entry(symbol, category="stablecoin"):
    return {
        "symbol": symbol,
        "base_asset": symbol.removesuffix("USDT"),
        "category": category,
        "design_intent": "stated on day one",
        "source": "https://example.invalid",
    }


EMPTY_LIST = ExclusionList.from_document(
    {"version": "empty-v1", "as_of": "2021-01-01", "entry": []}
)


def flat_volume(index, symbols, value):
    return pd.DataFrame(float(value), index=index, columns=list(symbols))


# --- The exclusion list is a dated, versioned artifact ----------------------


def test_the_shipped_exclusion_list_loads_with_its_version_and_date():
    exclusions = load_exclusion_list(EXCLUSIONS_V1)

    assert exclusions.version == "v1"
    assert exclusions.as_of == "2026-08-31"
    assert exclusions.path == str(EXCLUSIONS_V1)
    assert len(exclusions.sha256) == 64


def test_the_shipped_list_classifies_stablecoins_and_wrapped_assets():
    exclusions = load_exclusion_list(EXCLUSIONS_V1)

    assert exclusions.entry_for("USTUSDT").category == "stablecoin"
    assert exclusions.entry_for("WBTCUSDT").category == "wrapped_asset"
    assert exclusions.entry_for("WBETHUSDT").category == "wrapped_asset"
    assert exclusions.entry_for("BTCUSDT") is None


def test_every_shipped_entry_states_the_design_intent_it_was_classified_on():
    """ADR-0006 classifies on stated intent, so an entry without one is not evidence."""
    for excluded in load_exclusion_list(EXCLUSIONS_V1).entries:
        assert excluded.design_intent.strip()
        assert excluded.source.strip()


def test_an_unknown_category_is_rejected_rather_than_carried():
    with pytest.raises(PolicyError, match="category"):
        exclusion_list([entry("XYZUSDT", category="speculative")])


def test_a_symbol_listed_twice_is_rejected():
    with pytest.raises(PolicyError, match="twice"):
        exclusion_list([entry("USTUSDT"), entry("USTUSDT")])


def test_a_list_without_a_version_is_rejected():
    with pytest.raises(PolicyError, match="version"):
        ExclusionList.from_document({"as_of": "2021-01-01", "entry": []})


# --- Exclusions are permanent, and UST is the case that proves it -----------


def test_ust_is_excluded_for_the_whole_sample_including_after_the_depeg():
    """Per ADR-0006: intent is knowable at listing, so this uses no future information."""
    panel = panel_of(
        coverage("USTUSDT", "2021-01", "2022-12"), coverage("BTCUSDT", "2021-01", "2022-12")
    )

    policy = apply_universe_policy(
        panel, exclusions=exclusion_list([entry("USTUSDT")]), bracket=BINANCE_FULL
    )

    assert not policy.tradeable["USTUSDT"].any()
    assert policy.tradeable_on("2021-06-01") == ("BTCUSDT",)
    # After the May 2022 depeg, where the return series is most tempting.
    assert policy.tradeable_on("2022-06-01") == ("BTCUSDT",)


def test_wrapped_and_liquid_staked_assets_are_excluded_permanently():
    panel = panel_of(
        coverage("WBTCUSDT", "2021-01", "2022-12"),
        coverage("WBETHUSDT", "2022-05", "2022-12"),
        coverage("ETHUSDT", "2021-01", "2022-12"),
    )

    policy = apply_universe_policy(
        panel,
        exclusions=exclusion_list(
            [entry("WBTCUSDT", "wrapped_asset"), entry("WBETHUSDT", "wrapped_asset")]
        ),
        bracket=BINANCE_FULL,
    )

    assert not policy.tradeable["WBTCUSDT"].any()
    assert not policy.tradeable["WBETHUSDT"].any()
    assert policy.tradeable["ETHUSDT"].all()


def test_an_asset_the_list_does_not_name_is_left_alone():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    policy = apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket=BINANCE_FULL)

    assert policy.tradeable["BTCUSDT"].all()


def test_a_symbol_is_classified_at_its_first_tradeable_date_not_retroactively():
    """The classification date is read off archive coverage, never typed in."""
    panel = panel_of(
        coverage("USTUSDT", "2021-03", "2022-12"), coverage("BTCUSDT", "2021-01", "2022-12")
    )

    policy = apply_universe_policy(
        panel, exclusions=exclusion_list([entry("USTUSDT")]), bracket=BINANCE_FULL
    )

    classified = policy.metadata["exclusion_list"]["classified_at_ts_utc"]
    assert classified["USTUSDT"] == "2021-03-01T00:00:00Z"


def test_a_listing_the_list_never_saw_is_reported_rather_than_judged():
    """A stale list must not pass silently: an asset first traded after `as_of` is named."""
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2022-12"), coverage("NEWUSDT", "2022-06", "2022-12")
    )

    policy = apply_universe_policy(
        panel,
        exclusions=exclusion_list([entry("USTUSDT")], as_of="2022-01-01"),
        bracket=BINANCE_FULL,
    )

    assert policy.metadata["exclusion_list"]["unclassified_listings_since_as_of"] == [
        "NEWUSDT"
    ]
    assert policy.tradeable["NEWUSDT"].any()


def test_the_policy_records_which_exclusion_list_produced_it():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")
    exclusions = load_exclusion_list(EXCLUSIONS_V1)

    policy = apply_universe_policy(panel, exclusions=exclusions, bracket=BINANCE_FULL)

    recorded = policy.metadata["exclusion_list"]
    assert recorded["version"] == "v1"
    assert recorded["as_of"] == "2026-08-31"
    assert recorded["sha256"] == exclusions.sha256
    assert recorded["n_entries"] == len(exclusions.entries)


def test_the_policy_carries_the_universe_metadata_it_was_built_from():
    """The archive floor must survive the policy layer, not be dropped at it."""
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    policy = apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket=BINANCE_FULL)

    assert policy.metadata["universe"]["archive_floor_ts_utc"] == "2017-08-17T00:00:00Z"


# --- The liquidity floor is a data-quality gate, and it is point-in-time -----


def test_the_floor_reads_only_bars_strictly_before_the_decision_bar():
    """A volume spike on the Decision Bar cannot lift that bar's own gate."""
    dates = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC", name="ts_utc")
    dollar_volume = pd.DataFrame(
        {"AUSDT": [10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0]}, index=dates
    )

    mask = LiquidityFloor(floor_usd=100.0, window_days=3).mask(dollar_volume)

    assert list(mask["AUSDT"]) == [False, False, False, False, False, True]


def test_a_symbol_without_a_full_window_of_history_does_not_pass_the_floor():
    dates = pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC", name="ts_utc")
    dollar_volume = flat_volume(dates, ["AUSDT"], 1_000_000)

    mask = LiquidityFloor(floor_usd=100.0, window_days=3).mask(dollar_volume)

    assert list(mask["AUSDT"]) == [False, False, False, True]


def test_a_hole_in_the_index_does_not_widen_the_trailing_window():
    """Thirty days means thirty days: three rows spanning a fortnight are not a window."""
    dates = pd.DatetimeIndex(
        ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-14"], tz="UTC", name="ts_utc"
    )
    dollar_volume = flat_volume(dates, ["AUSDT"], 1_000)

    mask = LiquidityFloor(floor_usd=100.0, window_days=3).mask(dollar_volume)

    assert not mask["AUSDT"].any()


def test_a_symbol_below_the_floor_is_dropped_from_the_universe():
    panel = panel_of(
        coverage("BIGUSDT", "2021-01", "2021-06"),
        coverage("THINUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )
    dates = panel.tradeable.index
    dollar_volume = pd.concat(
        [
            flat_volume(dates, ["BIGUSDT"], 5_000_000),
            flat_volume(dates, ["THINUSDT"], 1_000),
        ],
        axis=1,
    )

    policy = apply_universe_policy(
        panel,
        exclusions=EMPTY_LIST,
        bracket=BINANCE_FULL,
        dollar_volume=dollar_volume,
        floor=LiquidityFloor(floor_usd=100_000.0),
    )

    assert policy.tradeable_on("2021-05-01") == ("BIGUSDT",)
    assert not policy.tradeable["THINUSDT"].any()


def test_the_floor_is_recorded_as_a_data_quality_gate_with_its_parameters():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")
    dollar_volume = flat_volume(panel.tradeable.index, ["BTCUSDT"], 5_000_000)

    policy = apply_universe_policy(
        panel,
        exclusions=EMPTY_LIST,
        bracket=BINANCE_FULL,
        dollar_volume=dollar_volume,
        floor=LiquidityFloor(floor_usd=100_000.0),
    )

    recorded = policy.metadata["liquidity_floor"]
    assert recorded["applied"] is True
    assert recorded["floor_usd"] == 100_000.0
    assert recorded["window_days"] == 30
    assert "capacity" in recorded["purpose"]


def test_a_panel_built_without_volume_says_the_floor_was_not_applied():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    policy = apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket=BINANCE_FULL)

    assert policy.metadata["liquidity_floor"]["applied"] is False


def test_a_floor_without_the_volume_to_apply_it_is_an_error():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    with pytest.raises(PolicyError, match="dollar volume"):
        apply_universe_policy(
            panel,
            exclusions=EMPTY_LIST,
            bracket=BINANCE_FULL,
            floor=LiquidityFloor(floor_usd=100_000.0),
        )


def test_a_symbol_missing_from_the_volume_frame_is_an_error_not_a_silent_drop():
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2021-06"),
        coverage("ETHUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )
    dollar_volume = flat_volume(panel.tradeable.index, ["BTCUSDT"], 5_000_000)

    with pytest.raises(PolicyError, match="ETHUSDT"):
        apply_universe_policy(
            panel,
            exclusions=EMPTY_LIST,
            bracket=BINANCE_FULL,
            dollar_volume=dollar_volume,
            floor=LiquidityFloor(floor_usd=100_000.0),
        )


def test_dollar_volume_is_built_from_the_bars_of_each_symbol():
    dates = pd.date_range("2021-01-01", periods=2, freq="D", tz="UTC", name="ts_utc")
    bars = {
        "AUSDT": pd.DataFrame({"close": [10.0, 20.0], "volume": [3.0, 4.0]}, index=dates),
        "BUSDT": pd.DataFrame({"close": [1.0, 2.0], "volume": [5.0, 6.0]}, index=dates),
    }

    dollar_volume = dollar_volume_from_bars(bars)

    assert list(dollar_volume["AUSDT"]) == [30.0, 80.0]
    assert list(dollar_volume["BUSDT"]) == [5.0, 12.0]


def test_a_symbol_absent_on_a_date_carries_no_dollar_volume_there():
    """A union index must not read a missing bar as a zero-volume one."""
    early = pd.date_range("2021-01-01", periods=2, freq="D", tz="UTC", name="ts_utc")
    late = pd.date_range("2021-01-02", periods=2, freq="D", tz="UTC", name="ts_utc")
    bars = {
        "AUSDT": pd.DataFrame({"close": [10.0, 10.0], "volume": [1.0, 1.0]}, index=early),
        "BUSDT": pd.DataFrame({"close": [10.0, 10.0], "volume": [2.0, 2.0]}, index=late),
    }

    dollar_volume = dollar_volume_from_bars(bars)

    assert dollar_volume["BUSDT"].isna().iloc[0]
    assert dollar_volume["AUSDT"].isna().iloc[-1]


# --- The bracket, reported both ways ----------------------------------------


def test_the_shipped_venue_listing_loads_with_its_status_and_version():
    listing = load_venue_listing(TOKOCRYPTO_V1)

    assert listing.venue == "tokocrypto"
    assert listing.version == "v1"
    assert listing.status == "stub"
    assert "BTCUSDT" in listing.symbols


def test_the_lower_bound_keeps_only_what_our_own_venue_lists():
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2021-06"),
        coverage("SRMUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )
    listing = VenueListing.from_document(
        {
            "venue": "tokocrypto",
            "version": "test-v1",
            "recorded_at": "2021-01-01",
            "status": "recorded",
            "symbols": ["BTCUSDT"],
        }
    )

    policy = apply_universe_policy(
        panel, exclusions=EMPTY_LIST, bracket=TOKOCRYPTO, venue_listing=listing
    )

    assert policy.tradeable_on("2021-05-01") == ("BTCUSDT",)
    assert not policy.tradeable["SRMUSDT"].any()


def test_the_upper_bound_is_the_full_binance_universe():
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2021-06"),
        coverage("SRMUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )

    policy = apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket=BINANCE_FULL)

    assert policy.tradeable_on("2021-05-01") == ("BTCUSDT", "SRMUSDT")


def test_the_bracket_is_reported_both_ways_from_one_call():
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2021-06"),
        coverage("SRMUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )
    listing = load_venue_listing(TOKOCRYPTO_V1)

    bracket = universe_bracket(panel, exclusions=EMPTY_LIST, venue_listing=listing)

    assert bracket[BINANCE_FULL].tradeable_on("2021-05-01") == ("BTCUSDT", "SRMUSDT")
    assert bracket[TOKOCRYPTO].tradeable_on("2021-05-01") == ("BTCUSDT",)
    assert bracket[BINANCE_FULL].metadata["bracket"]["bound"] == "upper"
    assert bracket[TOKOCRYPTO].metadata["bracket"]["bound"] == "lower"


def test_a_lower_bound_built_on_a_stub_listing_says_so_in_its_metadata():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    policy = apply_universe_policy(
        panel,
        exclusions=EMPTY_LIST,
        bracket=TOKOCRYPTO,
        venue_listing=load_venue_listing(TOKOCRYPTO_V1),
    )

    recorded = policy.metadata["bracket"]["venue_listing"]
    assert recorded["status"] == "stub"
    assert recorded["version"] == "v1"


def test_the_lower_bound_needs_a_venue_listing_to_stand_on():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    with pytest.raises(PolicyError, match="venue listing"):
        apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket=TOKOCRYPTO)


def test_an_unknown_bracket_is_rejected():
    panel = panel_of(coverage("BTCUSDT", "2021-01", "2021-06"), end="2021-06-30")

    with pytest.raises(PolicyError, match="bracket"):
        apply_universe_policy(panel, exclusions=EMPTY_LIST, bracket="coinbase")


# --- The three gates compose ------------------------------------------------


def test_the_gates_compose_into_one_tradeable_flag():
    panel = panel_of(
        coverage("BTCUSDT", "2021-01", "2021-06"),
        coverage("USTUSDT", "2021-01", "2021-06"),
        coverage("THINUSDT", "2021-01", "2021-06"),
        coverage("SRMUSDT", "2021-01", "2021-06"),
        end="2021-06-30",
    )
    dates = panel.tradeable.index
    dollar_volume = pd.concat(
        [
            flat_volume(dates, ["BTCUSDT", "USTUSDT", "SRMUSDT"], 5_000_000),
            flat_volume(dates, ["THINUSDT"], 1_000),
        ],
        axis=1,
    )
    listing = VenueListing.from_document(
        {
            "venue": "tokocrypto",
            "version": "test-v1",
            "recorded_at": "2021-01-01",
            "status": "recorded",
            "symbols": ["BTCUSDT", "USTUSDT", "THINUSDT"],
        }
    )

    policy = apply_universe_policy(
        panel,
        exclusions=exclusion_list([entry("USTUSDT")]),
        bracket=TOKOCRYPTO,
        venue_listing=listing,
        dollar_volume=dollar_volume,
        floor=LiquidityFloor(floor_usd=100_000.0),
    )

    # USTUSDT excluded by intent, THINUSDT by the floor, SRMUSDT by the bracket.
    assert policy.tradeable_on("2021-05-01") == ("BTCUSDT",)
    assert policy.metadata["n_symbols_tradeable_at_some_point"] == 1
