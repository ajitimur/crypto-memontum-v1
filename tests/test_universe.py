"""The Universe is reconstructed as it stood, delisted assets included.

The recorded-listing tests pin the survivorship claim against real bucket bytes;
the panel tests use hand-built coverage whose expected dates were worked out by
hand.
"""

import pandas as pd
import pytest

from conftest import DAILY_KLINES, KLINES, SR_PAGES, SR_PREFIX, daily_tail_marker
from crypto_momentum.data.binance_archive import ChecksumMismatch
from crypto_momentum.data.universe import (
    SymbolCoverage,
    SymbolNotCovered,
    UniverseError,
    bar_span_from_bars,
    build_archive_universe,
    build_universe_panel,
    coverage_for_symbol,
    coverage_from_keys,
    fetch_covered_month,
    symbols_in_archive,
)


def keys_for(symbol, interval, months, *, without_checksum=()):
    """The listing keys the archive publishes for `months` — zip plus checksum."""
    keys = []
    for month in months:
        name = f"{KLINES}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
        keys.append(name)
        if month not in without_checksum:
            keys.append(name + ".CHECKSUM")
    return keys


def coverage(symbol, months, interval="1d"):
    return SymbolCoverage(symbol=symbol, interval=interval, months=tuple(months))


def months_between(start, end):
    return tuple(
        stamp.strftime("%Y-%m") for stamp in pd.period_range(start, end, freq="M")
    )


# --- Enumeration comes from the bucket, so delisted symbols survive ----------


def srm_bucket(recorded_bucket):
    """SRMUSDT's monthly partitions, and the empty tail a delisted symbol has."""
    return recorded_bucket(
        {
            f"{KLINES}/SRMUSDT/1d/": ["SRMUSDT-1d.xml"],
            (
                f"{DAILY_KLINES}/SRMUSDT/1d/",
                daily_tail_marker("SRMUSDT", "1d", "2022-11"),
            ): ["SRMUSDT-1d-daily-tail.xml"],
        }
    )


def btc_bucket(recorded_bucket):
    """BTCUSDT's monthly partitions plus the running month it has not rolled up."""
    return recorded_bucket(
        {
            f"{KLINES}/BTCUSDT/1d/": ["BTCUSDT-1d.xml"],
            (
                f"{DAILY_KLINES}/BTCUSDT/1d/",
                daily_tail_marker("BTCUSDT", "1d", "2026-07"),
            ): ["BTCUSDT-1d-daily-tail.xml"],
        }
    )


def test_symbols_are_enumerated_from_the_bucket_listing(recorded_bucket):
    """Not from exchangeInfo: SRMUSDT stopped trading in 2022 and is still here."""
    symbols = symbols_in_archive(
        prefix=SR_PREFIX, open_url=recorded_bucket({SR_PREFIX: SR_PAGES})
    )

    assert symbols == ("SRMBIDR", "SRMBNB", "SRMBTC", "SRMBUSD", "SRMUSDT")


def test_a_quote_asset_filter_keeps_only_that_book(recorded_bucket):
    symbols = symbols_in_archive(
        quote_asset="USDT",
        prefix=SR_PREFIX,
        open_url=recorded_bucket({SR_PREFIX: SR_PAGES}),
    )

    assert symbols == ("SRMUSDT",)


def test_a_delisted_symbol_keeps_the_coverage_it_had(recorded_bucket):
    """SRMUSDT is the standing survivorship witness — it ends in November 2022."""
    srm = coverage_for_symbol("SRMUSDT", "1d", open_url=srm_bucket(recorded_bucket))

    assert srm.first_month == "2020-08"
    assert srm.last_month == "2022-11"
    assert srm.months == months_between("2020-08", "2022-11")
    # Delisted, so the archive has no running month for it.
    assert srm.daily_only_dates == ()
    assert srm.last_covered_date == pd.Timestamp("2022-11-30T00:00:00Z")


def test_a_still_trading_symbol_starts_at_the_archive_floor(recorded_bucket):
    btc = coverage_for_symbol("BTCUSDT", "1d", open_url=btc_bucket(recorded_bucket))

    assert btc.first_month == "2017-08"
    # Coverage is the partition's own span. The 2017-08 partition opens on the
    # 17th, but that is the archive floor and the panel applies it, not coverage.
    assert btc.first_covered_date == pd.Timestamp("2017-08-01T00:00:00Z")


def test_the_running_month_comes_from_the_daily_partitions(recorded_bucket):
    """The archive rolls a month up only once it is over; until then it is daily."""
    btc = coverage_for_symbol("BTCUSDT", "1d", open_url=btc_bucket(recorded_bucket))

    assert btc.last_month == "2026-07"
    assert btc.daily_only_dates[0] == "2026-08-01"
    assert btc.daily_only_dates[-1] == "2026-08-30"
    assert btc.last_covered_date == pd.Timestamp("2026-08-30T00:00:00Z")


def test_a_live_asset_is_not_reported_delisted_at_the_tail(recorded_bucket):
    """Reading only monthly partitions would call BTCUSDT delisted this month."""
    btc = coverage_for_symbol("BTCUSDT", "1d", open_url=btc_bucket(recorded_bucket))

    panel = build_universe_panel([btc], start="2026-06-01", end="2026-08-30")

    assert panel.tradeable["BTCUSDT"].all()
    assert panel.tradeable_on("2026-08-15") == ("BTCUSDT",)
    assert panel.metadata["n_symbols_delisted_within_window"] == 0


def test_a_window_ending_today_does_not_call_every_live_asset_delisted(recorded_bucket):
    """Today's bar is unpublished until the day closes; that is not a delisting."""
    btc = coverage_for_symbol("BTCUSDT", "1d", open_url=btc_bucket(recorded_bucket))
    srm = coverage_for_symbol("SRMUSDT", "1d", open_url=srm_bucket(recorded_bucket))

    # 2026-08-31 is "today": the archive's last partition is the 30th.
    panel = build_universe_panel([btc, srm], start="2022-01-01", end="2026-08-31")

    assert panel.metadata["delisting_reference_ts_utc"] == "2026-08-30T00:00:00Z"
    assert panel.metadata["n_symbols_delisted_within_window"] == 1
    assert panel.tradeable_on("2026-08-30") == ("BTCUSDT",)


def test_delisting_falls_back_to_the_window_when_nothing_is_still_running(
    recorded_bucket,
):
    """With no live symbol there is no frontier to read, so the window is it."""
    srm = coverage_for_symbol("SRMUSDT", "1d", open_url=srm_bucket(recorded_bucket))

    panel = build_universe_panel([srm], start="2022-10-01", end="2023-01-31")

    assert panel.metadata["delisting_reference_ts_utc"] == "2023-01-31T00:00:00Z"
    assert panel.metadata["n_symbols_delisted_within_window"] == 1


def test_the_days_after_the_last_daily_partition_are_still_untradeable(recorded_bucket):
    """The tail is exact, not open-ended: 2026-08-31 has no partition yet."""
    btc = coverage_for_symbol("BTCUSDT", "1d", open_url=btc_bucket(recorded_bucket))

    panel = build_universe_panel([btc], start="2026-08-01", end="2026-09-05")

    assert panel.tradeable["BTCUSDT"].loc["2026-08-30T00:00:00Z"]
    assert not panel.tradeable["BTCUSDT"].loc["2026-08-31T00:00:00Z":].any()


# --- Coverage is only what we could verify -----------------------------------


def test_a_month_published_without_a_checksum_is_not_coverage():
    """We would refuse to load it, so it must not claim to be tradeable either."""
    keys = keys_for("FAKEUSDT", "1d", ["2021-01", "2021-02"], without_checksum={"2021-02"})

    assert coverage_from_keys("FAKEUSDT", "1d", keys).months == ("2021-01",)


def test_other_symbols_and_intervals_in_a_listing_are_ignored():
    keys = keys_for("BTCUSDT", "1d", ["2021-01"]) + keys_for("BTCUSDT", "1h", ["2021-02"])
    keys.append(f"{KLINES}/BTCUSDT/1d/BTCUSDT-1d-2021-03-04.zip")

    assert coverage_from_keys("BTCUSDT", "1d", keys).months == ("2021-01",)


# --- The panel ---------------------------------------------------------------


def test_an_asset_whose_coverage_ends_mid_sample_is_untradeable_from_then_on():
    """The invariant this whole module exists for."""
    panel = build_universe_panel(
        [
            coverage("GONEUSDT", months_between("2021-01", "2021-03")),
            coverage("STAYSUSDT", months_between("2021-01", "2021-06")),
        ],
        start="2021-01-01",
        end="2021-06-30",
    )
    gone = panel.tradeable["GONEUSDT"]

    assert gone.loc["2021-03-30T00:00:00Z"]
    assert gone.loc["2021-03-31T00:00:00Z"]
    assert not gone.loc["2021-04-01T00:00:00Z"]
    assert not gone.loc["2021-04-01T00:00:00Z":].any()
    assert panel.tradeable["STAYSUSDT"].all()
    assert panel.tradeable_on("2021-02-15") == ("GONEUSDT", "STAYSUSDT")
    assert panel.tradeable_on("2021-04-15") == ("STAYSUSDT",)


def test_the_archive_floor_is_in_the_metadata_and_in_the_flags():
    """A window opening before the archive is not a window of dead assets."""
    panel = build_universe_panel(
        [coverage("BTCUSDT", ["2017-08", "2017-09"])],
        start="2017-01-01",
        end="2017-09-30",
    )
    btc = panel.tradeable["BTCUSDT"]

    assert panel.metadata["archive_floor_ts_utc"] == "2017-08-17T00:00:00Z"
    # 2017-01-01 through 2017-08-16 inclusive.
    assert panel.metadata["n_dates_before_archive_floor"] == 228
    assert not btc.loc[:"2017-08-16T00:00:00Z"].any()
    assert btc.loc["2017-08-17T00:00:00Z":].all()
    assert panel.metadata["symbols"]["BTCUSDT"]["first_tradeable_ts_utc"] == (
        "2017-08-17T00:00:00Z"
    )


def test_a_hole_in_coverage_is_a_hole_in_the_universe():
    """A symbol halted for a month was not tradeable that month."""
    panel = build_universe_panel(
        [coverage("HALTUSDT", ["2021-01", "2021-03"])],
        start="2021-01-01",
        end="2021-03-31",
    )
    halted = panel.tradeable["HALTUSDT"]

    assert halted.loc["2021-01-31T00:00:00Z"]
    assert not halted.loc["2021-02-01T00:00:00Z":"2021-02-28T00:00:00Z"].any()
    assert halted.loc["2021-03-01T00:00:00Z"]


def test_a_supplied_bar_span_narrows_a_mid_month_delisting_to_its_last_bar():
    """SRMUSDT's final partition runs to 2022-11-28; the month runs to the 30th."""
    panel = build_universe_panel(
        [coverage("SRMUSDT", months_between("2022-09", "2022-11"))],
        start="2022-11-01",
        end="2022-12-31",
        bar_span_by_symbol={
            "SRMUSDT": (
                pd.Timestamp("2022-09-01T00:00:00Z"),
                pd.Timestamp("2022-11-28T00:00:00Z"),
            )
        },
    )
    srm = panel.tradeable["SRMUSDT"]

    assert srm.loc["2022-11-28T00:00:00Z"]
    assert not srm.loc["2022-11-29T00:00:00Z"]
    assert not srm.loc["2022-11-30T00:00:00Z"]
    assert panel.metadata["symbols"]["SRMUSDT"]["last_tradeable_ts_utc"] == (
        "2022-11-28T00:00:00Z"
    )
    assert panel.metadata["symbols"]["SRMUSDT"]["narrowed_to_bars"]


def test_the_panel_reports_what_it_was_built_from():
    panel = build_universe_panel(
        [
            coverage("GONEUSDT", months_between("2021-01", "2021-03")),
            coverage("STAYSUSDT", months_between("2021-01", "2021-06")),
        ],
        start="2021-01-01",
        end="2021-06-30",
    )

    assert panel.metadata["n_symbols"] == 2
    assert panel.metadata["n_symbols_delisted_within_window"] == 1
    assert panel.metadata["interval"] == "1d"
    assert panel.metadata["n_dates"] == 31 + 28 + 31 + 30 + 31 + 30
    assert panel.metadata["coverage_requires_published_checksum"]
    assert "exchangeInfo" in panel.metadata["enumerated_from"]
    assert panel.tradeable.index.name == "ts_utc"
    assert panel.tradeable.dtypes.eq(bool).all()


def test_a_date_outside_the_panel_is_an_error_rather_than_an_empty_universe():
    panel = build_universe_panel(
        [coverage("BTCUSDT", ["2021-01"])], start="2021-01-01", end="2021-01-31"
    )

    with pytest.raises(UniverseError, match="outside the panel"):
        panel.tradeable_on("2021-02-01")


def test_a_symbol_given_twice_is_rejected():
    with pytest.raises(UniverseError, match="twice"):
        build_universe_panel(
            [coverage("BTCUSDT", ["2021-01"]), coverage("BTCUSDT", ["2021-02"])],
            start="2021-01-01",
            end="2021-02-28",
        )


def test_a_symbol_with_no_covered_months_is_rejected():
    with pytest.raises(UniverseError, match="no covered months"):
        build_universe_panel(
            [coverage("GHOSTUSDT", [])], start="2021-01-01", end="2021-01-31"
        )


def test_mixing_intervals_in_one_panel_is_rejected():
    with pytest.raises(UniverseError, match="one interval"):
        build_universe_panel(
            [
                coverage("BTCUSDT", ["2021-01"]),
                coverage("ETHUSDT", ["2021-01"], interval="1h"),
            ],
            start="2021-01-01",
            end="2021-01-31",
        )


# --- The one door to the bytes verifies them ---------------------------------


def test_fetching_a_covered_month_verifies_it_against_the_published_checksum(
    recorded_archive_file,
):
    from test_fetch import recorded_opener

    btc = coverage("BTCUSDT", ["2021-01"])
    _, expected_payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", "2021-01")

    payload, digest = fetch_covered_month(
        btc, "2021-01", open_url=recorded_opener(recorded_archive_file)
    )

    assert payload == expected_payload
    assert digest == checksum_text.split()[0]


def test_a_corrupted_partition_never_reaches_the_caller(recorded_archive_file):
    from test_fetch import recorded_opener

    btc = coverage("BTCUSDT", ["2021-01"])

    with pytest.raises(ChecksumMismatch):
        fetch_covered_month(
            btc, "2021-01", open_url=recorded_opener(recorded_archive_file, corrupt=True)
        )


def test_a_month_outside_coverage_is_refused_before_any_download():
    srm = coverage("SRMUSDT", months_between("2020-08", "2022-11"))

    def open_url(url):
        raise AssertionError("a month outside coverage must not be fetched")

    with pytest.raises(SymbolNotCovered, match="2022-12"):
        fetch_covered_month(srm, "2022-12", open_url=open_url)


# --- The composed path -------------------------------------------------------


def test_the_whole_universe_is_built_from_the_bucket_in_one_call(recorded_bucket):
    """Enumerate, gather coverage per symbol, and emit one panel."""
    panel = build_archive_universe(
        start="2022-10-01",
        end="2023-01-31",
        quote_asset="USDT",
        prefix=f"{KLINES}/",
        search_prefix=SR_PREFIX,
        open_url=recorded_bucket(
            {
                SR_PREFIX: SR_PAGES,
                f"{KLINES}/SRMUSDT/1d/": ["SRMUSDT-1d.xml"],
                (
                    f"{DAILY_KLINES}/SRMUSDT/1d/",
                    daily_tail_marker("SRMUSDT", "1d", "2022-11"),
                ): ["SRMUSDT-1d-daily-tail.xml"],
            }
        ),
    )

    assert list(panel.tradeable.columns) == ["SRMUSDT"]
    assert panel.tradeable_on("2022-11-15") == ("SRMUSDT",)
    assert panel.tradeable_on("2023-01-15") == ()
    assert panel.metadata["n_symbols_delisted_within_window"] == 1
    assert panel.metadata["n_symbols_narrowed_to_bars"] == 0


def test_metadata_edges_stay_inside_the_window_they_describe():
    """Read off the flags, so they cannot name a date the panel does not hold."""
    panel = build_universe_panel(
        [coverage("SRMUSDT", months_between("2020-08", "2022-11"))],
        start="2021-01-01",
        end="2021-12-31",
    )
    srm = panel.metadata["symbols"]["SRMUSDT"]

    assert srm["first_month"] == "2020-08"
    assert srm["first_tradeable_ts_utc"] == "2021-01-01T00:00:00Z"
    assert srm["last_tradeable_ts_utc"] == "2021-12-31T00:00:00Z"
    assert panel.metadata["n_symbols_delisted_within_window"] == 0


def test_a_symbol_the_window_misses_entirely_reports_no_tradeable_dates():
    panel = build_universe_panel(
        [coverage("SRMUSDT", ["2022-11"])], start="2023-01-01", end="2023-01-31"
    )
    srm = panel.metadata["symbols"]["SRMUSDT"]

    assert not panel.tradeable["SRMUSDT"].any()
    assert srm["first_tradeable_ts_utc"] is None
    assert srm["last_tradeable_ts_utc"] is None


def test_a_bar_span_is_read_off_built_bars():
    """The bridge from derived bars back to the panel's month-resolution edges."""
    bars = pd.DataFrame(
        {"close": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ["2022-11-26", "2022-11-27", "2022-11-28"], tz="UTC", name="ts_utc"
        ),
    )

    assert bar_span_from_bars(bars) == (
        pd.Timestamp("2022-11-26T00:00:00Z"),
        pd.Timestamp("2022-11-28T00:00:00Z"),
    )


def test_a_bar_span_needs_bars():
    with pytest.raises(UniverseError, match="at least one bar"):
        bar_span_from_bars(pd.DataFrame())
