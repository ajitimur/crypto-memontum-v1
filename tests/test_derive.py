"""`data/derived/` is rebuilt from raw by a script and is safe to delete."""

import pandas as pd
import pytest

from crypto_momentum.data.binance_archive import monthly_klines_file
from crypto_momentum.data.raw_store import RawStore
from crypto_momentum.derive import (
    DerivedStore,
    GapInWindow,
    build_daily_bars,
    rebuild_daily_bars,
)

FETCHED_AT = "2026-08-31T00:00:00Z"


@pytest.fixture
def raw_store(tmp_path, recorded_archive_file):
    store = RawStore(tmp_path / "raw")

    def load(*months):
        for month in months:
            archive_file = monthly_klines_file("BTCUSDT", "1d", month)
            _, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", month)
            store.write(
                archive_file,
                payload,
                sha256=checksum_text.split()[0],
                fetched_at_utc=FETCHED_AT,
            )
        return store

    return load


def test_two_stored_months_build_one_contiguous_daily_series(raw_store):
    store = raw_store("2021-01", "2021-02")

    bars = build_daily_bars(store, "BTCUSDT", "1d", ["2021-01", "2021-02"])

    assert len(bars) == 31 + 28
    assert bars.index[0] == pd.Timestamp("2021-01-01T00:00:00Z")
    assert bars.index[-1] == pd.Timestamp("2021-02-28T00:00:00Z")
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique


def test_a_missing_day_between_months_fails_loudly(raw_store):
    """A hole in the window is a data problem, not something to interpolate over."""
    store = raw_store("2021-01", "2025-03")

    with pytest.raises(GapInWindow, match="2021-02-01"):
        build_daily_bars(store, "BTCUSDT", "1d", ["2021-01", "2025-03"])


def test_derived_bars_round_trip_with_their_timezone_intact(tmp_path, raw_store):
    store = raw_store("2021-01")
    derived = DerivedStore(tmp_path / "derived")
    bars = build_daily_bars(store, "BTCUSDT", "1d", ["2021-01"])

    derived.write_bars("BTCUSDT", "1d", bars)

    pd.testing.assert_frame_equal(derived.read_bars("BTCUSDT", "1d"), bars)


def test_derived_data_can_be_deleted_and_rebuilt_from_raw(tmp_path, raw_store):
    store = raw_store("2021-01", "2021-02")
    derived = DerivedStore(tmp_path / "derived")
    months = ["2021-01", "2021-02"]
    first = rebuild_daily_bars(store, derived, "BTCUSDT", "1d", months)

    derived.clear()
    assert not derived.has("BTCUSDT", "1d")
    rebuilt = rebuild_daily_bars(store, derived, "BTCUSDT", "1d", months)

    pd.testing.assert_frame_equal(rebuilt, first)


def test_rebuilding_overwrites_rather_than_refusing(tmp_path, raw_store):
    """Unlike raw, derived data is disposable — a rebuild is the normal path."""
    store = raw_store("2021-01")
    derived = DerivedStore(tmp_path / "derived")

    rebuild_daily_bars(store, derived, "BTCUSDT", "1d", ["2021-01"])
    rebuild_daily_bars(store, derived, "BTCUSDT", "1d", ["2021-01"])

    assert derived.has("BTCUSDT", "1d")
