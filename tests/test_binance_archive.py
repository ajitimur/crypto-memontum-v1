"""Seam 2: the Binance archive adapter, tested against recorded fixture files."""

import zipfile

import pandas as pd
import pytest

from crypto_momentum.data.binance_archive import (
    ChecksumMismatch,
    MalformedArchiveFile,
    monthly_klines_file,
    parse_klines_zip,
    verify_sha256,
)


def test_monthly_klines_file_addresses_the_published_layout():
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    assert archive_file.filename == "BTCUSDT-1d-2021-01.zip"
    assert archive_file.url == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2021-01.zip"
    )
    assert archive_file.checksum_url == archive_file.url + ".CHECKSUM"


def test_a_recorded_file_matches_its_published_checksum(recorded_archive_file):
    filename, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", "2021-01")

    verify_sha256(payload, checksum_text, filename)  # does not raise


def test_a_corrupted_download_fails_loudly(recorded_archive_file):
    filename, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", "2021-01")
    corrupted = payload[:-1] + bytes([payload[-1] ^ 0xFF])

    with pytest.raises(ChecksumMismatch, match=filename):
        verify_sha256(corrupted, checksum_text, filename)


def test_a_checksum_for_a_different_file_is_refused(recorded_archive_file):
    filename, payload, _ = recorded_archive_file("BTCUSDT", "1d", "2021-01")
    _, _, other_checksum = recorded_archive_file("BTCUSDT", "1d", "2021-02")

    with pytest.raises(ChecksumMismatch, match="names"):
        verify_sha256(payload, other_checksum, filename)


def test_parses_a_recorded_month_into_daily_bars(recorded_archive_file):
    _, payload, _ = recorded_archive_file("BTCUSDT", "1d", "2021-01")

    bars = parse_klines_zip(payload)

    assert len(bars) == 31
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.name == "ts_utc"
    assert str(bars.index.tz) == "UTC"
    assert bars.index[0] == pd.Timestamp("2021-01-01T00:00:00Z")
    assert bars.index[-1] == pd.Timestamp("2021-01-31T00:00:00Z")
    # Read off the published CSV: the first bar of January 2021.
    assert bars["open"].iloc[0] == pytest.approx(28923.63)
    assert bars["close"].iloc[0] == pytest.approx(29331.69)
    assert bars["high"].iloc[0] == pytest.approx(29600.00)
    assert bars["low"].iloc[0] == pytest.approx(28624.57)


def test_the_index_is_the_bar_open_not_its_close(recorded_archive_file):
    """The archive carries open_time and close_time. We index on open_time, so a
    bar stamped 2021-01-01 covers 00:00:00Z to 23:59:59Z on that date."""
    _, payload, _ = recorded_archive_file("BTCUSDT", "1d", "2021-02")

    bars = parse_klines_zip(payload)

    assert bars.index[0] == pd.Timestamp("2021-02-01T00:00:00Z")
    assert bars.index[-1] == pd.Timestamp("2021-02-28T00:00:00Z")
    assert bars["close"].iloc[-1] == pytest.approx(45135.66)


def test_bars_are_unique_and_ordered(recorded_archive_file):
    _, payload, _ = recorded_archive_file("BTCUSDT", "1d", "2021-01")

    bars = parse_klines_zip(payload)

    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique


def test_microsecond_timestamps_are_read_as_the_same_calendar_days(recorded_archive_file):
    """Binance switched the archive's time unit from milliseconds to microseconds
    in 2025. Read the wrong unit and the bars land in 1970."""
    _, payload, _ = recorded_archive_file("BTCUSDT", "1d", "2025-03")

    bars = parse_klines_zip(payload)

    assert bars.index[0] == pd.Timestamp("2025-03-01T00:00:00Z")
    assert bars.index[-1] == pd.Timestamp("2025-03-31T00:00:00Z")
    assert len(bars) == 31


def test_a_zip_that_is_not_a_kline_csv_is_rejected(tmp_path):
    buffer = tmp_path / "junk.zip"
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("junk.csv", "a,b\n1,2\n")

    with pytest.raises(MalformedArchiveFile):
        parse_klines_zip(buffer.read_bytes())


def test_bytes_that_are_not_a_zip_are_rejected():
    with pytest.raises(MalformedArchiveFile):
        parse_klines_zip(b"<html>404</html>")
