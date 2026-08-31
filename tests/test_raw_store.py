"""Seam 3: `data/raw/` is append-only. Re-fetching a window is an error."""

import json

import pytest

from crypto_momentum.data.binance_archive import monthly_klines_file
from crypto_momentum.data.raw_store import RawStore, RawWindowAlreadyStored, RawWindowMissing

FETCHED_AT = "2026-08-31T00:00:00Z"
DIGEST = "6ff53d94f600e2a208882bfd5c00e2133cff8f07e57b7f373af29f85f86e0284"


@pytest.fixture
def store(tmp_path):
    return RawStore(tmp_path / "raw")


def test_a_stored_window_reads_back_byte_identical(store):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    store.write(archive_file, b"payload-bytes", sha256=DIGEST, fetched_at_utc=FETCHED_AT)

    assert store.read(archive_file) == b"payload-bytes"
    assert store.has(archive_file)


def test_refetching_an_existing_window_is_an_error_not_an_overwrite(store):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")
    store.write(archive_file, b"original", sha256=DIGEST, fetched_at_utc=FETCHED_AT)

    with pytest.raises(RawWindowAlreadyStored, match="BTCUSDT-1d-2021-01.zip"):
        store.write(archive_file, b"replacement", sha256=DIGEST, fetched_at_utc=FETCHED_AT)

    assert store.read(archive_file) == b"original"


def test_a_stored_file_is_not_writable(store):
    """Append-only is enforced by the filesystem, not only by the store's own check."""
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    path = store.write(archive_file, b"payload", sha256=DIGEST, fetched_at_utc=FETCHED_AT)

    assert path.stat().st_mode & 0o222 == 0


def test_the_manifest_records_what_the_protocol_asks_for(store):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    store.write(archive_file, b"payload", sha256=DIGEST, fetched_at_utc=FETCHED_AT)
    manifest = store.manifest(archive_file)

    assert manifest["venue"] == "binance-spot"
    assert manifest["symbol"] == "BTCUSDT"
    assert manifest["symbol_convention"] == "base+quote concatenated, uppercase"
    assert manifest["interval"] == "1d"
    assert manifest["month"] == "2021-01"
    assert manifest["timezone"] == "UTC"
    assert manifest["bar_close_convention"] == "index is bar open_time; bar covers open_time to next open"
    assert manifest["url"] == archive_file.url
    assert manifest["sha256"] == DIGEST
    assert manifest["fetched_at_utc"] == FETCHED_AT
    assert manifest["bytes"] == len(b"payload")


def test_the_manifest_is_written_beside_the_file_as_json(store):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    path = store.write(archive_file, b"payload", sha256=DIGEST, fetched_at_utc=FETCHED_AT)

    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    assert json.loads(sidecar.read_text())["sha256"] == DIGEST


def test_reading_a_window_that_was_never_fetched_names_it(store):
    archive_file = monthly_klines_file("ETHUSDT", "1d", "2019-05")

    assert not store.has(archive_file)
    with pytest.raises(RawWindowMissing, match="ETHUSDT-1d-2019-05.zip"):
        store.read(archive_file)


def test_windows_are_partitioned_by_symbol_and_interval(store):
    path = store.write(
        monthly_klines_file("BTCUSDT", "1d", "2021-01"),
        b"payload",
        sha256=DIGEST,
        fetched_at_utc=FETCHED_AT,
    )

    assert path.relative_to(store.root).as_posix() == (
        "binance/spot/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2021-01.zip"
    )
