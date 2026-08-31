"""Fetching verifies against the published checksum before anything is stored.

The opener is injected, so no test in this suite reaches the live archive.
"""

import pytest

from crypto_momentum.data.binance_archive import ChecksumMismatch, monthly_klines_file
from crypto_momentum.data.fetch import ArchiveUnavailable, fetch_archive_file


def recorded_opener(recorded_archive_file, month="2021-01", corrupt=False):
    filename, payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", month)
    if corrupt:
        payload = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    served = {
        f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/{filename}": payload,
        f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/{filename}.CHECKSUM": (
            checksum_text.encode()
        ),
    }

    def open_url(url: str) -> bytes:
        if url not in served:
            raise ArchiveUnavailable(f"404 for {url}")
        return served[url]

    return open_url


def test_a_verified_download_returns_its_payload_and_digest(recorded_archive_file):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")
    _, expected_payload, checksum_text = recorded_archive_file("BTCUSDT", "1d", "2021-01")

    payload, digest = fetch_archive_file(
        archive_file, open_url=recorded_opener(recorded_archive_file)
    )

    assert payload == expected_payload
    assert digest == checksum_text.split()[0]


def test_a_corrupted_download_is_never_returned(recorded_archive_file):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2021-01")

    with pytest.raises(ChecksumMismatch):
        fetch_archive_file(
            archive_file, open_url=recorded_opener(recorded_archive_file, corrupt=True)
        )


def test_a_month_the_archive_does_not_publish_is_reported_as_unavailable(
    recorded_archive_file,
):
    archive_file = monthly_klines_file("BTCUSDT", "1d", "2016-01")

    with pytest.raises(ArchiveUnavailable, match="2016-01"):
        fetch_archive_file(archive_file, open_url=recorded_opener(recorded_archive_file))
