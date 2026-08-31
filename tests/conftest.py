from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = FIXTURES / "binance" / "spot" / "monthly" / "klines"


@pytest.fixture
def recorded_archive_file():
    """Read a recorded `data.binance.vision` monthly kline file and its checksum.

    These bytes were downloaded once and committed. The adapter is never tested
    against the live archive.
    """

    def read(symbol: str, interval: str, month: str) -> tuple[str, bytes, str]:
        directory = ARCHIVE / symbol / interval
        filename = f"{symbol}-{interval}-{month}.zip"
        payload = (directory / filename).read_bytes()
        checksum_text = (directory / f"{filename}.CHECKSUM").read_text()
        return filename, payload, checksum_text

    return read


LISTINGS = FIXTURES / "binance" / "listing"


@pytest.fixture
def recorded_listing_page():
    """Read a recorded `ListBucketResult` page from the archive bucket.

    Like the kline fixtures, these bytes were listed once and committed. Nothing
    in the suite lists the live bucket.
    """

    def read(name: str) -> bytes:
        return (LISTINGS / name).read_bytes()

    return read
