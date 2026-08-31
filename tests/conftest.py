from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

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

KLINES = "data/spot/monthly/klines"
DAILY_KLINES = "data/spot/daily/klines"
# Enumeration over the real `SR` sub-prefix, recorded two keys at a time so the
# pagination path in the fixtures is real rather than simulated.
SR_PREFIX = f"{KLINES}/SR"
SR_PAGES = ("klines-SR-page1.xml", "klines-SR-page2.xml", "klines-SR-page3.xml")


def daily_tail_marker(symbol: str, interval: str, after_month: str) -> str:
    """The exclusive lower bound `coverage_for_symbol` seeds its tail listing with."""
    prefix = f"{DAILY_KLINES}/{symbol}/{interval}/"
    return f"{prefix}{symbol}-{interval}-{after_month}-99"


@pytest.fixture
def recorded_listing_page():
    """Read a recorded `ListBucketResult` page from the archive bucket.

    Like the kline fixtures, these bytes were listed once and committed. Nothing
    in the suite lists the live bucket.
    """

    def read(name: str) -> bytes:
        return (LISTINGS / name).read_bytes()

    return read


@pytest.fixture
def recorded_bucket(recorded_listing_page):
    """Serve recorded listing pages as an injectable `open_url`.

    Takes `{prefix: [page names]}`, or `{(prefix, start_after): [page names]}`
    where a listing is seeded with an opening marker. Successive pages under one
    prefix are chained by the marker each page itself reports, so the pagination
    a test exercises is the bucket's own.

    Requests are matched on `(prefix, marker)` and not on the whole URL, because
    the page size a caller asks for is its own business and the fixtures were
    recorded at several.
    """
    from crypto_momentum.data.archive_listing import parse_listing_page
    from crypto_momentum.data.fetch import ArchiveUnavailable

    def build(sources: dict) -> "Callable[[str], bytes]":
        served: dict[tuple[str, str | None], bytes] = {}
        for key, names in sources.items():
            prefix, marker = key if isinstance(key, tuple) else (key, None)
            for name in names:
                payload = recorded_listing_page(name)
                served[(prefix, marker)] = payload
                marker = parse_listing_page(payload).next_marker

        def open_url(url: str) -> bytes:
            query = parse_qs(urlparse(url).query)
            key = (query["prefix"][0], query.get("marker", [None])[0])
            if key not in served:
                raise ArchiveUnavailable(f"no recorded page for {key}")
            return served[key]

        return open_url

    return build
