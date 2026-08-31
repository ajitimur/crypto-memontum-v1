"""Record the bucket-listing fixtures under `tests/fixtures/binance/listing/`.

Run once, from the repo root, when a fixture needs re-recording:

    uv run python scripts/record_listing_fixtures.py

This is the only code in the repo that lists the live bucket. The suite reads
the committed bytes it writes, so a test never depends on what Binance is
serving today.

Three fixtures, each chosen for what it pins down:

- `klines-SR-page*.xml` — symbol enumeration, listed two keys at a time so the
  pagination path is real rather than simulated. Contains `SRMUSDT` alongside
  four other-quote SRM pairs, so the quote filter has something to reject.
- `SRMUSDT-1d.xml` — the survivorship witness: an asset delisted in 2022 whose
  partitions the archive still publishes.
- `BTCUSDT-1d.xml` — a still-trading asset whose coverage opens at the
  2017-08-17 archive floor.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from crypto_momentum.data.archive_listing import listing_url, parse_listing_page

OUT = Path("tests/fixtures/binance/listing")
KLINES_PREFIX = "data/spot/monthly/klines"
# Small enough that the recorded SR listing spans three real pages.
ENUMERATION_MAX_KEYS = 2


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def record(name: str, prefix: str, marker: str | None = None, max_keys: int = 1000):
    payload = get(listing_url(prefix, marker=marker, max_keys=max_keys))
    page = parse_listing_page(payload)
    (OUT / name).write_bytes(payload)
    print(
        f"{name}: {len(payload)}B prefixes={len(page.prefixes)} "
        f"keys={len(page.keys)} truncated={page.is_truncated}"
    )
    return page


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    marker = None
    for page_number in range(1, 9):
        page = record(
            f"klines-SR-page{page_number}.xml",
            f"{KLINES_PREFIX}/SR",
            marker=marker,
            max_keys=ENUMERATION_MAX_KEYS,
        )
        if not page.is_truncated:
            break
        marker = page.next_marker

    record("SRMUSDT-1d.xml", f"{KLINES_PREFIX}/SRMUSDT/1d/")
    record("BTCUSDT-1d.xml", f"{KLINES_PREFIX}/BTCUSDT/1d/")


if __name__ == "__main__":
    main()
