"""A synthetic archive and vendor panel, shared by the end-to-end suites.

The archive and the CoinMarketCap panel are both synthesised rather than
recorded, because what these suites test is the wiring — coverage, bars, the
point-in-time Universe, policy, the vendor join, the simulator, the result file
— and a fixture whose prices we chose is the only kind that lets the wiring be
checked against a number worked out by hand. The recorded-bytes tests live one
layer down, against the adapters this exercises.

Not a test module: it holds the fixtures two of them import. It lives beside
them rather than in `conftest.py` because nothing else in the suite wants a
whole synthetic venue in scope.
"""

import hashlib
import io
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from crypto_momentum.data.cmc_panel import CmcPanelStore
from crypto_momentum.data.fetch import ArchiveUnavailable
from crypto_momentum.runner import Workspace

RUN_AT = "2026-08-31T09:00:00Z"
MONTHS = ("2021-01", "2021-02", "2021-03")

# Six names, so a quintile is two and the cross-section clears MIN_UNIVERSE.
# Each compounds at its own steady rate, so the ranking never changes and the
# expected selection can be read straight off this table.
DAILY_RATE = {
    "BTCUSDT": 0.010,
    "ETHUSDT": 0.008,
    "BNBUSDT": 0.006,
    "ADAUSDT": 0.004,
    "XRPUSDT": 0.002,
    "DOGEUSDT": 0.000,
}
CMC_ID = {
    "BTCUSDT": (1, "BTC", "Bitcoin"),
    "ETHUSDT": (1027, "ETH", "Ethereum"),
    "BNBUSDT": (1839, "BNB", "BNB"),
    "ADAUSDT": (2010, "ADA", "Cardano"),
    "XRPUSDT": (52, "XRP", "XRP"),
    "DOGEUSDT": (74, "DOGE", "Dogecoin"),
}
# Deliberately the reverse of the return ranking, so a value-weighted portfolio
# cannot be mistaken for an equal-weighted or a signal-weighted one.
MARKET_CAP = {
    "BTCUSDT": 1e9,
    "ETHUSDT": 2e9,
    "BNBUSDT": 3e9,
    "ADAUSDT": 4e9,
    "XRPUSDT": 5e9,
    "DOGEUSDT": 6e9,
}

WINDOW = pd.date_range("2021-01-01", "2021-03-31", freq="D", tz="UTC")


def close_of(symbol: str, day: pd.Timestamp) -> float:
    """The synthetic close: a steady compounding ramp from 100 on the first day."""
    step = (day - WINDOW[0]).days
    return 100.0 * (1.0 + DAILY_RATE[symbol]) ** step


def klines_zip(symbol: str, month: str) -> bytes:
    """A monthly partition in the archive's own shape: a zip of a headerless CSV."""
    days = [day for day in WINDOW if day.strftime("%Y-%m") == month]
    rows = []
    for day in days:
        close = close_of(symbol, day)
        # The open is the previous close, so a fill price is checkable by hand.
        open_price = close / (1.0 + DAILY_RATE[symbol])
        open_ms = int(day.value // 1_000_000)
        close_ms = open_ms + 86_399_999
        rows.append(
            f"{open_ms},{open_price:.8f},{close:.8f},{open_price:.8f},{close:.8f},"
            f"1000.0,{close_ms},1000000.0,100,500.0,500000.0,0"
        )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"{symbol}-1d-{month}.csv", "\n".join(rows) + "\n")
    return payload.getvalue()


def listing_page(keys: list[str]) -> bytes:
    contents = "".join(f"<Contents><Key>{key}</Key></Contents>" for key in keys)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
    ).encode()


def panel_csv() -> bytes:
    """A weekly vendor panel covering the window, plus what the store demands of it.

    The first row reaches back to CoinMarketCap's own first snapshot so the
    stored manifest is not stamped with a window the payload does not cover, and
    the two known dead assets are present so the panel is not refused as
    survivorship-biased.
    """
    header = (
        "ts_utc,cmc_id,symbol,name,cmc_rank,price_usd,market_cap_usd,"
        "volume_24h_usd,circulating_supply"
    )
    rows = ["2013-04-28,1,BTC,Bitcoin,1,135.30,1500000000.0,0.0,11091000.0"]
    # The dead assets, listed once each and never again — which is exactly the
    # shape a survivorship-free panel has.
    rows.append("2018-01-07,827,BCC,BitConnect,207,29.10,268000000.0,3100000.0,9200000.0")
    rows.append("2020-08-16,6187,SRM,Serum,144,1.42,68000000.0,41000000.0,47900000.0")
    for snapshot in pd.date_range("2020-12-06", "2021-04-04", freq="7D"):
        for rank, (symbol, (cmc_id, ticker, name)) in enumerate(CMC_ID.items(), start=1):
            rows.append(
                f"{snapshot.date()},{cmc_id},{ticker},{name},{rank},1.0,"
                f"{MARKET_CAP[symbol]},1000000.0,1000000.0"
            )
    return ("\n".join([header, *rows]) + "\n").encode()


@pytest.fixture
def archive():
    """An opener serving the synthetic archive. Nothing here touches the network."""
    files: dict[str, bytes] = {}
    listings: dict[str, bytes] = {}
    for symbol in DAILY_RATE:
        keys = []
        for month in MONTHS:
            filename = f"{symbol}-1d-{month}.zip"
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{filename}"
            payload = klines_zip(symbol, month)
            files[url] = payload
            files[url + ".CHECKSUM"] = (
                f"{hashlib.sha256(payload).hexdigest()}  {filename}".encode()
            )
            key = f"data/spot/monthly/klines/{symbol}/1d/{filename}"
            keys.extend([key, key + ".CHECKSUM"])
        listings[f"data/spot/monthly/klines/{symbol}/1d/"] = listing_page(keys)
        # The window is long over, so the archive has rolled every month up and
        # there are no daily-only partitions at the tail.
        listings[f"data/spot/daily/klines/{symbol}/1d/"] = listing_page([])

    def open_url(url: str) -> bytes:
        query = parse_qs(urlparse(url).query)
        if "prefix" in query:
            prefix = query["prefix"][0]
            if prefix not in listings:
                raise ArchiveUnavailable(f"no listing for {prefix}")
            return listings[prefix]
        if url not in files:
            raise ArchiveUnavailable(f"404 for {url}")
        return files[url]

    return open_url


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(args, cwd=root, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (root / "README.md").write_text("cross-sectional\n")
    run("git", "add", ".")
    run("git", "commit", "-qm", "first")

    workspace = Workspace.under(root)
    # The policy artefacts and the vendor override table are committed data, so
    # the run reads the repo's own copies rather than a test-local invention.
    source = Path(__file__).parent.parent
    for relative in (
        "policy/exclusions-v1.toml",
        "policy/tokocrypto-listing-v1.toml",
        "configs/vendor-symbol-map.toml",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / relative).read_bytes())

    CmcPanelStore(workspace.raw_root).write(
        panel_csv(), pulled_at_utc=RUN_AT, window_start=date(2013, 4, 28)
    )
    return workspace
