"""The CoinMarketCap panel: one immutable pull, survivorship-free, checksummed.

Per ADR-0008 this is a single one-time pull, not a live dependency. Nothing here
touches R or the network — the R invocation is injected, so these tests pin the
one-time discipline without a working `crypto2` install.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from crypto_momentum.data.cmc_panel import (
    PANEL_START,
    CmcPanelStore,
    PanelAlreadyStored,
    PanelMissing,
    PanelWindowNotCovered,
    SurvivorshipBiasedPanel,
    parse_panel_csv,
    pull_panel,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coinmarketcap"
PULLED_AT = "2026-08-31T00:00:00Z"

# The fixture is a hand-cut slice, not the real panel, so its pulls ask for the
# window it actually has. The real pull asks for PANEL_START and is checked
# against it — see test_a_panel_that_does_not_reach_the_requested_start.
FIXTURE_START = date(2017, 1, 1)


@pytest.fixture
def panel_csv() -> bytes:
    return (FIXTURES / "cmc-panel-sample.csv").read_bytes()


@pytest.fixture
def store(tmp_path) -> CmcPanelStore:
    return CmcPanelStore(tmp_path / "raw")


@pytest.fixture
def survivors_only() -> bytes:
    """The same panel as it would come back from the survivorship-biased endpoint."""
    return b"\n".join(
        line
        for line in (FIXTURES / "cmc-panel-sample.csv").read_bytes().splitlines()
        if b",827," not in line and b",6187," not in line
    )


def _runner(payload: bytes, calls: list[str]):
    """An injected stand-in for `Rscript scripts/pull_cmc_panel.R`."""

    def run(destination: Path) -> None:
        calls.append(str(destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    return run


# --- the pull is one-time -------------------------------------------------


def test_the_first_pull_invokes_crypto2_and_stores_the_panel(store, panel_csv):
    calls: list[str] = []

    path = pull_panel(
        store,
        run_pull=_runner(panel_csv, calls),
        pulled_at_utc=PULLED_AT,
        window_start=FIXTURE_START,
    )

    assert len(calls) == 1
    assert path.read_bytes() == panel_csv


def test_rebuilding_does_not_re_fetch_the_panel(store, panel_csv):
    """`crypto2` breaches CoinMarketCap's terms; a second call is a real cost."""
    calls: list[str] = []
    runner = _runner(panel_csv, calls)

    first = pull_panel(
        store, run_pull=runner, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START
    )
    second = pull_panel(
        store,
        run_pull=runner,
        pulled_at_utc="2026-09-30T00:00:00Z",
        window_start=FIXTURE_START,
    )

    assert len(calls) == 1
    assert first == second
    assert store.manifest()["pulled_at_utc"] == PULLED_AT


def test_writing_over_a_stored_panel_is_an_error_not_an_overwrite(store, panel_csv):
    store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    with pytest.raises(PanelAlreadyStored, match="cmc"):
        store.write(b"replacement", pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    assert store.read() == panel_csv


def test_a_stored_panel_is_not_writable(store, panel_csv):
    path = store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    assert path.stat().st_mode & 0o222 == 0


def test_reading_a_panel_that_was_never_pulled_says_so(store):
    assert not store.has_panel()
    with pytest.raises(PanelMissing):
        store.read()


# --- checksum and provenance ---------------------------------------------


def test_the_panel_is_checksummed_and_the_digest_reads_back(store, panel_csv):
    import hashlib

    store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    assert store.manifest()["sha256"] == hashlib.sha256(panel_csv).hexdigest()


def test_the_manifest_records_what_the_protocol_asks_for(store, panel_csv):
    path = store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)
    manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())

    assert manifest["vendor"] == "coinmarketcap"
    assert manifest["source"] == "crypto2::crypto_listings(which='historical')"
    assert manifest["symbol_convention"] == "CoinMarketCap numeric id; symbol is not stable"
    assert manifest["bar_close_convention"] == "snapshot as of 00:00 UTC on ts_utc"
    assert manifest["timezone"] == "UTC"
    assert manifest["window_start"] == FIXTURE_START.isoformat()
    assert manifest["window_end"] == "2026-08-31"
    assert manifest["pulled_at_utc"] == PULLED_AT
    assert manifest["bytes"] == len(panel_csv)


def test_the_manifest_records_the_dead_coins_it_found_not_a_claim_of_freedom(
    store, panel_csv
):
    """`survivorship_free = true` would be an assertion about a file we did not read."""
    store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    assert store.manifest()["dead_assets_present"] == [827, 6187]


def test_the_manifest_separates_the_window_asked_for_from_what_came_back(
    store, panel_csv
):
    """The request and the delivery are different facts and both get recorded.

    Asking for 2018 onward and being handed history back to 2017 is fine — the
    panel covers the request. What the manifest must not do is round the two
    into one number, so a later reader can see which is which.
    """
    store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=date(2018, 1, 1))
    manifest = store.manifest()

    assert manifest["window_start"] == "2018-01-01"
    assert manifest["first_snapshot"] == "2017-01-01"
    assert manifest["last_snapshot"] == "2023-01-01"


def test_writing_a_short_panel_under_a_long_window_is_refused(store, panel_csv):
    """The check lives in `write`, not only in `pull_panel`.

    A manifest is only worth trusting if no path can write one it has not
    earned, so stamping the fixture's 2017 history with the real 2013 window
    has to fail here exactly as it does through the pull.
    """
    with pytest.raises(PanelWindowNotCovered, match="2013-04-28"):
        store.write(panel_csv, pulled_at_utc=PULLED_AT, window_start=PANEL_START)

    assert not store.has_panel()


def test_a_panel_that_does_not_reach_the_requested_start_is_refused(store, panel_csv):
    """A short panel stored under a long window would mislead every later reader."""

    def run(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(panel_csv)

    with pytest.raises(PanelWindowNotCovered, match="2013-04-28"):
        pull_panel(store, run_pull=run, pulled_at_utc=PULLED_AT)

    assert not store.has_panel()


def test_a_biased_panel_cannot_be_stored_even_by_writing_it_directly(
    store, survivors_only
):
    """The check lives in `write`, so no path can put a biased panel on disk."""
    with pytest.raises(SurvivorshipBiasedPanel):
        store.write(
            survivors_only, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START
        )

    assert not store.has_panel()


# --- the panel is survivorship-free --------------------------------------


def test_the_panel_covers_the_first_coinmarketcap_snapshot_onward():
    assert PANEL_START == date(2013, 4, 28)


def test_a_known_dead_coin_is_present_in_the_panel(panel_csv):
    """BitConnect collapsed in January 2018. A biased panel drops it entirely."""
    panel = parse_panel_csv(panel_csv)

    bitconnect = panel[panel["cmc_id"] == 827]
    assert not bitconnect.empty
    assert bitconnect["symbol"].iloc[0] == "BCC"
    assert bitconnect.index.max().date() == date(2018, 1, 7)


def test_a_panel_missing_its_dead_coins_is_rejected_at_pull_time(
    store, survivors_only
):
    """`cryptocurrency/historical` returns zero rows for delisted coins (ADR-0008)."""

    def run(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(survivors_only)

    with pytest.raises(SurvivorshipBiasedPanel, match="827"):
        pull_panel(store, run_pull=run, pulled_at_utc=PULLED_AT, window_start=FIXTURE_START)

    assert not store.has_panel(), "a biased panel must not reach data/raw/"


# --- parsing --------------------------------------------------------------


def test_one_row_is_one_asset_on_one_snapshot_date(panel_csv):
    panel = parse_panel_csv(panel_csv)

    assert panel.index.name == "ts_utc"
    assert str(panel.index.tz) == "UTC"
    assert not panel.reset_index().duplicated(["ts_utc", "cmc_id"]).any()


def test_market_cap_parses_as_a_float_in_us_dollars(panel_csv):
    panel = parse_panel_csv(panel_csv)

    btc_at_start = panel[(panel["cmc_id"] == 1) & (panel.index == "2017-01-01")]
    assert btc_at_start["market_cap_usd"].iloc[0] == pytest.approx(15_491_000_000.0)


def test_the_same_id_can_carry_different_symbols_at_different_dates(panel_csv):
    """Terra Classic was LUNA before the May 2022 unwind and LUNC after it."""
    panel = parse_panel_csv(panel_csv)

    terra_classic = panel[panel["cmc_id"] == 4172]
    assert set(terra_classic["symbol"]) == {"LUNA", "LUNC"}
