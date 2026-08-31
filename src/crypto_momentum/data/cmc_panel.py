"""The CoinMarketCap market-capitalisation panel — pulled once, then frozen.

ADR-0008 buys market caps with a single immutable `crypto2` pull rather than a
live vendor dependency: the endpoints are undocumented, the route breaches
CoinMarketCap's terms, and every extra call is another chance at an IP ban. So
the discipline this module enforces is that the second pull never happens. Once
the CSV is in `data/raw/`, `pull_panel` returns it without invoking R at all.

Recorded source conventions (`docs/agents/quant-research.md` asks for these):

- **Vendor**: CoinMarketCap, via `crypto2::crypto_listings(which = "historical")`.
  That is the survivorship-free endpoint; `cryptocurrency/historical` returns
  zero rows for delisted coins and would quietly bias the Universe.
- **Symbol convention**: the numeric CoinMarketCap id is the identity. The
  ticker is *not* stable — CoinMarketCap renamed id 4172 from LUNA to LUNC —
  so nothing here joins on a symbol. See `symbol_map`.
- **Bar close convention**: each row is a snapshot as of 00:00 UTC on `ts_utc`,
  not a bar. Market cap on `ts_utc` is known at `ts_utc`, so a rebalance on that
  date may read it; a signal formed on it may not read the next one.
- **Timezone**: UTC throughout.
- **Window**: 2013-04-28, CoinMarketCap's first snapshot, onward.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from crypto_momentum.data.immutable import (
    RawArtifactAlreadyStored,
    RawArtifactMissing,
    read_manifest,
    sha256_of,
    write_immutable,
)

VENDOR = "coinmarketcap"
SOURCE = "crypto2::crypto_listings(which='historical')"
SYMBOL_CONVENTION = "CoinMarketCap numeric id; symbol is not stable"
BAR_CLOSE_CONVENTION = "snapshot as of 00:00 UTC on ts_utc"
TIMEZONE = "UTC"

# CoinMarketCap's first historical listings snapshot.
PANEL_START = date(2013, 4, 28)

# Snapshots were published weekly for most of the panel's life, so weekly is the
# grain available across the whole window. It also matches the weekly rebalance
# of the paper the Replication Gate reproduces.
PANEL_INTERVAL = "7d"

_PARTITION = "coinmarketcap"
_FILENAME = "cmc-listings-historical.csv"
_R_SCRIPT = Path("scripts") / "pull_cmc_panel.R"

PANEL_COLUMNS = (
    "ts_utc",
    "cmc_id",
    "symbol",
    "name",
    "cmc_rank",
    "price_usd",
    "market_cap_usd",
    "volume_24h_usd",
    "circulating_supply",
)

# Assets that were listed, then died. A panel from the survivorship-*biased*
# endpoint drops them entirely, which is the failure this pull exists to avoid,
# and it is invisible in aggregate — so we name two and check for them by id.
KNOWN_DEAD_ASSETS = {
    827: "BitConnect, collapsed January 2018",
    6187: "Serum, delisted from Binance November 2022",
}


class PanelAlreadyStored(RawArtifactAlreadyStored):
    """The panel is already in `data/raw/`. It is pulled once and never replaced."""


class PanelMissing(RawArtifactMissing):
    """The panel was read before it was pulled."""


class MalformedPanel(Exception):
    """Bytes that are not a readable CoinMarketCap listings panel."""


class SurvivorshipBiasedPanel(Exception):
    """The panel is missing assets that died. It would bias every Universe built on it."""


class PanelPullFailed(Exception):
    """`Rscript` did not produce a panel."""


class PanelWindowNotCovered(Exception):
    """The panel does not reach back as far as the window that was asked for."""


PanelPuller = Callable[[Path], None]


def stamped_date(pulled_at_utc: str) -> date:
    """The UTC date of an ISO-seconds timestamp such as `2026-08-31T00:00:00Z`."""
    return date.fromisoformat(pulled_at_utc[:10])


class CmcPanelStore:
    """The one stored CoinMarketCap panel, in the append-only raw layer."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def has_panel(self) -> bool:
        return self.panel_path().exists()

    def panel_path(self) -> Path:
        return self.root / _PARTITION / _FILENAME

    def write(
        self,
        payload: bytes,
        *,
        pulled_at_utc: str,
        window_start: date,
        window_end: date | None = None,
    ) -> Path:
        """Store the panel. Raises if it is already stored, biased, or too short.

        `pulled_at_utc` is passed in rather than read from the clock, so the
        stored manifest is reproducible. Both checks run here rather than in
        `pull_panel`, so no path can put a bad panel on disk — a manifest is
        only worth trusting if nothing can write one it has not earned.

        `window_start` has no default for the same reason. The manifest records
        it as the window that was asked for, so the caller has to say what it
        asked for rather than inherit a claim it never made.

        The manifest separates the request (`window_start`, `window_end`) from
        what actually came back (`first_snapshot`, `last_snapshot`,
        `dead_assets_present`).
        """
        path = self.panel_path()
        if path.exists():
            raise PanelAlreadyStored(
                f"{_FILENAME} is already in {path.parent}. Per ADR-0008 the "
                "CoinMarketCap panel is pulled once; a second pull is a bug "
                "report, not an overwrite."
            )
        observed = parse_panel_csv(payload)
        assert_survivorship_free(observed)
        assert_covers_window(observed, window_start)
        listed = set(observed["cmc_id"].unique())
        return write_immutable(
            path,
            payload,
            {
                "vendor": VENDOR,
                "source": SOURCE,
                "symbol_convention": SYMBOL_CONVENTION,
                "bar_close_convention": BAR_CLOSE_CONVENTION,
                "timezone": TIMEZONE,
                "interval": PANEL_INTERVAL,
                "window_start": window_start.isoformat(),
                "window_end": (window_end or stamped_date(pulled_at_utc)).isoformat(),
                "first_snapshot": observed.index.min().date().isoformat(),
                "last_snapshot": observed.index.max().date().isoformat(),
                "assets": int(observed["cmc_id"].nunique()),
                # Evidence, not a claim: the ids of assets known to have died
                # that this payload actually lists.
                "dead_assets_present": sorted(
                    cmc_id for cmc_id in KNOWN_DEAD_ASSETS if cmc_id in listed
                ),
                "sha256": sha256_of(payload),
                "pulled_at_utc": pulled_at_utc,
                "bytes": len(payload),
            },
        )

    def read(self) -> bytes:
        path = self.panel_path()
        if not path.exists():
            raise PanelMissing(
                f"the CoinMarketCap panel has not been pulled into {path.parent}; "
                "run `momentum pull-cmc-panel`"
            )
        return path.read_bytes()

    def manifest(self) -> dict[str, Any]:
        if not self.has_panel():
            raise PanelMissing(f"no panel in {self.panel_path().parent}")
        return read_manifest(self.panel_path())

    def read_panel(self) -> pd.DataFrame:
        """The stored panel as a frame. One row is one asset on one snapshot date."""
        return parse_panel_csv(self.read())


def pull_panel(
    store: CmcPanelStore,
    *,
    run_pull: PanelPuller | None = None,
    pulled_at_utc: str,
    repo_root: Path | str = ".",
    window_start: date = PANEL_START,
) -> Path:
    """Return the stored panel, pulling it through `crypto2` only if absent.

    The early return is the whole point: re-running a build must not re-fetch.
    A panel that fails its checks never reaches `data/raw/`, so a bad pull
    leaves the store empty and can simply be run again.

    The window's end is taken from `pulled_at_utc` rather than read from the
    clock, so the same call reproduces the same request.
    """
    if store.has_panel():
        return store.panel_path()

    window_end = stamped_date(pulled_at_utc)
    puller = run_pull or rscript_puller(
        Path(repo_root) / _R_SCRIPT, window_start=window_start, window_end=window_end
    )
    with tempfile.TemporaryDirectory() as staging:
        destination = Path(staging) / _FILENAME
        puller(destination)
        if not destination.exists():
            raise PanelPullFailed(f"the pull produced no file at {destination}")
        payload = destination.read_bytes()

    # Both the survivorship and window checks live in `write`, which parses the
    # payload once. Checking here too would parse the same bytes twice.
    return store.write(
        payload,
        pulled_at_utc=pulled_at_utc,
        window_start=window_start,
        window_end=window_end,
    )


def rscript_puller(
    script: Path,
    *,
    window_start: date,
    window_end: date,
    interval: str = PANEL_INTERVAL,
) -> PanelPuller:
    """Build the puller that shells out to `Rscript`, the one R boundary we cross.

    Both window bounds are required. The R script will not read the clock for
    itself, so an unrepeatable window cannot be requested by accident.
    """

    def run(destination: Path) -> None:
        command = [
            "Rscript",
            str(script),
            "--start",
            window_start.isoformat(),
            "--end",
            window_end.isoformat(),
            "--interval",
            interval,
            "--out",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise PanelPullFailed(
                "Rscript is not on PATH. The panel needs R with the crypto2 "
                "package; see ADR-0008."
            ) from error
        except subprocess.CalledProcessError as error:
            raise PanelPullFailed(
                f"{' '.join(command)} exited {error.returncode}"
            ) from error

    return run


def parse_panel_csv(payload: bytes) -> pd.DataFrame:
    """Parse the panel CSV into a frame.

    One row is one asset on one snapshot date. The index is `ts_utc`, the
    snapshot's UTC date; `cmc_id` is the asset's permanent identity and `symbol`
    is only what it was called on that date.
    """
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except (ValueError, pd.errors.ParserError) as error:
        raise MalformedPanel(f"could not parse the panel CSV: {error}") from error

    missing = [column for column in PANEL_COLUMNS if column not in raw.columns]
    if missing:
        raise MalformedPanel(f"the panel is missing columns {missing}")

    panel = raw.loc[:, list(PANEL_COLUMNS)].copy()
    try:
        panel["cmc_id"] = panel["cmc_id"].astype(int)
        for column in (
            "cmc_rank",
            "price_usd",
            "market_cap_usd",
            "volume_24h_usd",
            "circulating_supply",
        ):
            panel[column] = pd.to_numeric(panel[column])
        snapshot = pd.to_datetime(panel.pop("ts_utc"), utc=True, format="ISO8601")
    except (ValueError, TypeError) as error:
        raise MalformedPanel(f"the panel's columns do not parse: {error}") from error

    panel.index = pd.DatetimeIndex(snapshot)
    panel.index.name = "ts_utc"
    panel = panel.sort_index(kind="stable")

    duplicated = panel.reset_index().duplicated(["ts_utc", "cmc_id"])
    if duplicated.any():
        offender = panel.iloc[duplicated.to_numpy().argmax()]
        raise MalformedPanel(
            f"cmc_id {offender['cmc_id']} appears twice on one snapshot; "
            "one row must be one asset on one date"
        )
    return panel


def assert_covers_window(panel: pd.DataFrame, window_start: date) -> None:
    """Raise unless the panel reaches back to `window_start`.

    A pull that quietly returns a shorter history would otherwise be stored with
    a manifest naming a window it does not have, and every later reader would
    trust the manifest.
    """
    first_snapshot = panel.index.min().date()
    if first_snapshot > window_start:
        raise PanelWindowNotCovered(
            f"the panel starts at {first_snapshot}, later than the requested "
            f"{window_start}. Storing it would stamp a window it does not cover."
        )


def assert_survivorship_free(panel: pd.DataFrame) -> None:
    """Raise unless assets known to have died are present.

    A panel drawn from the biased endpoint looks entirely normal — it is simply
    missing the coins that failed, which are exactly the ones a momentum
    cross-section needs in order not to overstate itself.
    """
    listed = set(panel["cmc_id"].unique())
    absent = {
        cmc_id: description
        for cmc_id, description in KNOWN_DEAD_ASSETS.items()
        if cmc_id not in listed
    }
    if absent:
        named = "; ".join(f"{cmc_id} ({why})" for cmc_id, why in sorted(absent.items()))
        raise SurvivorshipBiasedPanel(
            f"the panel does not list {named}. It came from a survivorship-biased "
            "source and must not be stored — see ADR-0008."
        )
