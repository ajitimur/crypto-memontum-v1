"""Matching CoinMarketCap ids to Binance tickers, as of a date.

ADR-0008 names this the hybrid's known bug surface, and the reason is that the
two vendors disagree about what an identity is. A CoinMarketCap id is permanent:
id 4172 is the original Terra for as long as CoinMarketCap exists. A Binance base
is a *name*, and Binance reuses names — it reassigned `LUNAUSDT` to Terra 2.0 on
2022-05-31 while the original chain's ticker became `LUNC`.

So the mapping is time-varying by construction. There is no answer to "what is
LUNA"; there is only an answer to "what was LUNA on 2022-05-22" (Terra Classic,
mid-collapse) and "what was LUNA on 2022-06-15" (Terra 2.0, a two-week-old
token). A mapping that collapses those two into one asset hands the 2022
cross-section a 99.99% drawdown and a fresh listing as if they were one price
series. `build_symbol_map` refuses to build such a mapping rather than
returning it.

Most of the mapping derives from the panel itself, because CoinMarketCap records
the rename: id 4172's symbol changes from LUNA to LUNC across the May 2022
snapshots. But it records it on *its* grid, and the vendor's snapshot date is not
the venue's cutover date — CoinMarketCap's fixture snapshot lands on 2022-05-29
where Binance renamed on 2022-05-31. The bar that fills an order is a Binance
bar, so the venue's date is the one that has to win. That is what
`configs/vendor-symbol-map.toml` is for, and `vendor_symbol_map` is the entry
point that applies it; `build_symbol_map` is the pure core underneath.

This module answers identity, not tradeability: "which asset was this base on
this date". Whether that asset could be traded on that date is the Universe's
question, and it is answered from the archive's own file date ranges. So a spell
deliberately spans a gap in listing — an asset relisted under the same symbol is
still the same asset, and pretending its identity lapsed would be a second bug
rather than a fix for the first.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd


# The committed hand-resolved links, relative to the repository root.
DEFAULT_OVERRIDE_TABLE = Path("configs") / "vendor-symbol-map.toml"


class AmbiguousTicker(Exception):
    """Two assets claim one identity on one date. Resolve it in the override table."""


class MalformedOverrideTable(Exception):
    """The committed override table is not readable as vendor links."""


class _HalfOpen:
    """`valid_from` inclusive, `valid_until` exclusive, `None` meaning still open.

    A mixin rather than a base dataclass so that both users keep their own field
    order — the identity reads first in each, which is how they are constructed.
    """

    valid_from: date
    valid_until: date | None

    def covers(self, as_of: date) -> bool:
        if as_of < self.valid_from:
            return False
        return self.valid_until is None or as_of < self.valid_until

    def overlaps(self, other: "_HalfOpen") -> bool:
        starts_before_other_ends = (
            other.valid_until is None or self.valid_from < other.valid_until
        )
        other_starts_before_this_ends = (
            self.valid_until is None or other.valid_from < self.valid_until
        )
        return starts_before_other_ends and other_starts_before_this_ends


@dataclass(frozen=True)
class SymbolSpell(_HalfOpen):
    """A stretch of time over which CoinMarketCap called `cmc_id` by `symbol`.

    `valid_from` is inclusive and `valid_until` exclusive, both UTC dates.
    `valid_until` is `None` while the spell is still the asset's current name.
    """

    cmc_id: int
    name: str
    symbol: str
    valid_from: date
    valid_until: date | None


@dataclass(frozen=True)
class VendorLink(_HalfOpen):
    """`cmc_id` is the asset Binance quoted as `binance_base` over this interval.

    Half-open: `valid_from` inclusive, `valid_until` exclusive, `None` for open.
    """

    cmc_id: int
    binance_base: str
    valid_from: date
    valid_until: date | None


@dataclass(frozen=True)
class SymbolMap:
    """Which CoinMarketCap id a Binance base referred to, at any date.

    `unmatched_cmc_ids` and `unmatched_binance_bases` are part of the result
    rather than a log line: an asset on one vendor and not the other is out of
    the Universe, and the caller has to be able to see which ones and how many.
    """

    links: tuple[VendorLink, ...]
    unmatched_cmc_ids: tuple[int, ...]
    unmatched_binance_bases: tuple[str, ...]

    def binance_base_for(self, cmc_id: int, as_of: date) -> str | None:
        for link in self.links:
            if link.cmc_id == cmc_id and link.covers(as_of):
                return link.binance_base
        return None

    def cmc_id_for(self, binance_base: str, as_of: date) -> int | None:
        for link in self.links:
            if link.binance_base == binance_base and link.covers(as_of):
                return link.cmc_id
        return None


def symbol_spells(panel: pd.DataFrame) -> tuple[SymbolSpell, ...]:
    """Derive each asset's naming history from the panel.

    One spell per contiguous run of snapshots on which an id carried the same
    symbol. A spell ends where the next one for that id begins, or — for an
    asset that simply stopped being listed — at the first snapshot on which it
    is gone, so that no date between two snapshots falls into a hole.
    """
    snapshots = sorted({timestamp.date() for timestamp in panel.index})
    next_snapshot = {
        day: snapshots[position + 1] for position, day in enumerate(snapshots[:-1])
    }

    spells: list[SymbolSpell] = []
    frame = panel.reset_index()
    frame["snapshot"] = frame["ts_utc"].dt.date
    for cmc_id, rows in frame.groupby("cmc_id", sort=True):
        ordered = rows.sort_values("snapshot")
        run_symbol: str | None = None
        run_start: date | None = None
        run_end: date | None = None
        run_name = ""
        for row in ordered.itertuples(index=False):
            if row.symbol != run_symbol:
                if run_symbol is not None:
                    spells.append(
                        SymbolSpell(
                            int(cmc_id), run_name, run_symbol, run_start, row.snapshot
                        )
                    )
                run_symbol, run_start, run_name = row.symbol, row.snapshot, row.name
            run_end = row.snapshot
        if run_symbol is not None:
            spells.append(
                SymbolSpell(
                    int(cmc_id),
                    run_name,
                    run_symbol,
                    run_start,
                    next_snapshot.get(run_end),
                )
            )
    return tuple(spells)


def build_symbol_map(
    spells: Iterable[SymbolSpell],
    binance_bases: Iterable[str],
    *,
    overrides: Iterable[VendorLink] = (),
) -> SymbolMap:
    """Match CoinMarketCap ids to the Binance bases in `binance_bases`.

    A spell matches when its symbol is one of `binance_bases`. The check is on
    the name only — `binance_bases` carries no dates, so this cannot tell that a
    base was listed for part of a spell and not the rest. Tradeability is the
    Universe's question, resolved from the archive's own file date ranges; this
    function answers identity.

    Overrides win outright: naming an id in the override table discards every
    derived link for it, so a hand-resolved asset is described in exactly one
    place rather than merged with a guess.

    Raises `AmbiguousTicker` if the result would have one base meaning two
    assets on one date, or one asset trading under two bases on one date.
    """
    bases = set(binance_bases)
    overridden = tuple(overrides)
    hand_resolved = {link.cmc_id for link in overridden}

    derived = tuple(
        VendorLink(spell.cmc_id, spell.symbol, spell.valid_from, spell.valid_until)
        for spell in spells
        if spell.cmc_id not in hand_resolved and spell.symbol in bases
    )
    # Overrides are kept whether or not the base appears in `binance_bases`: the
    # base list is usually a window's worth of the archive, and silently dropping
    # a curated link because the window is short is the opposite of explicit.
    links = derived + overridden

    _assert_no_overlap(links, key=lambda link: link.binance_base, subject="Binance base")
    _assert_no_overlap(links, key=lambda link: link.cmc_id, subject="CoinMarketCap id")

    seen_ids = {spell.cmc_id for spell in spells}
    linked_ids = {link.cmc_id for link in links}
    linked_bases = {link.binance_base for link in links}
    return SymbolMap(
        links=links,
        unmatched_cmc_ids=tuple(sorted(seen_ids - linked_ids)),
        unmatched_binance_bases=tuple(sorted(bases - linked_bases)),
    )


def vendor_symbol_map(
    panel: pd.DataFrame,
    binance_bases: Iterable[str],
    *,
    repo_root: Path | str = ".",
    overrides_path: Path | str | None = None,
) -> SymbolMap:
    """The mapping as it is actually used: panel spells, corrected by the table.

    This is the entry point production code should call. Going through
    `build_symbol_map` alone silently accepts the vendor's snapshot grid as the
    boundary between two assets sharing a ticker, and for LUNA that grid is two
    days out from Binance's own rename.
    """
    path = Path(overrides_path) if overrides_path else Path(repo_root) / DEFAULT_OVERRIDE_TABLE
    return build_symbol_map(
        symbol_spells(panel), binance_bases, overrides=load_overrides(path)
    )


def load_overrides(path: Path | str) -> tuple[VendorLink, ...]:
    """Read the committed hand-resolved links from TOML.

    TOML is inert data — the same reason `config` uses it. A mapping that could
    execute code would be a mapping nobody can audit.
    """
    path = Path(path)
    try:
        document = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise MalformedOverrideTable(f"{path} is not valid TOML: {error}") from error

    links = []
    for entry in document.get("link", []):
        try:
            links.append(
                VendorLink(
                    cmc_id=int(entry["cmc_id"]),
                    binance_base=str(entry["binance_base"]),
                    valid_from=_as_date(entry["valid_from"]),
                    valid_until=_as_date(entry["valid_until"])
                    if entry.get("valid_until") is not None
                    else None,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MalformedOverrideTable(
                f"{path} has an unreadable [[link]] entry {entry!r}: {error}"
            ) from error
    return tuple(links)


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    raise TypeError(f"{value!r} is not a date; write it bare, as 2022-05-31")


def _assert_no_overlap(
    links: Sequence[VendorLink],
    *,
    key: Callable[[VendorLink], object],
    subject: str,
) -> None:
    grouped: dict[object, list[VendorLink]] = {}
    for link in links:
        grouped.setdefault(key(link), []).append(link)
    for identity, group in grouped.items():
        ordered = sorted(group, key=lambda link: link.valid_from)
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier.overlaps(later):
                raise AmbiguousTicker(
                    f"{subject} {identity} is claimed by both "
                    f"{_describe(earlier)} and {_describe(later)}, which overlap. "
                    "Resolve it explicitly in configs/vendor-symbol-map.toml — a "
                    "collapsed ticker corrupts the cross-section silently."
                )


def _describe(link: VendorLink) -> str:
    until = link.valid_until.isoformat() if link.valid_until else "open"
    return f"cmc_id {link.cmc_id} as {link.binance_base} [{link.valid_from}, {until})"
