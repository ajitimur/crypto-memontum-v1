"""Policy: from what existed to what we would consider holding.

`crypto_momentum.data.universe` reconstructs what the archive published on each
date. That is a fact about the world. This module applies the three judgements
we layer on top of it, and they are judgements rather than facts, which is why
they live in the simulation core where a fixture can pin them down instead of in
the data adapter where they would be invisible:

1. **Exclusions.** Stablecoins and Wrapped Assets come out permanently, on the
   design intent each asset stated at listing (ADR-0006). The list is dated and
   versioned, and its version travels into every panel it produces, because a
   hand-maintained list is exactly the sort of artifact an error hides in.
2. **The liquidity floor.** A trailing median dollar volume gate, read strictly
   from bars before the Decision Bar. It is there to drop artefacted bars — a
   pair with four trades a day prices nothing — and not to model capacity. At
   USD 5,000-10,000 of deployed capital, capacity binds on almost nothing the
   archive publishes, so a floor sized as a capacity constraint would throw away
   the cross-section this research is about.
3. **The bracket.** The full Binance universe is the upper bound; the assets our
   own venue lists today are the lower. Both are reported, and neither is the
   answer on its own: the gap between them is venue-listing risk, and picking
   the flattering bound hides it.

No date is typed in anywhere. A symbol is classified on its first tradeable date
as the panel reports it, so a classification can never be applied before the
asset existed, and a symbol that first traded after the list's `as_of` date is
reported as unclassified rather than silently judged by a list written before it
listed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import pandas as pd

# The two ends of the bracket. `binance-full` is everything the archive ever
# published; `tokocrypto` is what we could actually have bought from Indonesia.
BINANCE_FULL = "binance-full"
TOKOCRYPTO = "tokocrypto"
BRACKETS = (BINANCE_FULL, TOKOCRYPTO)

CATEGORIES = ("stablecoin", "wrapped_asset")

# Thirty days of trailing bars, matching the issue's gate. Long enough that one
# quiet week does not evict an asset, short enough to catch a pair going dark.
DEFAULT_WINDOW_DAYS = 30

FLOOR_PURPOSE = (
    "data-quality gate on artefacted bars, not a capacity constraint — "
    "see the module docstring"
)


class PolicyError(Exception):
    """A policy could not be read or applied. Nothing is guessed at."""


class PointInTimePanel(Protocol):
    """What policy needs from a Universe, stated here rather than imported.

    `crypto_momentum.data.universe.UniversePanel` is the implementation, but the
    simulation core reaches for no adapter: it declares the shape it consumes and
    the data layer satisfies it. That keeps this module a pure function of the
    frame it is handed, which is the invariant `tests/test_simulation.py` asserts.
    """

    tradeable: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExclusionEntry:
    """One asset kept out of the Universe, and the stated intent that keeps it out."""

    symbol: str
    base_asset: str
    category: str
    design_intent: str
    source: str


@dataclass(frozen=True)
class ExclusionList:
    """A dated, versioned list of assets excluded by design intent.

    `version` and `as_of` are what a result quotes. `sha256` is what proves the
    quote: a list edited without a version bump would otherwise produce two
    different Universes under one name.
    """

    version: str
    as_of: str
    entries: tuple[ExclusionEntry, ...]
    sha256: str
    path: str | None = None

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, sha256: str = "", path: str | None = None
    ) -> "ExclusionList":
        version = _require_str(document, "version", "version")
        as_of = _require_date(document, "as_of", "as_of")
        raw_entries = document.get("entry", [])
        if not isinstance(raw_entries, list):
            raise PolicyError("entry must be a list of tables")

        entries: list[ExclusionEntry] = []
        seen: set[str] = set()
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise PolicyError(f"each entry must be a table, got {type(raw).__name__}")
            symbol = _require_str(raw, "symbol", "entry.symbol")
            if symbol in seen:
                raise PolicyError(f"{symbol} appears twice in the exclusion list")
            seen.add(symbol)
            category = _require_str(raw, "category", f"{symbol}.category")
            if category not in CATEGORIES:
                raise PolicyError(
                    f"{symbol} has category {category!r}; a category must be one of "
                    f"{', '.join(CATEGORIES)}"
                )
            entries.append(
                ExclusionEntry(
                    symbol=symbol,
                    base_asset=_require_str(raw, "base_asset", f"{symbol}.base_asset"),
                    category=category,
                    design_intent=_require_str(
                        raw, "design_intent", f"{symbol}.design_intent"
                    ),
                    source=_require_str(raw, "source", f"{symbol}.source"),
                )
            )
        return cls(
            version=version,
            as_of=as_of,
            entries=tuple(entries),
            sha256=sha256,
            path=path,
        )

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(excluded.symbol for excluded in self.entries)

    def entry_for(self, symbol: str) -> ExclusionEntry | None:
        for excluded in self.entries:
            if excluded.symbol == symbol:
                return excluded
        return None

    def excludes(self, symbol: str) -> bool:
        return symbol in self.symbols


@dataclass(frozen=True)
class VenueListing:
    """What one venue lists, captured on a date.

    `status` is `recorded` for a real capture and `stub` for a placeholder. It
    travels into the panel metadata so a bracket resting on a stub reports as
    one instead of passing for a measurement.
    """

    venue: str
    version: str
    recorded_at: str
    status: str
    symbols: tuple[str, ...]
    sha256: str = ""
    path: str | None = None

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any], *, sha256: str = "", path: str | None = None
    ) -> "VenueListing":
        raw_symbols = document.get("symbols", [])
        if not isinstance(raw_symbols, list) or not all(
            isinstance(symbol, str) for symbol in raw_symbols
        ):
            raise PolicyError("symbols must be a list of venue tickers")
        status = _require_str(document, "status", "status")
        if status not in ("recorded", "stub"):
            raise PolicyError(f"status must be 'recorded' or 'stub', got {status!r}")
        return cls(
            venue=_require_str(document, "venue", "venue"),
            version=_require_str(document, "version", "version"),
            recorded_at=_require_date(document, "recorded_at", "recorded_at"),
            status=status,
            symbols=tuple(raw_symbols),
            sha256=sha256,
            path=path,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "version": self.version,
            "recorded_at": self.recorded_at,
            "status": self.status,
            "n_symbols": len(self.symbols),
            "sha256": self.sha256,
            "path": self.path,
        }


@dataclass(frozen=True)
class LiquidityFloor:
    """A trailing median dollar volume gate, read strictly before the Decision Bar."""

    floor_usd: float
    window_days: int = DEFAULT_WINDOW_DAYS

    def mask(self, dollar_volume: pd.DataFrame) -> pd.DataFrame:
        """True where the trailing median clears the floor on that date.

        The shift is the point-in-time boundary and is the whole reason this is a
        method rather than a one-liner at the call site: the window ends on the
        bar *before* the Decision Bar, so a volume spike on the Decision Bar
        itself cannot admit an asset we would have had no way to see.

        A symbol without a full window of history is False rather than NaN: not
        enough bars to judge is not the same as passing, and a data-quality gate
        that defaults to open is not a gate.

        `dollar_volume` is a frame of one row per UTC date and one column per
        venue symbol, indexed on `ts_utc`. The window is measured in days rather
        than in rows, so a frame with a hole in its index does not quietly widen
        it into a 30-row window spanning more than 30 days.
        """
        if self.window_days < 1:
            raise PolicyError(f"window_days must be at least 1, got {self.window_days}")
        if not isinstance(dollar_volume.index, pd.DatetimeIndex):
            raise PolicyError("dollar volume must be indexed on ts_utc timestamps")
        if not dollar_volume.index.is_monotonic_increasing:
            raise PolicyError("dollar volume must be sorted by ts_utc")
        trailing_median = (
            dollar_volume.shift(1)
            .rolling(f"{self.window_days}D", min_periods=self.window_days)
            .median()
        )
        return (trailing_median >= self.floor_usd).fillna(False)

    def to_metadata(self, *, applied: bool, n_symbol_dates_dropped: int) -> dict[str, Any]:
        return {
            "applied": applied,
            "floor_usd": self.floor_usd,
            "window_days": self.window_days,
            "basis": "trailing median dollar volume, strictly before the Decision Bar",
            "purpose": FLOOR_PURPOSE,
            "n_symbol_dates_dropped": n_symbol_dates_dropped,
        }


def dollar_volume_from_bars(bars_by_symbol: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily dollar volume per symbol, from each symbol's own bars.

    Close times volume. For a USDT-quoted pair that is the bar's traded value in
    dollars, priced at the close rather than at the venue's own quote-volume
    figure, which the derived bars do not carry. The difference is a within-bar
    VWAP effect and cannot move a 30-day median across a floor set an order of
    magnitude below any real pair.

    Symbols are aligned on the union of their dates, so a date a symbol has no
    bar for is NaN — absent, which the floor reads as untradeable, and never a
    zero that would read as a real day of no trading.
    """
    if not bars_by_symbol:
        raise PolicyError("dollar volume needs at least one symbol's bars")
    columns = {}
    for symbol, bars in bars_by_symbol.items():
        missing = {"close", "volume"} - set(bars.columns)
        if missing:
            raise PolicyError(
                f"{symbol} bars are missing {', '.join(sorted(missing))}, so no "
                "dollar volume can be built from them"
            )
        columns[symbol] = bars["close"].astype(float) * bars["volume"].astype(float)
    dollar_volume = pd.DataFrame(columns).sort_index()
    dollar_volume.index.name = "ts_utc"
    dollar_volume.columns.name = "symbol"
    return dollar_volume


@dataclass(frozen=True)
class PolicyPanel:
    """The Universe after policy: what we would consider holding, per date.

    Same shape as a `UniversePanel` — one row per UTC date, one column per venue
    symbol — with the archive's own metadata nested under `universe`, so the
    archive floor and the survivorship claim survive this layer rather than
    being dropped at it.
    """

    tradeable: pd.DataFrame
    metadata: dict[str, Any]

    def tradeable_on(self, date: pd.Timestamp | str) -> tuple[str, ...]:
        """The post-policy Universe as of one rebalance date."""
        timestamp = _as_utc(date)
        if timestamp not in self.tradeable.index:
            raise PolicyError(
                f"{timestamp.date()} is outside the panel, which runs "
                f"{self.tradeable.index[0].date()} to {self.tradeable.index[-1].date()}"
            )
        row = self.tradeable.loc[timestamp]
        return tuple(row.index[row.to_numpy()])


def apply_universe_policy(
    panel: PointInTimePanel,
    *,
    exclusions: ExclusionList,
    bracket: str,
    venue_listing: VenueListing | None = None,
    dollar_volume: pd.DataFrame | None = None,
    floor: LiquidityFloor | None = None,
) -> PolicyPanel:
    """Apply exclusions, the bracket and the liquidity floor to a point-in-time panel.

    Every gate is a conjunction with what the archive says was tradeable, so
    policy can only ever remove — nothing here can make an asset tradeable on a
    date the archive has no bar for.

    `floor` without `dollar_volume` is an error rather than a skipped gate: a run
    that asked for a floor and quietly did not get one is the kind of result that
    cannot be told apart from a run that never asked.
    """
    if bracket not in BRACKETS:
        raise PolicyError(
            f"bracket must be one of {', '.join(BRACKETS)}, got {bracket!r}"
        )
    if bracket == TOKOCRYPTO and venue_listing is None:
        raise PolicyError(
            "the tokocrypto bracket needs a venue listing to stand on; pass one "
            "loaded from policy/, or select the binance-full bracket"
        )
    if floor is not None and dollar_volume is None:
        raise PolicyError(
            "a liquidity floor needs a dollar volume frame to apply it to"
        )

    tradeable = panel.tradeable.copy()
    symbols = list(tradeable.columns)

    excluded_here = [symbol for symbol in symbols if exclusions.excludes(symbol)]
    for symbol in excluded_here:
        tradeable[symbol] = False

    bracket_symbols = None if bracket == BINANCE_FULL else set(venue_listing.symbols)
    dropped_by_bracket: list[str] = []
    if bracket_symbols is not None:
        dropped_by_bracket = [
            symbol for symbol in symbols if symbol not in bracket_symbols
        ]
        for symbol in dropped_by_bracket:
            tradeable[symbol] = False

    floor_applied = floor is not None and dollar_volume is not None
    n_dropped_by_floor = 0
    if floor_applied:
        passes_floor = _floor_mask_for(panel, dollar_volume, floor)
        n_dropped_by_floor = int((tradeable & ~passes_floor).to_numpy().sum())
        tradeable &= passes_floor

    metadata = {
        "universe": panel.metadata,
        "exclusion_list": _exclusion_metadata(exclusions, panel, symbols, excluded_here),
        "bracket": {
            "selected": bracket,
            "bound": "upper" if bracket == BINANCE_FULL else "lower",
            "upper_bound": BINANCE_FULL,
            "lower_bound": TOKOCRYPTO,
            "venue_listing": (
                venue_listing.to_metadata() if venue_listing is not None else None
            ),
            "n_symbols_dropped_by_bracket": len(dropped_by_bracket),
        },
        "liquidity_floor": (
            floor.to_metadata(applied=True, n_symbol_dates_dropped=n_dropped_by_floor)
            if floor_applied
            else {"applied": False, "purpose": FLOOR_PURPOSE}
        ),
        "n_symbols_in_universe": len(symbols),
        "n_symbols_tradeable_at_some_point": int(tradeable.any().sum()),
    }
    return PolicyPanel(tradeable=tradeable, metadata=metadata)


def universe_bracket(
    panel: PointInTimePanel,
    *,
    exclusions: ExclusionList,
    venue_listing: VenueListing,
    dollar_volume: pd.DataFrame | None = None,
    floor: LiquidityFloor | None = None,
) -> dict[str, PolicyPanel]:
    """Both ends of the bracket from one call, keyed by bracket name.

    Reported together on purpose. A result quoted on one bound alone is a result
    that has chosen which listing risk to show, and the choice is invisible in
    the number.
    """
    return {
        name: apply_universe_policy(
            panel,
            exclusions=exclusions,
            bracket=name,
            venue_listing=venue_listing,
            dollar_volume=dollar_volume,
            floor=floor,
        )
        for name in BRACKETS
    }


def _floor_mask_for(
    panel: PointInTimePanel, dollar_volume: pd.DataFrame, floor: LiquidityFloor
) -> pd.DataFrame:
    """The floor mask, aligned to the panel's dates and symbols.

    A symbol the volume frame does not carry is an error, not a silent drop: an
    asset vanishing from the Universe because its volume column was forgotten
    looks exactly like an asset that failed the gate.
    """
    missing = [
        symbol for symbol in panel.tradeable.columns if symbol not in dollar_volume.columns
    ]
    if missing:
        raise PolicyError(
            "the dollar volume frame is missing "
            f"{', '.join(sorted(missing))}, so the liquidity floor cannot be "
            "applied to the whole Universe"
        )
    # Masked on the volume frame's own index first, so volume history reaching
    # back before the panel starts still feeds the trailing median.
    mask = floor.mask(dollar_volume)
    return (
        mask.reindex(index=panel.tradeable.index, columns=panel.tradeable.columns)
        .fillna(False)
        .astype(bool)
    )


def _exclusion_metadata(
    exclusions: ExclusionList,
    panel: PointInTimePanel,
    symbols: list[str],
    excluded_here: list[str],
) -> dict[str, Any]:
    """What the result quotes about the list, and what the list could not see.

    `classified_at_ts_utc` is each excluded symbol's first tradeable date in the
    panel — the classification is dated to the listing, never to today. A symbol
    that first traded after `as_of` is named under
    `unclassified_listings_since_as_of`, because a list written before an asset
    existed has said nothing about it, and silence is not a decision to include.
    """
    per_symbol = panel.metadata.get("symbols", {})
    as_of = _as_utc(exclusions.as_of)
    unclassified = []
    for symbol in symbols:
        if exclusions.excludes(symbol):
            continue
        first_tradeable = per_symbol.get(symbol, {}).get("first_tradeable_ts_utc")
        if first_tradeable is not None and _as_utc(first_tradeable) > as_of:
            unclassified.append(symbol)
    return {
        "version": exclusions.version,
        "as_of": exclusions.as_of,
        "sha256": exclusions.sha256,
        "path": exclusions.path,
        "n_entries": len(exclusions.entries),
        "n_stablecoins": sum(
            1 for excluded in exclusions.entries if excluded.category == "stablecoin"
        ),
        "n_wrapped_assets": sum(
            1 for excluded in exclusions.entries if excluded.category == "wrapped_asset"
        ),
        "excluded_symbols_in_universe": sorted(excluded_here),
        "classified_at_ts_utc": {
            symbol: per_symbol.get(symbol, {}).get("first_tradeable_ts_utc")
            for symbol in sorted(excluded_here)
        },
        "classified_at_basis": (
            "the symbol's first tradeable date in the point-in-time Universe, "
            "read from archive coverage (ADR-0006)"
        ),
        "unclassified_listings_since_as_of": sorted(unclassified),
    }


def _require_str(document: Mapping[str, Any], key: str, label: str) -> str:
    value = document.get(key)
    if value is None:
        raise PolicyError(f"missing {label}")
    if not isinstance(value, str) or not value.strip():
        raise PolicyError(f"{label} must be a non-empty string")
    return value


def _require_date(document: Mapping[str, Any], key: str, label: str) -> str:
    value = _require_str(document, key, label)
    try:
        pd.Timestamp(value)
    except ValueError as error:
        raise PolicyError(f"{label} must be a YYYY-MM-DD date, got {value!r}") from error
    return value


def _as_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
