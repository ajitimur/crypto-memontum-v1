"""Run configuration: inert TOML data in, a validated `RunConfig` out.

TOML is chosen over YAML or a Python module because it has no tag, no
constructor and no import: a config file is data and cannot execute anything.
Every value is additionally range- and shape-checked here, so a typo is a
rejected run rather than a silently different result.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_momentum.sim.cross_sectional import (
    DEFAULT_CAP_STALENESS_DAYS,
    MIN_UNIVERSE,
)
from crypto_momentum.sim.universe_policy import (
    BINANCE_FULL,
    BRACKETS,
    DEFAULT_WINDOW_DAYS,
)

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{4,20}$")
# `name` becomes a path segment under `results/`, so it is restricted to
# characters that cannot escape it. A config called `../../etc/passwd` would
# otherwise write a result outside the results directory.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

SUPPORTED_VENUES = ("binance-spot",)
SUPPORTED_INTERVALS = ("1d",)

BUY_AND_HOLD = "buy_and_hold"
CROSS_SECTIONAL = "cross_sectional"
SUPPORTED_STRATEGIES = (BUY_AND_HOLD, CROSS_SECTIONAL)

# A strategy declares which knobs it has. A knob belonging to another strategy is
# rejected rather than ignored, because a `lookback_days` sitting unread in a
# buy-and-hold config reads, in the trials log, exactly like one that was used.
_STRATEGY_KEYS: dict[str, tuple[str, ...]] = {
    BUY_AND_HOLD: (),
    CROSS_SECTIONAL: (
        "lookback_days",
        "holding_days",
        "quantile",
        "min_universe",
        "max_cap_staleness_days",
    ),
}

_SCHEMA: dict[str, tuple[str, ...]] = {
    "data": ("venue", "symbol", "symbols", "interval", "start_month", "end_month"),
    "strategy": ("kind", *_STRATEGY_KEYS[CROSS_SECTIONAL]),
    "universe": ("bracket", "liquidity_floor_usd", "liquidity_window_days"),
    "costs": ("fee_bps_per_side", "slippage_bps_per_side"),
}
_TOP_LEVEL_KEYS = ("name",)


class ConfigError(Exception):
    """A config was missing, malformed, or out of range. The run does not start."""


@dataclass(frozen=True)
class RunConfig:
    """One backtest run, fully specified.

    `name` is half the result key — the other half is the commit hash — so it
    identifies a configuration across every run that ever used it.
    """

    name: str
    venue: str
    symbol: str | None
    interval: str
    start_month: str
    end_month: str
    strategy_kind: str
    fee_bps_per_side: float
    slippage_bps_per_side: float
    # A single-asset hold names one `symbol`; a cross-section names `symbols`.
    # Exactly one of the two is set, so `universe_symbols` is the only thing
    # downstream code has to look at.
    symbols: tuple[str, ...] = ()
    lookback_days: int | None = None
    holding_days: int | None = None
    quantile: float | None = None
    min_universe: int = MIN_UNIVERSE
    max_cap_staleness_days: int = DEFAULT_CAP_STALENESS_DAYS
    bracket: str = BINANCE_FULL
    liquidity_floor_usd: float | None = None
    liquidity_window_days: int = DEFAULT_WINDOW_DAYS

    @property
    def universe_symbols(self) -> tuple[str, ...]:
        """Every symbol the run needs bars for, whichever way it named them."""
        return self.symbols if self.symbol is None else (self.symbol,)

    def months(self) -> list[str]:
        """The exact window fetched, as inclusive `YYYY-MM` archive partitions."""
        start_year, start_month = (int(part) for part in self.start_month.split("-"))
        end_year, end_month = (int(part) for part in self.end_month.split("-"))
        first = start_year * 12 + (start_month - 1)
        last = end_year * 12 + (end_month - 1)
        return [f"{index // 12:04d}-{index % 12 + 1:02d}" for index in range(first, last + 1)]

    @property
    def cost_bps_per_side(self) -> float:
        """Total round-trip-halved cost charged on both buys and sells.

        Per ADR-0007 the Indonesian PPh applies to each leg, so this is charged
        on entry as well as exit — never haircut off a gross number afterwards.
        """
        return self.fee_bps_per_side + self.slippage_bps_per_side


def load_config(path: Path | str) -> RunConfig:
    """Read and validate a run config. Raises `ConfigError` on anything wrong."""
    path = Path(path)
    try:
        raw_text = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error

    try:
        document = tomllib.loads(raw_text.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error

    _reject_unknown_keys(document)

    name = _require_str(document, "name", "name")
    if not NAME_PATTERN.match(name) or ".." in name:
        raise ConfigError(
            "name must be a filename-safe identifier of letters, digits, dots, "
            f"dashes and underscores — it becomes a path under results/ — got {name!r}"
        )
    data = _require_table(document, "data")
    strategy = _require_table(document, "strategy")
    costs = _require_table(document, "costs")

    venue = _require_choice(data, "venue", "data.venue", SUPPORTED_VENUES)
    strategy_kind = _require_choice(strategy, "kind", "strategy.kind", SUPPORTED_STRATEGIES)
    _reject_foreign_strategy_keys(strategy, strategy_kind)
    symbol, symbols = _require_symbols(data, strategy_kind)
    interval = _require_choice(data, "interval", "data.interval", SUPPORTED_INTERVALS)
    start_month = _require_month(data, "start_month", "data.start_month")
    end_month = _require_month(data, "end_month", "data.end_month")
    if end_month < start_month:
        raise ConfigError(
            f"data.end_month {end_month!r} is before data.start_month {start_month!r}"
        )

    universe = _optional_table(document, "universe")
    config = RunConfig(
        name=name,
        venue=venue,
        symbol=symbol,
        symbols=symbols,
        interval=interval,
        start_month=start_month,
        end_month=end_month,
        strategy_kind=strategy_kind,
        fee_bps_per_side=_require_non_negative(costs, "fee_bps_per_side", "costs.fee_bps_per_side"),
        slippage_bps_per_side=_require_non_negative(
            costs, "slippage_bps_per_side", "costs.slippage_bps_per_side"
        ),
        bracket=_require_choice_or(
            universe, "bracket", "universe.bracket", BRACKETS, BINANCE_FULL
        ),
        liquidity_floor_usd=_optional_non_negative(
            universe, "liquidity_floor_usd", "universe.liquidity_floor_usd"
        ),
        liquidity_window_days=_optional_positive_int(
            universe, "liquidity_window_days", "universe.liquidity_window_days",
            DEFAULT_WINDOW_DAYS,
        ),
        **_strategy_parameters(strategy, strategy_kind),
    )
    return config


def _reject_unknown_keys(document: dict[str, Any]) -> None:
    for key in document:
        if key not in _TOP_LEVEL_KEYS and key not in _SCHEMA:
            raise ConfigError(f"unknown key {key!r} at the top level of the config")
    for section, allowed in _SCHEMA.items():
        table = document.get(section)
        if not isinstance(table, dict):
            continue
        for key in table:
            if key not in allowed:
                raise ConfigError(f"unknown key {key!r} in [{section}]")


def _require_symbols(
    data: dict[str, Any], strategy_kind: str
) -> tuple[str | None, tuple[str, ...]]:
    """Read the run's assets in whichever shape its strategy takes.

    A single-asset hold names `data.symbol`; a cross-section names
    `data.symbols`. Naming both, or the wrong one for the strategy, is refused:
    a config that carries an unread symbol list looks in the trials log exactly
    like one that ran on it.
    """
    has_symbol = "symbol" in data
    has_symbols = "symbols" in data
    if has_symbol and has_symbols:
        raise ConfigError(
            "data.symbol and data.symbols cannot both be set; a run holds one "
            "asset or ranks a cross-section of them, not both"
        )

    if strategy_kind == CROSS_SECTIONAL:
        if not has_symbols:
            raise ConfigError(
                f"the {CROSS_SECTIONAL} strategy ranks a cross-section, so it "
                "needs data.symbols — a list of venue tickers"
            )
        raw = data["symbols"]
        if not isinstance(raw, list) or not raw:
            raise ConfigError("data.symbols must be a non-empty list of venue tickers")
        symbols = tuple(_checked_symbol(value, "data.symbols") for value in raw)
        duplicated = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicated:
            raise ConfigError(
                f"data.symbols lists {', '.join(duplicated)} more than once; a "
                "symbol appearing twice would be weighted twice"
            )
        return None, symbols

    if not has_symbol:
        raise ConfigError(f"the {strategy_kind} strategy needs data.symbol")
    return _checked_symbol(data["symbol"], "data.symbol"), ()


def _checked_symbol(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must hold strings, got {type(value).__name__}")
    if not SYMBOL_PATTERN.match(value):
        raise ConfigError(
            f"{label} must be an uppercase venue ticker such as BTCUSDT, got {value!r}"
        )
    return value


def _reject_foreign_strategy_keys(strategy: dict[str, Any], strategy_kind: str) -> None:
    allowed = set(_STRATEGY_KEYS[strategy_kind])
    for key in strategy:
        if key != "kind" and key not in allowed:
            raise ConfigError(
                f"[strategy] key {key!r} does not belong to the {strategy_kind} "
                "strategy, so nothing would read it"
            )


def _strategy_parameters(strategy: dict[str, Any], strategy_kind: str) -> dict[str, Any]:
    """The strategy's own knobs, each required and range-checked where it applies."""
    if strategy_kind != CROSS_SECTIONAL:
        return {}
    lookback_days = _require_positive_int(
        strategy, "lookback_days", "strategy.lookback_days"
    )
    holding_days = _require_positive_int(
        strategy, "holding_days", "strategy.holding_days"
    )
    quantile = _require_non_negative(strategy, "quantile", "strategy.quantile")
    if not 0.0 < quantile <= 1.0:
        raise ConfigError(
            f"strategy.quantile is the share of the cross-section held and must "
            f"be in (0, 1], got {quantile}"
        )
    return {
        "lookback_days": lookback_days,
        "holding_days": holding_days,
        "quantile": quantile,
        # Both change what a run holds, so both are config rather than a literal
        # in the simulator: a date that held cash has to be accountable to a
        # number someone wrote down.
        "min_universe": _optional_positive_int(
            strategy, "min_universe", "strategy.min_universe", MIN_UNIVERSE
        ),
        "max_cap_staleness_days": _optional_positive_int(
            strategy,
            "max_cap_staleness_days",
            "strategy.max_cap_staleness_days",
            DEFAULT_CAP_STALENESS_DAYS,
        ),
    }


def _optional_table(document: dict[str, Any], section: str) -> dict[str, Any]:
    table = document.get(section, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{section}] must be a table, got {type(table).__name__}")
    return table


def _require_table(document: dict[str, Any], section: str) -> dict[str, Any]:
    table = document.get(section)
    if table is None:
        raise ConfigError(f"config is missing the [{section}] section")
    if not isinstance(table, dict):
        raise ConfigError(f"[{section}] must be a table, got {type(table).__name__}")
    return table


def _require_str(table: dict[str, Any], key: str, label: str) -> str:
    value = table.get(key)
    if value is None:
        raise ConfigError(f"config is missing {label}")
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string, got {type(value).__name__}")
    return value


def _require_choice(
    table: dict[str, Any], key: str, label: str, allowed: tuple[str, ...]
) -> str:
    value = _require_str(table, key, label)
    if value not in allowed:
        raise ConfigError(f"{label} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _require_month(table: dict[str, Any], key: str, label: str) -> str:
    value = _require_str(table, key, label)
    if not MONTH_PATTERN.match(value):
        raise ConfigError(f"{label} must be a YYYY-MM month, got {value!r}")
    return value


def _require_choice_or(
    table: dict[str, Any], key: str, label: str, allowed: tuple[str, ...], default: str
) -> str:
    if key not in table:
        return default
    return _require_choice(table, key, label, allowed)


def _require_positive_int(table: dict[str, Any], key: str, label: str) -> int:
    value = table.get(key)
    if value is None:
        raise ConfigError(f"config is missing {label}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be a whole number of days, got {value!r}")
    if value < 1:
        raise ConfigError(f"{label} must be at least 1, got {value}")
    return value


def _optional_positive_int(
    table: dict[str, Any], key: str, label: str, default: int
) -> int:
    if key not in table:
        return default
    return _require_positive_int(table, key, label)


def _optional_non_negative(
    table: dict[str, Any], key: str, label: str
) -> float | None:
    if key not in table:
        return None
    return _require_non_negative(table, key, label)


def _require_non_negative(table: dict[str, Any], key: str, label: str) -> float:
    value = table.get(key)
    if value is None:
        raise ConfigError(f"config is missing {label}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ConfigError(f"{label} must not be negative, got {value}")
    return float(value)
