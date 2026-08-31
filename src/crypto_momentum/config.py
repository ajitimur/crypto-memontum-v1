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

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{4,20}$")

SUPPORTED_VENUES = ("binance-spot",)
SUPPORTED_INTERVALS = ("1d",)
SUPPORTED_STRATEGIES = ("buy_and_hold",)

_SCHEMA: dict[str, tuple[str, ...]] = {
    "data": ("venue", "symbol", "interval", "start_month", "end_month"),
    "strategy": ("kind",),
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
    symbol: str
    interval: str
    start_month: str
    end_month: str
    strategy_kind: str
    fee_bps_per_side: float
    slippage_bps_per_side: float

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
    data = _require_table(document, "data")
    strategy = _require_table(document, "strategy")
    costs = _require_table(document, "costs")

    venue = _require_choice(data, "venue", "data.venue", SUPPORTED_VENUES)
    symbol = _require_str(data, "symbol", "data.symbol")
    if not SYMBOL_PATTERN.match(symbol):
        raise ConfigError(
            f"data.symbol must be an uppercase venue ticker such as BTCUSDT, got {symbol!r}"
        )
    interval = _require_choice(data, "interval", "data.interval", SUPPORTED_INTERVALS)
    start_month = _require_month(data, "start_month", "data.start_month")
    end_month = _require_month(data, "end_month", "data.end_month")
    if end_month < start_month:
        raise ConfigError(
            f"data.end_month {end_month!r} is before data.start_month {start_month!r}"
        )

    config = RunConfig(
        name=name,
        venue=venue,
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        end_month=end_month,
        strategy_kind=_require_choice(strategy, "kind", "strategy.kind", SUPPORTED_STRATEGIES),
        fee_bps_per_side=_require_non_negative(costs, "fee_bps_per_side", "costs.fee_bps_per_side"),
        slippage_bps_per_side=_require_non_negative(
            costs, "slippage_bps_per_side", "costs.slippage_bps_per_side"
        ),
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


def _require_non_negative(table: dict[str, Any], key: str, label: str) -> float:
    value = table.get(key)
    if value is None:
        raise ConfigError(f"config is missing {label}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ConfigError(f"{label} must not be negative, got {value}")
    return float(value)
