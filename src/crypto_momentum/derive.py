"""`data/derived/` — bars rebuilt from `data/raw/` by a script, and disposable.

Nothing here is a source of truth. Delete the whole directory and the next
rebuild reproduces it byte-for-byte from the raw archive files, which is what
lets the raw layer stay append-only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from crypto_momentum.data.binance_archive import monthly_klines_file, parse_klines_zip
from crypto_momentum.data.raw_store import RawStore

_INTERVAL_FREQ = {"1d": pd.Timedelta(days=1)}


class GapInWindow(Exception):
    """The stored months do not form a contiguous series of bars."""


class DerivedBarsMissing(Exception):
    """Derived bars were read before they were built."""


class DerivedStore:
    """Built bars on disk. Overwriting is the normal path — this data is derived."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, symbol: str, interval: str) -> Path:
        return self.root / "bars" / interval / f"{symbol}.parquet"

    def has(self, symbol: str, interval: str) -> bool:
        return self.path_for(symbol, interval).exists()

    def write_bars(self, symbol: str, interval: str, bars: pd.DataFrame) -> Path:
        path = self.path_for(symbol, interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(path)
        return path

    def read_bars(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self.path_for(symbol, interval)
        if not path.exists():
            raise DerivedBarsMissing(
                f"no derived {interval} bars for {symbol}; rebuild them from data/raw/"
            )
        return pd.read_parquet(path)

    def clear(self) -> None:
        """Throw the whole derived layer away. Safe by construction."""
        if self.root.exists():
            shutil.rmtree(self.root)


def build_daily_bars(
    raw_store: RawStore, symbol: str, interval: str, months: list[str]
) -> pd.DataFrame:
    """Parse the stored monthly partitions for `months` into one bar series.

    One row is one bar, indexed on its UTC open time. Raises `GapInWindow` if the
    months do not join up — a hole is a data problem to investigate, never
    something to interpolate across.
    """
    frames = [
        parse_klines_zip(raw_store.read(monthly_klines_file(symbol, interval, month)))
        for month in months
    ]
    bars = pd.concat(frames).sort_index()
    duplicated = bars.index.duplicated()
    if duplicated.any():
        raise GapInWindow(
            f"{symbol} {interval} has duplicate bars at "
            f"{bars.index[duplicated][0].date()}; the stored months overlap"
        )
    _assert_contiguous(bars.index, symbol, interval)
    return bars


def rebuild_daily_bars(
    raw_store: RawStore,
    derived_store: DerivedStore,
    symbol: str,
    interval: str,
    months: list[str],
) -> pd.DataFrame:
    """Rebuild `symbol`'s derived bars from raw and store them."""
    bars = build_daily_bars(raw_store, symbol, interval, months)
    derived_store.write_bars(symbol, interval, bars)
    return bars


def _assert_contiguous(index: pd.DatetimeIndex, symbol: str, interval: str) -> None:
    step = _INTERVAL_FREQ.get(interval)
    if step is None or len(index) < 2:
        return
    gaps = index.to_series().diff().iloc[1:] != step
    if gaps.any():
        resumes_at = index[1:][gaps.to_numpy()][0]
        missing = index[index < resumes_at][-1] + step
        raise GapInWindow(
            f"{symbol} {interval} bars are missing {missing.date()}: the series "
            f"jumps to {resumes_at.date()}. A month is absent from data/raw/."
        )
