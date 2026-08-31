"""The path from a config file to a recorded result.

Fetch what is missing, rebuild derived bars from raw, simulate, file the result
under `(commit, config)`, and append a line to the trials log. Every step that
touches the outside world lives here or below it; the simulation core does not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_momentum.config import RunConfig, load_config
from crypto_momentum.data.binance_archive import monthly_klines_file
from crypto_momentum.data.fetch import UrlOpener, fetch_archive_file
from crypto_momentum.data.raw_store import RawStore
from crypto_momentum.derive import DerivedStore, rebuild_daily_bars
from crypto_momentum.provenance import describe_head
from crypto_momentum.results import ResultStore, RunRecord
from crypto_momentum.sim.buy_and_hold import simulate_buy_and_hold
from crypto_momentum.sim.report import RunResult
from crypto_momentum.trials import TRIALS_FILENAME, append_trial

# Timestamps that cross the JSON boundary are second-resolution UTC throughout.
ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Workspace:
    """Where the three data layers and the trials log live for a run."""

    repo_root: Path
    raw_root: Path
    derived_root: Path
    results_root: Path
    trials_path: Path

    @classmethod
    def under(cls, repo_root: Path | str) -> "Workspace":
        repo_root = Path(repo_root)
        return cls(
            repo_root=repo_root,
            raw_root=repo_root / "data" / "raw",
            derived_root=repo_root / "data" / "derived",
            results_root=repo_root / "results",
            trials_path=repo_root / TRIALS_FILENAME,
        )


def run_config(
    config_path: Path | str,
    workspace: Workspace,
    *,
    run_at_utc: str,
    open_url: UrlOpener | None = None,
) -> RunRecord:
    """Run one config end to end and return the recorded result.

    `run_at_utc` is passed in rather than read from the clock, so the runner
    itself stays reproducible; the CLI supplies the wall-clock value.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    provenance = describe_head(workspace.repo_root)

    bars = load_bars(config, workspace, fetched_at_utc=run_at_utc, open_url=open_url)
    result = simulate_buy_and_hold(bars, cost_bps_per_side=config.cost_bps_per_side)

    record = RunRecord(
        commit=provenance.commit,
        working_tree_dirty=provenance.working_tree_dirty,
        run_at_utc=run_at_utc,
        config=config,
        config_sha256=config_sha256,
        config_path=_relative_to_repo(config_path, workspace.repo_root),
        metrics=metrics_of(result),
        window={
            "months": config.months(),
            "first_bar_ts_utc": _iso(bars.index[0]),
            "last_bar_ts_utc": _iso(bars.index[-1]),
            "n_bars": len(bars),
        },
    )
    ResultStore(workspace.results_root).write(record)
    append_trial(workspace.trials_path, record.trial_line())
    return record


def load_bars(
    config: RunConfig,
    workspace: Workspace,
    *,
    fetched_at_utc: str,
    open_url: UrlOpener | None = None,
) -> pd.DataFrame:
    """Ensure the config's window is in `data/raw/`, then rebuild derived bars.

    Only months that are not already stored are fetched: raw data is append-only,
    so a second run of the same config reads what the first run stored.
    """
    raw_store = RawStore(workspace.raw_root)
    for month in config.months():
        archive_file = monthly_klines_file(config.symbol, config.interval, month)
        if raw_store.has(archive_file):
            continue
        payload, digest = fetch_archive_file(archive_file, open_url=open_url)
        raw_store.write(
            archive_file, payload, sha256=digest, fetched_at_utc=fetched_at_utc
        )
    return rebuild_derived(config, workspace)


def rebuild_derived(config: RunConfig, workspace: Workspace) -> pd.DataFrame:
    """Rebuild `config`'s derived bars from whatever is already in `data/raw/`.

    Fetches nothing: this is the path a researcher takes after deleting
    `data/derived/`, and it must work with the network unplugged.
    """
    return rebuild_daily_bars(
        RawStore(workspace.raw_root),
        DerivedStore(workspace.derived_root),
        config.symbol,
        config.interval,
        config.months(),
    )


def metrics_of(result: RunResult) -> dict[str, Any]:
    """Flatten a `RunResult` into JSON-serialisable numbers.

    The equity curve is deliberately left out: it belongs in a result artefact,
    not in a one-line trial summary, and the two share this shape.
    """
    return {
        "decision_ts_utc": _iso(result.decision_ts_utc),
        "entry_ts_utc": _iso(result.entry_ts_utc),
        "exit_ts_utc": _iso(result.exit_ts_utc),
        "exit_reason": result.exit_reason,
        # ADR-0001: the reporting block carries liquidation count and dates, and
        # an empty list is the explicit "none" — not a missing field.
        "liquidation_count": len(result.liquidation_dates),
        "liquidation_dates": [_iso(ts) for ts in result.liquidation_dates],
        "entry_price": result.entry_price,
        "exit_price": result.exit_price,
        "n_marks": result.n_marks,
        "cost_bps_per_side": result.cost_bps_per_side,
        "gross_return": result.gross_return,
        "net_return": result.net_return,
        "ann_return_gross": result.ann_return_gross,
        "ann_return_net": result.ann_return_net,
        "ann_vol_net": result.ann_vol_net,
        "sharpe_net": result.sharpe_net,
        "mean_log_return_daily_net": result.mean_log_return_daily_net,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_peak_ts_utc": _iso(result.max_drawdown_peak_ts_utc),
        "max_drawdown_trough_ts_utc": _iso(result.max_drawdown_trough_ts_utc),
        "cost_drag_annualised": result.cost_drag_annualised,
        "cost_drag_as_fraction_of_gross": result.cost_drag_as_fraction_of_gross,
    }


def _iso(timestamp: pd.Timestamp | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.strftime(ISO_SECONDS)


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
