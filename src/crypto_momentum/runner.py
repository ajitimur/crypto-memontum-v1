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

from crypto_momentum.config import CROSS_SECTIONAL, RunConfig, load_config
from crypto_momentum.data.binance_archive import monthly_klines_file
from crypto_momentum.data.cmc_panel import CmcPanelStore
from crypto_momentum.data.fetch import ArchiveUnavailable, UrlOpener, fetch_archive_file
from crypto_momentum.data.market_caps import market_cap_panel
from crypto_momentum.data.raw_store import RawStore, RawWindowMissing
from crypto_momentum.data.universe import (
    SymbolCoverage,
    UniverseError,
    bar_span_from_bars,
    build_universe_panel,
    coverage_for_symbol,
)
from crypto_momentum.derive import DerivedStore, build_daily_bars, rebuild_daily_bars
from crypto_momentum.policy import (
    EXCLUSIONS_FILENAME,
    TOKOCRYPTO_LISTING_FILENAME,
    load_exclusion_list,
    load_venue_listing,
    policy_root,
)
from crypto_momentum.provenance import describe_head
from crypto_momentum.results import ResultStore, RunRecord
from crypto_momentum.sim.benchmarks import (
    BENCHMARK_SYMBOL,
    btc_buy_and_hold,
    cap_weighted_market,
    deployment_hurdle,
)
from crypto_momentum.sim.buy_and_hold import NotEnoughBars, simulate_buy_and_hold
from crypto_momentum.sim.cross_sectional import CrossSectionalRun, simulate_cross_sectional
from crypto_momentum.sim.report import RunResult
from crypto_momentum.sim.universe_policy import (
    TOKOCRYPTO,
    LiquidityFloor,
    apply_universe_policy,
    dollar_volume_from_bars,
)
from crypto_momentum.trials import TRIALS_FILENAME, append_trial, read_trials

# Timestamps that cross the JSON boundary are second-resolution UTC throughout.
ISO_SECONDS = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class RunOutput:
    """What one strategy path hands back to be recorded.

    Four blocks rather than one dict, because they answer four different
    questions: what came out, what window it came from, how the positions were
    formed, and what the run is read against.
    """

    metrics: dict[str, Any]
    window: dict[str, Any]
    portfolio: dict[str, Any]
    benchmarks: dict[str, Any]


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

    if config.strategy_kind == CROSS_SECTIONAL:
        output = _run_cross_sectional(
            config, workspace, run_at_utc=run_at_utc, open_url=open_url
        )
    else:
        output = _run_single_asset(
            config, workspace, run_at_utc=run_at_utc, open_url=open_url
        )

    record = RunRecord(
        commit=provenance.commit,
        working_tree_dirty=provenance.working_tree_dirty,
        run_at_utc=run_at_utc,
        config=config,
        config_sha256=config_sha256,
        config_path=_relative_to_repo(config_path, workspace.repo_root),
        metrics=output.metrics,
        window=output.window,
        portfolio=output.portfolio,
        benchmarks=output.benchmarks,
        # Two different counts, because they answer two different questions. The
        # protocol asks how many *configurations* were tried to reach a quoted
        # number, so that is distinct config fingerprints and a config re-run
        # unchanged does not inflate it — a 21-cell Grid run twice is 21 tried,
        # not 42. How many runs were recorded is kept beside it. This run counts
        # in both: it is one of the configurations the number came from. The log
        # is appended below, so both are taken before it.
        **_counts_tried(workspace.trials_path, config_sha256=config_sha256),
    )
    ResultStore(workspace.results_root).write(record)
    append_trial(workspace.trials_path, record.trial_line())
    return record


def _run_single_asset(
    config: RunConfig,
    workspace: Workspace,
    *,
    run_at_utc: str,
    open_url: UrlOpener | None,
) -> RunOutput:
    """The walking skeleton's path: one symbol, held from the fill to the end."""
    bars = load_bars(config, workspace, fetched_at_utc=run_at_utc, open_url=open_url)
    result = simulate_buy_and_hold(bars, cost_bps_per_side=config.cost_bps_per_side)
    return RunOutput(
        metrics=metrics_of(result),
        window={
            "months": config.months(),
            "first_bar_ts_utc": _iso(bars.index[0]),
            "last_bar_ts_utc": _iso(bars.index[-1]),
            "n_bars": len(bars),
        },
        # A hold has no selection to describe, but the reporting block still
        # asks how much was invested and in how many names, and the answer for
        # a long-only unlevered single hold is: all of it, in one, every mark.
        portfolio={
            "strategy": config.strategy_kind,
            "long_only": True,
            "levered": False,
            "trend_gate": False,
            "n_rebalances": 1,
            "mean_n_positions": 1.0,
            "mean_gross_exposure": 1.0,
            "mean_net_exposure": 1.0,
            "max_gross_exposure": 1.0,
            "mean_rebalance_turnover": 1.0,
        },
        benchmarks=_benchmarks(
            result,
            config=config,
            workspace=workspace,
            bars_by_symbol={config.symbol: bars} if config.symbol else {},
            run_at_utc=run_at_utc,
            open_url=open_url,
            market=None,
            market_unavailable=(
                "a single-asset hold has no point-in-time Universe to weight by "
                "capitalisation"
            ),
        ),
    )


def _run_cross_sectional(
    config: RunConfig,
    workspace: Workspace,
    *,
    run_at_utc: str,
    open_url: UrlOpener | None,
) -> RunOutput:
    """The cross-sectional path, where the three layers meet.

    In order: the archive says which months each symbol ever published and the
    bars are built from those; the point-in-time Universe is reconstructed from
    that same coverage, narrowed to the days real bars exist for; policy removes
    what we would not consider holding; the vendor panel supplies the weights;
    and only then does the simulator see anything.

    Nothing here reaches past the run's own window, and nothing in the
    simulation core reaches back out to a store.
    """
    bars_by_symbol, coverages = load_cross_section(
        config, workspace, fetched_at_utc=run_at_utc, open_url=open_url
    )
    panel = build_universe_panel(
        coverages,
        start=_window_start(config),
        end=_window_end(config),
        bar_span_by_symbol={
            symbol: bar_span_from_bars(bars) for symbol, bars in bars_by_symbol.items()
        },
    )

    policy = policy_root(workspace.repo_root)
    floor = (
        LiquidityFloor(
            floor_usd=config.liquidity_floor_usd,
            window_days=config.liquidity_window_days,
        )
        if config.liquidity_floor_usd is not None
        else None
    )
    dollar_volume = dollar_volume_from_bars(bars_by_symbol)
    after_policy = apply_universe_policy(
        panel,
        exclusions=load_exclusion_list(policy / EXCLUSIONS_FILENAME),
        bracket=config.bracket,
        # The lower bound of the bracket is the only one that needs a listing,
        # but it is loaded either way so the two ends run off one artefact.
        venue_listing=load_venue_listing(policy / TOKOCRYPTO_LISTING_FILENAME),
        dollar_volume=dollar_volume if floor is not None else None,
        floor=floor,
    )

    caps = market_cap_panel(
        CmcPanelStore(workspace.raw_root).read_panel(),
        config.universe_symbols,
        repo_root=workspace.repo_root,
    )
    run = simulate_cross_sectional(
        bars_by_symbol,
        tradeable=after_policy.tradeable,
        market_caps=caps,
        lookback_days=config.lookback_days,
        holding_days=config.holding_days,
        quantile=config.quantile,
        min_universe=config.min_universe,
        max_cap_staleness_days=config.max_cap_staleness_days,
        cost_bps_per_side=config.cost_bps_per_side,
    )

    # The market portfolio runs on the same Universe, the same panel and the
    # same cadence as the strategy, so the only difference between the two paths
    # is that one of them ranked on the signal.
    market = cap_weighted_market(
        bars_by_symbol,
        tradeable=after_policy.tradeable,
        market_caps=caps,
        lookback_days=config.lookback_days,
        holding_days=config.holding_days,
        max_cap_staleness_days=config.max_cap_staleness_days,
        cost_bps_per_side=config.cost_bps_per_side,
    )

    spans = [bars.index for bars in bars_by_symbol.values()]
    return RunOutput(
        metrics=metrics_of(run.result),
        window={
            "months": config.months(),
            "first_bar_ts_utc": _iso(min(span[0] for span in spans)),
            "last_bar_ts_utc": _iso(max(span[-1] for span in spans)),
            "n_symbols": len(bars_by_symbol),
            "n_bars_by_symbol": {
                symbol: len(bars) for symbol, bars in sorted(bars_by_symbol.items())
            },
            "universe": after_policy.metadata,
        },
        portfolio=run.to_metadata(),
        benchmarks=_benchmarks(
            run.result,
            config=config,
            workspace=workspace,
            bars_by_symbol=bars_by_symbol,
            run_at_utc=run_at_utc,
            open_url=open_url,
            market=market,
            market_unavailable=None,
        ),
    )


def _counts_tried(trials_path: Path, *, config_sha256: str) -> dict[str, int]:
    """How many configurations have been tried, and how many runs recorded.

    A configuration is its fingerprint, not its name: editing a config and
    running it again is a second configuration tried, and running the same bytes
    twice is not. A trial logged before fingerprints were recorded has no
    fingerprint to count and is left out of the configuration count rather than
    collapsed into one — it is still counted as a run.
    """
    trials = read_trials(trials_path)
    fingerprints = {
        trial["config_sha256"] for trial in trials if trial.get("config_sha256")
    }
    return {
        "configurations_tried": len(fingerprints | {config_sha256}),
        "trials_recorded": len(trials) + 1,
    }


def _benchmarks(
    result: RunResult,
    *,
    config: RunConfig,
    workspace: Workspace,
    bars_by_symbol: dict[str, pd.DataFrame],
    run_at_utc: str,
    open_url: UrlOpener | None,
    market: CrossSectionalRun | None,
    market_unavailable: str | None,
) -> dict[str, Any]:
    """What the run is read against, per ADR-0005.

    BTC buy-and-hold is computed on every run, whatever the strategy, because it
    is the hurdle and not an option. Where it cannot be computed — the archive
    published no Bitcoin over the window — the block says so and the hurdle
    fails, rather than the field going quietly missing.
    """
    btc, btc_unavailable = _btc_over_the_same_window(
        result,
        config=config,
        workspace=workspace,
        bars_by_symbol=bars_by_symbol,
        run_at_utc=run_at_utc,
        open_url=open_url,
    )
    return {
        "btc_buy_and_hold": {
            "symbol": BENCHMARK_SYMBOL,
            "computed": btc is not None,
            **({"reason": btc_unavailable} if btc is None else metrics_of(btc)),
        },
        "cap_weighted_market": {
            "computed": market is not None,
            **(
                {"reason": market_unavailable}
                if market is None
                else {
                    **metrics_of(market.result),
                    # The same shape the strategy's own block reports, so the
                    # reference is read on the criteria the run is read on.
                    "n_rebalances": market.n_rebalances,
                    "mean_n_positions": market.mean_n_positions,
                    "mean_rebalance_turnover": market.mean_rebalance_turnover,
                    "mean_gross_exposure": market.mean_gross_exposure,
                    "mean_net_exposure": market.mean_gross_exposure,
                }
            ),
        },
        "deployment_hurdle": deployment_hurdle(result, btc=btc).to_metadata(),
    }


def _btc_over_the_same_window(
    result: RunResult,
    *,
    config: RunConfig,
    workspace: Workspace,
    bars_by_symbol: dict[str, pd.DataFrame],
    run_at_utc: str,
    open_url: UrlOpener | None,
) -> tuple[RunResult | None, str | None]:
    """BTC bought and held over the run's own window, or why it could not be.

    Both edges are the strategy's, not the config's. The hold starts at the
    strategy's Decision Bar, because a lookback the strategy spends warming up is
    not a window BTC was held over; and it ends at the strategy's last mark,
    because a run that halted or liquidated in February must not be read against
    a Bitcoin that kept compounding until March. Comparing a hold over more days
    to one over fewer is not the comparison ADR-0005 asks for.

    The bars are reused when the run already loaded them, and fetched through
    the same raw-then-derived path when it did not, so the hurdle never reads a
    price the run itself could not have.
    """
    bars = bars_by_symbol.get(BENCHMARK_SYMBOL)
    if bars is None:
        try:
            loaded = _load_symbol_bars(
                BENCHMARK_SYMBOL, config, workspace, fetched_at_utc=run_at_utc, open_url=open_url
            )
        except (ArchiveUnavailable, UniverseError) as error:
            return None, f"{BENCHMARK_SYMBOL} could not be read: {error}"
        if loaded is None:
            return None, (
                f"the archive publishes no {config.interval} bars for "
                f"{BENCHMARK_SYMBOL} in {config.start_month} to {config.end_month}"
            )
        bars, _ = loaded

    held = bars.loc[
        (bars.index >= result.decision_ts_utc) & (bars.index <= result.exit_ts_utc)
    ]
    try:
        return btc_buy_and_hold(held, cost_bps_per_side=config.cost_bps_per_side), None
    except NotEnoughBars as error:
        return None, f"{BENCHMARK_SYMBOL} has no window to be held over: {error}"


def load_cross_section(
    config: RunConfig,
    workspace: Workspace,
    *,
    fetched_at_utc: str,
    open_url: UrlOpener | None = None,
) -> tuple[dict[str, pd.DataFrame], list[SymbolCoverage]]:
    """Bars for every symbol in the cross-section, and the coverage behind them.

    A symbol is fetched only for the months the archive actually publishes it
    in. Asking a delisted asset for a month after it stopped trading is a 404
    that reads like a network fault; asking coverage first turns it into the
    fact it is, which is that the asset was gone by then.

    A symbol the archive never published at this interval, or published nothing
    for inside the window, is dropped rather than carried as an empty column.
    Its absence is visible in the Universe metadata, which counts the symbols
    that made it in.
    """
    wanted = set(config.months())

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    coverages: list[SymbolCoverage] = []
    for symbol in config.universe_symbols:
        loaded = _load_symbol_bars(
            symbol, config, workspace, fetched_at_utc=fetched_at_utc, open_url=open_url
        )
        if loaded is None:
            continue
        bars, coverage = loaded
        bars_by_symbol[symbol] = bars
        # The coverage handed to the panel is clipped to the run's own window, so
        # a symbol is not marked tradeable on months this run never looked at.
        coverages.append(
            SymbolCoverage(
                symbol=symbol,
                interval=coverage.interval,
                months=tuple(
                    month for month in config.months() if month in set(coverage.months)
                ),
                daily_only_dates=tuple(
                    day for day in coverage.daily_only_dates if day[:7] in wanted
                ),
            )
        )
    if not bars_by_symbol:
        raise RawWindowMissing(
            f"the archive publishes no {config.interval} bars for any of "
            f"{', '.join(config.universe_symbols)} in {config.start_month} to "
            f"{config.end_month}"
        )
    return bars_by_symbol, coverages


def _load_symbol_bars(
    symbol: str,
    config: RunConfig,
    workspace: Workspace,
    *,
    fetched_at_utc: str,
    open_url: UrlOpener | None = None,
) -> tuple[pd.DataFrame, SymbolCoverage] | None:
    """One symbol's daily bars over the config's window, and its archive coverage.

    Only the months the archive actually publishes the symbol in are fetched, and
    only those not already in `data/raw/`: raw data is append-only, and asking a
    delisted asset for a month after it stopped trading is a 404 that reads like
    a network fault.

    `None` when the archive published nothing for the symbol inside the window.
    """
    coverage = coverage_for_symbol(symbol, config.interval, open_url=open_url)
    months = [month for month in config.months() if month in set(coverage.months)]
    if not months:
        return None

    raw_store = RawStore(workspace.raw_root)
    for month in months:
        archive_file = monthly_klines_file(symbol, config.interval, month)
        if raw_store.has(archive_file):
            continue
        payload, digest = fetch_archive_file(archive_file, open_url=open_url)
        raw_store.write(
            archive_file, payload, sha256=digest, fetched_at_utc=fetched_at_utc
        )
    bars = build_daily_bars(raw_store, symbol, config.interval, months)
    DerivedStore(workspace.derived_root).write_bars(symbol, config.interval, bars)
    return bars, coverage


def _window_start(config: RunConfig) -> pd.Timestamp:
    return pd.Timestamp(f"{config.start_month}-01T00:00:00Z")


def _window_end(config: RunConfig) -> pd.Timestamp:
    start = pd.Timestamp(f"{config.end_month}-01T00:00:00Z")
    return start + pd.offsets.MonthEnd(0)


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


def rebuild_all_derived(
    config: RunConfig, workspace: Workspace
) -> dict[str, pd.DataFrame]:
    """Rebuild every symbol the config names, from whatever is in `data/raw/`.

    Fetches nothing, and lists nothing: this is the path a researcher takes
    after deleting `data/derived/`, and it must work with the network unplugged.
    A symbol is rebuilt from the months actually stored for it, so a run that
    only ever fetched a delisted asset's live months rebuilds exactly those.
    """
    raw_store = RawStore(workspace.raw_root)
    derived_store = DerivedStore(workspace.derived_root)
    built: dict[str, pd.DataFrame] = {}
    for symbol in config.universe_symbols:
        months = [
            month
            for month in config.months()
            if raw_store.has(monthly_klines_file(symbol, config.interval, month))
        ]
        if not months:
            continue
        built[symbol] = rebuild_daily_bars(
            raw_store, derived_store, symbol, config.interval, months
        )
    if not built:
        raise RawWindowMissing(
            f"data/raw/ holds no {config.interval} months for "
            f"{', '.join(config.universe_symbols)}; run the config to fetch them"
        )
    return built


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
        # ADR-0002: the mean log return and its Newey-West t-statistic decide;
        # the mean return is reported beside them for comparison with published
        # work, and their disagreeing in sign is itself a reported finding.
        "mean_return_daily_net": result.mean_return_daily_net,
        "mean_log_return_daily_net": result.mean_log_return_daily_net,
        "mean_log_return_t_stat": result.mean_log_return_t_stat,
        "newey_west_lags": result.newey_west_lags,
        "mean_return_sign_divergence": result.mean_return_sign_divergence,
        "clears_profitability_bar": result.clears_profitability_bar,
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
