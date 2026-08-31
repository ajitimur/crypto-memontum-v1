"""`momentum` — run a config, rebuild derived data, pull the panel, or read trials."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from crypto_momentum.config import ConfigError, load_config
from crypto_momentum.data.binance_archive import ChecksumMismatch, MalformedArchiveFile
from crypto_momentum.data.cmc_panel import (
    CmcPanelStore,
    MalformedPanel,
    PanelAlreadyStored,
    PanelMissing,
    PanelPullFailed,
    PanelWindowNotCovered,
    SurvivorshipBiasedPanel,
    pull_panel,
)
from crypto_momentum.data.fetch import ArchiveUnavailable
from crypto_momentum.data.raw_store import RawWindowAlreadyStored, RawWindowMissing
from crypto_momentum.derive import GapInWindow
from crypto_momentum.provenance import NotAGitRepository
from crypto_momentum.data.archive_listing import MalformedListing
from crypto_momentum.data.market_caps import UnmappableSymbol
from crypto_momentum.data.symbol_map import AmbiguousTicker, MalformedOverrideTable
from crypto_momentum.data.universe import SymbolNotCovered, UniverseError
from crypto_momentum.runner import (
    ISO_SECONDS,
    Workspace,
    rebuild_all_derived,
    run_config,
)
from crypto_momentum.sim.buy_and_hold import NotEnoughBars
from crypto_momentum.sim.cross_sectional import (
    NotEnoughHistory,
    SelectionError,
    TurnoverBudgetBreached,
)
from crypto_momentum.sim.report import PROFITABILITY_T_BAR
from crypto_momentum.sim.universe_policy import PolicyError
from crypto_momentum.trials import read_trials

EXIT_REFUSED = 2

# Anything a researcher can cause with a bad config or a bad download. These are
# reported as a one-line refusal rather than a traceback.
REFUSALS = (
    AmbiguousTicker,
    ArchiveUnavailable,
    ChecksumMismatch,
    ConfigError,
    GapInWindow,
    MalformedArchiveFile,
    MalformedListing,
    MalformedOverrideTable,
    MalformedPanel,
    NotAGitRepository,
    NotEnoughBars,
    NotEnoughHistory,
    PanelAlreadyStored,
    PanelMissing,
    PanelPullFailed,
    PanelWindowNotCovered,
    PolicyError,
    RawWindowAlreadyStored,
    RawWindowMissing,
    SelectionError,
    SurvivorshipBiasedPanel,
    SymbolNotCovered,
    # A breach of ADR-0007's turnover budget is a refusal like any other: the
    # config is one we will not trade, and saying so in a line is the answer.
    TurnoverBudgetBreached,
    UniverseError,
    UnmappableSymbol,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="momentum", description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository whose data/, results/ and trials.jsonl the run uses",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run one config end to end")
    run.add_argument("config", type=Path)

    build = subcommands.add_parser(
        "build-derived", help="rebuild derived bars from data/raw/ without fetching"
    )
    build.add_argument("config", type=Path)

    subcommands.add_parser(
        "pull-cmc-panel",
        help="pull the CoinMarketCap market-cap panel once (ADR-0008); a no-op if stored",
    )

    subcommands.add_parser("trials", help="show every configuration tried")

    args = parser.parse_args(argv)
    workspace = Workspace.under(args.repo_root)

    try:
        if args.command == "run":
            return _run(args.config, workspace)
        if args.command == "build-derived":
            return _build_derived(args.config, workspace)
        if args.command == "pull-cmc-panel":
            return _pull_cmc_panel(workspace)
        return _trials(workspace)
    except REFUSALS as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


def _run(config_path: Path, workspace: Workspace) -> int:
    # The wall clock is read here, at the edge, and passed in as a parameter so
    # that nothing below the CLI depends on it.
    run_at_utc = datetime.now(UTC).strftime(ISO_SECONDS)
    record = run_config(config_path, workspace, run_at_utc=run_at_utc)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    # The result JSON is the machine's copy and stays alone on stdout; the
    # reporting block is written beside it for the researcher reading along.
    for line in (
        f"liquidation: {_describe_liquidation(record.metrics)}",
        f"profitability: {_describe_profitability(record.metrics)}",
        f"means: {_describe_divergence(record.metrics)}",
        f"hurdle: {_describe_hurdle(record.benchmarks)}",
        f"configurations tried: {record.configurations_tried}",
    ):
        print(line, file=sys.stderr)
    if record.working_tree_dirty:
        print(
            "warning: the working tree was dirty, so this result is not "
            f"reproducible from commit {record.commit[:12]} alone",
            file=sys.stderr,
        )
    return 0


def _build_derived(config_path: Path, workspace: Workspace) -> int:
    config = load_config(config_path)
    built = rebuild_all_derived(config, workspace)
    total = sum(len(bars) for bars in built.values())
    print(
        f"rebuilt {total} {config.interval} bars across {len(built)} symbol(s) "
        "from data/raw/"
    )
    for symbol, bars in sorted(built.items()):
        print(f"  {symbol}  {len(bars)} bars  {bars.index[0].date()} to {bars.index[-1].date()}")
    return 0


def _pull_cmc_panel(workspace: Workspace) -> int:
    """Pull the panel if it is absent, and say plainly when it is not.

    The wall clock is read here, at the edge, and passed down as a parameter.
    """
    store = CmcPanelStore(workspace.raw_root)
    already_stored = store.has_panel()
    path = pull_panel(
        store,
        pulled_at_utc=datetime.now(UTC).strftime(ISO_SECONDS),
        repo_root=workspace.repo_root,
    )
    manifest = store.manifest()
    if already_stored:
        print(
            f"the panel is already at {path}, pulled {manifest['pulled_at_utc']}. "
            "Per ADR-0008 it is pulled once; nothing was fetched."
        )
        return 0
    print(
        f"pulled {manifest['assets']} assets from {manifest['first_snapshot']} to "
        f"{manifest['last_snapshot']} into {path}\n"
        f"  sha256 {manifest['sha256']}"
    )
    return 0


def _trials(workspace: Workspace) -> int:
    trials = read_trials(workspace.trials_path)
    plural = "" if len(trials) == 1 else "s"
    print(f"{len(trials)} configuration{plural} tried")
    for trial in trials:
        print(f"  {trial.get('run_at_utc')}  {trial.get('config_name')}  {_outcome(trial)}")
    return 0


def _outcome(reported: dict) -> str:
    """What the configuration gave, or why it gave nothing.

    A run refused on its turnover budget produced no metrics, so printing
    `net_return=None` beside it would render a configuration rejected on its
    merits identically to a result that failed to record — the same confusion of
    an absence with a value that `_describe_liquidation` exists to avoid.
    """
    if reported.get("refused"):
        realised = reported.get("weekly_rebalance_turnover")
        budget = reported.get("turnover_budget_weekly")
        measured = "unmeasured" if realised is None else f"{realised:.1%}"
        allowed = "unstated" if budget is None else f"{budget:.1%}"
        return f"refused={reported['refused']}  weekly_turnover={measured} vs budget {allowed}"
    return (
        f"net_return={reported.get('net_return')}  "
        f"liquidation={_describe_liquidation(reported)}"
    )


def _describe_liquidation(reported: dict) -> str:
    """The reporting block's liquidation line: a count with dates, or "none".

    ADR-0001 asks for an explicit "none" rather than a silent absence. A trial
    logged before the run was marked daily has no answer to give and says so,
    rather than reading like a run that was checked and survived.
    """
    if "liquidation_dates" not in reported:
        return "not recorded"
    dates = reported["liquidation_dates"]
    if not dates:
        return "none"
    plural = "" if len(dates) == 1 else "s"
    return f"{len(dates)} event{plural} — {', '.join(dates)}"


def _describe_profitability(reported: dict) -> str:
    """The statistic that decides, as ADR-0002 has it: mean log return at t > 3.0.

    Quoted with its bandwidth, because a t-statistic without the lag count behind
    it is not something a later run can be compared to.
    """
    mean_log = reported.get("mean_log_return_daily_net")
    t_statistic = reported.get("mean_log_return_t_stat")
    if t_statistic is None:
        return (
            "no t-statistic — the path has no finite mean log return to test "
            f"(mean log return {mean_log})"
        )
    # The verdict is read off the result rather than re-derived here, so the bar
    # is applied in exactly one place and moving it moves both.
    verdict = "clears" if reported.get("clears_profitability_bar") else "below"
    # The number of marks is quoted because the estimator carries no small-sample
    # correction: on a short window the standard error is biased down and the
    # t-statistic up, towards clearing. A reader cannot weigh that without T.
    return (
        f"mean log return {mean_log:.6f}/day, t = {t_statistic:.2f} "
        f"(Newey-West, {reported.get('newey_west_lags')} lags, "
        f"{reported.get('n_marks')} marks) — "
        f"{verdict} the t > {PROFITABILITY_T_BAR} bar"
    )


def _describe_divergence(reported: dict) -> str:
    """Whether the two means disagree in sign — ADR-0002 asks for this out loud.

    A positive mean return on a negative mean log return is a strategy that
    tests significant while losing money compounded, so it is stated rather than
    left for a reader to spot in two adjacent numbers.
    """
    mean_return = reported.get("mean_return_daily_net")
    mean_log = reported.get("mean_log_return_daily_net")
    if mean_log is None:
        # The most extreme divergence there is — a path that compounds to
        # nothing has no mean log return at all — so it is said, not skipped.
        return (
            f"mean return {mean_return} has no mean log return to be compared "
            "with: the run was liquidated, and a wipeout compounds to nothing "
            "whatever its mean return says"
        )
    if not reported.get("mean_return_sign_divergence"):
        return f"mean return {mean_return} and mean log return {mean_log} agree in sign"
    return (
        f"mean return {mean_return} and mean log return {mean_log} disagree in "
        "sign — the log return governs, and the divergence is diagnostic of the "
        "tail behaviour that kills these strategies"
    )


def _describe_hurdle(benchmarks: dict) -> str:
    """ADR-0005's three conditions, and which of them a run failed on."""
    hurdle = benchmarks.get("deployment_hurdle")
    if hurdle is None:
        return "not recorded"
    conditions = ("sharpe_above_btc", "drawdown_no_worse_than_btc", "clears_profitability_bar")
    if hurdle.get("clears"):
        return (
            "cleared — better Sharpe than BTC, no worse drawdown, and "
            f"t > {PROFITABILITY_T_BAR}"
        )
    failed = [name for name in conditions if hurdle.get(name) is not True]
    return f"not cleared — {', '.join(failed)}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
