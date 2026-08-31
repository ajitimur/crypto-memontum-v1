"""`momentum` — run a config or a Grid, rebuild derived data, pull the panel, or read trials."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

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
from crypto_momentum.data.cmc_prices import NoPanelPrices, PanelGrainTooCoarse
from crypto_momentum.data.fetch import ArchiveUnavailable
from crypto_momentum.gate import Criterion, GateError, RunGap
from crypto_momentum.data.raw_store import RawWindowAlreadyStored, RawWindowMissing
from crypto_momentum.derive import GapInWindow
from crypto_momentum.provenance import NotAGitRepository
from crypto_momentum.results import GridCellRecord
from crypto_momentum.data.archive_listing import MalformedListing
from crypto_momentum.data.market_caps import UnmappableSymbol
from crypto_momentum.data.symbol_map import AmbiguousTicker, MalformedOverrideTable
from crypto_momentum.data.universe import SymbolNotCovered, UniverseError
from crypto_momentum.runner import (
    ISO_SECONDS,
    Workspace,
    rebuild_all_derived,
    run_config,
    run_gate,
    run_grid,
)
from crypto_momentum.sim.buy_and_hold import NotEnoughBars
from crypto_momentum.sim.cross_sectional import (
    NotEnoughHistory,
    SelectionError,
    TurnoverBudgetBreached,
)
from crypto_momentum.sim.grid import GridError
from crypto_momentum.sim.published import (
    CITATION,
    LEGS,
    LONG_ONLY,
    LONG_SHORT,
    PublishedTableError,
)
from crypto_momentum.sim.report import PROFITABILITY_T_BAR
from crypto_momentum.sim.universe_policy import PolicyError
from crypto_momentum.trials import read_trials

EXIT_REFUSED = 2
# A gate that fails is a finding, not a fault, so it gets its own code rather
# than sharing the one a bad config gets. ADR-0003 expects it to be hard to pass.
EXIT_GATE_FAILED = 3

# Anything a researcher can cause with a bad config or a bad download. These are
# reported as a one-line refusal rather than a traceback.
REFUSALS = (
    AmbiguousTicker,
    ArchiveUnavailable,
    ChecksumMismatch,
    ConfigError,
    GapInWindow,
    GateError,
    GridError,
    MalformedArchiveFile,
    MalformedListing,
    MalformedOverrideTable,
    MalformedPanel,
    NotAGitRepository,
    NoPanelPrices,
    NotEnoughBars,
    NotEnoughHistory,
    PanelAlreadyStored,
    PanelGrainTooCoarse,
    PanelMissing,
    PanelPullFailed,
    PanelWindowNotCovered,
    PolicyError,
    PublishedTableError,
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

    grid = subcommands.add_parser(
        "grid", help="run every cell of a config's Grid in one invocation"
    )
    grid.add_argument("config", type=Path)

    gate = subcommands.add_parser(
        "gate",
        help=(
            "run both halves of the Replication Gate and state an explicit pass "
            "or fail against ADR-0003's fixed tolerances"
        ),
    )
    gate.add_argument("faithful", type=Path, help="the CoinMarketCap-priced grid config")
    gate.add_argument("venue", type=Path, help="the Binance-archive-priced grid config")
    gate.add_argument(
        "--reference-leg",
        choices=LEGS,
        default=LONG_SHORT,
        help=(
            "which leg of Han, Kang and Ryu's Table 14 the verdict is read "
            f"against (default {LONG_SHORT}, the leg ADR-0003 fixes its "
            "tolerances on). ADR-0004 makes this repo long-only, and a long-only "
            f"run cannot produce that leg's liquidations — see {LONG_ONLY}"
        ),
    )

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
        if args.command == "grid":
            return _grid(args.config, workspace)
        if args.command == "gate":
            return _gate(args.faithful, args.venue, workspace, leg=args.reference_leg)
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


def _grid(config_path: Path, workspace: Workspace) -> int:
    """Run the whole Grid and print it as a table, cell by cell.

    The JSON summary stays alone on stdout for the machine; the table is written
    to stderr for the researcher reading along, in the grid's published order.
    Never sorted by outcome — a grid ordered best-first is a ranking, and reading
    a ranking is the mistake the Grid exists to prevent.
    """
    run_at_utc = datetime.now(UTC).strftime(ISO_SECONDS)
    record = run_grid(config_path, workspace, run_at_utc=run_at_utc)
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))

    print(
        f"{record.grid}: {record.n_recorded} of {len(record.cells)} cells recorded, "
        f"{record.n_refused} refused, {record.n_liquidated} liquidated",
        file=sys.stderr,
    )
    # "turnover/wk" rather than "turnover": CONTEXT.md reserves the bare word,
    # and this column is Rebalance Turnover on a weekly basis.
    print(
        f"{'cell':>10}  {'sharpe':>8}  {'t(log)':>8}  {'turnover/wk':>11}  outcome",
        file=sys.stderr,
    )
    for cell in record.cells:
        print(f"{cell.name:>10}  {_describe_cell(cell)}", file=sys.stderr)
    print(f"configurations tried: {record.configurations_tried}", file=sys.stderr)
    if record.working_tree_dirty:
        print(
            "warning: the working tree was dirty, so this grid is not "
            f"reproducible from commit {record.commit[:12]} alone",
            file=sys.stderr,
        )
    return 0


def _gate(
    faithful_path: Path, venue_path: Path, workspace: Workspace, *, leg: str
) -> int:
    """Run the Replication Gate and print the verdict, criterion by criterion.

    Exits `EXIT_GATE_FAILED` on a failure so a script can branch on it. That is
    not an error: ADR-0003 expects the gate to be hard to pass — under both of
    Han et al.'s corrections none of their cross-sectional portfolios clears
    t > 3.0 — and a failure means the pipeline measured something and it did not
    match, which is the finding the gate exists to produce.
    """
    run_at_utc = datetime.now(UTC).strftime(ISO_SECONDS)
    outcome = run_gate(
        faithful_path, venue_path, workspace, run_at_utc=run_at_utc, leg=leg
    )
    print(json.dumps(outcome.record.to_dict(), indent=2, sort_keys=True))

    verdict = "PASS" if outcome.passes else "FAIL"
    print(
        f"\nReplication Gate: {verdict} — read against Han, Kang and Ryu's "
        f"{leg} leg\n{CITATION}",
        file=sys.stderr,
    )
    for run_verdict in (outcome.faithful, outcome.venue):
        print(
            f"\n{run_verdict.run} run ({'passes' if run_verdict.passes else 'fails'}), "
            f"{run_verdict.n_cells_compared} of 21 cells comparable",
            file=sys.stderr,
        )
        # The window and the cost assumption before the numbers they qualify: a
        # net Sharpe without its cost model is not a result (`CONTEXT.md`, Net),
        # and the archive floor is why the two runs do not cover one sample.
        print(
            f"  {_describe_window(outcome.record.windows.get(run_verdict.run, {}))}",
            file=sys.stderr,
        )
        print(
            f"  {_describe_costs(outcome.record.costs.get(run_verdict.run, {}))}",
            file=sys.stderr,
        )
        for criterion in run_verdict.criteria:
            print(f"  {_describe_criterion(criterion)}", file=sys.stderr)
        for warning in run_verdict.warnings:
            print(f"  warning: {warning}", file=sys.stderr)

    print(f"\n{_describe_gap(outcome.gap)}", file=sys.stderr)
    print(_describe_bracket(outcome.record.universe_bracket), file=sys.stderr)
    print(f"\nrecorded at {outcome.path}", file=sys.stderr)
    if outcome.record.working_tree_dirty:
        print(
            "warning: the working tree was dirty, so this verdict is not "
            f"reproducible from commit {outcome.record.commit[:12]} alone",
            file=sys.stderr,
        )
    return 0 if outcome.passes else EXIT_GATE_FAILED


def _describe_window(window: dict[str, Any]) -> str:
    """The window covered and the floor that bounded it, in the result itself.

    Issue #11 asks for the Venue Run's 2017-08-17 archive floor "stated in the
    result rather than footnoted", and a comment in a config file is a footnote.
    The published sample is printed beside it so the shortfall is read rather
    than worked out.
    """
    if not window:
        return "window: not recorded"
    covered = f"{_date(window.get('covered_start_ts_utc'))} to {_date(window.get('covered_end_ts_utc'))}"
    floor = _date(window.get("price_source_floor_ts_utc"))
    below = window.get("n_dates_below_floor")
    shortfall = "" if not below else f", {below} dates below it"
    return (
        f"window: {covered} on {window.get('price_source')} prices "
        f"(floor {floor}{shortfall}); published sample "
        f"{window.get('published_sample')}"
    )


def _describe_costs(costs: dict[str, Any]) -> str:
    """What the run's net figures are net *of* — the Net invariant, on the line.

    A Sharpe quoted without its cost assumption is not a result, so the
    assumption is printed with the criteria rather than left in the JSON.
    """
    if not costs:
        return "costs: not recorded"
    return (
        f"costs: {costs.get('cost_model')} at {costs.get('total_bps_per_side')}bp "
        f"per side, slippage {costs.get('slippage_bps_per_side')}bp, "
        f"{costs.get('funding')}"
    )


def _date(timestamp: str | None) -> str:
    return "—" if not timestamp else timestamp[:10]


def _describe_criterion(criterion: Criterion) -> str:
    """One criterion as a line: what was measured, what it needed, and the mark.

    A criterion that does not apply prints "not required" where its bar would go
    rather than a tick, because a bar nobody was held to is not a bar cleared.
    """
    mark = "pass" if criterion.passed else "FAIL"
    observed = "—" if criterion.observed is None else f"{criterion.observed:.4g}"
    bar = (
        "measured, not required"
        if criterion.required is None
        else f"needs {criterion.required}"
    )
    return f"[{mark:>4}] {criterion.name:<28} {observed:>8}  {bar}"


def _describe_gap(gap: RunGap) -> str:
    """The distance between the two runs — ADR-0003 calls this a result itself.

    It measures how much of the published effect is an artefact of cross-exchange
    aggregate pricing rather than something that could have been traded on one
    venue. Nobody in the surveyed literature reports it.
    """
    correlation = (
        "—" if gap.spearman_between_runs is None else f"{gap.spearman_between_runs:.3f}"
    )
    best = "—" if gap.best_net_sharpe_gap is None else f"{gap.best_net_sharpe_gap:+.3f}"
    return (
        "gap (venue minus faithful), a result in its own right:\n"
        f"  best net Sharpe            {best}\n"
        f"  mean |Sharpe| gap          {gap.mean_absolute_sharpe_gap:.3f} "
        f"across {gap.n_cells_compared} cells\n"
        f"  rank correlation of runs   {correlation}\n"
        f"  liquidation count          {gap.liquidation_count_gap:+d}"
    )


def _describe_bracket(bracket: dict[str, Any]) -> str:
    """The Universe as both bounds, never as one chosen number.

    A count quoted on one bound alone has chosen which listing risk to show, and
    the choice is invisible in the number.
    """
    lines = ["universe bracket, both bounds (symbols tradeable at some point):"]
    for run, bounds in bracket.items():
        if not bounds:
            lines.append(f"  {run:<9} not recorded")
            continue
        stated = ", ".join(f"{name} {count}" for name, count in sorted(bounds.items()))
        lines.append(f"  {run:<9} {stated}")
    return "\n".join(lines)


def _describe_cell(cell: GridCellRecord) -> str:
    """One row of the grid table: the two numbers that decide, or the refusal.

    A refused cell prints its reason where its numbers would be, rather than
    dashes across the row — it was tried, and what stopped it is the finding.
    """
    if not cell.recorded:
        return f"{'—':>8}  {'—':>8}  {'—':>11}  refused: {cell.refused}"
    outcome = (
        "liquidated"
        if cell.liquidated
        else ("clears hurdle" if cell.clears_deployment_hurdle else "below hurdle")
    )
    return (
        f"{_number(cell.metrics.get('sharpe_net')):>8}  "
        f"{_number(cell.metrics.get('mean_log_return_t_stat')):>8}  "
        f"{_percent(cell.weekly_rebalance_turnover):>11}  {outcome}"
    )


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


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
