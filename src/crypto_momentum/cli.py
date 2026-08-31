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
from crypto_momentum.runner import ISO_SECONDS, Workspace, rebuild_derived, run_config
from crypto_momentum.sim.buy_and_hold import NotEnoughBars
from crypto_momentum.trials import read_trials

EXIT_REFUSED = 2

# Anything a researcher can cause with a bad config or a bad download. These are
# reported as a one-line refusal rather than a traceback.
REFUSALS = (
    ArchiveUnavailable,
    ChecksumMismatch,
    ConfigError,
    GapInWindow,
    MalformedArchiveFile,
    MalformedPanel,
    NotAGitRepository,
    NotEnoughBars,
    PanelAlreadyStored,
    PanelMissing,
    PanelPullFailed,
    PanelWindowNotCovered,
    RawWindowAlreadyStored,
    RawWindowMissing,
    SurvivorshipBiasedPanel,
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
    if record.working_tree_dirty:
        print(
            "warning: the working tree was dirty, so this result is not "
            f"reproducible from commit {record.commit[:12]} alone",
            file=sys.stderr,
        )
    return 0


def _build_derived(config_path: Path, workspace: Workspace) -> int:
    config = load_config(config_path)
    bars = rebuild_derived(config, workspace)
    print(f"rebuilt {len(bars)} {config.interval} bars for {config.symbol} from data/raw/")
    return 0


def _pull_cmc_panel(workspace: Workspace) -> int:
    """Pull the panel if it is absent, and say plainly when it is not.

    The wall clock is read here, at the edge, and passed down as a parameter.
    """
    store = CmcPanelStore(workspace.raw_root)
    already_stored = store.has()
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
        print(
            f"  {trial.get('run_at_utc')}  {trial.get('config_name')}  "
            f"net_return={trial.get('net_return')}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
