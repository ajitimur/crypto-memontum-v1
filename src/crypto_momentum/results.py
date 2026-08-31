"""`results/` — one JSON file per (commit, config).

The key is the pair, because either half alone is ambiguous: the same config at
two commits is two results, and two configs at one commit are two results.

A Grid adds a third level and no new ambiguity: its cells are 21 configs at one
commit, filed together under the grid's own directory because they are read
together. The shape of the grid is the result; a cell on its own is not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crypto_momentum.config import RunConfig


def configuration_fingerprint(config_sha256: str, *, cell: str | None = None) -> str:
    """What makes this run one of the configurations tried, rather than another.

    The config file's bytes for a plain run — editing a config and running it
    again is a second configuration, and running the same bytes twice is not.

    For a cell of a Grid it is those bytes *and* the cell, because 21 cells are
    21 configurations tried and not one. That is the multiple testing the
    reporting protocol asks to be counted, and understating it is the exact
    error Han, Kang and Ryu correct for when they hold their own grid to
    t > 3.0 rather than t > 2.0.
    """
    if cell is None:
        return config_sha256
    return hashlib.sha256(f"{config_sha256}\n{cell}".encode()).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to say what was run, on what code, and what came out."""

    commit: str
    working_tree_dirty: bool
    run_at_utc: str
    config: RunConfig
    config_sha256: str
    config_path: str
    metrics: dict[str, Any]
    window: dict[str, Any] = field(default_factory=dict)
    # How the positions were formed: the strategy's own knobs, its turnover, its
    # exposure. Empty for a strategy that holds one thing and never trades again.
    portfolio: dict[str, Any] = field(default_factory=dict)
    # What the run was priced in: the named cost model broken into fee, tax and
    # levy, plus the slippage assumption. A net figure without this beside it is
    # not a result, per the Net invariant.
    costs: dict[str, Any] = field(default_factory=dict)
    # What the run is read against: BTC buy-and-hold, the cap-weighted market
    # portfolio, and ADR-0005's three-condition hurdle over the same window.
    benchmarks: dict[str, Any] = field(default_factory=dict)
    # How many configurations had been tried, this one included, when it ran —
    # counted by config fingerprint, so a re-run of the same bytes is not a
    # second configuration. The protocol requires it beside every quoted number,
    # so it travels with the number rather than living only in a log someone has
    # to go and count. `trials_recorded` is the run count beside it.
    configurations_tried: int = 0
    trials_recorded: int = 0
    # The published Grid this run is one cell of, if it is one — the name in
    # `sim/grid.py`, so a cell says which grid it belongs to and not merely that
    # it belongs to one.
    grid: str = ""
    # Two configs running `han-kang-ryu-21` over different windows are two
    # grids, so the cells are filed under the config's name and not the
    # published grid's — under which the second would overwrite the first.
    grid_config_name: str = ""
    # Which of the configurations tried this one is — see
    # `configuration_fingerprint`. Empty falls back to the config file's digest,
    # which is what it is for anything that is not a grid cell.
    fingerprint: str = ""

    @property
    def configuration_fingerprint(self) -> str:
        return self.fingerprint or self.config_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "run_at_utc": self.run_at_utc,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "configuration_fingerprint": self.configuration_fingerprint,
            # Only where there is one. A `"grid": ""` on every single-run result
            # would read as a grid that failed to name itself.
            **({"grid": self.grid} if self.grid else {}),
            "config": self._config_as_written(),
            "window": self.window,
            "costs": self.costs,
            "portfolio": self.portfolio,
            "benchmarks": self.benchmarks,
            "configurations_tried": self.configurations_tried,
            "trials_recorded": self.trials_recorded,
            "metrics": self.metrics,
        }

    def _config_as_written(self) -> dict[str, Any]:
        """The config as the TOML file states it, cost model named rather than spelt.

        The model's fee, tax and levy live in the `costs` block, once. Nesting
        them here as well would put two copies of the same three numbers in one
        file — they cannot disagree today, but a reader has no way to know which
        is the authority, and a later edit to one of them would make it matter.
        """
        written = asdict(self.config)
        written["cost_model"] = self.config.cost_model.name
        return written

    def trial_line(self) -> dict[str, Any]:
        """The flat one-line summary appended to the trials log.

        Carries the result key and the headline numbers, so the log answers
        "how many configurations were tried, and what did they give" without
        opening every result file.
        """
        return {
            **trial_identity(
                self.config,
                commit=self.commit,
                working_tree_dirty=self.working_tree_dirty,
                run_at_utc=self.run_at_utc,
                config_path=self.config_path,
                config_sha256=self.config_sha256,
                configuration_fingerprint=self.configuration_fingerprint,
                grid=self.grid,
            ),
            "configurations_tried": self.configurations_tried,
            "trials_recorded": self.trials_recorded,
            # Whether the run cleared ADR-0005's hurdle, so the log answers "did
            # any of these beat holding Bitcoin" without opening every result.
            **(
                {"clears_deployment_hurdle": self.benchmarks["deployment_hurdle"]["clears"]}
                if "deployment_hurdle" in self.benchmarks
                else {}
            ),
            # The headline shape of the portfolio, so the log answers "how much
            # did this one trade" without opening the result file. The full
            # block, halt exits and all, stays in the result.
            **{
                key: self.portfolio[key]
                for key in (
                    "n_rebalances",
                    "mean_n_positions",
                    "mean_rebalance_turnover",
                    "weekly_rebalance_turnover",
                    "mean_net_exposure",
                )
                if key in self.portfolio
            },
            **self.metrics,
        }


def trial_identity(
    config: RunConfig,
    *,
    commit: str,
    working_tree_dirty: bool,
    run_at_utc: str,
    config_path: str,
    config_sha256: str,
    configuration_fingerprint: str | None = None,
    grid: str = "",
) -> dict[str, Any]:
    """Which run this line is about — the half every trials line shares.

    One definition, because there are two kinds of line: a run that produced a
    result, and a run refused before it could. Both have to be countable and
    identifiable in the same way, and building the keys twice is how the two
    quietly drift into two schemas.

    A cell of a Grid names its grid and the two knobs the grid gave it, so the
    log can be read cell by cell without opening 21 result files.
    """
    return {
        "run_at_utc": run_at_utc,
        "commit": commit,
        "working_tree_dirty": working_tree_dirty,
        "config_name": config.name,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "configuration_fingerprint": configuration_fingerprint or config_sha256,
        **({"grid": grid} if grid else {}),
        **(
            {
                "lookback_days": config.lookback_days,
                "holding_days": config.holding_days,
            }
            if config.lookback_days is not None
            else {}
        ),
        "symbol": config.symbol,
        "n_symbols": len(config.universe_symbols),
        "interval": config.interval,
        "strategy_kind": config.strategy_kind,
        "start_month": config.start_month,
        "end_month": config.end_month,
        "cost_model": config.cost_model.name,
        "cost_bps_per_side": config.cost_bps_per_side,
    }


TURNOVER_BUDGET_BREACHED = "turnover_budget_breached"


def refused_trial_line(
    config: RunConfig,
    *,
    commit: str,
    working_tree_dirty: bool,
    run_at_utc: str,
    config_path: str,
    config_sha256: str,
    configurations_tried: int,
    trials_recorded: int,
    refused: str,
    reason: str,
    configuration_fingerprint: str | None = None,
    grid: str = "",
    realised_weekly_turnover: float | None = None,
    budget: float | None = None,
) -> dict[str, Any]:
    """The trials line for a configuration that was tried and then refused.

    Deliberately the same identity as a successful trial, so the log can be
    counted without filtering, and deliberately carrying no metrics, because none
    were produced. `refused` is what tells the two apart, and it names the kind
    of refusal; a turnover breach also names the two numbers that did it.
    """
    return {
        **trial_identity(
            config,
            commit=commit,
            working_tree_dirty=working_tree_dirty,
            run_at_utc=run_at_utc,
            config_path=config_path,
            config_sha256=config_sha256,
            configuration_fingerprint=configuration_fingerprint,
            grid=grid,
        ),
        # A refused configuration is still one of the configurations tried, so it
        # carries the same counts a recorded run does — otherwise the search
        # behind a later result would silently omit the ones the ceiling stopped.
        "configurations_tried": configurations_tried,
        "trials_recorded": trials_recorded,
        "refused": refused,
        "refused_reason": reason,
        # Only where they were measured. A cell refused before it walked has no
        # realised turnover, and a zero there would read as one that traded
        # nothing rather than one that never got to trade.
        **(
            {}
            if realised_weekly_turnover is None
            else {"weekly_rebalance_turnover": realised_weekly_turnover}
        ),
        **({} if budget is None else {"turnover_budget_weekly": budget}),
    }


RECORDED = "recorded"
REFUSED = "refused"

GRID_FILENAME = "grid.json"


@dataclass(frozen=True)
class GridCellRecord:
    """What one cell of a Grid came to, in the shape the grid is read across.

    Carries the headline numbers rather than the whole result, because the file
    a reader opens first is the grid and the question they open it with is about
    the column, not the cell. The cell's own result file has everything.
    """

    lookback_days: int
    holding_days: int
    name: str
    config_name: str
    outcome: str
    metrics: dict[str, Any] = field(default_factory=dict)
    n_rebalances: int | None = None
    weekly_rebalance_turnover: float | None = None
    clears_deployment_hurdle: bool | None = None
    refused: str | None = None
    refused_reason: str | None = None

    @property
    def recorded(self) -> bool:
        return self.outcome == RECORDED

    @property
    def liquidated(self) -> bool:
        """Whether the cell was wiped out mid-holding-period, per ADR-0001.

        Read off the count rather than compared against it at each call site, so
        the grid's tally and the line a researcher reads cannot disagree.
        """
        return self.metrics.get("liquidation_count", 0) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.name,
            "config_name": self.config_name,
            "lookback_days": self.lookback_days,
            "holding_days": self.holding_days,
            "outcome": self.outcome,
            **(
                {"refused": self.refused, "refused_reason": self.refused_reason}
                if self.outcome == REFUSED
                else {}
            ),
            "n_rebalances": self.n_rebalances,
            "weekly_rebalance_turnover": self.weekly_rebalance_turnover,
            "clears_deployment_hurdle": self.clears_deployment_hurdle,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class GridRecord:
    """A whole Grid, run once: every cell, in the grid's own order.

    The order is the published one and is never sorted by outcome. A grid
    ordered best-first is a ranking, and reading a ranking is the mistake the
    Grid exists to prevent.
    """

    commit: str
    working_tree_dirty: bool
    run_at_utc: str
    grid: str
    config: RunConfig
    config_sha256: str
    config_path: str
    cells: tuple[GridCellRecord, ...]
    configurations_tried: int = 0
    trials_recorded: int = 0
    # The Universe all 21 cells ran on, after policy. A property of the grid and
    # not of a cell: the cells differ in two knobs and see one Universe, which is
    # what makes them comparable with each other. It carries `bracket_bounds`,
    # both ends of the bracket, because a Universe quoted on one bound alone has
    # chosen which listing risk to show.
    universe: dict[str, Any] = field(default_factory=dict)

    @property
    def n_recorded(self) -> int:
        return sum(1 for cell in self.cells if cell.recorded)

    @property
    def n_refused(self) -> int:
        return sum(1 for cell in self.cells if cell.outcome == REFUSED)

    @property
    def n_liquidated(self) -> int:
        """Cells wiped out mid-holding-period. ADR-0001's headline count.

        Han, Kang and Ryu liquidate five of their 21, and the Replication Gate
        is read against that number, so it is counted here rather than derived
        by whoever opens the file.
        """
        return sum(1 for cell in self.cells if cell.liquidated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "run_at_utc": self.run_at_utc,
            "grid": self.grid,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "config": {**asdict(self.config), "cost_model": self.config.cost_model.name},
            "n_cells": len(self.cells),
            "n_recorded": self.n_recorded,
            "n_refused": self.n_refused,
            "n_liquidated": self.n_liquidated,
            "configurations_tried": self.configurations_tried,
            "trials_recorded": self.trials_recorded,
            "universe": self.universe,
            "cells": [cell.to_dict() for cell in self.cells],
        }


GATE_FILENAME = "gate.json"


@dataclass(frozen=True)
class GateRecord:
    """The Replication Gate, run once: two Grids, two verdicts, and the gap.

    The verdicts and the gap arrive already serialised, from `gate.py`. That
    module decides what a verdict *is* and this one decides where it lands on
    disk; typing the record against `GateVerdict` would put an import cycle
    between them for no gain, since nothing here reads inside them.

    `passes` is a conjunction of the two runs. ADR-0003 makes the Faithful Run
    the gate proper — if it fails, nothing downstream means anything — but the
    Venue Run has its own three binding criteria and a gate that reported "pass"
    while one of them failed would be reporting the half it liked.
    """

    commit: str
    working_tree_dirty: bool
    run_at_utc: str
    leg: str
    faithful: dict[str, Any]
    venue: dict[str, Any]
    gap: dict[str, Any]
    faithful_config_name: str
    venue_config_name: str
    universe_bracket: dict[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        return bool(self.faithful.get("passes")) and bool(self.venue.get("passes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "run_at_utc": self.run_at_utc,
            "leg": self.leg,
            "passes": self.passes,
            "faithful_config_name": self.faithful_config_name,
            "venue_config_name": self.venue_config_name,
            "universe_bracket": self.universe_bracket,
            "faithful": self.faithful,
            "venue": self.venue,
            "gap": self.gap,
        }


class ResultStore:
    """Backtest output on disk, keyed by commit hash and config name.

    A Grid's cells are keyed under the grid as well, so the 21 files that are
    read together sit together and nothing outside the grid is mixed in with
    them.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def directory_for(self, commit: str, *, working_tree_dirty: bool, grid: str = "") -> Path:
        # A dirty tree means the commit hash does not identify the code that ran,
        # so its output is filed apart from the commit's own results.
        directory = self.root / (f"{commit}-dirty" if working_tree_dirty else commit)
        return directory / grid if grid else directory

    def path_for(
        self,
        commit: str,
        config_name: str,
        *,
        working_tree_dirty: bool,
        grid: str = "",
    ) -> Path:
        directory = self.directory_for(
            commit, working_tree_dirty=working_tree_dirty, grid=grid
        )
        return directory / f"{config_name}.json"

    def write(self, record: RunRecord) -> Path:
        path = self.path_for(
            record.commit,
            record.config.name,
            working_tree_dirty=record.working_tree_dirty,
            grid=record.grid_config_name,
        )
        return _written(path, record.to_dict())

    def write_grid(self, record: GridRecord) -> Path:
        """The grid's summary, beside the cells it summarises."""
        path = (
            self.directory_for(
                record.commit,
                working_tree_dirty=record.working_tree_dirty,
                grid=record.config.name,
            )
            / GRID_FILENAME
        )
        return _written(path, record.to_dict())

    def write_gate(self, record: GateRecord) -> Path:
        """The gate's verdict, at the commit's own root.

        Beside the two grid directories it was read from rather than inside
        either of them: it is a statement about the pair, and filing it under one
        run would make it look like a property of that run.
        """
        path = (
            self.directory_for(
                record.commit, working_tree_dirty=record.working_tree_dirty
            )
            / GATE_FILENAME
        )
        return _written(path, record.to_dict())


def _written(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
