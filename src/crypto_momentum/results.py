"""`results/` — one JSON file per (commit, config).

The key is the pair, because either half alone is ambiguous: the same config at
two commits is two results, and two configs at one commit are two results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crypto_momentum.config import RunConfig


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "run_at_utc": self.run_at_utc,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
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
) -> dict[str, Any]:
    """Which run this line is about — the half every trials line shares.

    One definition, because there are two kinds of line: a run that produced a
    result, and a run refused for breaching its turnover budget. Both have to be
    countable and identifiable in the same way, and building the keys twice is
    how the two quietly drift into two schemas.
    """
    return {
        "run_at_utc": run_at_utc,
        "commit": commit,
        "working_tree_dirty": working_tree_dirty,
        "config_name": config.name,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "symbol": config.symbol,
        "n_symbols": len(config.universe_symbols),
        "interval": config.interval,
        "strategy_kind": config.strategy_kind,
        "start_month": config.start_month,
        "end_month": config.end_month,
        "cost_model": config.cost_model.name,
        "cost_bps_per_side": config.cost_bps_per_side,
    }


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
    realised_weekly_turnover: float,
    budget: float,
    reason: str,
) -> dict[str, Any]:
    """The trials line for a configuration that was tried and then refused.

    Deliberately the same identity as a successful trial, so the log can be
    counted without filtering, and deliberately carrying no metrics, because none
    were produced. `refused` is what tells the two apart, and it names the number
    that did it.
    """
    return {
        **trial_identity(
            config,
            commit=commit,
            working_tree_dirty=working_tree_dirty,
            run_at_utc=run_at_utc,
            config_path=config_path,
            config_sha256=config_sha256,
        ),
        # A refused configuration is still one of the configurations tried, so it
        # carries the same counts a recorded run does — otherwise the search
        # behind a later result would silently omit the ones the ceiling stopped.
        "configurations_tried": configurations_tried,
        "trials_recorded": trials_recorded,
        "refused": "turnover_budget_breached",
        "weekly_rebalance_turnover": realised_weekly_turnover,
        "turnover_budget_weekly": budget,
        "refused_reason": reason,
    }


class ResultStore:
    """Backtest output on disk, keyed by commit hash and config name."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path_for(self, commit: str, config_name: str, *, working_tree_dirty: bool) -> Path:
        # A dirty tree means the commit hash does not identify the code that ran,
        # so its output is filed apart from the commit's own results.
        directory = f"{commit}-dirty" if working_tree_dirty else commit
        return self.root / directory / f"{config_name}.json"

    def write(self, record: RunRecord) -> Path:
        path = self.path_for(
            record.commit,
            record.config.name,
            working_tree_dirty=record.working_tree_dirty,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
        return path
