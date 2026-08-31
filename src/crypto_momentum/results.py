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
    # What the run is read against: BTC buy-and-hold, the cap-weighted market
    # portfolio, and ADR-0005's three-condition hurdle over the same window.
    benchmarks: dict[str, Any] = field(default_factory=dict)
    # How many configurations had been tried, this one included, when it ran.
    # The protocol requires it beside every quoted number, so it travels with
    # the number rather than living only in a log someone has to go and count.
    configurations_tried: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "run_at_utc": self.run_at_utc,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "config": asdict(self.config),
            "window": self.window,
            "portfolio": self.portfolio,
            "benchmarks": self.benchmarks,
            "configurations_tried": self.configurations_tried,
            "metrics": self.metrics,
        }

    def trial_line(self) -> dict[str, Any]:
        """The flat one-line summary appended to the trials log.

        Carries the result key and the headline numbers, so the log answers
        "how many configurations were tried, and what did they give" without
        opening every result file.
        """
        return {
            "run_at_utc": self.run_at_utc,
            "commit": self.commit,
            "working_tree_dirty": self.working_tree_dirty,
            "config_name": self.config.name,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "symbol": self.config.symbol,
            "n_symbols": len(self.config.universe_symbols),
            "interval": self.config.interval,
            "strategy_kind": self.config.strategy_kind,
            "start_month": self.config.start_month,
            "end_month": self.config.end_month,
            "cost_bps_per_side": self.config.cost_bps_per_side,
            "configurations_tried": self.configurations_tried,
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
                    "mean_net_exposure",
                )
                if key in self.portfolio
            },
            **self.metrics,
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
