# Quant Research Protocol

How to run and report a backtest in this repo. The invariants it serves live in `CLAUDE.md`; this file is the procedure.

## Data

Three layers, each a distinct path:

- **`data/raw/`** — exactly as fetched from the venue or vendor, append-only. Re-fetching an existing window is a bug report, not an overwrite.
- **`data/derived/`** — resampled bars, returns, features. Rebuilt from raw by a script, so it is disposable.
- **`results/`** — backtest output, keyed by commit hash and config file name.

All three are gitignored: they are large and regenerable. Git tracks the fetch and build config that produces them.

Record for each raw source: venue, symbol convention, bar close convention, timezone, and the exact window fetched. Crypto venues differ on all five, and a mismatch shows up as an unexplained edge.

## Building a signal

1. State the economic hypothesis in one sentence before writing the code. "12-week returns rank ahead of 1-week reversals in majors" is a hypothesis; "momentum works" is not.
2. Write the feature as a pure function of a point-in-time frame, so it is testable on a hand-built fixture with a known answer.
3. Test the lookahead boundary explicitly: feed a frame whose last row is the decision bar and assert the feature ignores it.
4. Handle the survivorship edge: an asset that stops trading mid-holding-period exits at its last tradeable price, and that path is exercised by a test.

## Running a backtest

Every run takes a config file and emits a result keyed by `(commit, config)`. Vary one thing per run.

Costs enter the simulation, never a post-hoc haircut:

- **Taker fee** per side at the venue's actual tier.
- **Funding** on any perp position, at the realized rate for that window.
- **Slippage** as an explicit model, sized against the asset's depth at the rebalance hour. Small-cap crypto legs are where a paper edge dies.

## Reporting a result

Report these together — Sharpe alone hides the tail and the turnover that pays for it:

- Annualized return and volatility, net
- Sharpe, net
- Max drawdown and its date range
- Turnover per rebalance
- Average gross and net exposure
- Number of positions held
- The count of configurations tried to reach this one

When a result looks strong, state the most likely way it is wrong before defending it. In crypto momentum the usual suspects are: a stablecoin or wrapped asset inflating the cross-section, a venue outage producing stale bars that read as low volatility, and a single 2021-scale move carrying the whole series.

## Recording a decision

A parameter choice that survives out-of-sample is an architectural decision. Write it as an ADR under `docs/adr/` with the alternatives that lost and the evidence that separated them, so a later run does not silently re-litigate it.
