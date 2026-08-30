# Coding Standards

Loaded by the reviewer agent during code review. The research invariants in `CLAUDE.md` are review criteria; this file names what they look like in a diff.

Python, pandas-shaped.

## Research correctness

Check these before style — a clean function that leaks the future is worse than a messy one that doesn't.

- **Trace the index arithmetic.** Every new `.shift`, `.rolling`, `.resample`, `.ewm`, or `merge_asof` is a place a bar can slip across the decision boundary. Confirm the feature reads only bars that closed before the bar it trades on.
- **Treat an explicit lag as load-bearing.** A `.shift(1)`, a `closed=` or `label=` argument, an `.iloc[:-1]` slice: these encode point-in-time meaning and often look like removable noise. Keep them. Change one only alongside a test that pins the boundary.
- **Costs live in the simulation.** A path that produces a return applies fees, funding, and slippage as it trades. A gross number haircut afterwards is a finding.
- **Universe resolves as of the rebalance timestamp**, not from a today-shaped list of symbols. An asset that stops trading mid-hold exits at its last tradeable price.
- **Results are reproducible.** Stochastic code takes an explicit seed. A wall-clock read (`datetime.now()`, `date.today()`) inside a path that produces a result makes the run unrepeatable — take the timestamp as a parameter instead.

## Style

- Annotate function signatures. For a frame, say in the docstring what one row is and what the index carries.
- Name for the domain concept, not the pipeline position: `ranked_returns` over `df2`. Where `CONTEXT.md` defines a term, use that term.
- Put units and conventions in the name: `fee_bps`, `funding_rate_8h`, `ts_utc`, `lookback_bars`. A bare `price`, `rate`, or `window` invites a unit mismatch that survives every test.
- Keep frame transforms pure — take a frame, return a new one. In-place mutation of a caller's frame is a finding.
- Comments carry the why: the venue quirk, the convention, the reason a lag is where it is.

## Testing

- pytest. Every feature function has a test against a hand-built fixture whose expected value was worked out by hand, rather than a snapshot of what the code currently returns.
- Every feature has a lookahead-boundary test: a frame whose last row is the decision bar, asserting the feature ignores it.
- Cover the delisting path — an asset that disappears mid-holding-period.
- Assert the portfolio invariants that are cheap to state: weights sum to the target gross, a long-only book holds no negative weight, position count matches the config.
- Compare floats with an explicit tolerance.

## Architecture

- Keep four stages separable: data access, feature computation, portfolio construction, simulation. A feature function that reaches for data cannot be tested on a fixture, so pass frames in.
- Parameters live in config files, not as literals spread through the code. A number that changes a result is config.
- Notebooks explore; logic that produces a committed result lives in a module with tests, imported by the notebook.
