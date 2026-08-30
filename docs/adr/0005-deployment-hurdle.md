# ADR-0005: What this strategy must beat before it gets money

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** [ADR-0002](0002-log-return-profitability-bar.md), [ADR-0004](0004-long-only-spot-no-short-leg.md)

## Context

ADR-0004 makes v1 a long-only directional bet on crypto. That raises an obvious question a Sharpe ratio does not answer: why not just hold Bitcoin? Holding BTC requires no research, no rebalancing, no turnover, and — under the tax structure in ADR-0007 — almost no cost. It is the honest alternative use of the money, so it is the hurdle.

## Decision

Three conditions, **all required** before any capital is deployed:

1. Net Sharpe above BTC buy-and-hold over the same window.
2. Mean log return with t > 3.0, per ADR-0002.
3. **Maximum drawdown no worse than BTC buy-and-hold's** over the same window.

The cap-weighted market portfolio is logged as a secondary reference for comparability with the literature, but BTC is what decides.

## Consequences

The third condition is the one that will bind, and it is deliberately strict. A strategy that edges BTC on Sharpe while drawing down harder is not obviously better for an account someone has to live with, and Han et al.'s ~88% drawdowns sit on the wrong side of it. Expect this condition, not the Sharpe, to be what fails.
