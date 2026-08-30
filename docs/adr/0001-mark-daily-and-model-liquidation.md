# ADR-0001: Mark portfolios daily and model liquidation explicitly

- **Status:** Accepted
- **Date:** 2026-08-30
- **Supersedes:** nothing
- **Related:** [ADR-0002](0002-log-return-profitability-bar.md), [ADR-0003](0003-replicate-before-innovate.md), `docs/research/crypto-momentum-strategies.md` §2.5

## Context

Our strategies rebalance weekly. The obvious simulator evaluates them on weekly bars: form the portfolio Monday, take the return to the following Monday, repeat. Every crypto momentum paper we surveyed does this, and it is wrong in a way that flatters the result.

A weekly bar hides what happened inside the week. Han, Kang and Ryu simulate daily instead and find that **five of their twenty-one cross-sectional momentum portfolios are liquidated** during 2014–2023 — wiped out mid-holding-period by moves that the weekly bar smooths over entirely. Their example: a single day's move in UNFI on 2021-07-07 could have taken the whole portfolio, and the week's return shows nothing unusual. Their conclusion is that weekly-bar evaluation "overestimates the Sharpe ratio and misses liquidation events."

This is not a rounding error. A liquidated portfolio has a terminal loss and no forward returns, so a simulator that cannot represent liquidation is not computing a slightly optimistic Sharpe — it is computing the Sharpe of a strategy that does not exist.

The exposure is worst on the short leg. Every short-only portfolio in Han et al.'s top-5% table sits at −99% or −100% cumulative, blown up in both the 2017 and 2020 bull markets. Our current design uses a BTC short leg rather than a loser basket, which reduces but does not remove this.

## Decision

The backtester marks every position **daily** through the holding period, not only at rebalance boundaries.

1. Daily marking is the simulator's native granularity. Weekly rebalance is a decision-frequency choice layered on top of it, never a substitute for daily accounting.
2. When cumulative portfolio loss breaches 100%, the portfolio is **liquidated**: the loss is terminal, all positions close, and the series ends. A liquidated run is reported as liquidated with its date, never as a return series that continues past the breach.
3. Liquidation count and dates join the standard reporting block (see the `docs/agents/quant-research.md` amendment).
4. The liquidation path is exercised by a test, on a hand-built fixture whose intra-period path breaches 100% while its period-boundary return does not. A simulator that passes on boundary returns alone has not been tested.

Han et al. note that even daily marking underestimates liquidation risk, since intraday moves are larger still. We accept daily as the floor, not as sufficient, and record that a future ADR may tighten it.

## Alternatives considered

**Weekly-bar evaluation, as the literature does it.** Rejected. It is the specific defect that makes the published Sharpes in this area untrustworthy, and adopting it would mean our numbers inherit the flaw we set out to avoid.

**Weekly bars plus a post-hoc drawdown haircut.** Rejected for the same reason `docs/agents/quant-research.md` rejects post-hoc cost haircuts: a liquidation is a path-dependent terminal event, and no scalar adjustment applied afterward reconstructs the path.

**Intraday marking from the outset.** Rejected for now on cost. It multiplies the data-fetch burden by roughly two orders of magnitude to sharpen an estimate we have not yet shown we need. Revisit if daily marking puts any candidate strategy near the liquidation boundary.

## Amendment, 2026-08-30

[ADR-0004](0004-long-only-spot-no-short-leg.md) makes v1 long-only unlevered spot, under which a portfolio cannot breach a 100% loss unless every holding simultaneously goes to zero. The liquidation trigger is therefore **inert for v1**. It is retained deliberately: it becomes live the moment leverage or a short leg appears, and the daily-marking requirement stands regardless, since drawdown and path risk do not care whether liquidation is reachable.

## Consequences

- The simulator needs daily bars for every asset in the universe across every holding period, not just bars at rebalance dates. This raises the fetch requirement and is the binding constraint on the data layer.
- Some strategies that look viable in the literature will terminate in our backtest. That is the ADR working, not a bug.
- Reported Sharpes will come in below published figures for the same nominal strategy. Comparisons to published numbers must state that ours are daily-marked and theirs are not.
