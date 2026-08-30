# ADR-0004: v1 is long-only spot, with no short leg

- **Status:** Accepted
- **Date:** 2026-08-30
- **Amends:** [ADR-0001](0001-mark-daily-and-model-liquidation.md)
- **Related:** [ADR-0005](0005-deployment-hurdle.md), [ADR-0007](0007-venue-and-cost-structure.md), `docs/research/crypto-momentum-strategies.md` §2.5 and §6

## Context

`docs/research/crypto-momentum-strategies.md` §6 recommends a BTC short leg against the momentum basket, to get a roughly market-neutral portfolio. Reading Han, Kang and Ryu in full undermines that.

Every short-only portfolio in their table 15b ends at −99% or −100% cumulative, wiped out in both the 2017 and 2020 bull markets — and they note this is "not confined to jumps of small coins," so it is not a microcap problem that a large-cap filter would fix. Their best long-only portfolio (net Sharpe 1.54) beats their best long-short (1.40). Their overall verdict is that a momentum long-short "that can generate steady, market-neutral profits appears unattainable."

A BTC short is not their loser-basket short and avoids that specific failure. But it requires margin or perpetual futures, which brings funding costs, genuine liquidation risk, and — on a EUR-equivalent USD 5,000–10,000 account — a materially worse risk profile than the position size justifies.

## Decision

Version one is **long-only spot**. No margin, no perpetuals, no short leg.

Risk control is the **Trend Gate** (`CONTEXT.md`), tested as a variant *after* the replication runs rather than folded into them. Its motivation is Han et al.'s strongest single result: time-series momentum long-only at net Sharpe 1.51, ahead of every cross-sectional variant they test. Applying it as an overlay on cross-sectional selection is the synthesis of their two findings rather than a new idea.

The short leg becomes a v2 variant to be tested, not the default.

## Consequences

- **ADR-0001's liquidation trigger becomes inert.** An unlevered long-only portfolio cannot breach a 100% loss unless every holding simultaneously goes to zero. The machinery is retained anyway: it becomes live the moment leverage or a short leg appears, and retrofitting accounting into working code is worse than carrying it unused. The fixture test stays too.
- **This is a directional bet on crypto, not an alpha strategy.** Han et al.'s long-only portfolios carry maximum drawdowns around 88%. Anyone reading a good Sharpe from this strategy should read that number next to it. ADR-0005 exists because of this.
