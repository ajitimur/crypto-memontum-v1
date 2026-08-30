# ADR-0007: Trade on Tokocrypto, and treat turnover as the binding constraint

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** [ADR-0008](0008-data-sources.md), [ADR-0003](0003-replicate-before-innovate.md), `CONTEXT.md` (Cost Drag, Rebalance Turnover)

## Context

Three facts, all verified against primary sources, reshape this project more than any research finding did.

**Offshore venues are blocked at the carrier.** `binance.com`, `bybit.com` and `okx.com` resolve from this network to 202.3.218.137, which whois identifies as `TELKOMSEL-ID` — Telkomsel's own block-page server, serving a certificate for `internetbaik.telkomsel.com` that expired in September 2023 and redirecting to a block page. This is the Internet Positif filtering regime, deliberate and selective, not a fault.

**Indonesian tax applies per leg.** PMK 50/2025, effective 2025-08-01, sets a 0.21% final PPh on crypto disposals on a licensed venue — and Pasal 11(2)(b) makes a crypto-to-crypto swap a *penyerahan* on **both** sides. A USDT-quoted rebalance is taxed twice. Offshore trading does not avoid this; it raises the rate to 1% and shifts remittance onto the taxpayer.

**Total costs are roughly 2.7× what the literature assumes.** Tokocrypto: 0.15% fee + 0.21% PPh + 0.0444% exchange levy = **0.4044% per side, charged on buys as well as sells**, so 0.8088% per round trip against the 0.30% Han et al. assume. At the ~68% weekly turnover the literature reports for these signals, that is **28.6% of annual Cost Drag** versus their 10.6%.

## Decision

**Venue: Tokocrypto** (licensed PAKD, S-14/D.07/2025).

**Turnover is budgeted, not observed.** A hard ceiling of **25% weekly Rebalance Turnover**, enforced by the config loader, which rejects a run rather than executing it. Plus an ex-post reporting rule: **Cost Drag must not exceed one third of gross annualised return.**

**The universe is reported as a bracket.** Tokocrypto lists 419 USDT pairs today and publishes no history of what it listed previously. Backtesting the full Binance universe credits us with unavailable trades; restricting to today's list applies a present-day roster backwards. So we report both — full Binance universe as the upper bound, today's Tokocrypto list as the lower — and treat the gap as the measurement of venue-listing risk rather than picking the flattering end.

## Considered Options

**Indodax.** Cheaper in the way that matters — IDR-quoted, so only the sell leg is taxed, giving ~0.65% round trip against Tokocrypto's 0.81%. Rejected because there is no survivorship-free archive for its IDR book, which would reintroduce exactly the bias ADR-0008 solves. Paying 16bp more per round trip to keep a clean universe is the right trade.

**Bybit Indonesia.** 363 USDT pairs, full V5 API, measured reachable from this network without a VPN. Its fee schedule could not be located during the original research, and this ADR was published with that as an open item. It has since been closed — see the amendment below. Rejected, but for different reasons than cost.

**Offshore via VPN.** Rejected on practicalities rather than on law: the tax rises rather than disappears, a live strategy would depend on a VPN holding 24/7 for both data and execution, and the account's registered residence would conflict with its access IP. Whether retail users incur liability under Pasal 304 UU P2SK could not be established from any primary source, and that uncertainty is itself a reason not to build on it.

## Amendment, 2026-08-30: Bybit Indonesia measured, decision unchanged

The open item is closed. Bybit Indonesia's spot VIP 0 is reported at 0.10% maker and taker. Assuming the levy stacks as it does elsewhere — Bybit Indonesia is a CFX member, and CFX's USDT levy is 0.0222% per Indodax's published post-March-2026 schedule:

| | Fee | PPh | Levy | Per side | Round trip | Cost Drag @ 68% turnover |
|---|---|---|---|---|---|---|
| Tokocrypto | 0.15% | 0.21% | 0.0444% | 0.4044% | 0.8088% | 28.6% |
| Bybit Indonesia | 0.10% | 0.21% | 0.0222% | 0.3322% | 0.6644% | 23.5% |

Bybit is genuinely cheaper, by 0.144% per round trip, which would lift the Rebalance Turnover ceiling from 25% to roughly 29%. **The decision stands anyway**, on two grounds.

**The saving does not change the regime.** 23.5% and 28.6% are both catastrophic against the literature's 10.6%. Switching venues moves us from "the tax probably kills this" to "the tax probably kills this." Turnover remains the binding constraint either way, so this margin does not decide anything.

**Bybit is worse where it actually matters.** The reason for Tokocrypto is that its USDT book *is* Binance's book, so the archive in ADR-0008 is the venue's own price history and backtest and execution share one source. Bybit's book is its own, and its klines begin at each symbol's listing date — BTCUSDT starts 2021-07-05. Against Han et al.'s 2017-01 to 2023-08 window that covers roughly two years of a seven-year replication. Backtesting Bybit on Bybit data guts the Replication Gate; backtesting it on Binance data prices a venue we do not trade. Either way the single-source property is gone.

Two caveats on the numbers, recorded so a later reader does not over-trust them. The fee figures match Bybit's *global* VIP 0 schedule exactly, which is suggestive but not confirmation that `bybit.id` matches; our own research could not reach a Bybit Indonesia fee page. And the 0.0222% levy is inferred from Indodax's published CFX rate, not published by Bybit Indonesia. Neither caveat changes the conclusion: at 0.10% flat with no levy at all, per side would be 0.31% and Cost Drag 21.9% — the same regime.

## Consequences

- **Turnover, not Sharpe, is now the primary design constraint.** The strategy worth shipping is the one with the best return per unit of turnover. This should shape how we search the grid: the longer holding periods in Han et al.'s table, which look mediocre on Sharpe, are the ones most likely to survive here.
- A 25% ceiling against the literature's 68% means the published configurations are mostly untradeable for us as published. Expect to need a no-trade buffer band — hold a name until its rank leaves a wider threshold than the one that bought it — rather than to find a low-turnover config by luck.
- Every number this project reports is net of a 0.4044%-per-side cost model with the tax included. Quoting a figure against the literature's 15bp is comparing to a world we do not trade in.
