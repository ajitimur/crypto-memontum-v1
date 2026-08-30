# ADR-0003: Replicate a published net result before testing our own signal

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** [ADR-0001](0001-mark-daily-and-model-liquidation.md), [ADR-0002](0002-log-return-profitability-bar.md), `docs/research/crypto-momentum-strategies.md` §2.5 and §6

## Context

Our survey (`docs/research/crypto-momentum-strategies.md`) produced four candidate strategies. Every one of them reports its headline Sharpe on a sample ending in or before 2023: Begušić and Kostanjčar stop in 2018, Fieberg et al. in May 2022, Han et al. in August 2023. Han et al. further report that their top-5% portfolios "perform steadily until early 2021, but they then move sideways generating almost no profits."

So the live question is not which of the four to implement. It is whether the effect still exists — and, before even that, whether our pipeline is capable of measuring it.

We have one strategy in the survey with a **net** result published alongside enough methodology to check ourselves against: Han et al.'s (14, 7) all-coins cross-sectional long-short, net Sharpe 1.28 at a 15bp cost built from Binance's real fee schedule, a 3.26bp average relative tick size, and per-coin slippage measured from trading data. That is a target we can aim a pipeline at and know whether we hit.

There is also a contradiction in the survey worth resolving cheaply. Han et al. find the long leg "worsens monotonically with volume," with the lowest-volume group earning a slightly higher Sharpe than the highest. Begušić and Kostanjčar's liquid-winners result has the opposite sign on overlapping data. Candidate §2.4 depends on which is right.

## Decision

Work proceeds in three ordered steps. Each gates the next.

**Step 1 — Replication gate.** Build the data layer, point-in-time universe, and daily-marked simulator (ADR-0001) against Han et al.'s (14, 7) all-coins long-short, and attempt to reproduce net Sharpe 1.28 over 2017-01-01 to 2023-08-28 at 15bp. This is a pipeline-validation exercise, not a strategy we intend to trade. If we cannot approximately reproduce a published net figure on its published sample, our pipeline is wrong and every subsequent number is uninterpretable.

Two sorts ride along on the same pipeline at negligible cost: a volume-decile sort on the long leg, which resolves the §2.4 contradiction, and a time-series momentum benchmark.

**Step 2 — Regime split.** Re-run the frozen Step 1 config split at 2021-01-01: 2017–2020 against 2021–2023. This asks whether the effect decayed, and it asks it **entirely within the development sample**, so it costs us nothing from the holdout.

**Step 3 — Fork on the Step 2 answer.**
- If the post-2021 regime clears ADR-0002's bar, proceed to the RSI-14 config in §6, then CTREND, and only then open the 2024-01-01 → 2026-08-30 holdout.
- If it does not, stop testing variants of past-return cross-sectional momentum. Tuning lookbacks after a null is the failure `docs/agents/quant-research.md` names. Take up time-series momentum, which Han et al. measure at net Sharpe 1.51 against the best cross-sectional 1.28, or record the null and stop.

The out-of-sample window stays untouched through Steps 1 and 2. It is not opened until a config is frozen and the trial count is logged.

## Amendment, 2026-08-30: the gate runs twice, with fixed tolerances

Step 1 splits into two runs, because it was conflating two different tests.

**Faithful Run** — CoinMarketCap prices and caps (ADR-0008), their exact 2017-01-01 to 2023-08-28 window. Tests whether *our pipeline is correct*, with vendor differences eliminated as an excuse. This is the gate proper: if it fails, nothing downstream means anything.

**Venue Run** — same config, Binance archive prices, 2017-08-17 floor. Tests whether the published effect survives contact with executable prices.

**The gap between them is a result, not an error.** It measures how much of the published effect is an artifact of cross-exchange aggregate pricing versus what could actually have been traded on one venue. Our survey found nobody reporting that number, and both runs existing makes it nearly free.

Pass criteria, fixed now so that no run produces a number to rationalise against:

| | Faithful Run | Venue Run |
|---|---|---|
| Spearman rank correlation across the 21 cells | ≥ 0.70 | ≥ 0.70 |
| Liquidation count vs their five | within ±2 | within ±2 |
| Sign agreement on log-return t-statistics | ≥ 18 of 21 | ≥ 18 of 21 |
| Best cell's net Sharpe vs 1.28 | within ±0.15 | **not required** |

Level agreement is required only on the Faithful Run, where the same vendor and window make it meaningful. On the Venue Run different prices and a truncated window make a level match meaningless, and demanding one would just invite fitting.

0.70 and ±0.15 are judgement calls, not derived quantities. What makes them useful is that they are fixed in advance; rank correlation of 0.70 across 21 cells is significant past p < 0.001, so it is a real bar rather than a formality.

## Alternatives considered

**Implement the RSI-14 config first**, as `docs/research/crypto-momentum-strategies.md` §6 originally recommended. Rejected as the *first* step, not on its merits. Its published figure is gross-only, from a single table, so a result from it is unfalsifiable — we would not know whether a disappointing number meant a dead signal or a broken simulator. It remains the intended Step 3 strategy.

**Go straight at the 2024+ holdout to answer the decay question.** Rejected. It spends the one asset we cannot replace, to answer a question that 2021–2023 development data already answers.

**Skip replication and trust the pipeline.** Rejected. Three of our four candidates disagree with each other on basic questions such as the sign of the volume effect. In a field where careful groups reach opposite conclusions, an unvalidated pipeline is not evidence.

## Consequences

- First deliverable is a strategy we do not intend to trade. Accepted deliberately: it is the cheapest way to find out whether we can measure anything.
- "Approximately reproduce" needs a tolerance, and Han et al. use CoinMarketCap aggregate data while we plan to use Binance venue data, so exact agreement is not expected. The tolerance is set and recorded before the run, not after seeing the result.
- Step 2 may end the project's central premise early. That is a good outcome delivered cheaply, and it should be written up as a null rather than quietly reopened.
