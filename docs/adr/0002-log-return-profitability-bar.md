# ADR-0002: Judge profitability on mean log return at t > 3.0

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** [ADR-0001](0001-mark-daily-and-model-liquidation.md), [ADR-0003](0003-replicate-before-innovate.md), `docs/research/crypto-momentum-strategies.md` §2.5

## Context

The standard test in this literature is a t-test on the mean return. It does not survive contact with crypto's return distribution.

When returns are fat-tailed and skewed, the linear approximation that makes arithmetic mean return a proxy for compounded wealth breaks down, and **a strategy can post a statistically significant positive mean return while actually losing money**. This is not hypothetical. In Han, Kang and Ryu's cross-sectional results, six portfolios have a positive mean return and a negative mean *log* return. Ten portfolios clear t > 2.0 on mean return; only three clear it on mean log return.

Sharpe has the same defect from the same cause. Mean and standard deviation are incomplete descriptions of a skewed, fat-tailed distribution, so a high Sharpe on crypto momentum is weak evidence of a strategy that compounds.

Separately, this research area is heavily mined. Fieberg et al. enumerate 55,296 research-design permutations of cross-sectional crypto momentum and find the median gross Sharpe is 0.83 and that results are statistically significant in only 49% of designs — roughly a coin flip on whether any given design "works." Under that much multiple testing, a t > 2.0 threshold selects noise. Harvey, Liu and Zhu (2016) propose t > 3.0 for exactly this situation.

Applying both corrections at once to the best-documented paper in the area: **no** cross-sectional momentum portfolio in Han et al. clears t > 3.0 on mean log return.

## Decision

A strategy is judged profitable in this repo on the **t-statistic of its mean log return, with the bar at t > 3.0**, Newey-West adjusted.

1. Mean log return and its t-statistic join the standard reporting block. Mean return and Sharpe continue to be reported — they are how we compare to published work — but they do not decide anything.
2. Any result quoted as a finding states its log-return t-statistic. A result quoted on Sharpe alone is incomplete.
3. Where mean return and mean log return disagree in sign, the log return governs and the divergence is called out explicitly, since it is diagnostic of exactly the tail behaviour that kills these strategies.
4. This bar applies to development-sample and out-of-sample results alike.

We are setting a threshold we may well fail. That is the point of setting it now, before we have a number we are attached to.

## Alternatives considered

**t > 2.0 on mean log return.** Rejected. It is the conventional bar, but conventional bars assume one hypothesis, and this area has thousands. Fieberg et al.'s 49%-significant finding is direct evidence that t > 2.0 does not discriminate here.

**t > 3.0 on mean arithmetic return.** Rejected. Raising the bar without fixing the statistic leaves the compounding defect in place; a fat-tailed series can clear it and still lose money.

**Deflated Sharpe ratio (Bailey & López de Prado).** A reasonable alternative that corrects for multiple testing directly. Rejected for now because it requires an explicit trial count, and ours is only well-defined once we begin the config log that `docs/agents/quant-research.md` mandates. Worth revisiting as a complement, not a replacement — it addresses selection, not the log-return problem.

## Consequences

- Some strategies that look publishable by the literature's own standard will fail our bar. Expect this to include most of them.
- If our first candidate fails, the correct response is to record the null, not to tune lookbacks until something clears. That reflex is the failure mode this ADR exists to make expensive.
- Comparisons to published numbers must note that ours are held to a different and stricter test.
