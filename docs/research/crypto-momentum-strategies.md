# Crypto momentum strategies: candidates, evidence, and what survives costs

Research note, written 2026-08-30. Every quantitative claim carries an inline URL. Where I could not open a primary source, I say so and quote nothing from it.

## 1. Summary

Four candidates, ranked by how much of each reported edge I could verify from a primary source net of a stated cost assumption, and by how cleanly each maps onto this repo's invariants.

1. **Cross-sectional rank on a bounded momentum oscillator, 14-day RSI, value-weighted quintiles, weekly rebalance.** This is the only candidate where a peer-reviewed paper reports a simple non-ML weekly cross-sectional signal earning 3.52% per week long-short, t = 5.41, over Apr 2015 to May 2022 ([Fieberg, Liedtke, Poddig, Walker and Zaremba, *JFQA* 2024, Table 2](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). That figure is **gross**. The same paper's cost analysis for its aggregate signal shows roughly 75% of the gross spread surviving 30/40bp costs at 68% weekly turnover.
2. **CTREND, a machine-learning aggregate of 28 technical indicators, cross-sectionally ranked, value-weighted quintiles, weekly.** Annualized **gross** Sharpe **1.94**. Long-short 3.87% per week gross becomes **2.90% per week net** at 30bp long and 40bp short, t = 3.89, on 68.5% weekly turnover, and 2.45% per week net when restricted to the largest 100 coins ([same paper, Tables 3 and 9](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Best-documented net result in this literature. Also the costliest to build and the most exposed to overfitting.
3. **LTW CMOM, plain 2-to-3-week past return, value-weighted quintiles, weekly rebalance.** The canonical crypto cross-sectional momentum factor ([Liu, Tsyvinski and Wu, *Journal of Finance* 2022](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119); [NBER WP 25882](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). The honest Sharpe estimate is a **median annualized 0.83 gross** across 55,296 research-design permutations, max 2.30, min -4.47, statistically significant in only 49% of designs ([Fieberg et al. 2024, section VI.A](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). This is the baseline to beat, not the strategy to ship.
4. **Long-only "liquid winners", a 14-day momentum by Amihud-liquidity double sort, top 30% momentum intersected with top 30% liquidity, equal-weighted, biweekly.** Reported **information ratio 1.59 gross, 1.43 at 50bp, 1.27 at 100bp** ([Begušić and Kostanjčar, arXiv 1904.00890, Table 2](https://arxiv.org/pdf/1904.00890)). Long-only, so no borrow problem, and it is cost-tested. But the sample ends in 2018, the metric is an information ratio against a cap-weighted crypto benchmark rather than a Sharpe, and the paper is a short unrefereed note.

The uncomfortable headline. The plain-vanilla version of the thing this repo is named after, cross-sectional momentum on past returns, has a gross median Sharpe below 1.0, and I found no primary source reporting a **net** Sharpe above 1.0 for it. Everything that clears 1.0 adds something: a bounded oscillator instead of raw returns, an ML aggregate, or a liquidity interaction.

**Update, 2026-08-30.** Han, Kang and Ryu (SSRN 4675565) has since been obtained and read in full, and it partly overturns that paragraph. They *do* report cross-sectional momentum clearing Sharpe 1.0 **net** of a 15bp cost built from real venue fees and measured slippage: 1.28 for the (14, 7) long-short on all coins, 1.40 for (14, 5) on the top 5%. But they also show why that number should not be trusted on its own. Once interim daily price moves are simulated, five of their 21 cross-sectional portfolios are outright **liquidated**; and once profitability is tested on mean *log* returns rather than mean returns, **none** of the portfolios clears the t > 3.0 bar appropriate for a literature this heavily mined. Their verdict is that a market-neutral momentum long-short "appears unattainable." Full treatment in section 2.5, which is now the most important section in this note.

---

## 2. Per-strategy sections

### 2.1 Cross-sectional oscillator momentum, RSI-14 rank

**Economic hypothesis.** Retail-dominated crypto markets underreact to sustained directional pressure, and a bounded oscillator measures that pressure without letting one microcap's 500% week dominate the cross-sectional rank the way a raw return does.

**Signal.** For each coin, the 14-day relative strength index on daily closes, the ratio of average gains to average losses over the preceding 14 days ([Fieberg et al. 2024, section III.B.1](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). At the start of each week, rank the cross-section by RSI, assign to quintiles, form **value-weighted** portfolios. Long the top quintile, short the bottom.

**Rebalance.** Weekly.

**Universe.** CoinMarketCap price, volume and market-cap data. All coins with non-missing observations and market cap at or above USD 1 million, returns truncated at the 0.5% and 99.5% percentiles. 3,244 unique coins over the sample, from under 100 in 2015 to over 2,000 in 2021 ([section III.A](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). CoinMarketCap lists both active and defunct coins, which is the standard survivorship defence in this literature ([Liu, Tsyvinski and Wu, NBER WP 25882, section 2](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)).

**Reported performance.** High-minus-low **3.52% per week, t = 5.41. CCAPM alpha 3.17%, LTW three-factor alpha 2.09%**. The quintile pattern is monotone: 0.00, 0.75, 1.75, 2.42, 3.52 percent ([Table 2](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Related oscillators in the same table, stochastic-K at 3.96% with t = 5.73 and CCI at 3.80% with t = 5.03.

**Gross or net.** Gross. Table 2 carries no cost adjustment. No Sharpe, no drawdown and no turnover is reported for the individual indicators, only for the CTREND aggregate. That is the largest single gap in the case for this strategy, and the first thing our own backtest has to fill in.

**Sample period.** Apr 2015 to May 2022, 423 weekly observations.

**Venue and instrument.** None. These are CoinMarketCap composite spot prices, not a tradeable venue. Implementing means picking a venue and re-deriving the signal from that venue's bars.

**Citation.** Christian Fieberg, Gerrit Liedtke, Thorsten Poddig, Thomas Walker, Adam Zaremba, "A Trend Factor for the Cross Section of Cryptocurrency Returns," *Journal of Financial and Quantitative Analysis*, 2024. DOI [10.1017/S0022109024000747](https://doi.org/10.1017/S0022109024000747). Open-access PDF, read in full: https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf, accessed 2026-08-30.

---

### 2.2 CTREND, ML-aggregated trend signal

**Economic hypothesis.** No single technical indicator captures the trend. A cross-sectionally fitted combination of many of them extracts the shared predictive content while averaging away each one's idiosyncratic noise.

**Signal.** 28 daily technical indicators per coin in four groups. Momentum oscillators: RSI-14, stochK-14, stochD, stochRSI, CCI. Price moving averages: SMA at 3, 5, 10, 20, 50, 100 and 200 days scaled by the week's closing price, plus MACD as a 12/26-day EMA difference expressed as a percentage of the fast EMA, plus MACD minus its 9-day signal. Volume: dollar-volume SMAs at the same seven horizons normalized by current volume, volume-MACD, volume-MACD-signal, Chaikin money flow. Volatility: Bollinger low, mid and high scaled by the current close, plus Bollinger bandwidth. Those 28 feed a cross-sectional elastic-net predictive regression estimated on a **rolling 52-week window**, weighted by market cap, producing a one-week-ahead expected return per coin ([sections III.B and III.C](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Rank on that forecast, quintile, value-weight, long the top and short the bottom.

**Rebalance.** Weekly.

**Universe.** As section 2.1. CoinMarketCap, market cap at or above $1M, 3,244 unique coins.

**Reported performance.** Long-short **3.87% per week gross, t = 5.19**, weekly standard deviation 13.9 to 16.0 percent by leg, **annualized gross Sharpe 1.94**. The LTW-model alpha is 2.62% per week, t = 4.22, on a momentum beta of 0.79. It correlates heavily with plain momentum without being subsumed by it ([Table 3](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)).

**Net.** This is the part worth having. Turnover is **68.5% of the portfolio per week**. At 30bp long and 40bp short, the authors' baseline taken from Bianchi, Babiak and Dickerson, *JBF* 2022, the long-short spread falls from 3.87% to **2.90% per week, t = 3.89**. At 40/50bp it is 2.62%, t = 3.53. At 50/60bp it is 2.35%, t = 3.16. Breakeven cost is 1.41% per side, and the cost at which significance is lost is 0.88%. Restricted to the **largest 100 coins per week**, gross 3.40% becomes net 2.45%, t = 3.22, breakeven 1.25%, BETC-5% 0.70% ([Table 9](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). No net Sharpe is reported. My own arithmetic, net return at 75% of gross with essentially unchanged volatility, implies a net Sharpe near 1.4 to 1.5, but that is my inference and not the paper's number.

**Robustness.** Across 55,296 alternative research designs, most CTREND specifications land in the Sharpe 0.5 to 2.5 band, and the Lo (2002) Sharpe test is significant at 5% in 79% of them. The baseline CS-C-ENet family spans Sharpe 1.45 to 10.92 with a **median of 1.34**, and the authors flag that the extreme upper tail comes specifically from combining equal weighting with switching off the $1M market-cap filter ([section VI.A](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Read that as a warning label, not a result.

**Sample period and venue.** Apr 2015 to May 2022, CoinMarketCap composite prices, no venue.

**Citation.** Same as section 2.1.

---

### 2.3 LTW CMOM, plain past-return cross-sectional momentum

**Economic hypothesis.** Coins with high recent returns keep outperforming over the following week, because attention and inflow chase recent winners.

**Signal.** Cumulative return over the previous k weeks, k in {1,2,3,4}. Each week, sort the cross-section into quintiles, value-weight, long quintile 5 and short quintile 1. The factor version, CMOM, uses **3-week momentum** with 30/40/30 breakpoints, value-weighted, top minus bottom ([NBER WP 25882, sections 3.2 and 5](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). The 2026 update from the same group uses **2-week** past returns ([Borri, Liu, Tsyvinski and Wu, arXiv 2510.14435](https://arxiv.org/html/2510.14435v4)).

**Rebalance.** Weekly.

**Universe.** Original: CoinMarketCap, all coins with price, volume and market cap above $1,000,000. 1,707 coins, growing from 109 in 2014 to 1,583 in 2018, non-return variables winsorized at 1% and 99% weekly. CoinMarketCap "lists both active and defunct cryptocurrencies, thus alleviating concerns about survivorship bias" ([section 2](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). The 2026 update moves to CoinGecko, 2013-12-31 to 2025-09-06, **explicitly excludes stablecoins and wrapped tokens**, requires at least 30 daily observations and market cap at or above $1M at formation, and winsorizes at 0.0025%, giving 16,468 unique coins ([arXiv 2510.14435](https://arxiv.org/html/2510.14435v4)).

**Reported performance, gross.**

| Signal | Long-short weekly | t | Source |
|---|---|---|---|
| 1-week momentum | 2.7% | 1.99 | [NBER Table 4](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf) |
| 2-week | 3.3% | 2.44 | same |
| 3-week | 4.1% | 2.74 | same |
| 4-week | 2.5% | 2.00 | same |
| 8, 16, 50, 100-week | not significant | | same |
| CMOM 2-week, 2014 to 2025 | 2.6% | 3.89 | [arXiv 2510.14435](https://arxiv.org/html/2510.14435v4) |
| CMOM 2-week, post-2020 | 2.1% | 3.70 | same |

**Gross or net.** All gross. Neither the NBER working paper nor the 2026 update reports any transaction-cost adjustment. Neither reports a Sharpe ratio for CMOM.

**Best available Sharpe estimate.** From an independent replication across 6,144 research designs, **median annualized Sharpe 0.83, maximum 2.30, minimum -4.47, significant at 5% in 49% of designs** ([Fieberg et al. 2024, section VI.A](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Still gross. That is the number I would put in front of anyone quoting "crypto momentum has Sharpe 2".

**Where the effect lives.** Momentum concentrates in the *larger* coins, which is unusual for a crypto anomaly. The below-median-size group earns an insignificant 0.6% per week, the above-median group a significant 4.2% per week ([NBER section 1](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). That is the most encouraging single fact for implementability in this whole note. The strategy also survives replacing the short leg with a short in Bitcoin, where "the results virtually do not change", which matters because most altcoins are not shortable in size.

**Contradicting evidence.** Grobys and Sapkota, *Economics Letters* 180 (2019), 6 to 10, run **monthly** momentum on 143 coins over 2014 to 2018 and find no significant payoffs. Momentum produced an insignificant raw payoff of 0.90% per month, and the significant payoffs they did find came from "small cryptocurrencies that contaminate the strategies' payoff" ([open-access copy](https://osuva.uwasa.fi/bitstream/handle/10024/10391/Osuva_Grobys_Sapkota_2019.pdf?sequence=2&isAllowed=y)). Their monthly time-series tests at K = 12, 6 and 1 months gave average payoffs of 17.71 and 18.22 and similar, also not significant at 5%. Read alongside LTW, the reading is that crypto cross-sectional momentum is a 1-to-4-week phenomenon that dies at monthly horizons. Any config we write should treat a monthly lookback as an expected null, not a variant worth many runs.

**Citations.** Liu, Tsyvinski and Wu, "Common Risk Factors in Cryptocurrency," *Journal of Finance* 77(2), 1133 to 1177, 2022, https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119. Paywalled, so I read the free NBER WP 25882 version at https://www.nber.org/system/files/working_papers/w25882/w25882.pdf. Borri, Liu, Tsyvinski and Wu, "Cryptocurrency as an Investable Asset Class: Coming of Age," arXiv:2510.14435v4, revised 2026-03-21, https://arxiv.org/html/2510.14435v4. Grobys and Sapkota, "Cryptocurrencies and momentum," *Economics Letters* 180, 2019, https://www.sciencedirect.com/science/article/pii/S0165176519301077.

---

### 2.4 Long-only liquid winners, momentum by liquidity double sort

**Economic hypothesis.** Herding drives momentum, and herding happens where people actually trade, so the momentum effect should be strongest in the most liquid coins, which is also where it is cheapest to trade.

**Signal.** Two independent 30/40/30 sorts. Momentum uses the 14-day cumulative return, C_i(t) = (P_i(t) - P_i(t-14)) / P_i(t-14), with winners the top 30%. Liquidity uses Amihud illiquidity, I_i(t) = (1/T) sum |R_i(tau)| / V_i(tau) over the same 14-day window with V in USD dollar volume, and liquid means the **bottom** 30% of I. "Liquid winners" is the intersection. Equal-weighted, long-only ([arXiv 1904.00890, section 3](https://arxiv.org/pdf/1904.00890)).

**Rebalance.** Every 14 days.

**Universe.** 711 coins met the inclusion criteria at least once, with a minimum market cap of $1M and roughly half a year of history at inclusion. The paper does not describe the underlying vendor's dead-coin coverage, so I cannot vouch for its survivorship handling.

**Reported performance.** Liquid winners: mean **daily** return 0.71%, daily standard deviation 7.43%, **information ratio 1.59** at zero cost, 1.56 at 10bp, 1.43 at 50bp, 1.27 at 100bp. The illiquid-losers portfolio scores higher at IR 2.49 gross, but the authors themselves warn its bid-ask costs would be "significantly higher" in practice ([Table 2 and section 3](https://arxiv.org/pdf/1904.00890)).

**Gross or net.** Cost-tested, and honestly so, since the cost ladder is the paper's own. But the benchmark for the IR is a cap-weighted crypto portfolio, so this is an information ratio and not a Sharpe, and it is not directly comparable to the 1.94 figure in section 2.2. No max drawdown and no turnover are reported.

**Sample period.** Roughly Jul 2014 to late 2018.

**Venue and instrument.** Unspecified aggregate spot prices.

**Citation.** Stjepan Begušić and Zvonko Kostanjčar, "Momentum and liquidity in cryptocurrencies," arXiv:1904.00890, 2019-04-01, https://arxiv.org/pdf/1904.00890, accessed 2026-08-30. Unrefereed preprint, weight accordingly.

---

### 2.5 Han, Kang and Ryu, the strongest counter-evidence, now read in full

**Obtained 2026-08-30.** The earlier draft of this note could not open this paper. It is now retrieved and read: the 116-page version including the Internet Appendix, hosted by AUT's Centre for Financial Research, saved at `docs/research/papers/han-kang-ryu-2023-crypto-momentum.pdf`. Chulwoo Han, Byeongguk Kang and Jehyeon Ryu, all of Sungkyunkwan University. Everything below is quoted from that PDF.

This is the most methodologically careful paper in the set, and it is the one that most directly threatens the recommendation in section 6. Read it before writing any code.

**What it does differently.** Three things, each of which the other papers in this note omit:

1. **Interim price fluctuation.** Every other study evaluates a weekly strategy on weekly returns, which hides what happened inside the week. Han et al. simulate daily and check whether the portfolio would have been wiped out mid-holding-period. In their words a portfolio whose loss exceeds 100% "is liquidated and the loss is" terminal. They give the concrete case: a single day's move in UNFI on 2021-07-07 "could have liquidated the entire portfolio. Yet, the return of the week" overshadows it. Weekly-bar evaluation "overestimates the Sharpe ratio and misses liquidation events."
2. **Log returns, not just mean returns.** With fat tails, "a portfolio can earn a negative profit even when the mean return is statistically significantly positive." They therefore t-test the mean *log* return as the long-term profitability test. This is the paper's sharpest methodological point and it is not optional.
3. **Costs from actual venue data.** 15bp per trade, built from Binance's 10bp spot / 4.5bp futures fee, a 3.26bp average relative tick size, and per-coin slippage measured from trading data (min 0.01bp, mean 1.53bp, max 11.81bp). They call 15bp "a reasonable estimate (or perhaps closer to the lower limit)."

**Universe.** All CoinMarketCap coins from 2013-12-28 to 2023-08-28, filtered each day on market cap at or above $1M **and** daily volume at or above $1M, with 96 stablecoins excluded. Count runs 5 coins at the start, peaks at 784 in Dec 2021, ends at 433. Trading volume, not market cap, is the binding filter. BTC and ETH average 79.0% market dominance across the sample.

**The headline results.**

| Strategy | Net Sharpe (15bp) | Note |
|---|---|---|
| Time-series momentum, (28, 5) long-only | **1.51** | market 0.84 over same window |
| Cross-sectional (14, 7) long-short, all coins | **1.28** | market 1.01 |
| Cross-sectional (14, 5) long-short, top 5% | **1.40** | market 1.01, table 15b |
| Cross-sectional (14, 5) long-only, top 5% | **1.54** | 88.2% max drawdown |

So cross-sectional momentum *does* clear Sharpe 1.0 here, net of 15bp. That is a correction to the earlier draft of this note, which reported the paper as finding cross-sectional momentum simply weak. The real finding is more specific and more damaging.

**Why the authors still call the cross-sectional evidence weak.** Of 21 cross-sectional portfolios over selected lookback/holding pairs, **five are liquidated** during the sample and only six beat the market. Ten portfolios show a positive mean return with t > 2.0, but only three have a mean *log* return with t > 2.0, and **none** clears the t > 3.0 bar that Harvey et al. (2016) propose for a market with this much multiple testing. Six portfolios with a positive mean return "are either liquidated or earn a negative profit." Their own summary: "A momentum-based long-short strategy that can generate steady, market-neutral profits appears unattainable."

**Four findings that bear directly on our design.**

- **The edge is in the long leg, and the short leg is what kills you.** Every short-only portfolio in table 15b is negative, most at -99% or -100% cumulative. "The majority of short-only portfolios plunge by 99% during the 2017 bull market and another 99% during the 2020 bull market." Note that our section 6 config uses a BTC short leg rather than a loser-basket short, which sidesteps this specific failure, but it means we should not expect the paper's long-short numbers to transfer.
- **Crypto momentum inverts the equity pattern.** In equities the profit comes from the short leg and small caps. Here "it originates mainly from the long leg and large coins." Outside a few of the largest coins, "the majority of the coins exhibit reversal rather than momentum."
- **This contradicts section 2.4.** Han et al. find the long leg "worsens monotonically with volume" and that the lowest-volume group earns a slightly *higher* Sharpe than the highest-volume group. That is the opposite sign to Begušić and Kostanjčar's liquid-winners result. Two papers, opposite conclusions, on overlapping data. Do not build on section 2.4 without resolving this.
- **The large-coin effect has decayed.** For the top-5% universe, both the (1,7) and (14,5) portfolios "perform steadily until early 2021, but they then move sideways generating almost no profits." Our holdout window is entirely inside that flat regime.

**The authors' own look-ahead admission.** "We test various pairs of look-back and holding periods and choose optimal combinations. This practice introduces a look-ahead bias... our findings should be regarded as an optimistic view." Their conclusion is that given the tail risk, the small number of liquid coins and the dominance of a few majors, "it is difficult to argue that a cryptocurrency momentum strategy is an attractive alternative investment vehicle to institutional investors."

**Citation.** Chulwoo Han, Byeongguk Kang, Jehyeon Ryu, "Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market: A Comprehensive Analysis under Realistic Assumptions," SSRN 4675565, posted 2023-12-26, doi 10.2139/ssrn.4675565. SSRN itself returns HTTP 403; the full text with Internet Appendix is open at https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf, retrieved 2026-08-30, 9.2MB, 116pp, local copy in `docs/research/papers/`.

---

### 2.6 Looked at, not recommending

**Gbadebo (2026), "Momentum Trading in Cryptocurrencies"**, *Buhalterinės apskaitos teorija ir praktika* 33. Open access, and useful precisely as a null. On **eight** majors, BTC, ETH, LTC, XRP, BNB, ADA, DOGE and SOL, over Jan 2020 to Oct 2025, daily-rebalanced cross-sectional EMA momentum returned 14.59% annually on a daily Sharpe of 0.0128, which annualizes to about **0.20**, with a 55.0% max drawdown, against 31.96% and daily Sharpe 0.0282, about 0.45 annualized, for time-series momentum. The paper states its results are before trading costs. https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419. An eight-name cross-section is barely a cross-section, but it is a real datapoint that the effect is not in the majors alone.

**"Failure of Cross-Sectional Alpha Screening on Cryptocurrency Perpetual Futures"** (SSRN 6701738, Mar 2026). SSRN returned 403 and I could not read it. The indexed summary reports net Sharpes of -3.22 and -2.91 for linear and gradient-boosted cross-sectional models on large-cap perps at an 8-hour horizon. Unverified, so treat it as a rumour and not a result. The claim is at least directionally consistent with everything above, that the edge is weekly and is not in the large-cap perp cross-section at intraday horizons. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738

**`phuazz/crypto-breadth`**, an open-source cross-sectional momentum implementation on Binance USDT spot. Composite 30/90/180-day risk-adjusted-return rank, top-4 equal-weight behind a breadth gate, 50-day MA trend filter, weekly Monday rebalance with daily exit overrides, signal at close T executed at close T+1, 10bp per side, and a rolling liquidity gate of at least $25M trailing 30-day ADV and at least 90 days of history. Claims 1.359 Sharpe, 74.9% CAGR and -39.5% max drawdown over 2018 to 2026, and 1.456 Sharpe from 2021. Secondary, self-reported and not peer reviewed, but valuable because its own README states the disqualifying caveat: the 25-name candidate list is hindsight-selected, coins that later delisted are absent, and the bias concentrates in altcoin-heavy years. That is exactly the failure our point-in-time-universe invariant exists to prevent. https://github.com/phuazz/crypto-breadth

---

## 3. Cost and capacity reality check

**What the papers assume.** Fieberg et al. use **30bp long and 40bp short per rebalance** as their baseline, sourced from Bianchi, Babiak and Dickerson (*JBF* 142, 2022), and stress-test at 40/50 and 50/60bp. They note the assumption "may be conservative" for their sample, because the Bianchi et al. cost data comes from CryptoCompare, which skews to larger coins, while their own sample includes many small ones ([section VI.B](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Begušić and Kostanjčar ladder 0, 10, 50 and 100bp. LTW and the 2026 Borri et al. update assume nothing, so those results are pure gross.

### 3.1 A network caveat that shapes what follows

From this machine, `binance.com`, `developers.binance.com`, `okx.com` and `bybit.com` all resolve to a single ISP sinkhole at 202.3.218.137 serving an expired 2023 certificate for `internetbaik.telkomsel.com`, an Indonesian Telkomsel content filter. That is the cause of the "certificate has expired" and connection-reset errors below, not anything wrong with the exchanges. Working around it with DNS-over-HTTPS and `curl --resolve` recovered the API hosts and some marketing hosts, but not the Binance fee pages. **Anyone re-running this research from an unfiltered network should redo section 3.2, and should not trust a fee number that this note marks unverified.**

### 3.2 Current venue costs

| Venue and instrument | Maker | Taker | Source | Verified |
|---|---|---|---|---|
| OKX spot, regular user | 0.080% (8bp) | 0.100% (10bp) | https://www.okx.com/fees and its backing `/v3/users/support/common/fee/fee-table` API | yes, fetched 2026-08-30 |
| OKX futures and perp, regular user | 0.020% (2bp) | 0.050% (5bp) | same | yes, fetched 2026-08-30 |
| Binance spot, VIP 0 | not verified | not verified | https://www.binance.com/en/fee/schedule | **no**, the page returned HTTP 202 anti-bot challenges on all four edge IPs |
| Binance USD-M perp, regular | not verified | not verified | https://www.binance.com/en/fee/futureFee | **no**, same block |
| Bybit spot and perp | not verified | not verified | https://www.bybit.com | **no**, 403 to direct-IP requests |

I am leaving the Binance and Bybit cells empty on purpose. The repo's net invariant requires stating the cost assumption alongside every number, and a fee I half-remember is not an assumption, it is a guess. Filling those three rows needs a network path that is not ISP-filtered.

**What the verified figures do to the papers' assumptions.** OKX's 10bp spot taker and 5bp perp taker sit well inside the 30 to 40bp the CTREND authors assume, on the fee line alone. Their cost figure is doing double duty as fee plus effective spread, and that is the right way to read it. The exchange fee is the small, known part of the cost. The spread plus market impact on a $5M-ADV altcoin at the Monday open is the large, unknown part. Our slippage model, not the fee schedule, decides whether this works.

### 3.3 Funding, if we implement on perps

Correcting a widely repeated claim: **Binance funding is not uniformly 8-hourly, and the cap is not a single global number.** Querying `GET /fapi/v1/fundingInfo` on 2026-08-30 returned 768 symbols, of which **441 settle every 4 hours, 324 every 8 hours and 3 every hour**. Caps are per symbol. BTCUSDT and ETHUSDT are 8-hourly with a cap and floor of plus or minus 0.30%, SOLUSDT is plus or minus 0.375%, and the modal cap across all symbols is plus or minus 2.00%, covering 693 of them. The commonly quoted "8 hours, plus or minus 0.75%" does not match what the API says today. Bybit says the same thing in its docs, that "each symbol has a distinct funding interval, which can be queried via the instruments-info endpoint", and its live API reports BTCUSDT at a 480-minute interval (https://bybit-exchange.github.io/docs/v5/market/history-fund-rate). OKX perp funding is 8-hourly with a cap of plus or minus 0.375% and an interest rate of 0.0001, from `GET /api/v5/public/funding-rate`.

A funding model that hardcodes 8 hours will silently misprice more than half the Binance symbol set, and the mispriced half skews toward the smaller caps, which is exactly the leg where the edge is supposed to live.

### 3.4 Where the gross numbers would not survive

**The short leg is the problem, not the long leg.** Every cross-sectional design here is long-short, and the short leg of a crypto momentum sort is by construction the worst-performing, lowest-attention, thinnest coins in the cross-section. Locating spot borrow for those is usually impossible, and on perps many have no listed contract at all. LTW's own robustness check, replacing the short quintile with a short in Bitcoin and finding the results "virtually do not change" ([NBER section 1](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)), is the practical answer, and we should adopt it from day one rather than discover it later.

**Turnover is the tax.** Replacing 68.5% of the book weekly ([Fieberg et al. Table 9](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)) is roughly 35 times annual portfolio turnover. Each 10bp of round-trip cost therefore costs about 3.5% per year of gross return, before any slippage. The published breakeven costs, 1.41% per side full sample and 1.25% for the largest 100, sound generous only because the gross spread is enormous. Halve the gross spread, which is what "not 2015 to 2022" is likely to mean, and the breakeven halves with it.

**The $1M market-cap floor is a fiction at our size.** A coin at $1M market cap does not have a book. The papers' floor exists to exclude what does not matter statistically. It excludes nothing that matters executionally. Any config we write must set the floor by **dollar volume**, not market cap, at a level where our target notional is a small fraction of rebalance-hour depth.

**The gross figure I trust least.** LTW's headline 4.1% per week for 3-week momentum, 2014 to 2018, runs on a universe whose *median* coin had a market cap of $8.17 million and *median* daily dollar volume of about $104,000 ([NBER Table 1](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). At 68% weekly turnover and even a 50bp effective round trip on names that size, a large fraction of that spread is gone, and the true cost on the median name in that sample is far above 50bp. This is the number most often quoted as evidence that crypto momentum works, and it is the weakest evidence of an implementable edge in this note.

**Funding is a cost we would be adding, not one the literature accounts for.** No paper here trades perps, so none carries a funding line. If we implement the long leg on perps to get leverage and shortability, funding on a portfolio that is by construction long the hottest coins is a systematically adverse carry, because winners run positive funding. That has to enter the simulation as a realized-rate series per position per settlement, per `docs/agents/quant-research.md`, and never as a haircut.

---

## 4. Data requirements

### 4.1 Bars

The **Binance public data archive** at `data.binance.vision` is the cheapest path to deep, free, venue-native history. It covers spot, USD-M futures and COIN-M futures, with klines, trades and aggTrades. Kline intervals are 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w and 1mo, in daily and monthly zips each with a `.CHECKSUM`. URL pattern: `https://data.binance.vision/data/{spot|futures/um|futures/cm}/{daily|monthly}/{klines|trades|aggTrades}/{SYMBOL}/{INTERVAL}/{file}.zip`. **Spot timestamps switch from milliseconds to microseconds on 2025-01-01**, a bar-convention trap the repo's raw-source record must capture. https://github.com/binance/binance-public-data

The archive appears to retain delisted symbols. The README's own worked example downloads `ADABKRW`, a pair whose quote asset was discontinued in 2021. That is empirical evidence, not a documented guarantee, since the README says only "All symbols are supported". Verify against two or three known-dead pairs before depending on it.

For incremental pulls, the REST limits differ between products and both were confirmed live on 2026-08-30. `GET /api/v3/klines` on spot defaults to 500 and caps at 1000, and a `limit` above 1000 is silently clamped rather than rejected. `GET /fapi/v1/klines` on USD-M futures caps at 1500, and 1501 returns error code -1130. Spot BTCUSDT 1d history starts 2017-08-17, futures BTCUSDT 1d starts 2019-09-08. Docs: https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md

Reachable alternatives, both fetched directly. Bybit `GET /v5/market/kline` takes `category` of spot, linear or inverse, plus `symbol`, `interval` in {1,3,5,15,30,60,120,240,360,720,D,W,M}, `start` and `end` in milliseconds, and `limit` 1 to 1000 defaulting to 200, returned newest-first as start, open, high, low, close, volume, turnover (https://bybit-exchange.github.io/docs/v5/market/kline). OKX `GET /api/v5/market/candles` caps at 300, with deeper history behind the separate `GET /api/v5/market/history-candles`, also capped at 300.

### 4.2 Funding

Funding history is needed per symbol per settlement for the whole backtest window, as a **realized** series. The protocol forbids a modelled or averaged rate, and section 3.3 explains why an assumed 8-hour grid is wrong for most Binance symbols. Join on actual settlement timestamps rather than resampling.

Two pagination traps, both confirmed live on 2026-08-30. Binance `GET /fapi/v1/fundingRate` accepts a `limit` up to 1000, rejecting 1001, but **returns at most 500 rows per call regardless**. A loader that trusts the accepted `limit` will silently drop data. BTCUSDT funding history begins 2019-09-10 08:00 UTC. Bybit `GET /v5/market/funding/history` caps at 200 per request and errors if you pass `startTime` alone (https://bybit-exchange.github.io/docs/v5/market/history-fund-rate). OKX `GET /api/v5/public/funding-rate-history` caps at 400. A multi-year pull per symbol is therefore thousands of requests. Budget for it and write results to `data/raw/` append-only.

### 4.3 Point-in-time, survivorship-free universe

This is the hard part, and the finding is blunt. There is exactly one first-party endpoint that hands you a historical ranked cross-section, and it is paid.

**CoinMarketCap `GET /v1/cryptocurrency/listings/historical`** returns the ranked list as of the end of a historical UTC day, carrying the *historical* `cmc_rank` alongside supply, price, volume and market cap per coin. Parameters are `date` (required), `start`, `limit`, `sort`, `sort_dir`, `cryptocurrency_type`, `convert` and `aux`. History depth by plan: Basic 1 year, Builder 3 years, and Startup, Growth, Professional or Enterprise back to **2013-04-28**. It costs 1 credit per 100 coins returned. https://pro.coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency/listings-historical.md

**CoinMarketCap `GET /v1/cryptocurrency/map`** with `listing_status=active,inactive,untracked` gives the ever-existed universe, and each record carries `first_historical_data` and `last_historical_data`, exactly the fields needed to know when an asset entered and left coverage. It is callable without a key via the `/public-api` prefix. https://pro.coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency/cryptocurrency-id-map.md

**CoinGecko cannot give a point-in-time top-N.** `GET /coins/markets` returns `market_cap_rank` but has no as-of-date parameter, so it is a current snapshot only (https://docs.coingecko.com/reference/coins-markets). `GET /coins/{id}/history?date=` returns market cap for one coin on one day but not `market_cap_rank` (https://docs.coingecko.com/reference/coins-id-history). Reconstructing rankings yourself means pulling the ever-listed universe via `GET /coins/list?status=inactive` plus `status=active`, where the `inactive` value "retrieve[s] coins no longer listed on CoinGecko" and **requires the Analyst plan or above** (https://docs.coingecko.com/reference/coins-list), then pulling per-coin `GET /coins/{id}/market_chart/range` with `vs_currency`, `from`, `to` and `interval`, which returns `prices`, `market_caps` and `total_volumes`, and ranking cross-sectionally at each rebalance date. Auto-granularity is 5-minutely for windows up to 1 day, hourly for 2 to 90 days and daily beyond 90. Hourly history starts 2018-01-30 and 5-minutely 2018-02-09. https://docs.coingecko.com/reference/coins-id-market-chart-range. Basic-plan history is capped at **2 years** (https://docs.coingecko.com/reference/coins-id-market-chart), which contradicts the widely repeated "365 days" figure that I could not find on any current CoinGecko page.

Whether CoinGecko keeps `market_chart` history queryable for `status=inactive` coins is not documented. Test it against a known-dead id before designing around it.

**Binance has no historical `exchangeInfo`, and no delisted status.** `GET /api/v3/exchangeInfo` is current-state only and takes an optional `symbolStatus` filter of TRADING, HALT or BREAK. The full spot enum is TRADING, END_OF_DAY, HALT, BREAK and CANCEL_ONLY (https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/enums.md). A live call on 2026-08-30 returned 3,685 spot symbols, 1,358 TRADING and 2,327 BREAK, and **no DELISTED value exists**: delisted symbols vanish from the response entirely. On USD-M futures, `GET /fapi/v1/exchangeInfo` returned 883 symbols with statuses TRADING (752), SETTLING (130) and PENDING_TRADING (1). Two consequences. First, derive historical Binance listing windows from file presence per symbol per month on `data.binance.vision`, and document the result as reconstructed rather than exact. Second, **start snapshotting `exchangeInfo` daily into `data/raw/` now**, because that history cannot be recovered later. Bybit is slightly better here, exposing a lifecycle enum of PreLaunch, Trading, Delivering and Closed (https://bybit-exchange.github.io/docs/v5/enum), which is the closest thing to a first-party delisting marker any of these venues offers.

**Stablecoin and wrapped-asset exclusion.** The 2026 Borri, Liu, Tsyvinski and Wu update excludes both from its cross-section (https://arxiv.org/html/2510.14435v4), and we should too. CoinGecko's keyless `GET /coins/categories/list` returns usable exclusion category ids including `usd-stablecoin`, `eur-stablecoin`, `algorithmic-stablecoin`, `crypto-backed-stablecoin`, `fiat-backed-stablecoin`, `commodity-backed-stablecoin` and `yield-bearing-stablecoins`, plus `wrapped-tokens`, `bridged-wbtc`, `bridged-weth`, `bridged-wsteth`, `bridged-wbnb` and `bridged-wavax`. `GET /coins/markets` accepts a `category` filter to pull members. https://docs.coingecko.com/reference/coins-categories-list. There is a lookahead warning attached. These tags are current. A coin's tag today is not its tag then, and CoinMarketCap's `listings/historical` endpoint does not appear to accept `tag` or `aux=tags` at all. Applying a present-day stablecoin tag to a 2018 cross-section is a small but real look-ahead, and the honest mitigation is a hand-curated dated exclusion list stored in the config.

**Coin Metrics community API** at `https://community-api.coinmetrics.io/v4` needs no key and allows 10 requests per 6 seconds per IP. It offers `GET /reference-data/assets` and `/catalog-v2/asset-metrics` with per-asset `min_time` and `max_time` coverage windows, which is a free way to derive first and last data dates. Market cap is `CapMrktCurUSD`. Whether that metric sits inside the free community subset, and at what depth, is not documented. Resolve it with one unauthenticated call. https://docs.coinmetrics.io/api/v4

### 4.4 Minimum viable fetch for the recommendation in section 6

1. CoinMarketCap `/v1/cryptocurrency/map?listing_status=active,inactive,untracked` for the ever-existed universe with entry and exit dates. One cheap call.
2. Either CoinMarketCap `listings/historical` weekly snapshots, which is paid and correct, or CoinGecko `market_chart/range` per coin plus our own ranking, which is cheaper but more work and still needs the Analyst tier for dead coins.
3. `data.binance.vision` monthly 1d spot klines for every symbol that ever existed, plus USD-M futures klines and funding history if we take the perp route.
4. Daily `exchangeInfo` snapshots from today onward, appended to `data/raw/`.
5. A dated stablecoin and wrapped-asset exclusion list, hand-curated and versioned in the config.

---

## 5. Known failure modes and red flags

**Regime concentration.** Every headline result here is measured over a window containing 2017 or 2021 or both. Fieberg et al. run Apr 2015 to May 2022, LTW run 2014 to 2018, Begušić and Kostanjčar end in 2018. The one genuinely encouraging piece of evidence against pure regime-fitting is that CMOM's weekly spread only falls from 2.6% full sample to 2.1% post-2020, with the t-statistic holding at 3.70 ([arXiv 2510.14435](https://arxiv.org/html/2510.14435v4)). But that is still gross, and it still includes 2021. Our own out-of-sample window has to exclude 2021 entirely to mean anything.

**The microcap engine.** Fieberg et al. state it outright. The extreme Sharpe ratios in their design-space sweep, as high as 10.92, come from one specific combination: equal weighting with the $1M market-cap filter switched off ([section VI.A](https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf)). Grobys and Sapkota reach the same conclusion from the other direction, finding that the momentum payoffs they measured as significant were driven by "small cryptocurrencies that contaminate the strategies' payoff" ([open-access PDF](https://osuva.uwasa.fi/bitstream/handle/10024/10391/Osuva_Grobys_Sapkota_2019.pdf?sequence=2&isAllowed=y)). If one of our runs produces a Sharpe above 2, the first hypothesis to test is that a $2M-market-cap coin with a 40% spread is doing the work.

**Stablecoins and wrapped assets polluting the cross-section.** A stablecoin has near-zero return variance, so it sits permanently mid-pack in a momentum rank and permanently at the top of a low-volatility rank. A depeg puts it violently at the bottom of a momentum rank right before it mean-reverts. Wrapped assets are duplicate exposure, and holding both WBTC and BTC in a supposedly diversified quintile is one position, not two. LTW's original 2014 to 2018 universe excluded neither. The 2026 update excludes both. Treat exclusion as mandatory.

**Look-ahead in universe selection.** Two distinct versions. The obvious one is picking today's top-N coins and backtesting them, the failure the `crypto-breadth` README self-diagnoses. The subtle one is using a *current* tag, category or symbol list to filter a *past* cross-section. Since Binance publishes no historical `exchangeInfo` and has no delisted status at all, any Binance-tradeability filter applied to 2019 is reconstructed and therefore approximate. Document how it was reconstructed rather than pretending it is exact.

**Stale bars reading as low volatility.** Called out in `docs/agents/quant-research.md` and worth repeating. A delisted or halted symbol that keeps emitting a flat last price looks like a zero-return, zero-volatility asset. In a momentum sort it lands in the middle quintile forever and quietly absorbs weight. In a vol-scaled sort it gets enormous weight. Test the survivorship path explicitly, as the protocol requires.

**Horizon-shopping.** LTW test eight lookbacks and report four as significant with t-statistics between 1.99 and 2.74, at one, two, three and four weeks, with 8, 16, 50 and 100 weeks insignificant ([NBER section 3.2](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)). Four significant out of eight tested, none with a t-statistic above 3, none multiple-testing-adjusted. The 2026 update settles on 2 weeks. The 2019 factor definition used 3 weeks because, in the paper's own footnote, "it generates the largest long-short spread in the data". That is selection on the outcome. Our config should fix the lookback before the first run and count every alternative tried.

**Cross-sectional against time-series.** Both Han, Kang and Ryu (now read in full, section 2.5, where TS momentum nets Sharpe 1.51 against the best CS long-short at 1.28) and Gbadebo report that time-series momentum outperforms cross-sectional momentum in crypto. Gbadebo's numbers are TS at 31.96% annual and daily Sharpe 0.0282 against CS at 14.59% and daily Sharpe 0.0128, with CS suffering the larger 55.0% max drawdown ([PDF](https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419)), though on an eight-coin universe that is barely a cross-section. This repo is committed to the cross-sectional variant, which is a legitimate scope choice, but we should log a time-series momentum benchmark alongside every result so we can see what we are giving up.

---

## 6. Recommendation

**Implement section 2.1. Cross-sectional RSI-14 rank, weekly, value-weighted quintiles, on a liquidity-filtered Binance-tradeable universe, with a Bitcoin short leg.**

Why this one first. It is a single deterministic function of daily closes, so it is a pure point-in-time feature that satisfies the protocol's "testable on a hand-built fixture" requirement with no fitting step. It is the best-performing simple indicator in a peer-reviewed table at 3.52% per week, t = 5.41. Its bounded 0-to-100 range means a microcap's 500% week cannot dominate the cross-sectional rank the way a raw return does, which attacks the microcap failure mode directly. And it is the natural single-feature ablation of CTREND, so if it works, CTREND becomes a well-motivated v2 rather than a leap.

The config, precisely.

- **Signal.** 14-day Wilder RSI on daily closes, computed strictly from bars closing at or before the decision bar's open.
- **Decision bar.** Weekly, Monday 00:00 UTC. Signal from data through Sunday 23:59:59 UTC, filling at the Monday 00:00 UTC bar open at the earliest.
- **Universe, as of each rebalance date.** Coins with a Binance USDT spot pair listed on that date, reconstructed from `data.binance.vision` file coverage and documented as reconstructed. At least 90 daily bars of history. Trailing 30-day median daily dollar volume at or above **$5,000,000**. Stablecoins and wrapped or bridged assets excluded per a dated hand-curated list. No market-cap floor, since the liquidity filter subsumes it and is the executionally meaningful constraint.
- **Ranking and weighting.** Rank on RSI, take the top quintile long, value-weight within the quintile, cap at 15% per name.
- **Short leg.** Short BTCUSDT at the long leg's notional. LTW show this substitution barely changes the result ([NBER section 1](https://www.nber.org/system/files/working_papers/w25882/w25882.pdf)) and it removes the borrow problem entirely.
- **Costs in the simulation.** Taker fee per side at the venue's actual rate, re-verified against the live fee page from an unfiltered network before the first run, since section 3.2 could not confirm Binance's. Plus an explicit slippage model sized against each name's order-book depth at the Monday 00:00 UTC hour. No post-hoc haircut. Log a sensitivity run at the papers' 30bp assumption so our number is comparable to theirs.
- **Reporting.** Net annualized return and volatility, net Sharpe, max drawdown with dates, per-rebalance turnover, average gross and net exposure, number of positions, and the count of configs tried, per `docs/agents/quant-research.md`.
- **Benchmarks on every run.** Buy-and-hold BTC, and a time-series momentum variant on the same universe.

**Three amendments forced by section 2.5, added 2026-08-30.** Now recorded as [ADR-0001](../adr/0001-mark-daily-and-model-liquidation.md), [ADR-0002](../adr/0002-log-return-profitability-bar.md) and [ADR-0003](../adr/0003-replicate-before-innovate.md), and folded into `docs/agents/quant-research.md`. **ADR-0003 changes the order of work: the first thing implemented is a replication of Han et al.'s (14, 7) portfolio, not the RSI-14 config below.** The config below remains the intended strategy, but it is now step 3, gated on the replication and the regime split.

1. **Simulate daily, not weekly.** The backtest must mark the portfolio every day inside the weekly holding period and record whether cumulative loss would have breached 100%. Evaluating a weekly strategy on weekly bars hides liquidation events and inflates the Sharpe. This is a change to the simulator, not a reporting detail.
2. **Report the t-statistic of the mean log return, not just the mean return.** Han et al. show these diverge badly on fat-tailed crypto returns, with six of their portfolios posting a positive mean return but a negative mean log return. Add it to the reporting block in `docs/agents/quant-research.md`.
3. **Set the bar at t > 3.0 on the log return.** Given how heavily this literature has been mined, Harvey et al. (2016)'s cutoff is the honest threshold, and no cross-sectional portfolio in Han et al. clears it. If ours does not either, that is the answer.

**Out-of-sample window to hold out: 2024-01-01 through 2026-08-30, untouched.**

Three reasons for that boundary. It excludes 2021 entirely, so the holdout cannot be carried by a single 2021-scale move, the exact failure `docs/agents/quant-research.md` names. It sits after the published evidence, since Fieberg et al. end in May 2022 and the most recent CMOM update ends 2025-09-06, so 2024 onward is outside the window in which the RSI result was discovered. And at roughly 138 weekly rebalances it is long enough that a Sharpe estimate is not pure noise. Everything from 2017-01-01 to 2023-12-31 is the development sample. Do not look at 2024 onward until a config is frozen, and record in the ADR how many configs were tried before it was.

If the net Sharpe on the development sample comes in below about 0.8, the honest conclusion is that we have reproduced section 2.3's median-0.83 result and there is no implementable edge in plain cross-sectional momentum for us. At that point the next move is CTREND (section 2.2) or the liquidity interaction (section 2.4), not more lookback tuning.

---

## 7. Sources

All accessed 2026-08-30.

### Papers

- Christian Fieberg, Gerrit Liedtke, Thorsten Poddig, Thomas Walker, Adam Zaremba, "A Trend Factor for the Cross Section of Cryptocurrency Returns," *Journal of Financial and Quantitative Analysis* (2024). DOI https://doi.org/10.1017/S0022109024000747. Open-access PDF read in full: https://unipub.lib.uni-corvinus.hu/11621/1/a-trend-factor-for-the-cross-section-of-cryptocurrency-returns.pdf
- Yukun Liu, Aleh Tsyvinski, Xi Wu, "Common Risk Factors in Cryptocurrency," *Journal of Finance* 77(2), 1133 to 1177 (2022): https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119. Read via NBER Working Paper 25882: https://www.nber.org/system/files/working_papers/w25882/w25882.pdf
- Nicola Borri, Yukun Liu, Aleh Tsyvinski, Xi Wu, "Cryptocurrency as an Investable Asset Class: Coming of Age," arXiv:2510.14435v4, revised 2026-03-21: https://arxiv.org/html/2510.14435v4
- Klaus Grobys, Niranjan Sapkota, "Cryptocurrencies and momentum," *Economics Letters* 180 (2019), 6 to 10: https://www.sciencedirect.com/science/article/pii/S0165176519301077. Read via author self-archive: https://osuva.uwasa.fi/bitstream/handle/10024/10391/Osuva_Grobys_Sapkota_2019.pdf?sequence=2&isAllowed=y
- Stjepan Begušić, Zvonko Kostanjčar, "Momentum and liquidity in cryptocurrencies," arXiv:1904.00890 (2019): https://arxiv.org/pdf/1904.00890
- Adedeji Daniel Gbadebo, "Momentum Trading in Cryptocurrencies: A Comparative Study of Time-Series and Cross-Sectional Strategies," *Buhalterinės apskaitos teorija ir praktika* 33 (2026): https://www.journals.vu.lt/BATP/en/article/download/44540/42590/138419
- Chulwoo Han, Byeongguk Kang, Jehyeon Ryu, "Time-Series and Cross-Sectional Momentum in the Cryptocurrency Market: A Comprehensive Analysis under Realistic Assumptions," SSRN 4675565, posted 2023-12-26, doi 10.2139/ssrn.4675565. SSRN landing page returns HTTP 403: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565. **Full text obtained 2026-08-30** from the AUT Centre for Financial Research: https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf (116pp with Internet Appendix). Local copy: `docs/research/papers/han-kang-ryu-2023-crypto-momentum.pdf`. See section 2.5.
- Azka Fayez Junior, "Failure of Cross-Sectional Alpha Screening on Cryptocurrency Perpetual Futures," SSRN 6701738 (2026): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6701738. **Inaccessible, HTTP 403. No figures quoted.**
- Leigh Drogen, Corey Hoffstein, Kevin Otte, "Cross-sectional Momentum in Cryptocurrency Markets," SSRN 4322637 (2023): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637. **Inaccessible, HTTP 403. No figures quoted.**

### Exchange documentation and APIs

- Binance public data archive: https://github.com/binance/binance-public-data
- Binance spot REST API, raw source: https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md. Enums: https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/enums.md
- Binance live endpoints queried directly on 2026-08-30 for the limits, statuses and funding intervals in sections 3.3, 4.1 and 4.3: `https://data-api.binance.vision/api/v3/exchangeInfo`, `https://fapi.binance.com/fapi/v1/exchangeInfo`, `/fapi/v1/klines`, `/fapi/v1/fundingRate`, `/fapi/v1/fundingInfo`
- Binance fee schedule: https://www.binance.com/en/fee/schedule and https://www.binance.com/en/fee/futureFee. **Unreachable, HTTP 202 anti-bot challenge on all edge IPs. No fee figures quoted.**
- OKX fee schedule: https://www.okx.com/fees, plus its backing API `https://www.okx.com/v3/users/support/common/fee/fee-table`. Fetched successfully.
- OKX market and funding endpoints: `GET /api/v5/market/candles`, `GET /api/v5/market/history-candles`, `GET /api/v5/public/funding-rate`, `GET /api/v5/public/funding-rate-history`
- Bybit V5 kline: https://bybit-exchange.github.io/docs/v5/market/kline. Funding rate history: https://bybit-exchange.github.io/docs/v5/market/history-fund-rate. Enums: https://bybit-exchange.github.io/docs/v5/enum
- Bybit fee schedule: https://www.bybit.com. **Unreachable, 403. No fee figures quoted.**

### Data and universe APIs

- CoinMarketCap `listings/historical`: https://pro.coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency/listings-historical.md
- CoinMarketCap `cryptocurrency/map`: https://pro.coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency/cryptocurrency-id-map.md
- CoinGecko `/coins/list`: https://docs.coingecko.com/reference/coins-list
- CoinGecko `/coins/{id}/market_chart/range`: https://docs.coingecko.com/reference/coins-id-market-chart-range
- CoinGecko `/coins/{id}/market_chart`: https://docs.coingecko.com/reference/coins-id-market-chart
- CoinGecko `/coins/{id}/history`: https://docs.coingecko.com/reference/coins-id-history
- CoinGecko `/coins/markets`: https://docs.coingecko.com/reference/coins-markets
- CoinGecko categories: https://docs.coingecko.com/reference/coins-categories-list
- Coin Metrics API v4: https://docs.coinmetrics.io/api/v4. Community data terms: https://gitbook-docs.coinmetrics.io/packages/coin-metrics-community-data. Market cap metric: https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/market/market-capitalization

### Secondary and low-trust, labelled as such in the text

- `phuazz/crypto-breadth` open-source implementation: https://github.com/phuazz/crypto-breadth
