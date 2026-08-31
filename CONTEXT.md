# crypto-memontum-v1

Cross-sectional momentum research on crypto assets. This glossary fixes the language the research uses, so a signal, a backtest, and an ADR all mean the same thing by the same word.

## Language

### Time and sampling

**Decision Bar**:
The bar at which a signal is formed. A signal uses only data timestamped strictly before it, and the resulting position fills at the next bar's open at the earliest.
_Avoid_: signal bar, formation bar, entry bar

**Development Sample**:
Data through 2023-12-31, which we are free to iterate on. Every configuration tried here is counted and reported.
_Avoid_: training set, in-sample period

**Holdout**:
2024-01-01 through 2026-08-30. Untouched until a configuration is frozen. Looking at it spends it.
_Avoid_: test set, out-of-sample period, validation set

### Portfolio and universe

**Universe**:
The set of assets tradeable as of a given rebalance date, reconstructed as it stood on that date and including assets that later delisted, depegged, or unwound.
_Avoid_: basket, watchlist, coin list, investable set

**Archive Coverage**:
The set of monthly partitions `data.binance.vision` publishes for one symbol, counted only where the zip and its published SHA256 are both present. This is what the Universe is reconstructed from, in place of an exchange listing that only knows what still trades.
_Avoid_: listing history, availability, symbol history

**Archive Floor**:
2017-08-17, the first date the archive publishes anything. A date below it is outside the archive rather than a date on which nothing traded, and a panel says which of the two it means.
_Avoid_: data start, earliest date, history start

**Grid**:
The set of (lookback, holding period) pairs a strategy is evaluated across — 21 of them in the paper we replicate. A result is judged on the shape of the grid, not on a single cell.
_Avoid_: parameter sweep, sensitivity analysis

**Liquidation**:
A cumulative portfolio loss breaching 100%. Terminal: positions close, the return series ends there, and the run reports as liquidated with its date rather than continuing.
_Avoid_: blowup, margin call, wipeout

**Halt**:
A bar at which an asset could not have been traded — no volume, or no price at all. A position held through one exits at the last tradeable price before it, and a print that appears after it is not an exit we could have chosen. A price of zero on real volume is not a halt but a trade at nothing, and is marked as the loss it is.
_Avoid_: delisting, suspension, stale bar

**Churn**:
An asset's traded volume relative to its market capitalisation. High churn marks speculative activity, where price continuation tends to be short-lived.
_Avoid_: turnover, volume ratio, activity

**Stablecoin**:
An asset whose stated design intent is to hold a peg, classified at its listing date and excluded from the Universe permanently — whether or not it later broke that peg.
_Avoid_: pegged asset, stable

**Wrapped Asset**:
An asset that represents a claim on another asset already in the Universe, such as a bridged or liquid-staked token. Excluded permanently, because including it double-counts a single bet.
_Avoid_: derivative token, synthetic, bridged token

**Exclusion List**:
The dated, versioned record of which assets are Stablecoins and which are Wrapped Assets, with the stated design intent each classification rests on. Hand-maintained, so every result quotes the version that produced it.
_Avoid_: blacklist, filter list, ban list

**Liquidity Floor**:
A trailing median dollar volume threshold an asset clears to enter the Universe, read only from bars before the Decision Bar. A data-quality gate on artefacted bars, never a capacity constraint — at this account size capacity binds on almost nothing the archive publishes.
_Avoid_: volume filter, minimum size, liquidity screen

**Universe Bracket**:
The pair of Universes a result is reported on: the full Binance archive as the upper bound and today's Tokocrypto listing as the lower. The gap between them is venue-listing risk, and quoting one bound alone hides which risk was chosen.
_Avoid_: universe variant, listing scope, coverage option

**Trend Gate**:
A portfolio-level switch that holds the selected assets only while the market's own recent return is positive. A time-series overlay on a cross-sectional selection, not a competing strategy.
_Avoid_: regime filter, market filter, risk overlay

### Evaluation

**Faithful Run**:
The Replication Gate executed on the paper's own vendor data over its own window, so vendor differences are eliminated as an explanation for disagreement. Tests whether our pipeline is correct.
_Avoid_: baseline run, control run

**Venue Run**:
The same configuration executed on the archive prices of the venue we would actually trade. Tests whether a published effect survives contact with executable prices. The gap between it and the Faithful Run is a result in its own right, not an error.
_Avoid_: live backtest, realistic run

**Cost Drag**:
The annualised return lost to fees, tax and slippage at a given Rebalance Turnover. Budgeted ahead of a run, not discovered after one.
_Avoid_: transaction costs, friction, fee load

**Replication Gate**:
Reproducing the shape of Han, Kang and Ryu's published grid on their own sample before any of our own numbers are trusted. A pipeline that fails the gate produces uninterpretable results, not merely disappointing ones.
_Avoid_: validation, sanity check, baseline

**Rebalance Turnover**:
The fraction of the portfolio traded at a rebalance. The quantity that pays for a strategy's edge, and reported alongside every result.
_Avoid_: turnover, churn, trading volume

**Net**:
After fees, funding, slippage and tax, with the assumption stated alongside the number. A figure quoted without its cost assumption is not a result.
_Avoid_: after costs, adjusted, all-in
