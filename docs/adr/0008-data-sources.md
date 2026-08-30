# ADR-0008: Binance archive for prices, a one-time CoinMarketCap pull for market caps

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** [ADR-0007](0007-venue-and-cost-structure.md), [ADR-0003](0003-replicate-before-innovate.md)

## Context

The point-in-time universe invariant in `CLAUDE.md` needs a price panel that retains delisted assets. Probing established that `data.binance.vision` does exactly that: `SRMUSDT-1d-2022-11.zip` downloads and unzips to 28 real daily bars ending after the delisting, and the bucket is S3-listable at 3,695 spot symbol directories, every file SHA256-checksummed, no API key. History starts 2017-08-17.

Two gaps remain. The archive publishes no market capitalisation, which value-weighting, the cap-weighted market portfolio and Churn all require. And its 2017-08 floor sits inside Han et al.'s sample, which starts 2017-01-01.

No free, licensed source closes those gaps. CoinGecko gates delisted coins behind its $129/mo Analyst tier. CoinMarketCap's cheapest full-depth tier is $79/mo. Its `cryptocurrency/historical` endpoint returns zero rows for delisted coins and is therefore survivorship-*biased*; only `listings/historical` is survivorship-free.

## Decision

**Prices: `data.binance.vision`.** The universe is built from archive file date ranges, **not** from `exchangeInfo` and not from Binance's own `fetch-all-trading-pairs.sh`, which enumerates via `exchangeInfo` without warning anywhere in its README that this is not survivorship-free.

This also keeps backtest and execution on one source: Tokocrypto's USDT book *is* Binance's book — simultaneous fetches of `tokocrypto.site` and `data-api.binance.vision` returned the identical BTCUSDT price of 78844.55.

**Market caps and the Faithful Run panel: one immutable pull via `crypto2`.** The MIT-licensed CRAN package reconstructs a survivorship-free CoinMarketCap panel and backs published work on this exact problem (Ammann, Burdorf, Liebi & Stöckl, SSRN 4287573, 3,904 coins). Run once, emit CSV into `data/raw/`, checksum, never call again. The R-to-Python boundary enforces the one-time discipline rather than obstructing it.

## Considered Options

**Pay CoinMarketCap $79 for one month and bulk-pull.** The clean version of the same thing, with no terms question. Rejected on cost. It remains the obvious fallback if the `crypto2` route breaks.

**Drop market capitalisation entirely** — volume-weight, use BTC as the market proxy, abandon Churn. Rejected because it forfeits two things that separate a result from an anecdote: the Faithful Run, which is the only way to distinguish a broken pipeline from a vendor difference, and Churn, without which the contradiction between Han et al. and Begušić & Kostanjčar on the sign of the volume effect (`docs/research/crypto-momentum-strategies.md` §2.4) can never be settled.

## Consequences

- **The `crypto2` route breaches CoinMarketCap's terms of service.** It uses undocumented public endpoints; no authentication is circumvented and there is no security dimension, but it is a contract term and this ADR records that the breach was chosen deliberately rather than stumbled into. Practical exposure is an IP ban, not liability.
- Undocumented endpoints change without notice. A one-time immutable pull converts that ongoing fragility into a stored artifact, which is the shape `docs/agents/quant-research.md` already wants raw data to have.
- The hybrid introduces a **symbol mapping between CoinMarketCap ids and Binance tickers**. That mapping is a known bug surface and deserves its own fixture tests.
