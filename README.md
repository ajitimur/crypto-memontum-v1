# crypto-memontum-v1

Cross-sectional momentum research on crypto assets. The research invariants are
in `CLAUDE.md`, the vocabulary in `CONTEXT.md`, the protocol in
`docs/agents/quant-research.md`, and the decisions in `docs/adr/`.

## Running

```sh
uv sync
uv run pytest
uv run momentum run configs/skeleton-btcusdt-2021q1.toml
```

`run` fetches any months of the window not already in `data/raw/`, verifies each
download against the archive's published SHA256, rebuilds `data/derived/`,
simulates, writes the result to `results/<commit>/<config-name>.json`, and
appends one line to `trials.jsonl`.

```sh
uv run momentum grid configs/xsec-grid-2021h1.toml                  # all 21 cells, one invocation
uv run momentum gate configs/gate-faithful.toml configs/gate-venue.toml
uv run momentum build-derived configs/skeleton-btcusdt-2021q1.toml  # rebuild without fetching
uv run momentum pull-cmc-panel                                      # the one-time market-cap pull
uv run momentum trials                                              # every configuration tried
```

## The Replication Gate

`momentum gate` runs both halves of ADR-0003's Step 1 and states an explicit
pass or fail against tolerances fixed before either ran.

The **Faithful Run** (`configs/gate-faithful.toml`) prices off the CoinMarketCap
panel over Han, Kang and Ryu's own 2017-01-01 to 2023-08-28 window, so vendor
differences are eliminated as an explanation for disagreement. It tests whether
our pipeline is correct, and it is the only one held to the level. The **Venue
Run** (`configs/gate-venue.toml`) prices off the Binance archive from its
2017-08-17 floor — stated in the result, not footnoted — and tests whether the
published effect survives contact with executable prices.

The four criteria are in `src/crypto_momentum/gate.py` and the published table
they are read against is transcribed in `src/crypto_momentum/sim/published.py`
beside its citation. The verdict lands at `results/<commit>/gate.json`; a
failing gate exits 3, which is a finding and not a fault — ADR-0003 expects the
gate to be hard to pass.

**The gap between the two runs is reported as a result in its own right.** It
measures how much of the published effect is an artefact of cross-exchange
aggregate pricing versus what could actually have been traded on one venue.
Nobody in the surveyed literature reports it.

Three things stand between the apparatus and a verdict anyone should quote, each
recorded where it bites rather than only here:

- **The stored vendor panel is weekly.** ADR-0008's one-time pull runs at
  `--interval 7d`, and `data/cmc_prices.py` refuses a panel too coarse to mark
  daily rather than resampling it. The Faithful Run needs a daily re-pull.
- **ADR-0003 and ADR-0004 disagree about which published leg to bind to.**
  ADR-0003 fixes its tolerances on the long-short leg; ADR-0004 makes this repo
  long-only and notes an unlevered long-only book cannot liquidate. The gate
  runs the comparison ADR-0003 specifies, warns in the verdict, and takes
  `--reference-leg long_only` for the like-for-like reading.
- **ADR-0007's turnover ceiling is a deployment constraint.** ADR-0003 is
  explicit that Step 1 is a strategy we do not intend to trade, so the grid's
  short holding periods may be refused for breaching a ceiling that was never
  meant to apply to a replication run.

## The Grid

A config whose `[strategy]` names a `grid` instead of a `lookback_days` and a
`holding_days` is 21 runs, not one, and `momentum grid` runs all of them. The 21
pairs are Han, Kang and Ryu's own, in `src/crypto_momentum/sim/grid.py` beside
the citation; a config chooses between published grids by name and cannot
compose one, for the reason `costs.model` is a name rather than a number of
basis points.

Each cell is filed at
`results/<commit>/<grid-config-name>/<grid-config-name>-l<j>-h<k>.json` with the
grid's own summary at `grid.json` beside them, and each appends its own line to
`trials.jsonl`. **21 cells count as 21 configurations tried, not one** — that is
the multiple testing the reporting protocol asks to be counted, and the reason
Han et al. hold their own grid to `t > 3.0`.

The grid reads the archive, the point-in-time Universe and the vendor panel once
for all 21 cells, so the cells differ in their two knobs and in nothing else. A
cell that breaches its turnover budget, or whose lookback the window is too short
for, is recorded as refused and the grid carries on: that is a finding about the
cell, and stopping would leave the remaining configurations both unrun and
uncounted. A fault in the *data* is not caught — it would fail all 21 identically,
and twenty-one refusals for one missing file would read as a result.

## Layout

| Path | What it is |
| --- | --- |
| `data/raw/` | Archive files exactly as fetched, plus a manifest per file. Append-only: re-fetching a stored window raises. Gitignored. |
| `data/derived/` | Bars rebuilt from raw by `momentum build-derived`. Disposable — delete it and the next run rebuilds it. Gitignored. |
| `results/` | One JSON result per `(commit, config)`; a Grid's cells sit together under the grid config's own directory, with `grid.json` summarising the shape. Gitignored. |
| `trials.jsonl` | Every run ever made, appended, git-tracked. The count of configurations tried that every reported result has to quote. |
| `configs/` | Run configs and `vendor-symbol-map.toml`. Inert TOML — the loader validates and cannot execute. |
| `scripts/pull_cmc_panel.R` | The one-time `crypto2` pull. Needs R; run once, never again. Both window bounds are required — it will not read the clock. |
| `src/crypto_momentum/sim/` | The simulation core. No network, no filesystem, no clock; a test asserts it. |

## Data source

`data.binance.vision`, spot, monthly kline partitions (ADR-0008). Symbols are
concatenated base+quote uppercase (`BTCUSDT`); bars are indexed on their UTC
`open_time`, so a `1d` bar stamped `2021-01-01` covers that whole UTC day. The
archive switched its timestamp unit from milliseconds to microseconds in 2025;
the adapter reads both, and a recorded fixture from each era pins it.

## Market caps and the symbol mapping

Value weighting, the cap-weighted market portfolio and Churn need market
capitalisation, which the Binance archive does not publish. ADR-0008 buys it
with a single immutable `crypto2` pull of CoinMarketCap's survivorship-free
`listings/historical` endpoint, stored as one checksummed CSV under
`data/raw/coinmarketcap/`.

`pull-cmc-panel` is a no-op once that file exists — it does not re-fetch, which
is the whole discipline, since the route uses undocumented endpoints and every
extra call risks an IP ban. A panel that does not list assets known to have died
is rejected before it reaches `data/raw/`, because a survivorship-biased panel
looks entirely normal in aggregate. Running the pull needs R with `crypto2`
installed; nothing else in the repo does.

Joining the two vendors needs a symbol mapping, and ADR-0008 names it the known
bug surface. It is time-varying by construction: a CoinMarketCap id is
permanent, a Binance base is a name Binance reuses. Binance reassigned
`LUNAUSDT` to Terra 2.0 on 2022-05-31 while the original chain became `LUNC`, so
there is no answer to "what is LUNA" — only to "what was LUNA on this date".
`vendor_symbol_map` is the entry point: it derives most links from the panel's
own renames, then applies `configs/vendor-symbol-map.toml` on top. The table
earns its place because the vendor's snapshot grid is not the venue's cutover
date — CoinMarketCap moves id 4172 to LUNC on its own schedule, Binance renamed
on 2022-05-31, and the bar that fills an order is a Binance bar. A mapping where
one base means two assets on one day raises rather than resolving itself, and
assets on one vendor and not the other are reported as unmatched rather than
dropped.
