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
uv run momentum build-derived configs/skeleton-btcusdt-2021q1.toml  # rebuild without fetching
uv run momentum pull-cmc-panel                                      # the one-time market-cap pull
uv run momentum trials                                              # every configuration tried
```

## Layout

| Path | What it is |
| --- | --- |
| `data/raw/` | Archive files exactly as fetched, plus a manifest per file. Append-only: re-fetching a stored window raises. Gitignored. |
| `data/derived/` | Bars rebuilt from raw by `momentum build-derived`. Disposable — delete it and the next run rebuilds it. Gitignored. |
| `results/` | One JSON result per `(commit, config)`. Gitignored. |
| `trials.jsonl` | Every run ever made, appended, git-tracked. The count of configurations tried that every reported result has to quote. |
| `configs/` | Run configs and `vendor-symbol-map.toml`. Inert TOML — the loader validates and cannot execute. |
| `scripts/pull_cmc_panel.R` | The one-time `crypto2` pull. Needs R; run once, never again. |
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
`build_symbol_map` derives most links from the panel's own renames and refuses
to build a mapping where one base means two assets on one day; the hand-resolved
cases live in `configs/vendor-symbol-map.toml`. Assets on one vendor and not the
other are reported as unmatched rather than dropped.
