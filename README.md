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
uv run momentum trials                                              # every configuration tried
```

## Layout

| Path | What it is |
| --- | --- |
| `data/raw/` | Archive files exactly as fetched, plus a manifest per file. Append-only: re-fetching a stored window raises. Gitignored. |
| `data/derived/` | Bars rebuilt from raw by `momentum build-derived`. Disposable — delete it and the next run rebuilds it. Gitignored. |
| `results/` | One JSON result per `(commit, config)`. Gitignored. |
| `trials.jsonl` | Every run ever made, appended, git-tracked. The count of configurations tried that every reported result has to quote. |
| `configs/` | Run configs. Inert TOML — the loader validates and cannot execute. |
| `src/crypto_momentum/sim/` | The simulation core. No network, no filesystem, no clock; a test asserts it. |

## Data source

`data.binance.vision`, spot, monthly kline partitions (ADR-0008). Symbols are
concatenated base+quote uppercase (`BTCUSDT`); bars are indexed on their UTC
`open_time`, so a `1d` bar stamped `2021-01-01` covers that whole UTC day. The
archive switched its timestamp unit from milliseconds to microseconds in 2025;
the adapter reads both, and a recorded fixture from each era pins it.
