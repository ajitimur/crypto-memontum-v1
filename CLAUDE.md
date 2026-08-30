# crypto-memontum-v1

Cross-sectional momentum research on crypto assets. Python.

## Research invariants

These bind every signal, backtest, and result in this repo.

- **Point-in-time.** Compute a signal only from data timestamped strictly before the decision bar. A signal formed on bar `t` fills at `t+1` open, at the earliest.
- **Point-in-time universe.** Build the tradeable universe as of each rebalance date, including assets later delisted, depegged, or unwound. A universe drawn from today's exchange listing carries survivorship bias.
- **Net.** Every reported number is net of fees, funding, and a slippage assumption. State the assumption alongside the number.
- **Out-of-sample.** Hold a final period out and leave it untouched while iterating. Report how many configurations were tried before quoting the result.
- **Reproducible.** Every result traces to a commit hash plus a config file. Seed anything stochastic; treat raw data pulls as immutable and write derived data to a separate path.

Full backtest and reporting protocol: `docs/agents/quant-research.md`.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `ajitimur/crypto-memontum-v1`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
