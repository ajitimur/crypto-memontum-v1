# ADR-0006: Exclude stablecoins and wrapped assets by design intent, permanently

- **Status:** Accepted
- **Date:** 2026-08-30
- **Related:** `CONTEXT.md` (Stablecoin, Wrapped Asset, Universe)

## Context

A pegged asset has near-zero momentum by construction and pollutes a cross-sectional ranking, so it has to come out of the Universe. Doing that without hindsight is harder than it sounds, and UST is the case that breaks the naive approaches.

UST was a stablecoin until May 2022, then depegged, collapsed, and rebounded violently. Excluding it by today's classification uses future information about a live 2021 asset. Including it during the collapse puts an enormous return series into the cross-section that is driven by a solvency event, not by the effect we are studying.

## Decision

Classify by **stated design intent, at the asset's listing date**, and exclude permanently. An asset designed to hold a peg is a Stablecoin for the whole sample, whether or not it ever broke that peg. The same rule, for a different reason, excludes Wrapped Assets — bridged and liquid-staked tokens — permanently.

The exclusion list is dated and versioned. New listings are classified at their listing date, never retroactively.

## Considered Options

**Exclude only while the peg held, include after a depeg.** Rejected. It sounds like the point-in-time-honest option and is the opposite: the depeg date is a judgement call made with hindsight, and the whole reason the asset becomes interesting is a fact we only know afterward.

**Exclude anything ever classified as a stablecoin, using today's labels.** Rejected as stated, though it reaches the same answer as our rule in almost every case. The reasoning matters: applying today's labels backwards *is* look-ahead, even when it is harmless. Design intent is different — it is knowable from the asset's own documentation on day one, so applying it across the whole sample uses no future information.

## Consequences

- A post-depeg stablecoin in the cross-section is a bet on catastrophic depegs. There have been roughly three. That is an unhedgeable lottery ticket, and one of them landing inside our sample would carry the entire result — the exact failure `docs/agents/quant-research.md` tells us to name before defending a strong number.
- Wrapped assets are excluded because they duplicate an asset already in the Universe, so including them double-counts a single bet rather than adding a position.
- The list is a hand-maintained artifact and therefore a place errors will hide. It gets versioned so a result can be traced to the list that produced it.
