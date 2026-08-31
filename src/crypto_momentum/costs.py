"""What a trade costs, and how much trading the budget allows.

ADR-0007 is the whole of this module. Two facts from it decide the shape:

**A cost is a structure, not a number.** Tokocrypto's 0.4044% per side is
0.15% venue fee + 0.21% PPh + 0.0444% exchange levy, and the middle term is a
tax rather than a fee. Collapsing the three into one literal in a config file
loses the only part a later reader would need — which component moved when the
number changes, and whether the change is ours to make. So a run names a model
and the model carries its parts.

**Turnover is budgeted rather than observed.** At 0.8088% per round trip, the
~68% weekly Rebalance Turnover the literature reports for these signals costs
28.6% a year. ADR-0007 therefore sets a hard ceiling of 25% weekly Rebalance
Turnover. The ceiling binds in two places, and both are needed:

- At load time a config may not *declare* a budget above the ceiling. That is
  the check ADR-0007 puts on the config loader, and it is what stops a run whose
  stated intent already breaches the ADR from ever starting.
- At run time the simulator holds the walk to the budget the config declared. A
  realised turnover is not knowable before the walk, so this is the only place
  it can be caught — but it is caught as a refusal, not filed as a result with a
  caveat attached. See `sim.cross_sectional`.

There is deliberately no funding model. v1 is unlevered long-only spot per
ADR-0004 and holds no perpetual position, so there is no funding leg to pay. A
zero-valued funding rate sitting in a config would read like a modelled
assumption rather than the absence of one.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4

# ADR-0007's hard ceiling on weekly Rebalance Turnover. A config may budget less
# than this and often should; it may not budget more.
TURNOVER_CEILING_WEEKLY = 0.25

# The ceiling is quoted per week, so a config that rebalances on any other cadence
# has its turnover put on a weekly footing before the comparison is made.
DAYS_PER_WEEK = 7


class CostModelError(Exception):
    """A cost model was named that does not exist, or does not add up."""


@dataclass(frozen=True)
class CostModel:
    """One venue's cost of transacting, per side, broken into its components.

    Every component is in basis points of the value traded on that side. They are
    kept apart rather than pre-summed because they answer to different things: the
    fee is the venue's tier, the tax is PMK 50/2025, and the levy is the exchange
    association's. A change to one is a different kind of event from a change to
    another.

    `tax_charged_on_buys` records Pasal 11(2)(b): a USDT-quoted swap is a
    *penyerahan* on both sides, so the disposal tax lands on the buy leg too. It
    is not decoration — the simulator charges one rate on both legs, so a model
    whose tax falls on the sell alone would be silently overcharged by it. Such a
    model is refused here rather than mispriced there.
    """

    name: str
    fee_bps_per_side: float
    tax_bps_per_side: float
    levy_bps_per_side: float
    tax_charged_on_buys: bool
    source: str

    def __post_init__(self) -> None:
        for label, value in (
            ("fee_bps_per_side", self.fee_bps_per_side),
            ("tax_bps_per_side", self.tax_bps_per_side),
            ("levy_bps_per_side", self.levy_bps_per_side),
        ):
            if value < 0.0:
                raise CostModelError(
                    f"{self.name}: {label} cannot be negative, got {value}"
                )
        if self.tax_bps_per_side > 0.0 and not self.tax_charged_on_buys:
            raise CostModelError(
                f"{self.name}: a tax charged on the sell leg alone cannot be "
                "simulated, because the walk charges one rate on both legs and "
                "would overcharge every buy. Model the asymmetry in the walk "
                "before adding such a venue — an IDR-quoted book is the case."
            )

    @property
    def bps_per_side(self) -> float:
        """The whole cost of one side, in basis points. Charged on buys and sells."""
        return self.fee_bps_per_side + self.tax_bps_per_side + self.levy_bps_per_side

    @property
    def round_trip_bps(self) -> float:
        """In and back out again — the figure ADR-0007's comparison table quotes."""
        return 2.0 * self.bps_per_side

    def total_bps_per_side(self, *, slippage_bps_per_side: float) -> float:
        """The model's own cost plus the run's slippage assumption.

        Slippage stays out of the model because it is not the venue's: it is an
        assumption about our own size against the book, and the research
        invariant is that it is stated alongside the number rather than baked
        into one.
        """
        if slippage_bps_per_side < 0.0:
            raise CostModelError(
                f"slippage cannot be negative, got {slippage_bps_per_side}"
            )
        return self.bps_per_side + slippage_bps_per_side

    def to_metadata(self) -> dict[str, float | str | bool]:
        """What a result records about what it paid to trade."""
        return {
            "cost_model": self.name,
            "fee_bps_per_side": self.fee_bps_per_side,
            "tax_bps_per_side": self.tax_bps_per_side,
            "levy_bps_per_side": self.levy_bps_per_side,
            "tax_charged_on_buys": self.tax_charged_on_buys,
            "bps_per_side": self.bps_per_side,
            "round_trip_bps": self.round_trip_bps,
            "source": self.source,
            # Said in words rather than as a rate. A `funding_bps: 0.0` here
            # would be the very thing this module's docstring argues against —
            # it reads as a funding model that happened to price at zero, when
            # what is true is that there is no funding leg to price.
            "funding": "none — unlevered long-only spot holds no perpetual position",
        }


# What Han, Kang and Ryu assume: 15bp a trade, no tax, no levy. Kept so a
# replication can be run against the literature's own cost world — quoting one of
# our numbers against it would be comparing to a market we cannot trade in.
PAPER = CostModel(
    name="paper",
    fee_bps_per_side=15.0,
    tax_bps_per_side=0.0,
    levy_bps_per_side=0.0,
    tax_charged_on_buys=False,
    source="Han, Kang and Ryu (2023): 15bp per trade, 0.30% per round trip",
)

# The venue we would actually trade, per ADR-0007. 0.15% + 0.21% + 0.0444%.
TOKOCRYPTO = CostModel(
    name="tokocrypto",
    fee_bps_per_side=15.0,
    tax_bps_per_side=21.0,
    levy_bps_per_side=4.44,
    tax_charged_on_buys=True,
    source="ADR-0007: Tokocrypto fee + PMK 50/2025 PPh + CFX levy, 0.4044% per side",
)

COST_MODELS: dict[str, CostModel] = {model.name: model for model in (PAPER, TOKOCRYPTO)}
COST_MODEL_NAMES = tuple(COST_MODELS)


def cost_model_named(name: str) -> CostModel:
    """Look a model up by the name a config gives it."""
    try:
        return COST_MODELS[name]
    except KeyError:
        raise CostModelError(
            f"unknown cost model {name!r}; the models are "
            f"{', '.join(COST_MODEL_NAMES)}"
        ) from None


def weekly_turnover(mean_rebalance_turnover: float, *, holding_days: int) -> float:
    """One-way Rebalance Turnover put on the weekly footing the ceiling uses.

    A rebalance turns over `mean_rebalance_turnover` of the book every
    `holding_days` days. ADR-0007 states its ceiling per week, so a fortnightly
    config trading 40% at each rebalance is at 20% weekly and inside the ceiling,
    while a daily one trading 10% is at 70% and nowhere near it. Comparing the
    per-rebalance figure to a weekly ceiling would rank those two backwards.
    """
    if holding_days < 1:
        raise CostModelError(f"holding_days must be at least 1, got {holding_days}")
    return mean_rebalance_turnover * DAYS_PER_WEEK / holding_days
