"""The two cost models, and the weekly footing the turnover ceiling is stated on.

Every expected figure here is worked out from ADR-0007's own arithmetic rather
than copied from what the code returns: 0.15 + 0.21 + 0.0444 is a claim about
Indonesian tax law, and a snapshot test would let a typo in it pass forever.
"""

import pytest

from crypto_momentum.costs import (
    PAPER,
    TOKOCRYPTO,
    TURNOVER_CEILING_WEEKLY,
    CostModel,
    CostModelError,
    cost_model,
    weekly_turnover,
)


class TestTheTwoModels:
    def test_the_paper_charges_fifteen_basis_points_a_trade(self):
        # Han et al.: 15bp per trade, so 0.30% per round trip.
        assert PAPER.bps_per_side == pytest.approx(15.0)
        assert PAPER.round_trip_bps == pytest.approx(30.0)
        assert PAPER.tax_bps_per_side == 0.0

    def test_tokocrypto_stacks_fee_tax_and_levy_to_forty_point_four_four(self):
        # 0.15% fee + 0.21% PPh + 0.0444% levy = 0.4044% per side, and 0.8088%
        # per round trip — ADR-0007's headline comparison against the paper.
        assert TOKOCRYPTO.fee_bps_per_side == pytest.approx(15.0)
        assert TOKOCRYPTO.tax_bps_per_side == pytest.approx(21.0)
        assert TOKOCRYPTO.levy_bps_per_side == pytest.approx(4.44)
        assert TOKOCRYPTO.bps_per_side == pytest.approx(40.44)
        assert TOKOCRYPTO.round_trip_bps == pytest.approx(80.88)

    def test_tokocryptos_tax_falls_on_the_buy_leg_too(self):
        # Pasal 11(2)(b): a USDT-quoted swap is a disposal on both sides. This is
        # the fact that makes the model 2.7x the literature's rather than 1.4x.
        assert TOKOCRYPTO.tax_charged_on_buys is True

    def test_tokocrypto_is_roughly_two_and_a_half_times_the_paper(self):
        ratio = TOKOCRYPTO.round_trip_bps / PAPER.round_trip_bps
        assert ratio == pytest.approx(2.696, abs=0.001)

    def test_slippage_adds_on_top_of_the_venues_own_cost(self):
        assert TOKOCRYPTO.total_bps_per_side(
            slippage_bps_per_side=5.0
        ) == pytest.approx(45.44)

    def test_neither_model_carries_a_funding_leg(self):
        # v1 is unlevered long-only spot per ADR-0004, so there is no perpetual
        # position to fund. Recorded as an explicit zero rather than an omission.
        for model in (PAPER, TOKOCRYPTO):
            assert model.to_metadata()["funding_bps"] == 0.0

    def test_a_model_records_where_its_numbers_came_from(self):
        assert "ADR-0007" in TOKOCRYPTO.to_metadata()["source"]


class TestLookingAModelUp:
    def test_a_model_is_found_by_the_name_a_config_gives_it(self):
        assert cost_model("tokocrypto") is TOKOCRYPTO
        assert cost_model("paper") is PAPER

    def test_an_unknown_model_names_the_ones_that_exist(self):
        with pytest.raises(CostModelError, match="paper, tokocrypto"):
            cost_model("binance")


class TestAModelTheSimulatorCannotPrice:
    def test_a_sell_only_tax_is_refused_rather_than_charged_on_both_legs(self):
        # An IDR-quoted book is taxed on the disposal alone. The walk charges one
        # rate on both legs, so such a model has to be refused here — otherwise
        # every buy silently pays a tax nobody owes.
        with pytest.raises(CostModelError, match="sell leg alone"):
            CostModel(
                name="indodax",
                fee_bps_per_side=15.0,
                tax_bps_per_side=21.0,
                levy_bps_per_side=2.22,
                tax_charged_on_buys=False,
                source="hypothetical IDR-quoted book",
            )

    def test_a_negative_component_is_refused(self):
        with pytest.raises(CostModelError, match="cannot be negative"):
            CostModel(
                name="rebate",
                fee_bps_per_side=-1.0,
                tax_bps_per_side=0.0,
                levy_bps_per_side=0.0,
                tax_charged_on_buys=False,
                source="a maker rebate is not a cost model",
            )

    def test_negative_slippage_is_refused(self):
        with pytest.raises(CostModelError, match="slippage cannot be negative"):
            TOKOCRYPTO.total_bps_per_side(slippage_bps_per_side=-1.0)


class TestTheWeeklyFooting:
    def test_a_weekly_rebalance_is_its_own_weekly_turnover(self):
        assert weekly_turnover(0.2, holding_days=7) == pytest.approx(0.2)

    def test_a_fortnightly_rebalance_halves_onto_the_weekly_footing(self):
        # 40% every fourteen days is 20% a week, which is inside the ceiling even
        # though the per-rebalance figure is well outside it.
        assert weekly_turnover(0.4, holding_days=14) == pytest.approx(0.2)
        assert weekly_turnover(0.4, holding_days=14) < TURNOVER_CEILING_WEEKLY

    def test_a_daily_rebalance_multiplies_onto_the_weekly_footing(self):
        # 10% a day is 70% a week — the literature's regime, and nowhere near the
        # ceiling, despite looking like the smallest number of the three.
        assert weekly_turnover(0.1, holding_days=1) == pytest.approx(0.7)
        assert weekly_turnover(0.1, holding_days=1) > TURNOVER_CEILING_WEEKLY

    def test_the_ceiling_is_the_twenty_five_percent_adr_0007_sets(self):
        assert TURNOVER_CEILING_WEEKLY == 0.25

    def test_a_holding_period_of_no_days_is_refused(self):
        with pytest.raises(CostModelError, match="holding_days"):
            weekly_turnover(0.2, holding_days=0)
