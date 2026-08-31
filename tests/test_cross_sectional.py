"""The cross-sectional strategy: signal, selection, weighting, and the path.

The fixtures here are hand-built rather than recorded, because every assertion is
about a number we can work out by hand. A frame whose last row is the Decision
Bar is the one that matters most: it is how a lookahead gets caught.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_momentum.sim.cross_sectional import (
    MIN_UNIVERSE,
    NotEnoughHistory,
    SelectionError,
    market_caps_before,
    past_return,
    select_top_quantile,
    simulate_cross_sectional,
    value_weights,
)


def dates(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC", name="ts_utc")


def frame(index: pd.DatetimeIndex, columns: dict[str, list[float]]) -> pd.DataFrame:
    built = pd.DataFrame(columns, index=index)
    built.columns.name = "symbol"
    return built


def ramp(index: pd.DatetimeIndex, *, start: float, daily: float) -> list[float]:
    """A price series compounding at `daily` per day from `start`."""
    return [start * (1.0 + daily) ** step for step in range(len(index))]


def bars_from_closes(closes: pd.DataFrame, *, volume: float = 1_000.0) -> dict:
    """Daily bars whose open is the previous close, so a fill price is checkable."""
    bars = {}
    for symbol in closes.columns:
        close = closes[symbol]
        opens = close.shift(1)
        opens.iloc[0] = close.iloc[0]
        bars[symbol] = pd.DataFrame(
            {
                "open": opens,
                "high": close,
                "low": close,
                "close": close,
                "volume": pd.Series(volume, index=close.index),
            }
        )
    return bars


def all_tradeable(closes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=closes.index, columns=closes.columns)


def flat_caps(closes: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    """Weekly market-cap snapshots that do not move, so weights are hand-checkable."""
    snapshots = closes.index[::7]
    return pd.DataFrame(
        {symbol: [cap] * len(snapshots) for symbol, cap in caps.items()},
        index=snapshots,
    )


# A ten-name cross-section: a quintile of it is two names, which is small enough
# to work the arithmetic through by hand and wide enough to be a real ranking.
TEN = [f"A{index}USDT" for index in range(10)]


def ten_name_market(n_days: int = 60) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    index = dates("2021-01-01", n_days)
    # A0 climbs fastest, A9 slowest, so the ranking is known before any code runs.
    closes = frame(
        index,
        {
            symbol: ramp(index, start=100.0, daily=0.010 - 0.001 * position)
            for position, symbol in enumerate(TEN)
        },
    )
    caps = flat_caps(closes, {symbol: 1_000.0 * (position + 1) for position, symbol in enumerate(TEN)})
    return closes, bars_from_closes(closes), all_tradeable(closes), caps


class TestPastReturn:
    def test_reads_only_bars_before_the_decision_bar(self):
        """The lookahead test the research protocol asks for, at the feature."""
        index = dates("2021-01-01", 11)
        closes = frame(index, {"AUSDT": ramp(index, start=100.0, daily=0.01)})
        decision_ts = index[-1]

        honest = past_return(closes, decision_ts=decision_ts, lookback_days=5)

        # The Decision Bar's own close is replaced by a number no return could
        # survive. A feature that reads it moves; one that does not, does not.
        with_a_spike = closes.copy()
        with_a_spike.loc[decision_ts, "AUSDT"] = 1e9

        assert past_return(
            with_a_spike, decision_ts=decision_ts, lookback_days=5
        ).equals(honest)

    def test_is_the_return_over_the_lookback_ending_the_bar_before(self):
        index = dates("2021-01-01", 11)
        closes = frame(index, {"AUSDT": [100.0] * 4 + [200.0] + [400.0] * 6})
        # The Decision Bar is index[6]; the formation bar is index[5] at 400, and
        # five days earlier is index[0] at 100.
        signal = past_return(closes, decision_ts=index[6], lookback_days=5)

        assert signal["AUSDT"] == pytest.approx(3.0)

    def test_a_lookback_reaching_before_the_frame_is_refused(self):
        index = dates("2021-01-01", 11)
        closes = frame(index, {"AUSDT": ramp(index, start=100.0, daily=0.01)})

        with pytest.raises(NotEnoughHistory):
            past_return(closes, decision_ts=index[3], lookback_days=5)

    def test_a_symbol_missing_either_end_of_the_lookback_is_unranked(self):
        index = dates("2021-01-01", 11)
        closes = frame(
            index,
            {
                "AUSDT": ramp(index, start=100.0, daily=0.01),
                "BUSDT": [np.nan] * 3 + ramp(index[3:], start=100.0, daily=0.01),
            },
        )

        signal = past_return(closes, decision_ts=index[6], lookback_days=5)

        assert np.isnan(signal["BUSDT"])
        assert not np.isnan(signal["AUSDT"])

    def test_a_zero_price_at_the_start_of_the_lookback_is_unranked(self):
        """A return divided by nothing is not a large return; it is not one at all."""
        index = dates("2021-01-01", 11)
        closes = frame(index, {"AUSDT": [0.0] + ramp(index[1:], start=100.0, daily=0.01)})

        signal = past_return(closes, decision_ts=index[6], lookback_days=5)

        assert np.isnan(signal["AUSDT"])


class TestSelection:
    def test_takes_the_top_fifth_rounded_up(self):
        signal = pd.Series({f"A{index}": float(index) for index in range(10)})

        assert select_top_quantile(signal, quantile=0.2) == ("A9", "A8")

    def test_a_cross_section_that_does_not_divide_evenly_rounds_up(self):
        """Seven names at a fifth is 1.4 — two, so the quintile is never empty."""
        signal = pd.Series({f"A{index}": float(index) for index in range(7)})

        assert select_top_quantile(signal, quantile=0.2) == ("A6", "A5")

    def test_unranked_symbols_are_not_selected(self):
        signal = pd.Series({"A": np.nan, "B": 0.1, "C": 0.2, "D": 0.3, "E": 0.4})

        assert "A" not in select_top_quantile(signal, quantile=0.2)

    def test_ties_break_on_the_symbol_so_a_run_is_reproducible(self):
        signal = pd.Series({"BUSDT": 1.0, "AUSDT": 1.0, "CUSDT": 0.0})

        assert select_top_quantile(signal, quantile=0.33) == ("AUSDT",)

    def test_a_quantile_outside_the_unit_interval_is_refused(self):
        signal = pd.Series({"AUSDT": 1.0})

        with pytest.raises(SelectionError):
            select_top_quantile(signal, quantile=1.5)


class TestValueWeights:
    def test_are_market_cap_shares_summing_to_one(self):
        weights = value_weights(pd.Series({"AUSDT": 300.0, "BUSDT": 100.0}))

        assert weights["AUSDT"] == pytest.approx(0.75)
        assert weights["BUSDT"] == pytest.approx(0.25)
        assert weights.sum() == pytest.approx(1.0)

    def test_are_never_negative_and_never_levered(self):
        weights = value_weights(pd.Series({"AUSDT": 1.0, "BUSDT": 2.0, "CUSDT": 7.0}))

        assert (weights >= 0.0).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_a_cross_section_with_no_capitalisation_is_refused(self):
        with pytest.raises(SelectionError):
            value_weights(pd.Series({"AUSDT": 0.0, "BUSDT": 0.0}))


class TestMarketCapsBeforeTheDecisionBar:
    def test_reads_the_last_snapshot_strictly_before_the_decision_bar(self):
        snapshots = pd.date_range("2021-01-01", periods=3, freq="7D", tz="UTC")
        caps = pd.DataFrame({"AUSDT": [10.0, 20.0, 30.0]}, index=snapshots)

        # The Decision Bar falls on the third snapshot itself: that snapshot is
        # timestamped at the Decision Bar, so it is not available to form on.
        read = market_caps_before(caps, decision_ts=snapshots[2], max_staleness_days=14)

        assert read["AUSDT"] == pytest.approx(20.0)

    def test_a_snapshot_older_than_the_staleness_bound_is_unusable(self):
        snapshots = pd.date_range("2021-01-01", periods=2, freq="7D", tz="UTC")
        caps = pd.DataFrame({"AUSDT": [10.0, 20.0]}, index=snapshots)

        read = market_caps_before(
            caps, decision_ts=pd.Timestamp("2021-03-01T00:00:00Z"), max_staleness_days=14
        )

        assert np.isnan(read["AUSDT"])


class TestCrossSectionalRun:
    def test_rebalances_weekly_and_fills_at_the_next_bars_open(self):
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        gaps = {
            (later.decision_ts_utc - earlier.decision_ts_utc).days
            for earlier, later in zip(run.selections, run.selections[1:])
        }
        assert gaps == {7}
        for selection in run.selections:
            assert selection.entry_ts_utc == selection.decision_ts_utc + pd.Timedelta(days=1)

    def test_holds_the_top_quintile_value_weighted(self):
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        first = run.selections[0]
        # A0 and A1 compound fastest, so they are the quintile; their caps are
        # 1000 and 2000, so the value weights are a third and two thirds.
        assert first.symbols == ("A0USDT", "A1USDT")
        assert first.weights["A1USDT"] == pytest.approx(2.0 / 3.0)
        assert first.weights["A0USDT"] == pytest.approx(1.0 / 3.0)

    def test_is_long_only_and_never_levered(self):
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=40.44,
        )

        for selection in run.selections:
            assert all(weight >= 0.0 for weight in selection.weights.values())
            assert sum(selection.weights.values()) <= 1.0 + 1e-12
        assert run.max_gross_exposure <= 1.0 + 1e-9

    def test_a_spike_on_the_decision_bar_cannot_change_what_is_held(self):
        """The lookahead test again, at the whole strategy rather than the feature."""
        closes, bars, tradeable, caps = ten_name_market()
        honest = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        # The worst name in the cross-section prints a moonshot close on every
        # Decision Bar. A signal reading the Decision Bar would buy it.
        spiked = {symbol: bars[symbol].copy() for symbol in bars}
        for selection in honest.selections:
            spiked["A9USDT"].loc[selection.decision_ts_utc, "close"] = 1e9

        run = simulate_cross_sectional(
            spiked,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        assert [selection.symbols for selection in run.selections] == [
            selection.symbols for selection in honest.selections
        ]

    def test_marks_every_day_of_every_holding_period(self):
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        marks = run.result.equity_net.index
        assert marks[0] == run.selections[0].entry_ts_utc
        assert (marks.to_series().diff().iloc[1:] == pd.Timedelta(days=1)).all()

    def test_an_asset_untradeable_on_the_decision_bar_is_not_selected(self):
        closes, bars, tradeable, caps = ten_name_market()
        # The policy panel keeps the best name out for the whole run: this is the
        # universe layer and the strategy meeting.
        tradeable = tradeable.copy()
        tradeable["A0USDT"] = False

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        assert all("A0USDT" not in selection.symbols for selection in run.selections)
        assert run.selections[0].symbols == ("A1USDT", "A2USDT")

    def test_a_cross_section_too_thin_to_have_a_quintile_holds_cash(self):
        index = dates("2021-01-01", 40)
        closes = frame(
            index,
            {
                symbol: ramp(index, start=100.0, daily=0.01 - 0.001 * position)
                for position, symbol in enumerate(TEN[: MIN_UNIVERSE - 1])
            },
        )
        caps = flat_caps(closes, {symbol: 1_000.0 for symbol in closes.columns})

        run = simulate_cross_sectional(
            bars_from_closes(closes),
            tradeable=all_tradeable(closes),
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        assert all(selection.symbols == () for selection in run.selections)
        assert run.result.net_return == pytest.approx(0.0)
        assert run.n_rebalances_held_cash == len(run.selections)

    def test_an_asset_that_halts_mid_holding_period_exits_at_its_last_price(self):
        closes, bars, tradeable, caps = ten_name_market()
        halt_ts = pd.Timestamp("2021-01-25T00:00:00Z")
        # The best name goes dark: no volume, so no order could have filled. Its
        # closes keep printing afterwards, and none of them may be earned.
        halted = {symbol: bars[symbol].copy() for symbol in bars}
        after_halt = halted["A0USDT"].index >= halt_ts
        halted["A0USDT"].loc[after_halt, "volume"] = 0.0
        halted["A0USDT"].loc[after_halt, "close"] *= 10.0

        run = simulate_cross_sectional(
            halted,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        exited = [event for event in run.halt_exits if event.symbol == "A0USDT"]
        assert exited, "the halted asset should have been exited"
        assert exited[0].exit_ts_utc == halt_ts
        assert exited[0].exit_price == pytest.approx(
            float(bars["A0USDT"]["close"].loc[halt_ts - pd.Timedelta(days=1)])
        )
        # The tenfold print after the halt is a price nobody could have sold
        # into, so no daily mark may show it being earned.
        assert run.result.equity_net.pct_change().max() < 0.5

    def test_costs_are_charged_on_both_legs_of_every_rebalance(self):
        closes, bars, tradeable, caps = ten_name_market()
        common = dict(
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
        )

        free = simulate_cross_sectional(bars, cost_bps_per_side=0.0, **common)
        charged = simulate_cross_sectional(bars, cost_bps_per_side=40.44, **common)

        assert charged.result.net_return < free.result.net_return
        # Gross is the same path without costs, so Cost Drag is the whole gap.
        assert charged.result.gross_return == pytest.approx(free.result.net_return)
        assert charged.result.cost_drag_annualised > 0.0

    def test_turnover_is_reported_per_rebalance(self):
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        # The first rebalance buys from an empty book, so it turns over the lot.
        assert run.selections[0].turnover == pytest.approx(1.0)
        assert 0.0 <= run.mean_rebalance_turnover <= 1.0

    def test_a_window_with_no_room_for_a_holding_period_is_refused(self):
        index = dates("2021-01-01", 16)
        closes = frame(
            index,
            {
                symbol: ramp(index, start=100.0, daily=0.01)
                for symbol in TEN
            },
        )

        with pytest.raises(NotEnoughHistory):
            simulate_cross_sectional(
                bars_from_closes(closes),
                tradeable=all_tradeable(closes),
                market_caps=flat_caps(closes, {symbol: 1.0 for symbol in TEN}),
                lookback_days=30,
                holding_days=7,
                cost_bps_per_side=0.0,
            )


class TestTheFillBoundary:
    def test_an_asset_that_halts_during_the_fill_session_is_still_bought(self):
        """Whether today trades is not knowable at today's open.

        Gating the fill on the fill bar's own volume would be a decision made
        with the session's outcome in hand. The honest path is to buy at the
        open, discover the halt when the session closes, and exit at the price
        actually paid — which is the last one anyone transacted at.
        """
        closes, bars, tradeable, caps = ten_name_market()
        fill_ts = pd.Timestamp("2021-01-17T00:00:00Z")
        halted = {symbol: bars[symbol].copy() for symbol in bars}
        halted["A0USDT"].loc[fill_ts, "volume"] = 0.0

        run = simulate_cross_sectional(
            halted,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        first = run.selections[0]
        assert first.entry_ts_utc == fill_ts
        assert "A0USDT" in first.symbols, "the open was there, so the order fills"
        assert first.unfilled == ()

        exited = [event for event in run.halt_exits if event.symbol == "A0USDT"]
        assert exited[0].exit_ts_utc == fill_ts
        assert exited[0].exit_price == pytest.approx(
            float(bars["A0USDT"]["open"].loc[fill_ts])
        )

    def test_an_asset_with_no_opening_price_is_not_bought(self):
        closes, bars, tradeable, caps = ten_name_market()
        fill_ts = pd.Timestamp("2021-01-17T00:00:00Z")
        gapped = {symbol: bars[symbol].copy() for symbol in bars}
        gapped["A0USDT"].loc[fill_ts, "open"] = np.nan

        run = simulate_cross_sectional(
            gapped,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        first = run.selections[0]
        assert "A0USDT" not in first.symbols
        assert first.unfilled == ("A0USDT",)
        # Its weight stays in cash rather than being spread over the rest.
        assert sum(first.weights.values()) < 1.0


class TestExposure:
    def test_the_closing_day_is_not_averaged_in_as_a_flat_one(self):
        """The book is closed on the last bar, but it was held through it."""
        closes, bars, tradeable, caps = ten_name_market()

        run = simulate_cross_sectional(
            bars,
            tradeable=tradeable,
            market_caps=caps,
            lookback_days=14,
            holding_days=7,
            cost_bps_per_side=0.0,
        )

        assert run.exposure_gross.iloc[-1] == pytest.approx(1.0)
        assert run.max_gross_exposure == pytest.approx(1.0)
        # Fully invested every day of the run, so the average is one too.
        assert run.mean_gross_exposure == pytest.approx(1.0)
