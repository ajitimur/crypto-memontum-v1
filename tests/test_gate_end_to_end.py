"""The Replication Gate end to end: two Grids, two verdicts, and the gap.

As with the Grid suite, no cell's number means anything here — the synthetic
prices were chosen. What is under test is that both halves run from one
invocation, that the verdict is stated rather than implied, that the gap between
the two runs is measured, and that the pair of configs is checked for being a
pair before 42 cells are spent finding out it was not.
"""

import json

import pytest

from crypto_momentum.cli import EXIT_GATE_FAILED, EXIT_REFUSED, main
from crypto_momentum.config import ConfigError
from crypto_momentum.data.cmc_prices import PanelGrainTooCoarse
from crypto_momentum.gate import FAITHFUL, VENUE
from crypto_momentum.runner import run_gate
from crypto_momentum.sim.published import LONG_ONLY, LONG_SHORT
from synthetic_cross_section import (  # noqa: F401
    RUN_AT,
    archive,
    daily_panel_workspace,
    workspace,
)

SYMBOLS = '["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]'


def config_text(*, name, price_source, start_month="2021-01", **overrides):
    fields = {
        "quantile": "0.2",
        "bracket": '"binance-full"',
        "model": '"paper"',
        "slippage": "0.0",
        "grid": '"han-kang-ryu-21"',
        **overrides,
    }
    return f"""
name = "{name}"

[data]
venue = "binance-spot"
price_source = "{price_source}"
symbols = {SYMBOLS}
interval = "1d"
start_month = "{start_month}"
end_month = "2021-03"

[strategy]
kind = "cross_sectional"
grid = {fields["grid"]}
quantile = {fields["quantile"]}

[universe]
bracket = {fields["bracket"]}

[costs]
model = {fields["model"]}
slippage_bps_per_side = {fields["slippage"]}
"""


@pytest.fixture
def configs(daily_panel_workspace):
    """The two halves of a gate, written where a real pair would live."""

    def write(**overrides):
        directory = daily_panel_workspace.repo_root / "configs"
        directory.mkdir(parents=True, exist_ok=True)
        faithful = directory / "gate-faithful.toml"
        venue = directory / "gate-venue.toml"
        faithful.write_text(
            config_text(
                name="gate-faithful",
                price_source="cmc-panel",
                **overrides.pop("faithful", {}),
            )
        )
        venue.write_text(
            config_text(
                name="gate-venue",
                price_source="binance-archive",
                **overrides.pop("venue", {}),
            )
        )
        return faithful, venue

    return write


@pytest.fixture
def outcome(daily_panel_workspace, configs, archive):
    faithful, venue = configs()
    return run_gate(
        faithful,
        venue,
        daily_panel_workspace,
        run_at_utc=RUN_AT,
        leg=LONG_ONLY,
        open_url=archive,
    )


class TestBothRuns:
    def test_one_invocation_runs_both_halves_of_the_gate(self, outcome):
        assert len(outcome.faithful_grid.cells) == 21
        assert len(outcome.venue_grid.cells) == 21

    def test_each_half_is_labelled_by_what_it_tests(self, outcome):
        assert outcome.faithful.run == FAITHFUL
        assert outcome.venue.run == VENUE

    def test_the_faithful_run_is_priced_off_the_vendor_panel(self, outcome):
        assert outcome.faithful_grid.config.price_source == "cmc-panel"
        assert "coinmarketcap" in outcome.faithful_grid.universe["universe"]["source"]

    def test_the_venue_run_is_priced_off_the_archive(self, outcome):
        assert outcome.venue_grid.config.price_source == "binance-archive"
        assert "binance.vision" in outcome.venue_grid.universe["universe"]["source"]

    def test_both_grids_are_judged_even_when_the_first_one_fails(self, outcome):
        """Stopping on a failure would leave the gap unmeasured exactly when the
        two runs disagree, which is when it is most interesting."""
        assert outcome.faithful.criteria
        assert outcome.venue.criteria


class TestVerdict:
    def test_the_gate_states_a_pass_or_a_fail_and_never_a_shrug(self, outcome):
        assert outcome.passes in (True, False)
        assert outcome.record.to_dict()["passes"] in (True, False)

    def test_the_verdict_carries_all_four_criteria_for_each_run(self, outcome):
        for run in (outcome.record.faithful, outcome.record.venue):
            assert [criterion["criterion"] for criterion in run["criteria"]] == [
                "spearman_rank_correlation",
                "liquidation_count",
                "t_statistic_sign_agreement",
                "best_net_sharpe",
            ]

    def test_the_level_binds_on_the_faithful_run_only(self, outcome):
        faithful_level = outcome.faithful.criterion("best_net_sharpe")
        venue_level = outcome.venue.criterion("best_net_sharpe")

        assert faithful_level.required is not None
        assert venue_level.required is None

    def test_the_verdict_names_the_table_it_was_read_against(self, outcome):
        assert "Table 14" in outcome.record.faithful["citation"]

    def test_each_run_states_the_floor_its_price_source_imposes(self, outcome):
        """Issue #11 wants the archive floor "stated in the result rather than
        footnoted", and a comment in a config file is a footnote."""
        venue = outcome.record.windows[VENUE]
        faithful = outcome.record.windows[FAITHFUL]

        assert venue["price_source_floor_ts_utc"].startswith("2017-08-17")
        assert faithful["price_source_floor_ts_utc"].startswith("2013-04-28")

    def test_each_run_states_the_window_it_actually_covered(self, outcome):
        for window in outcome.record.windows.values():
            assert window["covered_start_ts_utc"]
            assert window["covered_end_ts_utc"]
            assert window["published_sample"] == "2017-01-01 to 2023-08-28"

    def test_each_run_states_what_its_net_figures_are_net_of(self, outcome):
        """A net Sharpe quoted without its cost assumption is not a result."""
        for costs in outcome.record.costs.values():
            assert costs["cost_model"] == "paper"
            assert costs["total_bps_per_side"] == 15.0
            assert costs["slippage_bps_per_side"] == 0.0

    def test_the_gate_is_filed_at_the_commit_s_root(self, outcome, daily_panel_workspace):
        path = (
            daily_panel_workspace.results_root
            / f"{outcome.record.commit}-dirty"
            / "gate.json"
        )

        assert path.exists()
        written = json.loads(path.read_text())
        assert written["faithful_config_name"] == "gate-faithful"
        assert written["venue_config_name"] == "gate-venue"


class TestTheGap:
    def test_the_gap_is_reported_as_a_result_in_its_own_right(self, outcome):
        gap = outcome.record.gap

        assert gap["signed"] == "venue minus faithful"
        assert "spearman_between_runs" in gap
        assert "best_net_sharpe_gap" in gap
        assert "liquidation_count_gap" in gap

    def test_the_gap_is_reported_cell_by_cell(self, outcome):
        assert len(outcome.record.gap["sharpe_gap_by_cell"]) == 21

    def test_two_runs_on_the_same_prices_rank_the_cells_the_same_way(self, outcome):
        """The synthetic panel carries the archive's own closes, so the only
        difference left between the runs is the Universe each one builds. The
        two should therefore agree on the *shape* of the grid; a low correlation
        here would be a wiring fault rather than a finding.

        Shape and not level, because the fixture's prices compound at a steady
        rate and so produce Sharpes in the hundreds. The levels are meaningless
        and their difference is meaningless with them — which is the same reason
        ADR-0003 binds the Venue Run on rank and not on level."""
        assert outcome.gap.spearman_between_runs > 0.9


class TestTheUniverseBracket:
    def test_both_bounds_are_reported_for_each_run(self, outcome):
        bracket = outcome.record.universe_bracket

        for run in (FAITHFUL, VENUE):
            assert set(bracket[run]) == {"binance-full", "tokocrypto"}

    def test_the_lower_bound_is_never_above_the_upper(self, outcome):
        for bounds in outcome.record.universe_bracket.values():
            assert bounds["tokocrypto"] <= bounds["binance-full"]


class TestThePairIsChecked:
    def test_a_faithful_run_on_venue_prices_is_refused(
        self, daily_panel_workspace, configs, archive
    ):
        faithful, venue = configs(faithful={})
        faithful.write_text(
            config_text(name="gate-faithful", price_source="binance-archive")
        )
        with pytest.raises(ConfigError, match="CoinMarketCap"):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

    def test_a_venue_run_on_vendor_prices_is_refused(
        self, daily_panel_workspace, configs, archive
    ):
        faithful, venue = configs()
        venue.write_text(config_text(name="gate-venue", price_source="cmc-panel"))
        with pytest.raises(ConfigError, match="Binance archive"):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

    def test_two_runs_that_differ_in_more_than_prices_are_refused(
        self, daily_panel_workspace, configs, archive
    ):
        """Otherwise the gap between them would measure two things at once."""
        faithful, venue = configs(venue={"quantile": "0.4"})
        with pytest.raises(ConfigError, match="quantile"):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

    def test_two_runs_ranking_different_assets_are_refused(
        self, daily_panel_workspace, configs, archive
    ):
        """A gap between runs that ranked different cross-sections is a fact
        about the two lists, not about the prices."""
        faithful, venue = configs()
        venue.write_text(
            config_text(name="gate-venue", price_source="binance-archive").replace(
                SYMBOLS, '["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]'
            )
        )
        with pytest.raises(ConfigError, match="symbols"):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

    def test_two_runs_priced_in_different_cost_models_are_refused(
        self, daily_panel_workspace, configs, archive
    ):
        faithful, venue = configs(venue={"model": '"tokocrypto"'})
        with pytest.raises(ConfigError, match="cost model"):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

    def test_the_pair_is_checked_before_either_grid_runs(
        self, daily_panel_workspace, configs, archive
    ):
        """42 cells is a long way to get before finding out the comparison was
        never going to mean anything."""
        faithful, venue = configs(venue={"quantile": "0.4"})
        with pytest.raises(ConfigError):
            run_gate(
                faithful, venue, daily_panel_workspace, run_at_utc=RUN_AT, open_url=archive
            )

        assert not daily_panel_workspace.results_root.exists()


class TestTheWeeklyPanelIsRefused:
    def test_a_faithful_run_on_the_stored_weekly_panel_will_not_start(
        self, workspace, archive
    ):
        """ADR-0008 pulls at `--interval 7d` and ADR-0001 marks daily. Nothing
        resamples between them: filling six days in seven would invent prices."""
        directory = workspace.repo_root / "configs"
        directory.mkdir(parents=True, exist_ok=True)
        faithful = directory / "gate-faithful.toml"
        venue = directory / "gate-venue.toml"
        faithful.write_text(config_text(name="gate-faithful", price_source="cmc-panel"))
        venue.write_text(config_text(name="gate-venue", price_source="binance-archive"))

        with pytest.raises(PanelGrainTooCoarse):
            run_gate(faithful, venue, workspace, run_at_utc=RUN_AT, open_url=archive)


class TestTheCommandLine:
    def test_a_failing_gate_exits_on_its_own_code_rather_than_the_refusal_one(
        self, daily_panel_workspace, configs, monkeypatch, capsys
    ):
        """A gate that fails is a finding, not a fault."""
        faithful, venue = configs()
        monkeypatch.setattr(
            "crypto_momentum.runner.load_cross_section",
            _archive_reader(),
        )
        code = main(
            [
                "--repo-root",
                str(daily_panel_workspace.repo_root),
                "gate",
                str(faithful),
                str(venue),
                "--reference-leg",
                LONG_SHORT,
            ]
        )

        assert code in (0, EXIT_GATE_FAILED)
        assert code != EXIT_REFUSED
        captured = capsys.readouterr()
        assert "Replication Gate:" in captured.err
        assert json.loads(captured.out)["leg"] == LONG_SHORT

    def test_the_verdict_prints_the_gap_and_both_bracket_bounds(
        self, daily_panel_workspace, configs, monkeypatch, capsys
    ):
        faithful, venue = configs()
        monkeypatch.setattr(
            "crypto_momentum.runner.load_cross_section",
            _archive_reader(),
        )
        main(
            [
                "--repo-root",
                str(daily_panel_workspace.repo_root),
                "gate",
                str(faithful),
                str(venue),
                "--reference-leg",
                LONG_ONLY,
            ]
        )

        captured = capsys.readouterr()
        assert "gap (venue minus faithful)" in captured.err
        assert "universe bracket, both bounds" in captured.err
        assert "tokocrypto" in captured.err
        # The floor and the cost assumption are on the printed verdict, not only
        # in the JSON — a footnote in a config file is not "stated in the result".
        assert "floor 2017-08-17" in captured.err
        assert "costs: paper at 15.0bp per side" in captured.err


def _archive_reader():
    """`load_cross_section` bound to the synthetic archive.

    The CLI builds its own opener — it is the edge, and the edge reaches the
    network — so a CLI test has to substitute the archive one layer in.
    """
    from crypto_momentum import runner
    from synthetic_cross_section import archive as archive_fixture

    opener = archive_fixture.__wrapped__()
    original = runner.load_cross_section

    def read(config, workspace, *, fetched_at_utc, open_url=None):
        return original(
            config, workspace, fetched_at_utc=fetched_at_utc, open_url=opener
        )

    return read
