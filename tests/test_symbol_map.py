"""CoinMarketCap id to Binance ticker mapping.

ADR-0008 calls this out as the hybrid's known bug surface. The mapping is
time-varying by construction: a Binance base is a name Binance reuses, while a
CoinMarketCap id is permanent, so the only honest question is which id a base
referred to *on a given date*.
"""

from datetime import date
from pathlib import Path

import pytest

from crypto_momentum.data.cmc_panel import parse_panel_csv
from crypto_momentum.data.symbol_map import (
    AmbiguousTicker,
    MalformedOverrideTable,
    SymbolSpell,
    VendorLink,
    build_symbol_map,
    load_overrides,
    symbol_spells,
    vendor_symbol_map,
)

FIXTURES = Path(__file__).parent / "fixtures" / "coinmarketcap"
REPO_ROOT = Path(__file__).parent.parent

# The bases the archive lists in the fixture's window. PEPE is deliberately
# absent from the panel, and BitConnect is deliberately absent from here.
BINANCE_BASES = ("BTC", "ETH", "SRM", "LUNA", "LUNC", "PEPE")


@pytest.fixture
def panel():
    return parse_panel_csv((FIXTURES / "cmc-panel-sample.csv").read_bytes())


@pytest.fixture
def symbol_map(panel):
    return build_symbol_map(symbol_spells(panel), BINANCE_BASES)


# --- the ordinary case ----------------------------------------------------


def test_an_id_resolves_to_its_binance_base(symbol_map):
    assert symbol_map.binance_base_for(1, date(2020, 1, 1)) == "BTC"
    assert symbol_map.binance_base_for(1027, date(2020, 1, 1)) == "ETH"


def test_a_base_resolves_back_to_its_id(symbol_map):
    assert symbol_map.cmc_id_for("ETH", date(2020, 1, 1)) == 1027


def test_an_id_does_not_resolve_before_the_panel_first_lists_it(symbol_map):
    """Serum's first snapshot is 2020-08-16; resolving it earlier is a lookahead."""
    assert symbol_map.binance_base_for(6187, date(2019, 1, 1)) is None
    assert symbol_map.binance_base_for(6187, date(2020, 8, 16)) == "SRM"


def test_a_delisted_asset_still_resolves_over_its_own_history(symbol_map):
    """Serum stopped trading in November 2022. The Universe still holds it then."""
    assert symbol_map.binance_base_for(6187, date(2022, 5, 22)) == "SRM"
    assert symbol_map.binance_base_for(6187, date(2023, 6, 1)) is None


# --- present in one source, absent from the other -------------------------


def test_an_asset_on_coinmarketcap_but_not_on_binance_is_reported_unmatched(symbol_map):
    """BitConnect was never listed on Binance. It is out of the Universe, not lost."""
    assert 827 in symbol_map.unmatched_cmc_ids
    assert symbol_map.binance_base_for(827, date(2017, 6, 1)) is None


def test_an_asset_on_binance_but_not_in_the_panel_is_reported_unmatched(symbol_map):
    """No market cap means no value weight, so PEPE cannot silently be weighted."""
    assert "PEPE" in symbol_map.unmatched_binance_bases
    assert symbol_map.cmc_id_for("PEPE", date(2023, 6, 1)) is None


def test_every_base_is_either_mapped_or_named_as_unmatched(symbol_map):
    mapped = {link.binance_base for link in symbol_map.links}

    assert mapped | set(symbol_map.unmatched_binance_bases) == set(BINANCE_BASES)


# --- ticker reuse ---------------------------------------------------------


def test_luna_before_the_unwind_is_terra_classic(symbol_map):
    assert symbol_map.cmc_id_for("LUNA", date(2022, 5, 22)) == 4172


def test_luna_after_the_unwind_is_terra_two(symbol_map):
    """Binance reassigned LUNAUSDT to Terra 2.0 on 2022-05-31."""
    assert symbol_map.cmc_id_for("LUNA", date(2022, 6, 15)) == 20314


def test_the_original_terra_follows_its_rename_to_lunc(symbol_map):
    assert symbol_map.binance_base_for(4172, date(2022, 5, 22)) == "LUNA"
    assert symbol_map.binance_base_for(4172, date(2022, 6, 15)) == "LUNC"
    assert symbol_map.cmc_id_for("LUNC", date(2022, 6, 15)) == 4172


def test_the_two_terras_are_never_the_same_asset_on_the_same_day(symbol_map):
    """Collapsing them corrupts the 2022 cross-section — one fell 99.99%, one did not."""
    for day in (date(2022, 5, 22), date(2022, 6, 15), date(2022, 11, 13)):
        resolved = [symbol_map.cmc_id_for(base, day) for base in ("LUNA", "LUNC")]
        held = [cmc_id for cmc_id in resolved if cmc_id is not None]
        assert len(held) == len(set(held)), f"LUNA and LUNC collapse on {day}"


def test_two_ids_claiming_one_base_on_one_day_is_refused_not_silently_collapsed():
    """A vendor that never recorded the LUNA-to-LUNC rename must fail loudly."""
    collapsed = (
        SymbolSpell(4172, "Terra Classic", "LUNA", date(2019, 7, 26), date(2023, 1, 8)),
        SymbolSpell(20314, "Terra", "LUNA", date(2022, 5, 29), date(2023, 1, 8)),
    )

    with pytest.raises(AmbiguousTicker, match="LUNA"):
        build_symbol_map(collapsed, ("LUNA",))


def test_an_override_resolves_a_collapsed_ticker():
    collapsed = (
        SymbolSpell(4172, "Terra Classic", "LUNA", date(2019, 7, 26), date(2023, 1, 8)),
        SymbolSpell(20314, "Terra", "LUNA", date(2022, 5, 29), date(2023, 1, 8)),
    )
    # Both sides of the collision have to be named: resolving one and leaving
    # the other on its derived spell still overlaps, and still raises.
    overrides = (
        VendorLink(4172, "LUNA", date(2019, 7, 26), date(2022, 5, 31)),
        VendorLink(4172, "LUNC", date(2022, 5, 31), None),
        VendorLink(20314, "LUNA", date(2022, 5, 31), None),
    )

    resolved = build_symbol_map(collapsed, ("LUNA", "LUNC"), overrides=overrides)

    assert resolved.cmc_id_for("LUNA", date(2020, 1, 1)) == 4172
    assert resolved.cmc_id_for("LUNA", date(2022, 6, 15)) == 20314
    assert resolved.cmc_id_for("LUNC", date(2022, 6, 15)) == 4172


def test_an_override_replaces_every_derived_link_for_that_id():
    derived = (
        SymbolSpell(4172, "Terra Classic", "LUNA", date(2019, 7, 26), date(2023, 1, 8)),
    )
    overrides = (VendorLink(4172, "LUNC", date(2022, 5, 31), None),)

    resolved = build_symbol_map(derived, ("LUNA", "LUNC"), overrides=overrides)

    assert resolved.cmc_id_for("LUNA", date(2020, 1, 1)) is None
    assert resolved.binance_base_for(4172, date(2020, 1, 1)) is None
    assert resolved.binance_base_for(4172, date(2023, 6, 1)) == "LUNC"


def test_an_open_ended_link_still_resolves_far_in_the_future():
    resolved = build_symbol_map(
        (SymbolSpell(1, "Bitcoin", "BTC", date(2013, 4, 28), None),), ("BTC",)
    )

    assert resolved.binance_base_for(1, date(2030, 1, 1)) == "BTC"


# --- the committed override table ----------------------------------------


def test_the_committed_override_table_loads_and_names_both_terras():
    overrides = load_overrides(REPO_ROOT / "configs" / "vendor-symbol-map.toml")

    by_id = {link.cmc_id for link in overrides}
    assert {4172, 20314} <= by_id


def test_the_committed_override_table_is_itself_unambiguous():
    """The overrides are hand-written, so they get the same overlap check.

    `build_symbol_map` raises on overlap, so simply building is the assertion.
    """
    overrides = load_overrides(REPO_ROOT / "configs" / "vendor-symbol-map.toml")
    bases = tuple({link.binance_base for link in overrides})

    built = build_symbol_map((), bases, overrides=overrides)

    assert built.links


# --- spells derived from the panel ---------------------------------------


def test_a_spell_runs_from_first_snapshot_to_the_snapshot_after_its_last(panel):
    spells = {(s.cmc_id, s.symbol): s for s in symbol_spells(panel)}

    bitconnect = spells[(827, "BCC")]
    assert bitconnect.valid_from == date(2017, 1, 1)
    # Last seen 2018-01-07; the panel's next snapshot is 2018-01-14, which is
    # where BitConnect is gone and so where the spell must end.
    assert bitconnect.valid_until == date(2018, 1, 14)


def test_the_final_spell_stays_open_ended(panel):
    spells = {(s.cmc_id, s.symbol): s for s in symbol_spells(panel)}

    assert spells[(1, "BTC")].valid_until is None


def test_one_id_with_two_symbols_yields_two_spells_that_do_not_overlap(panel):
    terra = sorted(
        (s for s in symbol_spells(panel) if s.cmc_id == 4172), key=lambda s: s.valid_from
    )

    assert [s.symbol for s in terra] == ["LUNA", "LUNC"]
    assert terra[0].valid_until == terra[1].valid_from


# --- the mapping as production actually builds it -------------------------


def test_the_venue_rename_date_wins_over_the_vendors_snapshot_grid(panel):
    """The whole reason the override table exists, asserted end to end.

    CoinMarketCap's snapshot moves id 4172 from LUNA to LUNC on 2022-05-29.
    Binance renamed on 2022-05-31. Between those two dates a LUNAUSDT bar is
    still the original chain, so 2022-05-30 has to resolve to 4172 — the
    derived-only mapping gets this wrong, and that is what is being pinned.
    """
    derived_only = build_symbol_map(symbol_spells(panel), BINANCE_BASES)
    as_used = vendor_symbol_map(panel, BINANCE_BASES, repo_root=REPO_ROOT)

    assert derived_only.cmc_id_for("LUNA", date(2022, 5, 30)) == 20314
    assert as_used.cmc_id_for("LUNA", date(2022, 5, 30)) == 4172
    assert as_used.cmc_id_for("LUNA", date(2022, 5, 31)) == 20314
    assert as_used.cmc_id_for("LUNC", date(2022, 5, 31)) == 4172


def test_the_mapping_as_used_still_matches_the_ordinary_assets(panel):
    as_used = vendor_symbol_map(panel, BINANCE_BASES, repo_root=REPO_ROOT)

    assert as_used.binance_base_for(1, date(2020, 1, 1)) == "BTC"
    assert as_used.binance_base_for(6187, date(2022, 5, 22)) == "SRM"
    assert 827 in as_used.unmatched_cmc_ids
    assert "PEPE" in as_used.unmatched_binance_bases


def test_a_hand_resolved_link_without_a_reason_is_refused(tmp_path):
    """The table overrules the panel, so it has to say why in the file itself."""
    table = tmp_path / "vendor-symbol-map.toml"
    table.write_text(
        "[[link]]\n"
        "cmc_id = 4172\n"
        'binance_base = "LUNC"\n'
        "valid_from = 2022-05-31\n"
    )

    with pytest.raises(MalformedOverrideTable, match="no reason"):
        load_overrides(table)


def test_every_link_in_the_committed_table_says_why_it_is_there():
    assert load_overrides(REPO_ROOT / "configs" / "vendor-symbol-map.toml")
