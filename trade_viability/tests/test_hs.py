from datamexico.client import Member
from datamexico.hs import build_entries, format_code, hs_code, normalize, overlapping, search


def _entries():
    members = {
        2: [Member("629", "Organic chemicals")],
        4: [Member("62901", "Acyclic hydrocarbons"), Member("62902", "Cyclic hydrocarbons"),
            Member("52711", "Petroleum gases and other gaseous hydrocarbons")],
        6: [Member("6290121", "Ethylene"), Member("6290122", "Propene (propylene)"),
            Member("6290230", "Toluene"), Member("6290241", "o-Xylene"),
            Member("5271114", "Ethylene, propylene, butylene and butadiene, liquefied"),
            Member("7390210", "Polypropylene")],
    }
    alt = {6: {"6290230": "Tolueno", "6290122": "Propeno (propileno)"}}
    return build_entries(members, alt)


def test_hs_code_strips_section_prefix_and_pads():
    assert hs_code("6290230", 6) == "290230"
    assert hs_code("62902", 4) == "2902"
    assert hs_code("10101", 4) == "0101"
    assert hs_code("10111", 6) == "010111"
    assert hs_code("11520100", 6) == "520100"
    assert hs_code(629, 2) == "29"


def test_format_code():
    assert format_code("290230") == "2902.30"
    assert format_code("2902") == "2902"


def test_normalize_strips_accents():
    assert normalize("Químicos  Orgánicos!") == "quimicos organicos"


def test_search_by_name_english_and_spanish():
    entries = _entries()
    assert [e.code for e in search(entries, "toluene")] == ["290230"]
    assert [e.code for e in search(entries, "Tolueno")] == ["290230"]


def test_search_parent_label_attached():
    toluene = search(_entries(), "toluene")[0]
    assert toluene.parent_label == "Cyclic hydrocarbons"


def test_search_by_code_prefix_returns_heading_and_children():
    codes = [(e.digits, e.code) for e in search(_entries(), "2902")]
    assert (4, "2902") in codes and (6, "290230") in codes and (6, "290241") in codes
    assert all(code.startswith("2902") for _, code in codes)
    assert codes[0] == (4, "2902")
    assert [e.code for e in search(_entries(), "2902.30")] == ["290230"]


def test_search_curated_synonyms_rank_first():
    codes = [e.code for e in search(_entries(), "propylene")]
    # curated: propene 290122 and liquefied olefins 271114; keyword: polypropylene
    assert codes[:2] == ["271114", "290122"]
    assert "390210" in codes


def test_search_plural_and_level_filter():
    assert [e.code for e in search(_entries(), "xylenes")] == ["290241"]
    assert all(e.digits == 4 for e in search(_entries(), "hydrocarbons", levels=[4]))


def test_overlapping_selection_detected():
    entries = _entries()
    heading = next(e for e in entries[4] if e.code == "2902")
    toluene = next(e for e in entries[6] if e.code == "290230")
    ethylene = next(e for e in entries[6] if e.code == "290121")
    assert overlapping([heading, toluene, ethylene]) == [(heading, toluene)]
