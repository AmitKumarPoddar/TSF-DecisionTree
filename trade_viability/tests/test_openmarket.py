import pandas as pd
import pytest

from datamexico import openmarket as om
from datamexico.analysis import EXPORTS, IMPORTS


def _series(values, start=2021):
    return pd.Series(values, index=range(start, start + len(values)), dtype=float)


YEARS = list(range(2019, 2026))


@pytest.mark.parametrize("values,cls", [
    ([100, 104, 108, 112, 117], om.GROWING),
    ([100, 101, 99, 100, 101], om.STABLE),
    ([100, 95, 90, 85, 80], om.DECLINING),
    ([0, 0, 0, 0, 0], om.NO_IMPORTS),
])
def test_trend_classification(values, cls):
    t = om.import_trend(_series(values), list(range(2021, 2026)))
    assert t.classification == cls


def test_trend_uses_last_five_complete_years_only():
    s = _series([10, 10, 100, 101, 99, 100, 101], start=2019)
    t = om.import_trend(s, YEARS)
    assert t.years == [2021, 2022, 2023, 2024, 2025] and t.classification == om.STABLE


def test_spike_does_not_decide_trend():
    # Real propylene imports (USD m) with the 2021 spike: a least-squares line would read "declining";
    # the Theil-Sen trend is not swung by the single spike year.
    t = om.import_trend(_series([239, 501, 309, 262, 275], start=2020), list(range(2020, 2025)))
    assert t.first_last_cagr > 0
    assert t.slope_pct == pytest.approx(-0.0147, abs=0.001)
    assert t.classification == om.STABLE
    assert len(t.trend_line) == 5


def test_quantity_measure_detection():
    assert om.quantity_measure(["Trade Value", "Quantity"]) == "Quantity"
    assert om.quantity_measure(["Trade Value"]) is None


def test_hhi_series_and_direction():
    rows = []
    for year, shares in ((2021, {"US": 90, "CN": 10}), (2025, {"US": 40, "CN": 30, "KR": 30})):
        for c, v in shares.items():
            rows.append({"year": year, "flow": IMPORTS, "country": c, "value": v})
    rows.append({"year": 2025, "flow": EXPORTS, "country": "US", "value": 999})
    h = om.hhi_series(pd.DataFrame(rows), [2021, 2025])
    assert h.first == pytest.approx(8200) and h.latest == pytest.approx(3400)
    assert "falling" in h.direction and "concentrated supply" in h.note


def test_distinct_groups_merges_subsidiaries_and_legal_suffixes():
    df = om.companies_frame([
        {"company": "LyondellBasell México S.A. de C.V.", "group": "LyondellBasell", "include": True},
        {"company": "LyondellBasell Industries", "group": "LyondellBasell Group", "include": True},
        {"company": "Grupo Alpek", "group": "", "include": True},
        {"company": "Excluded Inc.", "group": "", "include": False},
        {"company": "", "group": "", "include": True},
    ])
    assert om.distinct_groups(df) == ["lyondellbasell", "alpek"]


def test_competitor_register_append_only():
    reg = []
    df = om.companies_frame([{"company": "A", "url": "u", "origin": "AI"}])
    om.log_companies(reg, df, "PP")
    om.log_companies(reg, df, "PP")
    assert len(reg) == 1
    df.loc[0, "country"] = "Mexico"
    om.log_companies(reg, df, "PP")
    assert len(reg) == 2


def _trend(cls):
    values = {om.GROWING: [100, 105, 110, 116, 122], om.STABLE: [100] * 5,
              om.DECLINING: [100, 90, 80, 70, 60], om.NO_IMPORTS: [0] * 5}[cls]
    return om.import_trend(_series(values), list(range(2021, 2026)))


@pytest.mark.parametrize("cls,code", [(om.GROWING, om.OPEN), (om.STABLE, om.OPEN),
                                      (om.DECLINING, om.NOT_OPEN), (om.NO_IMPORTS, om.NOT_OPEN)])
def test_path_a_exact_hs(cls, code):
    v = om.open_market_verdict(True, _trend(cls), None, False, False, None)
    assert v.code == code


@pytest.mark.parametrize("n,cls,code", [
    (4, om.GROWING, om.OPEN),
    (5, om.STABLE, om.OPEN),
    (4, om.DECLINING, om.WATCH),
    (2, om.GROWING, om.WATCH),
    (1, om.DECLINING, om.WATCH),
])
def test_path_b_base_hs(n, cls, code):
    assert om.open_market_verdict(False, _trend(cls), n, True, False, None).code == code


def test_path_b_zero_competitors_needs_confirmation():
    pending = om.open_market_verdict(False, _trend(om.GROWING), 0, True, False, None, "MFPP")
    assert pending.code == om.PENDING and "MFPP" in pending.detail
    final = om.open_market_verdict(False, _trend(om.GROWING), 0, True, True, None, "MFPP")
    assert final.code == om.NO_MARKET and final.headline == "No market in Mexico for this opportunity."


def test_path_b_requires_scan_and_hhi_is_note_only():
    h = om.HHISeries(pd.DataFrame({"year": [2025], "hhi": [9000.0]}), 9000.0, 9000.0)
    assert om.open_market_verdict(False, _trend(om.GROWING), None, False, False, h).code == om.PENDING
    v = om.open_market_verdict(False, _trend(om.GROWING), 6, True, False, h)
    assert v.code == om.OPEN and any("concentrated supply" in n for n in v.notes)


def test_pending_without_trade():
    assert om.open_market_verdict(True, None, None, False, False, None).code == om.PENDING


def _by_country(year_shares):
    rows = []
    for year, shares in year_shares.items():
        for c, v in shares.items():
            rows.append({"year": year, "flow": IMPORTS, "country": c, "value": v})
    return pd.DataFrame(rows)


def test_country_shares_per_year_with_other_bucket():
    df = _by_country({2023: {"US": 50, "CN": 30, "KR": 20}, 2024: {"US": 40, "CN": 20, "KR": 20, "DE": 10, "JP": 10}})
    sh = om.country_shares(df, [2023, 2024], top=3)
    assert list(sh.columns) == [2023, 2024]
    assert list(sh.index) == ["US", "CN", "KR", "Other countries"]
    assert sh.loc["US", 2024] == pytest.approx(0.4) and sh.loc["Other countries", 2024] == pytest.approx(0.2)
    assert sh[2023].sum() == pytest.approx(1.0)


def test_group_names_keep_display_names():
    df = om.companies_frame([
        {"company": "LyondellBasell México", "group": "LyondellBasell", "include": True},
        {"company": "LyondellBasell Industries", "group": "LyondellBasell Group", "include": True},
        {"company": "Avient", "group": "", "include": True},
    ])
    assert om.group_names(df) == ["LyondellBasell", "Avient"]


PLAIN_FORBIDDEN = ("(", ")", "—", " – ", ":")


def test_open_market_paragraph_open_fragmented():
    trend = _trend(om.GROWING)
    shares = om.country_shares(_by_country({2025: {"United States": 34, "China": 22, "South Korea": 15,
                                                   "Germany": 15, "Japan": 14}}), [2025])
    v = om.open_market_verdict(False, trend, 4, True, False, None)
    text = om.open_market_paragraph(v, trend, shares, ["A", "B", "C", "D"], "MFPP", "polypropylene, the base "
                                    "material of MFPP", False, True)
    assert text.startswith("Yes, the market for MFPP in Mexico is open for a new supplier.")
    assert "imports of polypropylene, the base material of MFPP, have increased by about" in text and "a year between 2021 and 2025" in text
    assert "fragmented across multiple countries, such as United States with 34%, China with 22% and" in text
    assert "Many manufacturers, such as A, B, C and D, supply MFPP" in text
    assert not any(ch in text for ch in PLAIN_FORBIDDEN), text


def test_open_market_paragraph_concentrated_and_not_open():
    trend = _trend(om.DECLINING)
    shares = om.country_shares(_by_country({2025: {"United States": 85, "China": 15}}), [2025])
    v = om.open_market_verdict(True, trend, None, False, False, None)
    text = om.open_market_paragraph(v, trend, shares, [], "MFPP", "MFPP", True, False)
    assert text.startswith("No, the market for MFPP in Mexico is not open for a new supplier.")
    assert "have decreased by about" in text
    assert "concentrated, with United States alone supplying 85%" in text
    assert text.endswith("This is not a good signal to enter the market.")
    assert not any(ch in text for ch in PLAIN_FORBIDDEN), text
