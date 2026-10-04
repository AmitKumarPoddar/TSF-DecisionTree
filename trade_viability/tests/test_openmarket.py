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
