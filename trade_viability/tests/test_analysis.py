import datetime as dt

import pandas as pd
import pytest

from datamexico.analysis import (
    EXPORTS, FAIL, IMPORTS, MANUAL, PASS, WATCH, Thresholds, cagr, concentration, fmt_value,
    full_years, overall_verdict, partial_years_from_coverage, viability_signals, yearly_summary,
)


def _trade(values):
    """values: {year: (imports, exports)}"""
    rows = []
    for year, (imp, exp) in values.items():
        rows += [{"year": year, "flow": IMPORTS, "value": imp}, {"year": year, "flow": EXPORTS, "value": exp}]
    return pd.DataFrame(rows)


def test_yearly_summary_net_imports_and_growth():
    s = yearly_summary(_trade({2020: (100, 10), 2021: (120, 20)}))
    assert list(s.index) == [2020, 2021]
    assert s.loc[2021, "net_imports"] == 100
    assert s.loc[2021, "imports_yoy"] == pytest.approx(0.2)
    assert pd.isna(s.loc[2020, "imports_yoy"])


def test_yearly_summary_fills_missing_years_with_zero():
    s = yearly_summary(_trade({2019: (5, 0), 2021: (7, 0)}))
    assert s.loc[2020, "imports"] == 0


def test_apparent_consumption_and_import_dependence_match_approach_doc():
    # Propylene 2024 in the approach document: production 141,725 t,
    # imports 266,649 t, consumption 408,374 t -> 65% import dependence.
    s = yearly_summary(_trade({2024: (266_649, 0)}), production={2024: 141_725})
    assert s.loc[2024, "apparent_consumption"] == 408_374
    assert round(s.loc[2024, "import_dependence"], 2) == 0.65


def test_cagr():
    assert cagr(100, 121, 2) == pytest.approx(0.10)
    assert cagr(0, 10, 3) is None
    assert cagr(10, 10, 0) is None


def test_concentration_hhi():
    df = pd.DataFrame({"year": [2024] * 3, "flow": [IMPORTS] * 3,
                       "country": ["A", "B", "C"], "value": [50, 30, 20]})
    c = concentration(df, "country", 2024)
    assert c.hhi == pytest.approx(50**2 + 30**2 + 20**2)
    assert c.top1_name == "A" and c.top1_share == pytest.approx(0.5)
    assert c.top3_share == pytest.approx(1.0)


def test_full_years_excludes_current_and_partial():
    s = yearly_summary(_trade({2022: (1, 0), 2023: (1, 0), 2024: (1, 0), 2025: (1, 0)}))
    assert full_years(s, [2024], today=dt.date(2025, 3, 1)) == [2022, 2023]
    assert partial_years_from_coverage({2023: 12, 2024: 12, 2025: 6}) == [2025]


def _signals(values, production=None, **thr):
    s = yearly_summary(_trade(values), production)
    return viability_signals(s, list(s.index), Thresholds(**thr), unit="USD")


def test_recurring_import_market_passes_tests_1_and_2():
    _, verdict = _signals({y: (10e6 * 1.05 ** i, 1e6) for i, y in enumerate(range(2019, 2024))})
    assert verdict["1. Recurring demand"] == PASS
    assert verdict["2. Open market"] == PASS
    assert verdict["3. Service repeatability"] == MANUAL
    assert overall_verdict(verdict) == "Pending analyst input"


def test_negligible_imports_fail_recurring_demand():
    # Like heptane/hexane in the approach document: tiny volumes.
    _, verdict = _signals({y: (300, 0) for y in range(2019, 2024)}, min_avg_imports=1000)
    assert verdict["1. Recurring demand"] == FAIL


def test_net_exporter_fails_open_market():
    # Like ethyl acetate: Mexico exports far more than it imports.
    _, verdict = _signals({y: (5_870, 105_662) for y in range(2019, 2024)}, min_avg_imports=1000)
    assert verdict["2. Open market"] == FAIL
    assert overall_verdict(verdict) == "Not viable"


def test_low_import_dependence_fails_open_market():
    # Acetic anhydride: domestic production covers demand (0.1% dependence).
    values = {y: (85, 0) for y in range(2019, 2024)}
    _, verdict = _signals(values, production={y: 60_094 for y in values}, min_avg_imports=10)
    assert verdict["2. Open market"] == FAIL


def test_volatile_imports_are_watch_and_manual_inputs_flow_through():
    values = {2019: (1e6, 0), 2020: (9e6, 0), 2021: (1e6, 0), 2022: (9e6, 0), 2023: (9e6, 0)}
    s = yearly_summary(_trade(values))
    signals, verdict = viability_signals(
        s, list(s.index), Thresholds(),
        manual={"3. Service repeatability": (PASS, "fits ambient tankage"),
                "4. Economic margin": (PASS, "19-22% distributor margin")},
    )
    assert verdict["1. Recurring demand"] == WATCH
    assert overall_verdict(verdict) == "Viable - with watch points"
    assert any(sig.value == "fits ambient tankage" for sig in signals)


def test_fmt_value():
    assert fmt_value(1_234_567) == "1.2M"
    assert fmt_value(-2_500) == "-2.5K"
    assert fmt_value(3.2e9) == "3.2B"
    assert fmt_value(12) == "12"
