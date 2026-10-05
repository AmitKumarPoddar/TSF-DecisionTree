import datetime as dt

import pandas as pd
import pytest

from datamexico import market as m


def _figs(*rows):
    return m.figures_frame(rows)


FX = {"MXN": 20.0, "EUR": 0.5}


def test_parse_unit():
    assert m.parse_unit("USD million") == ("value", "USD", 1.0)
    assert m.parse_unit("MXN billion") == ("value", "MXN", 1000.0)
    assert m.parse_unit("kt (thousand tonnes)") == ("volume", None, 1.0)
    assert m.parse_unit("tonnes") == ("volume", None, 0.001)
    assert m.parse_unit("furlongs")[0] == "unknown"


def test_normalize_converts_currency_scale_and_year():
    df = _figs(
        {"amount": 100, "unit": "USD million", "year": 2025, "geography": "Mexico"},
        {"amount": 2, "unit": "MXN billion", "year": 2025, "geography": "México"},  # 2000 MXN m / 20 = 100
        {"amount": 100, "unit": "USD million", "year": 2023, "cagr_pct": 10, "geography": "Mexico"},
        {"amount": 100, "unit": "USD million", "year": 2023, "geography": "Mexico"},
    )
    n = m.normalize(df, FX, ref_year=2025)
    assert list(n["comparable"].round(6)) == [100, 100, 100, 100]
    assert n.loc[2, "at_ref_year"] == pytest.approx(121.0)
    assert n.loc[3, "at_ref_year"] == 100 and "not adjusted" in n.loc[3, "adjustment"]
    assert n["mexico"].all()


def test_missing_fx_rate_makes_figure_unusable():
    n = m.normalize(_figs({"amount": 5, "unit": "BRL million", "year": 2025, "geography": "Mexico"}), FX, 2025)
    assert n.loc[0, "comparable"] is None and "no FX rate" in n.loc[0, "adjustment"]


def test_stats_range_average_all_and_selected():
    df = _figs(
        {"amount": 100, "unit": "USD million", "year": 2025, "geography": "Mexico"},
        {"amount": 300, "unit": "USD million", "year": 2025, "geography": "Mexico", "use": True},
        {"amount": 200, "unit": "USD million", "year": 2025, "geography": "Mexico", "use": True},
        {"amount": 999, "unit": "USD million", "year": 2025, "geography": "Latin America"},
    )
    s = m.market_stats(m.normalize(df, FX, 2025), "value")
    assert (s.minimum, s.maximum, s.average_all) == (100, 300, 200)  # Latin America excluded
    assert s.average_selected == 250 and s.used == 250 and s.n_selected == 2


def test_stats_no_selection_uses_average_of_all():
    df = _figs({"amount": 100, "unit": "USD million", "year": 2025, "geography": "Mexico"},
               {"amount": 300, "unit": "USD million", "year": 2025, "geography": "Mexico"})
    s = m.market_stats(m.normalize(df, FX, 2025), "value")
    assert s.used == 200 and "none selected" in s.basis


def test_only_non_mexico_figures_are_not_a_market_size():
    df = _figs({"amount": 100, "unit": "USD million", "year": 2025, "geography": "North America"})
    assert m.primary_stats(m.normalize(df, FX, 2025)) is None


def test_trade_market_is_net_imports_times_relevance():
    summary = _summary([1000e6, 1200e6, 1500e6, 1400e6, 2000e6], [100e6, 200e6, 300e6, 400e6, 500e6])
    rel = m.Relevance(n_categories=4, r1=m.default_share(4), n_products=5, r2=m.default_share(5))
    tm = m.trade_market(summary, list(range(2019, 2024)), rel.combined)
    assert tm.unit == "USD million" and tm.latest_year == 2023
    assert tm.estimate == pytest.approx(1500 * 0.05)          # 2023 net imports 1,500 m × 0.05
    res = m.market_result(None, None, tm, rel)
    assert res.method == m.BASE_HS and res.estimate == pytest.approx(75) and res.year == 2023
    gross = m.trade_market(summary, list(range(2019, 2024)), rel.combined, m.GROSS)
    assert gross.estimate == pytest.approx(2000 * 0.05)


def test_net_exporter_gives_no_trade_size():
    summary = _summary([100e6, 100e6], [50e6, 150e6])
    tm = m.trade_market(summary, [2019, 2020], 1.0)
    assert tm.estimate is None and "not positive" in tm.note
    assert m.market_result(tm, None, None).method == m.NONE
    assert m.trade_market(summary, [2019], 1.0).estimate == pytest.approx(50)


def _reports(amount):
    return m.market_stats(m.normalize(_figs({"amount": amount, "unit": "USD million", "year": 2025,
                                             "geography": "Mexico"}), FX, 2025), "value")


def test_step_priority_exact_hs_then_reports_then_base():
    exact = m.trade_market(_summary([90e6], [10e6]), [2019], 1.0)
    base = m.trade_market(_summary([1e9], [0]), [2019], 0.04)
    rel = m.Relevance(5, 0.2, 5, 0.2)
    assert m.market_result(exact, _reports(40), base, rel).method == m.EXACT_HS
    assert m.market_result(exact, _reports(40), base, rel).estimate == pytest.approx(80)
    res = m.market_result(None, _reports(40), base, rel, ref_year=2025)
    assert (res.method, res.estimate, res.year) == (m.REPORTS, 40, 2025)
    assert m.market_result(None, None, base, rel).estimate == pytest.approx(40)
    assert m.market_result(None, None, base, None).method == m.NONE   # relevance not set yet
    # Gap 2: an exact code with net exports falls through to the reports
    net_exporter = m.trade_market(_summary([10e6], [50e6]), [2019], 1.0)
    assert m.market_result(net_exporter, _reports(40), base, rel).method == m.REPORTS


def _summary(imports, exports, start=2019):
    years = range(start, start + len(imports))
    df = pd.DataFrame({"imports": imports, "exports": exports}, index=pd.Index(years, name="year"))
    df["net_imports"] = df["imports"] - df["exports"]
    return df


def test_trade_recurrence_uses_last_five_years_only():
    s = _summary([0, 0, 0, 10, 10, 10, 10, 10], [0] * 8, start=2017)  # 2017-2024
    t = m.trade_recurrence(s, list(range(2017, 2025)))
    assert t.years == [2020, 2021, 2022, 2023, 2024]
    assert t.years_with_imports == 5 and t.recurring and t.strength == "High"


def test_trade_recurrence_moderate_when_net_exporter_in_a_year():
    s = _summary([10, 10, 10, 10, 10], [1, 1, 20, 1, 1])
    t = m.trade_recurrence(s, list(range(2019, 2024)))
    assert t.recurring and t.strength == "Moderate"


def test_trade_not_recurring():
    s = _summary([10, 0, 0, 0, 10], [0] * 5)
    t = m.trade_recurrence(s, list(range(2019, 2024)), min_years=4)
    assert not t.recurring and t.strength == "Not recurring"


PLAIN_FORBIDDEN = ("(", ")", "—", " – ", ":")


@pytest.mark.parametrize("market_found,recurring,code", [
    (True, True, m.ESTABLISHED),
    (True, False, m.NOT_ESTABLISHED),
    (False, True, m.NOT_ESTABLISHED),
    (False, False, m.NOT_ESTABLISHED),
])
def test_verdict_matrix(market_found, recurring, code):
    market = m.MarketResult(m.REPORTS if market_found else m.NONE, 120.0 if market_found else None,
                            "USD million", 2025)
    s = _summary([10] * 5 if recurring else [0, 0, 0, 10, 10], [0] * 5)
    v = m.recurring_demand_verdict(market, m.trade_recurrence(s, list(range(2019, 2024))), "MFPP",
                                   "Polypropylene", base_material=True)
    assert v.code == code
    assert not any(ch in v.detail for ch in PLAIN_FORBIDDEN), v.detail
    if code == m.ESTABLISHED:
        assert v.detail.startswith("Yes, the demand for MFPP in Mexico is recurring.")
        assert "approximately USD 120 million as of 2025" in v.detail
        assert ("Polypropylene, the immediate base material used for MFPP, has been imported into Mexico in all "
                "of the last five years, from 2019 to 2023, and Mexico has been a net importer") in v.detail
    else:
        assert v.detail.startswith("No, the recurring demand for MFPP in Mexico could not be established.")
    if not market_found:
        assert "No market size could be found" in v.detail
    if not recurring:
        assert "in only two of the last five years" in v.detail


def test_verdict_mentions_net_exports():
    market = m.MarketResult(m.EXACT_HS, 2157.4, "USD million", 2024)
    s = _summary([10] * 5, [5, 20, 20, 5, 5])
    v = m.recurring_demand_verdict(market, m.trade_recurrence(s, list(range(2019, 2024))), "mineral-filled PP")
    assert v.code == m.ESTABLISHED and "net importer in three of those years and a net exporter in the other two" \
        in v.detail
    assert "Mineral-filled PP has been imported" in v.detail and "USD 2.2 billion as of 2024" in v.detail


def test_size_detail_and_format():
    tm = m.trade_market(_summary([1e9], [0]), [2019], 0.05)
    res = m.market_result(None, None, tm, m.Relevance(4, 0.25, 5, 0.2))
    assert m.size_detail(res, base="Polypropylene") == \
        "2019, net imports (imports − exports) of Polypropylene × relevance 0.0500"
    assert m.fmt_size(2157.4, "USD million") == "USD 2.16 bn"


def test_verdict_pending_without_trade():
    assert m.recurring_demand_verdict(m.market_result(None, None, None), None).code == m.PENDING


def test_register_is_append_only():
    reg = []
    df = _figs({"publisher": "A", "amount": 1, "unit": "USD million", "url": "u", "origin": "AI"})
    now = dt.datetime(2026, 10, 1, 12, 0)
    fps1 = m.log_figures(reg, df, "1A", "PP", now)
    assert len(reg) == 1 and reg[0]["origin"] == "AI"
    m.log_figures(reg, df, "1A", "PP", now)
    assert len(reg) == 1  # no duplicate
    edited = df.copy()
    edited.loc[0, "amount"] = 2
    fps2 = m.log_figures(reg, edited, "1A", "PP", now)
    assert len(reg) == 2  # edit logged as a new entry, old kept
    frame = m.register_frame(reg, fps2)
    assert list(frame["status"]) == ["edited / removed from calculation", "in calculation"]
    assert fps1 != fps2


def test_register_ignores_blank_rows():
    reg = []
    m.log_figures(reg, m.figures_frame([{"use": False}]), "1A", "PP")
    assert reg == []


def test_source_cagr_applied_to_same_source_series():
    # IMARC-style series: only the forecast states the CAGR.
    df = _figs(
        {"publisher": "IMARC", "title": "PP", "amount": 1.39, "unit": "Billion USD", "year": 2020, "geography": "Mexico", "url": "https://i.com/pp"},
        {"publisher": "IMARC", "title": "PP", "amount": 1.71, "unit": "Billion USD", "year": 2025, "geography": "Mexico", "url": "https://i.com/pp"},
        {"publisher": "IMARC", "title": "PP", "amount": 2.51, "unit": "Billion USD", "year": 2034, "geography": "Mexico", "url": "https://i.com/pp", "cagr_pct": 4.16},
        {"publisher": "Other", "title": "x", "amount": 1.5, "unit": "USD billion", "year": 2020, "geography": "Mexico"},
    )
    n = m.normalize(df, FX, 2025)
    assert n.loc[0, "at_ref_year"] == pytest.approx(1390 * 1.0416 ** 5)
    assert "source's stated CAGR" in n.loc[0, "adjustment"]
    assert n.loc[3, "at_ref_year"] == 1500 and "not adjusted" in n.loc[3, "adjustment"]  # other source untouched
