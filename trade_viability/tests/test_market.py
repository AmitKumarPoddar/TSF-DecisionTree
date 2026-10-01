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


def test_derived_market_uses_two_level_relevance():
    base = _figs({"amount": 1000, "unit": "USD million", "year": 2025, "geography": "Mexico"},
                 {"amount": 2000, "unit": "USD million", "year": 2025, "geography": "Mexico"})
    stats = m.market_stats(m.normalize(base, FX, 2025), "value")
    rel = m.Relevance(n_categories=4, r1=m.default_share(4), n_products=5, r2=m.default_share(5))
    res = m.market_result(None, stats, rel)
    assert res.method == "derived"
    assert res.estimate == pytest.approx(1500 * 0.25 * 0.2)
    assert (res.low, res.high) == (pytest.approx(50), pytest.approx(100))
    rel.level2 = False
    assert m.market_result(None, stats, rel).estimate == pytest.approx(375)


def test_exact_market_takes_precedence():
    exact = m.market_stats(m.normalize(_figs({"amount": 40, "unit": "USD million", "year": 2025,
                                              "geography": "Mexico"}), FX, 2025), "value")
    res = m.market_result(exact, exact, m.Relevance(5, 0.2, 5, 0.2))
    assert res.method == "exact" and res.estimate == 40


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


@pytest.mark.parametrize("market_found,recurring,code", [
    (True, True, m.ESTABLISHED),
    (True, False, m.NOT_RECURRING),
    (False, True, m.TRADE_ONLY),
    (False, False, m.NOT_ESTABLISHED),
])
def test_verdict_matrix(market_found, recurring, code):
    market = m.MarketResult("exact" if market_found else "none", None, None,
                            10.0 if market_found else None, None, None, None)
    s = _summary([10] * 5 if recurring else [0] * 5, [0] * 5)
    v = m.recurring_demand_verdict(market, m.trade_recurrence(s, list(range(2019, 2024))))
    assert v.code == code
    if code == m.NOT_ESTABLISHED:
        assert v.headline == "The recurring demand could not be established."


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
