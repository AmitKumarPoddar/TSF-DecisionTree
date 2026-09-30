"""End-to-end tests against the mock server, for both Tesseract generations."""

import pytest

from datamexico import service
from datamexico.analysis import IMPORTS, partial_years_from_coverage, yearly_summary
from datamexico.client import DataMexicoError, TesseractClient
from datamexico.hs import search


def test_connect_skips_dead_base_url(mock_api):
    client = TesseractClient.connect(["http://127.0.0.1:9/tesseract", mock_api], timeout=2)
    assert client.base_url == mock_api


def test_connect_reports_every_failure():
    with pytest.raises(DataMexicoError, match="127.0.0.1:9"):
        TesseractClient.connect(["http://127.0.0.1:9/tesseract"], timeout=2)


def test_discovers_state_level_trade_cube(mock_api):
    cubes = service.trade_cubes(TesseractClient(mock_api))
    best = cubes[0]
    assert best.name == "economy_foreign_trade_ent"
    assert {d: lvl.name for d, lvl in best.hs_levels.items()} == {2: "HS2", 4: "HS4", 6: "HS6"}
    assert best.flow.name == "Flow" and best.year.name == "Year"
    assert best.country.name == "Country" and best.state.name == "State"
    assert best.period.name == "Quarter"
    assert best.value_measure == "Trade Value"
    assert "economy_foreign_trade_mun" in [c.name for c in cubes]


def test_search_and_fetch_toluene(mock_api):
    client = TesseractClient(mock_api)
    tc = service.trade_cubes(client)[0]
    entries = service.hs_entries(client, tc, "en")
    toluene = search(entries, "tolueno")
    assert [(e.code, e.label) for e in toluene] == [("290230", "Toluene")]

    flows = service.flow_map(client, tc, "en")
    assert sorted(flows.values()) == ["Exports", "Imports"]

    by_hs = service.fetch_trade(client, tc, toluene, "Trade Value", flows, "hs")
    assert set(by_hs["code"]) == {"290230"}
    summary = yearly_summary(by_hs)
    assert list(summary.index) == list(range(2018, 2026))
    assert (summary["imports"] > summary["exports"]).all()

    by_country = service.fetch_trade(client, tc, toluene, "Trade Value", flows, "country")
    totals = by_country.groupby("year")["value"].sum()
    assert totals.round(0).equals(by_hs.groupby("year")["value"].sum().round(0))

    by_state = service.fetch_trade(client, tc, toluene, "Trade Value", flows, "state")
    tamaulipas_imports = by_state[(by_state["state"] == "Tamaulipas") & (by_state["flow"] == IMPORTS)]
    assert tamaulipas_imports.empty


def test_mixed_levels_one_request_per_level(mock_api):
    client = TesseractClient(mock_api)
    tc = service.trade_cubes(client)[0]
    entries = service.hs_entries(client, tc, "en")
    picked = [e for e in search(entries, "polypropylene")] + [e for e in search(entries, "2711", levels=[4])]
    flows = service.flow_map(client, tc, "en")
    df = service.fetch_trade(client, tc, picked, "Trade Value", flows, "hs")
    assert set(df["code"]) == {"390210", "2711"}
    assert set(df["hs_digits"]) == {4, 6}


def test_partial_year_detected(mock_api):
    client = TesseractClient(mock_api)
    tc = service.trade_cubes(client)[0]
    toluene = search(service.hs_entries(client, tc, "en"), "toluene")
    coverage = service.fetch_coverage(client, tc, toluene, "Trade Value")
    assert coverage[2024] == 4 and coverage[2025] == 2
    assert partial_years_from_coverage(coverage) == [2025]


def test_query_log_records_reproducible_urls(mock_api):
    client = TesseractClient(mock_api)
    tc = service.trade_cubes(client)[0]
    client.data(tc.name, ["Year", "Flow"], ["Trade Value"], {"HS6": ["6290230"]}, "en")
    url = client.log.urls[-1]
    assert "data.jsonrecords?" in url and "HS6=6290230" in url and "drilldowns=Year,Flow" in url


def test_bad_level_raises(mock_api):
    client = TesseractClient(mock_api)
    with pytest.raises(DataMexicoError):
        client.data("economy_foreign_trade_ent", ["Nope"], ["Trade Value"])
