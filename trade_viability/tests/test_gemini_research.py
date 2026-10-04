"""Gemini research with a fake client (no network, no quota used)."""

import json
from types import SimpleNamespace as NS

import pytest
from google.genai import errors as genai_errors

from datamexico import gemini_research as gr
from datamexico.ai_research import AIResearchError

FIGS = {"figures": [
    {"publisher": "IMARC Group", "title": "Mexico PP market", "amount": 1.71, "unit": "Billion USD", "year": 2025,
     "geography": "Mexico", "cagr_pct": 4.16, "url": "https://www.imarcgroup.com/mexico-polypropylene-market",
     "note": ""},
    {"publisher": "Made Up", "title": "x", "amount": 1, "unit": "USD million", "year": 2024, "geography": "Mexico",
     "cagr_pct": None, "url": "https://nowhere.example/x", "note": ""},
], "notes": ""}


def _err(code, status):
    return genai_errors.ClientError(code, {"error": {"code": code, "status": status, "message": status}}) \
        if code < 500 else genai_errors.ServerError(code, {"error": {"code": code, "status": status, "message": status}})


def _resp(text, grounding=None, url_meta=None):
    gm = NS(grounding_chunks=[NS(web=NS(uri=u, title=d, domain=d)) for u, d in (grounding or [])])
    um = NS(url_metadata=[NS(retrieved_url=u, url_retrieval_status=f"UrlRetrievalStatus.URL_RETRIEVAL_STATUS_{s}")
                          for u, s in (url_meta or [])])
    return NS(text=text, candidates=[NS(grounding_metadata=gm, url_context_metadata=um)])


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config=None):
        tools = [t for t in (getattr(config, "tools", None) or [])]
        kind = ("search" if any(getattr(t, "google_search", None) for t in tools)
                else "url" if any(getattr(t, "url_context", None) for t in tools) else "plain")
        self.calls.append((model, kind))
        out = self.script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _researcher(script, key="k1", model="gemini-flash-lite-latest"):
    gr._SEARCH_AVAILABLE.clear()
    models = FakeModels(script)
    r = gr.GeminiResearcher(key, model, client=NS(models=models), sleep=lambda s: None)
    return r, models


def test_grounded_search_used_when_available():
    answer = "```json\n" + json.dumps(FIGS) + "\n```"
    r, models = _researcher([_resp(answer, grounding=[("https://vertexaisearch.cloud.google.com/r/1", "imarcgroup.com")])])
    res = r.find_market_sizes("polypropylene")
    assert res.data["mode"] == "Google Search grounding"
    assert [f["verified_url"] for f in res.data["figures"]] == [True, False]  # matched by domain
    assert models.calls == [("gemini-flash-lite-latest", "search")]


def test_free_tier_falls_back_to_url_reading_and_remembers():
    urls = json.dumps({"urls": ["https://www.imarcgroup.com/mexico-polypropylene-market", "https://dead.example/x"]})
    answer = "```json\n" + json.dumps(FIGS) + "\n```"
    r, models = _researcher([
        _err(429, "RESOURCE_EXHAUSTED"),                      # grounding not allowed on the free plan
        _resp(urls),                                           # candidate URLs
        _resp(answer, url_meta=[("https://www.imarcgroup.com/mexico-polypropylene-market", "SUCCESS"),
                                ("https://dead.example/x", "ERROR")]),
    ])
    res = r.find_market_sizes("polypropylene")
    assert res.data["mode"].startswith("URL reading")
    assert [f["verified_url"] for f in res.data["figures"]] == [True, False]
    assert "read 1 of 2" in res.data["notes"]
    assert [k for _, k in models.calls] == ["search", "plain", "url"]
    # second call skips the grounding attempt
    models.script = [_resp(urls), _resp(answer, url_meta=[])]
    r.find_market_sizes("toluene")
    assert [k for _, k in models.calls[3:]] == ["plain", "url"]


def test_user_urls_are_read_first_without_search():
    answer = "```json\n" + json.dumps({"figures": [], "notes": ""}) + "\n```"
    r, models = _researcher([_resp(json.dumps({"urls": []})), _resp(answer, url_meta=[("https://a.com", "SUCCESS")])])
    res = r.find_market_sizes("pp", urls=["https://a.com"])
    assert [k for _, k in models.calls] == ["plain", "url"]
    assert res.consulted == [{"url": "https://a.com", "title": "page read"}]


def test_overloaded_model_retries_then_falls_back_to_flash_lite():
    answer = "```json\n" + json.dumps({"figures": [], "notes": ""}) + "\n```"
    gr._SEARCH_AVAILABLE["k1"] = False
    r, models = _researcher([_err(503, "UNAVAILABLE")] * 3 + [_resp(json.dumps({"urls": ["https://a.com"]})),
                                                               _resp(answer)], model="gemini-3.5-flash")
    gr._SEARCH_AVAILABLE["k1"] = False
    r.find_market_sizes("pp")
    assert [m for m, _ in models.calls[:4]] == ["gemini-3.5-flash"] * 3 + ["gemini-flash-lite-latest"]


def test_quota_error_message_after_retry():
    r, _ = _researcher([_err(429, "RESOURCE_EXHAUSTED"), _err(429, "RESOURCE_EXHAUSTED")])
    gr._SEARCH_AVAILABLE["k1"] = False
    with pytest.raises(AIResearchError, match="quota"):
        r.find_market_sizes("pp")


def test_hierarchy_without_search_is_flagged_as_model_knowledge():
    data = {"base_material": "Polypropylene", "categories": ["A", "B"], "opportunity_category": "B",
            "products": ["x", "y", "z"], "opportunity_product": "y", "rationale": "r", "sources": []}
    r, models = _researcher([_err(429, "RESOURCE_EXHAUSTED"), _resp(json.dumps(data))])
    res = r.suggest_hierarchy("Mineral-filled PP")
    assert res.data["mode"].startswith("Model knowledge")
    assert "not web sources" in res.data["rationale"]
    assert len(res.data["products"]) == 3


def test_list_text_models_filters_non_text():
    models = [NS(name="models/gemini-flash-lite-latest", supported_actions=["generateContent"]),
              NS(name="models/gemini-3.5-flash", supported_actions=["generateContent"]),
              NS(name="models/gemini-3.8-flash-tts", supported_actions=["generateContent"]),
              NS(name="models/gemini-3-pro-image", supported_actions=["generateContent"]),
              NS(name="models/text-embedding-004", supported_actions=["embedContent"])]
    client = NS(models=NS(list=lambda: models))
    assert gr.list_text_models(client) == ["gemini-flash-lite-latest", "gemini-3.5-flash"]


def test_competitors_free_tier_confirms_by_reading_pages():
    cands = {"companies": [
        {"company": "Compounder MX", "group": "Compounder MX", "type": "Mexican producer/compounder", "country": "Mexico",
         "product": "talc-filled PP", "url": "https://cmx.mx/pp", "evidence": "", "sells_in_mexico": True},
        {"company": "Ghost Co", "group": "", "type": "Other", "country": "US", "product": "x",
         "url": "https://ghost.example/x", "evidence": "", "sells_in_mexico": True}], "notes": ""}
    read = {"companies": [cands["companies"][0]], "notes": "one confirmed"}
    r, models = _researcher([
        _err(429, "RESOURCE_EXHAUSTED"),                       # no search on free plan
        _resp(json.dumps(cands)),                              # candidates (JSON mode)
        _resp("```json\n" + json.dumps(read) + "\n```", url_meta=[("https://cmx.mx/pp", "SUCCESS"),
                                                                  ("https://ghost.example/x", "ERROR")]),
    ])
    res = r.find_competitors("Mineral-filled PP", "Polypropylene")
    comps = {c["company"]: c for c in res.data["companies"]}
    assert comps["Compounder MX"]["include"] and comps["Compounder MX"]["verified_url"]
    assert not comps["Ghost Co"]["include"] and "not confirmed" in comps["Ghost Co"]["note"]
    assert "Confirmed 1 of 2" in res.data["notes"]
    assert [k for _, k in models.calls] == ["search", "plain", "url"]
