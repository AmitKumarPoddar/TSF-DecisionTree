"""AI research parsing, using a fake Anthropic client (no network, no cost)."""

import json
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

from datamexico import ai_research as ai


def _search_block(*urls):
    return NS(type="web_search_tool_result", content=[NS(url=u, title=f"title {u}") for u in urls])


def _text(t):
    return NS(type="text", text=t, citations=None)


class FakeClient:
    """Returns the queued responses for stream() calls, then for create() calls."""

    def __init__(self, stream_responses, create_responses=()):
        self.stream_responses = list(stream_responses)
        self.create_responses = list(create_responses)
        self.stream_calls = []
        self.beta = NS(messages=NS(stream=self._stream))
        self.messages = NS(create=self._create)

    @contextmanager
    def _stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        msg = self.stream_responses.pop(0)
        yield NS(get_final_message=lambda: msg)

    def _create(self, **kwargs):
        return self.create_responses.pop(0)


FIGS = {"figures": [
    {"publisher": "Pub A", "title": "PP Mexico", "amount": 2.1, "unit": "USD billion", "year": 2024,
     "geography": "Mexico", "cagr_pct": 4.2, "url": "https://pub-a.com/pp-mexico/", "note": ""},
    {"publisher": "Pub B", "title": "PP LatAm", "amount": 9000, "unit": "USD million", "year": 2023,
     "geography": "Latin America", "cagr_pct": None, "url": "https://made-up.example/x", "note": ""},
], "notes": "few Mexico sources"}


def test_find_market_sizes_parses_and_verifies_urls():
    answer = "Found two.\n```json\n" + json.dumps(FIGS) + "\n```"
    client = FakeClient([NS(stop_reason="end_turn", content=[_search_block("https://www.pub-a.com/pp-mexico"),
                                                              _text(answer)])])
    res = ai.find_market_sizes(client, "polypropylene")
    figs = res.data["figures"]
    assert [f["publisher"] for f in figs] == ["Pub A", "Pub B"]
    assert figs[0]["verified_url"] is True      # same page, www/trailing slash ignored
    assert figs[1]["verified_url"] is False     # URL never returned by search
    assert all(f["origin"] == "AI" and f["use"] is False for f in figs)
    call = client.stream_calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["tools"][0]["type"] == "web_search_20260209"
    assert call["fallbacks"] == "default"


def test_pause_turn_is_continued():
    answer = "```json\n" + json.dumps({"figures": [], "notes": ""}) + "\n```"
    client = FakeClient([
        NS(stop_reason="pause_turn", content=[_search_block("https://a.com")]),
        NS(stop_reason="end_turn", content=[_search_block("https://b.com"), _text(answer)]),
    ])
    res = ai.find_market_sizes(client, "toluene")
    assert len(client.stream_calls) == 2
    assert client.stream_calls[1]["messages"][1]["role"] == "assistant"
    assert {p["url"] for p in res.consulted} == {"https://a.com", "https://b.com"}


def test_unparseable_answer_falls_back_to_structured_output():
    client = FakeClient(
        [NS(stop_reason="end_turn", content=[_text("Pub A says USD 2.1 billion in 2024.")])],
        [NS(content=[_text(json.dumps(FIGS))])],
    )
    res = ai.find_market_sizes(client, "polypropylene")
    assert len(res.data["figures"]) == 2


def test_refusal_raises():
    client = FakeClient([NS(stop_reason="refusal", content=[])])
    with pytest.raises(ai.AIResearchError, match="declined"):
        ai.find_market_sizes(client, "x")


def test_suggest_hierarchy():
    data = {"base_material": "Polypropylene", "categories": ["Injection", "Film", "Fibre", "Compounds"],
            "opportunity_category": "Compounds",
            "products": ["Additive concentrates", "Glass-fibre PP", "Mineral-filled PP", "PP TPO", "PP TPV"],
            "opportunity_product": "Mineral-filled PP", "rationale": "r",
            "sources": [{"title": "s", "url": "https://src.com/a"}]}
    client = FakeClient([NS(stop_reason="end_turn", content=[_search_block("https://src.com/a"),
                                                              _text("```json\n" + json.dumps(data) + "\n```")])])
    res = ai.suggest_hierarchy(client, "Mineral-filled polypropylene")
    assert len(res.data["categories"]) == 4 and len(res.data["products"]) == 5
    assert res.data["sources"][0]["verified_url"] is True


def test_extract_json_prefers_last_fenced_block():
    text = '```json\n{"a": 1}\n``` then ```json\n{"a": 2}\n```'
    assert ai.extract_json(text) == {"a": 2}
    assert ai.extract_json("no json here") is None


def test_url_verified_subpage():
    consulted = [{"url": "https://x.com/report"}]
    assert ai.url_verified("https://x.com/report/page-2", consulted)
    assert not ai.url_verified("https://y.com/report", consulted)
    assert not ai.url_verified("", consulted)


def test_find_competitors_claude():
    data = {"companies": [
        {"company": "A SA de CV", "group": "A", "type": "Distributor/importer in Mexico", "country": "Mexico",
         "product": "mineral-filled PP", "url": "https://a.mx/p", "evidence": "catalogue", "sells_in_mexico": True},
        {"company": "B", "group": "B", "type": "Other", "country": "US", "product": "PP", "url": "https://b.com",
         "evidence": "", "sells_in_mexico": False}], "notes": ""}
    client = FakeClient([NS(stop_reason="end_turn", content=[_search_block("https://a.mx/p"),
                                                              _text("```json\n" + json.dumps(data) + "\n```")])])
    res = ai.find_competitors(client, "Mineral-filled PP", "Polypropylene")
    a, b = res.data["companies"]
    assert a["include"] and a["verified_url"]
    assert not b["include"] and "not confirmed" in b["note"]
    assert "Do NOT count companies that only sell Polypropylene" in client.stream_calls[0]["messages"][0]["content"]
