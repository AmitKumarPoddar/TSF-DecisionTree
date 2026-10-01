"""AI-assisted market research with Google Gemini (works on the free tier).

Two ways to find market-size figures, chosen automatically:

1. **Google Search grounding** - used when the key's plan allows it (paid
   tier). Gemini searches the web itself.
2. **URL discovery + page reading** - used when search is not available
   (free tier: grounding returns "quota exceeded"). Gemini proposes candidate
   report pages, then reads them with the URL-context tool; only figures from
   pages that were actually retrieved are marked as verified. Analyst-supplied
   URLs are always read too.

Free-tier capacity is limited and sometimes overloaded (HTTP 503), so calls
are retried with back-off and fall back to the Flash-Lite model.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import urlparse

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .ai_research import (
    _FIGURES_SCHEMA, _HIERARCHY_SCHEMA, AIResearchError, ResearchResult, _hierarchy_prompt, _market_prompt,
    clean_figures, extract_json, url_verified,
)

DEFAULT_MODEL = "gemini-flash-lite-latest"   # free-tier friendly alias for the current Flash-Lite model
FALLBACK_MODEL = "gemini-flash-lite-latest"
MAX_URLS = 20                                 # URL-context tool limit per request

# Model families that are not general text models.
_EXCLUDE = re.compile(r"tts|image|transcribe|robotics|computer-use|customtools|embedding|live|omni|aqa|banana", re.I)

# Remember per API key whether Google Search grounding is allowed.
_SEARCH_AVAILABLE: dict[str, bool] = {}


def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def list_text_models(client: genai.Client) -> list[str]:
    """Gemini models this key can use for text generation (names without 'models/')."""
    names = []
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in (m.supported_actions or []) and name.startswith("gemini") and not _EXCLUDE.search(name):
            names.append(name)
    return sorted(set(names), key=lambda n: (not n.endswith("-latest"), n))


def _is_quota(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _is_overloaded(exc: Exception) -> bool:
    return getattr(exc, "code", None) in (500, 503, 504) or "UNAVAILABLE" in str(exc)


class GeminiResearcher:
    """Provider adapter used by the app (same interface as ClaudeResearcher)."""

    provider = "Gemini (Google)"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, client=None, sleep=time.sleep):
        self.api_key = api_key
        self.client = client or make_client(api_key)
        self.model = model or DEFAULT_MODEL
        self.sleep = sleep
        self.requests = 0

    # ------------------------------------------------------------------ #
    def _generate(self, contents: str, config: types.GenerateContentConfig | None = None,
                  allow_quota_retry: bool = True):
        """generate_content with retries; falls back to Flash-Lite when overloaded."""
        models = [self.model] + ([FALLBACK_MODEL] if self.model != FALLBACK_MODEL else [])
        last: Exception | None = None
        for model in models:
            for attempt in range(3):
                try:
                    self.requests += 1
                    return self.client.models.generate_content(model=model, contents=contents, config=config)
                except genai_errors.APIError as exc:
                    last = exc
                    if _is_overloaded(exc):
                        self.sleep(4 * (attempt + 1))
                        continue
                    if _is_quota(exc):
                        if allow_quota_retry and attempt == 0:
                            self.sleep(30)  # per-minute limit: wait and try once more
                            continue
                        raise AIResearchError(
                            "Gemini quota reached for this API key (free-tier per-minute or per-day limit). "
                            "Wait a minute and retry; if it persists, the daily limit is used up - try tomorrow "
                            "or pick another model in the sidebar.") from exc
                    if getattr(exc, "code", None) in (401, 403):
                        raise AIResearchError("The Gemini API key was rejected. Check GEMINI_API_KEY in the app secrets.") from exc
                    if getattr(exc, "code", None) == 404:
                        break  # model not available to this key: try the fallback model
                    raise AIResearchError(f"Gemini error: {str(exc)[:300]}") from exc
                except Exception as exc:  # noqa: BLE001 - network problems
                    last = exc
                    self.sleep(3)
        raise AIResearchError(f"Gemini is unavailable right now (overloaded). Try again in a minute. ({str(last)[:150]})")

    def _search_allowed(self) -> bool | None:
        return _SEARCH_AVAILABLE.get(self.api_key)

    def _grounded(self, prompt: str):
        """Try a Google-Search-grounded call; returns None when the plan doesn't allow search."""
        if self._search_allowed() is False:
            return None
        config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        try:
            response = self._generate(prompt, config, allow_quota_retry=False)
        except AIResearchError as exc:
            if "quota" in str(exc).lower():
                _SEARCH_AVAILABLE[self.api_key] = False
                return None
            raise
        _SEARCH_AVAILABLE[self.api_key] = True
        return response

    def _structure(self, text: str, schema: dict) -> dict:
        config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=schema)
        response = self._generate("Convert this research answer into the required JSON. Keep every figure and URL "
                                  "exactly as written; do not add anything.\n\n" + text, config)
        try:
            return json.loads(response.text or "")
        except ValueError as exc:
            raise AIResearchError("The Gemini answer could not be read as JSON.") from exc

    def _json(self, text: str, schema: dict, key: str) -> dict:
        data = extract_json(text or "")
        if data is None or key not in data:
            data = self._structure(text or "", schema)
        return data

    # ------------------------------------------------------------------ #
    def find_market_sizes(self, material: str, opportunity: str = "", urls: list[str] | None = None) -> ResearchResult:
        urls = [u.strip() for u in (urls or []) if u and u.strip()]
        prompt = _market_prompt(material, opportunity)
        response = None if urls else self._grounded(prompt)
        if response is not None:
            consulted = grounding_sources(response)
            data = self._json(response.text, _FIGURES_SCHEMA, "figures")
            domains = {page["domain"] for page in consulted if page.get("domain")}
            data["figures"] = clean_figures(data.get("figures"), consulted,
                                            verify=lambda url: _domain(url) in domains)
            data["mode"] = "Google Search grounding"
            return ResearchResult(data=data, consulted=consulted, text=response.text or "")

        candidates = urls + [u for u in self._candidate_urls(material) if u not in urls]
        candidates = candidates[:MAX_URLS]
        if not candidates:
            raise AIResearchError("No pages to read: add report URLs in the box and try again.")
        read_prompt = (
            f"Read these pages. List every market-size figure for **{material}** in Mexico (or Latin America / "
            "North America, labelled with the true geography) that is stated on a page you successfully read. "
            "Do not use outside knowledge and do not estimate. For each figure give publisher, title, amount, "
            "unit (\"USD million\", \"USD billion\", \"MXN million\", \"kt (thousand tonnes)\", \"tonnes\"), year the "
            "figure refers to, geography, cagr_pct (number or null), url (the page it appears on) and a one-line "
            "note. Finish with JSON in a ```json block: {\"figures\": [...], \"notes\": \"...\"}.\n\n"
            + "\n".join(candidates))
        response = self._generate(read_prompt, types.GenerateContentConfig(tools=[types.Tool(url_context=types.UrlContext())]))
        retrieved, failed = url_retrieval(response)
        consulted = [{"url": u, "title": "page read"} for u in retrieved]
        data = self._json(response.text, _FIGURES_SCHEMA, "figures")
        data["figures"] = clean_figures(data.get("figures"), consulted)
        data["mode"] = "URL reading (Google Search not available on this key)"
        note = f"Successfully read {len(retrieved)} of {len(candidates)} candidate pages."
        data["notes"] = " ".join(x for x in (str(data.get("notes") or ""), note) if x).strip()
        return ResearchResult(data=data, consulted=consulted, text=response.text or "")

    def _candidate_urls(self, material: str) -> list[str]:
        prompt = (
            f"I need web pages that state a market size for **{material} in Mexico** specifically. List up to 15 "
            "exact URLs: market-research publishers' Mexico-specific report pages (for example "
            "'.../mexico-<product>-market'), Latin America pages with a Mexico breakdown, Mexican industry "
            "associations or government statistics, and news articles quoting the Mexican market value. "
            "Return JSON only: {\"urls\": [\"https://...\"]}")
        config = types.GenerateContentConfig(response_mime_type="application/json")
        response = self._generate(prompt, config)
        data = extract_json(response.text or "") or {}
        return [u for u in data.get("urls", []) if isinstance(u, str) and u.startswith("http")]

    def suggest_hierarchy(self, opportunity: str, base_material: str = "") -> ResearchResult:
        prompt = _hierarchy_prompt(opportunity, base_material)
        response = self._grounded(prompt)
        if response is not None:
            consulted = grounding_sources(response)
            data = self._json(response.text, _HIERARCHY_SCHEMA, "products")
            mode = "Google Search grounding"
        else:
            config = types.GenerateContentConfig(response_mime_type="application/json",
                                                 response_json_schema=_HIERARCHY_SCHEMA)
            response = self._generate(prompt.replace("using web search where useful", "using your knowledge"), config)
            consulted = []
            data = self._json(response.text, _HIERARCHY_SCHEMA, "products")
            mode = "Model knowledge (no web search on this key)"
        for key in ("categories", "products", "sources"):
            if not isinstance(data.get(key), list):
                data[key] = []
        data["categories"] = [str(c).strip() for c in data["categories"] if str(c).strip()]
        data["products"] = [str(p).strip() for p in data["products"] if str(p).strip()]
        for src in data["sources"]:
            if isinstance(src, dict):
                src["verified_url"] = url_verified(src.get("url", ""), consulted)
        data["mode"] = mode
        if mode.startswith("Model knowledge"):
            data["rationale"] = (str(data.get("rationale") or "") + " (Based on model knowledge, not web sources - "
                                 "please review.)").strip()
        return ResearchResult(data=data, consulted=consulted, text=response.text or "")


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    host = urlparse(str(url or "")).netloc.lower()
    return host.removeprefix("www.")


def grounding_sources(response) -> list[dict]:
    """Pages Google Search returned (titles are usually the site's domain)."""
    pages = []
    for cand in getattr(response, "candidates", None) or []:
        gm = getattr(cand, "grounding_metadata", None)
        for chunk in getattr(gm, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            domain = (getattr(web, "domain", None) or getattr(web, "title", None) or "").lower().removeprefix("www.")
            pages.append({"url": getattr(web, "uri", "") or "", "title": getattr(web, "title", "") or "",
                          "domain": domain})
    return pages


def url_retrieval(response) -> tuple[list[str], list[str]]:
    """(retrieved, failed) URLs reported by the URL-context tool."""
    ok, failed = [], []
    for cand in getattr(response, "candidates", None) or []:
        meta = getattr(cand, "url_context_metadata", None)
        for item in getattr(meta, "url_metadata", None) or []:
            status = str(getattr(item, "url_retrieval_status", ""))
            (ok if status.endswith("SUCCESS") else failed).append(getattr(item, "retrieved_url", ""))
    return ok, failed
