"""AI-assisted market research with Claude and its server-side web search tool.

Two research tasks feed the recurring-demand workflow:

* :func:`find_market_sizes` - published market-size figures for a material
  in Mexico, each tied to the web page it came from.
* :func:`suggest_hierarchy` - the immediate base material, its broad product
  categories (relevance level 1) and the opportunity's parallel products
  (relevance level 2).

Every figure's URL is checked against the pages the search actually returned
(``verified_url``); the user reviews everything before it is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

MODEL = "claude-opus-5-5"
MAX_SEARCHES = 8
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AIResearchError(RuntimeError):
    """Raised when the research request cannot be completed."""


@dataclass
class ResearchResult:
    data: dict
    consulted: list[dict] = field(default_factory=list)  # [{"url", "title"}] pages the search returned
    text: str = ""


# --------------------------------------------------------------------------- #
# Prompts and schemas
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are a petrochemical market analyst supporting a market-entry study for Mexico. "
    "Use web search to find published figures. Report only figures you actually found on a "
    "page returned by your searches, each with the exact URL of that page. Never estimate, "
    "derive or invent a number; if nothing is published, return an empty list."
)

_FIGURES_SCHEMA = {
    "type": "object",
    "properties": {
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "publisher": {"type": "string"},
                    "title": {"type": "string"},
                    "amount": {"type": "number"},
                    "unit": {"type": "string"},
                    "year": {"type": ["integer", "null"]},
                    "geography": {"type": "string"},
                    "cagr_pct": {"type": ["number", "null"]},
                    "url": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["publisher", "title", "amount", "unit", "year", "geography",
                             "cagr_pct", "url", "note"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["figures", "notes"],
    "additionalProperties": False,
}

_HIERARCHY_SCHEMA = {
    "type": "object",
    "properties": {
        "base_material": {"type": "string"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "opportunity_category": {"type": "string"},
        "products": {"type": "array", "items": {"type": "string"}},
        "opportunity_product": {"type": "string"},
        "rationale": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "url": {"type": "string"}},
                "required": ["title", "url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["base_material", "categories", "opportunity_category", "products",
                 "opportunity_product", "rationale", "sources"],
    "additionalProperties": False,
}


def _market_prompt(material: str, opportunity_context: str = "") -> str:
    context = f"\nContext: this supports the evaluation of '{opportunity_context}'." if opportunity_context else ""
    return f"""Find published market-size estimates for **{material}** in **Mexico**.{context}

Search market-research publishers, industry associations (e.g. ANIQ), government statistics and
company disclosures. List every distinct figure you find - different publishers often disagree, and
all of them are wanted. If no Mexico-specific figure exists, you may include Latin America or North
America figures, labelled with their true geography.

For each figure give:
- publisher and report/page title
- amount and unit: money as "<ISO currency> million" or "<ISO currency> billion" (e.g. "USD million",
  "MXN billion"); volume as "kt (thousand tonnes)", "tonnes" or "Mt (million tonnes)"
- year: the year the figure refers to (not the publication year); a forecast figure is a separate entry
- geography exactly as stated by the source
- cagr_pct: the growth rate the source states (number, e.g. 4.5), or null
- url: the page where the figure appears
- note: one line of context (e.g. forecast period, scope such as "compounds only")

Finish with the result as JSON in a ```json code block with this shape:
{{"figures": [{{"publisher": "", "title": "", "amount": 0, "unit": "USD million", "year": 2024,
"geography": "Mexico", "cagr_pct": null, "url": "", "note": ""}}], "notes": "coverage caveats"}}"""


def _hierarchy_prompt(opportunity: str, base_material: str = "") -> str:
    hint = f" The user believes the immediate base material is '{base_material}'." if base_material else ""
    return f"""The opportunity under evaluation is **{opportunity}**.{hint}

Determine, using web search where useful:
1. base_material: the immediate base material the opportunity is made from (one step up, e.g.
   mineral-filled polypropylene -> polypropylene).
2. categories (relevance level 1): the broad categories the base material's market divides into,
   at the same level as each other, one of which contains the opportunity (e.g. for polypropylene:
   injection-moulding grades, film and sheet grades, fibre and raffia grades, compounds and
   modified grades). 3-8 entries.
3. opportunity_category: which of those categories contains the opportunity.
4. products (relevance level 2): the immediate derivative products within that category that are
   parallel to the opportunity, including the opportunity itself (e.g. additive concentrates,
   glass-fibre reinforced PP, mineral-filled PP compounds, PP-based TPO, PP-based TPV). These are
   intermediate products, not consumer end products. 3-10 entries.
5. opportunity_product: the entry in products that is the opportunity.
6. rationale: two or three sentences.
7. sources: pages that support the split.

Finish with the result as JSON in a ```json code block with keys base_material, categories,
opportunity_category, products, opportunity_product, rationale, sources (list of {{title, url}})."""


# --------------------------------------------------------------------------- #
# Running a search conversation
# --------------------------------------------------------------------------- #
def make_client(api_key: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def _collect_urls(content, consulted: dict[str, str]) -> None:
    """Record pages returned by web search and cited in text blocks."""
    for block in content:
        btype = getattr(block, "type", "")
        if btype == "web_search_tool_result":
            results = getattr(block, "content", None)
            if isinstance(results, list):  # an error result is a single object
                for item in results:
                    url = getattr(item, "url", None)
                    if url:
                        consulted.setdefault(url, getattr(item, "title", "") or "")
        elif btype == "text":
            for cit in getattr(block, "citations", None) or []:
                url = getattr(cit, "url", None)
                if url:
                    consulted.setdefault(url, getattr(cit, "title", "") or "")


def _run_search(client: anthropic.Anthropic, prompt: str, effort: str = "high",
                max_searches: int = MAX_SEARCHES) -> tuple[str, list[dict]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    consulted: dict[str, str] = {}
    response = None
    try:
        for _ in range(4):  # continue up to 3 times if a long search turn pauses
            with client.beta.messages.stream(
                model=MODEL,
                max_tokens=32000,
                system=_SYSTEM,
                messages=messages,
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}],
                output_config={"effort": effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                response = stream.get_final_message()
            _collect_urls(response.content, consulted)
            if response.stop_reason == "pause_turn":
                messages = [messages[0], {"role": "assistant", "content": response.content}]
                continue
            break
    except anthropic.AuthenticationError as exc:
        raise AIResearchError("The Anthropic API key was rejected. Check the key in the sidebar or app secrets.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise AIResearchError("The API key is not allowed to use this model or web search.") from exc
    except anthropic.RateLimitError as exc:
        raise AIResearchError("The AI service is rate-limiting requests. Wait a minute and try again.") from exc
    except anthropic.APIStatusError as exc:
        raise AIResearchError(f"AI service error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise AIResearchError("Could not reach the AI service (network error).") from exc

    if response is None:
        raise AIResearchError("The AI service returned no response.")
    if response.stop_reason == "refusal":
        raise AIResearchError("The AI service declined this request.")
    text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
    return text, [{"url": u, "title": t} for u, t in consulted.items()]


def extract_json(text: str) -> dict | None:
    """Pull the last JSON object out of a model answer (fenced block preferred)."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = list(reversed(blocks))
    start = text.find("{")
    if start != -1:
        candidates.append(text[start:text.rfind("}") + 1])
    for raw in candidates:
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _structure(client: anthropic.Anthropic, text: str, schema: dict) -> dict:
    """Second pass: convert a free-text answer into schema-valid JSON."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content":
                       "Convert this research answer into the required JSON. Keep every figure and URL "
                       "exactly as written; do not add anything.\n\n" + text}],
        )
    except anthropic.APIError as exc:
        raise AIResearchError(f"Could not structure the AI answer: {exc}") from exc
    out = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
    try:
        return json.loads(out)
    except ValueError as exc:
        raise AIResearchError("The AI answer could not be read as JSON.") from exc


def _norm_url(url: str) -> str:
    url = str(url or "").strip().lower().split("#")[0]
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.rstrip("/")


def url_verified(url: str, consulted: list[dict]) -> bool:
    """True when ``url`` is one of (or a sub-page of) the pages the search returned."""
    target = _norm_url(url)
    if not target:
        return False
    for page in consulted:
        seen = _norm_url(page.get("url", ""))
        if seen and (target == seen or target.startswith(seen + "/") or seen.startswith(target + "/")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Public tasks
# --------------------------------------------------------------------------- #
def clean_figures(raw_figures, consulted: list[dict], verify=None) -> list[dict]:
    """Normalise model-reported figures into rows for the figures table.

    ``verify(url) -> bool`` decides ``verified_url``; by default the URL must
    be one of the ``consulted`` pages.
    """
    verify = verify or (lambda url: url_verified(url, consulted))
    figures = []
    for fig in raw_figures or []:
        if not isinstance(fig, dict):
            continue
        try:
            amount = float(fig.get("amount"))
        except (TypeError, ValueError):
            continue
        figures.append({
            "use": False,
            "publisher": str(fig.get("publisher") or "").strip(),
            "title": str(fig.get("title") or "").strip(),
            "amount": amount,
            "unit": str(fig.get("unit") or "").strip(),
            "year": fig.get("year"),
            "geography": str(fig.get("geography") or "").strip(),
            "cagr_pct": fig.get("cagr_pct"),
            "url": str(fig.get("url") or "").strip(),
            "origin": "AI",
            "verified_url": bool(verify(fig.get("url", ""))),
            "note": str(fig.get("note") or "").strip(),
        })
    return figures


def find_market_sizes(client: anthropic.Anthropic, material: str, opportunity: str = "",
                      urls: list[str] | None = None) -> ResearchResult:
    prompt = _market_prompt(material, opportunity)
    if urls:
        prompt += "\n\nAlso read these pages the analyst found:\n" + "\n".join(urls)
    text, consulted = _run_search(client, prompt)
    data = extract_json(text)
    if data is None or "figures" not in data:
        data = _structure(client, text, _FIGURES_SCHEMA)
    data["figures"] = clean_figures(data.get("figures"), consulted)
    data["mode"] = "Claude web search"
    return ResearchResult(data=data, consulted=consulted, text=text)


def suggest_hierarchy(client: anthropic.Anthropic, opportunity: str, base_material: str = "") -> ResearchResult:
    text, consulted = _run_search(client, _hierarchy_prompt(opportunity, base_material), max_searches=5)
    data = extract_json(text)
    if data is None or "products" not in data:
        data = _structure(client, text, _HIERARCHY_SCHEMA)
    for key in ("categories", "products", "sources"):
        if not isinstance(data.get(key), list):
            data[key] = []
    data["categories"] = [str(c).strip() for c in data["categories"] if str(c).strip()]
    data["products"] = [str(p).strip() for p in data["products"] if str(p).strip()]
    for src in data["sources"]:
        if isinstance(src, dict):
            src["verified_url"] = url_verified(src.get("url", ""), consulted)
    return ResearchResult(data=data, consulted=consulted, text=text)


class ClaudeResearcher:
    """Provider adapter used by the app (same interface as GeminiResearcher)."""

    provider = "Claude (Anthropic)"

    def __init__(self, api_key: str, model: str = MODEL):
        self.client = make_client(api_key)
        self.model = model

    def find_market_sizes(self, material: str, opportunity: str = "", urls: list[str] | None = None) -> ResearchResult:
        return find_market_sizes(self.client, material, opportunity, urls)

    def suggest_hierarchy(self, opportunity: str, base_material: str = "") -> ResearchResult:
        return suggest_hierarchy(self.client, opportunity, base_material)
