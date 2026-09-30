"""HTTP client for the Data México Tesseract API.

Data México (Secretaría de Economía) publishes its datasets through a
Tesseract OLAP server. Two generations of the server exist and Data México
has used both:

* tesseract-rs (Rust) "logic layer":
    GET {base}/cubes
    GET {base}/members.jsonrecords?cube=..&level=..&locale=..
        -> {"data": [{"ID": .., "Label": ..}, ...]}
    GET {base}/data.jsonrecords?cube=..&drilldowns=A,B&measures=M&Level=id1,id2
        -> {"data": [{"A ID": .., "A": .., "M": ..}, ...], "source": [...]}

* tesseract-olap (Python):
    GET {base}/cubes
    GET {base}/members?cube=..&level=..&locale=..
        -> {"name": .., "members": [{"key": .., "caption": ..}, ...]}
    GET {base}/data.jsonrecords?cube=..&drilldowns=A,B&measures=M&Level=id1,id2
        -> {"columns": [...], "data": [...], "page": {...}}

Both accept cuts as ``Level=id1,id2`` query parameters, so one query builder
serves both; only the response shapes differ and are normalised here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlencode

import requests

# Candidate API roots, tried in order. The first one is where the API lives
# since Data México moved under economia.gob.mx; the second is the original
# host. Override with the DATAMEXICO_API_BASE environment variable.
DEFAULT_BASE_URLS: tuple[str, ...] = (
    "https://www.economia.gob.mx/apidatamexico/tesseract",
    "https://api.datamexico.org/tesseract",
)

ENV_BASE_URL = "DATAMEXICO_API_BASE"

USER_AGENT = "trade-viability-explorer/1.0 (+https://www.economia.gob.mx/datamexico/)"


class DataMexicoError(RuntimeError):
    """Raised when the API cannot be reached or returns an error."""


@dataclass
class Member:
    """One member of a level, e.g. an HS6 product or a country."""

    id: str
    label: str


@dataclass
class QueryLog:
    """Keeps the URLs requested so the UI can show how to reproduce a result."""

    urls: list[str] = field(default_factory=list)

    def add(self, url: str) -> None:
        self.urls.append(url)


def candidate_base_urls(extra: Iterable[str] = ()) -> list[str]:
    """Return base URLs to try, most specific first, without duplicates."""
    ordered: list[str] = []
    env = os.environ.get(ENV_BASE_URL, "").strip()
    for url in (*extra, env, *DEFAULT_BASE_URLS):
        url = (url or "").strip().rstrip("/")
        if url and url not in ordered:
            ordered.append(url)
    return ordered


class TesseractClient:
    """Thin wrapper over the Tesseract REST API used by Data México."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        session: requests.Session | None = None,
        log: QueryLog | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.session.headers.setdefault("Accept", "application/json")
        self.log = log or QueryLog()
        self.samples: dict[str, str] = {}  # url -> start of raw response, for diagnostics

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(
        cls,
        base_urls: Sequence[str] | None = None,
        timeout: float = 20.0,
    ) -> "TesseractClient":
        """Return a client for the first base URL whose /cubes endpoint answers."""
        errors: list[str] = []
        for url in base_urls or candidate_base_urls():
            client = cls(url, timeout=timeout)
            try:
                cubes = client.cubes()
            except DataMexicoError as exc:
                errors.append(f"{url}: {exc}")
                continue
            if cubes:
                client.timeout = max(timeout, 60.0)
                return client
            errors.append(f"{url}: no cubes returned")
        raise DataMexicoError(
            "Could not reach the Data México API. Tried:\n  " + "\n  ".join(errors)
        )

    # ------------------------------------------------------------------ #
    # Low level
    # ------------------------------------------------------------------ #
    def url_for(self, path: str, params: Mapping[str, str] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url += "?" + urlencode(params, safe=",:;")
        return url

    def get_json(self, path: str, params: Mapping[str, str] | None = None):
        url = self.url_for(path, params)
        self.log.add(url)
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise DataMexicoError(f"request failed: {exc}") from exc
        if response.status_code >= 400:
            raise DataMexicoError(
                f"HTTP {response.status_code} for {url}: {_error_detail(response)}"
            )
        self.samples[url] = response.text[:1500]
        try:
            return response.json()
        except ValueError as exc:
            raise DataMexicoError(f"non-JSON response from {url}") from exc

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def cubes(self, locale: str | None = None) -> list[dict]:
        """Return the raw cube metadata list."""
        params = {"locale": locale} if locale else None
        payload = self.get_json("cubes", params)
        if isinstance(payload, dict):
            cubes = payload.get("cubes", [])
        elif isinstance(payload, list):
            cubes = payload
        else:
            cubes = []
        return [c for c in cubes if isinstance(c, dict) and c.get("name")]

    def members(self, cube: str, level: str, locale: str | None = None) -> list[Member]:
        """Return every member of ``level`` in ``cube``.

        Tries the tesseract-rs endpoint first and falls back to the
        tesseract-olap one; both shapes are normalised to :class:`Member`.
        """
        base = {"cube": cube, "level": level}
        attempts = [{**base, "locale": locale}, base] if locale else [base]
        last_error: DataMexicoError | None = None
        for params in attempts:  # retry without locale if the level has no translations
            for path in ("members.jsonrecords", "members"):
                try:
                    payload = self.get_json(path, params)
                except DataMexicoError as exc:
                    last_error = exc
                    continue
                members = parse_members(payload)
                if members is not None:
                    return members
        raise last_error or DataMexicoError(f"unexpected members response for {level}")

    def data(
        self,
        cube: str,
        drilldowns: Sequence[str],
        measures: Sequence[str],
        cuts: Mapping[str, Sequence[str]] | None = None,
        locale: str | None = None,
    ) -> list[dict]:
        """Run an aggregate query and return its rows as dicts."""
        params: dict[str, str] = {
            "cube": cube,
            "drilldowns": ",".join(drilldowns),
            "measures": ",".join(measures),
        }
        if locale:
            params["locale"] = locale
        for level, ids in (cuts or {}).items():
            ids = [str(i) for i in ids if str(i) != ""]
            if ids:
                params[level] = ",".join(ids)
        payload = self.get_json("data.jsonrecords", params)
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise DataMexicoError("unexpected data response (no 'data' list)")
        return rows


    def labels_from_data(
        self, cube: str, level: str, measure: str, locale: str | None = None
    ) -> list[Member]:
        """Member IDs and labels taken from an aggregate query drilled by ``level``.

        Used when the members endpoint returns IDs without captions: the data
        endpoint always returns the level's name column next to its ID.
        """
        rows = self.data(cube, [level], [measure], locale=locale)
        members: list[Member] = []
        for row in rows:
            key = row.get(f"{level} ID", row.get(level))
            if key is None:
                continue
            label = row.get(level, key)
            members.append(Member(id=str(key), label=str(label)))
        return members


def parse_members(payload) -> list[Member] | None:
    """Normalise a members response from either Tesseract generation."""
    if isinstance(payload, dict):
        if isinstance(payload.get("members"), list):
            rows = payload["members"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            return None
    elif isinstance(payload, list):
        rows = payload
    else:
        return None

    members: list[Member] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _first(row, ("ID", "key", "Key", "id"))
        if key is None:
            continue
        label = _first(row, ("Label", "caption", "Caption", "label", "name", "Name"))
        if label is None:
            # Unknown shape: take the first other text field as the label.
            label = next(
                (v for k, v in row.items()
                 if isinstance(v, str) and v.strip() and v != str(key)
                 and k not in ("ID", "key", "Key", "id")),
                None,
            )
        members.append(Member(id=str(key), label=str(label if label is not None else key)))
    return members


def _first(row: Mapping, keys: Iterable[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            if body.get(key) and body.get(key) is not True:
                return str(body[key])[:300]
    return str(body)[:300]
