"""A small fake Tesseract server for tests and offline UI development.

It serves SYNTHETIC numbers - never use it for analysis. It mimics the
response shapes of both Tesseract generations:

* ``rs``   - tesseract-rs logic layer (members.jsonrecords -> {"data": [{ID, Label}]})
* ``olap`` - tesseract-olap (members -> {"members": [{key, caption}]}, data has "page")

Run standalone:  python -m tests.mock_tesseract --port 8765 --dialect rs
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

CUBE = "economy_foreign_trade_ent"

HS2 = {"101": ("Live animals", "Animales vivos"),
       "527": ("Mineral fuels and oils", "Combustibles minerales"),
       "629": ("Organic chemicals", "Productos químicos orgánicos"),
       "739": ("Plastics and articles thereof", "Plástico y sus manufacturas")}
HS4 = {"10101": ("Live horses, asses, mules and hinnies", "Caballos, asnos, mulos y burdéganos, vivos"),
       "52711": ("Petroleum gases and other gaseous hydrocarbons", "Gas de petróleo y demás hidrocarburos gaseosos"),
       "62901": ("Acyclic hydrocarbons", "Hidrocarburos acíclicos"),
       "62902": ("Cyclic hydrocarbons", "Hidrocarburos cíclicos"),
       "73902": ("Polymers of propylene or of other olefins, in primary forms", "Polímeros de propileno o de otras olefinas")}
HS6 = {"1010121": ("Pure-bred breeding horses", "Caballos reproductores de raza pura"),
       "5271114": ("Ethylene, propylene, butylene and butadiene, liquefied", "Etileno, propileno, butileno y butadieno, licuados"),
       "6290121": ("Ethylene", "Etileno"),
       "6290122": ("Propene (propylene)", "Propeno (propileno)"),
       "6290230": ("Toluene", "Tolueno"),
       "6290241": ("o-Xylene", "o-Xileno"),
       "6290250": ("Styrene", "Estireno"),
       "7390210": ("Polypropylene", "Polipropileno")}
FLOWS = {"1": ("Exports", "Exportaciones"), "2": ("Imports", "Importaciones")}
COUNTRIES = {"nausa": ("United States", "Estados Unidos"), "aschn": ("China", "China"),
             "askor": ("South Korea", "Corea del Sur"), "eudeu": ("Germany", "Alemania"),
             "asjpn": ("Japan", "Japón")}
STATES = {"28": ("Tamaulipas", "Tamaulipas"), "9": ("Ciudad de México", "Ciudad de México"),
          "15": ("Estado de México", "Estado de México"), "19": ("Nuevo León", "Nuevo León")}
YEARS = range(2018, 2026)
LAST_YEAR_MONTHS = 6  # 2025 is a partial year in the mock

LEVELS = {
    # level name -> (dimension, member dict or None for time levels)
    "Year": ("Date", None), "Quarter": ("Date", None), "Month": ("Date", None),
    "HS2": ("Product", HS2), "HS4": ("Product", HS4), "HS6": ("Product", HS6),
    "Flow": ("Flow", FLOWS), "Country": ("Country", COUNTRIES), "State": ("Geography", STATES),
}


def _noise(*key) -> float:
    digest = hashlib.md5("|".join(map(str, key)).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_facts() -> pd.DataFrame:
    base = {"6290230": 9e6, "6290122": 22e6, "6290121": 0.3e6, "6290241": 1.5e6,
            "6290250": 50e6, "7390210": 30e6, "5271114": 12e6, "1010121": 0.05e6}
    rows = []
    for year, month, hs6, flow, country, state in itertools.product(
        YEARS, range(1, 13), HS6, FLOWS, COUNTRIES, STATES
    ):
        if year == max(YEARS) and month > LAST_YEAR_MONTHS:
            continue
        growth = 1.04 ** (year - min(YEARS))
        flow_scale = 1.0 if flow == "2" else 0.08
        if hs6 == "7390210" and flow == "1":
            flow_scale = 0.9  # Mexico exports polypropylene
        country_scale = {"nausa": 0.7, "aschn": 0.1, "askor": 0.1, "eudeu": 0.06, "asjpn": 0.04}[country]
        if hs6 == "6290230" and state == "28" and flow == "2":
            continue  # toluene imports do not clear through Tamaulipas
        value = base[hs6] / 12 * growth * flow_scale * country_scale / 4 * (0.8 + 0.4 * _noise(year, month, hs6, flow, country, state))
        rows.append((year, (year * 10 + (month - 1) // 3 + 1), year * 100 + month,
                     hs6[:-4], hs6[:-2], hs6, flow, country, state, value))
    return pd.DataFrame(rows, columns=["Year", "Quarter", "Month", "HS2", "HS4", "HS6",
                                       "Flow", "Country", "State", "Trade Value"])


FACTS = build_facts()


def cube_metadata(dialect: str) -> dict:
    def level(name, depth):
        if dialect in ("rs", "bare"):
            return {"name": name, "unique_name": None, "properties": None, "annotations": {}}
        return {"name": name, "caption": name, "depth": depth, "count": 0, "annotations": {}, "properties": []}

    def dim(name, dtype, levels):
        body = {"name": name, "type": dtype, "annotations": {},
                "hierarchies": [{"name": name, "annotations": {},
                                 "levels": [level(lv, i) for i, lv in enumerate(levels, 1)]}]}
        if dialect == "olap":
            body["caption"] = name
            body["default_hierarchy"] = name
        return body

    measure = ({"name": "Trade Value", "aggregator": {"name": "sum"}, "annotations": {}}
               if dialect in ("rs", "bare") else {"name": "Trade Value", "caption": "Trade Value", "aggregator": "sum",
                                        "type": "float64", "annotations": {}, "attached": []})
    trade_dims = [dim("Date", "time", ["Year", "Quarter", "Month"]),
                  dim("Product", "standard", ["Chapter", "HS2", "HS4", "HS6"]),
                  dim("Flow", "standard", ["Flow"]),
                  dim("Country", "standard", ["Continent", "Country"]),
                  dim("Geography", "geo" if dialect == "olap" else "standard", ["Nation", "State"])]
    cubes = [
        {"name": "inegi_enoe", "dimensions": [dim("Date", "time", ["Year"])], "measures": [measure], "annotations": {}},
        {"name": "economy_foreign_trade_mun", "dimensions": trade_dims, "measures": [measure], "annotations": {}},
        {"name": CUBE, "dimensions": trade_dims, "measures": [measure], "annotations": {}},
    ]
    if dialect == "olap":
        return {"name": "mock", "locales": ["en", "es"], "default_locale": "es", "annotations": {}, "cubes": cubes}
    return {"name": "mock", "annotations": {}, "cubes": cubes}


def _label(level: str, key, locale: str) -> str:
    members = LEVELS[level][1]
    if members is None:
        return str(key)
    en, es = members[str(key)]
    return es if locale == "es" else en


class Handler(BaseHTTPRequestHandler):
    dialect = "rs"
    requests_seen: list[str] = []

    def log_message(self, *args):  # silence
        pass

    def _send(self, status: int, body) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        self.requests_seen.append(self.path)
        params = {k: v[-1] for k, v in parse_qs(url.query).items()}
        path = url.path.rstrip("/").split("/tesseract", 1)[-1]
        try:
            if path == "/cubes":
                return self._send(200, cube_metadata(self.dialect))
            if path in ("/members", "/members.jsonrecords"):
                if self.dialect in ("rs", "bare") and path == "/members":
                    return self._send(400, "use members.jsonrecords")
                return self._members(params)
            if path == "/data.jsonrecords":
                return self._data(params)
            return self._send(404, {"error": True, "detail": f"no route {path}"})
        except KeyError as exc:
            return self._send(400, {"error": True, "detail": f"bad request: {exc}"})

    def _members(self, params):
        level, locale = params["level"], params.get("locale", "es")
        if params.get("cube") not in (CUBE, "economy_foreign_trade_mun"):
            return self._send(404, {"error": True, "detail": "cube not found"})
        if level not in LEVELS:
            return self._send(404, {"error": True, "detail": "Unable to find a level with the name provided"})
        keys = LEVELS[level][1] or sorted(FACTS[level].unique())
        if self.dialect == "bare":  # members without captions, as seen on the live API
            return self._send(200, {"data": [{"ID": int(k) if str(k).isdigit() else k} for k in keys]})
        if self.dialect in ("rs",):
            return self._send(200, {"data": [{"ID": k, "Label": _label(level, k, locale)} for k in keys]})
        return self._send(200, {"name": level, "caption": level, "depth": 1, "annotations": {},
                                "properties": [], "dtypes": {},
                                "members": [{"key": k, "caption": _label(level, k, locale)} for k in keys]})

    def _data(self, params):
        if params.get("cube") not in (CUBE, "economy_foreign_trade_mun"):
            return self._send(404, {"error": True, "detail": "cube not found"})
        locale = params.get("locale", "es")
        drills = [d for d in params["drilldowns"].split(",") if d]
        measures = [m for m in params["measures"].split(",") if m]
        for d in drills:
            if d not in LEVELS:
                return self._send(400, {"error": True, "detail": f"unknown level {d}"})
        cuts = {k: v.split(",") for k, v in params.items() if k in LEVELS}
        for item in filter(None, params.get("include", "").split(";")):
            level, members = item.split(":", 1)
            cuts[level] = members.split(",")
        df = FACTS
        for level, members in cuts.items():
            df = df[df[level].astype(str).isin(members)]
        grouped = df.groupby(drills, as_index=False)[measures].sum()
        records = []
        for row in grouped.to_dict("records"):
            rec = {}
            for d in drills:
                key = row[d].item() if hasattr(row[d], "item") else row[d]
                if d == "Year":
                    rec["Year"] = int(key)
                    continue
                # tesseract-rs returns integer keys for integer columns.
                rec[f"{d} ID"] = int(key) if self.dialect in ("rs", "bare") and d in ("Flow", "State") else key
                rec[d] = _label(d, key, locale)
            for m in measures:
                rec[m] = round(float(row[m]), 2)
            records.append(rec)
        if self.dialect in ("rs", "bare"):
            return self._send(200, {"data": records, "source": []})
        columns = list(records[0].keys()) if records else []
        return self._send(200, {"columns": columns, "data": records,
                                "page": {"limit": 0, "offset": 0, "total": len(records)}})


def serve(port: int = 0, dialect: str = "rs") -> tuple[ThreadingHTTPServer, str]:
    """Start the mock in a background thread; returns (server, base_url)."""
    handler = type(f"Handler_{dialect}", (Handler,), {"dialect": dialect, "requests_seen": []})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/tesseract"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dialect", choices=("rs", "olap", "bare"), default="rs")
    args = parser.parse_args()
    server, base = serve(args.port, args.dialect)
    print(f"Mock Tesseract ({args.dialect}, SYNTHETIC DATA) at {base}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
