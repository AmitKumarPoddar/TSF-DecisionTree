# Mexico trade explorer (Data México)

An interactive tool for the Phase-1 viability screen of the *Business
Opportunities in Petrochemicals* study. It pulls Mexico's yearly import and
export data from **Data México** (Secretaría de Economía) through its
Tesseract API.

1. **Type a material** (English or Spanish) or an HS code, e.g. `toluene`,
   `tolueno`, `propylene`, `2902`, `2902.30`.
2. **Pick HS codes** from the matches. Each match shows Data México's own
   description and its parent heading.
3. **Fetch trade data.** You get yearly imports, exports and net imports,
   a breakdown by HS code, origin and destination countries, and Mexican
   states. You also get signals for the four viability tests, plus an Excel
   export.

## Recurring-demand workflow (branch `viability_workflow`)

The app opens on **Recurring-demand workflow**. The original analysis is under
**View → Trade explorer** in the sidebar.

| Tab | What happens |
|---|---|
| **1 · Opportunity & market size** | Enter the opportunity, vertical and immediate base material. **1A**: Mexico market-size figures for the exact opportunity, found by AI web search or entered manually. **1B**, only if 1A has no usable Mexico figure: base-material figures × two-level relevance (r1 = 1 / number of categories, r2 = 1 / number of parallel products; both can be overridden). |
| **2 · Trade recurrence** | The same HS search, charts and export as the Trade explorer. Recurrence uses the **last 5 complete years only**: recurring if imports appear in at least 4 of 5 (adjustable); strength **High** if net imports are positive in all 5, otherwise **Moderate**. |
| **3 · Recurring-demand result** | Market found + recurring → ✅ established. Market found only → ⚠️ recurrence not evidenced. Trade only → ⚠️ established from trade (lower confidence). Neither → ❌ *The recurring demand could not be established.* Includes an imports-vs-market cross-check. |
| **🔒 Market sources** | Append-only register of every figure found or entered. Edits are logged as new entries and nothing can be deleted. Also lists the pages each AI search consulted, an Excel report, and save/load of the whole assessment (JSON). |

How market sizes are calculated:
- **Output.** For each step: minimum, maximum, average of all Mexico figures, average of the selected figures, and the value used (the selected average, or the average of all when none are selected).
- **Comparable figures.** Values are converted to USD million at live ECB rates (editable in the sidebar) and moved to the reference year using each source's own CAGR. Volume figures (kt) are kept separate from value figures. Non-Mexico figures are shown but left out of the range and averages unless selected.

**AI search** uses Claude (`claude-opus-5-5`) with web search. Add an Anthropic API key in
the sidebar, or as `ANTHROPIC_API_KEY` in the app's Streamlit **Secrets**. Each search uses API
credits. A figure is flagged "URL seen in search" only if its link was among the pages the
search returned; review every figure before ticking it.

## Run it

```bash
cd trade_viability
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. It needs internet access to
`www.economia.gob.mx`.

## How it talks to the Data México API

Data México serves its data from a Tesseract OLAP server. The client tries
these base URLs in order and uses the first one that answers `/cubes`:

| Order | Base URL |
|---|---|
| 1 | `$DATAMEXICO_API_BASE`, or the URL typed in the sidebar |
| 2 | `https://www.economia.gob.mx/apidatamexico/tesseract` |
| 3 | `https://api.datamexico.org/tesseract` (original host) |

Endpoints used, the same as in the Tesseract documentation (`tesseract.pdf`):

```
GET {base}/cubes                                     # schema: cubes, levels, measures
GET {base}/members.jsonrecords?cube=C&level=HS6&locale=en
GET {base}/data.jsonrecords?cube=C&drilldowns=Year,Flow,HS6&measures=Trade Value&HS6=<ids>&locale=en
```

Cube and level names are **not hard-coded**. The app reads `/cubes` and
picks the cube that has HS product levels (HS2/HS4/HS6), a `Flow` level, a
`Year` level and a trade-value measure. Country, state and month or quarter
levels are used when the cube has them. You can switch to another detected
cube in the sidebar. Both generations of the Tesseract server
(tesseract-rs and tesseract-olap) are supported.

Data México stores HS IDs with the HS section as a numeric prefix; for
example, toluene 2902.30 may be ID `6290230`. The app shows the standard HS
code and uses the raw ID for queries. Every URL it calls is listed under
**Data & export → API queries**, so any number can be reproduced in a
browser.

## Viability screen

| Test (approach document) | What the app computes | Default threshold (editable in sidebar) |
|---|---|---|
| 1. Recurring demand | Years with imports in the window; average annual imports; volatility (coefficient of variation) | ≥ 4 of 5 years; ≥ 1M USD; CV ≤ 0.5 |
| 2. Open market | Net importer in the latest complete year; import CAGR; supplier-country concentration (HHI, top-1 and top-3 share); import dependence, if production is entered | imports > exports; CAGR ≥ 0; HHI < 2,500; dependence ≥ 30% |
| 3. Service repeatability | Analyst judgement, recorded in the app | – |
| 4. Economic margin | Analyst judgement, recorded in the app | – |

- **Apparent consumption and import dependence.** Data México trade data has
  no domestic production. You can enter production per year (e.g. from
  ANIQ's *Anuario*) in the same unit as the measure. The app then computes
  *apparent consumption = production + imports − exports* and
  *import dependence = imports ÷ apparent consumption*, the method used in
  the approach document's worked example.
- **Years shown.** Only 2015 onwards; earlier years and placeholder years (e.g. 0) are dropped everywhere.
- **Partial years.** The current year, or any year with fewer reported months
  or quarters, is hatched and labelled *YTD* in charts. It is excluded from
  growth rates and screening.
- **Unit.** Trade values are in USD. Thresholds are in the unit of the
  selected measure.

## Project layout

```
app.py                  Streamlit UI
datamexico/client.py    HTTP client: base-URL detection, /cubes, members, data
datamexico/schema.py    Finds the trade cube and its HS/flow/year/country/state levels
datamexico/hs.py        HS code normalisation and search (keyword, code prefix, synonyms)
datamexico/service.py   Queries used by the UI (one request per HS level selected)
datamexico/analysis.py  Yearly summary, CAGR, HHI, apparent consumption, 4-test signals
datamexico/charts.py    Plotly figures
tests/                  pytest suite + mock Tesseract server (synthetic data)
```

## Tests

```bash
cd trade_viability
python -m pytest -q
```

The integration tests run against `tests/mock_tesseract.py`. This local fake
server reproduces the response formats of both Tesseract generations with
**synthetic numbers**. To try the UI offline:

```bash
python -m tests.mock_tesseract --port 8765 --dialect rs &
DATAMEXICO_API_BASE=http://127.0.0.1:8765/tesseract streamlit run app.py
```

Never use the mock's numbers for analysis.
