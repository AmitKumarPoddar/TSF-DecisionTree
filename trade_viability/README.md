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
| **1 · Recurring demand** | The conclusion sits at the top: Yes/No, the approximate market size, and how many of the last 5 complete years had imports. The three steps below it set the market size. The first step that applies is used, and the later ones are skipped. |
| **2 · Open market** | Import trend, supplier-country HHI and a competitor scan for the exact opportunity (see below). |
| **🔒 Market sources** | Append-only register of every figure found or entered and of every competitor found. Edits are logged as new entries and nothing can be deleted. Also lists the pages each AI search consulted, an Excel report, and save/load of the whole assessment (JSON). |

The three steps in tab 1:

| Step | Applies when | Approximate market size |
|---|---|---|
| **1 · Import data of the exact opportunity** | The opportunity has its own HS code and its net imports are positive in the latest complete year | Net imports of that code |
| **2 · Market reports of the exact opportunity** | Step 1 doesn't apply and at least one usable Mexico figure exists (AI search or manual entry) | Average of the selected reports (or of all Mexico figures if none is selected) |
| **3 · Import data of the immediate base material × relevance** | Steps 1 and 2 don't apply | Base material's net imports × r1 × r2 (r1 = 1 / number of categories, r2 = 1 / number of parallel products; both can be overridden, or suggested by AI) |

- **Recurrence check.** Uses the import data of the step that gave the market size. On the Step 2 path it uses the exact code's imports if they were fetched; otherwise only Step 3a (the base material's HS codes) is needed, and no relevance.
- **"Yes" rule.** A market size from Steps 1–3 AND imports in at least 4 of the last 5 complete years (adjustable in the sidebar). The sentence also says how many years Mexico was a net importer, and when it was a net exporter.
- **Net or gross imports.** Net imports (imports − exports) are the default; the sidebar can switch Steps 1 and 3 to gross imports. If the latest year's figure isn't positive, that step can't give a market size: an exact code then moves on to Step 2.
- **Why "approximate".** Trade data leaves out domestic production and the value added after import.
- **Report figures (Step 2).** Converted to USD million at live ECB rates (editable in the sidebar) and moved to the reference year using each source's own CAGR. Volume figures (kt) are kept separate from value figures. Non-Mexico figures are shown but left out unless selected.

How the open-market check works (tab 2):
- **Import trend.** Uses the HS codes from tab 1 (the exact code from Step 1 if fetched, otherwise the base material's from Step 3a) and the last 5 complete years. It uses volume when the cube has a quantity measure, otherwise value. The trend is a Theil–Sen line (the median of all pairwise slopes), so a single spike year can't decide it. Above +2%/yr is **Growing**, −2% to +2% is **Stable**, below −2% is **Declining**.
- **Conclusion paragraph.** Both tabs open with a plain-English conclusion (no brackets, dashes or colons), with a "Copy as plain text (for Excel)" box. The open-market paragraph covers the Yes/No result, the yearly import change, how the imports are spread across countries, and the manufacturers found.
- **Import origins.** The share of Mexico's imports from each country, per year, for the top countries of the latest year (the rest are grouped as Other countries). Also exported to Excel.
- **HHI by year.** Supplier-country concentration with its 5-year direction. It is a note only and never changes the result.
- **Competitors.** Companies, domestic or international, that supply the *exact opportunity* in Mexico. Companies that only sell the base material are not counted. Subsidiaries of one group count once. 4 or more groups means the market is fragmented.

| HS codes used | Result |
|---|---|
| Exact opportunity | Growing/stable imports → ✅ open. Declining imports or no imports → ❌ not open. The competitor scan is optional. |
| Base material | Scan required. 4+ groups with growing/stable imports → ✅ open. 4+ groups with declining imports, or 1–3 groups → ⚠️ watch. 0 groups → *No market in Mexico for this opportunity*, but only after you tick the confirmation box; until then the result is provisional. |

On a free Gemini key, competitor candidates come from the model's knowledge and are checked by reading their pages. Candidates that could not be confirmed are listed with **Include** unticked. Tick the ones you can confirm.

**AI research** is available with two providers, selected in the sidebar:

| Provider | Key (Streamlit **Secrets**) | Default model | How it finds figures |
|---|---|---|---|
| **Gemini (Google)**, the default | `GEMINI_API_KEY` (optional `GEMINI_MODEL`) | `gemini-flash-lite-latest` | Google Search grounding if your plan allows it; on the free tier, Gemini proposes candidate report pages and reads them with its URL-reading tool |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | Claude web search |

- **"URL seen in search" flag.** It is ticked only when the figure's page was actually returned or read. Review every figure before ticking it.
- **Your own URLs.** You can paste report URLs you found yourself, and the AI reads those first.
- **No web search on the free tier.** On a free Gemini key, the product-structure suggestion comes from the model's knowledge and is labelled as such.

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
datamexico/market.py    Recurring demand: market sizes, relevance, recurrence, sources register
datamexico/openmarket.py  Open market: import trend, HHI notes, competitor groups, verdict
datamexico/ai_research.py, gemini_research.py  AI research (Claude / Gemini)
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
