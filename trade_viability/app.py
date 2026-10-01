"""Mexico trade explorer - material -> HS codes -> yearly imports/exports.

Run:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import io
import os

import pandas as pd
import streamlit as st

from datamexico import charts, service
from datamexico.analysis import (
    EXPORTS, FAIL, IMPORTS, MANUAL, NA, PASS, WATCH, Thresholds, cagr, concentration, fmt_value,
    full_years, hhi_band, overall_verdict, partial_years_from_coverage, viability_signals, yearly_summary,
)
from datamexico.client import ENV_BASE_URL, DataMexicoError, DataMexicoUnreachable, TesseractClient, candidate_base_urls
from datamexico.hs import HSEntry, format_code, overlapping, search

st.set_page_config(page_title="Mexico Trade Explorer", page_icon="📦", layout="wide")

STATUS_ICON = {PASS: "✅ Pass", WATCH: "⚠️ Watch", FAIL: "❌ Fail", MANUAL: "✍️ Needs input", NA: "➖ n/a"}
# Years before this are not relevant to the study and are never shown or used.
MIN_YEAR = 2015
HS_LABEL = {2: "HS2 (chapter)", 4: "HS4 (heading)", 6: "HS6 (subheading)", 8: "HS8 (tariff line)", 10: "HS10"}


# ---------------------------------------------------------------------- #
# Cached API access (keyed by base URL so switching source refetches)
# ---------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def connect(candidates: tuple[str, ...]) -> str:
    return TesseractClient.connect(list(candidates)).base_url


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_cubes(base_url: str):
    client = TesseractClient(base_url)
    return service.trade_cubes(client), client.log.urls


def get_cube(base_url: str, name: str):
    cubes, _ = load_cubes(base_url)
    return next(c for c in cubes if c.name == name)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_hs(base_url: str, cube: str, locale: str):
    client = TesseractClient(base_url)
    return service.hs_entries(client, get_cube(base_url, cube), locale), client.log.urls


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_flows(base_url: str, cube: str, locale: str):
    client = TesseractClient(base_url)
    return service.flow_map(client, get_cube(base_url, cube), locale), client.log.urls


def _entries_from_keys(keys):
    return [HSEntry(digits=d, member_id=m, code="", label="") for d, m in keys]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_trade(base_url: str, cube: str, keys: tuple, measure: str, breakdown: str, locale: str):
    client = TesseractClient(base_url)
    tc = get_cube(base_url, cube)
    flows, _ = load_flows(base_url, cube, locale)
    df = service.fetch_trade(client, tc, _entries_from_keys(keys), measure, flows, breakdown, locale)
    return df, client.log.urls


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_coverage(base_url: str, cube: str, measure: str, locale: str):
    client = TesseractClient(base_url)
    cov = service.fetch_coverage(client, get_cube(base_url, cube), measure, locale, MIN_YEAR)
    return cov, client.log.urls


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_comparison(base_url: str, keys: tuple, labels: tuple, locale: str):
    client = TesseractClient(base_url)
    cubes, _ = load_cubes(base_url)
    entries = [HSEntry(digits=d, member_id=m, code="", label="") for d, m in keys]
    return service.compare_cubes(client, cubes, entries, locale)


def log_urls(urls) -> None:
    seen = st.session_state.setdefault("api_log", [])
    for url in urls:
        if url not in seen:
            seen.append(url)


def theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", "light") == "dark" else "light"


def entry_key(e: HSEntry) -> str:
    return f"{e.digits}:{e.member_id}"


def entry_name(e: HSEntry) -> str:
    return f"{format_code(e.code)} · {e.label}"


# ---------------------------------------------------------------------- #
# Sidebar: data source and screening thresholds
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data source")
    custom_url = st.text_input(
        "API base URL (optional)",
        value=os.environ.get(ENV_BASE_URL, ""),
        placeholder="auto-detect",
        help="Leave blank to try the known Data México endpoints in order. "
        "Paste a URL ending in /tesseract to override.",
    )
    locale = st.radio("Label language", ["en", "es"], horizontal=True,
                      format_func=lambda x: {"en": "English", "es": "Español"}[x])

candidates = tuple(candidate_base_urls([custom_url] if custom_url else []))
try:
    with st.spinner("Connecting to Data México…"):
        base_url = connect(candidates)
        cubes, urls = load_cubes(base_url)
        log_urls(urls)
except DataMexicoError as exc:
    st.title("Mexico trade explorer")
    st.error("Could not connect to the Data México API.")
    st.code(str(exc), language=None)
    st.markdown(
        "- Check that this machine can reach `www.economia.gob.mx` (corporate proxies or firewalls may block it).\n"
        "- If Data México moved its API, paste the new base URL (ending in `/tesseract`) in the sidebar, "
        f"or set the `{ENV_BASE_URL}` environment variable."
    )
    st.stop()

with st.sidebar:
    st.success(f"Connected: `{base_url}`")
    cube_name = st.selectbox(
        "Trade cube", [c.name for c in cubes], index=0,
        help="Auto-detected cubes with HS product, trade flow and year levels, best match first.",
    )
    tc = get_cube(base_url, cube_name)
    measure = st.selectbox("Measure", tc.measures, index=tc.measures.index(tc.value_measure))
    default_unit = "USD" if measure == tc.value_measure else ""
    unit = st.text_input("Unit of the measure", value=default_unit,
                         help="Data México trade values are reported in US dollars.")
    country_cubes = [c.name for c in service.cubes_with(cubes, "country")]
    state_cubes = [c.name for c in service.cubes_with(cubes, "state")]
    country_cube = st.selectbox(
        "Cube for country breakdown", country_cubes or ["(none)"],
        index=country_cubes.index(cube_name) if cube_name in country_cubes else 0,
        help="Partner-country split. Totals may differ from the main cube if this is another cube.")
    state_cube = st.selectbox(
        "Cube for state breakdown", state_cubes or ["(none)"], index=0,
        help="Mexican-state split. State-level cubes only include trade Data México could "
        "attribute to a state, so they can sum to less than the national total.")
    with st.expander("Cube structure"):
        for c in cubes:
            st.caption(("**→** " if c.name == cube_name else "") + c.describe())
    with st.expander("Diagnostics"):
        st.caption("Raw API responses, useful if labels or flows look wrong.")
        if st.button("Run API diagnostics"):
            diag = TesseractClient(base_url)
            probes = [
                ("members.jsonrecords", {"cube": cube_name, "level": tc.flow.param, "locale": locale}),
                ("members", {"cube": cube_name, "level": tc.flow.param, "locale": locale}),
                ("data.jsonrecords", {"cube": cube_name, "drilldowns": tc.flow.param,
                                      "measures": tc.value_measure, "locale": locale}),
                ("members.jsonrecords", {"cube": cube_name, "level": max(tc.hs_levels.items())[1].param,
                                         "locale": locale}),
            ]
            for path, params in probes:
                url = diag.url_for(path, params)
                try:
                    diag.get_json(path, params)
                    body = diag.samples.get(url, "")
                except DataMexicoError as exc:
                    body = f"ERROR: {exc}"
                st.code(f"{url}\n{body[:800]}", language=None)

    st.header("Screening thresholds")
    thresholds = Thresholds(
        window_years=st.slider("Assessment window (complete years)", 3, 10, 5),
        min_years_with_imports=st.slider("Min. years with imports in window", 1, 10, 4),
        min_avg_imports=st.number_input(f"Min. average annual imports ({unit or 'units'})", min_value=0.0,
                                        value=1_000_000.0, step=100_000.0, format="%.0f"),
        max_cv=st.slider("Max. volatility (coef. of variation)", 0.1, 2.0, 0.5, 0.05),
        min_import_dependence=st.slider("Min. import dependence", 0.0, 1.0, 0.30, 0.05,
                                        help="Only used when domestic production is entered."),
        max_hhi=float(st.slider("Supplier concentration: HHI above this is flagged", 1000, 10000, 2500, 100)),
        min_import_cagr=st.slider("Min. import CAGR", -0.2, 0.2, 0.0, 0.01, format="%.2f"),
    )

mode = theme_mode()

st.title("Mexico trade explorer")
st.caption(
    "Material → HS codes → yearly imports and exports for Mexico, with signals for the "
    "four Phase-1 viability tests. Source: Data México (Secretaría de Economía)."
)

# ---------------------------------------------------------------------- #
# Step 1: search HS codes
# ---------------------------------------------------------------------- #
st.subheader("1 · Find HS codes for a material")
with st.spinner("Loading the HS product catalogue…"):
    try:
        hs_catalog, urls = load_hs(base_url, cube_name, locale)
        log_urls(urls)
    except DataMexicoError as exc:
        st.error(f"Could not load HS codes: {exc}")
        st.stop()

c1, c2, c3 = st.columns([3, 2, 1.2])
query = c1.text_input("Material name or HS code", placeholder="e.g. toluene, propylene, tolueno, 2902, 2902.30")
levels = c2.multiselect("HS levels", list(hs_catalog), default=[d for d in (4, 6) if d in hs_catalog] or list(hs_catalog),
                        format_func=lambda d: HS_LABEL.get(d, f"HS{d}"))
use_curated = c3.checkbox("Use synonyms", value=True,
                          help="Adds known HS6 codes for common petrochemical names "
                          "(e.g. propylene → 2901.22 and 2711.14).")

basket: dict[str, HSEntry] = st.session_state.setdefault("basket", {})

if query:
    results = search(hs_catalog, query, levels=levels, use_curated=use_curated)
    if not results:
        st.info("No HS codes matched. Try a shorter word, the Spanish name, or an HS code prefix such as 2902.")
    else:
        table = pd.DataFrame({
            "Level": [f"HS{e.digits}" for e in results],
            "HS code": [format_code(e.code) for e in results],
            "Description (Data México)": [e.label for e in results],
            "Parent heading": [e.parent_label for e in results],
            "In analysis": ["✓" if entry_key(e) in basket else "" for e in results],
        })
        st.caption(f"{len(results)} matches. Select rows, then add them to the analysis.")
        event = st.dataframe(
            table, hide_index=True, on_select="rerun", selection_mode="multi-row",
            key=f"results_{query}_{locale}", height=min(420, 38 + 35 * len(table)),
            column_config={"Description (Data México)": st.column_config.TextColumn(width="large")},
        )
        picked = [results[i] for i in event.selection.rows]
        if st.button(f"Add {len(picked)} selected code(s) to analysis", type="primary", disabled=not picked):
            for e in picked:
                basket[entry_key(e)] = e
            st.rerun()

# ---------------------------------------------------------------------- #
# Step 2: selected codes
# ---------------------------------------------------------------------- #
st.subheader("2 · Codes in the analysis")
if not basket:
    st.info("No codes selected yet. Search for a material above and add one or more HS codes.")
    st.stop()

keys_in_basket = list(basket)
kept = st.multiselect("Selected HS codes (remove with ×)", keys_in_basket, default=keys_in_basket,
                      format_func=lambda k: entry_name(basket[k]))
if set(kept) != set(keys_in_basket):
    st.session_state.basket = {k: basket[k] for k in kept}
    st.rerun()

for parent, child in overlapping(basket.values()):
    st.warning(f"{format_code(child.code)} is part of {format_code(parent.code)}: selecting both double counts "
               f"its trade. Remove one of them.")

b1, b2 = st.columns([1, 5])
if b1.button("Fetch trade data", type="primary"):
    st.session_state.show_results = True
if b2.button("Clear selection"):
    st.session_state.basket = {}
    st.session_state.show_results = False
    st.rerun()

if not st.session_state.get("show_results"):
    st.stop()

# ---------------------------------------------------------------------- #
# Step 3: data
# ---------------------------------------------------------------------- #
keys = tuple(sorted((e.digits, e.member_id) for e in basket.values()))
names = {(e.digits, e.member_id): e for e in basket.values()}
try:
    with st.spinner("Fetching yearly imports and exports…"):
        by_hs, urls = load_trade(base_url, cube_name, keys, measure, "hs", locale)
        log_urls(urls)
        by_country = by_state = None
        if country_cubes:
            cc = get_cube(base_url, country_cube)
            by_country, urls = load_trade(base_url, country_cube, keys,
                                          measure if country_cube == cube_name else cc.value_measure,
                                          "country", locale)
            log_urls(urls)
        if state_cubes:
            sc = get_cube(base_url, state_cube)
            by_state, urls = load_trade(base_url, state_cube, keys,
                                        measure if state_cube == cube_name else sc.value_measure,
                                        "state", locale)
            log_urls(urls)
        coverage, urls = load_coverage(base_url, cube_name, measure, locale)
        log_urls(urls)
except DataMexicoUnreachable as exc:
    connect.clear()
    st.error("The Data México server is not responding, so no data could be fetched.")
    st.markdown(
        f"- Details: `{exc}`\n"
        "- This is a network problem between this app and `www.economia.gob.mx`, not a problem with "
        "the codes you selected. The site may be down or rate-limiting, or it may be blocking this "
        "app's hosting provider.\n"
        "- Check whether [Data México](https://www.economia.gob.mx/datamexico/) opens in your browser. "
        "If it does, try again in a few minutes, or run the app on your own computer "
        "(`streamlit run app.py`), which connects from your network instead."
    )
    st.stop()
except DataMexicoError as exc:
    st.error(f"Data request failed: {exc}")
    st.stop()


def recent(df):
    """Keep rows from MIN_YEAR on (also drops placeholder years such as 0)."""
    return df if df is None or df.empty else df[df["year"] >= MIN_YEAR].reset_index(drop=True)


by_hs, by_country, by_state = recent(by_hs), recent(by_country), recent(by_state)
coverage = {y: n for y, n in coverage.items() if y >= MIN_YEAR}

if by_hs.empty:
    st.warning(f"Data México returned no trade records for the selected codes from {MIN_YEAR} onwards.")
    st.stop()

by_hs["product"] = [
    entry_name(names[(d, m)]) if (d, m) in names else c
    for d, m, c in zip(by_hs["hs_digits"], by_hs["hs_id"], by_hs["code"])
]
summary = yearly_summary(by_hs)
partial = set(partial_years_from_coverage(coverage))
this_year = dt.date.today().year
if not coverage:
    partial |= {y for y in summary.index if y >= this_year}
complete = full_years(summary, partial)
years = list(summary.index)

st.subheader("3 · Mexico's trade in the selected codes")
st.caption(f"Data from {MIN_YEAR} onwards.")
if partial:
    months = {y: coverage.get(y) for y in partial}
    detail = ", ".join(f"{y}" + (f" ({n} of {max(coverage.values())} periods)" if n else "") for y, n in months.items())
    st.caption(f"Hatched bars marked YTD are partial years and are excluded from growth and screening: {detail}.")

if complete:
    last = complete[-1]
    window = complete[-thresholds.window_years:]
    growth = cagr(summary.loc[window[0], "imports"], summary.loc[last, "imports"], len(window) - 1)
    k1, k2, k3, k4, k5 = st.columns(5)
    prev = summary.loc[last - 1] if last - 1 in summary.index else None
    k1.metric(f"Imports {last}", f"{fmt_value(summary.loc[last, 'imports'])} {unit}",
              None if prev is None or pd.isna(summary.loc[last, "imports_yoy"]) else f"{summary.loc[last, 'imports_yoy']:+.1%} vs {last - 1}")
    k2.metric(f"Exports {last}", f"{fmt_value(summary.loc[last, 'exports'])} {unit}",
              None if prev is None or pd.isna(summary.loc[last, "exports_yoy"]) else f"{summary.loc[last, 'exports_yoy']:+.1%} vs {last - 1}")
    k3.metric(f"Net imports {last}", f"{fmt_value(summary.loc[last, 'net_imports'])} {unit}",
              help="Imports minus exports. Positive means Mexico relies on imports.")
    k4.metric(f"Import CAGR {window[0]}–{last}", "–" if growth is None else f"{growth:+.1%}")
    k5.metric("Years with imports", f"{int((summary.loc[complete, 'imports'] > 0).sum())} of {len(complete)}")
else:
    st.info("No complete years available yet for these codes.")

tabs = st.tabs(["Trend", "By HS code", "Origin & destination countries", "Mexican states",
                "Viability screen", "Data & export"])

# ---- Trend ----------------------------------------------------------- #
with tabs[0]:
    st.plotly_chart(charts.trade_trend(summary, partial, unit, mode), key="trend")
    st.plotly_chart(charts.net_imports(summary.loc[complete] if complete else summary, set(), unit, mode), key="net")
    st.caption("Net imports are shown for complete years only.")
    with st.expander("Table"):
        show = summary[["imports", "exports", "net_imports", "imports_yoy", "exports_yoy"]].copy()
        show.index = charts.year_labels(show.index, partial)
        st.dataframe(show.style.format({"imports": "{:,.0f}", "exports": "{:,.0f}", "net_imports": "{:,.0f}",
                                        "imports_yoy": "{:+.1%}", "exports_yoy": "{:+.1%}"}, na_rep="–"))

# ---- By HS code ------------------------------------------------------ #
with tabs[1]:
    if by_hs["product"].nunique() == 1:
        st.caption("Only one code is selected: the breakdown equals the trend.")
    order = list(by_hs.groupby("product")["value"].sum().sort_values(ascending=False).index)
    for flow in (IMPORTS, EXPORTS):
        st.plotly_chart(charts.stacked_by(by_hs, "product", flow, years, partial, unit,
                                          f"{flow} by HS code", mode, order=order), key=f"hs_{flow}")
    with st.expander("Table"):
        pivot = by_hs.pivot_table(index=["product", "flow"], columns="year", values="value", aggfunc="sum").fillna(0)
        st.dataframe(pivot.style.format("{:,.0f}"))

# ---- Countries ------------------------------------------------------- #
with tabs[2]:
    if by_country is None or by_country.empty:
        st.info("This cube has no partner-country breakdown.")
    else:
        flow = st.radio("Flow", [IMPORTS, EXPORTS], horizontal=True, key="country_flow",
                        format_func=lambda f: "Imports by origin" if f == IMPORTS else "Exports by destination")
        st.plotly_chart(charts.stacked_by(by_country, "country", flow, years, partial, unit,
                                          f"{flow} by partner country", mode), key="country_stack")
        pick_years = complete or years
        year = st.select_slider("Year", options=pick_years, value=pick_years[-1], key="country_year")
        conc = concentration(by_country, "country", year, flow)
        if conc:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Partner countries", conc.n_partners)
            m2.metric("Top partner", conc.top1_name, f"{conc.top1_share:.0%} share", delta_color="off",
                      delta_arrow="off")
            m3.metric("Top-3 share", f"{conc.top3_share:.0%}")
            m4.metric("HHI", f"{conc.hhi:,.0f}", hhi_band(conc.hhi), delta_color="off", delta_arrow="off",
                      help="Herfindahl-Hirschman index of partner shares (0-10,000). "
                      "Below 1,500 unconcentrated; above 2,500 highly concentrated.")
            st.plotly_chart(charts.ranking_bar(conc.shares, "partner", "value", unit,
                                               f"{flow} by partner country, {year}", mode, share_col="share"),
                            key="country_rank")
        with st.expander("Table"):
            st.dataframe(by_country.pivot_table(index=["country", "flow"], columns="year", values="value",
                                                aggfunc="sum").fillna(0).style.format("{:,.0f}"))

# ---- States ---------------------------------------------------------- #
with tabs[3]:
    if by_state is None or by_state.empty:
        st.info("This cube has no Mexican-state breakdown.")
    else:
        st.caption(f"Source cube: `{state_cube}`. State-level data only include trade Data México "
                   "could attribute to a state, so totals can be lower than the national figures "
                   "in the other tabs.")
        flow = st.radio("Flow", [IMPORTS, EXPORTS], horizontal=True, key="state_flow")
        pick_years = complete or years
        year = st.select_slider("Year", options=pick_years, value=pick_years[-1], key="state_year")
        sub = by_state[(by_state["year"] == year) & (by_state["flow"] == flow)]
        table = sub.groupby("state", as_index=False)["value"].sum()
        table = table[table["value"] > 0]
        if table.empty:
            st.info(f"No {flow.lower()} recorded by state in {year}.")
        else:
            table["share"] = table["value"] / table["value"].sum()
            st.plotly_chart(charts.ranking_bar(table, "state", "value", unit, f"{flow} by state, {year}",
                                               mode, share_col="share", top=32), key="state_rank")
        with st.expander("Table"):
            st.dataframe(by_state.pivot_table(index=["state", "flow"], columns="year", values="value",
                                              aggfunc="sum").fillna(0).style.format("{:,.0f}"))

# ---- Viability ------------------------------------------------------- #
signals_df = pd.DataFrame()
with tabs[4]:
    st.markdown(
        "Tests 1 and 2 are screened from the trade data. Tests 3 and 4 depend on Indelpro's assets and "
        "commercial knowledge, so record your assessment below."
    )
    with st.expander("Optional: domestic production → apparent consumption and import dependence",
                     expanded=False):
        st.caption(f"Enter Mexican production per year in the same unit as the measure ({unit or 'measure unit'}), "
                   "e.g. from ANIQ's Anuario. Apparent consumption = production + imports − exports.")
        prod_key = "production_" + "_".join(f"{d}-{m}" for d, m in keys)
        prod_df = st.data_editor(
            pd.DataFrame({"year": years, "production": [None] * len(years)}).astype({"production": "float"}),
            hide_index=True, disabled=["year"], key=prod_key,
            column_config={"year": st.column_config.NumberColumn(format="%d"),
                           "production": st.column_config.NumberColumn(f"production ({unit})", format="%.0f")},
        )
    production = {int(r.year): r.production for r in prod_df.itertuples() if pd.notna(r.production)}
    full_summary = yearly_summary(by_hs, production or None)

    col3, col4 = st.columns(2)
    manual = {}
    for col, test in ((col3, "3. Service repeatability"), (col4, "4. Economic margin")):
        with col:
            status = st.selectbox(test, [MANUAL, PASS, WATCH, FAIL], key=f"status_{test}_{prod_key}",
                                  format_func=lambda s: STATUS_ICON[s])
            note = st.text_input("Evidence / note", key=f"note_{test}_{prod_key}",
                                 placeholder="e.g. ambient liquid, fits existing Altamira tankage"
                                 if test.startswith("3") else "e.g. distributors earn 19-22% gross margin")
            manual[test] = (status, note)

    supplier = None
    if by_country is not None and complete:
        supplier = concentration(by_country, "country", complete[-1], IMPORTS)
    signals, verdict = viability_signals(full_summary, complete, thresholds, supplier, manual, unit)
    overall = overall_verdict(verdict)
    banner = {"Viable": st.success, "Not viable": st.error}.get(overall, st.warning)
    banner(f"Overall: **{overall}** — " + " · ".join(f"{t}: {STATUS_ICON[s]}" for t, s in verdict.items()))

    signals_df = pd.DataFrame([{"Test": s.test, "Indicator": s.indicator, "Value": s.value,
                                "Status": STATUS_ICON.get(s.status, s.status), "Note": s.note} for s in signals])
    st.dataframe(signals_df, hide_index=True, column_config={
        "Test": st.column_config.TextColumn(width="medium"),
        "Indicator": st.column_config.TextColumn(width="medium"),
        "Value": st.column_config.TextColumn(width="medium"),
        "Status": st.column_config.TextColumn(width="medium"),
        "Note": st.column_config.TextColumn(width="large"),
    })

    if production:
        st.plotly_chart(charts.consumption_chart(full_summary, partial, unit, mode), key="consumption")
        st.plotly_chart(charts.dependence_chart(full_summary, partial, thresholds.min_import_dependence, mode),
                        key="dependence")

# ---- Data & export --------------------------------------------------- #
with tabs[5]:
    codes_df = pd.DataFrame([{"level": f"HS{e.digits}", "hs_code": format_code(e.code), "data_mexico_id": e.member_id,
                              "description": e.label, "parent_heading": e.parent_label} for e in basket.values()])
    st.markdown("**Selected codes**")
    st.dataframe(codes_df, hide_index=True)

    st.markdown("**Check totals across cubes**")
    st.caption("Imports and exports of the selected codes in every detected trade cube. "
               "Use the cube whose totals match Data México's published figures.")
    if st.button("Compare cubes"):
        with st.spinner("Querying every trade cube…"):
            comparison = load_comparison(base_url, keys, tuple(entry_name(e) for e in basket.values()), locale)
        if comparison.empty:
            st.info("No comparable cubes.")
        else:
            comparison = comparison[[c for c in comparison.columns if pd.notna(c) and c >= MIN_YEAR]]
            st.dataframe(comparison.style.format("{:,.0f}", na_rep="–"))

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        codes_df.to_excel(writer, sheet_name="Codes", index=False)
        full_summary.to_excel(writer, sheet_name="Yearly summary")
        if not signals_df.empty:
            signals_df.to_excel(writer, sheet_name="Viability", index=False)
        by_hs.to_excel(writer, sheet_name="By HS code", index=False)
        if by_country is not None:
            by_country.to_excel(writer, sheet_name="By country", index=False)
        if by_state is not None:
            by_state.to_excel(writer, sheet_name="By state", index=False)
        pd.DataFrame({"api_url": st.session_state.get("api_log", [])}).to_excel(
            writer, sheet_name="API queries", index=False)
    stem = "_".join(format_code(e.code) for e in basket.values())[:60]
    d1, d2 = st.columns(2)
    d1.download_button("Download Excel workbook", buffer.getvalue(), file_name=f"mexico_trade_{stem}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    d2.download_button("Download yearly summary (CSV)", full_summary.to_csv().encode(),
                       file_name=f"mexico_trade_{stem}.csv", mime="text/csv")

    st.markdown("**Raw rows by HS code**")
    st.dataframe(by_hs, hide_index=True)
    with st.expander("API queries used (paste in a browser to reproduce)"):
        for url in st.session_state.get("api_log", []):
            st.code(url, language=None)
