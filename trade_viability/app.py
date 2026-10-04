"""Opportunity viability app: recurring-demand workflow + Mexico trade explorer.

Run:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os

import pandas as pd
import streamlit as st

from datamexico import ai_research as ai
from datamexico import gemini_research as gem
from datamexico import charts, service
from datamexico import market as mk
from datamexico import openmarket as om
from datamexico.analysis import (
    EXPORTS, FAIL, IMPORTS, MANUAL, NA, PASS, WATCH, Thresholds, cagr, concentration, fmt_value,
    full_years, hhi_band, overall_verdict, partial_years_from_coverage, viability_signals, yearly_summary,
)
from datamexico.client import ENV_BASE_URL, DataMexicoError, DataMexicoUnreachable, TesseractClient, candidate_base_urls
from datamexico.hs import HSEntry, format_code, overlapping, search

st.set_page_config(page_title="Opportunity Viability", page_icon="📦", layout="wide")

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
VIEW_WORKFLOW = "Recurring-demand workflow"
VIEW_EXPLORER = "Trade explorer"

with st.sidebar:
    view = st.radio("View", [VIEW_WORKFLOW, VIEW_EXPLORER], key="view",
                    help="The workflow checks an opportunity step by step; the trade explorer is the "
                    "original import/export analysis.")
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

    if view == VIEW_EXPLORER:
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
    else:
        thresholds = Thresholds()

mode = theme_mode()


def trade_section(step: str = "", default_query: str = "", show_viability: bool = True):
    """HS search -> basket -> trade data, charts and export. Returns the trade results or None."""

    def num(n: int) -> str:
        return f"{step}.{n}" if step else str(n)


    # ---------------------------------------------------------------------- #
    # Step 1: search HS codes
    # ---------------------------------------------------------------------- #
    st.subheader(f"{num(1)} · Find HS codes for a material")
    with st.spinner("Loading the HS product catalogue…"):
        try:
            hs_catalog, urls = load_hs(base_url, cube_name, locale)
            log_urls(urls)
        except DataMexicoError as exc:
            st.error(f"Could not load HS codes: {exc}")
            return None

    c1, c2, c3 = st.columns([3, 2, 1.2])
    if default_query and st.session_state.get("_hs_default") != default_query:
        st.session_state.hs_query = default_query  # prefill when the opportunity/base material changes
        st.session_state._hs_default = default_query
    query = c1.text_input("Material name or HS code", key="hs_query", placeholder="e.g. toluene, propylene, tolueno, 2902, 2902.30")
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
    st.subheader(f"{num(2)} · Codes in the analysis")
    if not basket:
        st.info("No codes selected yet. Search for a material above and add one or more HS codes.")
        return None

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
        return None

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
        return None
    except DataMexicoError as exc:
        st.error(f"Data request failed: {exc}")
        return None


    def recent(df):
        """Keep rows from MIN_YEAR on (also drops placeholder years such as 0)."""
        return df if df is None or df.empty else df[df["year"] >= MIN_YEAR].reset_index(drop=True)


    by_hs, by_country, by_state = recent(by_hs), recent(by_country), recent(by_state)
    coverage = {y: n for y, n in coverage.items() if y >= MIN_YEAR}

    if by_hs.empty:
        st.warning(f"Data México returned no trade records for the selected codes from {MIN_YEAR} onwards.")
        return None

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

    st.subheader(f"{num(3)} · Mexico's trade in the selected codes")
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

    tab_names = ["Trend", "By HS code", "Origin & destination countries", "Mexican states",
                 "Viability screen", "Data & export"]
    if not show_viability:
        tab_names.remove("Viability screen")
    tab = dict(zip(tab_names, st.tabs(tab_names)))

    # ---- Trend ----------------------------------------------------------- #
    with tab["Trend"]:
        st.plotly_chart(charts.trade_trend(summary, partial, unit, mode), key="trend")
        st.plotly_chart(charts.net_imports(summary.loc[complete] if complete else summary, set(), unit, mode), key="net")
        st.caption("Net imports are shown for complete years only.")
        with st.expander("Table"):
            show = summary[["imports", "exports", "net_imports", "imports_yoy", "exports_yoy"]].copy()
            show.index = charts.year_labels(show.index, partial)
            st.dataframe(show.style.format({"imports": "{:,.0f}", "exports": "{:,.0f}", "net_imports": "{:,.0f}",
                                            "imports_yoy": "{:+.1%}", "exports_yoy": "{:+.1%}"}, na_rep="–"))

    # ---- By HS code ------------------------------------------------------ #
    with tab["By HS code"]:
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
    with tab["Origin & destination countries"]:
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
    with tab["Mexican states"]:
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
    full_summary = summary
    if show_viability:
        with tab["Viability screen"]:
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
    with tab["Data & export"]:
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

    return {"summary": summary, "complete": complete, "partial": partial, "years": years,
            "basket": list(basket.values()), "by_hs": by_hs, "by_country": by_country, "keys": keys,
            "unit": unit}


# ====================================================================== #
# Recurring-demand workflow
# ====================================================================== #
VERTICALS = ["Propylene", "Ethylene", "Chemical/Petrochemical Distribution"]
STEP_1A = "1A · Exact opportunity"
STEP_1B = "1B · Base material"
HS_EXACT = "The exact opportunity"
HS_BASE = "The immediate base material"


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_fx():
    try:
        return mk.fetch_fx()
    except Exception:  # noqa: BLE001 - any failure means "enter rates manually"
        return {}, ""


@st.cache_data(ttl=3600, show_spinner=False)
def gemini_models(api_key: str) -> list[str]:
    try:
        return gem.list_text_models(gem.make_client(api_key))
    except Exception:  # noqa: BLE001 - bad key or network: fall back to the default model
        return []


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # noqa: BLE001 - no secrets file configured
        return ""


def _wf_state() -> dict:
    """All workflow state lives under st.session_state.wf."""
    wf = st.session_state.setdefault("wf", {})
    wf.setdefault("figs", {STEP_1A: mk.empty_figures(), STEP_1B: mk.empty_figures()})
    wf.setdefault("ver", {STEP_1A: 0, STEP_1B: 0, "cat": 0, "prod": 0})
    wf.setdefault("categories", pd.DataFrame({"Category": pd.Series(dtype="str"),
                                              "Contains opportunity": pd.Series(dtype="bool")}))
    wf.setdefault("products", pd.DataFrame({"Parallel product": pd.Series(dtype="str"),
                                            "Is the opportunity": pd.Series(dtype="bool")}))
    wf.setdefault("rationale", "")
    wf.setdefault("register", [])
    wf.setdefault("consulted", [])
    wf.setdefault("notes", {})
    wf.setdefault("companies", om.empty_companies())
    wf["ver"].setdefault("comp", 0)
    wf.setdefault("comp_register", [])
    wf.setdefault("comp_scan_done", False)
    return wf


def _log_consulted(wf: dict, step: str, material: str, pages: list[dict]) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    seen = {(c["step"], c["url"]) for c in wf["consulted"]}
    for page in pages:
        if (step, page.get("url")) not in seen:
            wf["consulted"].append({"logged_at": now, "step": step, "material": material,
                                    "title": page.get("title", ""), "url": page.get("url", "")})


FIG_COLUMN_CONFIG = {
    "use": st.column_config.CheckboxColumn("Use", help="Tick the figures you trust. None ticked = average of all "
                                           "Mexico figures.", default=False),
    "publisher": st.column_config.TextColumn("Publisher"),
    "title": st.column_config.TextColumn("Report / page"),
    "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    "unit": st.column_config.TextColumn("Unit", default="USD million",
                                        help="e.g. USD million, MXN billion, EUR million, kt (thousand tonnes), tonnes"),
    "year": st.column_config.NumberColumn("Year", format="%d", help="Year the figure refers to"),
    "geography": st.column_config.TextColumn("Geography", default="Mexico"),
    "cagr_pct": st.column_config.NumberColumn("CAGR %", format="%.1f", help="Growth rate stated by the source"),
    "url": st.column_config.LinkColumn("Source URL"),
    "origin": st.column_config.TextColumn("Origin", default="Manual"),
    "verified_url": st.column_config.CheckboxColumn("URL seen in search", default=False,
                                                    help="AI figures only: the URL was among the pages the web "
                                                    "search returned."),
    "note": st.column_config.TextColumn("Note"),
}


def _stats_block(stats: mk.MarketStats | None, label: str) -> None:
    if stats is None:
        st.info(f"No usable Mexico market-size figure for {label} yet.")
        return
    u = "USD m" if stats.kind == "value" else "kt"

    def f(v):
        return "–" if v is None else f"{v:,.1f}"
    c = st.columns(5)
    c[0].metric(f"Minimum ({u})", f(stats.minimum))
    c[1].metric(f"Maximum ({u})", f(stats.maximum))
    c[2].metric(f"Average of all · {stats.n_all} ({u})", f(stats.average_all))
    c[3].metric(f"Average of selected · {stats.n_selected} ({u})", f(stats.average_selected))
    c[4].metric(f"Used ({u})", f(stats.used), help=stats.basis)
    st.caption(f"Used value: {stats.basis}. Figures are in {stats.unit} at the reference year; "
               "only Mexico figures enter the range and the average of all.")
    for w in stats.warnings:
        st.warning(w)


def _figures_step(wf: dict, step: str, material: str, opportunity: str, researcher, fx: dict,
                  ref_year: int) -> tuple[pd.DataFrame, mk.MarketStats | None]:
    """Editable figures table + AI search + normalised view + statistics."""
    edited = st.data_editor(
        wf["figs"][step], key=f"figs_{step}_{wf['ver'][step]}", num_rows="dynamic", hide_index=True,
        column_config=FIG_COLUMN_CONFIG, disabled=["verified_url"], width="stretch",
    )
    urls_text = st.text_area(
        "Report URLs for the AI to read (optional, one per line)", key=f"urls_{step}", height=68,
        placeholder="https://www.example.com/mexico-polypropylene-market",
        help="Pages you found yourself (e.g. a publisher's Mexico report page). The AI reads them and extracts "
        "the figures. Leave empty to let the AI find candidate pages.")
    b1, b2 = st.columns([2, 3])
    if b1.button(f"🔎 Find market sizes for “{material or '…'}” with AI", key=f"ai_{step}",
                 disabled=researcher is None or not material):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip().startswith("http")]
        with st.spinner("Researching sources (usually 10 seconds to 2 minutes)…"):
            try:
                res = researcher.find_market_sizes(material, opportunity, urls)
            except ai.AIResearchError as exc:
                st.error(str(exc))
                res = None
        if res is not None:
            new = mk.figures_frame(res.data["figures"])
            wf["figs"][step] = pd.concat([edited, new], ignore_index=True) if len(edited) else new
            wf["ver"][step] += 1
            wf["notes"][step] = res.data.get("notes", "")
            _log_consulted(wf, step, material, res.consulted)
            mk.log_figures(wf["register"], new, step, material)
            st.session_state["_flash"] = (f"AI ({res.data.get('mode', researcher.provider)}) added {len(new)} "
                                          f"figure(s) for {material}." if len(new) else
                                          f"AI ({res.data.get('mode', researcher.provider)}) found no figures for "
                                          f"{material}. Try adding report URLs, or enter figures manually.")
            st.rerun()
    if researcher is None:
        b2.caption("AI search is off: add a Gemini (or Anthropic) API key in the sidebar. You can always add rows "
                   "manually (click + below the table).")
    if wf["notes"].get(step):
        st.caption(f"AI coverage note: {wf['notes'][step]}")

    norm = mk.normalize(mk.figures_frame(edited.to_dict("records")), fx, ref_year)
    skipped = norm[norm["comparable"].isna() & norm["amount"].notna()]
    if len(skipped):
        st.warning("Not used in the calculation: " + "; ".join(
            f"{r.publisher or 'row'} ({r.adjustment})" for r in skipped.itertuples()))
    if len(norm):
        view = norm[["use", "publisher", "amount", "unit", "year", "geography", "at_ref_year", "kind",
                     "adjustment"]].rename(columns={"at_ref_year": f"comparable @ {ref_year}"})
        with st.expander("Comparable figures (converted to USD million or kt, moved to the reference year)"):
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={f"comparable @ {ref_year}": st.column_config.NumberColumn(format="%.1f")})
    stats = mk.primary_stats(norm) if len(norm) else None
    _stats_block(stats, material or "this material")
    vol = mk.market_stats(norm, "volume") if len(norm) else None
    if stats is not None and stats.kind == "value" and vol is not None and vol.found:
        st.caption(f"Volume figures also found: average {vol.average_all or vol.used:,.1f} kt "
                   "(shown separately, never mixed with value figures).")
    return edited, stats


def _list_editor(wf: dict, key: str, df_key: str, col: str, flag: str) -> pd.DataFrame:
    return st.data_editor(
        wf[df_key], key=f"{key}_{wf['ver'][key]}", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={col: st.column_config.TextColumn(col, width="medium"),
                       flag: st.column_config.CheckboxColumn(flag, default=False)},
    )


def _share_input(label: str, n: int, key: str) -> float:
    default = mk.default_share(n)
    c1, c2 = st.columns([2, 3])
    override = c1.checkbox(f"Override {label}", key=f"ovr_{key}")
    value = c2.number_input(label, min_value=0.0, max_value=1.0, value=round(default, 4), step=0.01,
                            format="%.4f", key=f"val_{key}_{n}", disabled=not override)
    share = value if override else default
    st.caption(f"Default {label} = 1 / {n or 1} = {default:.4f}; using **{share:.4f}**"
               + (" (manual override)" if override else ""))
    return share


def _apply_loaded(data: dict) -> None:
    """Restore a saved workflow (JSON from the Market sources tab)."""
    wf = _wf_state()
    st.session_state["opp_name"] = data.get("opportunity", "")
    st.session_state["opp_vertical"] = data.get("vertical", VERTICALS[0])
    st.session_state["opp_base"] = data.get("base_material", "")
    st.session_state["hs_represents"] = data.get("hs_represents", HS_BASE)
    for step in (STEP_1A, STEP_1B):
        wf["figs"][step] = mk.figures_frame(data.get("figures", {}).get(step, []))
        wf["ver"][step] += 1
    wf["categories"] = pd.DataFrame(data.get("categories", []), columns=["Category", "Contains opportunity"])
    wf["products"] = pd.DataFrame(data.get("products", []), columns=["Parallel product", "Is the opportunity"])
    wf["ver"]["cat"] += 1
    wf["ver"]["prod"] += 1
    wf["rationale"] = data.get("rationale", "")
    wf["register"] = list(data.get("register", []))
    wf["consulted"] = list(data.get("consulted", []))
    wf["notes"] = dict(data.get("notes", {}))
    wf["companies"] = om.companies_frame(data.get("competitors", []))
    wf["ver"]["comp"] = wf["ver"].get("comp", 0) + 1
    wf["comp_register"] = list(data.get("competitor_register", []))
    wf["comp_scan_done"] = bool(data.get("competitor_scan_done", False))
    st.session_state["om_none_confirmed"] = bool(data.get("no_supplier_confirmed", False))
    st.session_state.basket = {
        f"{b['digits']}:{b['member_id']}": HSEntry(digits=int(b["digits"]), member_id=str(b["member_id"]),
                                                   code=b.get("code", ""), label=b.get("label", ""),
                                                   parent_label=b.get("parent_label", ""))
        for b in data.get("hs_codes", [])
    }



COMP_COLUMN_CONFIG = {
    "include": st.column_config.CheckboxColumn("Include", default=True,
                                               help="Counts toward the number of competitor groups."),
    "company": st.column_config.TextColumn("Company"),
    "group": st.column_config.TextColumn("Parent group", help="Subsidiaries of one group count once."),
    "type": st.column_config.SelectboxColumn("Type", options=om.COMPANY_TYPES, default="Other"),
    "country": st.column_config.TextColumn("HQ country"),
    "product": st.column_config.TextColumn("Product (as the company describes it)"),
    "url": st.column_config.LinkColumn("Source URL"),
    "origin": st.column_config.TextColumn("Origin", default="Manual"),
    "verified_url": st.column_config.CheckboxColumn("Page seen", default=False,
                                                    help="AI only: the page was actually returned or read."),
    "note": st.column_config.TextColumn("Evidence / note"),
}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_quantity(base_url_: str, cube: str, keys: tuple, measure: str, locale_: str):
    client = TesseractClient(base_url_)
    tc_ = get_cube(base_url_, cube)
    flows, _ = load_flows(base_url_, cube, locale_)
    df = service.fetch_trade(client, tc_, _entries_from_keys(keys), measure, flows, "hs", locale_)
    return df


def _open_market_tab(wf: dict, name: str, base: str, trade_out, researcher) -> dict:
    """Open-market check: import trend, HHI, competitor scan and verdict."""
    mode_ = theme_mode()
    out = {"verdict": None, "evidence": [], "trend_table": None, "hhi_table": None, "comp_fps": [],
           "companies": om.empty_companies()}
    exact_hs = st.session_state.get("hs_represents", HS_BASE) == HS_EXACT
    st.caption("Uses the same HS codes as step 2. "
               + ("They are the **exact opportunity's** codes: imports alone decide the result; the competitor scan "
                  "is optional evidence." if exact_hs else
                  "They are the **base material's** codes: the competitor scan for the exact opportunity is "
                  "required."))

    # ---- Step 1: import trend ------------------------------------------
    st.subheader("4.1 · Import trend (last 5 complete years)")
    trend = None
    if trade_out is None:
        st.info("Select HS codes and fetch trade data in tab 2 first.")
    else:
        series = trade_out["summary"]["imports"]
        measure_used, unit_used = measure, trade_out["unit"]
        qty = om.quantity_measure(tc.measures)
        if qty and qty != measure:
            try:
                qdf = _load_quantity(base_url, cube_name, trade_out["keys"], qty, locale)
                qdf = qdf[qdf["year"] >= MIN_YEAR]
                qsum = qdf[qdf["flow"] == IMPORTS].groupby("year")["value"].sum()
                if qsum.sum() > 0:
                    series, measure_used, unit_used = qsum, qty, ""
            except DataMexicoError:
                pass
        trend = om.import_trend(series, trade_out["complete"], measure_used)
        val_trend = om.import_trend(trade_out["summary"]["imports"], trade_out["complete"], measure)
        if trend.years:
            st.plotly_chart(charts.import_trend_chart(trend.years, trend.values, trend.trend_line, unit_used,
                                                      f"Imports ({measure_used}) and trend", mode_),
                            key="om_trend")
        c = st.columns(4)
        c[0].metric("Trend", trend.classification,
                    help="Growing ≥ +2 %/yr, Stable between −2 % and +2 %, Declining < −2 % (Theil–Sen trend).")
        c[1].metric("Average yearly change (trend)", "–" if trend.slope_pct is None else f"{trend.slope_pct:+.1%}")
        c[2].metric("First → last growth", "–" if trend.first_last_cagr is None else f"{trend.first_last_cagr:+.1%}/yr")
        c[3].metric("Years", f"{trend.years[0]}–{trend.years[-1]}" if trend.years else "–")
        if measure_used != measure and val_trend.slope_pct is not None:
            st.caption(f"Trend based on {measure_used} (volume), so price swings don't distort it. "
                       f"By {measure}: {val_trend.classification} ({val_trend.slope_pct:+.1%}/yr).")
        else:
            st.caption(f"Trend based on {measure_used} (no quantity measure in this cube, so price swings can "
                       "affect it).")
        out["trend_table"] = pd.DataFrame({"year": trend.years, measure_used: trend.values,
                                           "trend line": trend.trend_line})

    # ---- Step 2: HHI ------------------------------------------------------
    st.subheader("4.2 · Supplier-country concentration (HHI) - note only")
    hhi = None
    if trade_out is not None and trend is not None and trend.years:
        hhi = om.hhi_series(trade_out.get("by_country"), trend.years)
        if hhi.table.empty:
            st.info("No partner-country breakdown available for these codes.")
        else:
            st.plotly_chart(charts.hhi_chart(hhi.table, mode_), key="om_hhi")
            st.caption(hhi.note + " HHI never changes the pass/fail result.")
            out["hhi_table"] = hhi.table

    # ---- Step 3: competitors ----------------------------------------------
    st.subheader("4.3 · Competitors supplying the exact opportunity in Mexico"
                 + (" (optional)" if exact_hs else " (required)"))
    st.caption(f"Any domestic or international company selling **{name or 'the opportunity'}** in Mexico counts; "
               "companies selling only the base material do not. Subsidiaries of one group count once; "
               f"{om.MIN_COMPETITORS}+ groups = fragmented market. Add or correct rows manually.")
    edited = st.data_editor(wf["companies"], key=f"companies_{wf['ver']['comp']}", num_rows="dynamic",
                            hide_index=True, column_config=COMP_COLUMN_CONFIG, disabled=["verified_url"],
                            width="stretch")
    b1, b2 = st.columns([2, 3])
    if b1.button(f"🔎 Find companies supplying “{name or '…'}” in Mexico with AI", key="ai_comp",
                 disabled=researcher is None or not name):
        with st.spinner("Searching for suppliers (usually 10 seconds to 2 minutes)…"):
            try:
                res = researcher.find_competitors(name, base)
            except ai.AIResearchError as exc:
                st.error(str(exc))
                res = None
        if res is not None:
            new = om.companies_frame(res.data.get("companies", []))
            wf["companies"] = pd.concat([edited, new], ignore_index=True) if len(edited) else new
            wf["ver"]["comp"] += 1
            wf["comp_scan_done"] = True
            wf["notes"]["competitors"] = res.data.get("notes", "")
            _log_consulted(wf, "Open market · competitors", name, res.consulted)
            om.log_companies(wf["comp_register"], new, name)
            st.session_state["_flash"] = (f"AI ({res.data.get('mode', researcher.provider)}) returned {len(new)} "
                                          "company record(s): review them below.")
            st.rerun()
    if researcher is None:
        b2.caption("AI search is off: add an API key in the sidebar, or enter companies manually.")
    if wf["notes"].get("competitors"):
        st.caption(f"AI note: {wf['notes']['competitors']}")
    out["comp_fps"] = om.log_companies(wf["comp_register"], edited, name)
    out["companies"] = edited
    groups = om.distinct_groups(edited)
    if len(edited.dropna(how="all")):
        wf["comp_scan_done"] = True  # manual entries count as a scan
    n = len(groups)
    reading = ("fragmented" if n >= om.MIN_COMPETITORS else "present but concentrated" if n else "none found")
    st.metric("Distinct competitor groups", n, reading, delta_color="off", delta_arrow="off")
    none_confirmed = False
    if n == 0 and not exact_hs:
        none_confirmed = st.checkbox(
            "Search reviewed: I confirm no domestic or international company supplies this product in Mexico",
            key="om_none_confirmed",
            help="Required before the app concludes 'No market in Mexico'. Check the AI results and any other "
            "sources (expert interviews, distributor catalogues) first.")
    elif "om_none_confirmed" in st.session_state and n:
        st.session_state["om_none_confirmed"] = False

    # ---- Step 4: verdict ---------------------------------------------------
    st.subheader("4.4 · Open-market result")
    verdict = om.open_market_verdict(exact_hs, trend, n if wf["comp_scan_done"] else None, wf["comp_scan_done"],
                                     none_confirmed, hhi, name or "the opportunity")
    banner = {om.OPEN: st.success, om.NOT_OPEN: st.error, om.NO_MARKET: st.error, om.PENDING: st.info}.get(
        verdict.code, st.warning)
    banner(f"{verdict.icon} **{verdict.headline}**  \n{verdict.detail}")
    for note_ in verdict.notes:
        st.caption(f"Note: {note_}")
    evidence = [
        ("HS codes represent", HS_EXACT if exact_hs else HS_BASE),
        ("Import trend", "–" if trend is None else
         f"{trend.classification} ({'–' if trend.slope_pct is None else f'{trend.slope_pct:+.1%}/yr'}, "
         f"{trend.measure}, {trend.years[0] if trend.years else ''}–{trend.years[-1] if trend.years else ''})"),
        ("HHI (latest)", "–" if hhi is None or hhi.latest is None else f"{hhi.latest:,.0f}, {hhi.direction}"),
        ("Competitor groups", str(n) + ("" if wf["comp_scan_done"] else " (scan not run)")),
        ("Groups counted", ", ".join(groups) or "–"),
    ]
    st.dataframe(pd.DataFrame(evidence, columns=["Item", "Value"]), hide_index=True, width="stretch",
                 column_config={"Value": st.column_config.TextColumn(width="large")})
    out.update(verdict=verdict, evidence=evidence)
    return out

def recurring_demand_workflow(researcher, fx: dict, ref_year: int, min_years: int) -> None:
    if "_pending_load" in st.session_state:  # restore a saved assessment before any widget exists
        try:
            _apply_loaded(st.session_state.pop("_pending_load"))
        except (KeyError, TypeError, ValueError) as exc:
            st.session_state["_flash_error"] = f"The saved file is incomplete or damaged: {exc}"
    wf = _wf_state()
    if "_pending_base" in st.session_state:  # set by the AI hierarchy suggestion
        st.session_state["opp_base"] = st.session_state.pop("_pending_base")
    flash = st.session_state.pop("_flash", None)

    st.title("Opportunity viability")
    st.caption("Recurring demand (tabs 1–3): market size in Mexico, imports recurring over the last 5 complete "
               "years, and the conclusion. Open market (tab 4): import trend, supplier concentration and "
               "competitors. Every source used is locked into the Market sources tab.")
    # Always present, so messages appearing or disappearing never shift the tabs
    # (a layout shift above st.tabs resets the selected tab).
    notice = st.container()
    if flash:
        notice.success(flash)
    if st.session_state.get("_flash_error"):
        notice.error(st.session_state.pop("_flash_error"))

    t1, t2, t3, t_om, t4 = st.tabs(["1 · Opportunity & market size", "2 · Trade recurrence",
                                    "3 · Recurring-demand result", "4 · Open market", "🔒 Market sources"])

    # ------------------------------------------------------------------ #
    with t1:
        st.subheader("Opportunity")
        c1, c2, c3 = st.columns([3, 2, 3])
        name = c1.text_input("Opportunity", key="opp_name", placeholder="e.g. Mineral-filled polypropylene")
        c2.selectbox("Vertical", VERTICALS, key="opp_vertical")
        base = c3.text_input("Immediate base material", key="opp_base", placeholder="e.g. Polypropylene",
                             help="The material one step up the chain. Used when the exact opportunity has no "
                             "Mexico market size or HS code.")
        if not name:
            st.info("Enter the opportunity to start.")

        st.subheader(f"Step {STEP_1A}: market size of “{name or 'the opportunity'}” in Mexico")
        edited_a, stats_a = _figures_step(wf, STEP_1A, name, name, researcher, fx, ref_year)
        current_fps = mk.log_figures(wf["register"], edited_a, STEP_1A, name)

        exact_found = stats_a is not None and stats_a.found
        relevance = None
        stats_b = None
        if exact_found:
            st.success("Exact Mexico market size found: step 1B is not needed.")
        with st.expander(f"Step {STEP_1B}: market size of the base material, scaled by relevance",
                         expanded=not exact_found):
            if exact_found:
                st.caption("Not used while the exact opportunity has a usable figure. Kept for reference.")
            st.markdown(f"**Market size of “{base or 'the base material'}” in Mexico**")
            edited_b, stats_b = _figures_step(wf, STEP_1B, base, name, researcher, fx, ref_year)
            current_fps += mk.log_figures(wf["register"], edited_b, STEP_1B, base)

            st.markdown("**Relevance: how much of the base-material market belongs to the opportunity**")
            st.caption("Level 1 splits the base material into its broad categories; level 2 splits the "
                       "opportunity's category into its parallel products. Each defaults to an equal split "
                       "(1 / number of entries) and can be overridden.")
            if st.button("🔎 Suggest base material, categories and parallel products with AI",
                         disabled=researcher is None or not name, key="ai_hier"):
                with st.spinner("Researching the product structure (usually 1–2 minutes)…"):
                    try:
                        res = researcher.suggest_hierarchy(name, base)
                    except ai.AIResearchError as exc:
                        st.error(str(exc))
                        res = None
                if res is not None:
                    d = res.data
                    wf["categories"] = pd.DataFrame({
                        "Category": d["categories"],
                        "Contains opportunity": [c == d.get("opportunity_category") for c in d["categories"]]})
                    wf["products"] = pd.DataFrame({
                        "Parallel product": d["products"],
                        "Is the opportunity": [p == d.get("opportunity_product") for p in d["products"]]})
                    wf["ver"]["cat"] += 1
                    wf["ver"]["prod"] += 1
                    wf["rationale"] = d.get("rationale", "")
                    _log_consulted(wf, "Relevance", d.get("base_material") or base,
                                   res.consulted + [s for s in d.get("sources", []) if isinstance(s, dict)])
                    if d.get("base_material") and not base:
                        st.session_state["_pending_base"] = d["base_material"]
                    st.session_state["_flash"] = (f"AI ({d.get('mode', researcher.provider)}) suggested the product "
                                                  "structure: review it below.")
                    st.rerun()
            if wf["rationale"]:
                st.caption(f"AI rationale: {wf['rationale']}")

            l1, l2 = st.columns(2)
            with l1:
                st.markdown(f"Level 1 · categories of {base or 'the base material'}")
                cats = _list_editor(wf, "cat", "categories", "Category", "Contains opportunity")
                n1 = int(cats["Category"].fillna("").astype(str).str.strip().ne("").sum())
                r1 = _share_input("r1", n1, "r1")
            with l2:
                level2 = st.checkbox("Apply level 2 (parallel products)", value=True, key="level2")
                st.markdown("Level 2 · parallel products within the opportunity's category")
                prods = _list_editor(wf, "prod", "products", "Parallel product", "Is the opportunity")
                n2 = int(prods["Parallel product"].fillna("").astype(str).str.strip().ne("").sum())
                r2 = _share_input("r2", n2, "r2") if level2 else 1.0
            relevance = mk.Relevance(n1, r1, n2, r2, level2)
            st.info(f"Combined relevance = r1 × r2 = {r1:.4f} × {r2 if level2 else 1:.4f} = **{relevance.combined:.4f}**")

        market = mk.market_result(stats_a, stats_b, relevance)
        st.subheader("Market size result")
        if market.found:
            u = "USD m" if market.unit.startswith("USD") else "kt"
            how = ("exact opportunity figures" if market.method == "exact"
                   else f"base material × relevance {relevance.combined:.4f}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Estimated market, used ({u})", f"{market.estimate:,.1f}", help=how)
            m2.metric(f"Range, min – max ({u})", f"{market.low:,.1f} – {market.high:,.1f}")
            m3.metric(f"Average of all figures ({u})",
                      f"{market.average_all:,.1f}" if market.average_all is not None else "–")
            m4.metric("Method", "Exact" if market.method == "exact" else "Derived")
            st.caption(f"Reference year {ref_year}. Method: {how}.")
        else:
            st.warning("No market size established yet (neither the exact opportunity nor the base material has a "
                       "usable Mexico figure).")

    # ------------------------------------------------------------------ #
    trade_rec = None
    trade_out = None
    with t2:
        st.session_state.setdefault("hs_represents", HS_BASE)
        st.radio("The HS codes you select represent", [HS_EXACT, HS_BASE], key="hs_represents", horizontal=True)
        st.caption("Use the exact opportunity's HS code when one exists; otherwise the base material's. Note that "
                   "compounded and filled grades are often still classified under the base polymer's heading, so a "
                   "base-material code usually includes the opportunity along with much else.")
        trade_out = trade_section(step="2", default_query=base or name, show_viability=False)
        if trade_out is not None:
            trade_rec = mk.trade_recurrence(trade_out["summary"], trade_out["complete"], min_years)
            st.subheader("2.4 · Recurrence over the last 5 complete years")
            if not trade_rec.years:
                st.warning("No complete years of trade data available.")
            else:
                tbl = trade_rec.table.copy()
                tbl["imports present"] = ["✓" if v > 0 else "✗" for v in tbl["imports"]]
                tbl["net imports positive"] = ["✓" if v > 0 else "✗" for v in tbl["net_imports"]]
                st.dataframe(tbl.style.format({"imports": "{:,.0f}", "exports": "{:,.0f}", "net_imports": "{:,.0f}"}),
                             width="stretch")
                r1c, r2c, r3c = st.columns(3)
                r1c.metric("Years with imports", f"{trade_rec.years_with_imports} of {len(trade_rec.years)}",
                           help=f"Recurring needs at least {trade_rec.min_years}.")
                r2c.metric("Years with positive net imports", f"{trade_rec.net_positive_years} of {len(trade_rec.years)}")
                r3c.metric("Recurrence", trade_rec.strength)

    # ------------------------------------------------------------------ #
    verdict = mk.recurring_demand_verdict(market, trade_rec)
    with t3:
        banner = {mk.ESTABLISHED: st.success, mk.NOT_ESTABLISHED: st.error, mk.PENDING: st.info}.get(
            verdict.code, st.warning)
        banner(f"{verdict.icon} **{verdict.headline}**  \n{verdict.detail}")
        hs_codes = ", ".join(entry_name(e) for e in (trade_out or {}).get("basket", [])) or "–"
        evidence = [
            ("Opportunity", name or "–"),
            ("Vertical", st.session_state.get("opp_vertical", "")),
            ("Immediate base material", base or "–"),
            ("Market size method", {"exact": "Exact opportunity figures", "derived": "Base material × relevance",
                                    "none": "Not found"}[market.method]),
            ("Estimated market (used)", f"{market.estimate:,.1f} {market.unit}" if market.found else "–"),
            ("Market range (min – max)", f"{market.low:,.1f} – {market.high:,.1f} {market.unit}" if market.found else "–"),
            ("Average of all figures", f"{market.average_all:,.1f} {market.unit}"
             if market.found and market.average_all is not None else "–"),
            ("Relevance (r1 × r2)", f"{relevance.combined:.4f}" if market.method == "derived" else "–"),
            ("HS codes", hs_codes),
            ("HS codes represent", st.session_state.get("hs_represents", HS_BASE) if trade_out else "–"),
            ("Trade years assessed", ", ".join(map(str, trade_rec.years)) if trade_rec and trade_rec.years else "–"),
            ("Years with imports", f"{trade_rec.years_with_imports} of {len(trade_rec.years)}" if trade_rec else "–"),
            ("Years with positive net imports", f"{trade_rec.net_positive_years} of {len(trade_rec.years)}"
             if trade_rec else "–"),
            ("Recurrence strength", trade_rec.strength if trade_rec else "–"),
        ]
        evidence_df = pd.DataFrame(evidence, columns=["Item", "Value"])
        st.dataframe(evidence_df, hide_index=True, width="stretch", height=36 * (len(evidence_df) + 1) + 4,
                     column_config={"Value": st.column_config.TextColumn(width="large")})

        # Cross-check: imports should not exceed the market they belong to.
        if market.found and market.unit.startswith("USD") and trade_rec and trade_rec.years:
            last = trade_rec.years[-1]
            imports_m = float(trade_out["summary"].loc[last, "imports"]) / 1e6
            represents_base = st.session_state.get("hs_represents") == HS_BASE
            compare_to = (market.stats.used if (represents_base and market.method == "derived")
                          else market.estimate)
            label = "base-material market" if represents_base and market.method == "derived" else "estimated market"
            if compare_to and imports_m > compare_to:
                st.warning(f"Cross-check: {last} imports ({imports_m:,.1f} USD m) exceed the {label} "
                           f"({compare_to:,.1f} USD m). The market-size figures may be too low or not comparable.")
            elif compare_to:
                st.caption(f"Cross-check: {last} imports are {imports_m:,.1f} USD m, "
                           f"{imports_m / compare_to:.0%} of the {label} ({compare_to:,.1f} USD m).")

    # ------------------------------------------------------------------ #
    with t_om:
        om_out = _open_market_tab(wf, name, base, trade_out, researcher)

    # ------------------------------------------------------------------ #
    with t4:
        st.subheader("🔒 Market sources")
        st.caption("Every market-size figure found by AI or entered manually is logged here and cannot be removed. "
                   "Edited figures are logged again as new entries; the earlier version stays.")
        reg = mk.register_frame(wf["register"], current_fps)
        if reg.empty:
            st.info("No market sources logged yet.")
        else:
            st.dataframe(reg.drop(columns=["fingerprint"]), hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("Source URL")})
        st.markdown("**Competitor evidence (open-market check)**")
        comp_reg = pd.DataFrame(wf["comp_register"])
        if comp_reg.empty:
            st.caption("None yet.")
        else:
            comp_reg["status"] = ["in calculation" if fp in om_out["comp_fps"] else
                                  "edited / removed from calculation" for fp in comp_reg["fingerprint"]]
            st.dataframe(comp_reg.drop(columns=["fingerprint"]), hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("Source URL")})
        st.markdown("**Pages consulted by AI searches**")
        consulted = pd.DataFrame(wf["consulted"], columns=["logged_at", "step", "material", "title", "url"])
        if consulted.empty:
            st.caption("None yet.")
        else:
            st.dataframe(consulted, hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("URL")})

        st.markdown("**Save or export this assessment**")
        saved = {
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "opportunity": name, "vertical": st.session_state.get("opp_vertical", ""), "base_material": base,
            "hs_represents": st.session_state.get("hs_represents", HS_BASE),
            "figures": {STEP_1A: edited_a.to_dict("records"), STEP_1B: edited_b.to_dict("records")},
            "categories": cats.to_dict("records"),
            "products": prods.to_dict("records"),
            "rationale": wf["rationale"], "register": wf["register"], "consulted": wf["consulted"],
            "notes": wf["notes"],
            "hs_codes": [{"digits": e.digits, "member_id": e.member_id, "code": e.code, "label": e.label,
                          "parent_label": e.parent_label} for e in st.session_state.get("basket", {}).values()],
            "verdict": {"code": verdict.code, "headline": verdict.headline, "detail": verdict.detail},
            "competitors": om_out["companies"].to_dict("records"),
            "competitor_register": wf["comp_register"],
            "competitor_scan_done": wf["comp_scan_done"],
            "no_supplier_confirmed": bool(st.session_state.get("om_none_confirmed", False)),
            "open_market_verdict": {"code": om_out["verdict"].code, "headline": om_out["verdict"].headline,
                                    "detail": om_out["verdict"].detail},
        }
        stem = "".join(ch if ch.isalnum() else "_" for ch in (name or "opportunity"))[:50]
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame([("Verdict", f"{verdict.icon} {verdict.headline}"), ("Detail", verdict.detail)]
                         + evidence, columns=["Item", "Value"]).to_excel(writer, sheet_name="Result", index=False)
            for s, df_ in ((STEP_1A, edited_a), (STEP_1B, edited_b)):
                df_.to_excel(writer, sheet_name=f"Figures {s[:2]}", index=False)
            cats.to_excel(writer, sheet_name="Relevance L1", index=False)
            prods.to_excel(writer, sheet_name="Relevance L2", index=False)
            if trade_rec is not None:
                trade_rec.table.to_excel(writer, sheet_name="Trade recurrence")
            reg.drop(columns=["fingerprint"]).to_excel(writer, sheet_name="Market sources", index=False)
            omv = om_out["verdict"]
            pd.DataFrame([("Verdict", f"{omv.icon} {omv.headline}"), ("Detail", omv.detail)]
                         + [("Note", n) for n in omv.notes] + om_out["evidence"],
                         columns=["Item", "Value"]).to_excel(writer, sheet_name="Open market", index=False)
            if om_out["trend_table"] is not None:
                om_out["trend_table"].to_excel(writer, sheet_name="Import trend", index=False)
            if om_out["hhi_table"] is not None and not om_out["hhi_table"].empty:
                om_out["hhi_table"].to_excel(writer, sheet_name="HHI by year", index=False)
            om_out["companies"].to_excel(writer, sheet_name="Competitors", index=False)
            consulted.to_excel(writer, sheet_name="Pages consulted", index=False)
        d1, d2 = st.columns(2)
        d1.download_button("Download Excel report", buffer.getvalue(), file_name=f"recurring_demand_{stem}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        d2.download_button("Save assessment (JSON)", json.dumps(saved, default=str, indent=1).encode(),
                           file_name=f"recurring_demand_{stem}.json", mime="application/json",
                           help="Reload it later with the uploader below to continue where you left off.")
        upload = st.file_uploader("Load a saved assessment (JSON)", type=["json"], key="wf_upload")
        if upload is not None and st.session_state.get("_loaded_upload") != (upload.name, upload.size):
            try:
                data = json.loads(upload.getvalue())
                if not isinstance(data, dict) or "figures" not in data:
                    raise ValueError("not a saved recurring-demand assessment")
            except ValueError as exc:
                st.error(f"Could not read that file: {exc}")
            else:
                st.session_state["_pending_load"] = data
                st.session_state["_loaded_upload"] = (upload.name, upload.size)
                st.session_state["_flash"] = f"Loaded assessment from {upload.name}."
                st.rerun()


# ====================================================================== #
# Page
# ====================================================================== #
if view == VIEW_EXPLORER:
    st.title("Mexico trade explorer")
    st.caption(
        "Material → HS codes → yearly imports and exports for Mexico, with signals for the "
        "four Phase-1 viability tests. Source: Data México (Secretaría de Economía)."
    )
    trade_section()
else:
    with st.sidebar:
        st.header("Recurring-demand settings")
        wf_min_years = st.slider("Min. years with imports (of last 5)", 1, 5, 4, key="wf_min_years")
        wf_ref_year = st.number_input("Market-size reference year", 2015, 2100, dt.date.today().year - 1,
                                      key="wf_ref_year",
                                      help="Figures for other years are moved to this year with their own CAGR.")
        fx_rates, fx_date = load_fx()
        with st.expander("FX rates (units per 1 USD)"):
            st.caption(f"ECB reference rates of {fx_date}." if fx_date else
                       "Could not load live rates: enter them for any non-USD currency you use.")
            fx_rates = dict(fx_rates)
            for cur in ("MXN", "EUR"):
                fx_rates[cur] = st.number_input(cur, min_value=0.0, value=float(fx_rates.get(cur, 0.0)),
                                                format="%.4f", key=f"fx_{cur}") or None
            fx_rates = {k: v for k, v in fx_rates.items() if v}
        st.header("AI research")
        gem_key = _secret("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        claude_key = _secret("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        providers = [gem.GeminiResearcher.provider, ai.ClaudeResearcher.provider]
        provider = st.radio("Provider", providers, key="ai_provider",
                            index=1 if claude_key and not gem_key else 0)
        if provider == gem.GeminiResearcher.provider:
            key = st.text_input("Gemini API key", type="password", key="gem_key", value=gem_key,
                                help="Store it in the app's Secrets as GEMINI_API_KEY so you don't have to paste it.")
            models = gemini_models(key) if key else []
            preferred = _secret("GEMINI_MODEL") or gem.DEFAULT_MODEL
            options = models or [preferred]
            if preferred not in options:
                options = [preferred] + options
            model = st.selectbox("Gemini model", options, index=options.index(preferred), key="gem_model",
                                 help="'-latest' aliases always point to Google's current model. Flash-Lite has the "
                                 "most generous free-tier limits; Flash is stronger but busier.")
            researcher = gem.GeminiResearcher(key, model) if key else None
            st.caption("Uses Google Search when your plan allows it, otherwise reads report pages. "
                       "About 2–3 requests per AI click.")
        else:
            key = st.text_input("Anthropic API key", type="password", key="ai_key", value=claude_key,
                                help="Store it in the app's Secrets as ANTHROPIC_API_KEY.")
            researcher = ai.ClaudeResearcher(key) if key else None
            st.caption(f"Model: {ai.MODEL} with web search. Each AI search uses API credits.")
    recurring_demand_workflow(researcher, fx_rates, int(wf_ref_year), int(wf_min_years))
