"""Turn raw Data México rows into yearly trade tables and viability signals.

The four viability tests come from the Phase-1 approach document:

1. Recurring demand      - does Mexican demand repeat year over year?
2. Open market           - can the market move to a new supplier?
3. Service repeatability - can Indelpro supply it repeatedly? (judgement)
4. Economic margin       - is a margin currently being earned? (judgement)

Tests 1 and 2 can be screened from trade data; 3 and 4 need the analyst's
input, which the UI collects and records next to the computed indicators.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .hs import hs_code
from .schema import Level

IMPORTS = "Imports"
EXPORTS = "Exports"

PASS, WATCH, FAIL, MANUAL, NA = "Pass", "Watch", "Fail", "Manual", "n/a"


# ---------------------------------------------------------------------- #
# Row normalisation
# ---------------------------------------------------------------------- #
def classify_flow(label: str) -> str | None:
    """Map a flow label (English or Spanish) to Imports/Exports."""
    text = str(label).lower()
    if re.search(r"import", text):
        return IMPORTS
    if re.search(r"export", text):
        return EXPORTS
    return None


def level_columns(columns: Iterable[str], level: Level) -> tuple[str | None, str | None]:
    """Return (id_column, label_column) for a drilled-down level."""
    cols = set(columns)
    names = [level.param, level.name]
    id_col = next((f"{n} ID" for n in names if f"{n} ID" in cols), None)
    label_col = next((n for n in names if n in cols), None)
    return id_col or label_col, label_col or id_col


def rows_to_frame(
    rows: Sequence[Mapping],
    year: Level,
    flow: Level,
    measure: str,
    flow_map: Mapping[str, str] | None = None,
    dims: Mapping[str, Level] | None = None,
    hs_digits: int | None = None,
) -> pd.DataFrame:
    """Normalise API rows into a long table.

    Output columns: ``year``, ``flow`` (Imports/Exports), one ``<key>_id`` and
    ``<key>`` column per entry in ``dims`` and ``value``. For the ``hs`` key an
    extra ``code`` column holds the standard HS code.
    """
    dims = dict(dims or {})
    base_cols = ["year", "flow"]
    for key in dims:
        base_cols += [f"{key}_id", key] + (["code"] if key == "hs" else [])
    if not rows:
        return pd.DataFrame(columns=base_cols + ["value"])

    raw = pd.DataFrame(list(rows))
    if measure not in raw.columns:
        raise KeyError(f"measure '{measure}' not in response columns {list(raw.columns)}")

    out = pd.DataFrame()
    year_id, year_label = level_columns(raw.columns, year)
    out["year"] = pd.to_numeric(raw[year_id], errors="coerce")
    if out["year"].isna().all() and year_label:
        out["year"] = pd.to_numeric(raw[year_label], errors="coerce")

    flow_id, flow_label = level_columns(raw.columns, flow)
    flow_map = {str(k): v for k, v in (flow_map or {}).items()}
    out["flow"] = [
        flow_map.get(str(fid)) or classify_flow(flab)
        for fid, flab in zip(raw[flow_id], raw[flow_label])
    ]

    for key, level in dims.items():
        id_col, label_col = level_columns(raw.columns, level)
        out[f"{key}_id"] = raw[id_col].astype(str)
        out[key] = raw[label_col].astype(str)
        if key == "hs":
            out["code"] = [hs_code(i, hs_digits or 6) for i in raw[id_col]]

    out["value"] = pd.to_numeric(raw[measure], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["year", "flow"])
    out["year"] = out["year"].astype(int)
    return out[base_cols + ["value"]].reset_index(drop=True)


# ---------------------------------------------------------------------- #
# Yearly metrics
# ---------------------------------------------------------------------- #
def yearly_summary(
    df: pd.DataFrame,
    production: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """Imports, exports, net imports and growth per year.

    If domestic ``production`` (same unit as the trade measure) is supplied,
    apparent consumption = production + imports - exports and
    import dependence = imports / apparent consumption are added, matching
    the method used in the approach document's worked example.
    """
    cols = ["imports", "exports", "net_imports", "imports_yoy", "exports_yoy"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    pivot = (
        df.groupby(["year", "flow"])["value"].sum().unstack("flow").reindex(columns=[IMPORTS, EXPORTS])
    )
    years = range(int(pivot.index.min()), int(pivot.index.max()) + 1)
    pivot = pivot.reindex(years).fillna(0.0)
    out = pd.DataFrame(index=pd.Index(years, name="year"))
    out["imports"] = pivot[IMPORTS]
    out["exports"] = pivot[EXPORTS]
    out["net_imports"] = out["imports"] - out["exports"]
    out["imports_yoy"] = _pct_change(out["imports"])
    out["exports_yoy"] = _pct_change(out["exports"])
    if production:
        prod = pd.Series({int(k): float(v) for k, v in production.items() if v is not None and not _isnan(v)})
        out["production"] = prod.reindex(out.index)
        out["apparent_consumption"] = out["production"] + out["imports"] - out["exports"]
        consumption = out["apparent_consumption"].where(out["apparent_consumption"] > 0)
        out["import_dependence"] = out["imports"] / consumption
    return out


def _pct_change(series: pd.Series) -> pd.Series:
    prev = series.shift(1)
    return (series - prev) / prev.where(prev != 0)


def _isnan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def cagr(first: float, last: float, periods: int) -> float | None:
    """Compound annual growth rate, or None when undefined."""
    if periods <= 0 or first is None or last is None or first <= 0 or last < 0:
        return None
    return (last / first) ** (1 / periods) - 1


def full_years(
    summary: pd.DataFrame,
    partial_years: Iterable[int] = (),
    today: _dt.date | None = None,
) -> list[int]:
    """Years that are complete (drops the current calendar year and known partial years)."""
    today = today or _dt.date.today()
    partial = set(partial_years)
    return [int(y) for y in summary.index if y < today.year and y not in partial]


def period_coverage(rows: Sequence[Mapping], year: Level, period: Level) -> dict[int, int]:
    """Count the distinct months/quarters reported per year."""
    if not rows:
        return {}
    raw = pd.DataFrame(list(rows))
    year_id, _ = level_columns(raw.columns, year)
    period_id, _ = level_columns(raw.columns, period)
    if year_id is None or period_id is None:
        return {}
    counts = raw.groupby(pd.to_numeric(raw[year_id], errors="coerce"))[period_id].nunique()
    return {int(y): int(n) for y, n in counts.items() if not _isnan(y)}


def partial_years_from_coverage(coverage: Mapping[int, int]) -> list[int]:
    """Years reporting fewer periods than the most complete year."""
    if not coverage:
        return []
    full = max(coverage.values())
    return sorted(y for y, n in coverage.items() if n < full)


# ---------------------------------------------------------------------- #
# Supplier concentration
# ---------------------------------------------------------------------- #
@dataclass
class Concentration:
    year: int
    total: float
    hhi: float  # 0..10,000
    top1_name: str
    top1_share: float
    top3_share: float
    n_partners: int
    shares: pd.DataFrame  # partner, value, share


def concentration(df: pd.DataFrame, partner_col: str, year: int, flow: str = IMPORTS) -> Concentration | None:
    """Herfindahl-Hirschman index and top shares of trade partners in ``year``."""
    sub = df[(df["year"] == year) & (df["flow"] == flow)]
    if sub.empty:
        return None
    shares = sub.groupby(partner_col)["value"].sum().sort_values(ascending=False)
    shares = shares[shares > 0]
    total = float(shares.sum())
    if total <= 0:
        return None
    frac = shares / total
    table = pd.DataFrame({"partner": shares.index, "value": shares.values, "share": frac.values})
    return Concentration(
        year=year,
        total=total,
        hhi=float((frac.mul(100) ** 2).sum()),
        top1_name=str(shares.index[0]),
        top1_share=float(frac.iloc[0]),
        top3_share=float(frac.iloc[:3].sum()),
        n_partners=int(len(shares)),
        shares=table,
    )


def hhi_band(hhi: float) -> str:
    """US DOJ/FTC market concentration bands."""
    if hhi < 1500:
        return "unconcentrated"
    if hhi < 2500:
        return "moderately concentrated"
    return "highly concentrated"


# ---------------------------------------------------------------------- #
# Viability screening
# ---------------------------------------------------------------------- #
@dataclass
class Thresholds:
    window_years: int = 5
    min_years_with_imports: int = 4
    min_avg_imports: float = 1_000_000.0  # in the selected measure's unit
    max_cv: float = 0.5
    min_import_dependence: float = 0.30
    max_hhi: float = 2500.0
    min_import_cagr: float = 0.0


@dataclass
class Signal:
    test: str
    indicator: str
    value: str
    status: str
    note: str = ""


def viability_signals(
    summary: pd.DataFrame,
    years: Sequence[int],
    thresholds: Thresholds,
    supplier: Concentration | None = None,
    manual: Mapping[str, tuple[str, str]] | None = None,
    unit: str = "",
) -> tuple[list[Signal], dict[str, str]]:
    """Compute indicator rows and an overall status for each of the 4 tests.

    ``years`` should be the complete years available; the last
    ``thresholds.window_years`` of them form the assessment window.
    ``manual`` maps test name -> (status, note) for tests 3 and 4.
    """
    t = thresholds
    signals: list[Signal] = []
    verdict: dict[str, str] = {}
    window = [y for y in sorted(years) if y in summary.index][-t.window_years:]
    manual = dict(manual or {})

    # ---- Test 1: recurring demand -------------------------------------
    test = "1. Recurring demand"
    if not window:
        signals.append(Signal(test, "Complete years of data", "0", FAIL, "No complete years returned"))
        verdict[test] = FAIL
    else:
        imp = summary.loc[window, "imports"]
        present = int((imp > 0).sum())
        avg = float(imp.mean())
        cv = float(imp.std(ddof=0) / avg) if avg > 0 else float("inf")
        s_present = PASS if present >= min(t.min_years_with_imports, len(window)) else FAIL
        s_avg = PASS if avg >= t.min_avg_imports else FAIL
        s_cv = PASS if cv <= t.max_cv else WATCH
        signals += [
            Signal(test, f"Years with imports ({window[0]}-{window[-1]})", f"{present} of {len(window)}", s_present,
                   f"needs >= {t.min_years_with_imports}"),
            Signal(test, "Average annual imports", f"{fmt_value(avg)} {unit}".strip(), s_avg,
                   f"needs >= {fmt_value(t.min_avg_imports)}"),
            Signal(test, "Volatility of imports (coef. of variation)", fmt_num(cv, 2), s_cv,
                   f"stable if <= {t.max_cv:.2f}"),
        ]
        verdict[test] = FAIL if FAIL in (s_present, s_avg) else (WATCH if s_cv == WATCH else PASS)

    # ---- Test 2: open market ------------------------------------------
    test = "2. Open market"
    if not window:
        verdict[test] = FAIL
    else:
        last = window[-1]
        row = summary.loc[last]
        net_importer = row["imports"] > row["exports"]
        signals.append(Signal(test, f"Net importer in {last}", f"net imports {fmt_value(row['net_imports'])} {unit}".strip(),
                              PASS if net_importer else FAIL, "imports exceed exports"))
        checks = [PASS if net_importer else FAIL]

        dep = row.get("import_dependence") if "import_dependence" in summary.columns else None
        if dep is not None and not _isnan(dep):
            s_dep = PASS if dep >= t.min_import_dependence else FAIL
            signals.append(Signal(test, f"Import dependence {last}", f"{dep:.0%}", s_dep,
                                  f"imports / apparent consumption; needs >= {t.min_import_dependence:.0%}"))
            checks.append(s_dep)
        else:
            signals.append(Signal(test, "Import dependence", NA, NA,
                                  "enter domestic production to compute apparent consumption"))

        growth = cagr(float(summary.loc[window[0], "imports"]), float(row["imports"]), len(window) - 1)
        s_growth = NA if growth is None else (PASS if growth >= t.min_import_cagr else WATCH)
        signals.append(Signal(test, f"Import CAGR {window[0]}-{last}", NA if growth is None else f"{growth:+.1%}",
                              s_growth, "growing market eases entry"))

        if supplier is not None:
            s_hhi = PASS if supplier.hhi < t.max_hhi else WATCH
            signals.append(Signal(test, f"Supplier-country concentration {supplier.year} (HHI)",
                                  f"{supplier.hhi:,.0f} ({hhi_band(supplier.hhi)})", s_hhi,
                                  f"top origin {supplier.top1_name} {supplier.top1_share:.0%}; top 3 {supplier.top3_share:.0%}"))
        else:
            s_hhi = NA
            signals.append(Signal(test, "Supplier-country concentration", NA, NA, "no country breakdown available"))

        if FAIL in checks:
            verdict[test] = FAIL
        elif WATCH in (s_growth, s_hhi):
            verdict[test] = WATCH
        else:
            verdict[test] = PASS

    # ---- Tests 3 & 4: analyst judgement ---------------------------------
    for test, hint in (
        ("3. Service repeatability",
         "Can existing terminal/storage/logistics/permits handle this material repeatedly?"),
        ("4. Economic margin",
         "Is a margin earned today (distributor spread, import premium, price gap)?"),
    ):
        status, note = manual.get(test, (MANUAL, ""))
        value = note or ("not yet assessed" if status == MANUAL else "no evidence recorded")
        signals.append(Signal(test, "Analyst assessment", value, status, hint))
        verdict[test] = status

    return signals, verdict


def overall_verdict(verdict: Mapping[str, str]) -> str:
    statuses = list(verdict.values())
    if FAIL in statuses:
        return "Not viable"
    if MANUAL in statuses:
        return "Pending analyst input"
    if WATCH in statuses:
        return "Viable - with watch points"
    return "Viable"


# ---------------------------------------------------------------------- #
# Formatting
# ---------------------------------------------------------------------- #
def fmt_value(value: float) -> str:
    """Compact number: 1.2B, 34.5M, 12.3K."""
    if value is None or _isnan(value):
        return "-"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= div:
            return f"{sign}{v / div:,.1f}{suffix}"
    return f"{sign}{v:,.0f}"


def fmt_num(value: float, decimals: int = 1) -> str:
    if value is None or _isnan(value) or math.isinf(value):
        return "-"
    return f"{value:,.{decimals}f}"
