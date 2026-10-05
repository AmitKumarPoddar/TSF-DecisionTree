"""Open-market check: can a new supplier enter the Mexican market?

Agreed rules:

* **Import trend** over the last 5 complete years, same HS codes as the
  recurring-demand step. Average yearly change from a Theil-Sen trend line
  (median of pairwise slopes, robust to a single spike year), shown with simple first-to-last growth:
  >= +2 %/yr Growing, between -2 % and +2 % Stable, < -2 % Declining.
  Uses a quantity measure when the cube has one (prices move values).
* **HHI** of import origins by country, per year: supporting note only, never
  changes the result. < 1,500 many countries; > 2,500 concentrated supply.
* **Competitor scan** for the exact opportunity only (never base-material
  sellers). Any domestic or international company selling it in Mexico counts;
  subsidiaries of one group count once. 4+ groups = fragmented.

Result
  Path A (HS code of the exact opportunity):
      no imports -> not open; declining -> not open; growing/stable -> open.
  Path B (HS code of the base material):
      4+ competitors & growing/stable -> open; 4+ & declining -> watch;
      1-3 -> watch (concentrated); 0 -> "no market in Mexico" once the analyst
      confirms the search was reviewed (provisional until then).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .analysis import IMPORTS, concentration, hhi_band

WINDOW = 5
BAND = 0.02            # +/- 2 % a year separates growing / stable / declining
MIN_COMPETITORS = 4    # distinct company groups for a fragmented market

GROWING, STABLE, DECLINING, NO_IMPORTS = "Growing", "Stable", "Declining", "No imports"

_QTY = re.compile(r"quantit|volume|weight|net ?weight|cantidad|peso|kg|kilo|ton", re.I)


def quantity_measure(measures: Iterable[str]) -> str | None:
    """A quantity-type measure of the cube, if any."""
    return next((m for m in measures if _QTY.search(m)), None)


# --------------------------------------------------------------------------- #
# Import trend
# --------------------------------------------------------------------------- #
@dataclass
class ImportTrend:
    years: list[int]
    values: list[float]
    measure: str
    slope_pct: float | None       # Theil-Sen slope / mean, per year
    first_last_cagr: float | None
    trend_line: list[float]
    band: float = BAND

    @property
    def classification(self) -> str:
        if not self.years or sum(self.values) <= 0:
            return NO_IMPORTS
        if self.slope_pct is None:
            return NO_IMPORTS
        if self.slope_pct >= self.band:
            return GROWING
        if self.slope_pct <= -self.band:
            return DECLINING
        return STABLE

    @property
    def ok(self) -> bool:
        return self.classification in (GROWING, STABLE)


def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust trend line: median of all pairwise slopes (one spike year cannot swing it)."""
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def import_trend(yearly: pd.Series, complete_years: Sequence[int], measure: str = "",
                 window: int = WINDOW, band: float = BAND) -> ImportTrend:
    """Classify the import trend of the last ``window`` complete years.

    ``yearly`` is imports indexed by year.
    """
    years = [y for y in sorted(complete_years) if y in yearly.index][-window:]
    values = [float(yearly.loc[y]) for y in years]
    if len(years) < 2 or sum(values) <= 0:
        return ImportTrend(years, values, measure, None, None, values[:], band)
    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    slope, intercept = theil_sen(x, y)
    mean = float(y.mean())
    slope_pct = float(slope / mean) if mean > 0 else None
    first, last = values[0], values[-1]
    cagr = (last / first) ** (1 / (len(years) - 1)) - 1 if first > 0 and last >= 0 else None
    line = [float(slope * xi + intercept) for xi in x]
    return ImportTrend(years, values, measure, slope_pct, cagr, line, band)


# --------------------------------------------------------------------------- #
# HHI by year
# --------------------------------------------------------------------------- #
@dataclass
class HHISeries:
    table: pd.DataFrame          # year, hhi, band, top_partner, top_share, partners
    latest: float | None
    first: float | None

    @property
    def direction(self) -> str:
        if self.latest is None or self.first is None or len(self.table) < 2:
            return "–"
        change = self.latest - self.first
        if change <= -250:
            return "falling (supply spreading across more countries)"
        if change >= 250:
            return "rising (supply concentrating)"
        return "broadly unchanged"

    @property
    def note(self) -> str:
        if self.latest is None:
            return "No partner-country data for HHI."
        band = hhi_band(self.latest)
        lead = ("Supply from many countries supports an open market." if self.latest < 1500 else
                "Watch: concentrated supply." if self.latest > 2500 else "Moderately concentrated supply.")
        return f"HHI {self.latest:,.0f} ({band}), {self.direction}. {lead}"


def hhi_series(by_country: pd.DataFrame | None, years: Sequence[int]) -> HHISeries:
    rows = []
    if by_country is not None and not by_country.empty:
        for y in years:
            c = concentration(by_country, "country", y, IMPORTS)
            if c:
                rows.append({"year": y, "hhi": c.hhi, "band": hhi_band(c.hhi), "top_partner": c.top1_name,
                             "top_share": c.top1_share, "partners": c.n_partners})
    table = pd.DataFrame(rows, columns=["year", "hhi", "band", "top_partner", "top_share", "partners"])
    return HHISeries(table, float(table["hhi"].iloc[-1]) if len(table) else None,
                     float(table["hhi"].iloc[0]) if len(table) else None)


def country_shares(by_country: pd.DataFrame | None, years: Sequence[int], top: int = 6) -> pd.DataFrame:
    """Share of Mexico's imports by origin country, per year (rows: countries, columns: years).

    Countries outside the ``top`` of the latest year are grouped as "Other countries".
    """
    if by_country is None or by_country.empty or not years:
        return pd.DataFrame()
    imp = by_country[(by_country["flow"] == IMPORTS) & (by_country["year"].isin(list(years)))]
    if imp.empty:
        return pd.DataFrame()
    pivot = imp.pivot_table(index="country", columns="year", values="value", aggfunc="sum").fillna(0.0)
    pivot = pivot[[y for y in years if y in pivot.columns]]
    shares = pivot / pivot.sum(axis=0).replace(0, np.nan)
    order = shares.iloc[:, -1].fillna(0).sort_values(ascending=False).index
    shares = shares.loc[order]
    if len(shares) > top:
        rest = shares.iloc[top:].sum(axis=0)
        shares = shares.iloc[:top]
        shares.loc["Other countries"] = rest
    shares.index.name = "country"
    return shares


# --------------------------------------------------------------------------- #
# Competitors
# --------------------------------------------------------------------------- #
COMP_COLUMNS = ["include", "company", "group", "type", "country", "product", "url", "origin",
                "verified_url", "note"]
COMPANY_TYPES = ["Mexican producer/compounder", "Distributor/importer in Mexico",
                 "International supplier selling into Mexico", "Other"]

_SUFFIX = re.compile(r"\b(s\.?a\.?(\s*de\s*c\.?v\.?)?|s\.?\s*de\s*r\.?l\.?(\s*de\s*c\.?v\.?)?|inc\.?|llc|ltd\.?|"
                     r"gmbh|corp(oration)?\.?|co\.?|company|group|grupo|plc|ag|bv|nv|spa|s\.?a\.?b\.?)\b", re.I)


def empty_companies() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COMP_COLUMNS})
    df["include"] = df["include"].astype("bool")
    df["verified_url"] = df["verified_url"].astype("bool")
    return df


def companies_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return empty_companies()
    for c in COMP_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COMP_COLUMNS].copy()
    df["include"] = df["include"].fillna(True).astype(bool)
    df["verified_url"] = df["verified_url"].fillna(False).astype(bool)
    return df.reset_index(drop=True)


def group_key(company, group) -> str:
    """Normalised parent group (falls back to the company name)."""
    name = str(group or "").strip()
    if not name or name.lower() in ("nan", "none"):
        name = str(company or "").strip()
    name = _SUFFIX.sub(" ", name.lower())
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def group_names(df: pd.DataFrame) -> list[str]:
    """Display name of each distinct included group: the group as written, else the company."""
    seen, out = set(), []
    for row in df.itertuples(index=False):
        if not bool(row.include) or not str(row.company or "").strip() or str(row.company) == "nan":
            continue
        key = group_key(row.company, row.group)
        if key and key not in seen:
            seen.add(key)
            g = str(row.group or "").strip()
            out.append(g if g and g.lower() not in ("nan", "none") else str(row.company).strip())
    return out


def distinct_groups(df: pd.DataFrame) -> list[str]:
    seen, out = set(), []
    for row in df.itertuples(index=False):
        if not bool(row.include) or not str(row.company or "").strip() or str(row.company) == "nan":
            continue
        key = group_key(row.company, row.group)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def log_companies(register: list[dict], df: pd.DataFrame, opportunity: str,
                  now: _dt.datetime | None = None) -> list[str]:
    """Append-only log of competitor evidence (like the market-size register)."""
    now = now or _dt.datetime.now()
    known = {e["fingerprint"] for e in register}
    current = []
    for row in df.to_dict("records"):
        if not str(row.get("company") or "").strip() or str(row.get("company")) == "nan":
            continue
        parts = [str(row.get(k, "")).strip().lower() for k in ("company", "group", "type", "country", "url")]
        fp = hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
        current.append(fp)
        if fp in known:
            continue
        register.append({"logged_at": now.strftime("%Y-%m-%d %H:%M"), "opportunity": opportunity,
                         **{k: row.get(k) for k in COMP_COLUMNS if k != "include"},
                         "origin": row.get("origin") or "Manual", "fingerprint": fp})
        known.add(fp)
    return current


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
OPEN, NOT_OPEN, WATCH, NO_MARKET, PENDING = "open", "not_open", "watch", "no_market", "pending"
NO_SUPPLIER_TEXT = "No company in Mexico, local or international, was found to supply {opp}."
NO_MARKET_TEXT = "No market in Mexico for this opportunity."


@dataclass
class OpenMarketVerdict:
    code: str
    icon: str
    headline: str
    detail: str
    notes: list[str] = field(default_factory=list)


def open_market_verdict(exact_hs: bool, trend: ImportTrend | None, n_groups: int | None,
                        scan_done: bool, none_confirmed: bool, hhi: HHISeries | None,
                        opportunity: str = "the opportunity", threshold: int = MIN_COMPETITORS) -> OpenMarketVerdict:
    notes = [hhi.note] if hhi is not None else []
    if trend is None:
        return OpenMarketVerdict(PENDING, "⏳", "Pending trade data",
                                 "Select HS codes and fetch trade data in step 2.", notes)
    cls = trend.classification
    chg = "" if trend.slope_pct is None else f" ({trend.slope_pct:+.1%}/yr)"

    if exact_hs:
        if cls == NO_IMPORTS:
            return OpenMarketVerdict(NOT_OPEN, "❌", "Not an open market",
                                     "The opportunity's own HS code shows no imports in the last 5 years.", notes)
        if cls == DECLINING:
            return OpenMarketVerdict(NOT_OPEN, "❌", "Not an open market",
                                     f"Direct imports are declining{chg}.", notes)
        if n_groups:
            notes.append(f"{n_groups} competitor group(s) recorded (optional evidence).")
        return OpenMarketVerdict(OPEN, "✅", "Open market",
                                 f"Direct imports of the product are {cls.lower()}{chg}: demand exists and a new "
                                 "supplier can enter.", notes)

    # Path B: base-material HS code -> competitor scan is required.
    if not scan_done:
        return OpenMarketVerdict(PENDING, "⏳", "Competitor scan needed",
                                 "The HS codes are the base material's, so run (or enter) the competitor scan for "
                                 "the exact opportunity.", notes)
    n = n_groups or 0
    if n == 0:
        if none_confirmed:
            return OpenMarketVerdict(NO_MARKET, "❌", NO_MARKET_TEXT,
                                     NO_SUPPLIER_TEXT.format(opp=opportunity) + " No supplier means no demand.", notes)
        return OpenMarketVerdict(PENDING, "⏳", "No supplier found yet (provisional)",
                                 NO_SUPPLIER_TEXT.format(opp=opportunity) + " Review the search (add any supplier you "
                                 "know) and tick the confirmation to conclude.", notes)
    trade_ok = trend.ok
    base = f"base-material imports {cls.lower()}{chg}"
    if n >= threshold:
        if trade_ok:
            return OpenMarketVerdict(OPEN, "✅", "Open market",
                                     f"{n} competitor groups (fragmented) and {base}.", notes)
        return OpenMarketVerdict(WATCH, "⚠️", "Fragmented market, but the base-material market is shrinking",
                                 f"{n} competitor groups; {base}.", notes)
    if trade_ok:
        return OpenMarketVerdict(WATCH, "⚠️", "Market exists but is concentrated",
                                 f"Only {n} competitor group(s) (fewer than {threshold}); {base}. Entry is harder.",
                                 notes)
    return OpenMarketVerdict(WATCH, "⚠️", "Concentrated and shrinking market",
                             f"Only {n} competitor group(s); {base}.", notes)


# --------------------------------------------------------------------------- #
# Conclusion paragraph (plain English, ready to paste)
# --------------------------------------------------------------------------- #
def join_names(names: Sequence[str]) -> str:
    names = [n for n in names if n]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _pct(x: float) -> str:
    v = abs(x) * 100
    return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"


def open_market_paragraph(verdict: OpenMarketVerdict, trend: ImportTrend | None, shares: pd.DataFrame,
                          companies: Sequence[str], opportunity: str, material: str, exact_hs: bool,
                          scan_done: bool, none_confirmed: bool = False) -> str:
    """The open-market conclusion as one plain paragraph, without brackets, dashes or colons."""
    opp = opportunity or "the opportunity"
    if verdict.code == OPEN:
        out = [f"Yes, the market for {opp} in Mexico is open for a new supplier."]
    elif verdict.code in (NOT_OPEN, NO_MARKET):
        out = [f"No, the market for {opp} in Mexico is not open for a new supplier."]
    elif verdict.code == WATCH:
        out = [f"The market for {opp} in Mexico may be open for a new supplier, but it needs a closer look "
               "before entering."]
    else:
        out = [f"The open market check for {opp} in Mexico is not complete yet."]

    if trend is not None and trend.years:
        y0, y1 = trend.years[0], trend.years[-1]
        what = f"Mexico's imports of {material}" + ("," if not exact_hs else "")
        cls = trend.classification
        if cls == NO_IMPORTS:
            out.append(f"Mexico recorded no imports of {material}{'' if exact_hs else ','} between {y0} and {y1}.")
        elif cls == GROWING:
            out.append(f"{what} have increased by about {_pct(trend.slope_pct)} a year between {y0} and {y1}.")
        elif cls == DECLINING:
            out.append(f"{what} have decreased by about {_pct(trend.slope_pct)} a year between {y0} and {y1}.")
        else:
            out.append(f"{what} have remained broadly stable between {y0} and {y1}.")

    if shares is not None and not shares.empty:
        year = shares.columns[-1]
        latest = shares[year].drop("Other countries", errors="ignore").dropna()
        latest = latest[latest > 0]
        if len(latest):
            hhi = float(((shares[year].fillna(0)) ** 2).sum() * 10000)
            named = [f"{c} with {_pct(v)}" for c, v in latest.head(3).items()]
            if len(latest) == 1 or latest.iloc[0] >= 0.5 or hhi > 2500:
                if latest.iloc[0] >= 0.5:
                    out.append(f"The imports are concentrated, with {latest.index[0]} alone supplying "
                               f"{_pct(latest.iloc[0])} of the total in {year}.")
                else:
                    out.append(f"The imports are concentrated in a few countries, mainly {join_names(named)} "
                               f"in {year}.")
            else:
                out.append(f"The imports are fragmented across multiple countries, such as {join_names(named)} "
                           f"in {year}.")

    names = list(companies)
    shown = join_names(names[:6])
    if names and len(names) >= MIN_COMPETITORS:
        out.append(f"Many manufacturers, such as {shown}, supply {opp} in the Mexican market, which validates "
                   "the fragmentation of the market and is a good signal to enter it.")
    elif names:
        out.append(f"Only a few manufacturers, such as {shown}, supply {opp} in the Mexican market, so the supply "
                   "is concentrated in a few hands.")
    elif scan_done and none_confirmed:
        out.append(f"No manufacturer supplying {opp} in the Mexican market could be found, which means there is "
                   "no market for it in Mexico yet.")
    elif not exact_hs:
        out.append(f"The manufacturers supplying {opp} in the Mexican market still need to be confirmed.")

    if verdict.code == OPEN and not (names and len(names) >= MIN_COMPETITORS):
        out.append("Overall, this is a good signal to enter the market.")
    elif verdict.code == WATCH and trend is not None and trend.classification == DECLINING:
        out.append("As the imports are shrinking, entry should be weighed carefully.")
    elif verdict.code in (NOT_OPEN, NO_MARKET):
        out.append("This is not a good signal to enter the market.")
    return " ".join(out)
