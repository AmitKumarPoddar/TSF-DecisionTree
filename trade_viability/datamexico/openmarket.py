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
