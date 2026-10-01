"""Recurring-demand workflow: market sizes, relevance, trade recurrence, verdict.

Workflow (agreed with the study team):

Step 1  Market size in Mexico
   1A  Figures for the exact opportunity. If any usable figure exists, 1B is skipped.
   1B  Figures for the immediate base material, scaled by a two-level relevance:
         opportunity market = base market x r1 x r2
       where r1 = share of the base material's category that contains the
       opportunity (default 1 / number of categories) and r2 = share of the
       opportunity among its parallel products (default 1 / number of products).
   Output: min, max and average of all figures, plus the average of the
   figures the user selects (all figures when none are selected).

Step 2  Trade recurrence (Data México), last 5 complete years only:
   recurring  = imports in at least ``min_years`` of those years
   strength   = High if net imports are positive in every one of them

Step 3  Verdict
   market found + recurring trade      -> recurring demand established
   market found + no recurring trade   -> demand exists, recurrence not evidenced
   no market    + recurring trade      -> established from trade only
   no market    + no recurring trade   -> the recurring demand could not be established
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import pandas as pd

TRADE_WINDOW = 5  # years of trade used for the recurrence conclusion

# --------------------------------------------------------------------------- #
# Market-size figures
# --------------------------------------------------------------------------- #
FIG_COLUMNS = [
    "use", "publisher", "title", "amount", "unit", "year", "geography",
    "cagr_pct", "url", "origin", "verified_url", "note",
]

UNIT_OPTIONS = [
    "USD million", "USD billion", "MXN million", "MXN billion", "EUR million", "EUR billion",
    "kt (thousand tonnes)", "tonnes", "Mt (million tonnes)",
]

_CURRENCY_UNIT = re.compile(r"^\s*([A-Za-z]{3})\s+(thousand|million|billion)\b", re.I)
_SCALE = {"thousand": 1e-3, "million": 1.0, "billion": 1e3}  # -> millions
_VOLUME_UNIT = {"kt": 1.0, "thousand tonnes": 1.0, "tonnes": 1e-3, "t": 1e-3, "mt": 1e3,
                "million tonnes": 1e3}


def empty_figures() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in FIG_COLUMNS})
    df["use"] = df["use"].astype("bool")
    df["verified_url"] = df["verified_url"].astype("bool")
    for c in ("amount", "year", "cagr_pct"):
        df[c] = df[c].astype("float")
    return df


def figures_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    """Build a figures table from dicts (missing columns filled)."""
    df = pd.DataFrame(list(rows))
    base = empty_figures()
    if df.empty:
        return base
    for c in FIG_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[FIG_COLUMNS].copy()
    df["use"] = df["use"].fillna(False).astype(bool)
    df["verified_url"] = df["verified_url"].fillna(False).astype(bool)
    for c in ("amount", "year", "cagr_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def parse_unit(unit: str) -> tuple[str, str | None, float | None]:
    """Return (kind, currency, multiplier-to-millions-or-kt) for a unit label.

    kind is ``value`` (money, normalised to millions of USD), ``volume``
    (normalised to kilotonnes) or ``unknown``.
    """
    text = str(unit or "").strip()
    m = _CURRENCY_UNIT.match(text)
    if m:
        return "value", m.group(1).upper(), _SCALE[m.group(2).lower()]
    low = text.lower()
    low = re.sub(r"\(.*?\)", "", low).strip()
    if low in _VOLUME_UNIT:
        return "volume", None, _VOLUME_UNIT[low]
    return "unknown", None, None


def is_mexico(geography) -> bool:
    return bool(re.search(r"m[eé]xic", str(geography or ""), re.I))


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def normalize(df: pd.DataFrame, fx_per_usd: Mapping[str, float], ref_year: int) -> pd.DataFrame:
    """Add comparable amounts: USD millions (value) or kilotonnes (volume), at ``ref_year``.

    ``fx_per_usd`` maps currency -> units per 1 USD (USD itself may be omitted).
    Amounts for another year are moved to ``ref_year`` with the figure's own
    CAGR when given; otherwise they are kept as reported and flagged.
    """
    out = df.copy()
    kinds, comparable, adjusted, notes = [], [], [], []
    for row in out.itertuples(index=False):
        kind, currency, mult = parse_unit(row.unit)
        amount = _num(row.amount)
        note = []
        base = None
        if kind == "unknown":
            note.append("unit not recognised")
        elif amount is None or amount <= 0:
            note.append("no amount")
        elif kind == "value":
            rate = 1.0 if currency == "USD" else _num((fx_per_usd or {}).get(currency))
            if rate in (None, 0):
                note.append(f"no FX rate for {currency}")
            else:
                base = amount * mult / rate
        else:
            base = amount * mult
        adj = base
        year, cagr = _num(row.year), _num(row.cagr_pct)
        if base is not None and year is not None and int(year) != ref_year:
            if cagr is not None:
                adj = base * (1 + cagr / 100) ** (ref_year - int(year))
                note.append(f"moved {int(year)}→{ref_year} at {cagr:g}%/yr")
            else:
                note.append(f"{int(year)} figure, not adjusted (no CAGR)")
        if not is_mexico(row.geography):
            note.append("not Mexico-specific")
        kinds.append(kind)
        comparable.append(base)
        adjusted.append(adj)
        notes.append("; ".join(note))
    out["kind"] = kinds
    out["comparable"] = comparable
    out["at_ref_year"] = adjusted
    out["mexico"] = [is_mexico(g) for g in out["geography"]]
    out["adjustment"] = notes
    return out


@dataclass
class MarketStats:
    kind: str                # "value" (USD m) or "volume" (kt)
    unit: str
    n_all: int               # Mexico figures with a usable amount
    minimum: float | None
    maximum: float | None
    average_all: float | None
    n_selected: int
    average_selected: float | None
    used: float | None       # average_selected if any selected, else average_all
    basis: str
    warnings: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.used is not None


def market_stats(norm: pd.DataFrame, kind: str) -> MarketStats:
    """Range and averages for one kind of figure (value or volume)."""
    unit = "USD million" if kind == "value" else "kt"
    usable = norm[(norm["kind"] == kind) & norm["at_ref_year"].notna()]
    mex = usable[usable["mexico"]]
    sel = usable[usable["use"].astype(bool)]
    vals = mex["at_ref_year"].astype(float)
    warnings = []
    avg_all = float(vals.mean()) if len(vals) else None
    avg_sel = float(sel["at_ref_year"].astype(float).mean()) if len(sel) else None
    if len(sel) and (~sel["mexico"]).any():
        warnings.append("A selected figure is not Mexico-specific.")
    if avg_sel is not None:
        used, basis = avg_sel, f"average of {len(sel)} selected figure(s)"
    elif avg_all is not None:
        used, basis = avg_all, f"average of all {len(vals)} Mexico figure(s) (none selected)"
    else:
        used, basis = None, "no usable Mexico figure"
    return MarketStats(
        kind=kind, unit=unit, n_all=len(vals),
        minimum=float(vals.min()) if len(vals) else None,
        maximum=float(vals.max()) if len(vals) else None,
        average_all=avg_all, n_selected=len(sel), average_selected=avg_sel,
        used=used, basis=basis, warnings=warnings,
    )


def primary_stats(norm: pd.DataFrame) -> MarketStats | None:
    """Value statistics when available, otherwise volume; None if neither."""
    for kind in ("value", "volume"):
        stats = market_stats(norm, kind)
        if stats.found:
            return stats
    return None


# --------------------------------------------------------------------------- #
# Two-level relevance
# --------------------------------------------------------------------------- #
def default_share(n: int) -> float:
    return 1.0 / n if n and n > 0 else 1.0


@dataclass
class Relevance:
    n_categories: int
    r1: float
    n_products: int
    r2: float
    level2: bool = True

    @property
    def combined(self) -> float:
        return self.r1 * (self.r2 if self.level2 else 1.0)


@dataclass
class MarketResult:
    method: str                    # "exact", "derived" or "none"
    stats: MarketStats | None      # stats of the figures used (exact or base)
    relevance: Relevance | None    # only for "derived"
    estimate: float | None         # opportunity market (used x relevance)
    low: float | None
    high: float | None
    average_all: float | None
    unit: str = ""

    @property
    def found(self) -> bool:
        return self.estimate is not None


def market_result(exact: MarketStats | None, base: MarketStats | None,
                  relevance: Relevance | None) -> MarketResult:
    """Combine step 1A and 1B into the opportunity's market size."""
    if exact is not None and exact.found:
        return MarketResult("exact", exact, None, exact.used, exact.minimum, exact.maximum,
                            exact.average_all, exact.unit)
    if base is not None and base.found and relevance is not None:
        r = relevance.combined

        def scale(v):
            return None if v is None else v * r
        return MarketResult("derived", base, relevance, scale(base.used), scale(base.minimum),
                            scale(base.maximum), scale(base.average_all), base.unit)
    return MarketResult("none", None, None, None, None, None, None)


# --------------------------------------------------------------------------- #
# Trade recurrence
# --------------------------------------------------------------------------- #
@dataclass
class TradeRecurrence:
    years: list[int]
    years_with_imports: int
    net_positive_years: int
    min_years: int
    recurring: bool
    net_positive_all: bool
    table: pd.DataFrame

    @property
    def strength(self) -> str:
        if not self.recurring:
            return "Not recurring"
        return "High" if self.net_positive_all else "Moderate"


def trade_recurrence(summary: pd.DataFrame, complete_years: Sequence[int],
                     min_years: int = 4, window: int = TRADE_WINDOW) -> TradeRecurrence:
    """Recurrence over the last ``window`` complete years only."""
    years = [y for y in sorted(complete_years) if y in summary.index][-window:]
    table = summary.loc[years, ["imports", "exports", "net_imports"]].copy() if years else \
        pd.DataFrame(columns=["imports", "exports", "net_imports"])
    present = int((table["imports"] > 0).sum()) if years else 0
    net_pos = int((table["net_imports"] > 0).sum()) if years else 0
    need = min(min_years, len(years)) if years else min_years
    recurring = bool(years) and present >= need
    return TradeRecurrence(years, present, net_pos, need, recurring,
                           bool(years) and net_pos == len(years), table)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
ESTABLISHED = "established"
NOT_RECURRING = "demand_not_recurring"
TRADE_ONLY = "trade_only"
NOT_ESTABLISHED = "not_established"
PENDING = "pending"

NOT_ESTABLISHED_TEXT = "The recurring demand could not be established."


@dataclass
class Verdict:
    code: str
    icon: str
    headline: str
    detail: str


def recurring_demand_verdict(market: MarketResult, trade: TradeRecurrence | None) -> Verdict:
    if trade is None:
        return Verdict(PENDING, "⏳", "Pending trade check",
                       "Select HS codes and fetch trade data in step 2 to complete the assessment.")
    if market.found and trade.recurring:
        return Verdict(ESTABLISHED, "✅", "Recurring demand established",
                       f"Market size found ({market.method}) and imports in {trade.years_with_imports} of "
                       f"the last {len(trade.years)} years; strength: {trade.strength}.")
    if market.found:
        return Verdict(NOT_RECURRING, "⚠️", "Demand exists, but recurring imports are not evidenced",
                       f"Market size found ({market.method}), but imports in only {trade.years_with_imports} "
                       f"of the last {len(trade.years)} years (needs {trade.min_years}).")
    if trade.recurring:
        return Verdict(TRADE_ONLY, "⚠️", "Recurring demand established from trade data only (lower confidence)",
                       f"No market-size figure found, but imports in {trade.years_with_imports} of the last "
                       f"{len(trade.years)} years; strength: {trade.strength}.")
    return Verdict(NOT_ESTABLISHED, "❌", NOT_ESTABLISHED_TEXT,
                   "No market-size figure was found and imports do not recur.")


# --------------------------------------------------------------------------- #
# Market sources register (append-only)
# --------------------------------------------------------------------------- #
REGISTER_COLUMNS = [
    "logged_at", "step", "material", "publisher", "title", "amount", "unit", "year",
    "geography", "cagr_pct", "url", "origin", "verified_url", "fingerprint",
]


def fingerprint(row: Mapping, step: str) -> str:
    parts = [step] + [str(row.get(k, "")).strip().lower()
                      for k in ("publisher", "title", "amount", "unit", "year", "geography", "url")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _has_content(row: Mapping) -> bool:
    return any(str(row.get(k) or "").strip() not in ("", "nan", "None")
               for k in ("publisher", "title", "url", "amount"))


def log_figures(register: list[dict], df: pd.DataFrame, step: str, material: str,
                now: _dt.datetime | None = None) -> list[str]:
    """Append every figure not yet logged; returns current fingerprints.

    Entries are never removed or overwritten: an edited figure is logged as a
    new entry and the earlier version stays in the register.
    """
    now = now or _dt.datetime.now()
    known = {e["fingerprint"] for e in register}
    current = []
    for row in df.to_dict("records"):
        if not _has_content(row):
            continue
        fp = fingerprint(row, step)
        current.append(fp)
        if fp in known:
            continue
        entry = {k: row.get(k) for k in REGISTER_COLUMNS if k in row}
        entry.update(logged_at=now.strftime("%Y-%m-%d %H:%M"), step=step, material=material,
                     fingerprint=fp, origin=row.get("origin") or "Manual")
        register.append(entry)
        known.add(fp)
    return current


def register_frame(register: Sequence[dict], current: Iterable[str]) -> pd.DataFrame:
    current = set(current)
    df = pd.DataFrame(list(register), columns=REGISTER_COLUMNS)
    df["status"] = ["in calculation" if fp in current else "edited / removed from calculation"
                    for fp in df["fingerprint"]]
    return df


# --------------------------------------------------------------------------- #
# FX
# --------------------------------------------------------------------------- #
FX_URL = "https://api.frankfurter.app/latest?from=USD"


def fetch_fx(session=None, timeout: float = 10.0) -> tuple[dict[str, float], str]:
    """Latest ECB reference rates as units per USD, with their date."""
    import requests
    sess = session or requests
    resp = sess.get(FX_URL, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return {k.upper(): float(v) for k, v in body.get("rates", {}).items()}, str(body.get("date", ""))
