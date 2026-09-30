"""High-level operations used by the UI: catalogue, HS search, trade queries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import pandas as pd

from .analysis import EXPORTS, IMPORTS, classify_flow, period_coverage, rows_to_frame
from .client import DataMexicoError, TesseractClient
from .hs import HSEntry, build_entries
from .schema import Level, TradeCube, find_trade_cubes


def trade_cubes(client: TesseractClient) -> list[TradeCube]:
    cubes = find_trade_cubes(client.cubes())
    if not cubes:
        raise DataMexicoError(
            "No cube with HS product, trade flow and year levels was found in /cubes."
        )
    return cubes


def hs_entries(client: TesseractClient, tc: TradeCube, locale: str = "en") -> dict[int, list[HSEntry]]:
    """Fetch every HS level's members (labels in ``locale`` plus the other language)."""
    members = {d: client.members(tc.name, lvl.param, locale) for d, lvl in tc.hs_levels.items()}
    alt_locale = "es" if locale == "en" else "en"
    alt: dict[int, dict[str, str]] = {}
    for d, lvl in tc.hs_levels.items():
        try:
            alt[d] = {m.id: m.label for m in client.members(tc.name, lvl.param, alt_locale)}
        except DataMexicoError:
            alt[d] = {}
    return build_entries(members, alt)


def flow_map(client: TesseractClient, tc: TradeCube, locale: str = "en") -> dict[str, str]:
    """Map Flow member IDs to Imports/Exports using their labels."""
    mapping = {}
    for m in client.members(tc.name, tc.flow.param, locale):
        kind = classify_flow(m.label)
        if kind:
            mapping[m.id] = kind
    if set(mapping.values()) != {IMPORTS, EXPORTS}:
        raise DataMexicoError(f"Could not identify import/export flows from members {mapping}")
    return mapping


def _group_by_level(selected: Iterable[HSEntry]) -> dict[int, list[HSEntry]]:
    groups: dict[int, list[HSEntry]] = defaultdict(list)
    for entry in selected:
        groups[entry.digits].append(entry)
    return dict(sorted(groups.items()))


def fetch_trade(
    client: TesseractClient,
    tc: TradeCube,
    selected: Sequence[HSEntry],
    measure: str,
    flows: dict[str, str],
    breakdown: str = "hs",
    locale: str = "en",
) -> pd.DataFrame:
    """Yearly imports/exports for the selected HS codes.

    ``breakdown`` adds one extra dimension to each row:
    ``"hs"`` (per HS code), ``"country"`` (partner country) or ``"state"``
    (Mexican state). One request is made per HS level selected.
    """
    frames = []
    for digits, entries in _group_by_level(selected).items():
        hs_level = tc.hs_levels[digits]
        cuts = {hs_level.param: [e.member_id for e in entries]}
        extra: dict[str, Level]
        if breakdown == "hs":
            extra = {"hs": hs_level}
        elif breakdown == "country" and tc.country:
            extra = {"country": tc.country}
        elif breakdown == "state" and tc.state:
            extra = {"state": tc.state}
        else:
            raise DataMexicoError(f"Breakdown '{breakdown}' is not available in cube {tc.name}")
        drilldowns = [tc.year.param, tc.flow.param] + [lvl.param for lvl in extra.values()]
        rows = client.data(tc.name, drilldowns, [measure], cuts, locale)
        frame = rows_to_frame(rows, tc.year, tc.flow, measure, flows, extra, hs_digits=digits)
        if breakdown == "hs":
            frame["hs_digits"] = digits
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_coverage(
    client: TesseractClient,
    tc: TradeCube,
    selected: Sequence[HSEntry],
    measure: str,
    locale: str = "en",
) -> dict[int, int]:
    """Months (or quarters) reported per year, used to flag partial years."""
    if tc.period is None:
        return {}
    coverage: dict[int, int] = {}
    for digits, entries in _group_by_level(selected).items():
        hs_level = tc.hs_levels[digits]
        rows = client.data(
            tc.name,
            [tc.year.param, tc.period.param],
            [measure],
            {hs_level.param: [e.member_id for e in entries]},
            locale,
        )
        for year, n in period_coverage(rows, tc.year, tc.period).items():
            coverage[year] = max(coverage.get(year, 0), n)
    return coverage
