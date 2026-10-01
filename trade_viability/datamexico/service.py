"""High-level operations used by the UI: catalogue, HS search, trade queries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import pandas as pd

from .analysis import EXPORTS, IMPORTS, classify_flow, period_coverage, rows_to_frame
from .client import DataMexicoError, DataMexicoUnreachable, Member, TesseractClient
from .hs import HSEntry, build_entries
from .schema import Level, TradeCube, find_trade_cubes


def trade_cubes(client: TesseractClient) -> list[TradeCube]:
    cubes = find_trade_cubes(client.cubes())
    if not cubes:
        raise DataMexicoError(
            "No cube with HS product, trade flow and year levels was found in /cubes."
        )
    return cubes


def _unlabelled(members: Sequence[Member]) -> bool:
    """True when most members came back without a caption (label == ID)."""
    if not members:
        return True
    bare = sum(1 for m in members if m.label.strip() == m.id.strip())
    return bare > len(members) / 2


def level_members(client: TesseractClient, tc: TradeCube, level: Level, locale: str) -> list[Member]:
    """Members of ``level`` with labels, falling back to the data endpoint."""
    members: list[Member] = []
    try:
        members = client.members(tc.name, level.param, locale)
    except DataMexicoError:
        pass
    if _unlabelled(members):
        from_data = client.labels_from_data(tc.name, level.param, tc.value_measure, locale)
        if from_data and not _unlabelled(from_data):
            return from_data
        if not members:
            members = from_data
    if not members:
        raise DataMexicoError(f"No members returned for level {level.param}")
    return members


def hs_entries(client: TesseractClient, tc: TradeCube, locale: str = "en") -> dict[int, list[HSEntry]]:
    """Fetch every HS level's members (labels in ``locale`` plus the other language)."""
    members = {d: level_members(client, tc, lvl, locale) for d, lvl in tc.hs_levels.items()}
    alt_locale = "es" if locale == "en" else "en"
    alt: dict[int, dict[str, str]] = {}
    for d, lvl in tc.hs_levels.items():
        try:
            alt[d] = {m.id: m.label for m in level_members(client, tc, lvl, alt_locale)}
        except DataMexicoError:
            alt[d] = {}
    return build_entries(members, alt)


def flow_map(client: TesseractClient, tc: TradeCube, locale: str = "en") -> dict[str, str]:
    """Map Flow member IDs to Imports/Exports using their labels."""
    attempts = []
    for loc in (locale, "es" if locale == "en" else "en"):
        for fetch in (
            lambda: client.members(tc.name, tc.flow.param, loc),
            lambda: client.labels_from_data(tc.name, tc.flow.param, tc.value_measure, loc),
        ):
            try:
                members = fetch()
            except DataMexicoError:
                continue
            attempts.append({m.id: m.label for m in members})
            mapping = {m.id: kind for m in members if (kind := classify_flow(m.label))}
            if set(mapping.values()) == {IMPORTS, EXPORTS}:
                return mapping
    raise DataMexicoError(
        f"Could not identify import/export flows; Flow members returned: {attempts}"
    )


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
    measure: str,
    locale: str = "en",
    min_year: int = 2015,
) -> dict[int, int]:
    """Months (or quarters) reported per year across the whole cube.

    Measured over all products, not the selected ones: a product that is not
    traded every month must not make a complete year look partial.
    """
    if tc.period is None:
        return {}
    import datetime as _dt
    years = [str(y) for y in range(min_year, _dt.date.today().year + 1)]
    try:  # restricting to recent years keeps this whole-cube query light
        rows = client.data(tc.name, [tc.year.param, tc.period.param], [measure],
                           {tc.year.param: years}, locale)
    except DataMexicoUnreachable:
        raise
    except DataMexicoError:  # some year members may not exist in the cube
        rows = client.data(tc.name, [tc.year.param, tc.period.param], [measure], locale=locale)
    return period_coverage(rows, tc.year, tc.period)


def cubes_with(cubes: Sequence[TradeCube], attr: str) -> list[TradeCube]:
    """Trade cubes having a ``country`` or ``state`` level, best first."""
    return [c for c in cubes if getattr(c, attr) is not None]


def compare_cubes(
    client: TesseractClient,
    cubes: Sequence[TradeCube],
    selected: Sequence[HSEntry],
    locale: str = "en",
) -> pd.DataFrame:
    """Yearly imports and exports of the selection in every detected trade cube.

    Lets the analyst check which cube matches the published national totals.
    """
    frames = []
    for tc in cubes:
        usable = [e for e in selected if e.digits in tc.hs_levels]
        if not usable:
            continue
        try:
            flows = flow_map(client, tc, locale)
            df = fetch_trade(client, tc, usable, tc.value_measure, flows, "hs", locale)
        except DataMexicoError as exc:
            frames.append(pd.DataFrame([{"cube": tc.name, "year": None, "flow": f"error: {exc}"[:120], "value": None}]))
            continue
        df = df.groupby(["year", "flow"], as_index=False)["value"].sum()
        df["cube"] = tc.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.pivot_table(index=["cube", "flow"], columns="year", values="value", aggfunc="sum")
