"""Discover the foreign-trade cube and its levels from the API's /cubes metadata.

The exact cube and level names on Data México have changed over time, so they
are not hard-coded. Instead every cube is inspected and scored on whether it
looks like an HS-product x trade-flow x year cube.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Level-name patterns (English and Spanish), matched case-insensitively.
_YEAR = re.compile(r"^(year|a[nñ]o|anio)$", re.I)
_PERIOD = re.compile(r"^(month|quarter|mes|trimestre)$", re.I)
_FLOW = re.compile(r"^(flow|trade flow|flujo|flujo comercial)$", re.I)
_COUNTRY = re.compile(r"^(country|pa[ií]s|partner country|origin country)$", re.I)
_STATE = re.compile(r"^(state|estado|entidad|entidad federativa)$", re.I)
_HS_DIGITS = re.compile(r"\bHS\s*-?\s*(\d{1,2})\b|\b(\d{1,2})\s*d[ií]git", re.I)
_TARIFF = re.compile(r"fracci[oó]n|tariff", re.I)
_VALUE_MEASURE = re.compile(r"trade value|valor comercial|^valor$|^value$", re.I)
_TRADE_CUBE = re.compile(r"foreign_trade|comercio_exterior|trade", re.I)


@dataclass(frozen=True)
class Level:
    """A level of a cube hierarchy, e.g. ``HS6`` in ``Product > HS``."""

    name: str
    dimension: str
    hierarchy: str
    depth: int
    dim_type: str = "standard"
    unique_name: str | None = None

    @property
    def param(self) -> str:
        """Name used in drilldowns/cuts (logic layer uses the unique name)."""
        return self.unique_name or self.name


@dataclass
class Cube:
    name: str
    levels: list[Level]
    measures: list[str]
    annotations: dict = field(default_factory=dict)

    def find(self, pattern: re.Pattern, time_only: bool | None = None) -> Level | None:
        for level in self.levels:
            if time_only is True and level.dim_type != "time":
                continue
            if time_only is False and level.dim_type == "time":
                continue
            if pattern.search(level.name) or (
                level.unique_name and pattern.search(level.unique_name)
            ):
                return level
        return None


@dataclass
class TradeCube:
    """A cube that can answer "imports/exports of HS code X by year"."""

    cube: Cube
    year: Level
    flow: Level
    hs_levels: dict[int, Level]
    country: Level | None
    state: Level | None
    period: Level | None
    value_measure: str
    score: float

    @property
    def name(self) -> str:
        return self.cube.name

    @property
    def measures(self) -> list[str]:
        return self.cube.measures

    def describe(self) -> str:
        hs = ", ".join(f"HS{d}={lvl.param}" for d, lvl in sorted(self.hs_levels.items()))
        extras = [
            f"country={self.country.param}" if self.country else None,
            f"state={self.state.param}" if self.state else None,
            f"period={self.period.param}" if self.period else None,
        ]
        return f"{self.name} [{hs}; flow={self.flow.param}; year={self.year.param}; " + ", ".join(
            e for e in extras if e
        ) + "]"


def parse_cubes(raw_cubes: list[dict]) -> list[Cube]:
    """Convert raw /cubes JSON (either Tesseract generation) into :class:`Cube`s."""
    cubes: list[Cube] = []
    for raw in raw_cubes:
        levels: list[Level] = []
        for dim in raw.get("dimensions", []) or []:
            dim_name = dim.get("name", "")
            dim_type = str(dim.get("type") or dim.get("dimension_type") or "standard").lower()
            for hie in dim.get("hierarchies", []) or []:
                for depth, lvl in enumerate(hie.get("levels", []) or [], start=1):
                    if not lvl.get("name"):
                        continue
                    levels.append(
                        Level(
                            name=lvl["name"],
                            dimension=dim_name,
                            hierarchy=hie.get("name", ""),
                            depth=int(lvl.get("depth") or depth),
                            dim_type=dim_type,
                            unique_name=lvl.get("unique_name") or None,
                        )
                    )
        measures = [m["name"] for m in raw.get("measures", []) or [] if m.get("name")]
        annotations = raw.get("annotations") or {}
        cubes.append(Cube(raw["name"], levels, measures, annotations if isinstance(annotations, dict) else {}))
    return cubes


def hs_digits(level: Level) -> int | None:
    """Return the number of HS digits a level represents, if any."""
    for text in (level.name, level.unique_name or ""):
        match = _HS_DIGITS.search(text)
        if match:
            digits = int(match.group(1) or match.group(2))
            if digits in (2, 4, 6, 8, 10):
                return digits
        if _TARIFF.search(text):
            return 8
    return None


def as_trade_cube(cube: Cube) -> TradeCube | None:
    """Return a :class:`TradeCube` if ``cube`` has HS, flow and year levels."""
    hs_levels: dict[int, Level] = {}
    for level in cube.levels:
        digits = hs_digits(level)
        if digits and digits not in hs_levels:
            hs_levels[digits] = level
    year = cube.find(_YEAR, time_only=True) or cube.find(_YEAR)
    flow = cube.find(_FLOW, time_only=False)
    if not hs_levels or year is None or flow is None or not cube.measures:
        return None

    value_measure = next((m for m in cube.measures if _VALUE_MEASURE.search(m)), cube.measures[0])
    country = cube.find(_COUNTRY, time_only=False)
    state = cube.find(_STATE, time_only=False)
    period = cube.find(_PERIOD, time_only=True)

    score = 0.0
    score += 5 if _TRADE_CUBE.search(cube.name) else 0
    score += 3 if 6 in hs_levels else 0
    score += 1 if 4 in hs_levels else 0
    score += 2 if country else 0
    score += 1 if state else 0
    score += 1 if period else 0
    score += 1 if _VALUE_MEASURE.search(value_measure) else 0
    # Municipal cubes cover less of total trade; legacy/test copies are stale.
    score -= 2 if re.search(r"_mun\b|municip", cube.name, re.I) else 0
    score -= 3 if re.search(r"legacy|test|old|deprecated", cube.name, re.I) else 0

    return TradeCube(
        cube=cube,
        year=year,
        flow=flow,
        hs_levels=dict(sorted(hs_levels.items())),
        country=country,
        state=state,
        period=period,
        value_measure=value_measure,
        score=score,
    )


def find_trade_cubes(raw_cubes: list[dict]) -> list[TradeCube]:
    """Return every usable trade cube, best candidate first."""
    found = [tc for tc in (as_trade_cube(c) for c in parse_cubes(raw_cubes)) if tc]
    return sorted(found, key=lambda tc: (-tc.score, tc.name))
