"""Search HS (Harmonized System) product codes by material name or code.

Data México product IDs carry the HS section as a numeric prefix (e.g. the HS6
code 290230 "Toluene" may be stored as 6290230). :func:`hs_code` strips that
prefix so users always see the standard HS code, while the raw ID is kept for
API queries.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

from .client import Member

# Common petrochemical names -> HS6 codes. Trade labels often use IUPAC or
# customs wording ("Propene (propylene)", "Ethanediol"), so plain keyword
# search can miss them. These codes only *add* candidates to the results;
# the description shown is always the one returned by Data México.
CURATED_HS6: dict[str, tuple[str, ...]] = {
    "ethylene": ("290121", "271114"),
    "ethene": ("290121",),
    "propylene": ("290122", "271114"),
    "propene": ("290122",),
    "butadiene": ("290124", "271114"),
    "butylene": ("290123", "271114"),
    "butene": ("290123",),
    "ethane": ("271119", "271129", "290110"),
    "propane": ("271112",),
    "butane": ("271113",),
    "hexane": ("290110",),
    "heptane": ("290110",),
    "cyclohexane": ("290211",),
    "benzene": ("290220",),
    "toluene": ("290230",),
    "xylene": ("290241", "290242", "290243", "290244"),
    "orthoxylene": ("290241",),
    "o-xylene": ("290241",),
    "paraxylene": ("290243",),
    "p-xylene": ("290243",),
    "styrene": ("290250",),
    "cumene": ("290270",),
    "vinyl chloride": ("290321",),
    "vcm": ("290321",),
    "ethylene dichloride": ("290315",),
    "edc": ("290315",),
    "methanol": ("290511",),
    "isopropanol": ("290512",),
    "isopropyl alcohol": ("290512",),
    "butanol": ("290513",),
    "2-ethylhexanol": ("290516",),
    "ethylene glycol": ("290531",),
    "monoethylene glycol": ("290531",),
    "meg": ("290531",),
    "propylene glycol": ("290532",),
    "diethylene glycol": ("290941",),
    "glycerol": ("290545",),
    "glycerine": ("290545",),
    "mtbe": ("290919",),
    "phenol": ("290711",),
    "ethylene oxide": ("291010",),
    "propylene oxide": ("291020",),
    "epichlorohydrin": ("291030",),
    "acetone": ("291411",),
    "acetic acid": ("291521",),
    "acetic anhydride": ("291524",),
    "ethyl acetate": ("291531",),
    "vinyl acetate": ("291532",),
    "acrylic acid": ("291611",),
    "adipic acid": ("291712",),
    "maleic anhydride": ("291714",),
    "phthalic anhydride": ("291735",),
    "terephthalic acid": ("291736",),
    "pta": ("291736",),
    "ethanolamine": ("292211", "292212", "292215"),
    "ethanolamines": ("292211", "292212", "292215"),
    "monoethanolamine": ("292211",),
    "diethanolamine": ("292212",),
    "triethanolamine": ("292215",),
    "acrylonitrile": ("292610",),
    "caprolactam": ("293371",),
    "ammonia": ("281410", "281420"),
    "carbon black": ("280310",),
    "polypropylene": ("390210", "390230"),
    "polyethylene": ("390110", "390120", "390140"),
    "pvc": ("390410", "390421", "390422"),
    "polyvinyl chloride": ("390410", "390421", "390422"),
    "polystyrene": ("390311", "390319"),
    "pet": ("390761", "390769"),
}


@dataclass
class HSEntry:
    """One HS product as shown in the search table."""

    digits: int
    member_id: str
    code: str
    label: str
    parent_label: str = ""
    match: str = ""


def hs_code(member_id: str | int, digits: int) -> str:
    """Return the standard ``digits``-long HS code for a Data México member ID.

    >>> hs_code("6290230", 6)
    '290230'
    >>> hs_code("10101", 4)   # section 1 prefix + heading 0101
    '0101'
    >>> hs_code("10111", 6)   # un-prefixed HS6 010111 with its zero dropped
    '010111'
    """
    only_digits = re.sub(r"\D", "", str(member_id))
    if not only_digits:
        return str(member_id)
    if len(only_digits) >= digits:
        return only_digits[-digits:]
    return only_digits.zfill(digits)


def format_code(code: str) -> str:
    """Pretty-print an HS code as 2902.30 style."""
    if len(code) <= 4:
        return code
    return f"{code[:4]}.{code[4:]}"


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation (keeps digits and hyphens)."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2] if token[:-2].endswith(("x", "ch", "sh")) else token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def build_entries(
    members_by_digits: Mapping[int, Iterable[Member]],
    alt_labels: Mapping[int, Mapping[str, str]] | None = None,
) -> dict[int, list[HSEntry]]:
    """Turn raw members into :class:`HSEntry` lists keyed by HS digit count.

    ``alt_labels`` optionally maps digits -> {member_id: label in another
    locale}; those labels are appended so Spanish or English queries both work.
    """
    entries: dict[int, list[HSEntry]] = {}
    for digits, members in sorted(members_by_digits.items()):
        extra = (alt_labels or {}).get(digits, {})
        entries[digits] = [
            HSEntry(
                digits=digits,
                member_id=m.id,
                code=hs_code(m.id, digits),
                label=m.label,
                match=normalize(m.label + " " + extra.get(m.id, "")),
            )
            for m in members
        ]

    # Attach the parent heading's label so a user sees the context of each code.
    by_code = {e.code: e.label for level in entries.values() for e in level}
    for digits, level in entries.items():
        for entry in level:
            for parent_digits in (d for d in sorted(entries, reverse=True) if d < digits):
                parent = by_code.get(entry.code[:parent_digits])
                if parent:
                    entry.parent_label = parent
                    break
    return entries


def search(
    entries: Mapping[int, list[HSEntry]],
    query: str,
    levels: Iterable[int] | None = None,
    use_curated: bool = True,
    limit: int = 300,
) -> list[HSEntry]:
    """Find HS entries matching a material name or an HS code prefix.

    * Numeric queries ("2902", "2902.30") match codes starting with those digits.
    * Text queries match entries whose description contains every word
      (plural-insensitive, accent-insensitive, English or Spanish label).
    * Known petrochemical names also pull in curated HS6 codes.
    """
    levels = set(levels) if levels is not None else set(entries)
    raw = query.strip()
    if not raw:
        return []

    results: dict[tuple[int, str], tuple[int, HSEntry]] = {}

    def add(entry: HSEntry, rank: int) -> None:
        key = (entry.digits, entry.member_id)
        if key not in results or rank < results[key][0]:
            results[key] = (rank, entry)

    digits_only = re.sub(r"[\s.\-]", "", raw)
    if digits_only.isdigit():
        for d, level in entries.items():
            if d not in levels:
                continue
            for entry in level:
                if entry.code.startswith(digits_only):
                    add(entry, 0 if entry.code == digits_only else 1)
    else:
        q = normalize(raw)
        tokens = [_stem(t) for t in q.split() if t]
        phrase = re.compile(rf"\b{re.escape(q)}", re.I)

        if use_curated:
            curated_codes = set()
            for name, codes in CURATED_HS6.items():
                if q == name or _stem(q) == _stem(name):
                    curated_codes.update(codes)
            for d, level in entries.items():
                if d not in levels:
                    continue
                for entry in level:
                    if entry.code in curated_codes:
                        add(entry, 0)

        for d, level in entries.items():
            if d not in levels:
                continue
            for entry in level:
                if all(t in entry.match for t in tokens):
                    add(entry, 1 if phrase.search(entry.match) else 2)

    ranked = sorted(results.values(), key=lambda item: (item[0], item[1].code, item[1].digits))
    return [entry for _, entry in ranked[:limit]]


def overlapping(selected: Iterable[HSEntry]) -> list[tuple[HSEntry, HSEntry]]:
    """Return (parent, child) pairs where one selected code contains another.

    Selecting both would double count trade in the totals.
    """
    items = list(selected)
    pairs = []
    for parent in items:
        for child in items:
            if child.digits > parent.digits and child.code.startswith(parent.code):
                pairs.append((parent, child))
    return pairs
