"""Plotly figures for the trade explorer.

Conventions: one y-axis per chart (never dual axis), categorical hues assigned
in a fixed order and kept per entity, thin bars with rounded ends, hairline
grid, partial years hatched and labelled "YTD".
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (fixed order), light and dark steps.
PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
OTHER = {"light": "#a3a29c", "dark": "#6f6e69"}
GRID = {"light": "#e7e6e2", "dark": "#34342f"}
# Streamlit page background; used as the 2px gap between stacked segments.
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}
MAX_SERIES = 7  # the 8th slot is reserved for "Other"


def year_labels(years: Iterable[int], partial: Iterable[int]) -> list[str]:
    partial = set(partial)
    return [f"{y} (YTD)" if y in partial else str(y) for y in years]


def _layout(fig: go.Figure, mode: str, title: str, unit: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        barmode=fig.layout.barmode or "group",
        bargap=0.35,
        bargroupgap=0.08,
        barcornerradius=4,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1, "title": None},
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(type="category", showgrid=False, linecolor=GRID[mode], ticks="")
    fig.update_yaxes(title=unit or None, gridcolor=GRID[mode], gridwidth=1, zeroline=True,
                     zerolinecolor=GRID[mode], tickformat="~s")
    return fig


def _patterns(labels: Sequence[str]) -> list[str]:
    return ["/" if label.endswith("(YTD)") else "" for label in labels]


def trade_trend(summary: pd.DataFrame, partial: Iterable[int], unit: str, mode: str = "light") -> go.Figure:
    """Imports vs exports per year (grouped columns)."""
    colors = PALETTE[mode]
    x = year_labels(summary.index, partial)
    fig = go.Figure()
    for col, name, color in (("imports", "Imports", colors[0]), ("exports", "Exports", colors[1])):
        fig.add_bar(
            x=x, y=summary[col], name=name, marker={"color": color, "pattern": {"shape": _patterns(x), "solidity": 0.35}},
            hovertemplate=f"{name}: %{{y:,.0f}} {unit}<extra></extra>",
        )
    return _layout(fig, mode, "Imports and exports by year", unit)


def net_imports(summary: pd.DataFrame, partial: Iterable[int], unit: str, mode: str = "light") -> go.Figure:
    """Net imports (imports - exports); positive = Mexico buys more than it sells."""
    x = year_labels(summary.index, partial)
    fig = go.Figure()
    fig.add_scatter(
        x=x, y=summary["net_imports"], mode="lines+markers", name="Net imports",
        line={"color": PALETTE[mode][0], "width": 2}, marker={"size": 8},
        hovertemplate=f"Net imports: %{{y:,.0f}} {unit}<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    fig = _layout(fig, mode, "Net imports (imports − exports)", unit, height=300)
    fig.update_yaxes(rangemode="tozero")
    return fig


def stacked_by(
    df: pd.DataFrame,
    key: str,
    flow: str,
    years: Sequence[int],
    partial: Iterable[int],
    unit: str,
    title: str,
    mode: str = "light",
    order: Sequence[str] | None = None,
) -> go.Figure:
    """Stacked columns of ``flow`` by year, split by ``key`` (top 7 + Other).

    Colours follow the entity: the order is fixed by total value over the
    whole period (or ``order``), so it does not change when years change.
    """
    sub = df[df["flow"] == flow]
    pivot = sub.pivot_table(index="year", columns=key, values="value", aggfunc="sum").reindex(years).fillna(0.0)
    ranked = list(order) if order else list(pivot.sum().sort_values(ascending=False).index)
    keep = ranked[:MAX_SERIES]
    rest = [c for c in pivot.columns if c not in keep]
    x = year_labels(years, partial)
    fig = go.Figure()
    for i, name in enumerate(keep):
        if name not in pivot.columns:
            continue
        fig.add_bar(
            x=x, y=pivot[name], name=str(name),
            marker={"color": PALETTE[mode][i],
                    "pattern": {"shape": _patterns(x), "solidity": 0.35}},
            hovertemplate=f"{name}: %{{y:,.0f}} {unit}<extra></extra>",
        )
    if rest:
        fig.add_bar(
            x=x, y=pivot[rest].sum(axis=1), name=f"Other ({len(rest)})",
            marker={"color": OTHER[mode], "pattern": {"shape": _patterns(x), "solidity": 0.35}},
            hovertemplate=f"Other: %{{y:,.0f}} {unit}<extra></extra>",
        )
    fig.update_layout(barmode="stack", bargap=0.35, legend_traceorder="normal")
    fig = _layout(fig, mode, title, unit)
    fig.update_traces(marker_line={"width": 2, "color": SURFACE[mode]})
    return fig


def ranking_bar(
    table: pd.DataFrame, label_col: str, value_col: str, unit: str, title: str,
    mode: str = "light", share_col: str | None = None, top: int = 15,
) -> go.Figure:
    """Horizontal bars, one colour (identity is the row label, not hue)."""
    data = table.sort_values(value_col, ascending=False).head(top).iloc[::-1]
    text = None
    if share_col:
        text = [f"{s:.0%}" for s in data[share_col]]
    fig = go.Figure(go.Bar(
        x=data[value_col], y=data[label_col], orientation="h", text=text, textposition="outside",
        marker={"color": PALETTE[mode][0]}, cliponaxis=False,
        hovertemplate=f"%{{y}}: %{{x:,.0f}} {unit}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=max(260, 34 * len(data) + 90), margin={"l": 10, "r": 40, "t": 50, "b": 10},
        barcornerradius=4, bargap=0.35, showlegend=False,
    )
    fig.update_xaxes(title=unit or None, gridcolor=GRID[mode], tickformat="~s")
    fig.update_yaxes(type="category", showgrid=False, ticks="")
    return fig


def consumption_chart(summary: pd.DataFrame, partial: Iterable[int], unit: str, mode: str = "light") -> go.Figure:
    """Apparent consumption vs production vs imports (same unit, one axis)."""
    colors = PALETTE[mode]
    x = year_labels(summary.index, partial)
    fig = go.Figure()
    for col, name, color in (("apparent_consumption", "Apparent consumption", colors[0]),
                             ("production", "Domestic production", colors[2]),
                             ("imports", "Imports", colors[1])):
        fig.add_scatter(x=x, y=summary[col], name=name, mode="lines+markers",
                        line={"color": color, "width": 2}, marker={"size": 8},
                        hovertemplate=f"{name}: %{{y:,.0f}} {unit}<extra></extra>")
    return _layout(fig, mode, "Apparent consumption = production + imports − exports", unit, height=340)


def dependence_chart(summary: pd.DataFrame, partial: Iterable[int], threshold: float, mode: str = "light") -> go.Figure:
    x = year_labels(summary.index, partial)
    fig = go.Figure()
    fig.add_scatter(x=x, y=summary["import_dependence"], name="Import dependence", mode="lines+markers",
                    line={"color": PALETTE[mode][0], "width": 2}, marker={"size": 8},
                    hovertemplate="Import dependence: %{y:.0%}<extra></extra>")
    fig.add_hline(y=threshold, line={"color": GRID[mode] if mode == "dark" else "#a3a29c", "width": 1},
                  annotation_text=f"threshold {threshold:.0%}", annotation_position="top left")
    fig.update_layout(showlegend=False)
    fig = _layout(fig, mode, "Import dependence (imports ÷ apparent consumption)", "", height=300)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig
