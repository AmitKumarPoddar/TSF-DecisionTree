import pandas as pd
import pytest

from datamexico import charts
from datamexico.analysis import EXPORTS, IMPORTS, yearly_summary


@pytest.fixture
def trade():
    rows = []
    for year in range(2019, 2025):
        for i, code in enumerate("ABCDEFGHIJ"):
            rows.append({"year": year, "flow": IMPORTS, "product": code, "value": 10.0 * (i + 1)})
            rows.append({"year": year, "flow": EXPORTS, "product": code, "value": 1.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_all_figures_build(trade, mode):
    summary = yearly_summary(trade, production={y: 500.0 for y in range(2019, 2025)})
    partial = {2024}
    figs = [
        charts.trade_trend(summary, partial, "USD", mode),
        charts.net_imports(summary, set(), "USD", mode),
        charts.consumption_chart(summary, partial, "USD", mode),
        charts.dependence_chart(summary, partial, 0.3, mode),
    ]
    for fig in figs:
        assert fig.layout.yaxis.overlaying is None  # never a second y-axis
    assert list(figs[0].data[0].x)[-1] == "2024 (YTD)"


def test_stacked_folds_tail_into_other_and_keeps_entity_colours(trade):
    fig = charts.stacked_by(trade, "product", IMPORTS, list(range(2019, 2025)), set(), "USD", "t")
    names = [t.name for t in fig.data]
    assert len(names) == charts.MAX_SERIES + 1 and names[-1] == "Other (3)"
    # Colour follows the entity given a fixed order, not its rank in a subset.
    order = list("JIHGFEDCBA")
    fig_a = charts.stacked_by(trade, "product", IMPORTS, [2019], set(), "USD", "t", order=order)
    fig_b = charts.stacked_by(trade[trade["product"] != "J"], "product", IMPORTS, [2019], set(), "USD", "t", order=order)
    colour_a = {t.name: t.marker.color for t in fig_a.data}
    colour_b = {t.name: t.marker.color for t in fig_b.data}
    assert colour_a["I"] == colour_b["I"]


def test_ranking_bar_orders_largest_on_top():
    table = pd.DataFrame({"state": ["a", "b", "c"], "value": [1, 3, 2], "share": [1 / 6, 1 / 2, 1 / 3]})
    fig = charts.ranking_bar(table, "state", "value", "USD", "t", share_col="share")
    assert list(fig.data[0].y) == ["a", "c", "b"]  # plotted bottom-up
    assert list(fig.data[0].text) == ["17%", "33%", "50%"]
