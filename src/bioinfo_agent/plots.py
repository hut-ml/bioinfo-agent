"""Plots built from the precomputed results only."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bioinfo_agent import queries
from bioinfo_agent.analysis import N_PCS

CONDITION_COLORS = {"control": "#1f77b4", "treated": "#d95f02"}
LANE_COLORS = {"L1": "#2a9d8f", "L2": "#7b2cbf"}
# One marker per condition-lane group, so the three groups stay identifiable in both panels.
GROUP_SYMBOLS = {
    "control-L1": "circle",
    "treated-L1": "square",
    "treated-L2": "diamond",
}


def _check_components(x: int, y: int) -> None:
    if not (1 <= x <= N_PCS and 1 <= y <= N_PCS) or x == y:
        raise ValueError(f"x and y must be two different components from 1 to {N_PCS}")


def pca_figure(x: int = 1, y: int = 2, subset: str = "all") -> go.Figure:
    """Sample PCA. 'all': twice, coloured by condition (left) and by lane (right).
    'treated': treated samples only, coloured by lane."""
    _check_components(x, y)
    scores = queries.load_pca(subset)
    _, summary = queries.load_results()
    pca = summary["lane_check"]["pca"]
    explained = (
        pca["treated_only"]["variance_explained"]
        if subset == "treated"
        else pca["variance_explained"]
    )
    x_name, y_name = f"PC{x}", f"PC{y}"
    if subset == "treated":
        panels = (("lane", LANE_COLORS),)
        titles = ("Treated samples only, coloured by lane",)
        groups = {g: v for g, v in GROUP_SYMBOLS.items() if g.startswith("treated")}
    else:
        panels = (("condition", CONDITION_COLORS), ("lane", LANE_COLORS))
        titles = ("Coloured by condition", "Coloured by lane")
        groups = GROUP_SYMBOLS

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=titles,
        horizontal_spacing=0.04,
    )
    for col, (field, colors) in enumerate(panels, start=1):
        for group, symbol in groups.items():
            condition, lane = group.split("-")
            points = scores[
                (scores["condition"] == condition) & (scores["lane"] == lane)
            ]
            fig.add_trace(
                go.Scatter(
                    x=points[x_name],
                    y=points[y_name],
                    mode="markers",
                    name=group,
                    legendgroup=group,
                    showlegend=col == 1,
                    text=points.index,
                    hovertemplate="%{text}<br>"
                    + f"{x_name}=%{{x:.1f}}, {y_name}=%{{y:.1f}}",
                    marker={
                        "symbol": symbol,
                        "size": 9,
                        "color": colors[condition if field == "condition" else lane],
                        "line": {"width": 0.5, "color": "white"},
                    },
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text=f"{x_name} ({explained[x - 1]:.0%})", row=1, col=col
        )
    fig.update_yaxes(title_text=f"{y_name} ({explained[y - 1]:.0%})", row=1, col=1)
    fig.update_layout(
        title=(
            f"Sample PCA ({pca['n_variable_genes']} most variable genes, VST counts"
            f"{', treated only' if subset == 'treated' else ''})"
        ),
        template="plotly_white",
        height=420,
        margin={"t": 90, "b": 50, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig
