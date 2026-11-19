"""Plots built from the precomputed results only."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bioinfo_agent import queries
from bioinfo_agent.config import N_PCS

CONDITION_COLORS = {"control": "#1f77b4", "treated": "#d95f02"}
LANE_COLORS = {"L1": "#2a9d8f", "L2": "#7b2cbf"}
# One marker per condition-lane group, so the three groups stay identifiable in both panels.
GROUP_SYMBOLS = {
    "control-L1": "circle",
    "treated-L1": "square",
    "treated-L2": "diamond",
}


def check_components(x: int, y: int) -> None:
    if not (1 <= x <= N_PCS and 1 <= y <= N_PCS) or x == y:
        raise ValueError(f"x and y must be two different components from 1 to {N_PCS}")


def pca_figure(x: int = 1, y: int = 2, subset: str = "all") -> go.Figure:
    """Sample PCA. 'all': twice, coloured by condition (left) and by lane (right).
    'treated': treated samples only, coloured by lane."""
    check_components(x, y)
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


VOLCANO_MODELS = {
    "headline": ("log2fc", "padj", "direction", "treated vs control, ~ condition"),
    "lane_adjusted": (
        "log2fc_lane_adj",
        "padj_lane_adj",
        "direction_lane_adj",
        "treated vs control, ~ lane + condition",
    ),
}
DIRECTION_STYLE = {
    "up": ("Up in treated", "#d95f02"),
    "down": ("Down in treated", "#1f77b4"),
    "ns": ("Not significant", "#bbbbbb"),
}


def volcano_figure(
    model: str = "headline", highlight: list[str] | tuple[str, ...] = ()
) -> go.Figure:
    """Volcano plot of every tested gene: shrunken log2FC against -log10(adjusted p-value).

    model: 'headline' (~ condition) or 'lane_adjusted' (~ lane + condition). highlight: gene
    symbols to mark and label (case-insensitive; unknown symbols are ignored).
    """
    if model not in VOLCANO_MODELS:
        raise ValueError(f"model must be one of {tuple(VOLCANO_MODELS)}")
    fc_col, padj_col, dir_col, description = VOLCANO_MODELS[model]
    table, summary = queries.load_results()
    thresholds = summary["thresholds"]
    padj = table[padj_col].fillna(1.0)
    # padj underflows to 0 for the strongest genes: draw them just above the highest finite value.
    finite = -np.log10(padj[padj > 0])
    ceiling = float(finite.max()) * 1.1 if len(finite) else 1.0
    neg_log_padj = pd.Series(
        np.where(padj > 0, -np.log10(padj.clip(lower=1e-300)), ceiling),
        index=table.index,
    )
    capped = int((padj == 0).sum())

    fig = go.Figure()
    for direction in ("ns", "down", "up"):
        name, color = DIRECTION_STYLE[direction]
        sel = table[dir_col] == direction
        fig.add_trace(
            go.Scattergl(
                x=table.loc[sel, fc_col],
                y=neg_log_padj[sel],
                mode="markers",
                name=f"{name} ({int(sel.sum())})",
                text=table.index[sel],
                customdata=np.column_stack([padj[sel], table.loc[sel, "base_mean"]]),
                hovertemplate="<b>%{text}</b><br>log2FC=%{x:.2f}<br>padj=%{customdata[0]:.2g}"
                "<br>mean count=%{customdata[1]:.0f}<extra></extra>",
                marker={
                    "size": 4 if direction == "ns" else 6,
                    "color": color,
                    "opacity": 0.5 if direction == "ns" else 0.85,
                },
            )
        )
    upper = pd.Series(table.index, index=table.index.str.upper())
    wanted = dict.fromkeys(h.upper() for h in highlight)
    marked = [upper[g] for g in wanted if g in upper.index]
    if marked:
        fig.add_trace(
            go.Scatter(
                x=table.loc[marked, fc_col],
                y=neg_log_padj[marked],
                mode="markers+text",
                name="Selected",
                text=marked,
                textposition="top center",
                hoverinfo="skip",
                marker={
                    "size": 11,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": "black"},
                },
            )
        )
    fig.add_hline(
        y=-np.log10(thresholds["padj"]),
        line={"dash": "dash", "color": "#888", "width": 1},
    )
    for x in (-thresholds["abs_log2fc"], thresholds["abs_log2fc"]):
        fig.add_vline(x=x, line={"dash": "dash", "color": "#888", "width": 1})
    if capped:
        fig.add_annotation(
            text=f"{capped} genes with padj = 0 are drawn at the top",
            xref="paper",
            yref="paper",
            x=1,
            y=1.02,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font={"size": 11, "color": "#666"},
        )
    fig.update_layout(
        title=f"Volcano plot ({description})",
        xaxis_title="log2 fold change (shrunken, treated vs control)",
        yaxis_title="-log10(adjusted p-value)",
        template="plotly_white",
        height=520,
        margin={"t": 70, "b": 50, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.18},
    )
    return fig
