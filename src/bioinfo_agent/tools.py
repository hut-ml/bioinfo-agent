"""The tools: the only way the agent reads the precomputed results. The LLM calls them through
`toolset`; offline mode calls the same functions directly. Docstrings are the LLM-facing descriptions."""

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from bioinfo_agent import queries
from bioinfo_agent.plots import VOLCANO_MODELS, check_components

MAX_GENES = 50
PLOT_TOOLS = ("show_pca", "show_volcano")
TABLE_TOOLS = ("list_de_genes", "get_genes", "list_lane_sensitive_genes")
TABLE_COLUMNS = (
    "gene",
    "direction",
    "log2fc",
    "padj",
    "base_mean",
    "lane_flag",
    "log2fc_lane_adjusted",
    "padj_lane_adjusted",
)
_TABLE_TITLES = {
    "list_de_genes": "Differentially expressed genes",
    "get_genes": "Gene lookup",
    "list_lane_sensitive_genes": "Lane-sensitive genes",
}

toolset = FunctionToolset()


@toolset.tool_plain
def get_overview() -> dict:
    """Analysis design, comparison, significance thresholds, group sizes and the number of
    up and down regulated genes."""
    return queries.overview()


@toolset.tool_plain
def list_de_genes(
    direction: str = "all", n: int = 10, sort_by: str = "padj"
) -> list[dict]:
    """List significant DE genes. direction: 'all', 'up' or 'down' (in treated).
    sort_by: 'padj' (most significant first), 'log2fc' (largest |fold change|) or
    'base_mean' (most highly expressed). n is capped at 50."""
    try:
        return queries.top_genes(direction, min(max(n, 1), MAX_GENES), sort_by)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc


@toolset.tool_plain
def get_genes(names: list[str]) -> list[dict]:
    """Look up the DE result of specific genes by symbol (case-insensitive). Unknown symbols
    come back with found=false and close-match suggestions."""
    return queries.get_genes(names[:MAX_GENES])


@toolset.tool_plain
def list_lane_sensitive_genes(flag: str = "all", n: int = 10) -> list[dict]:
    """List genes whose differential expression call changes when sequencing lane is added to
    the model (lane is partly confounded with condition). flag: 'lost' (DE without lane, not DE
    with it: possible lane artefacts), 'gained' (the opposite: lane may mask a real effect),
    'flipped' (reverses direction) or 'all'. n is capped at 50."""
    try:
        return queries.lane_sensitive_genes(flag, min(max(n, 1), MAX_GENES))
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc


@toolset.tool_plain
def get_lane_check() -> dict:
    """Is sequencing lane confounded with condition? Group sizes (every L2 sample is treated),
    DE gene counts with and without the lane term, how many calls change, and what the PCA shows
    (variance explained, group means on each PC, p-values for the condition and lane effects, and
    the same for a PCA of the treated samples only)."""
    return queries.lane_check()


@toolset.tool_plain
def get_l2_concordance() -> dict:
    """Independent directional check of the lane-adjusted DE genes. Those genes are estimated from
    L1 samples only (treated-L1 vs control-L1). This reports how many move in the same direction in
    treated-L2 vs control-L1, overall, for up and down genes, and for genes also significant in L2,
    next to a baseline of genes that are not DE (the lane shift alone pushes that above 50%).
    Control-L1 samples are shared, so it is supporting evidence, not a fully independent replicate."""
    return queries.l2_concordance()


@toolset.tool_plain
def show_pca(x: int = 1, y: int = 2, subset: str = "all") -> dict:
    """Show a PCA plot of the samples to the user. subset 'all': all 90 samples, drawn twice
    (left coloured by condition, right by lane). subset 'treated': the 50 treated samples only,
    coloured by lane, which removes the treatment effect so any L1/L2 separation is a lane effect.
    x and y are principal components 1 to 5. The plot appears in the side panel; use get_lane_check
    to read the numbers behind it."""
    try:
        check_components(x, y)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
    if subset not in queries.PCA_SUBSETS:
        raise ModelRetry(f"subset must be one of {queries.PCA_SUBSETS}")
    return {"plot": "pca", "x": x, "y": y, "subset": subset, "shown": True}


@toolset.tool_plain
def show_volcano(model: str = "headline", genes: list[str] | None = None) -> dict:
    """Show a volcano plot of every tested gene (log2 fold change against -log10 adjusted p-value,
    coloured up / down / not significant) to the user, to see which genes are up or down regulated.
    model: 'headline' (~ condition, the default) or 'lane_adjusted' (~ lane + condition).
    genes: optional gene symbols to mark and label on the plot. The plot appears in the side panel;
    use list_de_genes or get_genes to read the numbers behind it."""
    if model not in VOLCANO_MODELS:
        raise ModelRetry(f"model must be one of {tuple(VOLCANO_MODELS)}")
    found = [
        g["gene"] for g in queries.get_genes((genes or [])[:MAX_GENES]) if g["found"]
    ]
    return {"plot": "volcano", "model": model, "genes": found, "shown": True}


def format_call(name: str, args: dict) -> str:
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"


def summarise_result(name: str, result) -> str:
    """One line describing a tool result."""
    if name == "get_overview":
        return f"{result['design']}; {result['n_up']} up, {result['n_down']} down"
    if name == "list_de_genes":
        if not result:
            return "no genes"
        names = ", ".join(g["gene"] for g in result[:5])
        return f"{len(result)} genes: {names}{', ...' if len(result) > 5 else ''}"
    if name == "get_genes":
        return "; ".join(
            f"{g['gene']}: {g['direction']}"
            + (f" (lane: {g['lane_flag']})" if g["lane_flag"] != "same" else "")
            if g["found"]
            else f"{g['query']}: not found"
            for g in result
        )
    if name == "list_lane_sensitive_genes":
        if not result:
            return "no genes"
        names = ", ".join(f"{g['gene']} ({g['lane_flag']})" for g in result[:5])
        return f"{len(result)} genes: {names}{', ...' if len(result) > 5 else ''}"
    if name == "get_lane_check":
        flags = result["flags"]
        return (
            f"{flags.get('lost', 0)} lost, {flags.get('gained', 0)} gained, "
            f"{flags.get('flipped', 0)} flipped when lane is added"
        )
    if name == "get_l2_concordance":
        de, base = result["lane_adjusted_de"], result["baseline_not_de"]
        return (
            f"{de['n_same_direction']}/{de['n_genes']} lane-adjusted DE genes agree in L2 "
            f"({de['fraction']:.1%}); baseline {base['fraction']:.1%}"
        )
    if name == "show_pca":
        what = (
            "treated samples only" if result["subset"] == "treated" else "all samples"
        )
        return f"PCA plot ({what}), PC{result['x']} vs PC{result['y']}"
    if name == "show_volcano":
        marked = f", marking {', '.join(result['genes'])}" if result["genes"] else ""
        return f"volcano plot ({result['model'].replace('_', '-')} model){marked}"
    return str(result)[:100]


def result_table(name: str, args: dict, result: list[dict]) -> dict | None:
    """A table for the side panel from a gene-list tool result, or None if there is nothing to show."""
    if name not in TABLE_TOOLS:
        return None
    rows = [
        {col: g[col] for col in TABLE_COLUMNS} for g in result if g.get("found", True)
    ]
    if not rows:
        return None
    detail = ", ".join(f"{k}={v}" for k, v in args.items() if k != "names")
    title = _TABLE_TITLES[name] + (f" ({detail})" if detail else "")
    return {"title": title, "rows": rows}
