"""The tools: the only way the agent reads the precomputed results. The LLM calls them through
`toolset`; offline mode calls the same functions directly. Docstrings are the LLM-facing descriptions."""

from pydantic_ai import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from bioinfo_agent import queries

MAX_GENES = 50

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
            if g["found"]
            else f"{g['query']}: not found"
            for g in result
        )
    return str(result)[:100]
