"""Read the precomputed results. Every function returns plain, JSON-friendly data."""

import difflib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bioinfo_agent.analysis import (
    LOG2FC_THRESHOLD,
    PADJ_THRESHOLD,
    RESULTS_FILE,
    SUMMARY_FILE,
    results_dir,
)

SORT_KEYS = ("padj", "log2fc", "base_mean")


class ResultsNotFound(FileNotFoundError):
    pass


@lru_cache(maxsize=4)
def _load(directory: str) -> tuple[pd.DataFrame, dict]:
    path = Path(directory)
    try:
        table = pd.read_csv(path / RESULTS_FILE, index_col="gene")
        summary = json.loads((path / SUMMARY_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultsNotFound(
            f"No results in {path}. Run `bioinfo-agent precompute` first."
        ) from exc
    return table, summary


def load_results(directory: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    return _load(str(directory or results_dir()))


def _num(value, digits: int = 4):
    return None if pd.isna(value) else float(f"{value:.{digits}g}")


def overview(directory: Path | str | None = None) -> dict:
    """Design, thresholds, group sizes and gene counts."""
    _, summary = load_results(directory)
    return {k: v for k, v in summary.items() if k not in ("size_factors", "versions")}


def top_genes(
    direction: str = "all",
    n: int = 10,
    sort_by: str = "padj",
    directory: Path | str | None = None,
) -> list[dict]:
    """Significant DE genes (padj and |log2FC| pass the thresholds)."""
    if direction not in ("all", "up", "down"):
        raise ValueError("direction must be 'all', 'up' or 'down'")
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}")
    table, _ = load_results(directory)
    hits = table[table["direction"] != "ns"]
    if direction != "all":
        hits = hits[hits["direction"] == direction]
    if sort_by == "log2fc":
        hits = hits.reindex(hits["log2fc"].abs().sort_values(ascending=False).index)
    elif sort_by == "padj":
        # padj underflows to 0 for the strongest genes, so break ties on |stat|.
        hits = hits.assign(_abs_stat=hits["stat"].abs()).sort_values(
            ["padj", "_abs_stat"], ascending=[True, False]
        )
    else:
        hits = hits.sort_values(sort_by, ascending=False)
    return [_record(gene, row) for gene, row in hits.head(max(n, 0)).iterrows()]


def get_genes(names: list[str], directory: Path | str | None = None) -> list[dict]:
    """Look up genes case-insensitively; unknown names get 'did you mean' suggestions."""
    table, _ = load_results(directory)
    by_upper = {g.upper(): g for g in table.index}
    out = []
    for name in names:
        gene = by_upper.get(name.strip().upper())
        if gene is None:
            close = difflib.get_close_matches(name.strip().upper(), by_upper, n=3)
            out.append(
                {
                    "query": name,
                    "found": False,
                    "did_you_mean": [by_upper[c] for c in close],
                    "verdict": (
                        f"{name} is not in the results (not in the dataset, or removed by "
                        "the low-count filter)."
                    ),
                }
            )
        else:
            out.append({"query": name, "found": True, **_record(gene, table.loc[gene])})
    return out


def _record(gene: str, row: pd.Series) -> dict:
    return {
        "gene": gene,
        "log2fc": _num(row["log2fc"]),
        "padj": _num(row["padj"]),
        "base_mean": _num(row["base_mean"]),
        "direction": row["direction"],
        "verdict": verdict(row),
    }


def verdict(row: pd.Series) -> str:
    padj, lfc = row["padj"], row["log2fc"]
    if pd.isna(padj):
        text = "no adjusted p-value (flagged as an outlier or too low counts to test)"
    elif row["direction"] != "ns":
        word = "higher" if row["direction"] == "up" else "lower"
        text = f"differentially expressed: {word} in treated (log2FC {lfc:+.2f}, padj {padj:.2g})"
    elif padj < PADJ_THRESHOLD:
        text = (
            f"statistically significant (padj {padj:.2g}) but the change is small "
            f"(log2FC {lfc:+.2f}, below {LOG2FC_THRESHOLD:g})"
        )
    else:
        text = f"not differentially expressed (log2FC {lfc:+.2f}, padj {padj:.2g})"
    return text
