"""Differential expression (treated vs control) with PyDESeq2."""

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from bioinfo_agent.data import PROJECT_ROOT, Dataset, add_lane

PADJ_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
MIN_TOTAL_COUNT = 10

DESIGN = "~ condition"

RESULTS_FILE = "de_results.csv"
SUMMARY_FILE = "summary.json"


def results_dir() -> Path:
    return Path(os.environ.get("BIOINFO_RESULTS_DIR", PROJECT_ROOT / "results"))


def _fit(counts: pd.DataFrame, metadata: pd.DataFrame):
    """Fit DESeq2 and return (results with shrunken log2FC, size factors)."""
    dds = DeseqDataSet(
        counts=counts.T,
        metadata=metadata[["condition"]],
        design=DESIGN,
        ref_level=["condition", "control"],
        quiet=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=["condition", "treated", "control"],
        alpha=PADJ_THRESHOLD,
        quiet=True,
    )
    stats.summary()
    res = stats.results_df.copy()
    stats.lfc_shrink(coeff="condition[T.treated]")
    res["log2fc_shrunk"] = stats.results_df["log2FoldChange"]
    return res, dds.obs["size_factors"]


def call_direction(padj: pd.Series, log2fc: pd.Series) -> pd.Series:
    """'up' / 'down' in treated when padj and |log2FC| pass the thresholds, else 'ns'."""
    significant = (padj < PADJ_THRESHOLD) & (log2fc.abs() >= LOG2FC_THRESHOLD)
    direction = np.where(log2fc > 0, "up", "down")
    return pd.Series(np.where(significant, direction, "ns"), index=padj.index)


def run_de(dataset: Dataset) -> tuple[pd.DataFrame, dict]:
    metadata = add_lane(dataset.metadata)
    counts = dataset.counts.loc[:, metadata.index]
    kept = counts.loc[counts.sum(axis=1) >= MIN_TOTAL_COUNT]

    main, size_factors = _fit(kept, metadata)

    table = pd.DataFrame(
        {
            "base_mean": main["baseMean"],
            "log2fc": main["log2fc_shrunk"],
            "log2fc_unshrunk": main["log2FoldChange"],
            "lfc_se": main["lfcSE"],
            "stat": main["stat"],
            "pvalue": main["pvalue"],
            "padj": main["padj"],
        }
    )
    table.index.name = "gene"
    table["direction"] = call_direction(table["padj"], table["log2fc"])

    summary = {
        "design": DESIGN,
        "comparison": "treated vs control (positive log2FC = higher in treated)",
        "log2fc": "shrunken (apeglm-style, PyDESeq2 lfc_shrink)",
        "thresholds": {
            "padj": PADJ_THRESHOLD,
            "abs_log2fc": LOG2FC_THRESHOLD,
            "min_total_count": MIN_TOTAL_COUNT,
        },
        "n_samples": int(counts.shape[1]),
        # Observation only: lane is not part of the model.
        "samples": {
            f"{cond}-{lane}": int(n)
            for (cond, lane), n in metadata.groupby(["condition", "lane"])
            .size()
            .items()
        },
        "n_genes_input": int(counts.shape[0]),
        "n_genes_tested": int(kept.shape[0]),
        "n_genes_with_padj": int(table["padj"].notna().sum()),
        "n_up": int((table["direction"] == "up").sum()),
        "n_down": int((table["direction"] == "down").sum()),
        "size_factors": {k: round(float(v), 4) for k, v in size_factors.items()},
        "versions": {"pydeseq2": version("pydeseq2"), "pandas": pd.__version__},
    }
    return table, summary


def save_results(table: pd.DataFrame, summary: dict, directory: Path | None = None):
    directory = directory or results_dir()
    directory.mkdir(parents=True, exist_ok=True)
    table.sort_values("padj").to_csv(
        directory / RESULTS_FILE, float_format="%.6g", lineterminator="\n"
    )
    (directory / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return directory
