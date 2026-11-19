"""Differential expression (treated vs control) with PyDESeq2."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import binomtest, mannwhitneyu

from bioinfo_agent.config import (
    DESIGN,
    DESIGN_LANE_ADJUSTED,
    LOG2FC_THRESHOLD,
    MIN_TOTAL_COUNT,
    N_PCS,
    N_VARIABLE_GENES,
    PADJ_THRESHOLD,
    PCA_FILE,
    PCA_TREATED_FILE,
    RESULTS_FILE,
    SUMMARY_FILE,
    results_dir,
)
from bioinfo_agent.data import Dataset, add_lane


def _fit(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: str,
    factor: str,
    level: str,
    reference: str,
) -> pd.DataFrame:
    """Fit DESeq2 and return the results with an added shrunken log2FC (`log2fc_shrunk`).

    The effect reported is `level` vs `reference` of `factor`. The reference is set through the
    category order (PyDESeq2 0.6 ignores `ref_level`), which also names the coefficient
    `factor[T.level]` used for shrinkage.
    """
    design_metadata = metadata[["condition", "lane"]].copy()
    design_metadata[factor] = pd.Categorical(
        design_metadata[factor], categories=[reference, level]
    )
    dds = DeseqDataSet(
        counts=counts.T,
        metadata=design_metadata,
        design=design,
        quiet=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=[factor, level, reference],
        alpha=PADJ_THRESHOLD,
        quiet=True,
    )
    stats.summary()
    res = stats.results_df.copy()
    stats.lfc_shrink(coeff=f"{factor}[T.{level}]")
    res["log2fc_shrunk"] = stats.results_df["log2FoldChange"]
    return res


def call_direction(padj: pd.Series, log2fc: pd.Series) -> pd.Series:
    """'up' / 'down' in treated when padj and |log2FC| pass the thresholds, else 'ns'."""
    significant = (padj < PADJ_THRESHOLD) & (log2fc.abs() >= LOG2FC_THRESHOLD)
    direction = np.where(log2fc > 0, "up", "down")
    return pd.Series(np.where(significant, direction, "ns"), index=padj.index)


def lane_flag(headline: pd.Series, adjusted: pd.Series) -> pd.Series:
    """Compare the call without and with the lane term.

    same: unchanged (including not significant in both). lost: DE without lane, not DE with it.
    gained: the opposite. flipped: DE in both but in opposite directions.
    """
    both_de = (headline != "ns") & (adjusted != "ns")
    flag = np.select(
        [
            (headline != "ns") & (adjusted == "ns"),
            (headline == "ns") & (adjusted != "ns"),
            both_de & (headline != adjusted),
        ],
        ["lost", "gained", "flipped"],
        default="same",
    )
    return pd.Series(flag, index=headline.index)


def _agreement(same: pd.Series) -> dict:
    """Count and fraction of genes whose sign agrees, with a two-sided binomial test against 50%."""
    n, k = int(same.size), int(same.sum())
    return {
        "n_genes": n,
        "n_same_direction": k,
        "fraction": round(k / n, 4) if n else None,
        "p_vs_half": float(binomtest(k, n, 0.5).pvalue) if n else None,
    }


def l2_concordance(table: pd.DataFrame) -> dict:
    """Do the lane-adjusted DE genes (estimated from L1 samples only) move the same way in treated-L2
    vs control-L1?

    Not fully independent: control-L1 samples are shared, and L2 carries the lane shift. So the
    same comparison is made for genes that are not DE as a baseline: the lane shift alone already
    pushes that above 50%.
    """
    lane_adjusted = table["direction_lane_adj"]
    same = np.sign(table["log2fc_lane_adj"]) == np.sign(table["log2fc_l2_vs_control"])
    de = lane_adjusted != "ns"
    significant_l2 = table["padj_l2_vs_control"] < PADJ_THRESHOLD
    return {
        "lane_adjusted_de": _agreement(same[de]),
        "up": _agreement(same[lane_adjusted == "up"]),
        "down": _agreement(same[lane_adjusted == "down"]),
        "significant_in_l2": _agreement(same[de & significant_l2]),
        "baseline_not_de": _agreement(
            same[(lane_adjusted == "ns") & (table["direction"] == "ns")]
        ),
    }


def _pca_scores(
    counts: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample scores (PC1..PCn, with condition and lane) and variance explained, from
    variance-stabilised counts of the most variable genes (blind to the design)."""
    dds = DeseqDataSet(
        counts=counts.T,
        metadata=metadata[["condition", "lane"]],
        design=DESIGN,
        quiet=True,
    )
    dds.fit_size_factors()
    dds.vst(use_design=False)
    vst = pd.DataFrame(
        dds.layers["vst_counts"], index=counts.columns, columns=counts.index
    )
    top = vst.var().nlargest(N_VARIABLE_GENES).index
    centred = vst[top] - vst[top].mean()
    u, d, _ = np.linalg.svd(centred.to_numpy(), full_matrices=False)
    names = [f"PC{i + 1}" for i in range(N_PCS)]
    scores = pd.DataFrame(u[:, :N_PCS] * d[:N_PCS], index=counts.columns, columns=names)
    scores.insert(0, "lane", metadata["lane"])
    scores.insert(0, "condition", metadata["condition"])
    scores.index.name = "sample_id"
    return scores, d**2 / (d**2).sum()


def _p_values(scores: pd.DataFrame, group: pd.Series, a: str, b: str) -> dict:
    """Mann-Whitney p-value per PC between groups a and b (unadjusted, descriptive only)."""
    return {
        pc: float(
            mannwhitneyu(scores.loc[group == a, pc], scores.loc[group == b, pc]).pvalue
        )
        for pc in scores.columns[2:]
    }


def run_pca(
    counts: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """PCA of all samples, and of the treated samples only.

    The all-sample PCA is summarised by how well each PC separates conditions with lane held
    fixed (treated-L1 vs control-L1) and lanes with condition held fixed (treated-L2 vs
    treated-L1). The treated-only PCA drops the treatment axis, so any L1/L2 separation in it
    is a lane effect.
    """
    scores, explained = _pca_scores(counts, metadata)
    group = metadata["condition"] + "-" + metadata["lane"]

    treated = metadata.index[metadata["condition"] == "treated"]
    scores_t, explained_t = _pca_scores(counts[treated], metadata.loc[treated])
    lane = metadata.loc[treated, "lane"]
    names = list(scores.columns[2:])

    summary = {
        "n_variable_genes": N_VARIABLE_GENES,
        "variance_explained": [round(float(v), 4) for v in explained[:N_PCS]],
        "group_means": {
            g: {pc: round(float(v), 2) for pc, v in row.items()}
            for g, row in scores[names].groupby(group).mean().iterrows()
        },
        "condition_effect_p": _p_values(scores, group, "treated-L1", "control-L1"),
        "lane_effect_p": _p_values(scores, group, "treated-L2", "treated-L1"),
        "treated_only": {
            "variance_explained": [round(float(v), 4) for v in explained_t[:N_PCS]],
            "lane_group_means": {
                g: {pc: round(float(v), 2) for pc, v in row.items()}
                for g, row in scores_t[names].groupby(lane).mean().iterrows()
            },
            "lane_effect_p": _p_values(scores_t, lane, "L2", "L1"),
        },
    }
    return scores, scores_t, summary


def run_de(
    dataset: Dataset,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Returns (per-gene table, summary, PCA scores, treated-only PCA scores).

    The headline call is `~ condition`; the lane-adjusted model and the treated-only lane
    comparison are sensitivity checks."""
    metadata = add_lane(dataset.metadata)
    counts = dataset.counts.loc[:, metadata.index]
    kept = counts.loc[counts.sum(axis=1) >= MIN_TOTAL_COUNT]

    main = _fit(kept, metadata, DESIGN, "condition", "treated", "control")
    adjusted = _fit(
        kept, metadata, DESIGN_LANE_ADJUSTED, "condition", "treated", "control"
    )
    treated = metadata.index[metadata["condition"] == "treated"]
    lane_effect = _fit(
        kept[treated], metadata.loc[treated], "~ lane", "lane", "L2", "L1"
    )

    l2_samples = metadata.index[
        ((metadata["condition"] == "control") & (metadata["lane"] == "L1"))
        | ((metadata["condition"] == "treated") & (metadata["lane"] == "L2"))
    ]
    l2_vs_control = _fit(
        kept[l2_samples],
        metadata.loc[l2_samples],
        DESIGN,
        "condition",
        "treated",
        "control",
    )

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
    table["log2fc_lane_adj"] = adjusted["log2fc_shrunk"]
    table["padj_lane_adj"] = adjusted["padj"]
    table["direction_lane_adj"] = call_direction(
        table["padj_lane_adj"], table["log2fc_lane_adj"]
    )
    table["log2fc_lane_effect"] = lane_effect["log2fc_shrunk"]
    table["padj_lane_effect"] = lane_effect["padj"]
    table["log2fc_l2_vs_control"] = l2_vs_control["log2fc_shrunk"]
    table["padj_l2_vs_control"] = l2_vs_control["padj"]
    table["lane_flag"] = lane_flag(table["direction"], table["direction_lane_adj"])

    scores, scores_treated, pca_summary = run_pca(kept, metadata)
    lane_de = call_direction(table["padj_lane_effect"], table["log2fc_lane_effect"])

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
        "lane_check": {
            "lane_adjusted_design": DESIGN_LANE_ADJUSTED,
            "n_up_lane_adjusted": int((table["direction_lane_adj"] == "up").sum()),
            "n_down_lane_adjusted": int((table["direction_lane_adj"] == "down").sum()),
            "flags": {k: int(v) for k, v in table["lane_flag"].value_counts().items()},
            # Genes whose call changes and that also differ between the two treated lanes.
            "lane_effect_among_changed": {
                k: int(
                    (
                        (table["lane_flag"] == k)
                        & (table["padj_lane_effect"] < PADJ_THRESHOLD)
                    ).sum()
                )
                for k in ("lost", "gained", "flipped")
            },
            "n_lane_effect_padj": int(
                (table["padj_lane_effect"] < PADJ_THRESHOLD).sum()
            ),
            "n_lane_effect_de": int((lane_de != "ns").sum()),
            "l2_concordance": {
                "comparison": f"treated-L2 vs control-L1 ({len(l2_samples)} samples)",
                **l2_concordance(table),
            },
            "pca": pca_summary,
        },
        "versions": {"pydeseq2": version("pydeseq2"), "pandas": pd.__version__},
    }
    return table, summary, scores, scores_treated


def save_results(
    table: pd.DataFrame,
    summary: dict,
    pca: pd.DataFrame,
    pca_treated: pd.DataFrame,
    directory: Path | None = None,
):
    directory = directory or results_dir()
    directory.mkdir(parents=True, exist_ok=True)
    table.sort_values("padj").to_csv(
        directory / RESULTS_FILE, float_format="%.6g", lineterminator="\n"
    )
    for frame, name in ((pca, PCA_FILE), (pca_treated, PCA_TREATED_FILE)):
        frame.to_csv(directory / name, float_format="%.6g", lineterminator="\n")
    (directory / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return directory
