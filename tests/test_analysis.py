import json
import os
import shutil

import pandas as pd
import pytest
from conftest import DOWN, LANE, UP

from bioinfo_agent import queries
from bioinfo_agent.analysis import call_direction, l2_concordance, lane_flag
from bioinfo_agent.data import add_lane, load_dataset


def test_call_direction_applies_both_thresholds():
    padj = pd.Series([0.001, 0.001, 0.2, 0.001], index=list("abcd"))
    lfc = pd.Series([1.5, -2.0, 3.0, 0.4], index=list("abcd"))
    assert list(call_direction(padj, lfc)) == ["up", "down", "ns", "ns"]


def test_lane_comes_from_sample_id_suffix():
    meta = add_lane(load_dataset().metadata)
    table = meta.groupby(["condition", "lane"]).size()
    assert table.to_dict() == {
        ("control", "L1"): 40,
        ("treated", "L1"): 25,
        ("treated", "L2"): 25,
    }


def test_missing_lane_suffix_raises():
    meta = pd.DataFrame(
        {"condition": ["control"]}, index=pd.Index(["abc"], name="sample_id")
    )
    with pytest.raises(ValueError, match="lane suffix"):
        add_lane(meta)


def test_lane_flag_compares_headline_and_lane_adjusted_calls():
    headline = pd.Series(["up", "up", "ns", "up", "ns", "down"], index=list("abcdef"))
    adjusted = pd.Series(
        ["ns", "up", "down", "down", "ns", "down"], index=list("abcdef")
    )
    assert list(lane_flag(headline, adjusted)) == [
        "lost",
        "same",
        "gained",
        "flipped",
        "same",
        "same",
    ]


def test_l2_concordance_counts_sign_agreement_for_lane_adjusted_de_genes():
    table = pd.DataFrame(
        {
            "direction": ["up", "up", "down", "ns", "ns", "ns"],
            "direction_lane_adj": ["up", "up", "down", "ns", "ns", "up"],
            "log2fc_lane_adj": [1.5, 1.2, -1.1, 0.1, -0.2, 1.4],
            "log2fc_l2_vs_control": [2.0, -0.3, -1.0, -0.5, -0.1, 0.9],
            "padj_l2_vs_control": [0.001, 0.4, 0.001, 0.9, 0.9, 0.001],
        },
        index=list("abcdef"),
    )
    out = l2_concordance(table)
    assert out["lane_adjusted_de"]["n_genes"] == 4
    assert out["lane_adjusted_de"]["n_same_direction"] == 3
    assert out["up"]["n_same_direction"] == 2 and out["down"]["fraction"] == 1.0
    assert out["significant_in_l2"]["n_genes"] == 3
    assert out["significant_in_l2"]["fraction"] == 1.0
    # Baseline: genes not DE in either model (d and e); only e agrees in sign.
    assert out["baseline_not_de"]["n_genes"] == 2
    assert out["baseline_not_de"]["n_same_direction"] == 1


def test_pipeline_recovers_planted_effects(synthetic_run):
    table, summary, _ = synthetic_run
    assert (table.loc[UP, "direction"] == "up").all()
    assert (table.loc[DOWN, "direction"] == "down").all()
    # Higher in L2 only: DE without lane in the model, not DE with it.
    assert (table.loc[LANE, "lane_flag"] == "lost").all()
    assert (table.loc[UP + DOWN, "lane_flag"] == "same").all()
    assert (summary["n_up"], summary["n_down"]) == (16, 4)
    assert summary["lane_check"]["flags"] == {"same": 142, "lost": 8}
    assert summary["lane_check"]["l2_concordance"]["comparison"] == (
        "treated-L2 vs control-L1 (14 samples)"
    )
    assert summary["samples"] == {"control-L1": 8, "treated-L1": 6, "treated-L2": 6}


def test_saved_results_can_be_queried(synthetic_run):
    _, _, results = synthetic_run
    assert sorted(p.name for p in results.iterdir()) == [
        "de_results.csv",
        "pca.csv",
        "pca_treated.csv",
        "summary.json",
    ]
    assert queries.overview(results)["n_up"] == 16
    assert queries.get_genes(["up0"], results)[0]["direction"] == "up"
    lost = queries.lane_sensitive_genes("lost", 20, results)
    assert {g["gene"] for g in lost} == set(LANE)
    treated = queries.load_pca("treated", results)
    assert set(treated["condition"]) == {"treated"} and len(treated) == 12


def test_results_are_reloaded_after_precompute_rewrites_them(synthetic_run, tmp_path):
    _, _, results = synthetic_run
    copy = tmp_path / "results"
    shutil.copytree(results, copy)
    assert queries.overview(copy)["n_up"] == 16
    summary = json.loads((copy / "summary.json").read_text())
    summary["n_up"] = 99
    (copy / "summary.json").write_text(json.dumps(summary))
    os.utime(copy / "summary.json", ns=(1, 1))
    assert queries.overview(copy)["n_up"] == 99
