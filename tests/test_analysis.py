import pandas as pd
import pytest

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
