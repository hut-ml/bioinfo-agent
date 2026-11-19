import pytest

from bioinfo_agent import queries
from bioinfo_agent.analysis import LOG2FC_THRESHOLD, PADJ_THRESHOLD

EXAMPLE_GENES = ["CIITA", "SPATS2L", "IDH3B", "STAT1"]


def test_example_genes_have_results():
    found = queries.get_genes(EXAMPLE_GENES)
    assert all(g["found"] and g["padj"] is not None for g in found)
    by_gene = {g["gene"]: g for g in found}
    assert by_gene["CIITA"]["direction"] == "up"
    assert by_gene["STAT1"]["direction"] == "ns"


def test_lookup_is_case_insensitive_and_suggests_for_unknown():
    assert queries.get_genes([" ciita "])[0]["gene"] == "CIITA"
    unknown = queries.get_genes(["STTA1"])[0]
    assert unknown["found"] is False
    assert "STAT1" in unknown["did_you_mean"]


@pytest.mark.parametrize("direction", ["up", "down"])
def test_top_genes_filtered_and_sorted(direction):
    genes = queries.top_genes(direction, n=15)
    assert len(genes) == 15
    assert {g["direction"] for g in genes} == {direction}
    assert all(g["padj"] < PADJ_THRESHOLD for g in genes)
    assert all(abs(g["log2fc"]) >= LOG2FC_THRESHOLD for g in genes)
    padjs = [g["padj"] for g in genes]
    assert padjs == sorted(padjs)


def test_top_genes_by_fold_change_and_counts_match_overview():
    genes = queries.top_genes("all", n=10, sort_by="log2fc")
    sizes = [abs(g["log2fc"]) for g in genes]
    assert sizes == sorted(sizes, reverse=True)
    ov = queries.overview()
    total = len(queries.top_genes("all", n=10_000))
    assert total == ov["n_up"] + ov["n_down"]


def test_invalid_arguments_and_missing_results(tmp_path):
    with pytest.raises(ValueError):
        queries.top_genes("sideways")
    with pytest.raises(queries.ResultsNotFound):
        queries.overview(tmp_path)


def test_lane_check_and_pca_are_consistent_with_the_results():
    check = queries.lane_check()
    assert check["samples"] == {"control-L1": 40, "treated-L1": 25, "treated-L2": 25}
    table, _ = queries.load_results()
    assert check["flags"] == table["lane_flag"].value_counts().drop(
        "same"
    ).to_dict() | {"same": int((table["lane_flag"] == "same").sum())}
    pca = queries.load_pca()
    assert pca.shape == (90, 7)
    assert pca.groupby(["condition", "lane"]).size().to_dict() == {
        ("control", "L1"): 40,
        ("treated", "L1"): 25,
        ("treated", "L2"): 25,
    }


def test_treated_only_pca_has_only_treated_samples_and_separates_lanes():
    treated = queries.load_pca("treated")
    assert set(treated["condition"]) == {"treated"}
    assert treated.groupby("lane").size().to_dict() == {"L1": 25, "L2": 25}
    only = queries.lane_check()["pca"]["treated_only"]
    # Treatment is held fixed here, so a strong L1/L2 separation is a lane effect.
    assert min(only["lane_effect_p"].values()) < 1e-6
    with pytest.raises(ValueError):
        queries.load_pca("control")


def test_lane_flagged_genes_get_a_caution_in_their_verdict():
    by_gene = {g["gene"]: g for g in queries.get_genes(EXAMPLE_GENES)}
    # SPATS2L is DE only when lane is ignored; CIITA holds either way.
    assert by_gene["SPATS2L"]["lane_flag"] == "lost"
    assert "may be a lane effect" in by_gene["SPATS2L"]["verdict"]
    assert by_gene["CIITA"]["lane_flag"] == "same"
    assert "holds after accounting for lane" in by_gene["CIITA"]["verdict"]
    assert "Caution" not in by_gene["STAT1"]["verdict"]


def test_lane_sensitive_genes_filtered_and_ranked_by_change_in_log2fc():
    genes = queries.lane_sensitive_genes("lost", n=20)
    assert {g["lane_flag"] for g in genes} == {"lost"}
    changes = [abs(g["log2fc"] - g["log2fc_lane_adjusted"]) for g in genes]
    assert changes == sorted(changes, reverse=True)
    with pytest.raises(ValueError):
        queries.lane_sensitive_genes("sideways")


def test_l2_concordance_is_high_for_lane_adjusted_genes_and_above_its_baseline():
    c = queries.l2_concordance()
    n_de = sum(queries.lane_check()[f"n_{d}_lane_adjusted"] for d in ("up", "down"))
    assert c["lane_adjusted_de"]["n_genes"] == n_de
    assert c["lane_adjusted_de"]["fraction"] > 0.9
    # The lane shift alone lifts the baseline above 50%, so the excess is what matters.
    assert c["baseline_not_de"]["fraction"] < c["lane_adjusted_de"]["fraction"] - 0.2
