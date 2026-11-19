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
