import pytest

from bioinfo_agent.data import cohort_table, load_dataset


def test_dataset_shape_and_groups():
    ds = load_dataset()
    assert (ds.n_genes, ds.n_samples) == (15831, 90)
    assert list(ds.counts.columns) == list(ds.metadata.index)
    table = cohort_table(ds.metadata)
    assert table["control"] == 40
    assert table["treated"] == 50


def test_mismatched_sample_ids_raise(tmp_path):
    (tmp_path / "counts.csv").write_text("gene,a,b\ng1,1,2\n")
    (tmp_path / "metadata.csv").write_text(
        "sample_id,condition\na,control\nc,treated\n"
    )
    with pytest.raises(ValueError, match="do not match"):
        load_dataset(tmp_path)
