import pytest

from bioinfo_agent.data import cohort_table, data_dir, load_dataset


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


def test_crlf_files_load(tmp_path):
    # Windows checkouts may convert the CSVs to CRLF line endings.
    (tmp_path / "counts.csv").write_bytes(b"gene,a,b\r\ng1,1,2\r\ng2,3,4\r\n")
    (tmp_path / "metadata.csv").write_bytes(
        b"sample_id,condition\r\na,control\r\nb,treated\r\n"
    )
    ds = load_dataset(tmp_path)
    assert (ds.n_genes, ds.n_samples) == (2, 2)
    assert list(ds.metadata["condition"]) == ["control", "treated"]
    assert list(ds.counts.index) == ["g1", "g2"]


def test_data_dir_env_override(tmp_path, monkeypatch):
    (tmp_path / "counts.csv").write_text("gene,a\ng1,1\n")
    (tmp_path / "metadata.csv").write_text("sample_id,condition\na,control\n")
    monkeypatch.setenv("BIOINFO_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert load_dataset().n_genes == 1
