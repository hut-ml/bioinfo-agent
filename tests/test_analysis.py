import pandas as pd
import pytest

from bioinfo_agent.analysis import call_direction
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
