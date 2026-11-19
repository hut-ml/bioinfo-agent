import gradio as gr
import pytest

from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.offline import Reply, Step
from bioinfo_agent.plots import pca_figure
from bioinfo_agent.ui import build_ui, reply_to_messages


def test_steps_become_collapsible_messages_before_the_answer():
    reply = Reply(
        "done",
        [Step("call", "get_genes(names=['STAT1'])"), Step("result", "STAT1: ns")],
        notice="fallback",
    )
    messages = reply_to_messages(reply)
    assert [m["metadata"]["title"] for m in messages[:2]] == [
        "Tool call",
        "Tool result",
    ]
    assert "fallback" in messages[2]["content"] and "metadata" not in messages[2]
    assert messages[-1] == {"role": "assistant", "content": "done"}


def test_offline_question_shows_tool_call_and_answer():
    messages = reply_to_messages(
        Assistant(Backend("offline")).ask("Is STAT1 differentially expressed?")
    )
    assert messages[0]["metadata"]["title"] == "Tool call"
    assert "STAT1" in messages[-1]["content"]


def test_ui_builds():
    assert build_ui(Backend("offline")) is not None


def test_pca_request_becomes_an_inline_plot_after_the_answer():
    messages = reply_to_messages(Assistant(Backend("offline")).ask("Show the PCA"))
    assert messages[0]["metadata"]["title"] == "Tool call"
    # The answer comes first, then the all-sample and the treated-only plots.
    assert [isinstance(m["content"], gr.Plot) for m in messages[-3:]] == [
        False,
        True,
        True,
    ]


def test_pca_figure_has_a_trace_per_group_in_both_panels():
    fig = pca_figure(1, 2)
    assert len(fig.data) == 6
    assert {t.name for t in fig.data} == {"control-L1", "treated-L1", "treated-L2"}
    treated = pca_figure(2, 1, "treated")
    assert {t.name for t in treated.data} == {"treated-L1", "treated-L2"}
    assert sum(len(t.x) for t in treated.data) == 50
    with pytest.raises(ValueError):
        pca_figure(1, 1)
    with pytest.raises(ValueError):
        pca_figure(1, 9)
