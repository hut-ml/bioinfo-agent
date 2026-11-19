import gradio as gr
import pytest

from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.offline import Reply, Step
from bioinfo_agent.plots import pca_figure, volcano_figure
from bioinfo_agent.ui import (
    TABLE_COLUMNS,
    artifact_figure,
    build_ui,
    history_frame,
    new_artifacts,
    reply_to_messages,
    table_frame,
)


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


def test_plots_and_tables_become_numbered_artifacts_not_chat_content():
    question = "Show the PCA"
    reply = Assistant(Backend("offline")).ask(question)
    first = new_artifacts([], reply, question)
    assert [a["id"] for a in first] == [1, 2]
    assert [a["kind"] for a in first] == ["plot", "plot"]
    assert first[1]["title"] == "PCA (treated samples only), PC1 vs PC2"
    # Ids continue from the existing artifacts; the chat only gets a pointer to the panel.
    table_reply = Assistant(Backend("offline")).ask("top 3 up genes")
    added = new_artifacts(first, table_reply, "top 3 up genes")
    assert [(a["id"], a["kind"]) for a in added] == [(3, "table")]
    messages = reply_to_messages(table_reply, added)
    assert "#3" in messages[-1]["content"]
    assert not any(isinstance(m["content"], gr.Plot) for m in messages)
    frame = history_frame([*first, *added])
    assert list(frame["#"]) == [3, 2, 1]
    assert list(frame["Type"]) == ["Table", "Plot", "Plot"]
    assert list(table_frame(added[0]["rows"]).columns) == TABLE_COLUMNS


def test_artifact_figures_are_rebuilt_from_their_spec():
    spec = {"plot": "volcano", "model": "headline", "genes": ["STAT1"]}
    assert artifact_figure(spec, ("CIITA",)).data[-1].text == ("STAT1", "CIITA")
    pca = {"plot": "pca", "x": 1, "y": 2, "subset": "treated"}
    assert len(artifact_figure(pca).data) == 2


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


def test_volcano_figure_splits_genes_by_call_and_marks_selected_genes():
    sizes = {t.name.split(" (")[0]: len(t.x) for t in volcano_figure().data}
    assert (sizes["Up in treated"], sizes["Down in treated"]) == (222, 100)
    adjusted = volcano_figure("lane_adjusted", ["stat1", "nope"])
    assert {t.name.split(" (")[0]: len(t.x) for t in adjusted.data}[
        "Up in treated"
    ] == 174
    assert adjusted.data[-1].name == "Selected" and adjusted.data[-1].text == ("STAT1",)
    with pytest.raises(ValueError):
        volcano_figure("sideways")
