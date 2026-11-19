import gradio as gr
import pytest

from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.plots import pca_figure, volcano_figure
from bioinfo_agent.reply import Reply, Step
from bioinfo_agent.ui import (
    PANEL_SIZE,
    TABLE_COLUMNS,
    artifact_figure,
    build_ui,
    history_frame,
    latest_volcano,
    mark_gene,
    new_artifacts,
    panel_for,
    reopen_artifact,
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


def _artifacts(*questions):
    out = []
    for question in questions:
        reply = Assistant(Backend("offline")).ask(question)
        out += new_artifacts(out, reply, question)
    return out


def test_panel_shows_the_newest_plot_and_table_and_switches_tab():
    artifacts = _artifacts("Show the volcano plot", "top 3 up genes")
    update = panel_for(artifacts, artifacts[1:])
    assert len(update) == PANEL_SIZE
    assert update[0].selected == "tables"
    assert update[4].startswith("**#2**") and len(update[5].value) == 3
    # No plot was added: the plot and its label are left alone.
    assert update[2] == update[1] and update[3] == update[1]
    assert list(update[6]["#"]) == [2, 1]


def test_history_click_reopens_an_artifact_and_ignores_unknown_ids():
    artifacts = _artifacts("Show the volcano plot", "top 3 up genes")
    reopened = reopen_artifact(artifacts, 1)
    assert len(reopened) == PANEL_SIZE
    assert reopened[0].selected == "plots" and reopened[1].startswith("**#1**")
    assert reopen_artifact(artifacts, 99) == [gr.update()] * PANEL_SIZE


def test_table_click_marks_the_gene_on_the_latest_volcano_model():
    artifacts = _artifacts("Show the lane-adjusted volcano plot")
    assert latest_volcano(artifacts)["spec"]["model"] == "lane_adjusted"
    assert latest_volcano([])["spec"]["model"] == "headline"
    tab, label, plot, detail = mark_gene(artifacts, "CIITA")
    assert tab.selected == "plots" and "selected: CIITA" in label
    assert plot.value["type"] == "plotly"  # a figure was built
    assert detail.startswith("**CIITA**: differentially expressed")
    assert mark_gene(artifacts, "NOT_A_GENE") == [gr.update()] * 4


def test_chat_round_trip_through_the_running_app():
    from gradio_client import Client

    demo = build_ui(Backend("offline"))
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True, server_port=None)
    try:
        out = Client(url, verbose=False).predict(
            "Show the volcano plot marking CIITA", api_name="/respond"
        )
    finally:
        demo.close()
    contents = [m["content"][0]["text"] for m in out[0] if m["role"] == "assistant"]
    assert any("CIITA" in c for c in contents)
    assert any("Added to the side panel" in c for c in contents)


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
    assert tuple(table_frame(added[0]["rows"]).columns) == TABLE_COLUMNS


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
