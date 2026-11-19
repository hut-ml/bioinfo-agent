"""Gradio front end over the same Assistant the CLI uses.

The chat is on the left. Plots and tables the agent produces are collected as artifacts in a
right-hand panel (Plots, Tables, History) so they stay available after the conversation moves on.
"""

import gradio as gr
import pandas as pd

from bioinfo_agent import queries
from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.plots import pca_figure, volcano_figure
from bioinfo_agent.reply import Reply
from bioinfo_agent.tools import TABLE_COLUMNS

EXAMPLES = [
    "How can you help me?",
    "What are the top upregulated genes?",
    "Is STAT1 differentially expressed?",
    "What about CIITA, SPATS2L and IDH3B?",
    "Show the volcano plot",
    "Show the PCA: does lane or condition drive the variation?",
    "Which genes are sensitive to the lane effect?",
]

WELCOME = (
    "### Ask about the treated vs control results\n"
    "Every number comes from a tool call, shown above the answer. Plots and tables open in "
    "the panel on the right."
)

_STEP_TITLES = {
    "thinking": "Reasoning",
    "call": "Tool call",
    "result": "Tool result",
}

HISTORY_COLUMNS = ["#", "Type", "Title", "Asked"]
PLOTS_TAB, TABLES_TAB, HISTORY_TAB = "plots", "tables", "history"
EMPTY_PLOT = "Plots requested in the chat (volcano, PCA) appear here."
EMPTY_TABLE = "Gene tables from the chat appear here. Click a row to mark that gene on the volcano plot."


def plot_title(spec: dict) -> str:
    if spec["plot"] == "volcano":
        marked = f", marking {', '.join(spec['genes'])}" if spec["genes"] else ""
        return f"Volcano plot ({spec['model'].replace('_', '-')} model{marked})"
    what = "treated samples only" if spec["subset"] == "treated" else "all samples"
    return f"PCA ({what}), PC{spec['x']} vs PC{spec['y']}"


def new_artifacts(artifacts: list[dict], reply: Reply, question: str) -> list[dict]:
    """The reply's plots and tables as numbered artifacts, to append to the session's list."""
    made = [
        {"kind": "plot", "title": plot_title(spec), "spec": spec, "question": question}
        for spec in reply.plots
    ] + [
        {
            "kind": "table",
            "title": table["title"],
            "rows": table["rows"],
            "question": question,
        }
        for table in reply.tables
    ]
    start = len(artifacts) + 1
    return [{"id": start + i, **a} for i, a in enumerate(made)]


def artifact_figure(spec: dict, highlight: tuple[str, ...] = ()):
    if spec["plot"] == "volcano":
        return volcano_figure(spec["model"], [*spec["genes"], *highlight])
    return pca_figure(spec["x"], spec["y"], spec["subset"])


def table_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def history_frame(artifacts: list[dict]) -> pd.DataFrame:
    """One row per artifact, newest first."""
    rows = [
        [a["id"], "Plot" if a["kind"] == "plot" else "Table", a["title"], a["question"]]
        for a in reversed(artifacts)
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def reply_to_messages(reply: Reply, added: list[dict] | None = None) -> list[dict]:
    """Chat messages for a reply: one collapsible message per step, the answer, then a pointer to
    the plots and tables that were added to the side panel."""
    messages = [
        {
            "role": "assistant",
            "content": step.text,
            "metadata": {"title": _STEP_TITLES.get(step.kind, step.kind)},
        }
        for step in reply.steps
    ]
    if reply.notice:
        messages.append({"role": "assistant", "content": f"⚠️ {reply.notice}"})
    messages.append({"role": "assistant", "content": reply.answer})
    if added:
        lines = [
            f"{'📈' if a['kind'] == 'plot' else '📋'} **#{a['id']}** {a['title']}"
            for a in added
        ]
        messages.append(
            {
                "role": "assistant",
                "content": "Added to the side panel:\n\n"
                + "\n".join(f"- {line}" for line in lines),
            }
        )
    return messages


def backend_badge(backend: Backend) -> str:
    return f"**Backend:** `{backend.label}`"


def _show_plot(artifact: dict, highlight: tuple[str, ...] = ()) -> tuple:
    """Updates for (plot, plot title) showing a plot artifact."""
    title = artifact["title"]
    if highlight:
        title += f"; selected: {', '.join(highlight)}"
    return gr.Plot(
        artifact_figure(artifact["spec"], highlight)
    ), f"**#{artifact['id']}** {title}"


def _show_table(artifact: dict) -> tuple:
    return (
        gr.Dataframe(table_frame(artifact["rows"])),
        f"**#{artifact['id']}** {artifact['title']}",
    )


# Outputs of the side panel, in order: tabs, plot label, plot, gene detail, table label, table, history.
PANEL_SIZE = 7


def panel_for(artifacts: list[dict], added: list[dict]) -> list:
    """Panel updates showing the newest added plot and table, and switching to the tab of the
    newest artifact. Everything else is left as it is."""
    keep = gr.update()
    plots = [a for a in added if a["kind"] == "plot"]
    tables = [a for a in added if a["kind"] == "table"]
    plot_up, plot_label_up = _show_plot(plots[-1]) if plots else (keep, keep)
    table_up, table_label_up = _show_table(tables[-1]) if tables else (keep, keep)
    tab = keep
    if added:
        tab = gr.Tabs(selected=PLOTS_TAB if added[-1]["kind"] == "plot" else TABLES_TAB)
    return [
        tab,
        plot_label_up,
        plot_up,
        "" if plots else keep,
        table_label_up,
        table_up,
        history_frame(artifacts),
    ]


def reopen_artifact(artifacts: list[dict], artifact_id) -> list:
    """History row click: reopen that plot or table (the history itself is left as it is)."""
    chosen = next((a for a in artifacts if a["id"] == artifact_id), None)
    if chosen is None:
        return [gr.update()] * PANEL_SIZE
    return panel_for(artifacts, [chosen])[:-1] + [gr.update()]


def latest_volcano(artifacts: list[dict]) -> dict:
    """The newest volcano plot of the conversation, else a headline one."""
    found = next(
        (
            a
            for a in reversed(artifacts)
            if a["kind"] == "plot" and a["spec"]["plot"] == "volcano"
        ),
        None,
    )
    if found:
        return found
    spec = {"plot": "volcano", "model": "headline", "genes": []}
    return {"id": "-", "kind": "plot", "title": plot_title(spec), "spec": spec}


def mark_gene(artifacts: list[dict], gene: str) -> list:
    """Table row click: mark the gene on a volcano plot and show its verdict."""
    found = queries.get_genes([gene])[0]
    if not found["found"]:
        return [gr.update()] * 4
    plot_up, label_up = _show_plot(latest_volcano(artifacts), (gene,))
    return [
        gr.Tabs(selected=PLOTS_TAB),
        label_up,
        plot_up,
        f"**{gene}**: {found['verdict']}",
    ]


def build_ui(backend: Backend) -> gr.Blocks:
    with gr.Blocks(title="bioinfo-agent") as demo:
        gr.Markdown(
            "# 🧬 bioinfo-agent\nAn assistant for the treated vs control differential "
            "expression results."
        )
        gr.Markdown(backend_badge(backend))
        # Per browser session: one Assistant (the model's history), the messages shown, and the
        # plot/table artifacts. The shown messages live in state, not read back from the chatbot.
        assistant_state = gr.State(None)
        shown_state = gr.State([])
        artifacts_state = gr.State([])

        chatbot = gr.Chatbot(
            height=600,
            label="Conversation",
            layout="bubble",
            buttons=["copy"],
            placeholder=WELCOME,
        )
        with gr.Row():
            box = gr.Textbox(
                placeholder="Ask about genes, e.g. Is STAT1 differentially expressed?",
                show_label=False,
                autofocus=True,
                scale=5,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=box, label="Try one of these")
        clear = gr.Button("New conversation", size="sm")

        # A collapsible right-hand sidebar: fold it away to give the chat the full width.
        with (
            gr.Sidebar(position="right", open=True, width=640, label="Plots & tables"),
            gr.Tabs(selected=PLOTS_TAB) as tabs,
        ):
            with gr.Tab("📈 Plots", id=PLOTS_TAB):
                plot_label = gr.Markdown(EMPTY_PLOT)
                plot_out = gr.Plot(show_label=False)
                gene_detail = gr.Markdown()
            with gr.Tab("📋 Tables", id=TABLES_TAB):
                table_label = gr.Markdown(EMPTY_TABLE)
                table_out = gr.Dataframe(
                    headers=list(TABLE_COLUMNS),
                    interactive=False,
                    show_search="filter",
                    max_height=520,
                )
            with gr.Tab("🕘 History", id=HISTORY_TAB):
                gr.Markdown(
                    "Every plot and table of this conversation. Click a row to reopen it."
                )
                history_out = gr.Dataframe(
                    headers=HISTORY_COLUMNS, interactive=False, wrap=True
                )

        panel = [
            tabs,
            plot_label,
            plot_out,
            gene_detail,
            table_label,
            table_out,
            history_out,
        ]
        keep = gr.update()

        def respond(question, shown, assistant, artifacts):
            question = (question or "").strip()
            if not question:
                yield [
                    shown,
                    shown,
                    assistant,
                    artifacts,
                    gr.update(),
                    *[keep] * len(panel),
                ]
                return
            assistant = assistant or Assistant(backend)
            shown = [*shown, {"role": "user", "content": question}]
            pending = {
                "role": "assistant",
                "content": "",
                "metadata": {"title": "Working…", "status": "pending"},
            }
            yield [
                [*shown, pending],
                shown,
                assistant,
                artifacts,
                "",
                *[keep] * len(panel),
            ]
            reply = assistant.ask(question)
            added = new_artifacts(artifacts, reply, question)
            artifacts = [*artifacts, *added]
            shown = [*shown, *reply_to_messages(reply, added)]
            yield [shown, shown, assistant, artifacts, "", *panel_for(artifacts, added)]

        def open_artifact(artifacts, evt: gr.SelectData):
            return reopen_artifact(artifacts, evt.row_value[0])

        def mark_selected_gene(artifacts, evt: gr.SelectData):
            return mark_gene(artifacts, str(evt.row_value[0]))

        chat_inputs = [box, shown_state, assistant_state, artifacts_state]
        chat_outputs = [
            chatbot,
            shown_state,
            assistant_state,
            artifacts_state,
            box,
            *panel,
        ]
        box.submit(respond, chat_inputs, chat_outputs)
        send.click(respond, chat_inputs, chat_outputs)

        history_out.select(open_artifact, artifacts_state, panel)
        table_out.select(
            mark_selected_gene,
            artifacts_state,
            [tabs, plot_label, plot_out, gene_detail],
        )
        clear.click(
            lambda: (
                [],
                [],
                None,
                [],
                "",
                gr.Tabs(selected=PLOTS_TAB),
                EMPTY_PLOT,
                None,
                "",
                EMPTY_TABLE,
                pd.DataFrame(columns=list(TABLE_COLUMNS)),
                pd.DataFrame(columns=HISTORY_COLUMNS),
            ),
            None,
            chat_outputs,
        )
    return demo


def launch(backend: Backend, host: str = "127.0.0.1", port: int = 7860) -> None:
    build_ui(backend).launch(
        server_name=host,
        server_port=port,
        inbrowser=False,
    )
