"""Gradio chat front end over the same Assistant the CLI uses."""

import gradio as gr

from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.offline import Reply
from bioinfo_agent.plots import pca_figure

EXAMPLES = [
    "How can you help me?",
    "What are the top upregulated genes?",
    "Is STAT1 differentially expressed?",
    "What about CIITA, SPATS2L and IDH3B?",
    "How many genes are differentially expressed, and how was that decided?",
    "Show the PCA: does lane or condition drive the variation?",
    "Which genes are sensitive to the lane effect?",
]

_STEP_TITLES = {
    "thinking": "Reasoning",
    "call": "Tool call",
    "result": "Tool result",
}


def reply_to_messages(reply: Reply) -> list[dict]:
    """Chat messages for a reply: one collapsible message per step, the answer, then any plots."""
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
    for plot in reply.plots:
        figure = pca_figure(plot["x"], plot["y"], plot["subset"])
        messages.append({"role": "assistant", "content": gr.Plot(figure)})
    return messages


def backend_badge(backend: Backend) -> str:
    return f"**Backend:** `{backend.label}`"


def build_ui(backend: Backend) -> gr.Blocks:
    with gr.Blocks(title="bioinfo-agent") as demo:
        gr.Markdown(
            "# bioinfo-agent\nAsk questions about the treated vs control differential "
            "expression results. Every number comes from a tool call, shown above the answer."
        )
        gr.Markdown(backend_badge(backend))
        chatbot = gr.Chatbot(height=520, label="Conversation", buttons=["copy"])
        box = gr.Textbox(
            placeholder="Ask about genes, e.g. Is STAT1 differentially expressed?",
            show_label=False,
            autofocus=True,
        )
        gr.Examples(EXAMPLES, inputs=box, label="Try one of these")
        clear = gr.Button("New conversation")
        # Per browser session: one Assistant (the model's history) and the messages shown. The
        # shown messages live in state, not read back from the chatbot, so plots are not round-tripped.
        assistant_state = gr.State(None)
        shown_state = gr.State([])

        def respond(question, shown, assistant):
            question = (question or "").strip()
            if not question:
                yield shown, shown, assistant, gr.update()
                return
            assistant = assistant or Assistant(backend)
            shown = [*shown, {"role": "user", "content": question}]
            pending = {
                "role": "assistant",
                "content": "",
                "metadata": {"title": "Working…", "status": "pending"},
            }
            yield [*shown, pending], shown, assistant, ""
            shown = [*shown, *reply_to_messages(assistant.ask(question))]
            yield shown, shown, assistant, ""

        box.submit(
            respond,
            [box, shown_state, assistant_state],
            [chatbot, shown_state, assistant_state, box],
        )
        clear.click(
            lambda: ([], [], None, ""),
            None,
            [chatbot, shown_state, assistant_state, box],
        )
    return demo


def launch(backend: Backend, host: str = "127.0.0.1", port: int = 7860) -> None:
    build_ui(backend).launch(server_name=host, server_port=port, inbrowser=False)
