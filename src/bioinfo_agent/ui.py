"""Gradio chat front end over the same Assistant the CLI uses."""

import gradio as gr

from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.offline import Reply

EXAMPLES = [
    "How can you help me?",
    "What are the top upregulated genes?",
    "Is STAT1 differentially expressed?",
    "What about CIITA, SPATS2L and IDH3B?",
    "How many genes are differentially expressed, and how was that decided?",
]

_STEP_TITLES = {
    "thinking": "Reasoning",
    "call": "Tool call",
    "result": "Tool result",
}


def reply_to_messages(reply: Reply) -> list[dict]:
    """Chat messages for a reply: one collapsible message per step, then the answer."""
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
        # One Assistant (and so one conversation history) per browser session.
        assistant_state = gr.State(None)

        def respond(question, history, assistant):
            question = (question or "").strip()
            if not question:
                yield history, assistant, gr.update()
                return
            assistant = assistant or Assistant(backend)
            history = [*history, {"role": "user", "content": question}]
            pending = {
                "role": "assistant",
                "content": "",
                "metadata": {"title": "Working…", "status": "pending"},
            }
            yield [*history, pending], assistant, ""
            yield [*history, *reply_to_messages(assistant.ask(question))], assistant, ""

        box.submit(
            respond,
            [box, chatbot, assistant_state],
            [chatbot, assistant_state, box],
        )
        clear.click(lambda: ([], None, ""), None, [chatbot, assistant_state, box])
    return demo


def launch(backend: Backend, host: str = "127.0.0.1", port: int = 7860) -> None:
    build_ui(backend).launch(server_name=host, server_port=port, inbrowser=False)
