from bioinfo_agent.agent import Assistant, Backend
from bioinfo_agent.offline import Reply, Step
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
