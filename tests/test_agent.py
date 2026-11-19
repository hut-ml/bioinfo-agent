import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from bioinfo_agent import offline
from bioinfo_agent.agent import (
    Assistant,
    Backend,
    _model_and_settings,
    build_agent,
    resolve_backend,
)


def test_backend_auto_prefers_compatible_then_openai_then_offline():
    assert resolve_backend("auto", {}).kind == "offline"
    assert resolve_backend("auto", {"OPENAI_API_KEY": "  "}).kind == "offline"
    assert resolve_backend("auto", {"OPENAI_API_KEY": "sk-x"}).kind == "openai"
    both = {"OPENAI_API_KEY": "sk-x", "LLM_BASE_URL": "http://localhost:11434/v1"}
    compatible = resolve_backend("auto", both)
    assert compatible.kind == "compatible" and compatible.api_key == "not-needed"
    assert resolve_backend("offline", both).kind == "offline"


def test_backend_explicit_choice_needs_its_settings():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        resolve_backend("openai", {})
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve_backend("compatible", {})
    with pytest.raises(ValueError):
        resolve_backend("nonsense", {})


def test_every_tool_runs_against_the_results():
    # TestModel calls each tool once with generated arguments (invalid ones are retried by the tool).
    agent = build_agent(TestModel(custom_output_text="done"))
    result = agent.run_sync("hello")
    called = {
        p.tool_name
        for m in result.all_messages()
        for p in m.parts
        if p.part_kind == "tool-call"
    }
    assert called == {"get_overview", "list_de_genes", "get_genes"}


def _scripted(*calls):
    """A fake model that makes the given tool calls, then answers with the last tool result."""
    pending = list(calls)

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if pending:
            name, args = pending.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("answer")])

    return FunctionModel(respond)


def test_assistant_reports_tool_calls_and_keeps_history():
    model = _scripted(("get_genes", {"names": ["ciita"]}))
    assistant = Assistant(Backend("openai"), build_agent(model))
    reply = assistant.ask("Is CIITA DE?")
    assert reply.answer == "answer"
    kinds = [s.kind for s in reply.steps]
    assert kinds == ["call", "result"]
    assert "CIITA: up" in reply.steps[1].text
    assert assistant._history


def test_invalid_tool_arguments_are_sent_back_to_the_model():
    model = _scripted(
        ("list_de_genes", {"direction": "sideways"}),
        ("list_de_genes", {"direction": "up", "n": 3}),
    )
    reply = Assistant(Backend("openai"), build_agent(model)).ask("top up genes")
    assert reply.answer == "answer"
    assert [s.kind for s in reply.steps].count("call") == 2
    assert any("rejected" in s.text for s in reply.steps)


def test_llm_failure_falls_back_to_offline_with_a_notice():
    def boom(messages, info):
        raise ModelHTTPError(401, "gpt-test", {"error": "bad key"})

    reply = Assistant(Backend("openai"), build_agent(FunctionModel(boom))).ask(
        "Is STAT1 DE?"
    )
    assert reply.notice and "offline mode" in reply.notice
    assert "STAT1: not differentially expressed" in reply.answer


def test_offline_answers_the_three_required_questions():
    assistant = Assistant(Backend("offline"))
    helped = assistant.ask("How can you help me?")
    assert "differentially expressed" in helped.answer and not helped.steps

    listing = assistant.ask(
        "Which genes are differentially expressed between treated and control samples?"
    )
    assert "322 genes" in listing.answer and offline.LANE_CAVEAT in listing.answer
    assert listing.steps[0].text.startswith("list_de_genes")

    genes = assistant.ask(
        "Is gene CIITA, SPATS2L, IDH3B or stat1 differentially expressed?"
    ).answer
    assert "CIITA: differentially expressed: higher in treated" in genes
    assert "IDH3B: not differentially expressed" in genes
    assert "STAT1: not differentially expressed" in genes


def test_offline_gene_detection_ignores_plain_words():
    # "was" is a gene symbol (WAS) but only counts when written as one.
    assert offline.answer("What was the effect of treatment?").steps == []
    assert (
        offline.answer("Is WAS differentially expressed?").steps[0].text
        == "get_genes(names=['WAS'])"
    )
    assert "Did you mean" in offline.answer("Is STTA1 differentially expressed?").answer
    top = offline.answer("Show the top 3 down genes")
    assert top.steps[0].text == "list_de_genes(direction='down', n=3, sort_by='padj')"


def test_reasoning_summary_is_shown_before_the_tool_call():
    responses = iter(
        [
            ModelResponse(
                parts=[
                    ThinkingPart("Checking whether CIITA passes the thresholds."),
                    ToolCallPart("get_genes", {"names": ["CIITA"]}),
                ]
            ),
            ModelResponse(parts=[TextPart("answer")]),
        ]
    )
    model = FunctionModel(lambda messages, info: next(responses))
    reply = Assistant(Backend("openai"), build_agent(model)).ask("Is CIITA DE?")
    assert [(s.kind, s.text) for s in reply.steps[:2]] == [
        ("thinking", "Checking whether CIITA passes the thresholds."),
        ("call", "get_genes(names=['CIITA'])"),
    ]


def test_openai_reasoning_settings_come_from_the_environment():
    env = {"OPENAI_API_KEY": "sk-x"}
    assert resolve_backend("openai", env).reasoning == "low"
    high = resolve_backend("openai", {**env, "OPENAI_REASONING": "HIGH"})
    _, settings = _model_and_settings(high)
    assert settings["openai_reasoning_effort"] == "high"
    assert settings["openai_reasoning_summary"] == "auto"

    off = resolve_backend("openai", {**env, "OPENAI_REASONING": "off"})
    assert _model_and_settings(off)[1] == {}
