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
    assert called == {
        "get_overview",
        "list_de_genes",
        "get_genes",
        "list_lane_sensitive_genes",
        "get_lane_check",
        "get_l2_concordance",
        "show_pca",
        "show_volcano",
    }


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


def test_offline_lane_and_pca_questions_use_the_lane_tools():
    lane = offline.answer("Which genes are sensitive to the lane effect?")
    assert [s.text.split("(")[0] for s in lane.steps if s.kind == "call"] == [
        "get_lane_check",
        "list_lane_sensitive_genes",
    ]
    assert "SPATS2L" in offline.answer("Is SPATS2L DE? Is it a lane artefact?").answer
    assert (
        offline.answer("Which genes are lost when adjusting for lane?")
        .steps[2]
        .text.startswith("list_lane_sensitive_genes(flag='lost'")
    )
    pca = offline.answer("Show me the PCA of L1 vs L2")
    assert [p["subset"] for p in pca.plots] == ["all", "treated"]
    assert "Among the treated samples only" in pca.answer


def test_show_pca_rejects_an_unknown_subset_and_asks_the_model_to_retry():
    model = _scripted(
        ("show_pca", {"subset": "control"}), ("show_pca", {"subset": "treated"})
    )
    reply = Assistant(Backend("openai"), build_agent(model)).ask("PCA of treated")
    assert any("rejected" in s.text for s in reply.steps)
    assert [p["subset"] for p in reply.plots] == ["treated"]


def test_assistant_returns_plot_requests_from_show_pca():
    model = _scripted(("show_pca", {"x": 1, "y": 3}))
    reply = Assistant(Backend("openai"), build_agent(model)).ask("Show a PCA")
    assert reply.plots == [
        {"plot": "pca", "x": 1, "y": 3, "subset": "all", "shown": True}
    ]
    assert reply.steps[1].text == "PCA plot (all samples), PC1 vs PC3"


def test_offline_concordance_question_calls_the_tool_and_reports_the_baseline():
    reply = offline.answer("Are the lane-adjusted genes confirmed in L2?")
    assert [s.text for s in reply.steps if s.kind == "call"] == ["get_l2_concordance()"]
    assert "of 277 genes" in reply.answer and "not DE agree" in reply.answer


def test_offline_volcano_question_returns_a_plot_marking_the_named_genes():
    reply = offline.answer("Show the volcano plot and mark STAT1 and nope123")
    assert reply.plots == [
        {"plot": "volcano", "model": "headline", "genes": ["STAT1"], "shown": True}
    ]
    assert offline.answer("lane adjusted volcano").plots[0]["model"] == "lane_adjusted"


def test_assistant_returns_volcano_plots_and_gene_tables():
    model = _scripted(
        ("show_volcano", {"model": "sideways"}),
        ("show_volcano", {"model": "lane_adjusted", "genes": ["ciita"]}),
        ("list_de_genes", {"direction": "up", "n": 3}),
        ("get_genes", {"names": ["CIITA", "NOPE1"]}),
    )
    reply = Assistant(Backend("openai"), build_agent(model)).ask("volcano")
    assert reply.plots == [
        {"plot": "volcano", "model": "lane_adjusted", "genes": ["CIITA"], "shown": True}
    ]
    assert [t["title"] for t in reply.tables] == [
        "Differentially expressed genes (direction=up, n=3)",
        "Gene lookup",
    ]
    assert [len(t["rows"]) for t in reply.tables] == [3, 1]
    assert offline.answer("top 5 genes").tables[0]["rows"][0]["gene"]
