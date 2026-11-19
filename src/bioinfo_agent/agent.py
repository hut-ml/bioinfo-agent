"""The agent: an LLM (hosted or local, via the OpenAI API) that answers by calling tools over the
precomputed results, with an offline keyword router as fallback."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.messages import (
    ModelMessage,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from bioinfo_agent import offline, tools
from bioinfo_agent.reply import Reply, Step

DEFAULT_OPENAI_MODEL = "gpt-6.1-terra"
DEFAULT_COMPATIBLE_MODEL = "default"

SYSTEM_PROMPT = f"""\
You help biologists explore a bulk RNA-seq experiment comparing treated and control samples.
The differential expression (DE) analysis is precomputed; you can only read its results through tools.

Rules:
- Every number (log2 fold change, adjusted p-value, counts of genes) must come from a tool result. \
Never estimate, recall or invent values. If a tool cannot answer, say so.
- For any question about genes, call a tool first. Call get_overview first when you need the \
thresholds, the design or the number of DE genes.
- State the call rule when you report DE genes: padj < threshold and |log2FC| >= threshold, \
taken from get_overview. Positive log2FC means higher in treated.
- A gene that is not significant is "not differentially expressed in this dataset", not proof of \
no biology. If a gene is not found, say so and offer the suggestions the tool returned.
- {offline.LANE_CAVEAT} Mention it when listing DE genes or when a call is borderline.
- Each gene result has a lane_flag. For 'lost', 'gained' or 'flipped' genes, tell the user the call \
changes when lane is added to the model and quote the lane-adjusted numbers from the tool. Never say \
a gene is a lane artefact as a fact: say it may be one. Use list_lane_sensitive_genes and \
get_lane_check for questions about lane, batch or confounding, and get_l2_concordance to \
ask whether the lane-adjusted genes are confirmed in L2 (always give the baseline next to the \
concordance; it is supporting evidence, not an independent replicate).
- For a PCA or a request to see the samples, call show_pca, then describe it using get_lane_check \
numbers only. For a volcano plot or a request to see which genes are up or down regulated overall, \
call show_volcano (optionally marking genes), then describe it using get_overview or list_de_genes \
numbers only. Plots and gene tables appear in a side panel next to the chat, not in the message.
- You cannot run new analyses or look up pathways yet. Say so if asked.
- Write for a biologist: short, plain language, gene symbols as given by the tool. Don't show \
raw JSON.
"""


@dataclass(frozen=True)
class Backend:
    kind: Literal["openai", "compatible", "offline"]
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    reasoning: str = "off"  # OpenAI only: "off" or an effort level

    @property
    def label(self) -> str:
        if self.kind == "offline":
            return "offline (keyword matching, no LLM)"
        where = f" at {self.base_url}" if self.kind == "compatible" else ""
        return f"{self.kind}: {self.model}{where}"


def resolve_backend(
    preference: str = "auto", env: Mapping[str, str] | None = None
) -> Backend:
    """Pick a backend from the environment.

    auto: an OpenAI-compatible server (local or hosted) if LLM_BASE_URL is set, else OpenAI if
    OPENAI_API_KEY is set, else offline. An explicit preference raises ValueError if its settings
    are missing.
    """
    env = os.environ if env is None else env

    def get(name: str) -> str | None:
        return env.get(name, "").strip() or None

    if preference not in ("auto", "openai", "compatible", "offline"):
        raise ValueError("backend must be auto, openai, compatible or offline")
    openai_key, base_url = get("OPENAI_API_KEY"), get("LLM_BASE_URL")
    if preference == "auto":
        preference = "compatible" if base_url else "openai" if openai_key else "offline"

    if preference == "compatible":
        if not base_url:
            raise ValueError(
                "Set LLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama)."
            )
        return Backend(
            "compatible",
            model=get("LLM_MODEL") or DEFAULT_COMPATIBLE_MODEL,
            base_url=base_url,
            api_key=get("LLM_API_KEY") or "not-needed",
        )
    if preference == "openai":
        if not openai_key:
            raise ValueError("Set OPENAI_API_KEY.")
        return Backend(
            "openai",
            model=get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
            api_key=openai_key,
            reasoning=(get("OPENAI_REASONING") or "low").lower(),
        )
    return Backend("offline")


def _model_and_settings(backend: Backend) -> tuple[Model, dict]:
    if backend.kind == "openai":
        model = OpenAIResponsesModel(
            backend.model, provider=OpenAIProvider(api_key=backend.api_key)
        )
        if backend.reasoning == "off":
            return model, {}
        settings = OpenAIResponsesModelSettings(
            openai_reasoning_effort=backend.reasoning, openai_reasoning_summary="auto"
        )
        return model, settings
    # Compatible servers (Ollama, llamafile, hosted gateways, ...) speak the Chat Completions dialect.
    provider = OpenAIProvider(base_url=backend.base_url, api_key=backend.api_key)
    # thinking=False: a gateway may route to an OpenAI reasoning model, which rejects tools over Chat Completions.
    return OpenAIChatModel(backend.model, provider=provider), {
        "temperature": 0,
        "thinking": False,
    }


def build_agent(model: Model | str, settings: dict | None = None) -> Agent:
    return Agent(
        model,
        instructions=SYSTEM_PROMPT,
        toolsets=[tools.toolset],
        model_settings=settings or None,
    )


def _steps(messages: list[ModelMessage]) -> list[Step]:
    steps = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ThinkingPart) and part.content.strip():
                steps.append(Step("thinking", part.content.strip()))
            elif isinstance(part, ToolCallPart):
                steps.append(
                    Step("call", tools.format_call(part.tool_name, part.args_as_dict()))
                )
            elif isinstance(part, ToolReturnPart):
                if part.outcome == "success":
                    text = tools.summarise_result(part.tool_name, part.content)
                else:
                    text = f"rejected, retrying: {str(part.content)[:100]}"
                steps.append(Step("result", text))
    return steps


def _plots(messages: list[ModelMessage]) -> list[dict]:
    """Plot requests returned by show_pca and show_volcano; the front end draws them."""
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name in tools.PLOT_TOOLS
        and part.outcome == "success"
    ]


def _tables(messages: list[ModelMessage]) -> list[dict]:
    """Gene tables for the side panel, from the gene-list tool results."""
    args: dict[str, dict] = {}
    tables = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                args[part.tool_call_id] = part.args_as_dict()
            elif isinstance(part, ToolReturnPart) and part.outcome == "success":
                table = (
                    tools.result_table(
                        part.tool_name, args.get(part.tool_call_id, {}), part.content
                    )
                    if part.tool_name in tools.TABLE_TOOLS
                    else None
                )
                if table:
                    tables.append(table)
    return tables


class Assistant:
    """Holds a backend and the conversation. Falls back to offline mode if the LLM call fails."""

    def __init__(self, backend: Backend | None = None, agent: Agent | None = None):
        self.backend = backend or resolve_backend()
        self._history: list[ModelMessage] = []
        if agent is not None:
            self._agent = agent
        elif self.backend.kind == "offline":
            self._agent = None
        else:
            self._agent = build_agent(*_model_and_settings(self.backend))

    def ask(self, question: str) -> Reply:
        if self._agent is None:
            return offline.answer(question)
        try:
            result = self._agent.run_sync(question, message_history=self._history)
        except AgentRunError as exc:
            reply = offline.answer(question)
            reply.notice = (
                f"The LLM call failed ({_describe(exc)}); answered in offline mode."
            )
            return reply
        self._history = result.all_messages()
        new = result.new_messages()
        return Reply(result.output, _steps(new), plots=_plots(new), tables=_tables(new))


def _describe(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:200]
