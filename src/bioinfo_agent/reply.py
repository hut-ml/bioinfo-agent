"""What an answer looks like to a front end (CLI or UI)."""

from dataclasses import dataclass, field


@dataclass
class Step:
    """Something shown while answering: a model summary, a tool call, or a tool result."""

    kind: str  # "thinking", "call" or "result"
    text: str


@dataclass
class Reply:
    answer: str
    steps: list[Step] = field(default_factory=list)
    notice: str | None = None
    plots: list[dict] = field(
        default_factory=list
    )  # e.g. {"plot": "pca", "x": 1, "y": 2}
    tables: list[dict] = field(default_factory=list)  # {"title": ..., "rows": [...]}
