"""Offline mode: a keyword router over the same tools the LLM uses. No model, no network."""

import re
from dataclasses import dataclass, field

from bioinfo_agent import queries, tools
from bioinfo_agent.tools import format_call, summarise_result

DEFAULT_TOP_N = 10

# Lane is not in the model; every L2 sample is treated.
LANE_CAVEAT = (
    "Caveat: the sample ID suffix (sequencing lane) is partly confounded with condition "
    "(every L2 sample is treated) and is not modelled, so treat borderline calls with care."
)


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


# Words that look like gene symbols once upper-cased but are almost always plain English.
_STOPWORDS = {
    "a", "all", "an", "and", "are", "as", "at", "be", "by", "can", "de", "do", "does", "down",
    "for", "from", "gene", "genes", "get", "how", "i", "if", "in", "is", "it", "me", "of",
    "on", "or", "show", "the", "to", "top", "up", "was", "what", "which", "with", "you",
}  # fmt: skip
# Upper-case abbreviations a biologist might type that are not genes.
_ABBREVIATIONS = {"DE", "DEG", "DEGS", "RNA", "DNA", "QC", "PCA", "GSEA", "NA", "OK"}
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]+$")


def _gene_candidates(question: str, known: set[str]) -> list[str]:
    """Tokens that look like gene symbols: upper-case, or a known gene that isn't an English word."""
    found: list[str] = []
    for token in re.findall(r"[A-Za-z0-9-]+", question):
        looks_like_symbol = bool(_SYMBOL.match(token)) and token not in _ABBREVIATIONS
        is_known_gene = token.upper() in known and token.lower() not in _STOPWORDS
        if (looks_like_symbol or is_known_gene) and token.upper() not in found:
            found.append(token.upper())
    return found


def _wants_help(q: str) -> bool:
    return bool(
        re.search(
            r"\bhelp\b|what can you|what do you do|how can you|what are you|\bhello\b|^hi\b",
            q,
        )
    )


def _wants_gene_list(q: str) -> bool:
    return bool(
        re.search(
            r"differentially|\bde\b|\bdegs?\b|top\b|significant|up-?regulated|down-?regulated|"
            r"\bup\b|\bdown\b|how many|which genes|list .*genes|most (changed|different)",
            q,
        )
    )


def _help_text() -> str:
    t = queries.overview()["thresholds"]
    return (
        "I can answer questions about the precomputed control vs treated RNA-seq comparison "
        "(DESeq2, shrunken log2 fold changes).\n"
        "- Ask which genes are differentially expressed, e.g. 'Which genes are upregulated?' "
        "or 'Show the top 20 down genes'.\n"
        "- Ask about specific genes, e.g. 'Is STAT1 differentially expressed?' or "
        "'CIITA and IDH3B'.\n"
        f"A gene counts as differentially expressed when padj < {t['padj']:g} and "
        f"|log2FC| >= {t['abs_log2fc']:g}.\n"
        "I am running in offline mode (keyword matching only). Set an LLM backend in .env for "
        "free-form questions; see the README."
    )


def _gene_answer(names: list[str]) -> Reply:
    found = tools.get_genes(names)
    steps = [
        Step("call", format_call("get_genes", {"names": names})),
        Step("result", summarise_result("get_genes", found)),
    ]
    lines = []
    for g in found:
        label = g["gene"] if g["found"] else g["query"]
        lines.append(f"{label}: {g['verdict']}")
        if g.get("did_you_mean"):
            lines.append(f"  Did you mean: {', '.join(g['did_you_mean'])}?")
    return Reply("\n".join(lines), steps)


def _list_answer(q: str) -> Reply:
    ov = queries.overview()
    t = ov["thresholds"]
    if re.search(r"down|lower|decreas|under-?expressed", q):
        direction = "down"
    elif re.search(r"\bup\b|up-?regulated|higher|increas|over-?expressed", q):
        direction = "up"
    else:
        direction = "all"
    sort_by = (
        "log2fc" if re.search(r"fold|log2|biggest|largest|strongest", q) else "padj"
    )
    match = re.search(r"\b(\d{1,3})\b", q)
    n = int(match.group(1)) if match and int(match.group(1)) > 0 else DEFAULT_TOP_N
    hits = tools.list_de_genes(direction, n, sort_by)
    args = {"direction": direction, "n": n, "sort_by": sort_by}
    steps = [
        Step("call", format_call("list_de_genes", args)),
        Step("result", summarise_result("list_de_genes", hits)),
    ]
    total = {"up": ov["n_up"], "down": ov["n_down"], "all": ov["n_up"] + ov["n_down"]}[
        direction
    ]
    header = (
        f"{total} genes are differentially expressed ({direction if direction != 'all' else 'up or down'}"
        f" in treated; padj < {t['padj']:g} and |log2FC| >= {t['abs_log2fc']:g}). "
        f"Top {len(hits)} by {'|log2FC|' if sort_by == 'log2fc' else 'adjusted p-value'}:"
    )
    rows = [f"{'gene':<12}{'log2FC':>8}{'padj':>11}{'dir':>6}"]
    rows += [
        f"{g['gene']:<12}{g['log2fc']:>+8.2f}{g['padj']:>11.2g}{g['direction']:>6}"
        for g in hits
    ]
    return Reply("\n".join([header, *rows, "", LANE_CAVEAT]), steps)


def answer(question: str) -> Reply:
    """Answer the three supported kinds of question, or explain what is supported."""
    q = question.lower()
    table, _ = queries.load_results()
    names = _gene_candidates(question, set(table.index.str.upper()))
    if names:
        return _gene_answer(names)
    if _wants_help(q):
        return Reply(_help_text())
    if _wants_gene_list(q):
        return _list_answer(q)
    return Reply(
        "I did not understand that in offline mode.\n\n" + _help_text(),
    )
