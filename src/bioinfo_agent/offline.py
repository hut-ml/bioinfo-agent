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
    plots: list[dict] = field(
        default_factory=list
    )  # e.g. {"plot": "pca", "x": 1, "y": 2}


# Words that look like gene symbols once upper-cased but are almost always plain English.
_STOPWORDS = {
    "a", "all", "an", "and", "are", "as", "at", "be", "by", "can", "de", "do", "does", "down",
    "for", "from", "gene", "genes", "get", "how", "i", "if", "in", "is", "it", "me", "of",
    "on", "or", "show", "the", "to", "top", "up", "was", "what", "which", "with", "you",
}  # fmt: skip
# Upper-case abbreviations a biologist might type that are not genes.
_ABBREVIATIONS = {
    "DE", "DEG", "DEGS", "RNA", "DNA", "QC", "PCA", "GSEA", "NA", "OK", "L1", "L2",
}  # fmt: skip
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


def _wants_pca(q: str) -> bool:
    return bool(re.search(r"\bpca\b|principal component|\bplot\b|cluster", q))


def _wants_concordance(q: str) -> bool:
    return bool(
        re.search(
            r"concordan|confirm|replicat|independent|robust|validat|same direction", q
        )
    )


def _wants_lane(q: str) -> bool:
    return bool(
        re.search(
            r"\blanes?\b|\bl1\b|\bl2\b|batch|confound|artefact|artifact|sensitiv", q
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
        "- Ask about the lane confounding, e.g. 'Which genes are lane sensitive?' or "
        "'Show the PCA' or 'Do the lane-adjusted genes replicate in L2?'.\n"
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


def lane_summary(lc: dict) -> list[str]:
    """Plain-language lines built only from the lane_check numbers."""
    s, flags = lc["samples"], lc["flags"]
    n_up, n_down = lc["n_up_lane_adjusted"], lc["n_down_lane_adjusted"]
    lines = [
        (
            f"Lane is confounded with condition: control-L1 {s['control-L1']}, treated-L1 "
            f"{s['treated-L1']}, treated-L2 {s['treated-L2']} samples (no control sample is in L2)."
        ),
        (
            f"Adding lane to the model leaves {n_up + n_down} DE genes ({n_up} up, {n_down} down). "
            f"{flags.get('lost', 0)} genes lose their call, {flags.get('gained', 0)} gain one and "
            f"{flags.get('flipped', 0)} flip direction."
        ),
        (
            f"The two treated lanes differ at padj < 0.05 for {lc['n_lane_effect_padj']} genes "
            f"({lc['n_lane_effect_de']} with |log2FC| >= 1), although every sample in both is treated."
        ),
    ]
    pca = lc["pca"]
    means = pca["group_means"]
    lines.append(
        f"PC1 ({pca['variance_explained'][0]:.0%} of the variance) puts control-L1 at "
        f"{means['control-L1']['PC1']:+.1f}, treated-L1 at {means['treated-L1']['PC1']:+.1f} and "
        f"treated-L2 at {means['treated-L2']['PC1']:+.1f}. Treated vs control (both L1): "
        f"p = {pca['condition_effect_p']['PC1']:.1g}; L2 vs L1 (both treated): "
        f"p = {pca['lane_effect_p']['PC1']:.1g}."
    )
    treated = pca["treated_only"]
    lane_pcs = sorted((p, pc) for pc, p in treated["lane_effect_p"].items() if p < 0.05)
    if lane_pcs:
        shown = ", ".join(f"{pc} (p = {p:.1g})" for p, pc in lane_pcs)
        lines.append(
            "Among the treated samples only, where treatment is held fixed, L1 and L2 "
            f"separate on {shown}."
        )
    else:
        lines.append(
            "Among the treated samples only, L1 and L2 do not separate on any of the first "
            f"{len(treated['lane_effect_p'])} PCs."
        )
    if pca["lane_effect_p"]["PC1"] < 0.05:
        lines.append(
            "So lane shifts samples along the same axis as the treatment: the confounding is "
            "real, and calls for genes marked 'lost' should be treated with caution."
        )
    return lines


def concordance_summary(c: dict) -> list[str]:
    """Plain-language lines built only from the l2_concordance numbers."""
    de, up, down = c["lane_adjusted_de"], c["up"], c["down"]
    sig, base = c["significant_in_l2"], c["baseline_not_de"]
    return [
        (
            f"Directional check in treated-L2 vs control-L1 (the lane-adjusted DE genes come from "
            f"L1 only): {de['n_same_direction']} of {de['n_genes']} genes ({de['fraction']:.1%}) "
            f"move the same way; up {up['fraction']:.1%}, down {down['fraction']:.1%}. Among the "
            f"{sig['n_genes']} also significant in L2, {sig['fraction']:.1%} agree."
        ),
        (
            f"For comparison, genes that are not DE agree {base['fraction']:.1%} of the time "
            "(above 50% because of the lane shift), so the excess over that baseline is what "
            "counts. Control-L1 samples are shared between the two comparisons, so this supports "
            "the genes but is not a fully independent replicate."
        ),
    ]


def _concordance_answer() -> Reply:
    c = tools.get_l2_concordance()
    steps = [
        Step("call", format_call("get_l2_concordance", {})),
        Step("result", summarise_result("get_l2_concordance", c)),
    ]
    return Reply("\n".join(concordance_summary(c)), steps)


def _lane_answer(q: str) -> Reply:
    lc = tools.get_lane_check()
    steps = [
        Step("call", format_call("get_lane_check", {})),
        Step("result", summarise_result("get_lane_check", lc)),
    ]
    flag = next(
        (f for f in ("lost", "gained", "flipped") if re.search(rf"\b{f}\b", q)), "all"
    )
    hits = tools.list_lane_sensitive_genes(flag, DEFAULT_TOP_N)
    steps += [
        Step(
            "call",
            format_call(
                "list_lane_sensitive_genes", {"flag": flag, "n": DEFAULT_TOP_N}
            ),
        ),
        Step("result", summarise_result("list_lane_sensitive_genes", hits)),
    ]
    rows = [
        f"{g['gene']:<14}{g['lane_flag']:<9}"
        f"headline {g['log2fc']:+.2f} (padj {g['padj']:.2g}), "
        f"lane-adjusted {g['log2fc_lane_adjusted']:+.2f} (padj {g['padj_lane_adjusted']:.2g})"
        for g in hits
    ]
    text = [
        *lane_summary(lc),
        *concordance_summary(lc["l2_concordance"]),
        "",
        f"Top {len(hits)} genes whose call changes with lane:",
        *rows,
    ]
    return Reply("\n".join(text), steps)


def _pca_answer(q: str) -> Reply:
    plots = [tools.show_pca(subset="all"), tools.show_pca(subset="treated")]
    lc = tools.get_lane_check()
    steps = []
    for plot in plots:
        steps += [
            Step("call", format_call("show_pca", {"subset": plot["subset"]})),
            Step("result", summarise_result("show_pca", plot)),
        ]
    steps += [
        Step("call", format_call("get_lane_check", {})),
        Step("result", summarise_result("get_lane_check", lc)),
    ]
    text = [
        (
            "PCA of all samples (left: coloured by condition, right: by lane), then of the "
            "treated samples only (coloured by lane)."
        ),
        *lane_summary(lc),
    ]
    return Reply("\n".join(text), steps, plots=plots)


def answer(question: str) -> Reply:
    """Answer the three supported kinds of question, or explain what is supported."""
    q = question.lower()
    table, _ = queries.load_results()
    names = _gene_candidates(question, set(table.index.str.upper()))
    if names:
        return _gene_answer(names)
    if _wants_pca(q):
        return _pca_answer(q)
    if _wants_concordance(q):
        return _concordance_answer()
    if _wants_lane(q):
        return _lane_answer(q)
    if _wants_help(q):
        return Reply(_help_text())
    if _wants_gene_list(q):
        return _list_answer(q)
    return Reply(
        "I did not understand that in offline mode.\n\n" + _help_text(),
    )
