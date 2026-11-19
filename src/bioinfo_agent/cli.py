from typing import Annotated

import typer

from bioinfo_agent import queries
from bioinfo_agent.data import cohort_table, load_dataset

app = typer.Typer(
    help="Ask questions about a differential expression analysis.", no_args_is_help=True
)


@app.callback()
def _root() -> None:
    """Ask questions about a differential expression analysis."""


@app.command()
def info() -> None:
    """Summarise the cohort: genes, samples, samples per condition."""
    ds = load_dataset()
    typer.echo(f"{ds.n_genes} genes x {ds.n_samples} samples\n")
    typer.echo(cohort_table(ds.metadata).to_string())


@app.command()
def precompute() -> None:
    """Run the differential expression analysis and write results/ (takes about a minute)."""
    from bioinfo_agent.analysis import run_de, save_results

    typer.echo("Running DESeq2 (~ condition)...")
    table, summary = run_de(load_dataset())
    directory = save_results(table, summary)
    typer.echo(
        f"{summary['n_genes_tested']} genes tested: {summary['n_up']} up, "
        f"{summary['n_down']} down."
    )
    typer.echo(f"Wrote {directory}")


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"Error: {exc}", err=True)
    return typer.Exit(1)


@app.command()
def genes(
    direction: Annotated[
        str, typer.Option(help="all, up or down (in treated)")
    ] = "all",
    n: Annotated[int, typer.Option("-n", help="Number of genes to show")] = 10,
    sort_by: Annotated[str, typer.Option(help="padj, log2fc or base_mean")] = "padj",
) -> None:
    """Show the analysis overview and the top differentially expressed genes."""
    try:
        info_ = queries.overview()
        hits = queries.top_genes(direction, n, sort_by)
    except (queries.ResultsNotFound, ValueError) as exc:
        raise _fail(exc) from exc
    t = info_["thresholds"]
    typer.echo(f"Design: {info_['design']}; {info_['comparison']}")
    typer.echo(
        f"Significant = padj < {t['padj']:g} and |log2FC| >= {t['abs_log2fc']:g}: "
        f"{info_['n_up']} up, {info_['n_down']} down "
        f"({info_['n_genes_tested']} genes tested)\n"
    )
    typer.echo(f"{'gene':<12}{'log2FC':>8}{'padj':>11}{'dir':>6}")
    for g in hits:
        typer.echo(
            f"{g['gene']:<12}{g['log2fc']:>+8.2f}{g['padj']:>11.2g}{g['direction']:>6}"
        )


@app.command()
def gene(names: Annotated[list[str], typer.Argument(help="Gene symbols")]) -> None:
    """Look up one or more genes, e.g. `gene STAT1 CIITA`."""
    try:
        found = queries.get_genes(names)
    except queries.ResultsNotFound as exc:
        raise _fail(exc) from exc
    for g in found:
        label = g["gene"] if g["found"] else g["query"]
        typer.echo(f"{label}: {g['verdict']}")
        if g.get("did_you_mean"):
            typer.echo(f"  Did you mean: {', '.join(g['did_you_mean'])}?")


def main() -> None:
    app()
