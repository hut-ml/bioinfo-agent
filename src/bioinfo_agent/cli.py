import typer

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


def main() -> None:
    app()
