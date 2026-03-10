"""CLI interface for ProfileRAG."""

import json
import logging
import sys

import click

from pipeline import RAGPipeline
from monitoring.metrics import metrics


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """ProfileRAG — production RAG over company documents."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s  %(message)s",
    )


@cli.command()
@click.argument("pdf_path")
@click.option("--force", is_flag=True, help="Force re-ingestion even if unchanged")
def ingest(pdf_path: str, force: bool):
    """Ingest a PDF into the vector store."""
    pipe = RAGPipeline()
    count = pipe.ingest(pdf_path, force=force)
    click.echo(f"Done — {count} chunks stored.")


@cli.command()
@click.argument("question")
@click.option("--pdf", default=None, help="PDF to ingest before querying")
@click.option("--top-k", default=5, help="Number of results to retrieve")
def query(question: str, pdf: str | None, top_k: int):
    """Ask a question over ingested documents."""
    pipe = RAGPipeline()

    if pdf:
        pipe.ingest(pdf)

    result = pipe.query(question, top_k=top_k)

    click.echo(f"\nAnswer:\n{result['answer']}\n")
    click.echo("Sources:")
    for src in result["sources"]:
        click.echo(f"  Page {src['page']} (score: {src['score']}): {src['text']}...")
    click.echo(
        f"\nMetrics: {result['metrics']['total_ms']:.0f}ms total, "
        f"${result['metrics']['cost_usd']:.4f}"
    )


@cli.command()
@click.argument("golden_path")
@click.option("--no-judge", is_flag=True, help="Skip LLM-as-judge scoring")
def evaluate(golden_path: str, no_judge: bool):
    """Evaluate against a golden QA dataset."""
    pipe = RAGPipeline()
    results = pipe.evaluate(golden_path, use_judge=not no_judge)

    click.echo("\n── Per-question results ──")
    for r in results["results"]:
        click.echo(
            f"  Q: {r['question'][:80]}\n"
            f"    Hit: {r['hit']}  MRR: {r['mrr']:.2f}  "
            f"Judge: {r['judge_score']}/5"
        )

    click.echo(f"\n── Aggregate ──\n{json.dumps(results['aggregate'], indent=2)}")


@cli.command()
@click.option("--last", default=50, help="Number of recent queries to aggregate")
def stats(last: int):
    """Show aggregated monitoring metrics."""
    summary = metrics.summary(last_n=last)
    if not summary:
        click.echo("No metrics recorded yet.")
        return
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
def serve(host: str, port: int):
    """Start the FastAPI server."""
    import uvicorn
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run("api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
