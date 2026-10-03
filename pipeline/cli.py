"""CLI entrypoints for the pipeline (called by GitHub Actions and lll-* scripts)."""

from __future__ import annotations

import datetime
from typing import Optional

import typer

app = typer.Typer(name="leadlaglab", add_completion=False)


@app.command("ingest")
def _ingest(
    date: Optional[str] = typer.Option(
        None, "--date", help="Date to ingest (YYYY-MM-DD). Defaults to today."
    ),
) -> None:
    """Run all data source collectors for a given date."""
    target = date or datetime.date.today().isoformat()
    typer.echo(f"[ingest] Running for {target} (not yet implemented — M3)")


@app.command("features")
def _features(
    date: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """Construct signal features from raw archive."""
    target = date or datetime.date.today().isoformat()
    typer.echo(f"[features] Running for {target} (not yet implemented — M4)")


@app.command("evaluate")
def _evaluate(
    as_of: Optional[str] = typer.Option(None, "--as-of"),
) -> None:
    """Run signal evaluation (IC, quintiles, Fama-MacBeth)."""
    typer.echo("[evaluate] Not yet implemented — M5")


@app.command("predict")
def _predict(
    date: Optional[str] = typer.Option(None, "--date"),
) -> None:
    """Generate and commit today's frozen predictions."""
    target = date or datetime.date.today().isoformat()
    typer.echo(f"[predict] Running for {target} (not yet implemented — M6)")


@app.command("build-site")
def _build_site() -> None:
    """Build site/public/data JSON from processed data."""
    typer.echo("[build-site] Not yet implemented — M7")


# pyproject.toml [project.scripts] entrypoints — each calls the app with its subcommand
def ingest() -> None:
    app(["ingest"], standalone_mode=True)


def features() -> None:
    app(["features"], standalone_mode=True)


def evaluate() -> None:
    app(["evaluate"], standalone_mode=True)


def predict() -> None:
    app(["predict"], standalone_mode=True)


def build_site() -> None:
    app(["build-site"], standalone_mode=True)
