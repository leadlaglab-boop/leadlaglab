"""CLI entrypoints for the pipeline (called by GitHub Actions and lll-* scripts)."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(name="leadlaglab", add_completion=False)

DEFAULT_DATA_REPO = Path(os.getenv("DATA_REPO_PATH", "../leadlaglab-data"))
DEFAULT_SITE_DATA = Path("site/public/data")
DEFAULT_CONFIG_PATH = Path("config/universe_retail.yaml")
DEFAULT_BACKFILL_START = "2020-01-01"


@app.command("universe")
def _universe(
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config"),
    as_of: str | None = typer.Option(None, "--as-of", help="YYYY-MM-DD"),
) -> None:
    """Build the SecurityMaster from Wikipedia + retail basket config."""
    from pipeline.universe.security_master import build_security_master

    target = datetime.date.fromisoformat(as_of) if as_of else None
    df = build_security_master(
        data_repo_path=data_repo.resolve(),
        config_path=config.resolve(),
        as_of=target,
    )
    typer.echo(f"SecurityMaster: {len(df)} rows written to {data_repo}/processed/security_master/")


@app.command("prices")
def _prices(
    tickers: str | None = typer.Option(
        None, "--tickers", help="Comma-separated list; defaults to all active S&P 500"
    ),
    start: str = typer.Option(DEFAULT_BACKFILL_START, "--start", help="YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="YYYY-MM-DD; defaults to today"),
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    live: bool = typer.Option(False, "--live", help="Mark as live data (default: backfilled)"),
) -> None:
    """Fetch and archive price data for the universe."""
    from pipeline.sources.base import DateRange
    from pipeline.sources.prices import PriceIngestor
    from pipeline.universe.security_master import get_active_tickers, load_security_master

    data_type = "live" if live else "backfilled"
    end_date = datetime.date.fromisoformat(end) if end else datetime.date.today()
    start_date = datetime.date.fromisoformat(start)
    dr = DateRange(start=start_date, end=end_date)

    sm = load_security_master(data_repo.resolve())

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = get_active_tickers(sm)

    # Build (ticker, security_id) pairs
    active = sm[sm["valid_to"].isna() & sm["ticker"].isin(ticker_list)]
    pairs = list(zip(active["ticker"], active["security_id"], strict=False))

    typer.echo(f"Fetching prices for {len(pairs)} tickers ({start_date} – {end_date})")

    ingestor = PriceIngestor(
        data_repo_path=data_repo.resolve(),
        tiingo_api_key=os.getenv("TIINGO_API_KEY"),
    )
    results = ingestor.batch_fetch(pairs, dr, data_type=data_type)

    failed = [t for t, n in results.items() if n < 0]
    if failed:
        typer.echo(f"WARNING: {len(failed)} tickers failed: {failed[:10]}", err=True)
    typer.echo(f"Done. {sum(v for v in results.values() if v > 0):,} rows written.")


@app.command("ingest")
def _ingest(
    start: str | None = typer.Option(
        None, "--start", help="Start date YYYY-MM-DD. Defaults to today."
    ),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD. Defaults to start."),
    sources: str | None = typer.Option(
        None, "--sources", help="Comma-separated: wikipedia,edgar,gdelt,trends. Defaults to all."
    ),
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    config_dir: Path = typer.Option(Path("config"), "--config-dir"),
    max_tickers: int | None = typer.Option(
        None, "--max-tickers", help="Limit to N tickers (for validation runs)."
    ),
    live: bool = typer.Option(False, "--live", help="Tag as live data (default: backfilled)."),
    no_finbert: bool = typer.Option(False, "--no-finbert", help="Skip FinBERT NLP (VADER only)."),
) -> None:
    """Run all signal source collectors for a date range (M3)."""
    from pipeline.signals.ingestor import SignalIngestor
    from pipeline.sources.base import DateRange

    start_date = datetime.date.fromisoformat(start) if start else datetime.date.today()
    end_date = datetime.date.fromisoformat(end) if end else start_date
    dr = DateRange(start=start_date, end=end_date)
    data_type = "live" if live else "backfilled"
    source_list = [s.strip() for s in sources.split(",")] if sources else None

    typer.echo(
        f"[ingest] {start_date} – {end_date}, sources={source_list or 'all'}, "
        f"data_type={data_type}"
    )

    ingestor = SignalIngestor(
        data_repo_path=data_repo.resolve(),
        config_dir=config_dir.resolve(),
        use_finbert=not no_finbert,
    )
    summary = ingestor.run(
        date_range=dr,
        data_type=data_type,
        sources=source_list,
        max_tickers=max_tickers,
    )
    typer.echo(f"[ingest] Done: {summary}")


@app.command("features")
def _features(
    start: str | None = typer.Option(
        None, "--start", help="Start date YYYY-MM-DD. Defaults to today."
    ),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD. Defaults to start."),
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    lookback: int = typer.Option(60, "--lookback", help="Rolling window in calendar days."),
) -> None:
    """Construct signal features from raw archive (M4)."""
    from pipeline.features.builder import FeatureBuilder

    start_date = datetime.date.fromisoformat(start) if start else datetime.date.today()
    end_date = datetime.date.fromisoformat(end) if end else start_date

    typer.echo(f"[features] Building {start_date} – {end_date}, lookback={lookback}d")
    builder = FeatureBuilder(data_repo_path=data_repo.resolve(), lookback_days=lookback)
    summary = builder.build_range(start_date, end_date)
    total = sum(summary.values())
    typer.echo(f"[features] Done. {total:,} feature rows written.")


@app.command("evaluate")
def _evaluate(
    start: str = typer.Option(DEFAULT_BACKFILL_START, "--start", help="Eval start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, "--end", help="Eval end date YYYY-MM-DD"),
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    fdr_q: float = typer.Option(0.05, "--fdr-q", help="BH FDR threshold"),
) -> None:
    """Run signal evaluation: IC, quintiles, Fama-MacBeth, walk-forward OOS, BH FDR."""
    from pipeline.evaluation.engine import run_evaluation

    start_date = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end) if end else datetime.date.today()
    out_dir = run_evaluation(
        data_repo=data_repo.resolve(),
        start=start_date,
        end=end_date,
        fdr_q=fdr_q,
    )
    typer.echo(f"[evaluate] Results written to {out_dir}")


@app.command("predict")
def _predict(
    date: str | None = typer.Option(None, "--date"),
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    score_only: bool = typer.Option(
        False, "--score-only", help="Score existing predictions; don't generate new ones"
    ),
) -> None:
    """Generate frozen daily predictions and score completed ones."""
    from pipeline.ledger.scorer import score_predictions
    from pipeline.ledger.writer import write_daily_predictions

    target = datetime.date.fromisoformat(date) if date else datetime.date.today()

    if not score_only:
        written = write_daily_predictions(data_repo=data_repo.resolve(), as_of=target)
        typer.echo(f"[predict] Wrote {len(written)} ledger file(s) for {target}")

    new_outcomes = score_predictions(data_repo=data_repo.resolve(), as_of=target)
    typer.echo(f"[predict] Scored {len(new_outcomes)} new outcomes")


@app.command("build-site")
def _build_site(
    data_repo: Path = typer.Option(DEFAULT_DATA_REPO, "--data-repo"),
    site_data: Path = typer.Option(DEFAULT_SITE_DATA, "--site-data"),
) -> None:
    """Build site/public/data JSON from processed archive."""
    from pipeline.build.site_data import build_all

    build_all(
        data_repo_path=data_repo.resolve(),
        site_data_path=site_data.resolve(),
    )


# pyproject.toml [project.scripts] entrypoints
def universe() -> None:
    app(["universe"], standalone_mode=True)


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
