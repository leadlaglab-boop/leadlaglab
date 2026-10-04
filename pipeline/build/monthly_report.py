"""
Generate a monthly "state of the study" report.

Reads from site/public/data/ JSON files (which are committed to the repo after
each successful pipeline run) and writes a Markdown report to
study/reports/YYYY-MM.md.

Returns a short summary string suitable for use as a GitHub Issue body.
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import structlog

log = structlog.get_logger()

# Minimum detectable IC (from pre-registration, section 4)
IC_TARGET = 0.03
# Typical cross-section size for power calculations
N_SP500 = 500
N_RETAIL = 15
# Minimum months of live data recommended (practitioners' rule of thumb,
# accounting for Newey-West autocorrelation inflation)
POWER_THRESHOLDS = {1: 6, 5: 12, 21: 24}  # horizon_days -> min live months
# Live data collection start date
LIVE_START_DATE = date(2026, 10, 3)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file; return empty dict if missing or unreadable."""
    try:
        return cast("dict[str, Any]", json.loads(path.read_text()))
    except Exception:
        return {}


def _months_live(as_of: date | None = None) -> float:
    ref = as_of or date.today()
    delta = ref - LIVE_START_DATE
    return round(delta.days / 30.44, 1)


def _power_note(months: float, horizon: int) -> str:
    threshold = POWER_THRESHOLDS[horizon]
    if months >= threshold:
        return f"✓ {months:.1f}mo (threshold {threshold}mo)"
    shortfall = threshold - months
    return f"⚠ {months:.1f}mo of {threshold}mo needed (~{shortfall:.0f}mo remaining)"


def _ic_significance_label(bh_reject: bool, pvalue: float | None) -> str:
    if bh_reject:
        return "**sig**"
    if pvalue is not None and pvalue < 0.10:
        return "marginal"
    return "ns"


def generate_report(
    site_data_path: Path,
    reports_dir: Path,
    data_repo_path: Path | None = None,
    month: date | None = None,
) -> tuple[str, Path]:
    """
    Generate the monthly report.

    Returns (summary_text, report_path) where summary_text is a short
    plain-text summary suitable for a GitHub Issue body.
    """
    as_of = month or date.today()
    period = as_of.strftime("%Y-%m")
    month_label = as_of.strftime("%B %Y")

    site_data = site_data_path.resolve()
    manifest = _load_json(site_data / "manifest.json")
    universe = _load_json(site_data / "universe.json")
    eval_results = _load_json(site_data / "eval_results.json")
    predictions = _load_json(site_data / "predictions.json")
    pipeline_status = _load_json(site_data / "pipeline_status.json")

    months_live = _months_live(as_of)
    sp500_count = universe.get("sp500_count", 0)
    retail_count = universe.get("retail_attention_count", 0)
    data_as_of = (
        manifest.get("build_time", "unknown")[:10] if manifest.get("build_time") else "unknown"
    )

    # ---- Signal performance -------------------------------------------------
    results = eval_results.get("results", [])
    sig_count = sum(1 for r in results if r.get("ic_bh_reject", False))
    total_tests = len(results)

    # Build IC table rows (only 5-day horizon for brevity; show all in detail)
    ic_rows: list[str] = []
    for r in sorted(
        results, key=lambda x: (x.get("feature", ""), x.get("horizon_days", 0), x.get("cohort", ""))
    ):
        feat = r.get("feature", "—")
        horizon = r.get("horizon_days", "—")
        cohort = r.get("cohort", "—")
        ic = r.get("ic_mean")
        tstat = r.get("ic_tstat")
        bh = r.get("ic_bh_reject", False)
        pval = r.get("ic_pvalue")
        qs = r.get("quintile_spread")
        ic_str = f"{ic:.4f}" if ic is not None else "—"
        tstat_str = f"{tstat:.2f}" if tstat is not None else "—"
        qs_str = f"{qs:.4f}" if qs is not None else "—"
        sig = _ic_significance_label(bh, pval)
        ic_rows.append(
            f"| {feat} | {horizon}d | {cohort} | {ic_str} | {tstat_str} | {qs_str} | {sig} |"
        )

    ic_table = ""
    if ic_rows:
        header = "| Feature | Horizon | Cohort | IC (mean) | t-stat | Q5−Q1 spread | BH-FDR |\n"
        header += "|---------|---------|--------|-----------|--------|--------------|--------|\n"
        ic_table = header + "\n".join(ic_rows)
    else:
        ic_table = "_No evaluation results yet. Evaluation requires at least 60 days of live data._"

    # ---- Prediction ledger scorecard ----------------------------------------
    preds = predictions.get("predictions", [])
    scored = [p for p in preds if p.get("realized_excess_return") is not None]
    unscored = len(preds) - len(scored)

    def _hit_rate(subset: list[dict[str, Any]]) -> str:
        correct = [p for p in subset if p.get("correct") is True]
        if not subset:
            return "—"
        return f"{100 * len(correct) / len(subset):.1f}% ({len(correct)}/{len(subset)})"

    def _horizon_subset(preds: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
        return [p for p in preds if p.get("horizon_days") == days]

    ledger_rows = []
    for days in [1, 5, 21]:
        h_all = _horizon_subset(preds, days)
        h_scored = [p for p in h_all if p.get("realized_excess_return") is not None]
        ledger_rows.append(f"| {days}d | {len(h_all)} | {len(h_scored)} | {_hit_rate(h_scored)} |")

    ledger_table = (
        "| Horizon | Predictions made | Outcomes scored | Hit rate |\n"
        "|---------|-----------------|-----------------|----------|\n" + "\n".join(ledger_rows)
    )

    # ---- Power estimate -----------------------------------------------------
    power_rows = "\n".join(f"| {h}d | {_power_note(months_live, h)} |" for h in [1, 5, 21])
    power_table = "| Horizon | Status |\n|---------|--------|\n" + power_rows

    # Estimate adequate-power date for the hardest horizon (21d)
    months_needed_21d = POWER_THRESHOLDS[21]
    months_remaining = max(0.0, months_needed_21d - months_live)
    adequate_date = date(
        as_of.year + int((as_of.month + math.ceil(months_remaining) - 1) / 12),
        ((as_of.month + math.ceil(months_remaining) - 1) % 12) + 1,
        1,
    )

    # ---- Pipeline health ----------------------------------------------------
    sources = pipeline_status.get("sources", [])
    source_rows = []
    for s in sources:
        src = s.get("source", "—")
        status = s.get("status", "unknown")
        last_ok = s.get("last_success") or "—"
        records = s.get("records_today", 0)
        icon = (
            "✓"
            if status == "ok"
            else ("⚠" if status == "stale" else "✗" if status == "error" else "·")
        )
        source_rows.append(
            f"| {src} | {icon} {status} | {last_ok[:10] if len(str(last_ok)) > 10 else last_ok} | {records} |"
        )

    source_table = ""
    if source_rows:
        source_table = (
            "| Source | Status | Last success | Records (last run) |\n"
            "|--------|--------|--------------|--------------------|\n" + "\n".join(source_rows)
        )
    else:
        source_table = "_Pipeline status data not yet available._"

    # ---- Archive size -------------------------------------------------------
    archive_size_mb: int | None = None
    if data_repo_path and data_repo_path.exists():
        with contextlib.suppress(Exception):
            result = subprocess.run(
                ["du", "-sm", str(data_repo_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            archive_size_mb = int(result.stdout.split()[0])

    archive_note = (
        f"{archive_size_mb} MB (warning threshold: 600 MB, R2 migration at ~800 MB)"
        if archive_size_mb is not None
        else "_Archive size unknown (data repo not checked in this run)._"
    )

    # ---- Overall status -----------------------------------------------------
    overall_status = pipeline_status.get("overall_status", "unknown")
    status_icon = (
        "🟢" if overall_status == "ok" else ("🟡" if overall_status == "degraded" else "🔴")
    )

    # ---- Build report -------------------------------------------------------
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""# Lead/Lag Lab — State of the Study: {month_label}

**Generated:** {generated_at}
**Data as of:** {data_as_of}
**Live data collection start:** {LIVE_START_DATE.isoformat()}
**Months of live data:** {months_live:.1f}
**Pipeline status:** {status_icon} {overall_status}

---

> This report is auto-generated monthly. It is a factual summary of current
> data coverage and statistical results — not an investment view or recommendation.
> With {months_live:.1f} months of live data, results are preliminary.

---

## 1. Universe Coverage

| Cohort | Securities |
|--------|-----------|
| S&P 500 | {sp500_count} |
| Retail-attention basket | {retail_count} |
| **Total** | **{sp500_count + retail_count}** |

---

## 2. Signal Performance

**{sig_count} of {total_tests} signal × horizon × cohort combinations pass BH-FDR correction (q=0.05).**

{ic_table}

_IC = Spearman rank IC (out-of-sample); Q5−Q1 = long-short quintile excess return spread.
BH-FDR: **sig** = passes Benjamini-Hochberg correction at q=0.05; marginal = uncorrected p < 0.10._

---

## 3. Prediction Ledger

{ledger_table}

**Total predictions in ledger:** {len(preds)} ({unscored} awaiting outcomes)

---

## 4. Statistical Power

IC target (pre-registration §4): **{IC_TARGET}** (minimum detectable effect, 80% power, BH-corrected).

{power_table}

**Estimated date of adequate power for 21-day horizon:** approximately {adequate_date.strftime("%B %Y")}.

_Note: power thresholds account for Newey-West autocorrelation correction and BH FDR penalty.
Until the 21-day threshold is met, 21-day results should be treated as exploratory._

---

## 5. Pipeline Health

{source_table}

---

## 6. Archive

{archive_note}

---

## 7. Key Decisions / Amendments This Month

_No amendments recorded. See `study/AMENDMENTS.md` for any deviations from the pre-registration._

---

## 8. Next Month Focus

- Continue daily signal collection
- {"Evaluate whether any signal passes the BH-FDR threshold after more data accrues." if sig_count == 0 else f"{sig_count} signal(s) currently significant — monitor stability over next month."}
- Monitor prediction ledger hit rates vs baselines as more outcomes arrive

---

_Auto-generated by `lll-monthly-report`. View live site: [leadlaglab.com](https://leadlaglab.com)_
"""

    # ---- Write report -------------------------------------------------------
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{period}.md"
    report_path.write_text(report)
    log.info("wrote monthly report", path=str(report_path), period=period)

    # ---- Build short summary for GitHub Issue -------------------------------
    summary = (
        f"## State of the Study: {month_label}\n\n"
        f"**Live data:** {months_live:.1f} months | "
        f"**Pipeline:** {overall_status} | "
        f"**Signals significant (BH):** {sig_count}/{total_tests}\n\n"
        f"**Predictions:** {len(preds)} made, {len(scored)} scored\n\n"
        f"**Power (21d horizon):** {_power_note(months_live, 21)}\n\n"
        f"Full report: `study/reports/{period}.md`"
    )

    return summary, report_path
