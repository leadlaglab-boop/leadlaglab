"""Tests for the monthly report generator."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pipeline.build.monthly_report import (
    LIVE_START_DATE,
    _months_live,
    _power_note,
    generate_report,
)


def _write_site_data(site_data_path: Path, **overrides: object) -> None:
    """Write minimal valid site data JSONs to a temp directory."""
    defaults: dict[str, object] = {
        "manifest.json": {
            "build_time": "2026-11-01T09:00:00Z",
            "schema_version": "1.0.0",
        },
        "universe.json": {
            "as_of": "2026-11-01",
            "sp500_count": 503,
            "retail_attention_count": 15,
            "securities": [],
        },
        "eval_results.json": {
            "as_of": "2026-11-01",
            "run_id": "abc123",
            "results": [
                {
                    "feature": "wiki_pageviews",
                    "horizon_days": 5,
                    "cohort": "sp500",
                    "ic_mean": 0.025,
                    "ic_tstat": 1.8,
                    "ic_pvalue": 0.08,
                    "ic_bh_reject": False,
                    "quintile_spread": 0.003,
                    "n_obs": 20,
                    "ic_series": [],
                },
                {
                    "feature": "gdelt_tone",
                    "horizon_days": 1,
                    "cohort": "sp500",
                    "ic_mean": 0.041,
                    "ic_tstat": 2.5,
                    "ic_pvalue": 0.02,
                    "ic_bh_reject": True,
                    "quintile_spread": 0.005,
                    "n_obs": 20,
                    "ic_series": [],
                },
            ],
        },
        "predictions.json": {
            "as_of": "2026-11-01",
            "predictions": [
                {
                    "prediction_id": "aaa",
                    "made_at": "2026-10-05T16:00:00Z",
                    "security_id": "sid_aapl",
                    "ticker": "AAPL",
                    "target_date_start": "2026-10-06",
                    "target_date_end": "2026-10-07",
                    "horizon_days": 1,
                    "model_id": "zero_v1",
                    "predicted_direction": "up",
                    "predicted_excess_return": 0.0,
                    "confidence_interval": [None, None],
                    "realized_excess_return": 0.012,
                    "scored_at": "2026-10-08T16:00:00Z",
                    "correct": True,
                },
                {
                    "prediction_id": "bbb",
                    "made_at": "2026-10-06T16:00:00Z",
                    "security_id": "sid_msft",
                    "ticker": "MSFT",
                    "target_date_start": "2026-10-07",
                    "target_date_end": "2026-10-14",
                    "horizon_days": 5,
                    "model_id": "momentum_v1",
                    "predicted_direction": "down",
                    "predicted_excess_return": -0.005,
                    "confidence_interval": [None, None],
                    "realized_excess_return": None,
                    "scored_at": None,
                    "correct": None,
                },
            ],
        },
        "pipeline_status.json": {
            "as_of": "2026-11-01T09:00:00Z",
            "overall_status": "ok",
            "sources": [
                {
                    "source": "wikipedia",
                    "status": "ok",
                    "last_success": "2026-11-01",
                    "records_today": 503,
                    "error_message": None,
                },
                {
                    "source": "edgar",
                    "status": "stale",
                    "last_success": "2026-10-29",
                    "records_today": 0,
                    "error_message": None,
                },
            ],
        },
    }
    for filename, data in {**defaults, **overrides}.items():
        (site_data_path / filename).write_text(json.dumps(data))


class TestMonthsLive:
    def test_on_start_date(self) -> None:
        assert _months_live(LIVE_START_DATE) == 0.0

    def test_one_month(self) -> None:
        # ~30 days from start
        result = _months_live(date(2026, 11, 2))
        assert 0.9 < result < 1.1

    def test_six_months(self) -> None:
        result = _months_live(date(2027, 4, 3))
        assert 5.8 < result < 6.2


class TestPowerNote:
    def test_below_threshold(self) -> None:
        note = _power_note(2.0, 21)
        assert "⚠" in note
        assert "remaining" in note

    def test_above_threshold(self) -> None:
        note = _power_note(30.0, 1)
        assert "✓" in note

    def test_at_threshold(self) -> None:
        note = _power_note(6.0, 1)
        assert "✓" in note


class TestGenerateReport:
    def test_writes_markdown_file(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        _write_site_data(site_data_path)

        summary, report_path = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )

        assert report_path.exists()
        assert report_path.name == "2026-11.md"
        content = report_path.read_text()
        assert "Lead/Lag Lab" in content
        assert "November 2026" in content

    def test_report_contains_signal_table(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        _write_site_data(site_data_path)

        _, report_path = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )
        content = report_path.read_text()
        assert "wiki_pageviews" in content
        assert "gdelt_tone" in content
        assert "**sig**" in content  # gdelt_tone is BH-significant in test data

    def test_report_contains_ledger_stats(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        _write_site_data(site_data_path)

        _, report_path = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )
        content = report_path.read_text()
        assert "Prediction Ledger" in content
        assert "1/1" in content  # 1 scored prediction, 1 correct

    def test_summary_string_format(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        _write_site_data(site_data_path)

        summary, _ = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )
        assert "State of the Study" in summary
        assert "November 2026" in summary

    def test_handles_empty_data_gracefully(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        # Write minimal universe but empty eval and predictions
        _write_site_data(
            site_data_path,
            **{
                "eval_results.json": {"as_of": "2026-11-01", "results": []},
                "predictions.json": {"as_of": "2026-11-01", "predictions": []},
                "pipeline_status.json": {
                    "as_of": "2026-11-01T00:00:00Z",
                    "overall_status": "degraded",
                    "sources": [],
                },
            },
        )

        # Should not raise
        summary, report_path = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )
        assert report_path.exists()
        assert "degraded" in summary

    def test_missing_json_files_graceful(self, tmp_path: Path) -> None:
        site_data_path = tmp_path / "data"
        site_data_path.mkdir()
        reports_dir = tmp_path / "reports"
        # Don't write any JSON files — all should fall back gracefully

        summary, report_path = generate_report(
            site_data_path=site_data_path,
            reports_dir=reports_dir,
            data_repo_path=None,
            month=date(2026, 11, 1),
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Lead/Lag Lab" in content
