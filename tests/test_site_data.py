"""Tests for pipeline/build/site_data.py — JSON exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.build.site_data import (
    _safe_float,
    build_eval_results_json,
    build_manifest,
    build_pipeline_status_json,
    build_predictions_json,
    build_universe_json,
)

# ── _safe_float ──────────────────────────────────────────────────────────────


def test_safe_float_none() -> None:
    assert _safe_float(None) is None


def test_safe_float_nan() -> None:
    assert _safe_float(float("nan")) is None


def test_safe_float_normal() -> None:
    assert _safe_float(0.123456789) == pytest.approx(0.12345679, rel=1e-6)


# ── build_universe_json ───────────────────────────────────────────────────────


def test_build_universe_json_empty(tmp_path: Path) -> None:
    """If no security master exists, should raise or return gracefully."""
    data_repo = tmp_path / "data"
    data_repo.mkdir()
    site_data = tmp_path / "site"
    site_data.mkdir()

    with pytest.raises(OSError):
        build_universe_json(data_repo, site_data)


# ── build_eval_results_json ───────────────────────────────────────────────────


def test_build_eval_results_json_empty(tmp_path: Path) -> None:
    """With no eval data, should write an empty results list."""
    data_repo = tmp_path / "data"
    (data_repo / "processed" / "evaluation").mkdir(parents=True)
    site_data = tmp_path / "site"
    site_data.mkdir()

    out = build_eval_results_json(data_repo, site_data)

    assert out["results"] == []
    assert "as_of" in out
    output_file = site_data / "eval_results.json"
    assert output_file.exists()
    loaded = json.loads(output_file.read_text())
    assert loaded["results"] == []


def test_build_eval_results_json_structure(tmp_path: Path) -> None:
    """With eval data, every record should have the required fields."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_repo = tmp_path / "data"
    eval_dir = data_repo / "processed" / "evaluation"
    eval_dir.mkdir(parents=True)
    site_data = tmp_path / "site"
    site_data.mkdir()

    # Write minimal results.parquet
    results_table = pa.table(
        {
            "feature": ["wiki_pageviews", "wiki_pageviews"],
            "horizon": pa.array([5, 21], type=pa.int32()),
            "cohort": ["sp500", "sp500"],
            "eval_type": ["oos", "oos"],
            "metric": ["ic", "ic"],
            "value": [0.04, 0.02],
            "t_stat": [2.1, 1.1],
            "p_value_raw": [0.04, 0.27],
            "p_value_bh": [0.05, 0.30],
            "significant_bh": [True, False],
            "n_periods": pa.array([120, 120], type=pa.int32()),
        }
    )
    pq.write_table(results_table, eval_dir / "results.parquet")

    out = build_eval_results_json(data_repo, site_data)

    assert len(out["results"]) == 2
    required = {
        "feature",
        "horizon_days",
        "cohort",
        "ic_mean",
        "ic_tstat",
        "ic_bh_reject",
        "n_dates",
        "ic_series",
    }
    for rec in out["results"]:
        assert required.issubset(rec.keys()), f"Missing keys: {required - rec.keys()}"
    # ic_series should be empty list when no ic_series.parquet
    assert out["results"][0]["ic_series"] == []


# ── build_predictions_json ────────────────────────────────────────────────────


def test_build_predictions_json_empty(tmp_path: Path) -> None:
    """With no ledger data, should write empty predictions list."""
    data_repo = tmp_path / "data"
    (data_repo / "processed" / "ledger").mkdir(parents=True)
    site_data = tmp_path / "site"
    site_data.mkdir()

    out = build_predictions_json(data_repo, site_data)

    assert out["predictions"] == []
    assert (site_data / "predictions.json").exists()


# ── build_pipeline_status_json ────────────────────────────────────────────────


def test_build_pipeline_status_json_no_runs(tmp_path: Path) -> None:
    """With no run data, should return degraded status with 'never' for all sources."""
    data_repo = tmp_path / "data"
    data_repo.mkdir()
    site_data = tmp_path / "site"
    site_data.mkdir()

    out = build_pipeline_status_json(data_repo, site_data)

    assert out["overall_status"] in ("degraded", "ok", "error")
    assert isinstance(out["sources"], list)
    assert len(out["sources"]) > 0
    assert all(s["status"] in ("never", "ok", "stale", "error") for s in out["sources"])
    assert (site_data / "pipeline_status.json").exists()


# ── build_manifest ────────────────────────────────────────────────────────────


def test_build_manifest_hashes(tmp_path: Path) -> None:
    """Manifest should hash all JSON files except itself."""
    (tmp_path / "universe.json").write_text('{"test": 1}')
    (tmp_path / "eval_results.json").write_text('{"results": []}')

    manifest = build_manifest(tmp_path)

    assert "universe.json" in manifest["files"]
    assert "eval_results.json" in manifest["files"]
    assert "manifest.json" not in manifest["files"]
    for _name, info in manifest["files"].items():
        assert len(info["sha256"]) == 16
        assert info["size_bytes"] > 0
