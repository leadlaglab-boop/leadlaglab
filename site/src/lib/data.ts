import { readFileSync, existsSync } from "fs";
import { join } from "path";

const DATA_DIR = join(process.cwd(), "public", "data");

function readJson<T>(filename: string, fallback: T): T {
  const path = join(DATA_DIR, filename);
  if (!existsSync(path)) return fallback;
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as T;
  } catch {
    return fallback;
  }
}

// ── Types ───────────────────────────────────────────────────────────────────

export interface Security {
  security_id: string;
  ticker: string;
  name: string;
  gics_sector: string;
  cohort: "sp500" | "retail_attention";
}

export interface Universe {
  as_of: string;
  sp500_count: number;
  retail_attention_count: number;
  securities: Security[];
}

export interface Manifest {
  schema_version: string;
  built_at: string;
  files: Record<string, { sha256: string; size_bytes: number }>;
}

export interface SignalSummary {
  as_of: string;
  signals: SignalMeta[];
}

export interface SignalMeta {
  key: string;           // e.g. "wiki_pageviews"
  name: string;          // e.g. "Wikipedia Pageviews"
  family: string;        // e.g. "Public Attention"
  source: string;
  description: string;
  coverage_pct: number;  // fraction of universe with data
  backfill_from: string | null;
  live_from: string | null;
  ic_mean: number | null;
  ic_tstat: number | null;
  ic_significant: boolean;
  n_dates: number;
}

export interface EvalResults {
  as_of: string;
  run_id: string;
  results: SignalEval[];
}

export interface SignalEval {
  feature: string;
  horizon_days: number;
  cohort: string;
  ic_mean: number;
  ic_std: number;
  ic_tstat: number;
  ic_pvalue: number;
  ic_bh_reject: boolean;
  quintile_spread: number;
  hit_rate: number;
  n_obs: number;
  n_dates: number;
  ic_series: Array<{ date: string; ic: number }>;
}

export interface PredictionLedger {
  as_of: string;
  predictions: Prediction[];
}

export interface Prediction {
  prediction_id: string;
  made_at: string;
  security_id: string;
  ticker: string;
  target_date_start: string;
  target_date_end: string;
  horizon_days: number;
  model_id: string;
  predicted_direction: "up" | "down";
  predicted_excess_return: number;
  confidence_interval: [number, number];
  realized_excess_return: number | null;
  scored_at: string | null;
  correct: boolean | null;
}

export interface PipelineStatus {
  as_of: string;
  overall_status: "ok" | "degraded" | "error";
  sources: SourceStatus[];
  recent_runs: RunRecord[];
}

export interface SourceStatus {
  source: string;
  last_success: string | null;
  last_attempt: string | null;
  status: "ok" | "stale" | "error" | "never";
  records_today: number;
  error_message: string | null;
}

export interface RunRecord {
  run_id: string;
  started_at: string;
  finished_at: string | null;
  status: "success" | "failed" | "running";
  sources_ok: number;
  sources_failed: number;
  error_summary: string | null;
}

// ── Loaders ────────────────────────────────────────────────────────────────

export function loadUniverse(): Universe {
  return readJson<Universe>("universe.json", {
    as_of: "",
    sp500_count: 0,
    retail_attention_count: 0,
    securities: [],
  });
}

export function loadManifest(): Manifest {
  return readJson<Manifest>("manifest.json", {
    schema_version: "1.0.0",
    built_at: "",
    files: {},
  });
}

export function loadSignalSummary(): SignalSummary {
  return readJson<SignalSummary>("signals_summary.json", {
    as_of: "",
    signals: [],
  });
}

export function loadEvalResults(): EvalResults {
  return readJson<EvalResults>("eval_results.json", {
    as_of: "",
    run_id: "",
    results: [],
  });
}

export function loadPredictionLedger(): PredictionLedger {
  return readJson<PredictionLedger>("predictions.json", {
    as_of: "",
    predictions: [],
  });
}

export function loadPipelineStatus(): PipelineStatus {
  return readJson<PipelineStatus>("pipeline_status.json", {
    as_of: "",
    overall_status: "error",
    sources: [],
    recent_runs: [],
  });
}

// ── Helpers ────────────────────────────────────────────────────────────────

export function formatDate(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatPct(n: number | null, digits = 1): string {
  if (n == null) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function formatNumber(n: number | null, digits = 3): string {
  if (n == null) return "—";
  return n.toFixed(digits);
}

export function getUniqueSignalFamilies(): string[] {
  const summary = loadSignalSummary();
  return [...new Set(summary.signals.map((s) => s.family))];
}

export function getSectorList(universe: Universe): string[] {
  return [...new Set(universe.securities.map((s) => s.gics_sector))].sort();
}
