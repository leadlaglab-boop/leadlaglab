"""
NLP scoring: FinBERT and VADER for financial text.

Two scorers run on every piece of text (article titles, summaries):
  1. FinBERT (ProsusAI/finbert) — fine-tuned BERT for financial sentiment
  2. VADER (vaderSentiment) — rule-based, very fast; serves as a baseline

Both scorers are run and stored separately so they can be compared (a study question).
Models are lazy-loaded: FinBERT is not imported until first use, keeping CI fast
when NLP scoring is not needed.

Output per text:
  finbert_positive, finbert_negative, finbert_neutral (sum to 1.0)
  vader_compound (-1 to +1), vader_positive, vader_negative, vader_neutral

Model info is stored with every scored record for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import structlog

log = structlog.get_logger()

FINBERT_MODEL = "ProsusAI/finbert"


@dataclass
class SentimentScores:
    finbert_positive: float
    finbert_negative: float
    finbert_neutral: float
    finbert_model_name: str
    vader_compound: float
    vader_positive: float
    vader_negative: float
    vader_neutral: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NLPScorer:
    """
    Lazy-loading NLP scorer. FinBERT is loaded on first call to avoid
    expensive model loading in imports.
    """

    def __init__(self, finbert_model: str = FINBERT_MODEL, use_finbert: bool = True) -> None:
        self._finbert_model_name = finbert_model
        self._use_finbert = use_finbert
        self._pipeline: Any = None
        self._vader: Any = None

    def _load_vader(self) -> None:
        if self._vader is None:
            from vaderSentiment.vaderSentiment import (
                SentimentIntensityAnalyzer,  # type: ignore[import-not-found]
            )

            self._vader = SentimentIntensityAnalyzer()

    def _load_finbert(self) -> None:
        if self._pipeline is None and self._use_finbert:
            try:
                from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]

                log.info("loading FinBERT", model=self._finbert_model_name)
                self._pipeline = hf_pipeline(
                    "text-classification",
                    model=self._finbert_model_name,
                    top_k=None,
                    device=-1,  # CPU
                )
                log.info("FinBERT loaded")
            except Exception as e:
                log.warning("FinBERT load failed; falling back to VADER-only", error=str(e))
                self._use_finbert = False

    def score_text(self, text: str) -> SentimentScores:
        """Score a single text string."""
        self._load_vader()

        # VADER scores
        vs = self._vader.polarity_scores(text)
        vader_compound = float(vs["compound"])
        vader_positive = float(vs["pos"])
        vader_negative = float(vs["neg"])
        vader_neutral = float(vs["neu"])

        # FinBERT scores
        if self._use_finbert:
            self._load_finbert()

        fb_pos = fb_neg = fb_neu = 1.0 / 3.0  # uniform prior if unavailable
        if self._pipeline is not None:
            try:
                # FinBERT expects text truncated to 512 tokens
                truncated = text[:500]
                result = self._pipeline(truncated)[0]
                scores = {r["label"].lower(): r["score"] for r in result}
                fb_pos = scores.get("positive", fb_pos)
                fb_neg = scores.get("negative", fb_neg)
                fb_neu = scores.get("neutral", fb_neu)
            except Exception as e:
                log.warning("FinBERT inference failed", error=str(e)[:100])

        return SentimentScores(
            finbert_positive=round(fb_pos, 6),
            finbert_negative=round(fb_neg, 6),
            finbert_neutral=round(fb_neu, 6),
            finbert_model_name=self._finbert_model_name if self._use_finbert else "unavailable",
            vader_compound=round(vader_compound, 6),
            vader_positive=round(vader_positive, 6),
            vader_negative=round(vader_negative, 6),
            vader_neutral=round(vader_neutral, 6),
        )

    def score_batch(self, texts: list[str]) -> list[SentimentScores]:
        """Score multiple texts. FinBERT runs in batches for efficiency."""
        if not texts:
            return []

        self._load_vader()

        # VADER (fast; no batching needed)
        vader_results = [self._vader.polarity_scores(t) for t in texts]

        # FinBERT (batch)
        if self._use_finbert:
            self._load_finbert()

        finbert_results = None
        if self._pipeline is not None:
            try:
                truncated = [t[:500] for t in texts]
                raw = self._pipeline(truncated, batch_size=16)
                finbert_results = [{r["label"].lower(): r["score"] for r in item} for item in raw]
            except Exception as e:
                log.warning("FinBERT batch inference failed", error=str(e)[:100])

        scores = []
        for i, _text in enumerate(texts):
            vs = vader_results[i]
            if finbert_results and i < len(finbert_results):
                fb = finbert_results[i]
                fb_pos = fb.get("positive", 1 / 3)
                fb_neg = fb.get("negative", 1 / 3)
                fb_neu = fb.get("neutral", 1 / 3)
            else:
                fb_pos = fb_neg = fb_neu = 1 / 3

            scores.append(
                SentimentScores(
                    finbert_positive=round(fb_pos, 6),
                    finbert_negative=round(fb_neg, 6),
                    finbert_neutral=round(fb_neu, 6),
                    finbert_model_name=(
                        self._finbert_model_name if self._use_finbert else "unavailable"
                    ),
                    vader_compound=round(float(vs["compound"]), 6),
                    vader_positive=round(float(vs["pos"]), 6),
                    vader_negative=round(float(vs["neg"]), 6),
                    vader_neutral=round(float(vs["neu"]), 6),
                )
            )

        return scores

    def score_gdelt_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Add NLP scores to GDELT signal_raw records (in-place by returning new list).
        Extracts article titles from payload_json.sample_articles and scores them.
        """
        scored = []
        for record in records:
            payload = json.loads(record.get("payload_json", "{}"))
            articles = payload.get("sample_articles", [])
            texts = [a.get("title", "") for a in articles if a.get("title")]

            if not texts:
                scored.append(record)
                continue

            article_scores = self.score_batch(texts)
            if not article_scores:
                scored.append(record)
                continue

            # Aggregate: average across articles
            avg_finbert_pos = sum(s.finbert_positive for s in article_scores) / len(article_scores)
            avg_finbert_neg = sum(s.finbert_negative for s in article_scores) / len(article_scores)
            avg_vader = sum(s.vader_compound for s in article_scores) / len(article_scores)

            payload["nlp"] = {
                "finbert_avg_positive": round(avg_finbert_pos, 6),
                "finbert_avg_negative": round(avg_finbert_neg, 6),
                "vader_avg_compound": round(avg_vader, 6),
                "finbert_model": self._finbert_model_name if self._use_finbert else "unavailable",
                "n_articles_scored": len(article_scores),
                "article_scores": [s.to_dict() for s in article_scores[:5]],
            }

            new_record = dict(record)
            new_record["payload_json"] = json.dumps(payload)
            scored.append(new_record)

        return scored
