"""Golden dataset evaluation — compute retrieval & answer-quality metrics."""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

log = logging.getLogger(__name__)

DEFAULT_K_VALUES = [1, 3, 5, 10]


@dataclass
class GoldenItem:
    question: str
    expected_answer: str
    expected_pages: list[int]


@dataclass
class RetrievalMetrics:
    """Retrieval-quality metrics for a single query."""
    hit_at_k: dict[int, bool] = field(default_factory=dict)   # {1: True, 3: True, ...}
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict) # {5: 0.82, 10: 0.75}
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    retrieved_pages: list[int] = field(default_factory=list)


@dataclass
class JudgeScores:
    """Per-criterion LLM-as-judge scores (1-5)."""
    overall: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    reasoning: str = ""


@dataclass
class EvalResult:
    question: str
    answer: str
    expected_answer: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    judge: JudgeScores = field(default_factory=JudgeScores)


class GoldenDataset:
    """Load and evaluate against a golden QA dataset.

    Expected JSON format:
    [
        {
            "question": "What is the company's revenue?",
            "expected_answer": "Revenue was $50M in 2024.",
            "expected_pages": [12, 13]
        }
    ]
    """

    def __init__(self, path: str):
        self.items = self._load(path)

    @staticmethod
    def _load(path: str) -> list[GoldenItem]:
        data = json.loads(Path(path).read_text())
        return [
            GoldenItem(
                question=item["question"],
                expected_answer=item["expected_answer"],
                expected_pages=item.get("expected_pages", []),
            )
            for item in data
        ]

    @staticmethod
    def compute_retrieval_metrics(
        expected_pages: list[int],
        retrieved_metas: list[dict],
        k_values: list[int] | None = None,
    ) -> RetrievalMetrics:
        """Compute Hit@K, MRR, NDCG@K, Precision@K, Recall@K."""
        ks = k_values or DEFAULT_K_VALUES

        retrieved_pages = [m.get("page", -1) for m in retrieved_metas]
        result = RetrievalMetrics(retrieved_pages=retrieved_pages)

        if not expected_pages:
            # No expected pages — treat as all-correct
            for k in ks:
                result.hit_at_k[k] = True
                result.ndcg_at_k[k] = 1.0
                result.precision_at_k[k] = 1.0
                result.recall_at_k[k] = 1.0
            result.mrr = 1.0
            return result

        expected_set = set(expected_pages)

        # Binary relevance for each retrieved page
        relevance = [1 if p in expected_set else 0 for p in retrieved_pages]

        # MRR — reciprocal rank of first relevant result
        result.mrr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                result.mrr = 1.0 / rank
                break

        for k in ks:
            top_k_rel = relevance[:k]
            hits_in_k = sum(top_k_rel)

            # Hit@K — did any expected page appear in top-k?
            result.hit_at_k[k] = hits_in_k > 0

            # Precision@K — fraction of top-k that are relevant
            result.precision_at_k[k] = round(hits_in_k / k, 4) if k > 0 else 0.0

            # Recall@K — fraction of expected pages found in top-k
            result.recall_at_k[k] = (
                round(hits_in_k / len(expected_set), 4) if expected_set else 1.0
            )

            # NDCG@K — normalized discounted cumulative gain
            dcg = sum(
                rel_i / math.log2(i + 2)  # i+2 because enumerate starts at 0
                for i, rel_i in enumerate(top_k_rel)
            )
            # Ideal DCG: all relevant docs at the top
            ideal_count = min(len(expected_set), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
            result.ndcg_at_k[k] = round(dcg / idcg, 4) if idcg > 0 else 0.0

        return result

    @staticmethod
    def aggregate(results: list[EvalResult], k_values: list[int] | None = None) -> dict:
        """Aggregate metrics across all evaluation results."""
        n = len(results)
        if n == 0:
            return {}

        ks = k_values or DEFAULT_K_VALUES
        agg: dict = {"count": n}

        # Retrieval metrics
        retrieval: dict = {
            "mrr": round(mean(r.retrieval.mrr for r in results), 4),
        }

        for k in ks:
            k_label = str(k)
            hits = [r.retrieval.hit_at_k.get(k, False) for r in results]
            precs = [r.retrieval.precision_at_k.get(k, 0.0) for r in results]
            recalls = [r.retrieval.recall_at_k.get(k, 0.0) for r in results]
            ndcgs = [r.retrieval.ndcg_at_k.get(k, 0.0) for r in results]

            retrieval[f"hit@{k_label}"] = round(sum(hits) / n, 4)
            retrieval[f"precision@{k_label}"] = round(mean(precs), 4)
            retrieval[f"recall@{k_label}"] = round(mean(recalls), 4)
            retrieval[f"ndcg@{k_label}"] = round(mean(ndcgs), 4)

        agg["retrieval"] = retrieval

        # Judge metrics
        scores = [r.judge.overall for r in results]
        if any(s > 0 for s in scores):
            faithfulness = [r.judge.faithfulness for r in results]
            completeness = [r.judge.completeness for r in results]
            conciseness = [r.judge.conciseness for r in results]

            agg["judge"] = {
                "overall_mean": round(mean(scores), 2),
                "overall_median": round(median(scores), 2),
                "overall_min": round(min(scores), 2),
                "overall_max": round(max(scores), 2),
                "faithfulness_mean": round(mean(faithfulness), 2),
                "completeness_mean": round(mean(completeness), 2),
                "conciseness_mean": round(mean(conciseness), 2),
            }
        else:
            agg["judge"] = {"overall_mean": 0.0}

        return agg
