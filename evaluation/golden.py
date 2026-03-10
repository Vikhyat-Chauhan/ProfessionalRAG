"""Golden dataset evaluation — compute retrieval & answer metrics."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class GoldenItem:
    question: str
    expected_answer: str
    expected_pages: list[int]


@dataclass
class EvalResult:
    question: str
    answer: str
    expected_answer: str
    hit_at_k: bool
    mrr: float
    judge_score: float = 0.0
    judge_reasoning: str = ""


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
    ) -> tuple[bool, float]:
        """Compute Hit@K and MRR based on page-level relevance."""
        if not expected_pages:
            return True, 1.0

        retrieved_pages = [m.get("page", -1) for m in retrieved_metas]

        hit = any(p in expected_pages for p in retrieved_pages)

        mrr = 0.0
        for rank, page in enumerate(retrieved_pages, start=1):
            if page in expected_pages:
                mrr = 1.0 / rank
                break

        return hit, mrr

    @staticmethod
    def aggregate(results: list[EvalResult]) -> dict:
        n = len(results)
        if n == 0:
            return {}
        return {
            "count": n,
            "hit_rate": round(sum(r.hit_at_k for r in results) / n, 3),
            "avg_mrr": round(sum(r.mrr for r in results) / n, 3),
            "avg_judge_score": round(
                sum(r.judge_score for r in results) / n, 2
            ),
        }
