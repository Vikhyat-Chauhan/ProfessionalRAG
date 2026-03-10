"""RAG pipeline — orchestrates ingestion, retrieval, reranking, and generation."""

import logging

from config import settings
from ingestion.reader import read_pdf, file_fingerprint
from ingestion.chunker import ChunkHelper, ChunkConfig
from retrieval.embedder import Embedder
from retrieval.store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import LLMClient
from monitoring.metrics import metrics

log = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = Reranker()
        self.llm = LLMClient()

    def ingest(self, pdf_path: str, force: bool = False) -> int:
        """Ingest a PDF into the vector store. Returns chunk count.

        Skips ingestion if the file hasn't changed (based on SHA-256),
        unless force=True.
        """
        fp = file_fingerprint(pdf_path)

        if not force and not self.store.needs_ingestion(fp):
            count = self.store.collection.count() - 1
            log.info("Skipping ingestion — file unchanged (%d chunks)", count)
            return count

        log.info("Reading PDF: %s", pdf_path)
        pages = read_pdf(pdf_path)
        for page in pages:
            page["source"] = pdf_path

        cfg = ChunkConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks, metas = ChunkHelper.split_documents(pages, cfg)
        log.info("Created %d chunks", len(chunks))

        log.info("Encoding chunks...")
        embeddings = self.embedder.embed_documents(chunks)

        count = self.store.ingest(chunks, embeddings, metas, fp)
        return count

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Run the full retrieve → rerank → generate pipeline."""
        k = top_k or settings.top_k
        metrics.start_query(question)

        # Retrieve
        with metrics.track_latency("retrieval"):
            query_vec = self.embedder.embed_query(question)
            chunks, metas = self.store.query(query_vec)

        # Rerank
        with metrics.track_latency("rerank"):
            ranked = self.reranker.rerank(question, chunks, metas, top_k=k)

        top_score = ranked[0][2] if ranked else 0.0
        metrics.record_retrieval(top_score, len(ranked))

        # Generate
        context = [(chunk, meta) for chunk, meta, _ in ranked]
        answer = self.llm.generate(question, context)

        query_metrics = metrics.finish_query()

        return {
            "answer": answer,
            "sources": [
                {"page": m.get("page", "?"), "score": round(s, 4), "text": c[:200]}
                for c, m, s in ranked
            ],
            "metrics": {
                "total_ms": round(query_metrics.total_latency_ms, 1),
                "llm_ms": round(query_metrics.llm_latency_ms, 1),
                "cost_usd": round(query_metrics.cost_usd, 6),
                "top_score": round(top_score, 4),
            },
        }

    def evaluate(
        self, golden_path: str, use_judge: bool = True
    ) -> dict:
        """Run evaluation against a golden dataset."""
        from evaluation.golden import GoldenDataset, EvalResult
        from evaluation.judge import LLMJudge

        dataset = GoldenDataset(golden_path)
        judge = LLMJudge() if use_judge else None
        results: list[EvalResult] = []

        for item in dataset.items:
            log.info("Evaluating: %s", item.question)

            query_vec = self.embedder.embed_query(item.question)
            chunks, metas = self.store.query(query_vec)
            ranked = self.reranker.rerank(item.question, chunks, metas)

            context = [(c, m) for c, m, _ in ranked]
            answer = self.llm.generate(item.question, context)

            retrieved_metas = [m for _, m, _ in ranked]
            hit, mrr = GoldenDataset.compute_retrieval_metrics(
                item.expected_pages, retrieved_metas
            )

            result = EvalResult(
                question=item.question,
                answer=answer,
                expected_answer=item.expected_answer,
                hit_at_k=hit,
                mrr=mrr,
            )

            if judge:
                score, reasoning = judge.score(
                    item.question, item.expected_answer, answer
                )
                result.judge_score = score
                result.judge_reasoning = reasoning

            results.append(result)

        return {
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "hit": r.hit_at_k,
                    "mrr": r.mrr,
                    "judge_score": r.judge_score,
                    "judge_reasoning": r.judge_reasoning,
                }
                for r in results
            ],
            "aggregate": GoldenDataset.aggregate(results),
        }
