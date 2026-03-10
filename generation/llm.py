"""LLM generation via Anthropic API with token/cost tracking."""

import logging
import anthropic

from config import settings
from monitoring.metrics import metrics

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly based on the provided context. "
    "If the answer is not present in the context, say \"I don't have enough information to answer that.\" "
    #"Always reference the page number when citing information."
)


class LLMClient:
    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Copy .env.example to .env and add your key."
                )
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def generate(
        self,
        query: str,
        context_blocks: list[tuple[str, dict]],
    ) -> str:
        prompt = self._build_prompt(query, context_blocks)

        with metrics.track_latency("llm_generation"):
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

        answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        metrics.record_tokens(input_tokens, output_tokens)
        log.info(
            "LLM: %d input tokens, %d output tokens",
            input_tokens,
            output_tokens,
        )
        return answer

    @staticmethod
    def _build_prompt(query: str, context_blocks: list[tuple[str, dict]]) -> str:
        parts = []
        for chunk, meta in context_blocks:
            page = meta.get("page", "?")
            source = meta.get("source", "")
            # Use "File:" for code files, "Page:" for PDFs
            if isinstance(page, str) and "/" in page:
                header = f"[File: {page}]"
            else:
                header = f"[Page {page}]"
            if source:
                header += f" ({source})"
            parts.append(f"{header}\n{chunk}")

        context = "\n\n---\n\n".join(parts)
        return f"Context:\n{context}\n\nQuestion: {query}"
