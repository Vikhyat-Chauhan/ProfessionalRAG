"""LLM generation via Anthropic API with token/cost tracking."""

import logging
import anthropic

from config import settings
from monitoring.metrics import metrics

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Vikhyat's personal AI assistant, embedded on his portfolio website. "
    "Your job is to help recruiters, hiring managers, and visitors learn about Vikhyat — "
    "his projects, skills, experience, and ideas — by answering their questions using the provided context.\n\n"
    "Guidelines:\n"
    "- Speak in a warm, professional, and confident tone — like a knowledgeable colleague advocating for Vikhyat.\n"
    "- Refer to him as \"Vikhyat\" (not \"the candidate\" or \"the user\").\n"
    "- When discussing his projects, highlight the technical depth, the problem solved, and the impact.\n"
    "- Ground every answer in the provided context. If the context doesn't cover something, say: "
    "\"That's not something I have details on, but feel free to reach out to Vikhyat directly!\"\n"
    "- Keep answers concise but substantive — recruiters are busy. Use bullet points when listing skills or projects.\n"
    "- If asked about fit for a role, connect Vikhyat's experience from the context to what's being asked.\n"
    "- Never fabricate details about Vikhyat's background. Stick to what's in the context."
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

    def generate_stream(
        self,
        query: str,
        context_blocks: list[tuple[str, dict]],
    ):
        """Yield text chunks as they arrive from Claude (streaming)."""
        prompt = self._build_prompt(query, context_blocks)

        with self.client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

            response = stream.get_final_message()
            metrics.record_tokens(
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

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
