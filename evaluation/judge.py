"""LLM-as-judge — scores answer quality on a 1-5 scale with per-criterion breakdown."""

import json
import logging
import re

import anthropic

from config import settings

log = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's answer \
to a question about a document.

## Criteria (each scored 1-5)

1. **Faithfulness** — Is the answer factually consistent with the expected answer \
and the retrieved context? Does it avoid hallucination?
   - 5: Fully faithful, no fabricated claims
   - 3: Mostly faithful, minor unsupported details
   - 1: Contradicts expected answer or hallucinates significantly

2. **Completeness** — Does the answer address all parts of the question \
and cover the key points from the expected answer?
   - 5: Covers every key point from the expected answer
   - 3: Covers some key points, misses others
   - 1: Misses the core point entirely

3. **Conciseness** — Is the answer focused and free of irrelevant information?
   - 5: Precise, no filler or off-topic content
   - 3: Some unnecessary detail but mostly on-topic
   - 1: Rambling, off-topic, or padded with irrelevant info

## Inputs

**Question:** {question}

**Expected Answer:** {expected_answer}

**Actual Answer:** {actual_answer}

**Retrieved Context (top chunks):**
{context}

## Instructions
Evaluate the actual answer against the expected answer and retrieved context.
Respond with ONLY a JSON object — no markdown, no explanation outside the JSON:

{{"faithfulness": <int 1-5>, "completeness": <int 1-5>, "conciseness": <int 1-5>, \
"overall": <int 1-5>, "reasoning": "<1-2 sentences explaining the scores>"}}

The overall score should reflect your holistic judgment, not just the average \
of the three criteria.
"""


class LLMJudge:
    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def score(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        context_chunks: list[str] | None = None,
    ) -> dict:
        """Return dict with faithfulness, completeness, conciseness, overall, reasoning."""
        context_str = "N/A"
        if context_chunks:
            numbered = [
                f"[{i+1}] {chunk[:500]}" for i, chunk in enumerate(context_chunks[:5])
            ]
            context_str = "\n\n".join(numbered)

        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            context=context_str,
        )

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "faithfulness": int(data.get("faithfulness", 0)),
                    "completeness": int(data.get("completeness", 0)),
                    "conciseness": int(data.get("conciseness", 0)),
                    "overall": int(data.get("overall", 0)),
                    "reasoning": data.get("reasoning", ""),
                }
        except (json.JSONDecodeError, KeyError, ValueError):
            log.warning("Failed to parse judge response: %s", text)

        return {
            "faithfulness": 0,
            "completeness": 0,
            "conciseness": 0,
            "overall": 0,
            "reasoning": f"Parse error: {text[:100]}",
        }
