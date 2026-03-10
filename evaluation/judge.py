"""LLM-as-judge — scores answer quality on a 1-5 scale."""

import json
import logging
import re

import anthropic

from config import settings

log = logging.getLogger(__name__)

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI assistant's answer.

## Criteria
1. **Faithfulness** — Is the answer supported by the provided context?
2. **Completeness** — Does it address all parts of the question?
3. **Conciseness** — Is it free of irrelevant information?

## Inputs
**Question:** {question}

**Expected Answer:** {expected_answer}

**Actual Answer:** {actual_answer}

## Instructions
Rate the answer from 1 to 5:
- 5: Perfect — faithful, complete, concise
- 4: Minor omission or wording issue
- 3: Partially correct, missing key details
- 2: Mostly wrong or misleading
- 1: Completely wrong or hallucinated

Respond with ONLY a JSON object:
{{"score": <int>, "reasoning": "<one sentence>"}}
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
    ) -> tuple[float, str]:
        """Return (score, reasoning)."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
        )

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        try:
            # Extract JSON from response (handle markdown code blocks)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data["score"]), data.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            log.warning("Failed to parse judge response: %s", text)

        return 0.0, f"Parse error: {text[:100]}"
