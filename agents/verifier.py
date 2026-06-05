"""
Verifier agent — checks the combined results for completeness, consistency,
and correctness before the aggregator commits to a final answer.

Returns verification_passed=True and the assembled answer if results are
sufficient, or verification_passed=False with specific feedback if not.
"""

import json
import logging
from typing import Any

import openai
from pydantic import BaseModel

from config import settings
from state import OrchestratorState
from tracing import make_entry, timed

logger = logging.getLogger(__name__)

_VERIFIER_SYSTEM = """\
You are a verification agent. You receive:
- The original question
- Retriever results (document-grounded facts)
- Executor results (reasoning and calculations)

Your job is to determine if these results are sufficient to answer the question
completely and correctly.

Return a JSON object with exactly these fields:
{
  "passed": true | false,
  "feedback": "one sentence explaining the verdict",
  "answer_draft": "draft of the final answer if passed=true, else empty string"
}

Be strict: if any part of the question is unanswered, or if there is a
calculation that has not been verified, return passed=false with specific
feedback about what is missing.
"""


class VerifierOutput(BaseModel):
    passed: bool
    feedback: str
    answer_draft: str


def verifier_node(state: OrchestratorState) -> dict[str, Any]:
    question = state["question"]
    retriever_results = state.get("retriever_results", [])
    executor_results = state.get("executor_results", [])

    logger.info("Verifier: checking results for question: %s", question[:80])

    context = ""
    if retriever_results:
        context += "Retriever results:\n" + "\n---\n".join(retriever_results) + "\n\n"
    if executor_results:
        context += "Executor results:\n" + "\n---\n".join(executor_results) + "\n\n"

    client = openai.OpenAI(api_key=settings.openai_api_key)

    with timed() as t:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _VERIFIER_SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\n{context}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    tokens = resp.usage.total_tokens if resp.usage else 0
    raw = resp.choices[0].message.content or "{}"

    try:
        parsed = json.loads(raw)
        output = VerifierOutput(**parsed)
    except Exception as exc:
        logger.error("Verifier parse error: %s — raw: %s", exc, raw[:200])
        output = VerifierOutput(passed=False, feedback="Verification parse failed.", answer_draft="")

    logger.info(
        "Verifier: passed=%s feedback=%s", output.passed, output.feedback[:80]
    )

    trace_entry = make_entry(
        agent="verifier",
        event="result",
        data={"passed": output.passed, "feedback": output.feedback, "answer_preview": output.answer_draft[:100]},
        latency_ms=t[0],
        tokens=tokens,
    )

    return {
        "verification_passed": output.passed,
        "verification_feedback": output.feedback,
        "final_answer": output.answer_draft if output.passed else "",
        "trace": [trace_entry],
        "tokens_used": {**state.get("tokens_used", {}), "verifier": tokens},
    }
