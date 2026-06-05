"""
Planner agent — decomposes the user question into an ordered list of subtasks
and assigns each to the appropriate specialist (retriever, executor, verifier).

The planner is the only agent that sees the raw question.  Every other agent
sees only its own subtask description and the results already collected, which
keeps context windows small and attribution clean.
"""

import json
import logging
import time
from typing import Any

import openai
from pydantic import BaseModel, Field

from config import settings
from state import OrchestratorState, SubTask
from tracing import make_entry, timed

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """\
You are a task planner for a multi-agent system. Given a question, decompose it
into an ordered list of 2–5 subtasks. Each subtask must be:

- Atomic: accomplishable by a single specialist agent in one pass.
- Typed: assigned to exactly one agent type:
    • "retriever" — for finding facts, definitions, or document-grounded information.
    • "executor"  — for calculations, reasoning over gathered facts, or synthesis.
    • "verifier"  — for checking consistency, completeness, and correctness of results.

Rules:
- Always end with a "verifier" subtask.
- Keep descriptions concise — one sentence each.
- Order matters: retrievers before executors, verifier last.

Return ONLY a JSON array. Example:
[
  {"id": 1, "description": "Look up the definition of KV cache.", "agent_type": "retriever"},
  {"id": 2, "description": "Explain why KV cache reduces latency.", "agent_type": "executor"},
  {"id": 3, "description": "Verify the explanation is complete and accurate.", "agent_type": "verifier"}
]
"""


class SubTaskSchema(BaseModel):
    id: int
    description: str
    agent_type: str = Field(..., pattern="^(retriever|executor|verifier)$")


def planner_node(state: OrchestratorState) -> dict[str, Any]:
    question = state["question"]
    logger.info("Planner: decomposing question: %s", question[:80])

    client = openai.OpenAI(api_key=settings.openai_api_key)

    with timed() as t:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    tokens = resp.usage.total_tokens if resp.usage else 0
    raw = resp.choices[0].message.content or "[]"

    try:
        parsed = json.loads(raw)
        # Model may return {"tasks": [...]} or just [...]
        if isinstance(parsed, dict):
            parsed = next(iter(parsed.values()))
        subtasks: list[SubTask] = []
        for item in parsed:
            validated = SubTaskSchema(**item)
            subtasks.append(
                SubTask(
                    id=validated.id,
                    description=validated.description,
                    agent_type=validated.agent_type,
                    result=None,
                    done=False,
                )
            )
    except Exception as exc:
        logger.error("Planner failed to parse subtask list: %s — raw: %s", exc, raw[:200])
        # Fallback: single retriever + verifier
        subtasks = [
            SubTask(id=1, description=question, agent_type="retriever", result=None, done=False),
            SubTask(id=2, description="Verify the answer is complete.", agent_type="verifier", result=None, done=False),
        ]

    logger.info("Planner produced %d subtasks", len(subtasks))

    trace_entry = make_entry(
        agent="planner",
        event="result",
        data={"subtasks": [f"[{s['agent_type']}] {s['description']}" for s in subtasks]},
        latency_ms=t[0],
        tokens=tokens,
    )

    return {
        "subtasks": subtasks,
        "current_task_idx": 0,
        "trace": [trace_entry],
        "tokens_used": {"planner": tokens},
    }
