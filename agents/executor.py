"""
Executor agent — handles calculation, reasoning, and synthesis subtasks.

Has access to the calculator tool and the collected retriever results.
Synthesizes gathered facts into a structured response for its subtask.
"""

import json
import logging
from typing import Any

import openai

from tools import calculator, TOOL_SCHEMAS, CalculatorArgs

from config import settings
from state import OrchestratorState
from tracing import make_entry, timed

logger = logging.getLogger(__name__)

_EXECUTOR_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "calculator"]

_EXECUTOR_SYSTEM = """\
You are a specialist execution agent for reasoning and calculation tasks.
You have access to results collected by a retrieval agent — use them as your evidence.

Your job:
1. Read the retriever results provided as context.
2. Fulfill your assigned subtask using the calculator tool for any arithmetic.
3. Return a concise, precise response. Show your reasoning briefly.

Do NOT speculate beyond the evidence provided.
"""


def executor_node(state: OrchestratorState) -> dict[str, Any]:
    idx = state["current_task_idx"]
    subtasks = state["subtasks"]
    task = subtasks[idx]
    retriever_context = "\n\n".join(state.get("retriever_results", []))

    logger.info("Executor: subtask [%d] %s", task["id"], task["description"][:80])

    client = openai.OpenAI(api_key=settings.openai_api_key)
    context_block = f"Retriever context:\n{retriever_context}\n\n" if retriever_context else ""

    messages: list[dict] = [
        {"role": "system", "content": _EXECUTOR_SYSTEM},
        {"role": "user", "content": f"{context_block}Subtask: {task['description']}"},
    ]

    total_tokens = 0
    trace_entries = []
    result_text = ""

    for call_num in range(6):
        with timed() as t:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=_EXECUTOR_TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )

        tok = resp.usage.total_tokens if resp.usage else 0
        total_tokens += tok
        choice = resp.choices[0]

        if not choice.message.tool_calls:
            result_text = choice.message.content or "(no result)"
            trace_entries.append(make_entry("executor", "result", result_text, t[0], tok))
            break

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                continue

            with timed() as tool_t:
                if tool_name == "calculator":
                    calc_result = calculator(**CalculatorArgs(**args).model_dump())
                else:
                    calc_result = f"Unknown tool: {tool_name}"

            trace_entries.append(
                make_entry("executor", "tool_call", {"tool": tool_name, "args": args, "result": calc_result}, tool_t[0], 0)
            )

            call_id = f"executor_call_{call_num}"
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}],
            })
            messages.append({"role": "tool", "tool_call_id": call_id, "content": calc_result})

    updated_subtasks = list(subtasks)
    updated_subtasks[idx] = {**task, "result": result_text, "done": True}

    return {
        "subtasks": updated_subtasks,
        "current_task_idx": idx + 1,
        "executor_results": [result_text],
        "trace": trace_entries,
        "tokens_used": {**state.get("tokens_used", {}), f"executor_{idx}": total_tokens},
    }
