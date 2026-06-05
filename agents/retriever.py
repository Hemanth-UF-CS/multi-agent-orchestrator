"""
Retriever agent — resolves information-gathering subtasks.

Uses the same search_docs + lookup tools from the react-agent project.
The agent issues a single LLM call with tool_choice="required" and returns
the aggregated tool results as a structured finding.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import openai

# Reuse tools from react-agent if available; otherwise inline stubs.
_REACT_AGENT_PATH = Path(__file__).parent.parent.parent / "react-agent"
if _REACT_AGENT_PATH.exists():
    sys.path.insert(0, str(_REACT_AGENT_PATH))
    from tools import search_docs, lookup, TOOL_SCHEMAS, SearchDocsArgs, LookupArgs  # noqa: E402
    sys.path.pop(0)
else:
    from tools_stub import search_docs, lookup, TOOL_SCHEMAS, SearchDocsArgs, LookupArgs  # noqa: E402

from config import settings
from state import OrchestratorState
from tracing import make_entry, timed

logger = logging.getLogger(__name__)

_RETRIEVER_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in {"search_docs", "lookup"}]

_RETRIEVER_SYSTEM = """\
You are a specialist information-retrieval agent.
Your only job is to fulfill the given subtask by calling the available tools.
Call search_docs or lookup (or both) to gather all relevant information for the subtask.
After gathering information, return a concise summary of your findings as plain text.
Do NOT attempt to answer questions outside your subtask scope.
"""


def retriever_node(state: OrchestratorState) -> dict[str, Any]:
    idx = state["current_task_idx"]
    subtasks = state["subtasks"]
    task = subtasks[idx]

    logger.info("Retriever: subtask [%d] %s", task["id"], task["description"][:80])

    client = openai.OpenAI(api_key=settings.openai_api_key)
    messages: list[dict] = [
        {"role": "system", "content": _RETRIEVER_SYSTEM},
        {"role": "user", "content": task["description"]},
    ]

    tool_results: list[str] = []
    total_tokens = 0
    trace_entries = []

    for call_num in range(4):  # max 4 retrieval tool calls per subtask
        with timed() as t:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                tools=_RETRIEVER_TOOLS,
                tool_choice="auto",
                temperature=0.0,
            )

        total_tokens += resp.usage.total_tokens if resp.usage else 0
        choice = resp.choices[0]

        if not choice.message.tool_calls:
            # Model chose not to call a tool — its text is the finding.
            finding = choice.message.content or "(no result)"
            tool_results.append(finding)
            trace_entries.append(
                make_entry("retriever", "result", finding, t[0], resp.usage.total_tokens if resp.usage else 0)
            )
            break

        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                continue

            with timed() as tool_t:
                if tool_name == "search_docs":
                    result = search_docs(**SearchDocsArgs(**args).model_dump())
                elif tool_name == "lookup":
                    result = lookup(**LookupArgs(**args).model_dump())
                else:
                    result = f"Unknown tool: {tool_name}"

            tool_results.append(result)
            trace_entries.append(
                make_entry("retriever", "tool_call", {"tool": tool_name, "args": args, "result": result[:120]}, tool_t[0], 0)
            )

            call_id = f"retriever_call_{call_num}"
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}],
            })
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

    combined = "\n\n".join(tool_results)

    # Mark subtask done
    updated_subtasks = list(subtasks)
    updated_subtasks[idx] = {**task, "result": combined, "done": True}

    return {
        "subtasks": updated_subtasks,
        "current_task_idx": idx + 1,
        "retriever_results": [combined],
        "trace": trace_entries,
        "tokens_used": {**state.get("tokens_used", {}), f"retriever_{idx}": total_tokens},
    }
