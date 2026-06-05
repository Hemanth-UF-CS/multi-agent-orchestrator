"""
Shared typed state object passed between all agents in the LangGraph graph.

Using TypedDict + Annotated[list, operator.add] means list fields are
*appended to* on each update rather than replaced — every agent contributes
to trace/results without clobbering previous entries.
"""

import operator
from typing import Annotated, Any, Optional, TypedDict


class SubTask(TypedDict):
    id: int
    description: str
    agent_type: str        # "retriever" | "executor" | "verifier"
    result: Optional[str]
    done: bool


class TraceEntry(TypedDict):
    agent: str
    event: str             # "start" | "tool_call" | "result" | "error"
    data: Any
    latency_ms: float
    tokens: int


class OrchestratorState(TypedDict):
    # Input
    question: str

    # Planning
    subtasks: list[SubTask]
    current_task_idx: int

    # Agent results
    retriever_results: Annotated[list[str], operator.add]
    executor_results: Annotated[list[str], operator.add]

    # Verification
    verification_passed: bool
    verification_feedback: str

    # Output
    final_answer: str

    # Observability — each agent appends its own entries
    trace: Annotated[list[TraceEntry], operator.add]

    # Token accounting per agent
    tokens_used: dict[str, int]

    # Error propagation
    error: Optional[str]
