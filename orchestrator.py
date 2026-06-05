"""
Multi-Agent Orchestrator built with LangGraph.

Graph topology:
                    ┌─────────┐
         START ──► │ planner │
                    └────┬────┘
                         │
                    ┌────▼─────┐
                    │ dispatch │ ◄─────────────────────┐
                    └────┬─────┘                       │
              ┌──────────┼──────────┐                  │
              │          │          │                   │
         ┌────▼──┐  ┌────▼───┐ ┌───▼───┐              │
         │retriev│  │executo │ │verifi │              │
         │  er   │  │  r     │ │  er   │              │
         └────┬──┘  └────┬───┘ └───┬───┘              │
              └──────────┴─────────┘                   │
                          │                            │
                    ┌─────▼──────┐                     │
                    │route_after │────── need more ─────┘
                    │  dispatch  │
                    └─────┬──────┘
                          │ done
                    ┌─────▼──────┐
                    │ aggregate  │
                    └─────┬──────┘
                          │
                         END

Conditional routing:
- dispatch:       routes to retriever / executor / verifier based on the
                  current subtask's agent_type, or to aggregate if all done.
- route_after_verify: if verifier passed → aggregate; if not → planner (retry).
"""

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.planner import planner_node
from agents.retriever import retriever_node
from agents.executor import executor_node
from agents.verifier import verifier_node
from state import OrchestratorState
from tracing import make_entry, timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (return the name of the next node)
# ---------------------------------------------------------------------------


def route_after_plan(state: OrchestratorState) -> Literal["dispatch"]:
    return "dispatch"


def dispatch_router(state: OrchestratorState) -> Literal["retriever", "executor", "verifier", "aggregate"]:
    """Route to the correct specialist based on the next pending subtask."""
    idx = state["current_task_idx"]
    subtasks = state.get("subtasks", [])

    if idx >= len(subtasks):
        return "aggregate"

    agent_type = subtasks[idx]["agent_type"]
    if agent_type == "retriever":
        return "retriever"
    if agent_type == "executor":
        return "executor"
    if agent_type == "verifier":
        return "verifier"

    logger.warning("Unknown agent_type '%s' — defaulting to retriever", agent_type)
    return "retriever"


def route_after_verify(state: OrchestratorState) -> Literal["aggregate", "planner"]:
    if state.get("verification_passed"):
        return "aggregate"
    return "planner"


# ---------------------------------------------------------------------------
# Aggregator node
# ---------------------------------------------------------------------------


def aggregate_node(state: OrchestratorState) -> dict[str, Any]:
    """Collect all results and produce the final answer if verifier didn't."""
    with timed() as t:
        final = state.get("final_answer") or ""
        if not final:
            parts = state.get("retriever_results", []) + state.get("executor_results", [])
            final = " ".join(parts).strip() or "(no answer produced)"

    trace_entry = make_entry(
        agent="aggregator",
        event="result",
        data=final[:200],
        latency_ms=t[0],
        tokens=0,
    )

    return {"final_answer": final, "trace": [trace_entry]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    graph = StateGraph(OrchestratorState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("dispatch", lambda s: s)   # pass-through routing node
    graph.add_node("retriever", retriever_node)
    graph.add_node("executor", executor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("aggregate", aggregate_node)

    # Entry
    graph.set_entry_point("planner")

    # Edges
    graph.add_conditional_edges("planner", route_after_plan, {"dispatch": "dispatch"})
    graph.add_conditional_edges(
        "dispatch",
        dispatch_router,
        {
            "retriever": "retriever",
            "executor": "executor",
            "verifier": "verifier",
            "aggregate": "aggregate",
        },
    )

    # After each specialist, return to dispatch to get the next subtask
    graph.add_edge("retriever", "dispatch")
    graph.add_edge("executor", "dispatch")

    # Verifier routes to aggregate or back to planner for a retry
    graph.add_conditional_edges(
        "verifier",
        route_after_verify,
        {"aggregate": "aggregate", "planner": "planner"},
    )

    graph.add_edge("aggregate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run API
# ---------------------------------------------------------------------------


def run_orchestrator(question: str, verbose: bool = False) -> dict[str, Any]:
    """
    Run the multi-agent orchestrator on a question.

    Returns a dict with:
      final_answer, trace, tokens_used, subtasks, verification_passed
    """
    import time

    app = build_graph()
    initial_state = OrchestratorState(
        question=question,
        subtasks=[],
        current_task_idx=0,
        retriever_results=[],
        executor_results=[],
        verification_passed=False,
        verification_feedback="",
        final_answer="",
        trace=[],
        tokens_used={},
        error=None,
    )

    wall_start = time.perf_counter()
    final_state = app.invoke(initial_state)
    elapsed_ms = (time.perf_counter() - wall_start) * 1000

    if verbose:
        from tracing import print_trace, summarize_tokens

        print_trace(final_state["trace"])
        token_summary = summarize_tokens(final_state["trace"])
        print("Token summary:", token_summary)
        print(f"Total elapsed: {elapsed_ms:.0f}ms")

    return {
        "question": question,
        "final_answer": final_state["final_answer"],
        "trace": final_state["trace"],
        "tokens_used": final_state.get("tokens_used", {}),
        "subtasks": final_state.get("subtasks", []),
        "verification_passed": final_state.get("verification_passed", False),
        "elapsed_ms": elapsed_ms,
    }
