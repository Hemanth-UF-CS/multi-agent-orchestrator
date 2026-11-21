"""Offline tests: routing, retry limit, graph wiring and tools. No API calls are made."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import pytest  # noqa: E402

import orchestrator  # noqa: E402
from config import settings  # noqa: E402
from tools import calculator  # noqa: E402


def _task(kind):
    return {"id": 1, "description": "x", "agent_type": kind, "result": None, "done": False}


@pytest.mark.parametrize("kind", ["retriever", "executor", "verifier"])
def test_dispatch_routes_to_the_next_subtask(kind):
    state = {"current_task_idx": 1, "subtasks": [_task("retriever"), _task(kind)]}
    assert orchestrator.dispatch_router(state) == kind


def test_dispatch_aggregates_when_subtasks_are_done():
    assert orchestrator.dispatch_router({"current_task_idx": 2, "subtasks": [_task("retriever")] * 2}) == "aggregate"


def test_unknown_agent_type_falls_back_to_retriever():
    assert orchestrator.dispatch_router({"current_task_idx": 0, "subtasks": [_task("poet")]}) == "retriever"


def _plans(n):
    return [{"agent": "planner", "event": "result"}] * n


def test_failed_verification_replans_until_the_retry_limit():
    limit = settings.max_planner_retries
    assert orchestrator.route_after_verify({"verification_passed": False, "trace": _plans(1)}) == "planner"
    assert orchestrator.route_after_verify({"verification_passed": False, "trace": _plans(limit)}) == "planner"
    assert orchestrator.route_after_verify({"verification_passed": False, "trace": _plans(limit + 1)}) == "aggregate"


def test_passed_verification_aggregates():
    assert orchestrator.route_after_verify({"verification_passed": True, "trace": _plans(1)}) == "aggregate"


def test_graph_compiles_with_every_agent():
    nodes = set(orchestrator.build_graph().get_graph().nodes)
    assert {"planner", "dispatch", "retriever", "executor", "verifier", "aggregate"} <= nodes


def test_executor_calculator_is_the_safe_one():
    assert calculator("6 * 7") == "42"
    assert calculator("__import__('os')").startswith("Error")
