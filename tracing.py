"""
Per-agent tracing and observability utilities.

Every agent call produces a list of TraceEntry dicts that get appended to
OrchestratorState.trace.  After a run completes, print_trace() renders a
human-readable timeline showing inputs, outputs, tool calls, token costs,
and latency per step — the same information you'd want in a production
observability dashboard.
"""

import time
from contextlib import contextmanager
from typing import Any, Generator

from state import TraceEntry


def make_entry(
    agent: str,
    event: str,
    data: Any,
    latency_ms: float = 0.0,
    tokens: int = 0,
) -> TraceEntry:
    return TraceEntry(
        agent=agent,
        event=event,
        data=data,
        latency_ms=latency_ms,
        tokens=tokens,
    )


@contextmanager
def timed() -> Generator[list[float], None, None]:
    """Context manager that measures elapsed ms.  Usage: with timed() as t: ...; ms = t[0]"""
    start = time.perf_counter()
    bucket: list[float] = [0.0]
    try:
        yield bucket
    finally:
        bucket[0] = (time.perf_counter() - start) * 1000


def print_trace(trace: list[TraceEntry]) -> None:
    """Print a human-readable trace timeline to stdout."""
    print("\n" + "=" * 70)
    print("AGENT TRACE")
    print("=" * 70)
    for i, entry in enumerate(trace, start=1):
        agent = entry["agent"].upper()
        event = entry["event"]
        latency = entry["latency_ms"]
        tokens = entry["tokens"]
        data = entry["data"]

        header = f"[{i:02d}] {agent:12s}  {event:12s}  {latency:6.0f}ms  {tokens:5d} tok"
        print(header)

        if isinstance(data, str):
            preview = data[:120].replace("\n", " ↵ ")
            print(f"      {preview}")
        elif isinstance(data, dict):
            for k, v in data.items():
                v_str = str(v)[:80].replace("\n", " ↵ ")
                print(f"      {k}: {v_str}")
        elif isinstance(data, list):
            for item in data[:3]:
                print(f"      • {str(item)[:100]}")
        print()
    print("=" * 70)


def summarize_tokens(trace: list[TraceEntry]) -> dict[str, int]:
    """Return total tokens consumed per agent."""
    totals: dict[str, int] = {}
    for entry in trace:
        agent = entry["agent"]
        totals[agent] = totals.get(agent, 0) + entry["tokens"]
    return totals
