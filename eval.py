"""
Evaluation comparing multi-agent orchestrator against the single ReAct agent
baseline from the react-agent project.

Metrics per task:
- task_success (keyword match)
- total_tokens
- elapsed_ms
- subtask_count (orchestrator only)
- agent_steps (single-agent only)

Usage:
    python eval.py                        # orchestrator only
    python eval.py --compare              # orchestrator + single-agent baseline
    python eval.py --compare --verbose
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from orchestrator import run_orchestrator

TASKS_FILE = Path(__file__).parent / "eval_tasks.json"

# Attempt to import single-agent baseline
_REACT_AGENT_PATH = Path(__file__).parent.parent / "react-agent"
_SINGLE_AGENT_AVAILABLE = _REACT_AGENT_PATH.exists()
if _SINGLE_AGENT_AVAILABLE:
    sys.path.insert(0, str(_REACT_AGENT_PATH))
    from agent import ReActAgent  # noqa: E402
    sys.path.pop(0)


def load_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return json.load(f)


def keyword_pass(answer: str | None, keywords: list[str]) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def run_orchestrator_eval(tasks: list[dict], verbose: bool = False) -> list[dict]:
    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"[ORCH {i:02d}/{len(tasks)}] {task['question'][:60]}", end="  ", flush=True)
        run = run_orchestrator(task["question"])
        passed = keyword_pass(run["final_answer"], task["expected_keywords"])
        total_tokens = sum(run["tokens_used"].values())
        print(
            f"{'PASS' if passed else 'FAIL'}  "
            f"subtasks={len(run['subtasks'])}  tokens={total_tokens}  {run['elapsed_ms']:.0f}ms"
        )
        if verbose and not passed:
            print(f"      answer: {run['final_answer'][:120]}")
        results.append({
            "id": task["id"],
            "category": task.get("category", ""),
            "question": task["question"],
            "passed": passed,
            "subtask_count": len(run["subtasks"]),
            "tokens": total_tokens,
            "elapsed_ms": run["elapsed_ms"],
        })
    return results


def run_single_agent_eval(tasks: list[dict], verbose: bool = False) -> list[dict]:
    agent = ReActAgent()
    results = []
    for i, task in enumerate(tasks, start=1):
        print(f"[SING {i:02d}/{len(tasks)}] {task['question'][:60]}", end="  ", flush=True)
        run = agent.run(task["question"])
        passed = keyword_pass(run.final_answer, task["expected_keywords"])
        print(
            f"{'PASS' if passed else 'FAIL'}  "
            f"steps={run.total_steps}  tokens={run.total_tokens}  {run.elapsed_ms:.0f}ms"
        )
        if verbose and not passed:
            print(f"      answer: {(run.final_answer or '')[:120]}")
        results.append({
            "id": task["id"],
            "category": task.get("category", ""),
            "question": task["question"],
            "passed": passed,
            "steps": run.total_steps,
            "tokens": run.total_tokens,
            "elapsed_ms": run.elapsed_ms,
        })
    return results


def print_summary(label: str, results: list[dict], step_key: str = "subtask_count") -> None:
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Task success rate:  {passed / n * 100:.1f}%  ({passed}/{n})")
    if step_key in results[0]:
        avg_steps = statistics.mean(r[step_key] for r in results)
        print(f"  Avg {step_key}/task:  {avg_steps:.2f}")
    print(f"  Avg tokens/task:    {statistics.mean(r['tokens'] for r in results):.0f}")
    print(f"  Avg latency:        {statistics.mean(r['elapsed_ms'] for r in results):.0f} ms")


def print_comparison(orch: list[dict], single: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("  HEAD-TO-HEAD COMPARISON (orchestrator vs single-agent)")
    print(f"{'=' * 60}")
    print(f"  {'Task':<50} {'ORCH':>4}  {'SING':>4}")
    print(f"  {'-' * 50}  ----  ----")
    for o, s in zip(orch, single):
        o_mark = "PASS" if o["passed"] else "FAIL"
        s_mark = "PASS" if s["passed"] else "FAIL"
        print(f"  {o['question'][:50]:<50} {o_mark:>4}  {s_mark:>4}")

    orch_pass = sum(1 for r in orch if r["passed"]) / len(orch) * 100
    sing_pass = sum(1 for r in single if r["passed"]) / len(single) * 100
    orch_tok = statistics.mean(r["tokens"] for r in orch)
    sing_tok = statistics.mean(r["tokens"] for r in single)
    orch_ms = statistics.mean(r["elapsed_ms"] for r in orch)
    sing_ms = statistics.mean(r["elapsed_ms"] for r in single)

    print(f"\n  Success rate:  orchestrator {orch_pass:.1f}%  vs  single-agent {sing_pass:.1f}%")
    print(f"  Avg tokens:    orchestrator {orch_tok:.0f}  vs  single-agent {sing_tok:.0f}")
    print(f"  Avg latency:   orchestrator {orch_ms:.0f}ms  vs  single-agent {sing_ms:.0f}ms")

    token_delta = (orch_tok - sing_tok) / sing_tok * 100
    latency_delta = (orch_ms - sing_ms) / sing_ms * 100
    print(f"\n  Multi-agent overhead: +{token_delta:.1f}% tokens, +{latency_delta:.1f}% latency")
    print(f"  (overhead is the cost of decomposition; benefit is on complex tasks)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Also run single-agent baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks()

    print("\nRunning orchestrator eval...")
    orch_results = run_orchestrator_eval(tasks, verbose=args.verbose)
    print_summary("ORCHESTRATOR", orch_results, step_key="subtask_count")

    if args.compare:
        if not _SINGLE_AGENT_AVAILABLE:
            print("\nSingle-agent baseline not available (react-agent project not found).")
            sys.exit(1)
        print("\nRunning single-agent baseline eval...")
        single_results = run_single_agent_eval(tasks, verbose=args.verbose)
        print_summary("SINGLE AGENT (ReAct baseline)", single_results, step_key="steps")
        print_comparison(orch_results, single_results)


if __name__ == "__main__":
    main()
