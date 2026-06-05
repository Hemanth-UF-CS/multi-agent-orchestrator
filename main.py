"""Interactive CLI for the multi-agent orchestrator."""

import logging
import sys

from orchestrator import run_orchestrator
from tracing import print_trace, summarize_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)


def main() -> None:
    print("Multi-Agent Orchestrator (type 'quit' to exit)\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print()
        result = run_orchestrator(question, verbose=False)

        print("\n--- Subtask Plan ---")
        for task in result["subtasks"]:
            status = "done" if task["done"] else "pending"
            print(f"  [{task['agent_type']:10s}] {task['description']}")
            if task.get("result"):
                print(f"    result: {task['result'][:100]}")

        print_trace(result["trace"])

        token_by_agent = summarize_tokens(result["trace"])
        total_tokens = sum(token_by_agent.values())

        print(f"\nAnswer: {result['final_answer'] or '(no answer)'}")
        print(f"Verified: {result['verification_passed']}")
        print(f"Tokens:   {total_tokens}  {token_by_agent}")
        print(f"Elapsed:  {result['elapsed_ms']:.0f}ms")
        print()


if __name__ == "__main__":
    main()
