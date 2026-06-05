# Multi-Agent Orchestrator with Hierarchical Delegation

A planner-orchestrator that decomposes tasks and delegates to specialist sub-agents over a shared typed state, built with **LangGraph**.  This extends the [react-agent](https://github.com/S-HEMANTH-REDDY/react-agent) project — the single-agent primitive becomes one of several specialists in a directed graph.

## Architecture

```
         ┌─────────────────────────────────────────────┐
  query ─►  PLANNER                                    │
         │  LLM decomposes question into 2–5 subtasks  │
         └──────────────┬──────────────────────────────┘
                        │ subtask list
         ┌──────────────▼──────────────────────────────┐
         │  DISPATCH (conditional router)               │
         └─────┬──────────────┬──────────────┬─────────┘
               │              │              │
         ┌─────▼──┐    ┌──────▼─┐    ┌──────▼──┐
         │RETRIEV-│    │EXECUT- │    │VERIF-   │
         │ER      │    │OR      │    │IER      │
         │search_ │    │calcula-│    │checks   │
         │docs,   │    │tor,    │    │complete-│
         │lookup  │    │reason  │    │ness     │
         └─────┬──┘    └──────┬─┘    └──────┬──┘
               └──────────────┘             │
                     back to dispatch       │ passed/failed
                                    ┌───────▼────────┐
                                    │  AGGREGATOR    │
                                    │  final answer  │
                                    └───────┬────────┘
                                            │
                                           END
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Planner** | Decomposes the question into an ordered subtask list (JSON). | LLM only |
| **Retriever** | Resolves information-gathering subtasks. | `search_docs`, `lookup` |
| **Executor** | Handles calculation and synthesis subtasks. | `calculator` |
| **Verifier** | Checks completeness and correctness of collected results. | LLM only |
| **Aggregator** | Combines verified results into the final answer. | None |

### Shared state

```python
class OrchestratorState(TypedDict):
    question: str
    subtasks: list[SubTask]       # planner output
    current_task_idx: int         # dispatch cursor
    retriever_results: list[str]  # appended by each retriever call
    executor_results: list[str]   # appended by each executor call
    verification_passed: bool
    final_answer: str
    trace: list[TraceEntry]       # full per-agent observability log
    tokens_used: dict[str, int]   # token accounting per agent
```

`Annotated[list[...], operator.add]` on list fields means updates are *appended*, not replaced — no agent can accidentally clobber another's results.

### Observability

Every agent appends `TraceEntry` dicts to `state.trace`:

```
[01] PLANNER       result        12ms    142 tok
      subtasks: ['[retriever] Look up KV cache', '[executor] Explain latency', '[verifier] Check answer']

[02] RETRIEVER     tool_call      3ms      0 tok
      tool: search_docs  args: {'query': 'KV cache LLM inference'}
      result: [1] KV cache stores key-value pairs from previously ...

[03] EXECUTOR      result        18ms    210 tok
      KV cache eliminates redundant attention recomputation by storing ...

[04] VERIFIER      result        14ms    185 tok
      passed: True  feedback: Answer is complete and grounded in retrieved evidence.
```

## Quick start

```bash
git clone https://github.com/S-HEMANTH-REDDY/multi-agent-orchestrator.git
cd multi-agent-orchestrator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set OPENAI_API_KEY in .env

# Interactive demo
python main.py

# Orchestrator-only eval
python eval.py

# Head-to-head comparison with single-agent baseline
# (requires react-agent cloned at ../react-agent)
python eval.py --compare --verbose
```

## Evaluation

The eval set contains **15 compound tasks** that require both retrieval and reasoning — the type where multi-agent decomposition has the clearest advantage over single-agent.

```
python eval.py --compare
```

Outputs:
- **Task success rate** for each system
- **Avg tokens per task** (multi-agent overhead is the cost of decomposition)
- **Head-to-head table** — which tasks each system passes/fails
- **Token and latency delta** — quantifies where multi-agent adds value vs. overhead

> Fill in measured numbers after running the eval.

## Project structure

```
multi-agent-orchestrator/
├── orchestrator.py        # LangGraph graph: nodes, routing, compile()
├── state.py               # OrchestratorState TypedDict
├── tracing.py             # TraceEntry helpers and print_trace()
├── config.py              # pydantic-settings
├── agents/
│   ├── planner.py         # Task decomposition
│   ├── retriever.py       # search_docs + lookup
│   ├── executor.py        # calculator + synthesis
│   └── verifier.py        # completeness/correctness check
├── eval.py                # Orchestrator eval + single-agent comparison
├── eval_tasks.json        # 15 compound tasks
├── main.py                # Interactive CLI
└── requirements.txt
```

## Design decisions

**Why LangGraph over a hand-rolled loop?**
LangGraph handles typed state propagation, conditional routing, and cycle detection out of the box.  Using it here turns "familiarity with LangGraph" into "built a multi-agent system in LangGraph" — a concrete, demonstrable skill for JDs that name it explicitly.

**Why `operator.add` on list fields?**
Each specialist appends its results to `retriever_results` / `executor_results` rather than overwriting.  This means late-arriving retriever results don't clobber earlier ones, and the verifier sees the full picture.

**Why a separate verifier agent?**
The single-agent self-critique in react-agent is a prompt nudge.  A dedicated verifier agent is a structural guarantee — it runs as its own LLM call with its own prompt and cannot be skipped by the planner.

**When does multi-agent help vs. hurt?**
Multi-agent helps on decomposition-heavy tasks (compound questions, multi-step reasoning).  It adds token and latency overhead on simple lookup tasks where a single ReAct agent would finalize in 2–3 steps.  The `--compare` eval quantifies this tradeoff.
