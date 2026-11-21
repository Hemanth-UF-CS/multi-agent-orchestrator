"""
Tool definitions, Pydantic schemas, and implementations.

Each tool has a strict Pydantic model for argument validation.  The agent loop
validates every LLM-produced tool call against these schemas before execution;
malformed or hallucinated arguments are rejected and retried rather than
silently passed through.
"""

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------


class SearchDocsArgs(BaseModel):
    query: str = Field(..., description="Search query for the document store")


class LookupArgs(BaseModel):
    term: str = Field(..., description="Term or concept to look up")


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ..., description="Mathematical expression to evaluate, e.g. '2 * (3 + 4)'"
    )


class FinalAnswerArgs(BaseModel):
    answer: str = Field(..., description="Final answer to the user's question")


# ---------------------------------------------------------------------------
# OpenAI function-calling schemas (injected into every LLM call)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search a document store for relevant passages. "
                "Use this when you need factual information, context, or details "
                "not yet in your reasoning trace."
            ),
            "parameters": SearchDocsArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": (
                "Look up a specific term, concept, technology, or entity "
                "to get a concise definition or description."
            ),
            "parameters": LookupArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression safely. "
                "Use for any arithmetic, percentages, powers, or numerical reasoning — "
                "never compute in your head."
            ),
            "parameters": CalculatorArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit the final answer. Only call this after self-critique confirms "
                "the answer is complete and consistent with the gathered evidence."
            ),
            "parameters": FinalAnswerArgs.model_json_schema(),
        },
    },
]

# Map tool name → (function, arg schema)
TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "search_docs": SearchDocsArgs,
    "lookup": LookupArgs,
    "calculator": CalculatorArgs,
    "final_answer": FinalAnswerArgs,
}

# ---------------------------------------------------------------------------
# In-memory document store (BM25-style keyword search)
# ---------------------------------------------------------------------------

_DOCS: list[str] = [
    "Python is a high-level, interpreted, dynamically typed programming language prized for readability and scientific computing.",
    "The Transformer architecture, introduced in 'Attention is All You Need' (Vaswani et al., 2017), uses self-attention mechanisms to process sequences in parallel.",
    "LangGraph is a library by LangChain for building stateful, multi-actor applications using directed graph control flow and typed state.",
    "ReAct (Reasoning + Acting) is an agent framework that interleaves chain-of-thought reasoning traces with tool calls, enabling self-correction.",
    "RAG (Retrieval-Augmented Generation) augments a language model's response by first retrieving relevant documents from an external store.",
    "KV cache in LLM inference stores the key-value pairs from previously processed tokens, eliminating redundant recomputation during autoregressive decoding.",
    "Apache Kafka is a distributed event-streaming platform designed for high-throughput, fault-tolerant, persistent message delivery across services.",
    "Redis is an in-memory data-structure store used as a cache, message broker (Pub/Sub), and sorted-set database for leaderboards and feeds.",
    "The University of Florida (UF) is a major public research university in Gainesville, Florida, consistently ranked in the top 10 US public universities.",
    "IFAS (Institute of Food and Agricultural Sciences) at UF conducts research and education in agriculture, natural resources, and life sciences.",
    "GPT-4 is OpenAI's large multimodal model (text + image inputs) with strong reasoning capabilities, released in 2023.",
    "Pydantic is a Python data-validation library that uses type annotations to parse and validate data at runtime, widely used with FastAPI and LLM tool calling.",
    "FastAPI is a modern, high-performance Python web framework for building APIs, built on Starlette and Pydantic with automatic OpenAPI generation.",
    "Docker is an open platform for developing, shipping, and running applications in portable containers that share the host OS kernel.",
    "Kubernetes (k8s) is an open-source container-orchestration system that automates deployment, scaling, and management of containerized workloads.",
    "PyTorch is an open-source ML framework developed by Meta with dynamic computation graphs, widely used for research and production model training.",
    "CUDA is NVIDIA's parallel-computing platform and programming model that enables GPU-accelerated computation for scientific and ML workloads.",
    "vLLM implements PagedAttention to manage KV-cache memory in non-contiguous blocks, achieving near-zero memory waste and higher throughput than naive batching.",
    "Prometheus is an open-source monitoring system that scrapes time-series metrics from instrumented services and stores them for alerting and dashboards.",
    "OpenAI is an AI research company that develops the GPT series of large language models and provides the OpenAI API for programmatic access.",
]


def search_docs(query: str) -> str:
    query_terms = set(query.lower().split())
    scored: list[tuple[int, str]] = []
    for doc in _DOCS:
        doc_terms = set(doc.lower().split())
        overlap = len(query_terms & doc_terms)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "No relevant documents found for that query."
    top = [doc for _, doc in scored[:3]]
    return "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(top))


# ---------------------------------------------------------------------------
# Inline knowledge base
# ---------------------------------------------------------------------------

_KB: dict[str, str] = {
    "python": "Python is a high-level, dynamically typed, interpreted language favored for readability, scientific computing, and rapid prototyping.",
    "transformer": "The Transformer is a neural architecture using multi-head self-attention (Vaswani et al., 2017). It is the foundation of all modern LLMs.",
    "langgraph": "LangGraph is a LangChain library for building stateful multi-agent applications as directed graphs with typed shared state.",
    "react": "ReAct (Reasoning + Acting) is an agent paradigm that interleaves chain-of-thought reasoning with tool calls, enabling grounded, self-correcting behavior.",
    "rag": "Retrieval-Augmented Generation (RAG) retrieves relevant documents at inference time and injects them into the LLM prompt to ground the response.",
    "kafka": "Apache Kafka is a distributed, fault-tolerant event-streaming platform for high-throughput producer-consumer pipelines.",
    "redis": "Redis is an in-memory key-value store supporting strings, hashes, sorted sets, and Pub/Sub messaging, commonly used for caching and queues.",
    "pydantic": "Pydantic validates Python data with type annotations at runtime. It is widely used for API request/response schemas and LLM tool-call validation.",
    "openai": "OpenAI is an AI safety and research company that develops and deploys the GPT series of large language models via the OpenAI API.",
    "gpt-4": "GPT-4 is OpenAI's large multimodal model with strong reasoning and coding abilities, released in March 2023.",
    "llm": "A Large Language Model (LLM) is a neural network with billions of parameters pre-trained on large text corpora, capable of generation and reasoning.",
    "cuda": "CUDA is NVIDIA's parallel-computing platform and C-extension API for writing GPU kernels, enabling scientific and ML acceleration.",
    "pytorch": "PyTorch is Meta's open-source ML framework with dynamic computation graphs, widely used for research and industrial model training.",
    "university of florida": "UF is a major public research university in Gainesville, FL, ranked among the top US public universities with strong engineering and agricultural programs.",
    "docker": "Docker packages applications and their dependencies into portable containers that share the host OS kernel, enabling consistent deployment.",
    "kubernetes": "Kubernetes (k8s) automates deployment, scaling, and management of containerized applications across clusters.",
    "fastapi": "FastAPI is a modern Python web framework for APIs, built on Starlette and Pydantic, with automatic OpenAPI docs and async support.",
    "vllm": "vLLM uses PagedAttention to store KV-cache in non-contiguous memory blocks, dramatically increasing LLM inference throughput.",
    "prometheus": "Prometheus is an open-source monitoring system that scrapes time-series metrics from instrumented services.",
    "kv cache": "KV cache stores key-value attention tensors from previous tokens during autoregressive decoding, eliminating redundant computation and cutting latency.",
}


def lookup(term: str) -> str:
    key = term.lower().strip()
    if key in _KB:
        return _KB[key]
    for k, v in _KB.items():
        if key in k or k in key:
            return v
    return f"No entry found for '{term}'. Try search_docs with a broader query."


# ---------------------------------------------------------------------------
# Calculator: arithmetic on an allow-list of AST nodes (no eval)
# ---------------------------------------------------------------------------

_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "int": int,
    "float": float,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 1000  # keeps 9**9**9-style inputs from hanging the agent


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent larger than {_MAX_EXPONENT}")
        return _BINARY[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_evaluate(a) for a in node.args]
        if node.func.id == "pow" and len(args) >= 2 and abs(args[1]) > _MAX_EXPONENT:
            raise ValueError(f"exponent larger than {_MAX_EXPONENT}")
        return _FUNCTIONS[node.func.id](*args)
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    expr = expression.strip()
    try:
        result = _evaluate(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        return f"Error evaluating '{expr}': {exc}"
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


# ---------------------------------------------------------------------------
# Dispatch map used by the agent loop
# ---------------------------------------------------------------------------

TOOL_FN_MAP: dict[str, tuple[Any, type[BaseModel]]] = {
    "search_docs": (search_docs, SearchDocsArgs),
    "lookup": (lookup, LookupArgs),
    "calculator": (calculator, CalculatorArgs),
}
