"""CLI runner for the prompt-evaluation framework and agent memory demo.

Usage:
    python main.py eval           # full grid: models x approaches x styles
    python main.py eval --style specific --approach cot
    python main.py memory         # run the agent-memory demo
"""

from __future__ import annotations

import argparse
import itertools
import sys

from gemini_api import (
    APPROACHES,
    MODELS,
    STYLES,
    GeminiAPI,
    apply_approach,
    build_prompt,
)
from memory import MemoryStore, memory_update

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Explain the concept of photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "How do I reverse a linked list in Python?",
]


def run_evaluation(args) -> None:
    api = GeminiAPI(seed=args.seed)
    models = args.models or list(MODELS)
    approaches = args.approaches or list(APPROACHES)
    styles = args.styles or list(STYLES)

    print(f"{'model':<16}{'approach':<10}{'style':<14}"
          f"{'accuracy':>9}{'coherence':>10}{'latency_ms':>11}")
    print("-" * 70)

    rows = []
    question = args.question or DEFAULT_QUESTIONS[0]
    for model, approach, style in itertools.product(models, approaches, styles):
        prompt = apply_approach(build_prompt(question, style), approach)
        result = api.evaluateprompt(prompt, model=model, approach=approach,
                                    style=style)
        rows.append(result)
        print(f"{model:<16}{approach:<10}{style:<14}"
              f"{result.accuracy:>9.4f}{result.coherence:>10.4f}"
              f"{result.latency_ms:>11.2f}")

    if args.best:
        best = max(rows, key=lambda r: r.accuracy)
        print("\nBest (by accuracy):")
        print(f"  model={best.model}, approach={best.approach}, "
              f"style={best.style}, accuracy={best.accuracy:.4f}, "
              f"coherence={best.coherence:.4f}, latency={best.latency_ms:.2f}ms")


def run_memory_demo(args) -> None:
    store = MemoryStore()
    print("Simulating a customer-support agent across multiple interactions.\n")

    interactions = [
        ("query", "user asks about refund policy",
         "agent explains the 30-day policy"),
        ("query", "user asks about refund policy again",
         "agent recalls previous answer"),
        ("query", "user asks about refund policy for the third time",
         "agent confirms and offers an exception"),
        ("complaint", "user complains about late delivery",
         "agent apologizes and opens a ticket"),
        ("order", "user places an order for a laptop",
         "agent confirms the order"),
    ]

    for i, (z, action, reaction) in enumerate(interactions, start=1):
        state = memory_update(store, z, action, reaction)
        print(f"--- step {i}: m_{i-1} -> m_{i} (z={z}) ---")
        print(f"  x_t = {action}")
        print(f"  r_t = {reaction}")
        print(f"  working   : {list(state.working)}")
        print(f"  semantic  : {[(f.subject, round(f.confidence, 2)) for f in state.semantic]}")
        print(f"  procedural: {[p.name for p in state.procedural]}")
        print()

    print("Reading memory for a follow-up query:")
    for hit in store.read("refund policy", top_k=3):
        print(f"  - {hit}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prompt engineering evaluation + memory demo")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("eval", help="run prompt evaluation grid")
    eval_p.add_argument("--model", dest="models", action="append",
                        choices=list(MODELS), help="restrict to a model (repeatable)")
    eval_p.add_argument("--approach", dest="approaches", action="append",
                        choices=list(APPROACHES), help="restrict to an approach (repeatable)")
    eval_p.add_argument("--style", dest="styles", action="append",
                        choices=list(STYLES), help="restrict to a style (repeatable)")
    eval_p.add_argument("--question", help="custom question to evaluate")
    eval_p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    eval_p.add_argument("--best", action="store_true", help="also print best combo")

    sub.add_parser("memory", help="run the agent-memory demo")

    args = parser.parse_args(argv)
    if args.command == "eval":
        run_evaluation(args)
    elif args.command == "memory":
        run_memory_demo(args)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
