"""Benchmark local Ollama models: NAIVE vs MANAGED (sliding-window+summary).

For every model it runs the same Q&A conversation twice -- once dumping the
full history, once using the sliding-window context -- and reports:

  * avg prompt tokens            (context size actually sent)
  * prompt_eval_ms               (time the model spends READING the prompt)
  * eval_ms                      (time spent GENERATING the answer)
  * avg wall_ms                  (end-to-end per turn)

Ollama's own timing counters are used, and each model is warmed up first so
model-load time does not pollute the numbers.

Run:
    python benchmark.py
    python benchmark.py --models llama3.1:8b --turns 3
    python benchmark.py --list
"""

from __future__ import annotations

import argparse
import time
from typing import List

from local_context import QUESTIONS, SAMPLE_DOCUMENT
from local_llm import LocalLLM
from sliding_context import SlidingWindowContext
from token_utils import count_tokens

DEFAULT_MODELS = ["llama3.1:8b"]


def run_strategy(llm: LocalLLM, corpus: str, questions: List[str], managed: bool,
                 max_tokens: int, summary_tokens: int, num_predict: int) -> List[dict]:
    rows: List[dict] = []
    if managed:
        ctx = SlidingWindowContext(max_tokens=max_tokens,
                                   summary_tokens=summary_tokens)
        ctx.add(corpus)
    history = [] if managed else [corpus]

    for i, q in enumerate(questions, 1):
        if managed:
            ctx.add(f"Question: {q}")
            prompt = ctx.render()
        else:
            prompt = "\n\n".join(history + [q])

        m = llm.generate_full(prompt, max_tokens=num_predict)
        rows.append(m)

        if managed:
            ctx.add(f"Answer: {m['response']}")
        else:
            history.append(f"Question: {q}")
            history.append(f"Answer: {m['response']}")

        print(f"      turn {i}: prompt_tok={m['prompt_tokens']:>4} "
              f"prompt_eval={m['prompt_eval_ms']:>8.1f}ms "
              f"gen={m['eval_ms']:>8.1f}ms")
    return rows


def summarize(rows: List[dict]) -> dict:
    n = max(1, len(rows))
    return {
        "avg_tokens": sum(r["prompt_tokens"] for r in rows) / n,
        "prompt_eval_ms": sum(r["prompt_eval_ms"] for r in rows),
        "eval_ms": sum(r["eval_ms"] for r in rows),
        "wall_ms": sum(r["wall_ms"] for r in rows),
        "avg_wall_ms": sum(r["wall_ms"] for r in rows) / n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark local models: naive vs managed context")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--turns", type=int, default=3, help="number of questions (default 3)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--summary-tokens", type=int, default=96)
    ap.add_argument("--num-predict", type=int, default=64,
                    help="max output tokens per answer (default 64)")
    args = ap.parse_args(argv)

    questions = QUESTIONS[: args.turns]
    results = {}

    for model in args.models:
        # Disable long "thinking" for reasoning models so timings are comparable.
        think = False if "qwen3" in model.lower() else None
        llm = LocalLLM(model=model, host=args.host, think=think)
        print(f"\n=== {model} (backend={llm.backend}) ===")
        if llm.backend != "ollama":
            print("  ! Ollama not reachable -- skipping real model")
            continue

        load_ms = llm.warmup()
        print(f"  warm-up load: {load_ms:.0f} ms")

        print("  NAIVE (full history)")
        naive_rows = run_strategy(llm, SAMPLE_DOCUMENT, questions, False,
                                  args.max_tokens, args.summary_tokens,
                                  args.num_predict)
        print("  MANAGED (sliding window + summary)")
        managed_rows = run_strategy(llm, SAMPLE_DOCUMENT, questions, True,
                                    args.max_tokens, args.summary_tokens,
                                    args.num_predict)

        results[model] = {
            "naive": summarize(naive_rows),
            "managed": summarize(managed_rows),
        }

    if not results:
        print("\nNo Ollama models were reachable.")
        return 1

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 88)
    print("RESULTS")
    print("=" * 88)
    header = (f"{'model':<20}{'strategy':<10}{'avg_tok':>9}"
              f"{'prompt_eval_ms':>17}{'gen_ms':>12}{'avg_wall_ms':>13}")
    print(header)
    print("-" * 88)
    for model, r in results.items():
        for strat in ("naive", "managed"):
            s = r[strat]
            print(f"{model:<20}{strat:<10}{s['avg_tokens']:>9.0f}"
                  f"{s['prompt_eval_ms']:>17.0f}{s['eval_ms']:>12.0f}"
                  f"{s['avg_wall_ms']:>13.0f}")
        print("-" * 88)

    print("\nSAVINGS FROM THE SLIDING-WINDOW CONTEXT")
    print(f"{'model':<20}{'token_save%':>13}{'prompt_eval_save%':>20}")
    print("-" * 55)
    for model, r in results.items():
        n, m = r["naive"], r["managed"]
        tok_save = 100 * (1 - m["avg_tokens"] / n["avg_tokens"]) if n["avg_tokens"] else 0
        pe_save = (100 * (1 - m["prompt_eval_ms"] / n["prompt_eval_ms"])
                   if n["prompt_eval_ms"] else 0)
        print(f"{model:<20}{tok_save:>12.1f}%{pe_save:>19.1f}%")

    print("\nNote: 'gen_ms' (answer generation) is roughly equal for both "
          "strategies -- the saving is in prompt size, i.e. context cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
