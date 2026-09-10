"""Evaluate a local LLM with vs. without sliding-window + summarization.

Two strategies are compared on the same conversation (an "object of
discussion" -- a document we ask questions about):

* naive   -- every turn sends the ENTIRE history so far as context.
* managed -- a SlidingWindowContext keeps a bounded window and summarizes
             overflow, so prompt size stays capped.

Metrics reported: total prompt tokens, latency, token reduction %, and speedup.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from local_llm import LocalLLM
from sliding_context import SlidingWindowContext
from summarizer import rouge1_recall
from token_utils import count_tokens, format_tokens

SAMPLE_DOCUMENT = """\
Large Language Models (LLMs) have a limited context window, which restricts how
much information they can consider at once. Context management is the practice
of selecting, ordering, and compressing the most relevant information before it
is sent to the model. Retrieval-Augmented Generation, or RAG, fetches relevant
documents from an external store prior to generation. Memory management lets a
model retain information across interactions. Short-term memory maintains
coherence within the current conversation. Long-term memory stores user
preferences and history across sessions. Vector databases enable semantic search
over stored knowledge using embedding vectors. A sliding window keeps only the
most recent turns and discards older ones. Summarization compresses older
context into a short digest instead of discarding it. Agentic AI systems perform
multi-step reasoning and use external tools over long tasks. Hierarchical memory
separates fast working memory from slower external storage. The memory update
function m_t equals f of z, x_t, r_t, and m_{t-1}. Accuracy measures how correct
an answer is. Coherence measures how logically consistent a response is.
Latency is the time taken to produce an answer. Extractive summarization selects
the most important existing sentences without rewriting them. Compression ratio
is the size of the original text divided by the size of the summary.
"""

QUESTIONS = [
    "What is the main limitation of LLMs discussed here?",
    "What does RAG stand for and what does it do?",
    "What is the difference between short-term and long-term memory?",
    "How does a sliding window differ from summarization?",
    "What is the memory update function formula?",
    "What do accuracy, coherence, and latency measure?",
]


def _run_conversation(
    llm: LocalLLM,
    corpus: str,
    questions: List[str],
    managed: bool,
    max_tokens: int,
    summary_tokens: int,
) -> dict:
    """Run the Q&A conversation and collect per-turn metrics."""
    if managed:
        ctx = SlidingWindowContext(
            max_tokens=max_tokens,
            summary_tokens=summary_tokens,
        )
        ctx.add(corpus)

    history: List[str] = [corpus] if not managed else []
    rows: List[dict] = []
    total_prompt_tokens = 0
    total_latency_ms = 0.0

    for q in questions:
        if managed:
            ctx.add(f"Question: {q}")
            prompt = ctx.render()
            answer, p_tokens, latency = llm.generate(prompt)
            ctx.add(f"Answer: {answer}")
        else:
            prompt = "\n\n".join(history + [q])
            answer, p_tokens, latency = llm.generate(prompt)
            history.append(f"Question: {q}")
            history.append(f"Answer: {answer}")

        total_prompt_tokens += p_tokens
        total_latency_ms += latency
        rows.append({
            "turn": q,
            "prompt_tokens": p_tokens,
            "latency_ms": latency,
            "answer": answer,
        })

    result = {
        "rows": rows,
        "total_prompt_tokens": total_prompt_tokens,
        "total_latency_ms": total_latency_ms,
    }
    if managed:
        result["context_stats"] = ctx.stats()
        result["retention"] = rouge1_recall(ctx.summary, ctx.overflow_source())
    return result


def _print_table(name: str, result: dict) -> None:
    print(f"\n[{name}]")
    print(f"{'turn':<42}{'prompt_tok':>11}{'latency_ms':>12}")
    print("-" * 66)
    for i, row in enumerate(result["rows"], start=1):
        print(f"{'Q' + str(i) + ': ' + row['turn'][:38]:<42}"
              f"{row['prompt_tokens']:>11}"
              f"{row['latency_ms']:>12.1f}")
    print("-" * 66)
    print(f"{'TOTAL':<42}"
          f"{result['total_prompt_tokens']:>11}"
          f"{result['total_latency_ms']:>12.1f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare naive vs sliding-window+summary context for a local LLM"
    )
    ap.add_argument("--model", default="llama3.2",
                    help="Ollama model name (default llama3.2)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="context budget for the managed window (default 256)")
    ap.add_argument("--summary-tokens", type=int, default=96,
                    help="budget for the summarized part (default 96)")
    ap.add_argument("--file", help="optional .txt file to use as the discussion corpus")
    args = ap.parse_args(argv)

    corpus = SAMPLE_DOCUMENT
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            corpus = f.read()

    llm = LocalLLM(model=args.model, host=args.host)
    print(f"Local LLM backend: {llm.backend}"
          + (f" (model={llm.model})" if llm.backend == "ollama" else
             " -- start Ollama to use a real model"))
    print(f"Corpus: {count_tokens(corpus)} tokens, {len(QUESTIONS)} questions")
    print(f"Window config: max={args.max_tokens} tokens, "
          f"summary={args.summary_tokens}")

    naive = _run_conversation(llm, corpus, QUESTIONS, managed=False,
                              max_tokens=args.max_tokens,
                              summary_tokens=args.summary_tokens)

    managed = _run_conversation(llm, corpus, QUESTIONS, managed=True,
                                max_tokens=args.max_tokens,
                                summary_tokens=args.summary_tokens)

    _print_table("NAIVE (full history)", naive)
    _print_table("MANAGED (sliding window + summary)", managed)

    n_tok = naive["total_prompt_tokens"]
    m_tok = managed["total_prompt_tokens"]
    n_lat = naive["total_latency_ms"]
    m_lat = managed["total_latency_ms"]

    print("\n===================== COMPARISON =====================")
    print(f"Prompt tokens : naive={format_tokens(n_tok)}  "
          f"managed={format_tokens(m_tok)}  "
          f"saved={100 * (1 - m_tok / n_tok):.1f}%")
    print(f"Latency (ms)  : naive={n_lat:,.1f}  "
          f"managed={m_lat:,.1f}  "
          f"speedup={n_lat / m_lat:.2f}x")
    if "context_stats" in managed:
        st = managed["context_stats"]
        print(f"Window stats  : max={st['max_tokens']}  "
              f"final_total={st['total_tokens']}  "
              f"compressions={st['compressions']}  "
              f"retention={managed['retention']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
