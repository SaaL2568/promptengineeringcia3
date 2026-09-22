"""Visualize the sliding-window + summarization context manager.

Run:
    python visualize.py                 # instant, simulated answers
    python visualize.py --real --model llama3.1:8b
    python visualize.py --no-color

It prints, turn by turn, how the NAIVE context keeps growing while the
MANAGED window stays capped -- and shows exactly when a summarization
("compression") happens and what the summary contains.
"""

from __future__ import annotations

import argparse
import sys

from local_context import QUESTIONS, SAMPLE_DOCUMENT
from local_llm import LocalLLM
from sliding_context import SlidingWindowContext
from token_utils import count_tokens

# --------------------------------------------------------------------------- #
# Tiny ANSI color helpers
# --------------------------------------------------------------------------- #
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    enabled = True


def c(text: str, color: str) -> str:
    if not C.enabled:
        return text
    return f"{color}{text}{C.RESET}"


def enable_ansi() -> None:
    # Windows consoles need a nudge to process ANSI escape codes.
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def bar(tokens: int, scale: int, width: int = 44) -> str:
    filled = int(round(width * min(tokens, scale) / scale)) if scale else 0
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def shorten(text: str, limit: int = 96) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# Collect a trace of both strategies
# --------------------------------------------------------------------------- #
def collect(llm: LocalLLM, corpus: str, questions, max_tokens: int,
            summary_tokens: int):
    # --- NAIVE: keep everything -------------------------------------------
    naive_counts = []
    history = [corpus]
    for q in questions:
        prompt = "\n\n".join(history + [q])
        naive_counts.append(count_tokens(prompt))
        answer, _, _ = llm.generate(prompt)
        history.append(f"Question: {q}")
        history.append(f"Answer: {answer}")

    # --- MANAGED: sliding window + summary --------------------------------
    ctx = SlidingWindowContext(max_tokens=max_tokens,
                               summary_tokens=summary_tokens)
    ctx.add(corpus)
    managed = []
    prev_compressions = 0
    for q in questions:
        ctx.add(f"Question: {q}")
        prompt = ctx.render()
        answer, _, _ = llm.generate(prompt)
        ctx.add(f"Answer: {answer}")
        managed.append({
            "tokens": ctx.total_tokens(),
            "summary": ctx.summary,
            "recent": list(ctx.recent),
            "compressed": ctx.compressions - prev_compressions,
            "compressions": ctx.compressions,
        })
        prev_compressions = ctx.compressions

    return naive_counts, managed


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render(llm, corpus, questions, max_tokens, summary_tokens) -> None:
    naive_counts, managed = collect(llm, corpus, questions, max_tokens,
                                    summary_tokens)
    scale = max(max(naive_counts), max_tokens)

    print()
    print(c("=" * 78, C.BOLD))
    print(c("  SLIDING-WINDOW CONTEXT  vs  NAIVE FULL-HISTORY CONTEXT", C.BOLD))
    print(c("=" * 78, C.BOLD))
    print(f"  corpus: {count_tokens(corpus)} tokens   "
          f"window limit: {max_tokens} tokens   "
          f"summary budget: {summary_tokens} tokens")
    print(f"  scale: 1 '#' ~ {scale / 44:.1f} tokens")
    print()

    for i, q in enumerate(questions, 1):
        m = managed[i - 1]
        n_tok = naive_counts[i - 1]
        m_tok = m["tokens"]

        print(c(f"TURN {i}", C.BOLD) + "  " + c(shorten(q, 60), C.DIM))
        print(f"  {c('NAIVE  ', C.RED)} {bar(n_tok, scale)} {n_tok:>4} tok "
              + c("(grows without limit)", C.DIM))
        print(f"  {c('MANAGED', C.CYAN)} {bar(m_tok, scale)} {m_tok:>4} tok "
              + c(f"(capped at {max_tokens})", C.DIM))

        if m["summary"]:
            print(c("    +-- SUMMARY -------------------------------------------", C.DIM))
            print(c("    | " + shorten(m["summary"], 90), C.BLUE))
            print(c("    +-----------------------------------------------------", C.DIM))
        recent_txt = " | ".join(shorten(r, 40) for r in m["recent"][-2:])
        print(c("    recent: " + shorten(recent_txt, 88), C.GREEN))

        if m["compressed"]:
            print(c(f"    >> COMPRESSED {m['compressed']} time(s): oldest turns "
                    f"summarized. total compressions = {m['compressions']}",
                    C.YELLOW))
        print()

    # Final comparison
    n_total = sum(naive_counts)
    m_total = sum(m["tokens"] for m in managed)
    print(c("=" * 78, C.BOLD))
    print(f"  total prompt tokens : naive={n_total}  managed={m_total}  "
          f"saved={100 * (1 - m_total / n_total):.1f}%")
    print(f"  final window        : {managed[-1]['tokens']} tokens "
          f"(naive ended at {naive_counts[-1]})")
    print(f"  summarizations      : {managed[-1]['compressions']}")
    print(c("=" * 78, C.BOLD))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Visualize sliding-window context")
    ap.add_argument("--real", action="store_true",
                    help="use the real Ollama model instead of simulated answers")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--summary-tokens", type=int, default=96)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args(argv)

    if args.no_color:
        C.enabled = False
    else:
        enable_ansi()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.real:
        llm = LocalLLM(model=args.model, host=args.host)
        print(f"backend: {llm.backend} ({args.model})")
    else:
        llm = LocalLLM(model="simulated")
        llm.backend = "simulated"  # force instant offline answers
        print("backend: simulated (instant). Use --real for Ollama.")

    render(llm, SAMPLE_DOCUMENT, QUESTIONS, args.max_tokens, args.summary_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
