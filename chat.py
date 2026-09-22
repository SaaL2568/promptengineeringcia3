"""Simple chat interface that sends every message to BOTH context strategies.

  * NAIVE   -- the whole conversation is re-sent every turn (grows forever).
  * MANAGED -- a sliding window + extractive summary keeps the prompt capped.

Type a message and you see both answers side by side, with the prompt size,
prompt-reading time, and generation time for each. Watch the managed window
stay flat while the naive context keeps growing.

Run:
    python chat.py
    python chat.py --model llama3.1:8b --max-tokens 256
    python chat.py --seed-file notes.txt

Commands inside the chat:
    /stats   show the current window state
    /reset   clear both conversations
    /quit    exit
"""

from __future__ import annotations

import argparse
import sys

from local_llm import LocalLLM
from sliding_context import SlidingWindowContext
from token_utils import count_tokens
from visualize import C, c, enable_ansi, shorten


def _print_block(title: str, color: str, metrics: dict, extra: str = "") -> None:
    print(c("+" + "-" * 74, color))
    line = (f"| {title}   prompt={metrics['prompt_tokens']} tok  "
            f"read={metrics['prompt_eval_ms']:.0f}ms  "
            f"gen={metrics['eval_ms']:.0f}ms")
    if extra:
        line += f"  {extra}"
    print(c(line, color))
    print(c("+" + "-" * 74, color))
    print("  " + metrics["response"].strip().replace("\n", "\n  "))
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chat: naive vs sliding-window context")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="managed window limit (default 256)")
    ap.add_argument("--summary-tokens", type=int, default=96)
    ap.add_argument("--num-predict", type=int, default=128,
                    help="max output tokens per answer (default 128)")
    ap.add_argument("--seed-file", help="optional .txt file to load as initial context")
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

    llm = LocalLLM(model=args.model, host=args.host)
    print(c(f"Model: {args.model}   backend: {llm.backend}", C.BOLD))
    if llm.backend != "ollama":
        print(c("  (Ollama not reachable -- using the simulated model)", C.YELLOW))
    print(c("Commands: /stats  /reset  /quit", C.DIM))
    print()

    corpus = ""
    if args.seed_file:
        with open(args.seed_file, encoding="utf-8") as f:
            corpus = f.read().strip()

    def fresh_state():
        ctx = SlidingWindowContext(max_tokens=args.max_tokens,
                                   summary_tokens=args.summary_tokens)
        history = []
        if corpus:
            ctx.add(f"[Background document]\n{corpus}")
            history.append(f"[Background document]\n{corpus}")
        return ctx, history

    ctx, history = fresh_state()

    while True:
        try:
            user = input(c("You: ", C.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            ctx, history = fresh_state()
            print(c("(conversation reset)", C.DIM))
            continue
        if user == "/stats":
            st = ctx.stats()
            print(c(f"managed window: {st['total_tokens']}/{st['max_tokens']} tok "
                    f"(summary={st['summary_tokens']}, recent={st['recent_tokens']}, "
                    f"compressions={st['compressions']})", C.CYAN))
            print()
            continue

        # ---- NAIVE: send the entire history every time ----
        naive_prompt = "\n\n".join(history + [f"User: {user}"]) + "\nAssistant:"
        naive = llm.generate_full(naive_prompt, max_tokens=args.num_predict)

        # ---- MANAGED: send only the capped window ----
        ctx.add(f"User: {user}")
        managed_prompt = ctx.render() + "\nAssistant:"
        managed = llm.generate_full(managed_prompt, max_tokens=args.num_predict)

        # Update both histories.
        history.append(f"User: {user}")
        history.append(f"Assistant: {naive['response'].strip()}")
        ctx.add(f"Assistant: {managed['response'].strip()}")

        st = ctx.stats()
        _print_block("NAIVE   (full history)", C.RED, naive,
                     extra=f"history_sent={count_tokens(naive_prompt)} tok")
        _print_block("MANAGED (sliding window)", C.CYAN, managed,
                     extra=f"window_after={st['total_tokens']}/{st['max_tokens']} "
                           f"summaries={st['compressions']}")

    print(c("bye", C.DIM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
