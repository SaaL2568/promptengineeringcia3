# Context & Memory Management for LLMs

A Prompt Engineering project implementing two ideas:

1. **Prompt evaluation framework** — inspired by the paper
   *"Enhancing Large Language Model Performance Through Context and Memory Management"*
   (`PromptEngineeringTeam8`): compare models × reasoning approaches × prompt styles
   on accuracy / coherence / latency, plus a four-scope agent memory.
2. **Sliding-window context with summarization** — the core idea: keep the
   context window bounded by summarizing old turns instead of re-sending the
   entire history, and measure the effect on a **real local LLM (Ollama)**.

---

## Requirements

- **Python 3.10+** (developed on 3.14). No third-party packages required — everything uses the standard library.
- **Ollama** running locally (`ollama serve`) with a model pulled. Available here:
  - `llama3.1:8b` (default)
  - `qwen3:8b`
  - `qwen2.5-coder:7b`

If Ollama is not reachable, every tool **falls back to a deterministic simulated
model** so it still runs — but those numbers are **fake** (see below).

---

## IMPORTANT: what is real vs simulated

| Component | Real? | Notes |
|---|---|---|
| `gemini_api.py` scores (accuracy/coherence/latency) | **No — simulated** | Formula-based; not from any real model. Illustrates the paper's framework only. |
| `local_llm.py` (Ollama path) | **Yes** | Real inference; real token counts + nanosecond timings from Ollama. |
| `local_llm.py` (simulated path) | **No** | Used only when Ollama is down. Deterministic fake numbers. |
| `summarizer.py` | **Yes** | Real extractive summarizer (frequency-based sentence scoring). |
| `sliding_context.py` | **Yes** | Real sliding-window + summary logic. |
| `token_utils.count_tokens` | **Approximate** | Word-level estimate, not the model's real tokenizer. (In the Ollama path, prompt tokens come from Ollama itself, so those are exact.) |
| `benchmark_results.*` | **Yes** | Measured from `llama3.1:8b`. Single run; timing is noisy. |

---

## Files

### 1. Prompt evaluation framework (from the paper)
- `gemini_api.py` — simulated `GeminiAPI.evaluateprompt`; models, reasoning approaches (`zero-shot/few-shot/cot/tot`), prompt styles.
- `memory.py` — four-scope agent memory (working / episodic / semantic / procedural) + the `m_t = f(z, x_t, r_t, m_{t-1})` update loop.
- `main.py` — CLI: `python main.py eval` and `python main.py memory`.

### 2. Sliding-window context (the main idea)
- `summarizer.py` — extractive summarizer (no LLM needed).
- `sliding_context.py` — `SlidingWindowContext`: bounded window, summarizes overflow.
- `token_utils.py` — token counting helpers.
- `local_llm.py` — Ollama client (streaming + detailed metrics + simulated fallback).

### 3. Demos & tools
- `local_context.py` — compares naive vs managed context in one run.
- `visualize.py` — turn-by-turn ASCII visualization of the window compressing.
- `chat.py` — terminal chat: each message goes to **both** naive and managed.
- `web_chat.py` — **live web UI** with two panels, tokens streamed as generated.
- `benchmark.py` — real Ollama benchmark (naive vs managed), Ollama timing counters.
- `benchmark_charts.py` — benchmark that exports **CSV + JSON + HTML charts**.

### Assets
- `PromptEngineeringTeam8.docx`, `PromptPPT.pptx` — source paper/slides.
- `benchmark_results.csv`, `.json`, `.html` — latest benchmark output.

---

## How to run

```powershell
# 1) Start Ollama (once)
ollama serve

# Terminal chat: compare naive vs managed live
python chat.py

# Live web UI (two streaming panels) -> http://localhost:8000
python web_chat.py
python web_chat.py --max-tokens 96          # smaller window -> compression sooner

# Visualize the window compressing (offline, instant)
python visualize.py

# Benchmark + export charts
python benchmark_charts.py --turns 12 --max-tokens 128 --num-predict 64

# Paper framework
python main.py eval
python main.py memory
```

---

## Benchmark results (latest run)

`llama3.1:8b`, 8 turns, window 256, summary 96, `num_predict=64`:

| metric | naive | managed | change |
|---|---|---|---|
| total prompt tokens | 4238 | 1949 | **−54.0%** |
| final window tokens | 781 | 197 | bounded |
| compressions | 0 | 13 | — |
| total prompt read ms | 3533 | 4049 | −14.6% (noisy) |

**Interpretation:**
- The **token savings are real and consistent** (≈30–55% depending on length/config).
- **Wall-clock is not faster**: answer *generation* dominates (5–8 s/turn), while prompt reading is sub-second. The benefit is **context size / cost / staying under the context limit**, not latency.
- `prompt_eval_ms` is noisy (cache/warm-up effects) — average over repeats before drawing conclusions.

Charts: open `benchmark_results.html`. Raw data: `benchmark_results.csv`.

---

## Status

**Done**
- [x] Paper's evaluation framework + memory (simulated)
- [x] Extractive summarizer + sliding-window context
- [x] Real Ollama integration (streaming, metrics)
- [x] Terminal chat, live web chat, visualizer
- [x] Benchmark with CSV/JSON/HTML export

**Known limitations**
- `gemini_api.py` metrics are simulated (not real model quality).
- `count_tokens` is approximate (word-based).
- Benchmark is single-run and small-scale; timings noisy.
- Single shared conversation state in the web UI (one local user).
- No automated answer-quality scoring for the managed summary yet.

**Possible next steps**
- Add quality/faithfulness scoring (e.g., ROUGE vs ground truth).
- Repeat runs + medians to reduce timing noise.
- Larger corpora / more turns to widen the naive-vs-managed gap.
- Make the summarizer pluggable (extractive vs LLM-based).
- Persist conversations (the `memory.py` long-term store) across sessions.

---

## Notes on Ollama

Ollama is a **stateless** inference server: it does not remember your
conversation. The client must re-send the whole context every turn. Ollama only
enforces a hard `num_ctx` cap (default ~4096) by silently truncating the front.
So **context management is always the client's job** — which is what
`sliding_context.py` does.
