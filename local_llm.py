"""Local LLM client.

Talks to a local Ollama server over its HTTP API. If Ollama is not running (or
the requested model is unavailable), it transparently falls back to a
deterministic simulated local LLM so the evaluation harness still runs.

Two interfaces:

    llm.generate(prompt, max_tokens=...)       -> (response, prompt_tokens, latency_ms)
    llm.generate_full(prompt, max_tokens=...)  -> dict of detailed metrics

``generate_full`` exposes Ollama's own timing breakdown, which is far more
reliable than wall-clock time for benchmarking:

    prompt_eval_ms : time spent READING the prompt (grows with prompt size)
    eval_ms        : time spent GENERATING the answer
    load_ms        : model load time (should be ~0 after a warm-up call)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from token_utils import count_tokens

OLLAMA_HOST = "http://localhost:11434"


def _ns_to_ms(value) -> float:
    try:
        return float(value) / 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


class LocalLLM:
    """Generate completions from a local model (Ollama or simulated)."""

    def __init__(self, model: str = "llama3.2", host: str = OLLAMA_HOST,
                 timeout: float = 600.0, think: Optional[bool] = None) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think
        self.backend = self._detect_backend()

    # ------------------------------------------------------------------ #
    def _detect_backend(self) -> str:
        if self._ollama_available():
            return "ollama"
        return "simulated"

    def _ollama_available(self) -> bool:
        try:
            req = urllib.request.Request(
                self.host + "/api/tags", method="GET",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Simple interface (kept for existing callers)
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, int, float]:
        """Return (response, prompt_tokens, latency_ms)."""
        m = self.generate_full(prompt, max_tokens)
        return m["response"], m["prompt_tokens"], m["wall_ms"]

    # ------------------------------------------------------------------ #
    # Detailed interface
    # ------------------------------------------------------------------ #
    def generate_full(self, prompt: str, max_tokens: int = 256) -> dict:
        if self.backend == "ollama":
            return self._ollama_full(prompt, max_tokens)
        return self._simulated_full(prompt, max_tokens)

    def _ollama_full(self, prompt: str, max_tokens: int) -> dict:
        options = {"num_predict": max_tokens}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self.think is not None:
            payload["think"] = self.think
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError:
            self.backend = "simulated"
            return self._simulated_full(prompt, max_tokens)
        wall_ms = (time.perf_counter() - start) * 1000.0

        return {
            "response": data.get("response", ""),
            "prompt_tokens": int(data.get("prompt_eval_count", count_tokens(prompt))),
            "eval_tokens": int(data.get("eval_count", 0)),
            "prompt_eval_ms": _ns_to_ms(data.get("prompt_eval_duration")),
            "eval_ms": _ns_to_ms(data.get("eval_duration")),
            "load_ms": _ns_to_ms(data.get("load_duration")),
            "total_ms": _ns_to_ms(data.get("total_duration")),
            "wall_ms": wall_ms,
            "backend": "ollama",
        }

    def _simulated_full(self, prompt: str, max_tokens: int) -> dict:
        """Deterministic stand-in local model (for offline evaluation)."""
        prompt_tokens = count_tokens(prompt)
        eval_tokens = min(20 + prompt_tokens // 20, max_tokens)
        # Simulate: prompt reading is cheap-ish, generation dominates.
        prompt_eval_ms = 5.0 + prompt_tokens * 0.6
        eval_ms = 30.0 + eval_tokens * 4.0
        lines = [ln for ln in prompt.strip().splitlines()
                 if ln.strip() and not ln.strip().endswith("Assistant:")]
        question = lines[-1] if lines else "the request"
        response = (
            f"(local model: {self.model}, simulated) "
            f"Answer to '{question[:80]}': a concise response synthesized "
            f"from the provided context."
        )
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "eval_tokens": eval_tokens,
            "prompt_eval_ms": prompt_eval_ms,
            "eval_ms": eval_ms,
            "load_ms": 0.0,
            "total_ms": prompt_eval_ms + eval_ms,
            "wall_ms": prompt_eval_ms + eval_ms,
            "backend": "simulated",
        }

    # ------------------------------------------------------------------ #
    # Streaming interface
    # ------------------------------------------------------------------ #
    def stream_generate(self, prompt: str, max_tokens: int = 256):
        """Yield {'type':'token','text':...} events, then one 'done' event."""
        if self.backend == "ollama":
            yield from self._ollama_stream(prompt, max_tokens)
        else:
            yield from self._simulated_stream(prompt, max_tokens)

    def _ollama_stream(self, prompt: str, max_tokens: int):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.URLError:
            self.backend = "simulated"
            yield from self._simulated_stream(prompt, max_tokens)
            return

        with resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield {"type": "token", "text": chunk}
                if data.get("done"):
                    yield {
                        "type": "done",
                        "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                        "eval_tokens": int(data.get("eval_count", 0)),
                        "prompt_eval_ms": _ns_to_ms(data.get("prompt_eval_duration")),
                        "eval_ms": _ns_to_ms(data.get("eval_duration")),
                        "load_ms": _ns_to_ms(data.get("load_duration")),
                    }
                    return

    def _simulated_stream(self, prompt: str, max_tokens: int):
        m = self._simulated_full(prompt, max_tokens)
        words = m["response"].split()
        for i, w in enumerate(words):
            yield {"type": "token", "text": (w if i == 0 else " " + w)}
            time.sleep(0.02)
        yield {
            "type": "done",
            "prompt_tokens": m["prompt_tokens"],
            "eval_tokens": m["eval_tokens"],
            "prompt_eval_ms": m["prompt_eval_ms"],
            "eval_ms": m["eval_ms"],
            "load_ms": m["load_ms"],
        }

    # ------------------------------------------------------------------ #
    def warmup(self) -> float:
        """Load the model into memory; return load time in ms."""
        m = self.generate_full("Hello", max_tokens=1)
        return m.get("load_ms", 0.0)
