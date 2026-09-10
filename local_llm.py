"""Local LLM client.

Talks to a local Ollama server over its HTTP API. If Ollama is not running (or
the requested model is unavailable), it transparently falls back to a
deterministic simulated local LLM so the evaluation harness still runs.

Exposes a small, uniform interface::

    llm.generate(prompt, max_tokens=...) -> (response, prompt_tokens, latency_ms)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from token_utils import count_tokens

OLLAMA_HOST = "http://localhost:11434"


class LocalLLM:
    """Generate completions from a local model (Ollama or simulated)."""

    def __init__(self, model: str = "llama3.2", host: str = OLLAMA_HOST,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.backend = self._detect_backend()

    # ------------------------------------------------------------------ #
    def _detect_backend(self) -> str:
        if self._ollama_available():
            return "ollama"
        return "simulated"

    def _ollama_available(self) -> bool:
        try:
            req = urllib.request.Request(
                self.host + "/api/tags", method="GET", headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, int, float]:
        """Return (response, prompt_tokens, latency_ms)."""
        if self.backend == "ollama":
            return self._generate_ollama(prompt, max_tokens)
        return self._generate_simulated(prompt, max_tokens)

    def _generate_ollama(self, prompt: str, max_tokens: int) -> Tuple[str, int, float]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
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
        except urllib.error.URLError as exc:
            # A model may be missing even if the server is up; fall back.
            self.backend = "simulated"
            return self._generate_simulated(prompt, max_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        response = data.get("response", "")
        prompt_tokens = int(data.get("prompt_eval_count", count_tokens(prompt)))
        return response, prompt_tokens, latency_ms

    def _generate_simulated(self, prompt: str, max_tokens: int) -> Tuple[str, int, float]:
        """Deterministic stand-in local model (for offline evaluation)."""
        prompt_tokens = count_tokens(prompt)
        # Simulate a local CPU/GPU model: latency grows roughly linearly with
        # the number of prompt tokens it must process.
        latency_ms = 40.0 + prompt_tokens * 1.8
        last_line = [ln for ln in prompt.strip().splitlines() if ln.strip()]
        question = last_line[-1] if last_line else "the request"
        response = (
            f"(local model: {self.model}, simulated) "
            f"Answer to '{question[:80]}': a concise response synthesized "
            f"from the provided context."
        )
        return response, prompt_tokens, latency_ms
