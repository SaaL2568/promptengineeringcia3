"""Simulated Gemini API for prompt evaluation.

This module mimics the ``GeminiAPI.evaluateprompt`` interface described in the
paper ("Enhancing Large Language Model Performance Through Context and Memory
Management") without requiring network access or an API key. It is fully
deterministic given a seed so experiments can be reproduced.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
# capability: 0..1, how well the model answers in general.
# base_latency_ms: fixed per-call latency floor (no reasoning overhead).
# context_limit_tokens: maximum size of the working context window.
MODELS: Dict[str, dict] = {
    "gemini-small": {
        "capability": 0.60,
        "base_latency_ms": 120.0,
        "context_limit_tokens": 4096,
    },
    "gemini-medium": {
        "capability": 0.80,
        "base_latency_ms": 320.0,
        "context_limit_tokens": 32768,
    },
    "gemini-large": {
        "capability": 0.95,
        "base_latency_ms": 750.0,
        "context_limit_tokens": 1_000_000,
    },
}

# --------------------------------------------------------------------------- #
# Reasoning approaches
# --------------------------------------------------------------------------- #
# Each approach has an accuracy multiplier, a coherence multiplier, and a
# latency multiplier that captures the cost of deeper reasoning.
APPROACHES: Dict[str, dict] = {
    "zero-shot": {
        "accuracy": 0.70,
        "coherence": 0.66,
        "latency": 1.00,
        "tokens": 1.0,
    },
    "few-shot": {
        "accuracy": 0.85,
        "coherence": 0.80,
        "latency": 1.15,
        "tokens": 1.6,
    },
    "cot": {
        "accuracy": 0.95,
        "coherence": 0.96,
        "latency": 2.20,
        "tokens": 2.8,
    },
    "tot": {
        "accuracy": 0.98,
        "coherence": 0.94,
        "latency": 3.60,
        "tokens": 4.2,
    },
}

# --------------------------------------------------------------------------- #
# Prompt styles
# --------------------------------------------------------------------------- #
STYLES: Dict[str, dict] = {
    "brief": {
        "accuracy": 0.72,
        "coherence": 0.70,
        "detail": 0.2,
        "creativity": 0.2,
        "examples": 0,
    },
    "specific": {
        "accuracy": 0.95,
        "coherence": 0.92,
        "detail": 0.9,
        "creativity": 0.2,
        "examples": 0,
    },
    "creative": {
        "accuracy": 0.78,
        "coherence": 0.68,
        "detail": 0.4,
        "creativity": 0.9,
        "examples": 0,
    },
    "example-based": {
        "accuracy": 0.90,
        "coherence": 0.86,
        "detail": 0.6,
        "creativity": 0.3,
        "examples": 2,
    },
}

# Human-readable templates used to build illustrative prompt text per style.
STYLE_TEMPLATES: Dict[str, str] = {
    "brief": "Answer: {question}",
    "specific": (
        "Answer the following question precisely and concisely, using only "
        "factually correct information. Provide exactly one answer and include "
        "a short justification. Question: {question}"
    ),
    "creative": (
        "Imagine you are a visionary storyteller. Answer the following in an "
        "original, expressive and imaginative way. Question: {question}"
    ),
    "example-based": (
        "Here are two solved examples to guide you.\n"
        "Example 1: {example1}\n"
        "Example 2: {example2}\n"
        "Now, using the same pattern, answer: {question}"
    ),
}

APPROACH_PREFIXES: Dict[str, str] = {
    "zero-shot": "",
    "few-shot": "Learn the format from the examples above, then ",
    "cot": "Think step by step, writing out each reasoning step before the final answer. ",
    "tot": "Explore several candidate lines of reasoning, compare them, then commit to the best answer. ",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class EvaluationResult:
    """Result of a single ``evaluateprompt`` call."""

    model: str
    approach: str
    style: str
    prompt: str
    response: str
    accuracy: float
    coherence: float
    latency_ms: float
    tokens_used: int

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "approach": self.approach,
            "style": self.style,
            "accuracy": round(self.accuracy, 4),
            "coherence": round(self.coherence, 4),
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
        }


def _count_tokens(text: str) -> int:
    """Cheap token estimator (word-level) to avoid external dependencies."""
    return len(re.findall(r"\w+|[^\w\s]", text))


class GeminiAPI:
    """Simulated ``GeminiAPI`` exposing ``evaluateprompt``."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Public interface (mirrors GeminiAPI.evaluateprompt)
    # ------------------------------------------------------------------ #
    def evaluateprompt(
        self,
        prompt: str,
        model: str = "gemini-medium",
        approach: str = "cot",
        style: str = "specific",
    ) -> EvaluationResult:
        """Evaluate a prompt against a (model, approach, style) combination."""
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(MODELS)}.")
        if approach not in APPROACHES:
            raise ValueError(
                f"Unknown approach '{approach}'. Choose from {list(APPROACHES)}."
            )
        if style not in STYLES:
            raise ValueError(f"Unknown style '{style}'. Choose from {list(STYLES)}.")

        model_cfg = MODELS[model]
        approach_cfg = APPROACHES[approach]
        style_cfg = STYLES[style]

        accuracy = self._score_accuracy(model_cfg, approach_cfg, style_cfg)
        coherence = self._score_coherence(model_cfg, approach_cfg, style_cfg)
        latency_ms = self._score_latency(model_cfg, approach_cfg, style_cfg, prompt)
        tokens_used = self._estimate_tokens(approach_cfg, style_cfg, prompt, model_cfg)
        response = self._render_response(prompt, model, approach, style)

        return EvaluationResult(
            model=model,
            approach=approach,
            style=style,
            prompt=prompt,
            response=response,
            accuracy=_clamp(accuracy),
            coherence=_clamp(coherence),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------ #
    # Metric computations
    # ------------------------------------------------------------------ #
    def _score_accuracy(self, model_cfg, approach_cfg, style_cfg) -> float:
        # Higher specificity and examples raise accuracy; pure creativity trades
        # some precision for originality.
        specificity = max(style_cfg["detail"], 1 - style_cfg["creativity"])
        base = model_cfg["capability"] * approach_cfg["accuracy"]
        base *= 0.6 + 0.4 * specificity
        noise = self._rng.uniform(-0.04, 0.04)
        return base + noise

    def _score_coherence(self, model_cfg, approach_cfg, style_cfg) -> float:
        base = model_cfg["capability"] * approach_cfg["coherence"]
        base *= 0.5 + 0.5 * style_cfg["coherence"]
        noise = self._rng.uniform(-0.03, 0.03)
        return base + noise

    def _score_latency(self, model_cfg, approach_cfg, style_cfg, prompt) -> float:
        # Latency grows with reasoning depth, prompt length, and detail requested.
        length_factor = 1.0 + math.log1p(len(prompt)) / 20.0
        detail_factor = 1.0 + 0.5 * style_cfg["detail"]
        latency = (
            model_cfg["base_latency_ms"]
            * approach_cfg["latency"]
            * length_factor
            * detail_factor
        )
        # Simulate the paper's observation that specific prompts lower latency.
        if style_cfg["detail"] >= 0.9:
            latency *= 0.85
        return latency

    def _estimate_tokens(self, approach_cfg, style_cfg, prompt, model_cfg) -> int:
        prompt_tokens = _count_tokens(prompt)
        prompt_tokens += style_cfg["examples"] * 24
        response_tokens = int(
            (20 + 30 * style_cfg["detail"]) * approach_cfg["tokens"]
        )
        total = prompt_tokens + response_tokens
        return min(total, model_cfg["context_limit_tokens"])

    def _render_response(self, prompt, model, approach, style) -> str:
        if approach in ("cot", "tot"):
            reasoning = (
                f"Step 1: parse the request ({model}).\n"
                f"Step 2: evaluate candidate answers.\n"
            )
            if approach == "tot":
                reasoning += (
                    f"Step 3: branch A -> plausible, branch B -> stronger.\n"
                    f"Step 4: select branch B.\n"
                )
            else:
                reasoning += "Step 3: synthesize the final answer.\n"
            return reasoning + f"Final answer: a confident response to '{prompt[:60]}'."
        if approach == "few-shot":
            return f"Following the example format, the answer to '{prompt[:60]}' is produced."
        return f"Answer: a direct response to '{prompt[:60]}'."


# --------------------------------------------------------------------------- #
# Prompt builders (creative / specific / brief / example-based)
# --------------------------------------------------------------------------- #
def build_prompt(question: str, style: str) -> str:
    """Construct a prompt for the given style using the paper's templates."""
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'.")
    return STYLE_TEMPLATES[style].format(
        question=question,
        example1="Q: What is 2+2? -> A: 4",
        example2="Q: Capital of France? -> A: Paris",
    )


def apply_approach(prompt: str, approach: str) -> str:
    """Wrap a prompt with the reasoning directive for the given approach."""
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach '{approach}'.")
    return APPROACH_PREFIXES[approach] + prompt
