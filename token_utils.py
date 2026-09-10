"""Shared, dependency-free token counting utilities."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap, deterministic token estimator (word-level).

    Good enough for relative comparisons in the evaluation harness; a real
    deployment would use the tokenizer of the specific model.
    """
    return len(_WORD_RE.findall(text))


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)
