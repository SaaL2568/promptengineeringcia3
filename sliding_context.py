"""Sliding-window context manager with automatic summarization.

Implements the user's idea:

    Keep the context as a *sliding window*. Whenever the total token count
    exceeds a defined limit, the oldest material is compressed into a summary
    (extractive) and the window keeps only the summary + the most recent turns.

This bounds the prompt size sent to the local LLM, which keeps it fast and
memory-light while retaining the gist of earlier context.
"""

from __future__ import annotations

from typing import List

from summarizer import summarize
from token_utils import count_tokens


class SlidingWindowContext:
    """A bounded context buffer that summarizes overflow."""

    def __init__(self, max_tokens: int = 512, summary_tokens: int = 128) -> None:
        self.max_tokens = max_tokens          # total context budget (the limit)
        # The summary must always be smaller than the whole window, otherwise
        # the window can never fit under the limit.
        if summary_tokens >= max_tokens:
            summary_tokens = max(16, max_tokens // 2)
        self.summary_tokens = summary_tokens  # budget for the distilled part

        self.summary = ""                     # distilled older context
        self.recent: List[str] = []           # verbatim turns (newest last)
        self._overflow: List[str] = []        # raw source text being summarized
        self.compressions = 0                 # how many times we summarized

    # ------------------------------------------------------------------ #
    # Size
    # ------------------------------------------------------------------ #
    def summary_token_count(self) -> int:
        return count_tokens(self.summary)

    def recent_token_count(self) -> int:
        return sum(count_tokens(t) for t in self.recent)

    def total_tokens(self) -> int:
        return self.summary_token_count() + self.recent_token_count()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def add(self, text: str) -> None:
        """Add a new turn; summarize the overflow if we exceed the limit."""
        self.recent.append(text)
        self._compress()

    def _compress(self) -> None:
        # While we are over the limit (and there is still something we can
        # evict), move the oldest turn into the overflow and rebuild the
        # summary from it. The summary is capped at summary_tokens.
        while self.total_tokens() > self.max_tokens and len(self.recent) > 1:
            self._overflow.append(self.recent.pop(0))
            self._rebuild_summary(self.summary_tokens)

        # If a single recent turn plus the summary still overflows, shrink the
        # summary further so the window actually respects the limit.
        if self.total_tokens() > self.max_tokens and self._overflow:
            budget = max(8, self.max_tokens - self.recent_token_count())
            self._rebuild_summary(budget)

    def _rebuild_summary(self, budget: int) -> None:
        self.summary = summarize(" ".join(self._overflow), budget=budget)
        self.compressions += 1

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """Build the prompt-ready context string."""
        parts: List[str] = []
        if self.summary:
            parts.append(f"[Summarized earlier context]\n{self.summary}")
        parts.extend(self.recent)
        return "\n\n".join(parts)

    def stats(self) -> dict:
        return {
            "summary_tokens": self.summary_token_count(),
            "recent_tokens": self.recent_token_count(),
            "total_tokens": self.total_tokens(),
            "max_tokens": self.max_tokens,
            "compressions": self.compressions,
        }

    def overflow_source(self) -> str:
        """Return the raw older text that has been folded into the summary."""
        return " ".join(self._overflow)
