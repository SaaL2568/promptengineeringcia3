"""Extractive text summarization (no LLM required).

Uses a lightweight TextRank-style sentence-scoring approach: sentences are
scored by the average frequency of their content words, then the top-scoring
sentences are selected (in original order) until a token budget is met.

This keeps summarization instant, local, and deterministic -- exactly what we
want for compressing the overflow of a sliding context window.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from token_utils import count_tokens

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _score_sentences(sentences: List[str]) -> List[float]:
    """Score each sentence by mean content-word frequency (TextRank-lite)."""
    freq: dict = {}
    word_lists: List[List[str]] = []
    for s in sentences:
        words = _WORD_RE.findall(s.lower())
        word_lists.append(words)
        for w in words:
            freq[w] = freq.get(w, 0) + 1

    scores: List[float] = []
    for words in word_lists:
        if not words:
            scores.append(0.0)
            continue
        scores.append(sum(freq[w] for w in words) / len(words))
    return scores


def summarize(text: str, budget: int = 128, top_k: int = 0) -> str:
    """Return an extractive summary of ``text`` within ``budget`` tokens.

    If ``top_k`` is set, that many top sentences are kept regardless of the
    token budget (still subject to the budget ceiling).
    """
    text = text.strip()
    if not text:
        return ""
    sentences = split_sentences(text)
    if not sentences:
        return text
    if count_tokens(text) <= budget:
        return text

    scores = _score_sentences(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)

    selected: List[Tuple[int, str]] = []
    used = 0
    for idx in ranked:
        if top_k and len(selected) >= top_k:
            break
        cost = count_tokens(sentences[idx])
        if used + cost > budget:
            continue
        selected.append((idx, sentences[idx]))
        used += cost

    # Reorder by original sentence position to preserve narrative flow.
    selected.sort(key=lambda pair: pair[0])
    return " ".join(s for _, s in selected)


def rouge1_recall(summary: str, source: str) -> float:
    """Fraction of source word-types retained in the summary (quality proxy)."""
    src = set(_WORD_RE.findall(source.lower()))
    summ = set(_WORD_RE.findall(summary.lower()))
    if not src:
        return 1.0
    return len(src & summ) / len(src)
