"""Agent memory architecture for LLM context and memory management.

Implements the four temporal scopes described in the paper:

* Working Memory   -- ephemeral, high-bandwidth context window (bounded).
* Episodic Memory  -- concrete historical logs (what happened, in sequence).
* Semantic Memory  -- distilled facts / learned conclusions.
* Procedural Memory-- executable skills / rules governing behavior.

The core loop is "write -> manage -> read":

1. ``write``     : store new observations.
2. ``manage``    : prune working memory and consolidate episodic -> semantic.
3. ``read``      : retrieve the most relevant memory for a query.

The evolution of memory is expressed by the paper's update function::

    m_t = f(z, x_t, r_t, m_{t-1})

where ``z`` is the interaction type, ``x_t`` the generated action, ``r_t`` the
observed reaction, and ``m_{t-1}`` the previous memory state.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class EpisodicEntry:
    """A single logged event (episodic memory)."""

    timestamp: float
    interaction_type: str
    action: str
    reaction: str


@dataclass
class SemanticFact:
    """A distilled fact (semantic memory)."""

    subject: str
    predicate: str
    confidence: float
    occurrences: int = 1


@dataclass
class Procedure:
    """An executable skill or rule (procedural memory)."""

    name: str
    trigger: str
    steps: List[str]


@dataclass
class MemoryState:
    """Snapshot of the agent's memory at time ``t`` (the ``m_t`` in the paper)."""

    working: Tuple[str, ...]
    episodic: Tuple[EpisodicEntry, ...]
    semantic: Tuple[SemanticFact, ...]
    procedural: Tuple[Procedure, ...]

    def to_dict(self) -> dict:
        return {
            "working": list(self.working),
            "episodic": [
                {
                    "type": e.interaction_type,
                    "action": e.action,
                    "reaction": e.reaction,
                }
                for e in self.episodic
            ],
            "semantic": [
                {"subject": f.subject, "predicate": f.predicate,
                 "confidence": round(f.confidence, 3)}
                for f in self.semantic
            ],
            "procedural": [p.name for p in self.procedural],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a: str, b: str) -> float:
    """Lightweight semantic similarity (bag-of-words cosine, no deps)."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    fa: Dict[str, int] = {}
    fb: Dict[str, int] = {}
    for t in ta:
        fa[t] = fa.get(t, 0) + 1
    for t in tb:
        fb[t] = fb.get(t, 0) + 1
    dot = sum(fa.get(k, 0) * v for k, v in fb.items())
    na = math.sqrt(sum(v * v for v in fa.values()))
    nb = math.sqrt(sum(v * v for v in fb.values()))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Memory store
# --------------------------------------------------------------------------- #


class MemoryStore:
    """A four-scope memory with a write-manage-read lifecycle."""

    def __init__(
        self,
        working_limit: int = 5,
        semantic_threshold: float = 0.5,
        consolidation_threshold: int = 2,
    ) -> None:
        self.working: Deque[str] = deque(maxlen=working_limit)
        self.episodic: List[EpisodicEntry] = []
        self.semantic: List[SemanticFact] = []
        self.procedural: List[Procedure] = []

        self.working_limit = working_limit
        self.semantic_threshold = semantic_threshold
        self.consolidation_threshold = consolidation_threshold

    # ------------------------------------------------------------------ #
    # WRITE
    # ------------------------------------------------------------------ #
    def write(
        self, interaction_type: str, action: str, reaction: str = ""
    ) -> None:
        """Record a new observation into working and episodic memory."""
        self.working.append(f"[{interaction_type}] {action}")
        self.episodic.append(
            EpisodicEntry(
                timestamp=time.time(),
                interaction_type=interaction_type,
                action=action,
                reaction=reaction,
            )
        )

    # ------------------------------------------------------------------ #
    # MANAGE
    # ------------------------------------------------------------------ #
    def manage(self) -> None:
        """Prune working memory and consolidate episodic -> semantic facts."""
        self._consolidate_semantic()
        self._distill_procedures()
        # Working memory is auto-pruned by its bounded deque; nothing else to do.

    def _consolidate_semantic(self) -> None:
        """Distill recurring patterns in episodic memory into semantic facts."""
        subjects = [e.interaction_type for e in self.episodic]
        for subject in set(subjects):
            count = subjects.count(subject)
            if count >= self.consolidation_threshold:
                self._upsert_fact(subject, "occurred repeatedly", count)

    def _upsert_fact(self, subject: str, predicate: str, occurrences: int) -> None:
        for fact in self.semantic:
            if fact.subject == subject and fact.predicate == predicate:
                fact.occurrences += occurrences
                fact.confidence = min(1.0, fact.confidence + 0.1)
                return
        self.semantic.append(
            SemanticFact(subject=subject, predicate=predicate, confidence=0.5)
        )

    def _distill_procedures(self) -> None:
        """Create a rule when a recurring interaction type is detected."""
        existing = {p.name for p in self.procedural}
        for fact in self.semantic:
            name = f"handle_{fact.subject}"
            if name not in existing and fact.confidence >= 0.6:
                self.procedural.append(
                    Procedure(
                        name=name,
                        trigger=fact.subject,
                        steps=["recognize trigger", "recall related facts",
                               "produce response"],
                    )
                )
                existing.add(name)

    # ------------------------------------------------------------------ #
    # READ
    # ------------------------------------------------------------------ #
    def read(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most relevant memory items for a query."""
        candidates: List[Tuple[float, str]] = []

        for item in self.working:
            candidates.append((_cosine(query, item), f"[working] {item}"))

        for entry in self.episodic:
            text = f"{entry.interaction_type} {entry.action} {entry.reaction}"
            candidates.append((_cosine(query, text), f"[episodic] {text}"))

        for fact in self.semantic:
            text = f"{fact.subject} {fact.predicate}"
            candidates.append((_cosine(query, text), f"[semantic] {text}"))

        candidates.sort(key=lambda c: c[0], reverse=True)
        return [text for score, text in candidates[:top_k] if score > 0.0]

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def state(self) -> MemoryState:
        return MemoryState(
            working=tuple(self.working),
            episodic=tuple(self.episodic),
            semantic=tuple(self.semantic),
            procedural=tuple(self.procedural),
        )


# --------------------------------------------------------------------------- #
# Memory update function:  m_t = f(z, x_t, r_t, m_{t-1})
# --------------------------------------------------------------------------- #
def memory_update(
    store: MemoryStore, interaction_type: str, action: str, reaction: str = ""
) -> MemoryState:
    """Apply the paper's memory update function and return the new state ``m_t``."""
    store.write(interaction_type, action, reaction)
    store.manage()
    return store.state()
