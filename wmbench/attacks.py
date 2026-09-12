"""Text-level robustness attacks (optional track)."""

from __future__ import annotations

import random


def word_deletion(text: str, ratio: float, seed: int) -> str:
    """Delete each whitespace-delimited word independently with probability ``ratio``.

    Same rule as MarkLLM's ``WordDeletion`` editor, but seeded per call so resumed or
    array-split runs reproduce the same attacked text.
    """
    if not text:
        return text
    rng = random.Random(seed)
    kept = [w for w in text.split() if rng.random() >= ratio]
    return " ".join(kept)
