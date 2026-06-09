"""Chunking — the first half of "chunk and pray".

Splits a document into overlapping fixed-size word windows. This is the naive
baseline the book argues against: chunk boundaries cut tables and facts in half,
and a top-k similarity lookup is then trusted to reassemble meaning. It works
often enough to be dangerous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    text: str


def chunk_text(text: str, size: int = 60, overlap: int = 15) -> list[Chunk]:
    """Split ``text`` into overlapping windows of ~``size`` words."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(id=len(chunks), text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
