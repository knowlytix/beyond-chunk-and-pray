"""NaiveRAG — chunk → embed → top-k → stuff-and-pray.

The deliberate "before" the book argues against. It retrieves the most similar
chunks and extracts a naive answer from them. Crucially it **never abstains**:
asked an unanswerable question, it still emits a confident answer. It has no
provenance, no number verification, and no notion of "I can't ground this" ---
exactly the failure modes GEODE/GMS fixes (see groundloop.gms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from groundloop.chunk import chunk_text
from groundloop.index import TfidfIndex

_NUM = re.compile(r"\d+(?:\.\d+)?")
_NUMERIC_CUE = re.compile(r"\b(revenue|income|equity|headcount|how many|topline|total|number)\b", re.I)
_STOP = {"the", "a", "an", "is", "are", "was", "of", "what", "who", "in", "for", "to", "and"}


@dataclass
class Answer:
    question: str
    answer: str
    decision: str           # "answer" | "abstain" — baseline always "answer"
    sources: list[int] = field(default_factory=list)
    context: str = ""


class NaiveRAG:
    def __init__(self, corpus: str, chunk_size: int = 60, overlap: int = 15, k: int = 3) -> None:
        self._chunks = chunk_text(corpus, size=chunk_size, overlap=overlap)
        self._index = TfidfIndex().fit(self._chunks)
        self._k = k

    def answer(self, question: str) -> Answer:
        hits = self._index.query(question, k=self._k)
        context = " ".join(h.chunk.text for h in hits)
        sources = [h.chunk.id for h in hits]
        top = hits[0].chunk.text if hits else ""

        # Naive extraction. NO abstention — it always returns *something*.
        if _NUMERIC_CUE.search(question):
            qterms = {w for w in re.findall(r"[a-z]+", question.lower()) if w not in _STOP}
            best_span, best_overlap = top, -1
            for line in re.split(r"(?<=[.!?])\s+|\n|\|", top):
                ov = len(qterms & set(re.findall(r"[a-z]+", line.lower())))
                if ov > best_overlap and _NUM.search(line):
                    best_span, best_overlap = line, ov
            m = _NUM.search(best_span)
            ans = m.group(0) if m else (top[:80] if top else "unknown")
        else:
            ans = re.split(r"(?<=[.!?])\s+", top)[0][:120] if top else "unknown"

        return Answer(question=question, answer=ans, decision="answer",
                      sources=sources, context=context)
