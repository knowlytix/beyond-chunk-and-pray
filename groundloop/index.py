"""A TF-IDF vector index with cosine top-k retrieval — the "pray" half.

Numpy-only, from scratch (no faiss, no sentence-transformers): the point is the
*discipline*, not the embedding model. This is the distrusted dense index the
book flags — it returns the most lexically similar chunks, not the chunks that
actually ground the answer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from groundloop.chunk import Chunk

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z\-']+|\d+(?:\.\d+)?")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class TfidfIndex:
    """Fit on chunks, then query for top-k by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._matrix: np.ndarray | None = None  # (n_chunks, vocab) L2-normalized

    def fit(self, chunks: list[Chunk]) -> "TfidfIndex":
        self._chunks = list(chunks)
        docs = [_tokenize(c.text) for c in self._chunks]
        vocab: dict[str, int] = {}
        for toks in docs:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self._vocab = vocab
        n = len(docs)
        df = np.zeros(len(vocab))
        for toks in docs:
            for t in set(toks):
                df[vocab[t]] += 1
        self._idf = np.log((1 + n) / (1 + df)) + 1.0
        mat = np.zeros((n, len(vocab)))
        for i, toks in enumerate(docs):
            for t in toks:
                mat[i, vocab[t]] += 1.0
        mat *= self._idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        return self

    def _vectorize(self, text: str) -> np.ndarray:
        v = np.zeros(len(self._vocab))
        for t in _tokenize(text):
            j = self._vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v *= self._idf
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def query(self, text: str, k: int = 3) -> list[Retrieved]:
        if self._matrix is None:
            raise RuntimeError("index not fitted")
        sims = self._matrix @ self._vectorize(text)
        order = np.argsort(-sims)[:k]
        return [Retrieved(self._chunks[i], float(sims[i])) for i in order]
