"""groundloop — trustworthy RAG, starting from the naive baseline.

The open package is a genuine "chunk and pray" RAG: chunk a document, build a
TF-IDF index, retrieve top-k, and stuff-and-answer (never abstaining). It is the
deliberate "before" the book *Beyond Chunk and Pray* argues against. The GEODE/GMS
"after" --- triple-mediated retrieval, Exact Numerical Memory, provenance and
abstention --- snaps in via the optional licensed knowlytix backend
(see groundloop.gms).
"""

from groundloop.chunk import Chunk, chunk_text
from groundloop.eval import CaseResult, evaluate, load_cohort, load_corpus
from groundloop.index import Retrieved, TfidfIndex
from groundloop.rag import Answer, NaiveRAG

__version__ = "0.0.1"

__all__ = [
    "Answer",
    "CaseResult",
    "Chunk",
    "NaiveRAG",
    "Retrieved",
    "TfidfIndex",
    "chunk_text",
    "evaluate",
    "load_cohort",
    "load_corpus",
]
