"""Tests for the open groundloop baseline."""

from __future__ import annotations

import os

from groundloop import NaiveRAG, TfidfIndex, chunk_text, evaluate, load_cohort, load_corpus

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _corpus() -> str:
    return load_corpus(os.path.join(_DATA, "annual_report.md"))


def test_chunking_overlaps_and_covers():
    chunks = chunk_text("one two three four five six", size=3, overlap=1)
    assert chunks
    assert all(c.text for c in chunks)
    # ids are contiguous from 0
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_index_retrieves_relevant_chunk():
    chunks = chunk_text(_corpus(), size=40, overlap=10)
    idx = TfidfIndex().fit(chunks)
    hits = idx.query("Cloud Platform revenue", k=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[-1].score  # sorted by score
    assert any("Cloud Platform" in h.chunk.text for h in hits)


def test_naive_rag_never_abstains():
    rag = NaiveRAG(_corpus())
    # even an unanswerable question gets a (wrong) confident answer
    a = rag.answer("What is management's outlook for fiscal 2026?")
    assert a.decision == "answer"
    assert a.answer  # non-empty
    assert a.sources  # cited some chunk ids


def test_eval_baseline_has_zero_abstention():
    corpus = _corpus()
    cohort = load_cohort(os.path.join(_DATA, "eval_cohort.json"))
    rep = evaluate(NaiveRAG(corpus), cohort)
    # the whole point: the naive baseline cannot abstain
    assert rep["abstention_rate"] == 0.0
    assert rep["n_abstain"] >= 1
    # and it gets at least one easy lookup right (sanity)
    assert rep["answer_accuracy"] > 0.0


def test_eval_report_shape():
    cohort = load_cohort(os.path.join(_DATA, "eval_cohort.json"))
    rep = evaluate(NaiveRAG(_corpus()), cohort)
    assert rep["n"] == len(cohort)
    assert len(rep["results"]) == len(cohort)
    assert 0.0 <= rep["overall"] <= 1.0
