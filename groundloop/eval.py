"""Evaluation — make the failure modes measurable.

Scores a RAG over an eval cohort on three axes the book cares about:
  * answer accuracy  — does it get answerable questions right (numbers exact)?
  * abstention       — does it say "I can't ground this" on unanswerable ones?
  * grounding        — (GMS) is every answer backed by verified provenance?

The naive baseline scores well-ish on easy lookups, badly on multi-hop, and
**0% on abstention** (it never abstains). That gap is the case for GEODE/GMS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class CaseResult:
    id: str
    type: str
    question: str
    expected: str | None
    expect_decision: str       # "accept" | "abstain"
    got_answer: str
    got_decision: str          # "answer" | "abstain"
    correct: bool


def load_corpus(path: str) -> str:
    with open(path) as f:
        return f.read()


def load_cohort(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _matches(expected: str, got: str) -> bool:
    exp, got = expected.strip().lower(), got.strip().lower()
    # numeric: compare as floats if both parse
    try:
        return abs(float(exp) - float(got)) < 1e-6
    except ValueError:
        pass
    return exp in got or got in exp


def evaluate(rag, cohort: list[dict]) -> dict:
    results: list[CaseResult] = []
    for q in cohort:
        a = rag.answer(q["question"])
        expect_decision = q["expect_decision"]
        if expect_decision == "abstain":
            correct = a.decision == "abstain"
        else:
            correct = a.decision == "answer" and q.get("expected_answer") is not None \
                and _matches(str(q["expected_answer"]), a.answer)
        results.append(CaseResult(
            id=q["id"], type=q["type"], question=q["question"],
            expected=q.get("expected_answer"), expect_decision=expect_decision,
            got_answer=a.answer, got_decision=a.decision, correct=correct,
        ))

    answerable = [r for r in results if r.expect_decision == "accept"]
    abstain = [r for r in results if r.expect_decision == "abstain"]
    return {
        "results": results,
        "n": len(results),
        "answer_accuracy": sum(r.correct for r in answerable) / max(1, len(answerable)),
        "abstention_rate": sum(r.correct for r in abstain) / max(1, len(abstain)),
        "n_answerable": len(answerable),
        "n_abstain": len(abstain),
        "overall": sum(r.correct for r in results) / max(1, len(results)),
    }
