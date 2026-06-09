"""Demo — naive "chunk and pray" RAG, and why it fails.

Runs the open NaiveRAG baseline over the Northwind annual-report cohort and
prints answer accuracy, abstention rate, and the per-question failure modes.
The baseline gets easy lookups right, botches multi-hop, grabs wrong numbers,
and never abstains on unanswerable questions — the case for GEODE/GMS.

Open baseline: TF-IDF chunk-and-pray (no license). GMS upgrade (optional): when
`knowlytix` is installed, the GEODE path grounds every answer through a verified
graph, gets numbers byte-exact, and abstains rather than guess.

Run:  python demos/naive_rag.py
"""

from __future__ import annotations

import os

import groundloop.gms as gms
from groundloop import NaiveRAG, evaluate, load_cohort, load_corpus

_HERE = os.path.dirname(__file__)
_DATA = os.path.join(_HERE, "..", "data")


def main() -> None:
    print("=" * 70)
    print('DEMO — "chunk and pray" RAG baseline (and why it fails)')
    print("=" * 70)
    print(f"GMS backend available: {gms.available()}\n")

    corpus = load_corpus(os.path.join(_DATA, "annual_report.md"))
    cohort = load_cohort(os.path.join(_DATA, "eval_cohort.json"))
    rep = evaluate(NaiveRAG(corpus), cohort)

    print("SCORES (open baseline)")
    print("-" * 70)
    print(f"  answer accuracy : {rep['answer_accuracy']*100:>3.0f}%   ({rep['n_answerable']} answerable)")
    print(f"  abstention rate : {rep['abstention_rate']*100:>3.0f}%   ({rep['n_abstain']} unanswerable)")
    print(f"  overall         : {rep['overall']*100:>3.0f}%\n")

    print("PER-QUESTION (failure modes)")
    print("-" * 70)
    for r in rep["results"]:
        mark = "✓" if r.correct else "✗"
        note = ""
        if r.expect_decision == "abstain" and not r.correct:
            note = "  ← hallucinated (should abstain)"
        elif r.type == "multi_hop" and not r.correct:
            note = "  ← top-k can't traverse the graph"
        elif not r.correct and r.expected is not None:
            note = "  ← wrong number / chunk"
        print(f"  {mark} [{r.type:<11}] {r.question[:40]:<40} got={r.got_answer[:18]!r}{note}")

    print("\nBASELINE vs GMS")
    print("-" * 70)
    if gms.available():
        print("  GMS active: GEODE grounds each answer through the verified graph,")
        print("  gets numbers byte-exact (ENM), and abstains on the unanswerable ones.")
    else:
        print(f"  The baseline scored {rep['answer_accuracy']*100:.0f}% answer accuracy and "
              f"{rep['abstention_rate']*100:.0f}% abstention.")
        print("  GEODE/GMS grounds every answer with provenance, gets numbers exact,")
        print("  and abstains rather than guess. Install the licensed backend:\n")
        print("  " + gms.INSTALL_HINT.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
