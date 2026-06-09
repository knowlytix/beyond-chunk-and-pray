# Notebooks — Beyond Chunk and Pray

The chapter notebooks for the trustworthy-RAG book: geometric memory,
provenance, triple-mediated retrieval, grounded synthesis, self-verification,
abstention, calibration, and the annual-report capstone.

## ⚠️ These are the Pro-tier (licensed) chapters

Unlike the agent-side repos, these notebooks are **GMS-native end-to-end** —
every one imports the licensed `knowlytix` backend. They are **not** built on the
open `groundloop` baseline (that's the free "chunk and pray" demo in
[`../demos`](../demos) / the `groundloop` package — the *before* the book argues
against). These notebooks are the *after*, and they require:

- the licensed [`knowlytix`](https://knowlytix.ai/) backend (Python 3.12), and
- a trained GMS store (the annual-report store).

```bash
# 1. licensed backend (see knowlytix install docs) + this repo's deps
pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required
export KNOWLYTIX_LICENSE_KEY=...        # your license key

# 2. build the trained store the notebooks query
python scripts/build_store.py          # writes data/gms_annual_report_store/

# 3. run
jupyter lab
```

`book_kit.py` (repo root) is the shared helper the notebooks import — it resolves
the corpus/store/cohort paths and loads the trained store with the exact config
it was built with.

## Notes

- **The store is not shipped.** `data/gms_annual_report_store/` is gitignored
  (licensed artifact). Rebuild it with `scripts/build_store.py` against your
  `knowlytix` version — a prebuilt store from a different `knowlytix` build may
  not load (the model shape is version-specific).
- The open corpus (`data/annual_report.md`, `corpus_facts.md`,
  `eval_cohort.json`) is shared with the `groundloop` baseline demo.
- For the free, runnable "before": see `../demos/naive_rag.py` and the
  `groundloop` package.
