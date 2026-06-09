# Beyond Chunk and Pray

*Building trustworthy RAG with geometric memory.*

Most RAG is "chunk and pray" — split the docs, embed them, retrieve the top-k by
similarity, and pray the model grounds its answer instead of hallucinating. This
is the open alternative. Retrieval is **triple-mediated**: questions are answered
*through* a verified knowledge graph with provenance, not by a vector lookup the
generator is trusted to use well. The dense index is **distrusted** — optional,
off by default, flagged when used. Every answer is **verifiable and cited**,
numbers are **byte-exact**, and the system **abstains rather than guess** when it
can't ground a claim.

The baseline runs with **no license and no GMS**. For production-grade grounding —
geometric retrieval, Exact Numerical Memory, contradiction detection, signed
answers — the optional **GMS backend** ([`knowlytix`](https://knowlytix.ai/))
snaps in via a lazy seam. Clone it, run the baseline, see exactly where GMS lifts
grounding and recall.

> Part of the **"Beyond … and Pray"** series:
> [governed agents](https://github.com/knowlytix/beyond-prompt-and-pray) ·
> [trustworthy RAG](https://github.com/knowlytix/beyond-chunk-and-pray) ·
> [test & validate](https://github.com/knowlytix/beyond-ship-and-pray) ·
> [LLMs from scratch](https://github.com/knowlytix/llm-from-scratch)

> **Status:** the open `groundloop` baseline (naive chunk→embed→top-k RAG, the
> deliberate "before") is under construction — it is the contrast that motivates
> the GMS "after". See the carve-out manifest in the planning notes.

## What's inside

- **Grounded retrieval** — answers mediated through a verified knowledge graph
- **Provenance + citations** — every claim traces to its source
- **Byte-exact numbers** — Exact Numerical Memory, not "close enough"
- **Abstention** — says "I can't ground that" instead of guessing
- **Distrusted dense index** — optional, flagged when used
- **GMS-optional** — baseline runs free; geometric guarantees via `knowlytix`

## Install

```bash
pip install groundloop                # naive RAG baseline (the "before")
pip install "groundloop[ml]"          # + embedders / open-weight models
pip install "groundloop[gms]"         # + the licensed GMS backend (knowlytix)
```

## The GMS upgrade (open-core)

`groundloop` runs fully without a license. GEODE-RAG — geometric retrieval, Exact
Numerical Memory, provenance — requires the licensed
[`knowlytix`](https://knowlytix.ai/) package, imported lazily:

```python
import groundloop.gms as gms
gms.available()   # True if the licensed backend is installed
```

The production-grade, GMS-native edition is the *Beyond Chunk and Pray, Pro
Edition* — see [knowlytix.ai](https://knowlytix.ai/).

## License

Apache-2.0. © 2026 Knowlytix. Authored by A. Sudjianto.
