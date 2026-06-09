"""Access to the optional, licensed GMS backend (``knowlytix``).

The GEODE-RAG features of this library --- triple-mediated retrieval through a
verified knowledge graph, Exact Numerical Memory, provenance and abstention ---
run on the ``knowlytix`` package. ``knowlytix`` is licensed and distributed
separately (it is not on public PyPI); a license is required.

Knowlytix: https://knowlytix.ai/

The open ``groundloop`` baseline (naive chunk-and-pray RAG) runs without it. Use
:func:`available` to branch, or :func:`require` for a clear, actionable error.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

KNOWLYTIX_URL = "https://knowlytix.ai/"

INSTALL_HINT = (
    "GEODE-RAG features require the licensed 'knowlytix' package, which is "
    "installed separately and is not on public PyPI.\n"
    f"  Obtain a license and the package index from Knowlytix: {KNOWLYTIX_URL}\n"
    "  pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required\n"
    "  pip install 'groundloop[gms]'\n"
    "See the README section 'The GMS upgrade'."
)


def available() -> bool:
    """Return ``True`` if the licensed ``knowlytix`` backend is importable."""
    return importlib.util.find_spec("knowlytix") is not None


def require() -> ModuleType:
    """Import and return ``knowlytix``, or raise a clear error pointing to it."""
    try:
        return importlib.import_module("knowlytix")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(INSTALL_HINT) from exc
