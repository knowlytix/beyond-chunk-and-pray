# SPDX-License-Identifier: Apache-2.0
"""Make `knowlytix` resolve to the GMS-knowlytix *branch* source (latest version).

The tutorial is developed against the unpublished branch of the library, not a
released wheel. Every script and notebook prepends ``KNOWLYTIX_SRC`` to
``sys.path`` so ``import knowlytix.knowledge.rag`` picks up that working tree.

The path is **configurable**: override with the ``KNOWLYTIX_SRC`` environment
variable, or edit ``_DEFAULT`` below, when the library moves (e.g. once it ships
as a wheel, drop this bootstrap entirely).
"""

from __future__ import annotations

import os
import sys

_DEFAULT = ""

KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", _DEFAULT)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def use_branch_library() -> str:
    """Prepend the branch library to sys.path; return the path used."""
    if KNOWLYTIX_SRC and KNOWLYTIX_SRC not in sys.path:
        sys.path.insert(0, KNOWLYTIX_SRC)
    return KNOWLYTIX_SRC


# Notebooks copy these three lines into their first cell instead of importing
# this module (so a notebook is self-contained):
#
#     import os, sys
#     KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")
#     sys.path.insert(0, KNOWLYTIX_SRC)
