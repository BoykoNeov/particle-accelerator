"""Reference-suite configuration.

Applies the Windows/clang-cl fix-up that makes xtrack's JIT C-kernel compilation
work on this toolchain (see ``_xtrack_jit`` for the full diagnosis). The patch is
a no-op on non-Windows platforms and when clang-cl is absent, so the reference
cross-checks skip gracefully where it cannot run.
"""

from __future__ import annotations

import os
import sys

# tests/ dirs are not import packages, so make this directory importable before
# pulling in the sibling fix-up module.
sys.path.insert(0, os.path.dirname(__file__))

import _xtrack_jit  # noqa: E402

_xtrack_jit.apply()
