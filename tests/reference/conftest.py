"""Reference-suite configuration.

Applies the Windows/clang-cl fix-up that makes xtrack's JIT C-kernel compilation
work on this toolchain (see ``_xtrack_jit`` for the full diagnosis). The patch is
a no-op on non-Windows platforms and when clang-cl is absent, so the reference
cross-checks skip gracefully where it cannot run.
"""

from __future__ import annotations

import os
import sys

# xtrack >= 0.111 refuses to JIT-compile its C kernels unless told to (it expects the
# ``xsuite`` umbrella's prebuilt ones, which do not build on Python 3.14 — see
# ``docs/CONVENTIONS.md`` -> *Toolchain*). Opt in before xtrack is imported anywhere;
# an explicit environment setting wins.
os.environ.setdefault("XSUITE_ALLOW_KERNEL_COMPILATION", "1")

# tests/ dirs are not import packages, so make this directory importable before
# pulling in the sibling fix-up module.
sys.path.insert(0, os.path.dirname(__file__))

import _xtrack_jit  # noqa: E402

_xtrack_jit.apply()
