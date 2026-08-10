"""Run pytest at a lowered process priority so a long suite does not make the desktop sluggish.

This box routinely runs several agent sessions at once, each driving its own ``pytest -n``
across all 16 logical CPUs. At normal priority accsim's suite joins that scrum and everything
— including the desktop — goes to treacle. Child processes inherit the priority class on both
platforms, so setting it once here covers the whole xdist worker pool. Run::

    .venv/Scripts/python.exe scripts/nicepytest.py -n 8          # analytic suite, yields the box
    .venv/Scripts/python.exe scripts/nicepytest.py tests/reference -n 8 -m reference

Every argument is forwarded verbatim to pytest, so this is a drop-in prefix for any invocation.
It changes *scheduling only* — never test selection, tolerances, or worker count. Pick ``-n``
yourself; ``docs/CONVENTIONS.md`` records why the default is 8 rather than ``auto``.

Lowered priority is not the same as fewer workers. Priority makes the suite *yield* a core it is
already holding; a smaller ``-n`` never takes the core in the first place. Under a CPU-bound
competitor only the second one reliably preserves that competitor's throughput, so the two are
complementary rather than alternatives.

The Windows path has a trap worth keeping written down: ``GetCurrentProcess()`` returns the
pseudo-handle ``(HANDLE)-1``. Without an explicit ``restype`` ctypes treats the return as a 32-bit
int, which zero-extends to ``0x00000000FFFFFFFF`` on x64 — an invalid handle. ``SetPriorityClass``
then fails *silently* (it returns 0, and nothing checks it), leaving every worker at normal
priority. The restype must be a pointer-width type and the result must be tested; both are done
below. This mirrors the sibling ``nicepytest.py`` scripts in other projects on this machine,
where the trap was first found.
"""

from __future__ import annotations

import os
import sys

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
POSIX_NICE_INCREMENT = 10


def _lower_priority() -> str:
    """Drop this process's scheduling priority. Returns a label for the log line."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # see module docstring — load-bearing
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]

        if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
        return "BelowNormal"

    # POSIX: os.nice returns the NEW niceness. Raising it needs no privilege; lowering it back
    # would, which is why this is one-way.
    return f"nice {os.nice(POSIX_NICE_INCREMENT)}"


print(f"[nicepytest] priority -> {_lower_priority()}", file=sys.stderr)

import pytest  # noqa: E402  (imported after the priority drop on purpose — it pulls in NumPy/SciPy)

sys.exit(pytest.main(sys.argv[1:]))
