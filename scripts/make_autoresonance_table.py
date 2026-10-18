"""Regenerate ``_AUTORESONANCE_MU_TH`` in ``src/accsim/twiss.py`` (W7, damped autoresonance).

The capture threshold ``mu_th(gamma)`` of the damped universal equation

    i Psi' + (tau - |Psi|^2) Psi = mu - i gamma Psi

on ``gamma = 0, 0.0025, ..., 0.1``. Started at rest with ``mu`` ramped on over ``tau in [-80,
-40]``; captured iff ``|Psi|^2 > tau / 2`` at ``tau_read = min(40, 0.7 |Psi|_+^2)``, with
``|Psi|_+ = (mu + sqrt(mu^2 - 2 gamma)) / (2 gamma)`` the energy balance's upper root, and no
capture below ``mu = sqrt(2 gamma)`` or with ``tau_read < 8``. Each entry is bisected from
``[0.40, 0.60]`` to ``1e-7``. The first printed entry is ``mu_c`` (0.41060 to ``5e-6``); the
package writes ``_AUTORESONANCE_MU_C`` there instead.
``tests/analytic/test_damped_autoresonance.py`` re-derives four entries and three spline
midpoints. Run (about two minutes on four workers)::

    .venv/Scripts/python.exe scripts/make_autoresonance_table.py

It prints the tuple to paste. The worker pool inherits the lowered priority set here.
"""

from __future__ import annotations

import math
import os
import sys
from multiprocessing import Pool

from scipy.integrate import solve_ivp

STEP, N = 0.0025, 41


def _lower_priority() -> None:
    """``scripts/nicepytest.py``'s priority drop (see its docstring for the restype trap)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000):
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
    else:
        os.nice(10)


def captured(mu: float, gamma: float) -> bool:
    if gamma > 0.0:
        disc = mu * mu - 2.0 * gamma
        if disc <= 0.0:
            return False
        t_read = min(40.0, 0.7 * ((mu + math.sqrt(disc)) / (2.0 * gamma)) ** 2)
        if t_read < 8.0:
            return False
    else:
        t_read = 40.0

    def rhs(t: float, y: list[float]) -> list[float]:
        psi = y[0] + 1j * y[1]
        m = mu * min(1.0, max(0.0, (t + 80.0) / 40.0))
        dpsi = -1j * (m - (t - abs(psi) ** 2) * psi) - gamma * psi
        return [dpsi.real, dpsi.imag]

    sol = solve_ivp(rhs, (-80.0, t_read), [0.0, 0.0], rtol=1e-10, atol=1e-12, method="DOP853")
    return sol.y[0, -1] ** 2 + sol.y[1, -1] ** 2 > 0.5 * t_read


def mu_th(gamma: float) -> float:
    lo, hi = 0.40, 0.60
    assert not captured(lo, gamma) and captured(hi, gamma)
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if captured(mid, gamma) else (mid, hi)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    _lower_priority()
    with Pool(4) as pool:
        values = pool.map(mu_th, [STEP * k for k in range(N)])
    rows = [", ".join(f"{v:.7f}" for v in values[i : i + 6]) for i in range(0, N, 6)]
    print("_AUTORESONANCE_MU_TH = (\n" + "".join(f"    {r},\n" for r in rows) + ")")
