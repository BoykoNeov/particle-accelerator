"""Shared MAD-X (cpymad) plumbing for the D3 second-reference cross-checks.

Why a second reference at all
----------------------------
xtrack already pins accsim's maps. But xsuite *deliberately* follows MAD-X's
coordinate conventions, so a convention error the two share by design would not
be caught by adding MAD-X. What D3 genuinely buys is an **independent numerical
implementation** and an independent derivation of the same maps: a sign slip or
arithmetic error in accsim, or a bug in xtrack, has to be reproduced by a
completely separate Fortran codebase to survive. That is the claim these tests
support -- no more.

The longitudinal frame (the part that has teeth)
------------------------------------------------
MAD-X canonical coordinates are ``(x, px, y, py, T, PT)``, *not* accsim's
``(x, px, y, py, zeta, delta)``:

* the transverse 4x4 block shares ordering and normalisation, so it compares
  entrywise with no transform at all;
* ``PT = (E - E0) / (p0 c)`` is an **energy** deviation, whereas ``delta = dp/p0``
  is a **momentum** deviation -- to first order ``PT = beta0 * delta``;
* ``T`` is the path-length/time variable with the opposite scaling, ``zeta =
  beta0 * T``.

So the change of variables from the MAD-X frame to ours is the diagonal
similarity transform :func:`to_accsim_frame`, ``R_us = M R_madx M^-1`` with
``M = diag(1, 1, 1, 1, beta0, 1/beta0)``.

**Both the scale and the sign of this transform were pinned empirically, not
remembered** (see ``docs/CONVENTIONS.md`` -> *MAD-X reference frame*). The scale
came from a drift, where MAD-X reports ``dT/dPT = L/(beta0^2 gamma0^2)`` against
accsim's ``R56 = L/gamma0^2`` -- a ratio of exactly ``beta0^2``, which fixes
``M``. The *sign* cannot be read off a drift, because a drift's only non-zero
longitudinal entry is that diagonal-adjacent term and it is even under a
simultaneous flip of ``T`` and ``PT``. It is pinned instead by the **dipole**,
whose ``R51``/``R52`` (path lengthening with transverse offset and angle) and
``R16``/``R26`` (dispersion) are odd under that flip, and which agrees entrywise
at ~2e-16 with the sign above.

Note what is deliberately *not* done here: the longitudinal block is never
dropped from the comparison. Comparing only the transverse 4x4 would make every
test pass while silently abandoning the ``R56 = L/gamma0^2`` convention that this
project has a standing note about -- precisely the error the gate exists to
catch.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import pytest


def import_madx():
    """The cpymad ``Madx`` class, or skip the test when cpymad is not installed."""
    madx_mod = pytest.importorskip("cpymad.madx", reason="cpymad (MAD-X) not installed")
    return madx_mod.Madx


@contextmanager
def madx_session() -> Iterator:
    """A quiet MAD-X subprocess, always torn down.

    cpymad runs MAD-X out of process (via minrpc), so an un-closed session leaks
    a child process for the rest of the pytest run. Skips rather than fails if
    the subprocess cannot be launched at all -- consistent with how the xtrack
    checks treat an unavailable JIT toolchain.
    """
    Madx = import_madx()
    try:
        m = Madx(stdout=False)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"MAD-X subprocess unavailable: {type(exc).__name__}: {exc}")
    try:
        yield m
    finally:
        m.quit()


def twiss_rmatrix(madx, sequence: str) -> np.ndarray:
    """The accumulated 6x6 R-matrix at the end of ``sequence``, in MAD-X's frame.

    Read live out of the ``twiss`` table's ``re<i><j>`` columns -- never
    transcribed, and never borrowed from the xtrack tests' expected values, so
    the two references stay genuinely independent.
    """
    table = madx.table.twiss
    return np.array(
        [[table[f"re{i + 1}{j + 1}"][-1] for j in range(6)] for i in range(6)],
        dtype=float,
    )


def to_accsim_frame(r_madx: np.ndarray, beta0: float) -> np.ndarray:
    """Change of variables ``(x,px,y,py,T,PT) -> (x,px,y,py,zeta,delta)``.

    See the module docstring for how ``M`` was pinned. The transverse 4x4 is
    untouched by construction; only the longitudinal row/column are rescaled.
    """
    m = np.diag([1.0, 1.0, 1.0, 1.0, beta0, 1.0 / beta0])
    return m @ r_madx @ np.linalg.inv(m)


def beam_beta0(madx, sequence: str) -> float:
    """``beta0`` of the beam attached to ``sequence``, as MAD-X computed it."""
    return float(madx.sequence[sequence].beam.beta)


class ElementMap(NamedTuple):
    """A MAD-X element map in both frames, so tests can assert against either.

    ``madx`` is the untouched ``(x, px, y, py, T, PT)`` matrix; ``accsim`` is the
    same map after :func:`to_accsim_frame`. Keeping the raw one reachable is what
    lets a test pin the *energy-vs-momentum* difference against a real number
    instead of asserting a tautology about its own transform.
    """

    madx: np.ndarray
    accsim: np.ndarray
    beta0: float


def single_element_rmatrix(element: str, length: float, particle: str, gamma0: float) -> ElementMap:
    """6x6 R-matrix of a one-element MAD-X sequence, in both frames.

    ``element`` is a MAD-X element definition body (e.g. ``"quadrupole, l=0.5,
    k1=1.2"``). The element is centred in a sequence of exactly its own length,
    so the accumulated R-matrix at the end of the line *is* that element's map.

    ``betx``/``bety`` are arbitrary here: the sequence is not periodic, and the
    ``re<i><j>`` columns are the transfer map, which does not depend on the
    initial optics handed to ``twiss``.
    """
    with madx_session() as madx:
        madx.input(f"""
            beam, particle={particle}, gamma={gamma0!r};
            el: {element};
            seq: sequence, l={length!r};
              el, at={length / 2.0!r};
            endsequence;
            use, sequence=seq;
            select, flag=twiss, clear;
            twiss, betx=1.0, bety=1.0, rmatrix;
        """)
        raw = twiss_rmatrix(madx, "seq")
        beta0 = beam_beta0(madx, "seq")
        return ElementMap(madx=raw, accsim=to_accsim_frame(raw, beta0), beta0=beta0)
