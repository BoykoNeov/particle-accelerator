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

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np
import pytest

from accsim.coords import DELTA, DIM, ZETA
from accsim.taylor import TaylorMap, compose


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


# ---------------------------------------------------------------------------
# P1: the frame change at second order
# ---------------------------------------------------------------------------
#
# ``M R M^-1`` is a similarity transform and stops being the right statement one order
# up, for two reasons the roadmap derived before the milestone was written: a rank-3
# object transforms with one ``M`` and two ``M^-1``, and ``PT`` is a *nonlinear* function
# of ``delta``,
#
#     PT = sqrt((1 + delta)^2 + 1/beta0^2 - 1) - 1/beta0 = beta0 delta + beta0 delta^2 / (2 gamma0^2) + ...
#
# whose quadratic term is ``1250x`` the ``1e-6`` gate at ``gamma0 = 20``. Rather than
# hand-writing the corrected tensor rule, the frame change is itself a second-order map
# and the conversion is two compositions, ``f_accsim = Phi^-1 . g . Phi`` — so the same
# :func:`accsim.taylor.compose` the milestone gates does the transform, and a wrong rule
# would fail the drift, whose ``T`` the analytic suite derives independently.
#
# PTC's sixth variable is ``-T`` (its ``c6_000010`` is ``-L/(beta0^2 gamma0^2)`` where
# MAD-X's ``R56`` is ``+``; pinned in ``test_second_order_map_madx.py``), hence ``time_sign``.


def _pt_of_delta(delta: float, beta0: float) -> tuple[float, float, float]:
    """``PT(delta)`` with its first two derivatives; ``c = 1/beta0^2 - 1 = 1/(beta0 gamma0)^2``."""
    c = 1.0 / beta0**2 - 1.0
    s = math.sqrt((1.0 + delta) ** 2 + c)
    return s - 1.0 / beta0, (1.0 + delta) / s, c / s**3


def _delta_of_pt(pt: float, beta0: float) -> tuple[float, float, float]:
    """``delta(PT) = sqrt(PT^2 + 2 PT/beta0 + 1) - 1`` with its first two derivatives."""
    s = math.sqrt(pt * pt + 2.0 * pt / beta0 + 1.0)
    return s - 1.0, (pt + 1.0 / beta0) / s, (1.0 - 1.0 / beta0**2) / s**3


def frame_maps(
    z0_accsim: np.ndarray, beta0: float, *, time_sign: float = 1.0
) -> tuple[TaylorMap, TaylorMap]:
    """``(Phi, Phi_inv)``: accsim -> MAD-X coordinates and back, each expanded to second order.

    ``Phi`` is expanded about ``z0_accsim`` and ``Phi_inv`` about ``Phi(z0_accsim)``, so
    ``Phi_inv . Phi`` composes without a mismatch. ``T_madx = time_sign * zeta / beta0``
    exactly (both are linear in the arrival time — gated, not assumed, by the drift's
    zeta row); ``PT`` is the nonlinear function above.
    """
    z0 = np.asarray(z0_accsim, dtype=float)
    pt, dpt, ddpt = _pt_of_delta(float(z0[DELTA]), beta0)
    k = z0.copy()
    k[ZETA] = time_sign * z0[ZETA] / beta0
    k[DELTA] = pt
    R = np.eye(DIM)
    R[ZETA, ZETA] = time_sign / beta0
    R[DELTA, DELTA] = dpt
    T = np.zeros((DIM,) * 3)
    T[DELTA, DELTA, DELTA] = 0.5 * ddpt
    phi = TaylorMap(z0, k, R, T)

    d, dd, ddd = _delta_of_pt(pt, beta0)
    k_inv = z0.copy()
    k_inv[DELTA] = d  # == z0[DELTA] to round-off: the round trip
    R_inv = np.eye(DIM)
    R_inv[ZETA, ZETA] = time_sign * beta0
    R_inv[DELTA, DELTA] = dd
    T_inv = np.zeros((DIM,) * 3)
    T_inv[DELTA, DELTA, DELTA] = 0.5 * ddd
    phi_inv = TaylorMap(k, k_inv, R_inv, T_inv)
    return phi, phi_inv


def to_accsim_frame_second_order(
    k_madx: np.ndarray,
    r_madx: np.ndarray,
    t_madx: np.ndarray,
    z0_accsim: np.ndarray,
    beta0: float,
    *,
    time_sign: float = 1.0,
) -> TaylorMap:
    """A MAD-X ``(k, R, T)`` about the accsim point ``z0_accsim``, as an accsim ``TaylorMap``.

    ``k_madx`` is the map's constant part *in MAD-X coordinates* (the image of the
    expansion point); for a map about the design orbit it is zero. The result is
    ``Phi_inv . g . Phi`` with ``Phi_inv`` expanded about ``k_madx`` — the exit point —
    which is what makes the conversion valid for an element map on a steered orbit and
    not only for a one-turn map that closes.
    """
    phi, _ = frame_maps(z0_accsim, beta0, time_sign=time_sign)
    g = TaylorMap(phi.k, np.asarray(k_madx, dtype=float), r_madx, t_madx)
    # Phi_inv must be expanded about g's exit point, in MAD-X coordinates. Its own
    # origin in accsim coordinates is whatever maps to that point; rebuild it there.
    exit_madx = np.asarray(k_madx, dtype=float)
    exit_accsim = exit_madx.copy()
    exit_accsim[ZETA] = time_sign * exit_madx[ZETA] * beta0
    exit_accsim[DELTA] = _delta_of_pt(float(exit_madx[DELTA]), beta0)[0]
    _, phi_inv = frame_maps(exit_accsim, beta0, time_sign=time_sign)
    return compose(phi_inv, compose(g, phi))


def sectormap_rows(
    madx, table: str = "smap"
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{element name: (k, R, T)}`` read live from a ``twiss, sectormap`` table.

    One row per *explicit* element (implicit drifts are folded into the following row,
    so a fixture must declare its drifts). MAD-X stores ``t<i><j><k>`` symmetric in
    ``j, k`` — measured on a thin sextupole, ``t413 = t431`` — the same convention as
    :mod:`accsim.taylor`, so no factor is applied here.
    """
    t = madx.table[table]
    out = {}
    for row, name in enumerate(t.name):
        k = np.array([t[f"k{i + 1}"][row] for i in range(6)], dtype=float)
        R = np.array(
            [[t[f"r{i + 1}{j + 1}"][row] for j in range(6)] for i in range(6)], dtype=float
        )
        T = np.array(
            [
                [[t[f"t{i + 1}{j + 1}{l + 1}"][row] for l in range(6)] for j in range(6)]
                for i in range(6)
            ],
            dtype=float,
        )
        out[str(name).strip()] = (k, R, T)
    return out


def ptc_maptable(madx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(k, R, T)`` decoded from a ``ptc_twiss, maptable``, in **MAD-X's** coordinate order.

    Rows are named ``c<i>_<e1 e2 e3 e4 e5 e6>``. PTC's own variable order is
    ``(x, px, y, py, PT, T')`` — the *momentum-like* variable fifth and the time-like one
    sixth, the reverse of MAD-X's ``(x, px, y, py, T, PT)`` — and its ``T'`` is ``-T``
    (both facts are pinned in ``test_second_order_map_madx.py`` on a bare drift: ``c5``
    is the conserved row, ``c6_000010 = -L/(beta0^2 gamma0^2)``). This reader swaps the
    last two indices so the result sits in MAD-X order, ``(x, px, y, py, T', PT)``; the
    sign of ``T'`` is left to the caller's ``time_sign``.

    A monomial's coefficient is the **sum** over the symmetric pair, so an off-diagonal
    ``T_ijk`` is *half* the ``e_j = e_k = 1`` coefficient and a diagonal one is the
    ``e_j = 2`` coefficient as is — PTC's ``c1_110000`` was measured to be exactly
    ``T[0,0,1] + T[0,1,0]`` of xtrack's symmetric tensor before the milestone was written
    (``docs/ROADMAP.md``, axis P).
    """
    t = madx.table.map_table
    k = np.zeros(6)
    R = np.zeros((6, 6))
    T = np.zeros((6, 6, 6))
    perm = [0, 1, 2, 3, 5, 4]  # PTC index -> MAD-X index
    for name, coef in zip(t.name, t.coef, strict=True):
        label = str(name).strip()
        i = perm[int(label[1]) - 1]
        exps = [int(c) for c in label.split("_")[1]]
        order = sum(exps)
        if order == 0:
            k[i] = coef
        elif order == 1:
            R[i, perm[exps.index(1)]] = coef
        elif order == 2:
            idx = [perm[n] for n, e in enumerate(exps) for _ in range(e)]
            j, m = idx
            if j == m:
                T[i, j, j] = coef
            else:
                T[i, j, m] = T[i, m, j] = 0.5 * coef
    return k, R, T
