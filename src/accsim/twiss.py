"""Linear (Courant-Snyder) optics: matched Twiss, propagation, tunes.

The transverse motion of the uncoupled linear lattice is described per plane by
the Twiss parameters ``(beta, alpha, gamma)`` with ``gamma = (1 + alpha^2)/beta``,
and a phase advance ``mu``. This module extracts the matched (periodic) Twiss
from a one-turn matrix and propagates it element-by-element.

Scope (Stage 1): pure transverse ``x`` and ``y`` from the 2x2 blocks of the 6x6
map. Drifts and quadrupoles neither couple the planes nor produce dispersion, so
the 2x2 reduction is exact here. **Dispersion** (the coupling to ``delta``) is
added with the :class:`~accsim.elements.dipole.Dipole` in a later change.

Conventions (see ``docs/CONVENTIONS.md``):

- The matched beta is positive by construction; the sign of ``sin mu`` is fixed
  by ``beta = M12 / sin mu > 0``.
- Phase is **accumulated continuously** along the lattice (``atan2`` per element),
  never via ``acos`` of the one-turn matrix — the latter only yields the
  *fractional* tune and loses the integer part. ``Q = mu_total / 2 pi``.
- Stability of a plane requires ``|1/2 Tr(block)| < 1`` (``|Tr| < 2``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .coords import PX, PY, X, Y
from .lattice import Lattice


class UnstableLatticeError(ValueError):
    """Raised when a plane's one-turn block is unstable (``|1/2 Tr| >= 1``).

    An unstable plane has no real matched (periodic) beta function — the betatron
    motion grows without bound — so Twiss matching is undefined.
    """


@dataclass(frozen=True)
class Twiss:
    """Courant-Snyder parameters in both planes at one longitudinal position ``s``.

    ``gamma_x``/``gamma_y`` are derived (``gamma = (1 + alpha^2)/beta``) and the
    phases ``mu_x``/``mu_y`` are in radians, accumulated from the lattice start.
    """

    s: float
    beta_x: float
    alpha_x: float
    mu_x: float
    beta_y: float
    alpha_y: float
    mu_y: float

    @property
    def gamma_x(self) -> float:
        return (1.0 + self.alpha_x**2) / self.beta_x

    @property
    def gamma_y(self) -> float:
        return (1.0 + self.alpha_y**2) / self.beta_y


def _matched_block(C: np.ndarray) -> tuple[float, float]:
    """Matched ``(beta, alpha)`` of a stable 2x2 one-turn block ``C``.

    Solves ``C = R(mu)`` in Courant-Snyder form
    ``[[cos mu + alpha sin mu, beta sin mu], [-gamma sin mu, cos mu - alpha sin mu]]``.
    """
    cos_mu = 0.5 * (C[0, 0] + C[1, 1])
    if abs(cos_mu) >= 1.0:
        raise UnstableLatticeError(
            f"unstable plane: |1/2 Tr| = {abs(cos_mu):.6g} >= 1 (no real matched beta)"
        )
    # beta > 0 forces sign(sin mu) = sign(M12); take that root of sin^2 = 1 - cos^2.
    sin_mu = math.copysign(math.sqrt(1.0 - cos_mu * cos_mu), C[0, 1])
    beta = C[0, 1] / sin_mu
    alpha = 0.5 * (C[0, 0] - C[1, 1]) / sin_mu
    return beta, alpha


def _propagate_block(C: np.ndarray, beta: float, alpha: float) -> tuple[float, float, float]:
    """Propagate ``(beta, alpha)`` through a 2x2 block ``C``; return ``(beta1, alpha1, dmu)``.

    Uses the sigma-matrix form ``B1 = C B C^T`` with ``B = [[beta, -alpha],
    [-alpha, gamma]]`` (exact and symplectic-faithful when ``det C = 1``), and the
    phase advance ``dmu = atan2(C12, beta*C11 - alpha*C12)``. For drifts and
    quadrupoles ``C12 >= 0``, so ``dmu in [0, pi)`` and the accumulated phase is
    monotone.
    """
    gamma = (1.0 + alpha * alpha) / beta
    B = np.array([[beta, -alpha], [-alpha, gamma]])
    B1 = C @ B @ C.T
    beta1 = B1[0, 0]
    alpha1 = -B1[0, 1]
    dmu = math.atan2(C[0, 1], beta * C[0, 0] - alpha * C[0, 1])
    if dmu < 0.0:
        dmu += 2.0 * math.pi  # keep phase monotone across the rare C12 < 0 element
    return beta1, alpha1, dmu


def _blocks(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the ``(x, px)`` and ``(y, py)`` 2x2 sub-blocks of a 6x6 map."""
    return M[np.ix_([X, PX], [X, PX])], M[np.ix_([Y, PY], [Y, PY])]


def match_periodic(one_turn: np.ndarray) -> Twiss:
    """Matched (periodic) Twiss at the start of a ring from its one-turn matrix.

    Raises :class:`UnstableLatticeError` if either plane is unstable. Phases are
    set to zero at this reference point.
    """
    cx, cy = _blocks(one_turn)
    beta_x, alpha_x = _matched_block(cx)
    beta_y, alpha_y = _matched_block(cy)
    return Twiss(0.0, beta_x, alpha_x, 0.0, beta_y, alpha_y, 0.0)


def closed_twiss(lattice: Lattice) -> Twiss:
    """Matched Twiss at the entrance of a periodic ``lattice``."""
    return match_periodic(lattice.one_turn_matrix())


def propagate_twiss(lattice: Lattice, twiss0: Twiss) -> list[Twiss]:
    """Twiss at every element boundary, starting from ``twiss0``.

    Returns ``len(lattice) + 1`` points: the entrance, then the exit of each
    element in order. Phase advances accumulate continuously, so the last point's
    ``mu`` over one period equals ``2 pi Q``.
    """
    points = [twiss0]
    s = twiss0.s
    bx, ax, mux = twiss0.beta_x, twiss0.alpha_x, twiss0.mu_x
    by, ay, muy = twiss0.beta_y, twiss0.alpha_y, twiss0.mu_y
    for elem in lattice.elements:
        cx, cy = _blocks(elem.matrix(lattice.ref))
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        mux += dmux
        muy += dmuy
        s += elem.length
        points.append(Twiss(s, bx, ax, mux, by, ay, muy))
    return points


def tunes(lattice: Lattice) -> tuple[float, float]:
    """Cell/ring tunes ``(Qx, Qy) = mu_total / 2 pi`` of a periodic ``lattice``.

    Matches the periodic Twiss, propagates once around, and divides the total
    accumulated phase advance by ``2 pi`` — so this returns the *full* tune
    (integer + fractional), not just the fractional part the one-turn matrix gives.
    """
    end = propagate_twiss(lattice, closed_twiss(lattice))[-1]
    return end.mu_x / (2.0 * math.pi), end.mu_y / (2.0 * math.pi)


def is_stable(one_turn: np.ndarray) -> bool:
    """True if both transverse planes are stable (``|1/2 Tr(block)| < 1``)."""
    cx, cy = _blocks(one_turn)
    return abs(0.5 * (cx[0, 0] + cx[1, 1])) < 1.0 and abs(0.5 * (cy[0, 0] + cy[1, 1])) < 1.0
