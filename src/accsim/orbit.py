"""Closed orbit: where the beam actually goes when the machine is imperfect.

Every optics quantity in the package so far — beta, tune, chromaticity,
dispersion — describes motion *about* the design orbit and assumes the beam is
on it. A real machine never is: a steering error, a misaligned quadrupole or a
deliberate :class:`~accsim.elements.corrector.Corrector` deflects the beam, and
the trajectory that closes on itself turn after turn is displaced. That
trajectory is the **closed orbit**, and steering it back is the single most
common operational task in an accelerator.

**The one equation.** A corrector makes the one-turn map affine,
``x -> M x + k`` (see :meth:`accsim.Lattice.transfer_map`). An orbit closes when
it is a fixed point of that map,

    x_co = M4 x_co + k4    =>    (I - M4) x_co = k4,

solved on the 4D transverse subspace ``(x, px, y, py)`` at ``delta = 0``. This is
*literally* the same solve as the matched dispersion
(:func:`~accsim.twiss._matched_dispersion`, ``D = (I - M4)^-1 d``); only the
inhomogeneity differs — there it is the map's ``delta`` column, here the
accumulated corrector kicks. The dispersion is nothing but the closed orbit of an
off-momentum particle, so sharing the algebra is the physics, not a shortcut. By
linearity a particle at momentum ``delta`` rides ``x_co + D * delta``.

``I - M4`` is singular exactly when the map has an eigenvalue 1 — an **integer
tune**, where a kick's effect adds up in phase turn after turn and nothing
closes. The resulting ``1/sin(pi Q)`` in the single-kick closed form

    x_co(s) = theta sqrt(beta_k beta(s)) / (2 sin(pi Q)) * cos(dpsi - pi Q)

is the same statement said with Twiss parameters. That form is a *consequence*
here, used in the tests as an independent reference; the module always solves the
fixed point.

**Scope, stated plainly.** This is linear, ``delta = 0`` orbit theory:

- Correctors change the orbit, never the optics — their linear map is the
  identity, so beta and the tunes are untouched (asserted in the analytic suite).
- **Sextupole feed-down is out of scope.** A sextupole's linear map is a drift
  only because its Jacobian is taken at ``(x, y) = 0``; on a *distorted* orbit it
  feeds down to a quadrupole (and a dipole) kick, so a real machine's optics do
  respond to the orbit. Nothing here models that. The claim "correctors do not
  move the optics" is a linear-order, on-axis-sextupole statement.
- Misalignments are not modelled as such. A quadrupole displaced by ``dx``
  produces a kick ``-k1 L dx``; represent it by placing an explicit
  :class:`~accsim.elements.corrector.Corrector` of that angle.
"""

from __future__ import annotations

import numpy as np

from .coords import PX, PY, X, Y
from .lattice import Lattice

_TRANSVERSE = [X, PX, Y, PY]  # the 4D transverse subspace (x, px, y, py)

# Above this condition number the fixed-point solve has lost every significant
# digit: the lattice is on (or numerically indistinguishable from) an integer
# resonance. Chosen well below 1/eps ~ 4.5e15 so the failure is reported rather
# than returned as a huge but meaningless orbit.
_COND_LIMIT = 1e12


class ClosedOrbitError(ValueError):
    """Raised when no closed orbit exists (or the solve for it is meaningless).

    The fixed-point condition ``(I - M4) x = k4`` fails exactly when ``M4`` has an
    eigenvalue 1 — an **integer tune** in one plane. Physically a kick then
    repeats in phase every turn and the excursion grows without bound instead of
    closing; the machine has no closed orbit to find.
    """


def _affine_4d(M: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The transverse ``(4x4, 4)`` part of a 6D affine map ``(M, k)``."""
    return M[np.ix_(_TRANSVERSE, _TRANSVERSE)], k[_TRANSVERSE]


def closed_orbit(lattice: Lattice) -> np.ndarray:
    """Closed orbit ``(x, px, y, py)`` at the entrance of a periodic ``lattice``.

    Solves the fixed point ``(I - M4) x_co = k4`` of the one-turn affine map. A
    lattice with no :class:`~accsim.elements.corrector.Corrector` has ``k4 = 0``
    and returns exactly zero — the design orbit.

    Raises :class:`ClosedOrbitError` on an integer tune, where ``I - M4`` is
    singular and no orbit closes.
    """
    m4, k4 = _affine_4d(*lattice.one_turn_map())
    if not k4.any():
        return np.zeros(4)  # a perfect machine sits exactly on the design orbit
    A = np.eye(4) - m4
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise ClosedOrbitError(
            f"no closed orbit: (I - M4) is singular to working precision "
            f"(condition number {cond:.3g}), i.e. the lattice is on an **integer** "
            "tune in at least one plane, where a kick repeats in phase every turn "
            "and the excursion never closes"
        )
    return np.linalg.solve(A, k4)


def propagate_orbit(lattice: Lattice, orbit0: np.ndarray | None = None) -> list[np.ndarray]:
    """Orbit ``(x, px, y, py)`` at every element boundary.

    Returns ``len(lattice) + 1`` points — the entrance, then the exit of each
    element in order — mirroring
    :func:`~accsim.twiss.propagate_twiss`. ``orbit0`` defaults to
    :func:`closed_orbit`, in which case the last point equals the first (that is
    what "closed" means); pass an explicit start to follow a **trajectory** down a
    transfer line instead.

    Each step is the element's own affine map ``x -> M4 x + k4``, so a corrector
    shows up as a kink in ``px`` and the orbit only starts moving in ``x``
    downstream of it.
    """
    o = closed_orbit(lattice) if orbit0 is None else np.asarray(orbit0, dtype=float)
    if o.shape != (4,):
        raise ValueError(f"orbit0 must be a length-4 (x, px, y, py) vector, got {o.shape}")
    points = [o]
    for elem in lattice.elements:
        m4, k4 = _affine_4d(elem.matrix(lattice.ref), elem.kick(lattice.ref))
        o = m4 @ o + k4
        points.append(o)
    return points
