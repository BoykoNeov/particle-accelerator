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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coords import PX, PY, X, Y
from .elements.corrector import Corrector
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
    singular and no orbit closes. **That check comes first, before the zero-kick
    shortcut**, so the contract does not depend on whether the machine happens to
    be perfect: on an integer tune a kick-free lattice does have zero as *a* fixed
    point, but not as the *only* one, and returning it would quietly claim a
    uniqueness the map does not have. It also keeps
    :func:`orbit_response_matrix`'s zeroed baseline on the same code path as its
    unit-kick columns, so the two cannot disagree about whether an orbit exists.
    """
    m4, k4 = _affine_4d(*lattice.one_turn_map())
    A = np.eye(4) - m4
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise ClosedOrbitError(
            f"no closed orbit: (I - M4) is singular to working precision "
            f"(condition number {cond:.3g}), i.e. the lattice is on an **integer** "
            "tune in at least one plane, where a kick repeats in phase every turn "
            "and the excursion never closes"
        )
    if not k4.any():
        return np.zeros(4)  # a perfect machine sits exactly on the design orbit
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


# ---------------------------------------------------------------------------
# Correction: response matrix and the SVD steering solve
# ---------------------------------------------------------------------------

_PLANES = {"x": (X, "kick_x"), "y": (Y, "kick_y")}


class OrbitCorrectionError(ValueError):
    """An orbit-correction problem is ill-posed (bad plane, monitor or corrector)."""


@dataclass(frozen=True)
class OrbitCorrection:
    """Outcome of :func:`correct_orbit`. The kicks are already applied.

    - ``kicks`` / ``initial_kicks`` — corrector angles [rad] after / before.
    - ``rms_before`` / ``rms_after`` — RMS orbit at the monitors [m]. **Both are
      measured**, by solving for the closed orbit of the actual lattice; neither
      is the linear prediction ``x0 + R dtheta``. That distinction is the whole
      point when the response matrix is one the machine handed you rather than
      one the model computed — see :func:`correct_orbit`.
    - ``singular_values`` — the full spectrum of the response matrix, largest
      first, so a caller can see the conditioning that was truncated.
    - ``n_used`` — how many singular values the solve kept.
    """

    kicks: np.ndarray
    initial_kicks: np.ndarray
    rms_before: float
    rms_after: float
    singular_values: np.ndarray
    n_used: int

    @property
    def n_correctors(self) -> int:
        return int(self.kicks.size)


def _check_plane(plane: str) -> tuple[int, str]:
    try:
        return _PLANES[plane]
    except KeyError:
        raise OrbitCorrectionError(
            f"plane must be 'x' or 'y', got {plane!r}; horizontal and vertical "
            "correction are separate problems in an uncoupled lattice"
        ) from None


def _check_correctors(lattice: Lattice, correctors: Sequence[Corrector]) -> list[Corrector]:
    corr = list(correctors)
    if not corr:
        raise OrbitCorrectionError("no correctors given; there is nothing to steer with")
    in_lattice = {id(e) for e in lattice.elements}
    seen: set[int] = set()
    for c in corr:
        if not isinstance(c, Corrector):
            raise OrbitCorrectionError(
                f"{type(c).__name__} cannot steer the orbit; correctors must be "
                "Corrector elements (a quadrupole moves the optics, not the orbit)"
            )
        if id(c) not in in_lattice:
            raise OrbitCorrectionError(
                f"{c!r} is not in this lattice, so it has no response; pass the "
                "element objects the lattice was built from"
            )
        if id(c) in seen:
            raise OrbitCorrectionError(
                f"{c!r} is listed twice, so its column would be ambiguous; list it "
                "once (repeated *placements* in the lattice are fine and are summed)"
            )
        seen.add(id(c))
    return corr


def _check_monitors(lattice: Lattice, monitors: Sequence[int] | None) -> list[int]:
    if monitors is None:
        return list(range(len(lattice) + 1))
    mon = [int(m) for m in monitors]
    if not mon:
        raise OrbitCorrectionError("no monitors given; there is nothing to measure")
    n = len(lattice)
    for m in mon:
        if not 0 <= m <= n:
            raise OrbitCorrectionError(
                f"monitor index {m} is outside [0, {n}]; monitors are element "
                "*boundaries*, the same points propagate_orbit returns"
            )
    return mon


def _orbit_at(lattice: Lattice, monitors: Sequence[int], coord: int) -> np.ndarray:
    """One plane's closed orbit sampled at the monitor boundaries."""
    table = propagate_orbit(lattice)
    return np.array([table[m][coord] for m in monitors])


def orbit_response_matrix(
    lattice: Lattice,
    correctors: Sequence[Corrector],
    monitors: Sequence[int] | None = None,
    plane: str = "x",
) -> np.ndarray:
    r"""Orbit response matrix ``R[i, j] = d(orbit at monitor i) / d(kick of corrector j)``.

    Shape ``(n_monitors, n_correctors)``, units m/rad. ``monitors`` are element
    **boundary** indices (``None`` = every boundary); ``plane`` selects ``'x'``
    (driving ``kick_x``) or ``'y'`` (driving ``kick_y``).

    **This response is exact, not a finite difference.** The closed orbit is
    strictly *affine* in the corrector kicks — the fixed point ``(I - M4)^-1 k4``
    is linear in ``k4``, and ``k4`` is linear in the kicks — so

        orbit(theta) = orbit(0) + R theta

    holds for any kick, however large, with no truncation error. Column ``j`` is
    therefore taken by setting corrector ``j`` to one radian, all other *listed*
    correctors to zero, and subtracting the baseline. That single consequence is
    what makes :func:`correct_orbit` one linear solve rather than an iteration,
    exactly as a sextupole's strict affineness makes
    :func:`~accsim.matching.match_chromaticity` an exact solve.

    Kicks not owned by a listed corrector (a steering *error* you are correcting
    against) stay put and cancel in the baseline subtraction, so the matrix is the
    machine's response and not a property of its current orbit — to round-off,
    since the cancellation is a floating-point ``(col + base) - base``, not an
    algebraic one.

    Like :func:`~accsim.matching.insertion_response_matrix` this **mutates the
    lattice while it works** and restores every corrector from a snapshot in a
    ``finally``, so an exception cannot leave the machine mis-set.
    """
    coord, attr = _check_plane(plane)
    corr = _check_correctors(lattice, correctors)
    mon = _check_monitors(lattice, monitors)

    snapshot = [getattr(c, attr) for c in corr]
    try:
        for c in corr:
            setattr(c, attr, 0.0)
        base = _orbit_at(lattice, mon, coord)
        columns = []
        for c in corr:
            setattr(c, attr, 1.0)
            columns.append(_orbit_at(lattice, mon, coord) - base)
            setattr(c, attr, 0.0)
    finally:
        for c, v in zip(corr, snapshot, strict=True):
            setattr(c, attr, v)
    return np.column_stack(columns)


def correct_orbit(
    lattice: Lattice,
    correctors: Sequence[Corrector],
    monitors: Sequence[int] | None = None,
    plane: str = "x",
    *,
    n_singular: int | None = None,
    response: np.ndarray | None = None,
) -> OrbitCorrection:
    r"""Steer the closed orbit to zero at the monitors, and apply the kicks.

    Solves ``R dtheta = -x0`` for the corrector changes, where ``x0`` is the
    present orbit at the monitors. Because ``R`` is exact (see
    :func:`orbit_response_matrix`) this is **one linear solve**, not an iteration:
    with enough independent correctors the orbit lands on zero to machine
    precision in a single application.

    ``n_singular`` keeps only the largest ``n`` singular values of ``R``, the
    standard defence against an ill-conditioned corrector set. Two correctors
    close together in betatron phase have nearly parallel response columns; the
    untruncated least-squares answer then buys a small orbit improvement with a
    huge, nearly cancelling pair of kicks that no real magnet could deliver.
    Truncation drops that direction and returns a sane, if slightly less perfect,
    correction. Default ``None`` keeps every singular value above the usual
    round-off cutoff.

    ``response`` lets the caller supply a response matrix instead of computing
    one — an operational machine *measures* its response matrix, and that
    measurement disagrees with the model. The consequence is deliberate and is
    why :attr:`OrbitCorrection.rms_after` is obtained by **re-solving the closed
    orbit of the corrected lattice**, never by evaluating ``x0 + R dtheta``: with
    a wrong ``R`` the prediction is perfect and the machine is not.

    N vs M is handled the way :func:`~accsim.matching.match_insertion` handles it:
    more correctors than monitors gives the minimum-norm solution, fewer gives the
    least-squares one. Neither is reported as exact — the returned RMS says what
    was actually achieved.

    Raises :class:`ClosedOrbitError` (and rolls the kicks back) if the corrected
    lattice has no closed orbit.
    """
    coord, attr = _check_plane(plane)
    corr = _check_correctors(lattice, correctors)
    mon = _check_monitors(lattice, monitors)

    R = (
        orbit_response_matrix(lattice, corr, mon, plane)
        if response is None
        else np.asarray(response, dtype=float)
    )
    if R.shape != (len(mon), len(corr)):
        raise OrbitCorrectionError(
            f"response has shape {R.shape}, expected ({len(mon)}, {len(corr)}) for "
            f"{len(mon)} monitors and {len(corr)} correctors"
        )
    if n_singular is not None and not 1 <= n_singular <= min(R.shape):
        raise OrbitCorrectionError(
            f"n_singular must be in [1, {min(R.shape)}] for a {R.shape} response "
            f"matrix, got {n_singular}"
        )

    x0 = _orbit_at(lattice, mon, coord)
    rms_before = float(np.sqrt(np.mean(x0**2)))

    u, s, vt = np.linalg.svd(R, full_matrices=False)
    keep = len(s) if n_singular is None else n_singular
    cutoff = s[0] * max(R.shape) * np.finfo(float).eps if s.size else 0.0
    keep = min(keep, int(np.count_nonzero(s > cutoff)))
    if keep == 0:
        raise OrbitCorrectionError(
            "the response matrix is numerically zero: none of these correctors "
            "moves the orbit at any of these monitors"
        )
    dtheta = -(vt[:keep].T @ ((u[:, :keep].T @ x0) / s[:keep]))

    initial = np.array([getattr(c, attr) for c in corr])
    kicks = initial + dtheta
    for c, v in zip(corr, kicks, strict=True):
        setattr(c, attr, float(v))
    try:
        # Measured, not predicted: the orbit of the machine as it now stands.
        x1 = _orbit_at(lattice, mon, coord)
    except ClosedOrbitError:
        for c, v in zip(corr, initial, strict=True):
            setattr(c, attr, float(v))
        raise
    return OrbitCorrection(
        kicks=kicks,
        initial_kicks=initial,
        rms_before=rms_before,
        rms_after=float(np.sqrt(np.mean(x1**2))),
        singular_values=s,
        n_used=keep,
    )
