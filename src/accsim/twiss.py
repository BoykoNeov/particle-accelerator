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
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coords import DELTA, PX, PY, X, Y
from .lattice import Lattice

_TRANSVERSE = [X, PX, Y, PY]  # the 4D transverse subspace (x, px, y, py)


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

    ``D*`` are the linear dispersion ``(Dx, Dpx, Dy, Dpy) = d(x, px, y, py)/ddelta``
    [m, rad, m, rad] of the matched off-momentum closed orbit. They default to
    zero (a dispersion-free lattice — e.g. drifts + quads only).
    """

    s: float
    beta_x: float
    alpha_x: float
    mu_x: float
    beta_y: float
    alpha_y: float
    mu_y: float
    disp_x: float = 0.0
    disp_px: float = 0.0
    disp_y: float = 0.0
    disp_py: float = 0.0

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
    phase advance ``dmu = atan2(C12, beta*C11 - alpha*C12)``. For drifts and thin
    quadrupoles ``C12 >= 0`` gives ``dmu in [0, pi)``; the ``dmu < 0`` wrap below
    exists for the rarer thick focusing quad with ``omega*L > pi``, where ``C12``
    (which is ``sin(omega L)/omega``) goes negative. Note this recovers only per-
    element advances up to ``2*pi`` — a single element with ``dmu > 2*pi`` (a thick
    quad with ``omega*L > 2*pi``) would be undercounted; guard for that in Stage 3+.
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


def _transverse_4d(M: np.ndarray) -> np.ndarray:
    """The 4x4 transverse block ``(x, px, y, py)`` of a 6x6 map."""
    return M[np.ix_(_TRANSVERSE, _TRANSVERSE)]


def _dispersive_kick(M: np.ndarray) -> np.ndarray:
    """The transverse coupling to ``delta``: ``[R16, R26, R36, R46]``."""
    return M[_TRANSVERSE, DELTA]


def _matched_dispersion(one_turn: np.ndarray) -> np.ndarray:
    """Matched 4D dispersion ``D = (I - M4)^-1 d`` of a periodic map.

    The off-momentum closed orbit obeys ``D = M4 @ D + d`` (it must close on
    itself each turn), so ``D = (I - M4)^-1 d``. For an uncoupled lattice with no
    vertical bending the vertical components come out zero automatically.
    """
    m4 = _transverse_4d(one_turn)
    d = _dispersive_kick(one_turn)
    return np.linalg.solve(np.eye(4) - m4, d)


def match_periodic(one_turn: np.ndarray) -> Twiss:
    """Matched (periodic) Twiss at the start of a ring from its one-turn matrix.

    Raises :class:`UnstableLatticeError` if either plane is unstable. Phases are
    set to zero at this reference point.
    """
    cx, cy = _blocks(one_turn)
    beta_x, alpha_x = _matched_block(cx)
    beta_y, alpha_y = _matched_block(cy)
    d = _matched_dispersion(one_turn)
    return Twiss(0.0, beta_x, alpha_x, 0.0, beta_y, alpha_y, 0.0, d[0], d[1], d[2], d[3])


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
    disp = np.array([twiss0.disp_x, twiss0.disp_px, twiss0.disp_y, twiss0.disp_py])
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        cx, cy = _blocks(M)
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        # Dispersion is the first-order off-momentum orbit: propagate it affinely,
        # D -> M4 @ D + d (matrix transport plus the element's dispersive kick),
        # NOT the quadratic B = C B C^T form used for beta/alpha.
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)
        mux += dmux
        muy += dmuy
        s += elem.length
        points.append(Twiss(s, bx, ax, mux, by, ay, muy, disp[0], disp[1], disp[2], disp[3]))
    return points


def beam_sigma(
    twiss: Sequence[Twiss],
    emit_x: float,
    emit_y: float | None = None,
    sigma_delta: float = 0.0,
) -> tuple[list[float], list[float]]:
    r"""1-sigma transverse beam envelopes ``(sigma_x, sigma_y)`` along a Twiss table.

    The RMS beam size at each point adds the betatron width and the
    momentum-spread offset **in quadrature** (they are statistically independent
    in a matched beam):

        sigma_u(s) = sqrt(emit_u * beta_u(s) + (D_u(s) * sigma_delta)^2),  u in {x, y}.

    Inputs (not computed — there is no radiation/RF yet to set an equilibrium):

    - ``emit_x`` / ``emit_y``: geometric (not normalised) emittances [m·rad].
      ``emit_y`` defaults to ``emit_x`` (round beam).
    - ``sigma_delta``: RMS relative momentum spread ``sigma(delta)`` (dimensionless);
      default ``0`` gives the pure betatron envelope ``sqrt(emit_u * beta_u)``.

    Each plane uses its own dispersion, so vertical dispersion is included for free
    if a lattice ever produces it (a flat, uncoupled lattice has ``D_y = 0``, so
    the vertical envelope is betatron-only there). Units: ``D_u`` [m], ``sigma_delta``
    dimensionless, ``emit_u * beta_u`` [m] — consistent, ``sigma_u`` in [m].
    """
    if emit_y is None:
        emit_y = emit_x
    sd2 = sigma_delta * sigma_delta
    sx = [math.sqrt(emit_x * t.beta_x + (t.disp_x * t.disp_x) * sd2) for t in twiss]
    sy = [math.sqrt(emit_y * t.beta_y + (t.disp_y * t.disp_y) * sd2) for t in twiss]
    return sx, sy


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


_INV_4PI = 1.0 / (4.0 * math.pi)


def natural_chromaticity(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Natural chromaticity ``(Q'_x, Q'_y) = (dQ_x/ddelta, dQ_y/ddelta)``.

    An off-momentum particle sees a quadrupole gradient weakened as
    ``k1 -> k1/(1 + delta)``, which shifts the tune. To first order this is the
    textbook β-weighted integral of the gradient over the matched lattice:

        Q'_x = -(1 / 4 pi) ∮ beta_x(s) k1(s) ds,
        Q'_y = +(1 / 4 pi) ∮ beta_y(s) k1(s) ds,

    (opposite signs because the quad focuses x with ``+k1`` and y with ``-k1``);
    both come out **negative** for an ordinary FODO. Only quadrupole gradients
    contribute here — drifts add nothing, and dipole weak-focusing / edge
    chromaticity is out of Stage 2 scope (flagged: a lattice with bends will have
    an additional, uncomputed dipole term).

    Thin quads are exact single-point contributions (``beta`` is continuous across
    a thin kick); thick quads are integrated by ``slices``-fold trapezoidal
    sub-stepping of ``beta`` across the body. The integer part of the tune is
    irrelevant to ``dQ/ddelta``, so no phase unwrapping is needed.
    """
    from .elements.quadrupole import Quadrupole, ThinQuadrupole, _focusing_block

    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    xi_x = xi_y = 0.0
    for elem in lattice.elements:
        if isinstance(elem, ThinQuadrupole):
            # beta is continuous across a thin kick, so the entrance beta is the
            # value "at" the quad; k1l is the signed integrated gradient.
            xi_x += -_INV_4PI * bx * elem.k1l
            xi_y += +_INV_4PI * by * elem.k1l
            cx, cy = _blocks(elem.matrix(lattice.ref))
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
        elif isinstance(elem, Quadrupole) and elem.k1 != 0.0 and elem.length > 0.0:
            ds = elem.length / slices
            xb = _focusing_block(elem.k1, ds)  # x'' + k1 x = 0
            yb = _focusing_block(-elem.k1, ds)  # y'' - k1 y = 0
            int_bx = 0.5 * bx  # trapezoid: half-weight the entrance sample
            int_by = 0.5 * by
            for i in range(slices):
                bx, ax, _ = _propagate_block(xb, bx, ax)
                by, ay, _ = _propagate_block(yb, by, ay)
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                int_bx += w * bx
                int_by += w * by
            xi_x += -_INV_4PI * elem.k1 * int_bx * ds
            xi_y += +_INV_4PI * elem.k1 * int_by * ds
        else:
            cx, cy = _blocks(elem.matrix(lattice.ref))
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
    return xi_x, xi_y


def _sextupole_feeddown(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Sextupole feed-down chromaticity ``(dQ_x/ddelta, dQ_y/ddelta)`` at dispersion.

    A sextupole at a point of dispersion sees ``x = x_beta + D_x delta``; its
    quadratic kick ``Delta px = -1/2 k2l (x^2 - y^2)`` then contains a
    ``delta``-dependent *linear* gradient ``k1_eff = k2 D_x delta`` (and the mirror
    term in ``y``). Feeding that through the same tune-shift bookkeeping as the
    quadrupole natural chromaticity gives

        dQ_x/ddelta = +(1/4pi) ∮ beta_x(s) k2(s) D_x(s) ds,
        dQ_y/ddelta = -(1/4pi) ∮ beta_y(s) k2(s) D_x(s) ds

    (opposite signs to the ``x^2 - y^2`` structure; the ``+``/``-`` split is what
    lets a sextupole at ``D_x > 0`` push a negative natural chromaticity back
    toward zero). ``D_x`` is the matched dispersion transported to each sextupole,
    so this vanishes on a dispersion-free (drift + quad) lattice.

    Thin sextupoles are exact single-point contributions (``beta`` and ``D_x`` are
    continuous across the zero-length kick); thick sextupoles are integrated by
    trapezoidal sub-slicing of ``beta_x D_x`` / ``beta_y D_x`` across the body,
    whose linear map is a drift (so ``beta`` and ``D_x`` transport as through a
    drift).
    """
    from .elements.quadrupole import _focusing_block
    from .elements.sextupole import Sextupole, ThinSextupole

    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    xi_x = xi_y = 0.0
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        if isinstance(elem, ThinSextupole):
            dx = float(disp[0])
            xi_x += +_INV_4PI * bx * elem.k2l * dx
            xi_y += -_INV_4PI * by * elem.k2l * dx
        elif isinstance(elem, Sextupole) and elem.k2 != 0.0 and elem.length > 0.0:
            ds = elem.length / slices
            db = _focusing_block(0.0, ds)  # sextupole linear map is a drift: [[1,ds],[0,1]]
            int_x = 0.5 * bx * float(disp[0])  # trapezoid: half-weight entrance
            int_y = 0.5 * by * float(disp[0])
            for i in range(slices):
                bx, ax, _ = _propagate_block(db, bx, ax)
                by, ay, _ = _propagate_block(db, by, ay)
                # Drift transport of dispersion: D_x += D_px ds, D_y += D_py ds.
                disp[0] += disp[1] * ds
                disp[2] += disp[3] * ds
                w = 0.5 if i == slices - 1 else 1.0  # half-weight exit
                int_x += w * bx * float(disp[0])
                int_y += w * by * float(disp[0])
            xi_x += +_INV_4PI * elem.k2 * int_x * ds
            xi_y += -_INV_4PI * elem.k2 * int_y * ds
            continue  # beta / disp already advanced across the body
        # Advance beta and dispersion across this element (non-thick-sextupole).
        cx, cy = _blocks(M)
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)
    return xi_x, xi_y


def momentum_compaction(lattice: Lattice, slices: int = 64) -> float:
    r"""Momentum-compaction factor ``alpha_c`` of a periodic ``lattice``.

    A higher-momentum particle rides the dispersion orbit ``x = D_x delta`` and,
    where the orbit is curved, travels a longer (or shorter) path. The fractional
    circumference change per unit momentum deviation is the purely geometric
    integral over the ring:

        alpha_c = (1 / C) ∮ D_x(s) h(s) ds,   h(s) = 1/rho(s),  C = circumference.

    Only bending magnets contribute (``h = 0`` in drifts, quads, sextupoles), so a
    straight (dispersion-free) lattice has ``alpha_c = 0``. The matched dispersion
    is transported along the lattice; inside each thick dipole ``D_x(s)`` is
    integrated by ``slices``-fold trapezoidal sub-stepping of the sub-bend map
    (``h`` is constant across a sector body). ``alpha_c`` carries **no** ``gamma0``
    dependence — it is geometry only.

    Cross-checked two independent ways (see ``tests/analytic/test_momentum_compaction.py``):
    the symplecticity identity ``alpha_c = 1/gamma0^2 - (R51 D_x + R52 D_px + R56)/C``
    from the one-turn longitudinal row (a different set of matrix entries than the
    dispersion-generating ones this integral uses), and xtrack's
    ``momentum_compaction_factor``.
    """
    from .elements.dipole import Dipole

    tw0 = closed_twiss(lattice)
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    integral = 0.0  # ∮ D_x h ds
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        if isinstance(elem, Dipole) and elem.angle != 0.0 and elem.length > 0.0:
            h = elem.curvature
            ds = elem.length / slices
            sub = Dipole(ds, h * ds).matrix(lattice.ref)  # one sector sub-slice
            sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
            acc = 0.5 * disp[0]  # trapezoid: half-weight the entrance sample
            for i in range(slices):
                disp = sub4 @ disp + subk
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                acc += w * disp[0]
            integral += h * acc * ds
            continue
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)
    return integral / lattice.length


def slip_factor(lattice: Lattice, slices: int = 64) -> float:
    r"""Phase-slip factor ``eta = alpha_c - 1/gamma0^2`` of a periodic ``lattice``.

    Combines the geometric path-lengthening (:func:`momentum_compaction`) with the
    velocity effect: a higher-momentum particle moves faster (``+``) but on a
    longer orbit (``-``). ``eta`` sets the sign of the longitudinal restoring force
    and vanishes at transition (``gamma0 = 1/sqrt(alpha_c)``); Stage 3's
    synchrotron tune ``Qs`` is built on it. The ``1/gamma0^2`` is taken from the
    reference particle, the same single source as the drift/dipole ``R56 = L/gamma0^2``.
    """
    return momentum_compaction(lattice, slices) - 1.0 / lattice.ref.gamma0**2


def chromaticity(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Total first-order chromaticity ``(Q'_x, Q'_y)`` = quads + sextupole feed-down.

    Adds the sextupole feed-down term (:func:`_sextupole_feeddown`) to the
    quadrupole :func:`natural_chromaticity`. This is the quantity sextupoles exist
    to control: with the right ``k2`` at a dispersive location, the feed-down
    cancels the (negative) natural chromaticity.

    **Not a complete absolute total.** Like :func:`natural_chromaticity`, this omits
    the dipole's own weak-focusing / edge chromaticity (out of Stage 2 scope), so on
    any lattice where the feed-down is nonzero — which *requires* bends for
    dispersion — an uncomputed dipole term is also present. The validated
    deliverables are therefore: (a) the *feed-down term itself*, pinned to a
    symbolic ``dQ/ddelta``; (b) accsim-internal *correction* (feed-down cancels the
    quad natural term); and (c) an xtrack cross-check of the *with-minus-without
    sextupoles difference*, in which the shared dipole term cancels exactly (a
    sextupole's linear map is a drift, so adding it leaves beta, dispersion, and the
    tunes — hence every other chromaticity term — untouched).
    """
    nx, ny = natural_chromaticity(lattice, slices)
    fx, fy = _sextupole_feeddown(lattice, slices)
    return nx + fx, ny + fy
