"""Linear (Courant-Snyder) optics: matched Twiss, propagation, tunes.

The transverse motion of the uncoupled linear lattice is described per plane by
the Twiss parameters ``(beta, alpha, gamma)`` with ``gamma = (1 + alpha^2)/beta``,
and a phase advance ``mu``. This module extracts the matched (periodic) Twiss
from a one-turn matrix and propagates it element-by-element.

Scope: the Courant-Snyder path (:func:`match_periodic`, :func:`tunes`,
:func:`propagate_twiss`, dispersion, chromaticity) reduces the 6x6 map to
independent ``(x, px)`` and ``(y, py)`` 2x2 blocks. That reduction is exact only
for a **transversely uncoupled** lattice (drifts, quads, dipoles, sextupoles are
all block-diagonal). A betatron-coupling element — a
:class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` — breaks it, so the
uncoupled entry points are **guarded** (:func:`_require_uncoupled` raises
:class:`CoupledLatticeError` rather than return decoupled-but-wrong betas/tunes),
and coupled motion goes through the **normal-mode** route instead: the eigen-tunes
(:func:`normal_mode_tunes`), the difference-resonance :func:`closest_tune_approach`,
and the Edwards-Teng normal-mode beta functions (:func:`coupled_twiss`,
:func:`propagate_coupled_twiss`) with the projected sizes a screen would see
(:func:`coupled_beam_sigma`). Dispersion (the coupling to ``delta``) is included in
the block path via the :class:`~accsim.elements.dipole.Dipole`.

Conventions (see ``docs/CONVENTIONS.md``):

- The matched beta is positive by construction; the sign of ``sin mu`` is fixed
  by ``beta = M12 / sin mu > 0``.
- Phase is **accumulated continuously** along the lattice (``atan2`` per element),
  never via ``acos`` of the one-turn matrix — the latter only yields the
  *fractional* tune and loses the integer part. ``Q = mu_total / 2 pi``.
- Stability of a plane requires ``|1/2 Tr(block)| < 1`` (``|Tr| < 2``); the coupled
  analogue is all four eigenvalues of the 4x4 on the unit circle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .coords import DELTA, DIM, PX, PY, ZETA, X, Y
from .lattice import Lattice

_TRANSVERSE = [X, PX, Y, PY]  # the 4D transverse subspace (x, px, y, py)

#: Below this, relative to the eigenvector's own size, a mode's symplectic norm counts
#: as zero and the map has no normal form (see :class:`NormalFormError`).
_DEGENERATE_NORM = 1e-10


class UnstableLatticeError(ValueError):
    """Raised when a plane's one-turn block is unstable (``|1/2 Tr| >= 1``).

    An unstable plane has no real matched (periodic) beta function — the betatron
    motion grows without bound — so Twiss matching is undefined.
    """


class CoupledLatticeError(ValueError):
    """Raised when uncoupled 2x2 Twiss is asked of an x-y *coupled* lattice.

    The Courant-Snyder machinery here (:func:`match_periodic`, :func:`tunes`,
    :func:`natural_chromaticity`, ...) reduces the one-turn map to independent
    ``(x, px)`` and ``(y, py)`` 2x2 blocks. That reduction is exact only when the
    off-diagonal blocks vanish — i.e. no betatron coupling. A
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` (or any element with
    ``R[px, y]`` / ``R[py, x]`` terms) breaks it, and a naive 2x2 match would return
    plausible-but-wrong betas and tunes. Use the coupled machinery for such a lattice
    instead of silently decoupling it: :func:`coupled_twiss` for the normal-mode
    (Edwards-Teng) beta functions and :func:`normal_mode_tunes` for the tunes.
    """


class ResonantLatticeError(ValueError):
    """Raised when a closed form asked for here sits on top of its own resonance.

    :func:`sextupole_detuning` divides by ``sin(pi Phi)`` for ``Phi`` in ``{Q_x,
    3 Q_x, Q_x + 2 Q_y, Q_x - 2 Q_y}`` — the lines a sextupole drives. On such a line
    the second-order normal form does not exist: the perturbation series has a zero
    denominator, and the tune of a particle at finite amplitude is not an analytic
    function of its action at all. The divergence *approaching* the line is physical
    and is returned; landing on it is not a number this package will invent.
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


def _coupling_norm(M: np.ndarray) -> float:
    """Largest ``|entry|`` of the two off-diagonal (x-y coupling) 2x2 blocks of a 6x6."""
    m = np.abs(M[np.ix_([X, PX], [Y, PY])]).max()
    return float(max(m, np.abs(M[np.ix_([Y, PY], [X, PX])]).max()))


def _require_uncoupled(one_turn: np.ndarray, atol: float = 1e-9) -> None:
    """Raise :class:`CoupledLatticeError` if the one-turn map has x-y coupling.

    The uncoupled 2x2 Courant-Snyder path is only valid when the transverse map is
    block-diagonal. Drifts, quads, dipoles and sextupoles are exactly block-diagonal
    (coupling norm 0), so this is a no-op for them; a skew quad trips it.
    """
    c = _coupling_norm(one_turn)
    if c > atol:
        raise CoupledLatticeError(
            f"lattice is x-y coupled (off-block norm {c:.3g} > {atol:g}); the 2x2 "
            "Courant-Snyder path would return wrong betas/tunes. Use "
            "normal_mode_tunes() for the coupled mode tunes."
        )


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

    Raises :class:`UnstableLatticeError` if either plane is unstable, or
    :class:`CoupledLatticeError` if the map has betatron (x-y) coupling — the 2x2
    reduction is only valid for a block-diagonal transverse map. Phases are set to
    zero at this reference point.
    """
    _require_uncoupled(one_turn)
    cx, cy = _blocks(one_turn)
    beta_x, alpha_x = _matched_block(cx)
    beta_y, alpha_y = _matched_block(cy)
    d = _matched_dispersion(one_turn)
    return Twiss(0.0, beta_x, alpha_x, 0.0, beta_y, alpha_y, 0.0, d[0], d[1], d[2], d[3])


def closed_twiss(lattice: Lattice) -> Twiss:
    """Matched Twiss at the entrance of a periodic ``lattice``."""
    return match_periodic(lattice.one_turn_matrix())


def propagate_twiss(
    lattice: Lattice, twiss0: Twiss, *, maps: Sequence[np.ndarray] | None = None
) -> list[Twiss]:
    """Twiss at every element boundary, starting from ``twiss0``.

    Returns ``len(lattice) + 1`` points: the entrance, then the exit of each
    element in order. Phase advances accumulate continuously, so the last point's
    ``mu`` over one period equals ``2 pi Q``.

    ``maps`` substitutes the transport and nothing else: one ``6x6`` per element,
    in beam order, used in place of ``elem.matrix()``. The lattice is still needed
    — ``s`` comes from the element lengths — and passing the on-axis matrices
    explicitly reproduces the default bit for bit. Its purpose is
    :func:`~accsim.orbit.linearised_element_maps`: the maps a particle near the
    *real* closed orbit sees, which differ from the design matrices wherever an
    off-axis sextupole feeds a gradient down. :func:`propagate_twiss_on_orbit` is
    that combination packaged.
    """
    if maps is not None and len(maps) != len(lattice.elements):
        raise ValueError(
            f"maps must have one matrix per element: got {len(maps)} for "
            f"{len(lattice.elements)} elements"
        )
    points = [twiss0]
    s = twiss0.s
    bx, ax, mux = twiss0.beta_x, twiss0.alpha_x, twiss0.mu_x
    by, ay, muy = twiss0.beta_y, twiss0.alpha_y, twiss0.mu_y
    disp = np.array([twiss0.disp_x, twiss0.disp_px, twiss0.disp_y, twiss0.disp_py])
    for i, elem in enumerate(lattice.elements):
        M = elem.matrix(lattice.ref) if maps is None else maps[i]
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
    """True if both transverse planes are stable (``|1/2 Tr(block)| < 1``).

    This is the *uncoupled* per-plane test; it inspects only the diagonal 2x2 blocks
    and is blind to x-y coupling (a coupled lattice can be unstable through the
    off-blocks while each diagonal block looks stable). For a coupled lattice use the
    eigenvalue stability implicit in :func:`normal_mode_tunes` (which raises if any
    eigenvalue leaves the unit circle).
    """
    cx, cy = _blocks(one_turn)
    return abs(0.5 * (cx[0, 0] + cx[1, 1])) < 1.0 and abs(0.5 * (cy[0, 0] + cy[1, 1])) < 1.0


def _j4() -> np.ndarray:
    """The 4x4 unit-symplectic form for ``(x, px, y, py)`` (block-diag [[0,1],[-1,0]])."""
    J = np.zeros((4, 4))
    J[0, 1] = J[2, 3] = 1.0
    J[1, 0] = J[3, 2] = -1.0
    return J


_J4 = _j4()


def normal_mode_tunes(lattice: Lattice, atol: float = 1e-6) -> tuple[float, float]:
    r"""Coupled **normal-mode** fractional tunes ``(Q1, Q2)`` of a periodic lattice.

    When the lattice couples ``x`` and ``y`` (e.g. a
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole`), the motion no longer
    separates into independent horizontal and vertical betatron oscillations. The
    right invariant description is the pair of **normal modes** — the eigenvectors of
    the transverse 4x4 one-turn matrix ``M4``. A stable symplectic ``M4`` has four
    eigenvalues on the unit circle in two complex-conjugate pairs
    ``e^{±i 2 pi Q1}, e^{±i 2 pi Q2}``; the mode tunes are the phases ``/2 pi``.

    Returns the two **fractional** tunes in ``[0, 1)`` (eigenvalues give only the
    fractional part — the integer turn count is lost). They are ordered by which
    plane dominates each mode's eigenvector, so for a weakly-coupled lattice
    ``Q1`` is the ``x``-like mode and ``Q2`` the ``y``-like one, and in the
    uncoupled limit ``(Q1, Q2)`` equals ``tunes(lattice) mod 1`` exactly. Exactly at
    the difference resonance the modes are 50/50 mixtures and the labelling is
    arbitrary, but the *pair* (and their gap, the observable) is well defined.

    The rotation sense of each mode is fixed by the sign of its eigenvector's
    symplectic norm ``Im(v* J v)`` — the standard convention that maps each
    conjugate pair to a single tune in ``[0, 1)`` (rather than the ambiguous
    ``acos`` fractional value in ``[0, 0.5]``). Raises
    :class:`UnstableLatticeError` if any eigenvalue leaves the unit circle by more
    than ``atol`` (a coupled instability — e.g. a sum-resonance stop-band — which
    the per-plane :func:`is_stable` cannot see).
    """
    m4 = _transverse_4d(lattice.one_turn_matrix())
    eigvals, eigvecs = np.linalg.eig(m4)
    if not np.allclose(np.abs(eigvals), 1.0, atol=atol, rtol=0.0):
        raise UnstableLatticeError(
            f"coupled lattice unstable: eigenvalue moduli {np.abs(eigvals)} are not "
            "all on the unit circle (betatron motion grows without bound)."
        )
    modes: list[tuple[float, float, float]] = []  # (tune, x-weight, y-weight)
    for k in range(4):
        v = eigvecs[:, k]
        snorm = float(np.imag(np.conj(v) @ _J4 @ v))
        if snorm <= 0.0:  # keep one eigenvector per conjugate pair (the +orientation)
            continue
        q = float(np.angle(eigvals[k]) / (2.0 * math.pi)) % 1.0
        xw = float(abs(v[0]) ** 2 + abs(v[1]) ** 2)
        yw = float(abs(v[2]) ** 2 + abs(v[3]) ** 2)
        modes.append((q, xw, yw))
    if len(modes) != 2:  # pragma: no cover - degenerate numerical edge
        raise UnstableLatticeError(
            f"could not resolve two normal modes (found {len(modes)}); the one-turn "
            "matrix may be at an exact resonance degeneracy."
        )
    # Label by dominant plane: the more x-like mode is Q1, the more y-like is Q2.
    (qa, xa, _ya), (qb, xb, _yb) = modes
    return (qa, qb) if xa >= xb else (qb, qa)


@dataclass(frozen=True)
class CoupledTwiss:
    """Edwards-Teng **normal-mode** optics at one longitudinal position ``s``.

    The coupled analogue of :class:`Twiss`. Modes ``1`` and ``2`` are the two
    betatron eigen-modes; ``beta_1``/``alpha_1`` are the Courant-Snyder parameters
    *of the mode*, not of a plane. In the uncoupled limit mode 1 is exactly the
    horizontal plane and mode 2 the vertical (``gamma_c = 1``, ``C = 0``).

    - ``gamma_c`` is the Edwards-Teng mixing parameter, ``cos`` of the coupling
      angle: ``gamma_c = 1`` uncoupled, ``gamma_c = 1/sqrt(2)`` fully mixed. It
      satisfies ``gamma_c**2 + det C = 1`` exactly (that is what makes the
      transformation symplectic), so ``gamma_c`` is always in ``[1/sqrt(2), 1]``.
    - ``c11..c22`` are the entries of the 2x2 coupling matrix ``C`` (see
      :attr:`c_matrix`); they carry the *orientation* of the mixing, which
      ``gamma_c`` alone does not.
    - ``D*`` are the matched 4D dispersion components as in :class:`Twiss`. They are
      solved from the **full coupled** 4x4, so a skew quadrupole at nonzero ``D_x``
      correctly produces vertical dispersion here.
    """

    s: float
    beta_1: float
    alpha_1: float
    beta_2: float
    alpha_2: float
    gamma_c: float
    c11: float = 0.0
    c12: float = 0.0
    c21: float = 0.0
    c22: float = 0.0
    disp_x: float = 0.0
    disp_px: float = 0.0
    disp_y: float = 0.0
    disp_py: float = 0.0

    @property
    def gamma_1(self) -> float:
        return (1.0 + self.alpha_1**2) / self.beta_1

    @property
    def gamma_2(self) -> float:
        return (1.0 + self.alpha_2**2) / self.beta_2

    @property
    def c_matrix(self) -> np.ndarray:
        """The 2x2 Edwards-Teng coupling matrix ``C``."""
        return np.array([[self.c11, self.c12], [self.c21, self.c22]])

    @property
    def coupling_angle(self) -> float:
        """The mixing angle ``phi = arccos(gamma_c)`` in radians, in ``[0, pi/4]``.

        ``0`` is uncoupled; ``pi/4`` (45 degrees) is full mixing, reached only on the
        difference resonance. ``sin(phi)**2 = det C`` is the fraction of mode 2's
        action carried in the horizontal plane (and vice versa).
        """
        return math.acos(min(1.0, max(-1.0, self.gamma_c)))

    @property
    def v_matrix(self) -> np.ndarray:
        """The 4x4 symplectic decoupling transformation ``V`` (block form below).

        ``V = [[gamma_c I, C], [-adj(C), gamma_c I]]`` maps normal-mode coordinates to
        laboratory ``(x, px, y, py)``: the one-turn map is ``M4 = V U V^-1`` with
        ``U = diag(A, B)`` block-diagonal.
        """
        C = self.c_matrix
        g = self.gamma_c
        return np.block([[g * np.eye(2), C], [-_adj2(C), g * np.eye(2)]])


def _adj2(C: np.ndarray) -> np.ndarray:
    """Symplectic conjugate (= adjugate) of a 2x2 matrix: ``C+ = -J C^T J``.

    For 2x2 this is the adjugate ``[[d, -b], [-c, a]]``, so ``C C+ = det(C) I``.
    """
    return np.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]])


def _edwards_teng(one_turn: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    r"""Edwards-Teng decomposition of a transverse one-turn map: ``(gamma_c, C, A, B)``.

    Factorises the transverse 4x4 as ``M4 = V U V^-1`` with

        V = [[gamma_c I, C], [-C+, gamma_c I]],   U = [[A, 0], [0, B]],

    where ``C+ = adj(C)`` and ``V`` is symplectic iff ``gamma_c^2 + det C = 1``. ``A``
    and ``B`` are then ordinary 2x2 Courant-Snyder blocks — one per normal mode.

    **Derivation (not recalled).** Writing ``M4`` in 2x2 blocks ``[[m, n], [p, q]]``
    and ``X = C / gamma_c``, the vanishing of the off-diagonal block of ``V^-1 M4 V``
    is the matrix Riccati equation ``n + m X - X q - X p X = 0``. Its root is
    proportional to ``H = n + adj(p)``, ``X = lambda H``, and with
    ``Delta = (tr m - tr q)/2``, ``R = sqrt(Delta^2 + det H)``:

        lambda  = -sgn(Delta) / (|Delta| + R),
        gamma_c^2 = 1 / (1 + det X) = 1/2 + |Delta| / (2 R),
        C       = gamma_c X = -sgn(Delta) H / (2 gamma_c R).

    The ``1/(|Delta| + R)`` normalisation (equivalently the ``2`` in ``2 gamma_c R``)
    is what makes ``gamma_c^2 + det C = 1``; it is verified in-test both by
    re-deriving ``lambda`` from the Riccati and by the residual ``||M4 - V U V^-1||``.

    Taking ``|Delta|`` (rather than ``Delta``) fixes ``gamma_c >= 1/sqrt(2)``, i.e. ``V``
    is the *smaller* of the two rotations, which labels mode 1 as the ``x``-like one —
    the same convention as :func:`normal_mode_tunes`. Exactly on the difference
    resonance ``Delta = 0`` the modes are 50/50 and the labelling is arbitrary (here
    ``sgn(0) = +1``); the *pair* is still well defined.

    **How an unstable coupled lattice actually fails here.** In practice the raise comes
    from :func:`_matched_block` on a normal-mode block (``|1/2 Tr(A)| >= 1``) — that is
    what fires on the known coupled-instability ring from the G1 tests, where the
    discriminant stays positive. The ``Delta^2 + det H < 0`` branch below is a
    **defensive** guard for the case where the two modes merge outright: a scan over
    FODO rings (both symmetric and split-tune, thin skews from ``0.05`` to ``2``,
    including near the sum resonance) found no lattice reachable with the current
    element set that triggers it, so it is documented as unexercised rather than
    claimed as the instability path. Either way the function raises
    :class:`UnstableLatticeError` and never returns a fabricated decomposition.
    """
    m4 = _transverse_4d(one_turn)
    m, n, p, q = m4[:2, :2], m4[:2, 2:], m4[2:, :2], m4[2:, 2:]
    H = n + _adj2(p)
    delta = 0.5 * (float(np.trace(m)) - float(np.trace(q)))
    disc = delta * delta + float(np.linalg.det(H))
    if disc < 0.0:  # pragma: no cover - defensive; see the docstring (no known trigger)
        raise UnstableLatticeError(
            f"no real Edwards-Teng decoupling: Delta^2 + det H = {disc:.3g} < 0 — the "
            "normal modes have merged (a coupled instability the per-plane |1/2 Tr| "
            "test cannot see)."
        )
    root = math.sqrt(disc)
    if root == 0.0:  # Delta = 0 and det H = 0: only reachable uncoupled (H = 0)
        if np.abs(H).max() > 0.0:  # pragma: no cover - degenerate singular-H edge
            raise UnstableLatticeError(
                "degenerate Edwards-Teng decomposition (Delta = 0 and det H = 0 with "
                "H != 0): the normal modes are not separable."
            )
        gamma_c, C = 1.0, np.zeros((2, 2))
    else:
        sgn = 1.0 if delta >= 0.0 else -1.0
        gamma_c = math.sqrt(0.5 + abs(delta) / (2.0 * root))
        C = -sgn * H / (2.0 * gamma_c * root)
    V = np.block([[gamma_c * np.eye(2), C], [-_adj2(C), gamma_c * np.eye(2)]])
    V_inv = np.block([[gamma_c * np.eye(2), -C], [_adj2(C), gamma_c * np.eye(2)]])
    U = V_inv @ m4 @ V
    return gamma_c, C, U[:2, :2], U[2:, 2:]


def match_periodic_coupled(one_turn: np.ndarray, s: float = 0.0) -> CoupledTwiss:
    """Matched Edwards-Teng normal-mode optics from a one-turn matrix.

    The coupled counterpart of :func:`match_periodic`: it does **not** require the
    map to be block-diagonal, and reduces to it exactly when it is (``gamma_c = 1``,
    ``C = 0``, ``beta_1 = beta_x``, ``beta_2 = beta_y`` to the last bit).

    Raises :class:`UnstableLatticeError` if either normal mode is unstable
    (``|1/2 Tr(A)| >= 1``) or the modes cannot be separated (see
    :func:`_edwards_teng`).
    """
    gamma_c, C, A, B = _edwards_teng(one_turn)
    beta_1, alpha_1 = _matched_block(A)
    beta_2, alpha_2 = _matched_block(B)
    d = _matched_dispersion(one_turn)
    return CoupledTwiss(
        s,
        beta_1,
        alpha_1,
        beta_2,
        alpha_2,
        gamma_c,
        float(C[0, 0]),
        float(C[0, 1]),
        float(C[1, 0]),
        float(C[1, 1]),
        float(d[0]),
        float(d[1]),
        float(d[2]),
        float(d[3]),
    )


def coupled_twiss(lattice: Lattice) -> CoupledTwiss:
    """Matched normal-mode (Edwards-Teng) optics at the entrance of a periodic lattice.

    Use this instead of :func:`closed_twiss` when the lattice contains a
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` — the uncoupled path
    raises :class:`CoupledLatticeError` there by design.
    """
    return match_periodic_coupled(lattice.one_turn_matrix())


def propagate_coupled_twiss(
    lattice: Lattice, *, maps: Sequence[np.ndarray] | None = None
) -> list[CoupledTwiss]:
    """Matched normal-mode optics at every element boundary of a periodic lattice.

    Returns ``len(lattice) + 1`` points (entrance, then each element's exit). Unlike
    :func:`propagate_twiss`, which transports ``(beta, alpha)`` forward, this
    re-matches at every point from the **local** one-turn map
    ``M(s) = T(s) M(0) T(s)^-1`` (``T(s)`` the transfer matrix from the start). That
    is exact, needs no transport rule for the coupling matrix ``C``, and keeps the
    mode labelling consistent with :func:`coupled_twiss`.

    **Scope:** no mode *phase* is accumulated — the returned points carry no ``mu``,
    so this cannot produce a tune (use :func:`normal_mode_tunes`). Mode labelling is
    per-point: on a lattice sitting exactly on the difference resonance the local
    ``Delta`` can pass through zero and modes 1/2 may swap between points; off
    resonance (the useful regime) it is stable, and the ``beta_1`` continuity is
    gated in the analytic tests.

    ``maps`` substitutes the transport exactly as in :func:`propagate_twiss` — the
    running transfer matrix and the one-turn map are both built from it. This is
    the path a *vertically* steered machine needs: a normal sextupole at
    ``y_co != 0`` is a skew quadrupole, so its on-orbit optics is coupled even
    though its design optics is exactly not.
    """
    if maps is not None and len(maps) != len(lattice.elements):
        raise ValueError(
            f"maps must have one matrix per element: got {len(maps)} for "
            f"{len(lattice.elements)} elements"
        )
    if maps is None:
        one_turn = lattice.one_turn_matrix()
    else:
        one_turn = np.eye(6)
        for m in maps:
            one_turn = m @ one_turn
    points = [match_periodic_coupled(one_turn, s=0.0)]
    transfer = np.eye(6)
    s = 0.0
    for i, elem in enumerate(lattice.elements):
        transfer = (elem.matrix(lattice.ref) if maps is None else maps[i]) @ transfer
        s += elem.length
        local = transfer @ one_turn @ np.linalg.inv(transfer)
        points.append(match_periodic_coupled(local, s=s))
    return points


def coupled_beam_sigma(
    twiss: Sequence[CoupledTwiss],
    emit_1: float,
    emit_2: float,
    sigma_delta: float = 0.0,
) -> tuple[list[float], list[float], list[float]]:
    r"""Projected beam sizes ``(sigma_x, sigma_y, tilt)`` of a **coupled** beam.

    The coupled counterpart of :func:`beam_sigma`. With betatron coupling the beam
    ellipse is no longer aligned with the ``x``/``y`` axes: what a screen measures is
    the *projection* of the 4D ellipsoid, which mixes both modes. Building the sigma
    matrix in the normal-mode basis and rotating it back with ``V``,

        Sigma = V diag(emit_1 B_1, emit_2 B_2) V^T,   B_i = [[beta_i, -alpha_i],
                                                            [-alpha_i, gamma_i]],

    and adding the dispersive contribution in quadrature (statistically independent
    of the betatron motion),

        sigma_x = sqrt(Sigma_xx + (D_x sigma_delta)^2),
        sigma_y = sqrt(Sigma_yy + (D_y sigma_delta)^2),
        tilt    = 1/2 atan2(2 <x y>, <x^2> - <y^2>),  <x y> = Sigma_xy + D_x D_y sigma_delta^2.

    ``emit_1``/``emit_2`` are the **eigen-mode** geometric emittances — exactly what
    :func:`~accsim.radiation.equilibrium_emittances_coupled` returns. Note the
    distinction that matters: ``sigma_y`` here is the *projected* vertical size, which
    on a coupled machine is larger than ``sqrt(emit_2 beta_2)`` because mode 1 leaks
    into the vertical plane. ``tilt`` is in radians, the angle of the beam ellipse's
    major axis in the ``x-y`` plane (zero for an uncoupled beam).
    """
    sx: list[float] = []
    sy: list[float] = []
    tilt: list[float] = []
    sd2 = sigma_delta * sigma_delta
    for t in twiss:
        b1 = np.array([[t.beta_1, -t.alpha_1], [-t.alpha_1, t.gamma_1]])
        b2 = np.array([[t.beta_2, -t.alpha_2], [-t.alpha_2, t.gamma_2]])
        mode = np.block([[emit_1 * b1, np.zeros((2, 2))], [np.zeros((2, 2)), emit_2 * b2]])
        V = t.v_matrix
        sigma = V @ mode @ V.T
        xx = sigma[0, 0] + t.disp_x * t.disp_x * sd2
        yy = sigma[2, 2] + t.disp_y * t.disp_y * sd2
        xy = sigma[0, 2] + t.disp_x * t.disp_y * sd2
        sx.append(math.sqrt(xx))
        sy.append(math.sqrt(yy))
        tilt.append(0.5 * math.atan2(2.0 * xy, xx - yy))
    return sx, sy, tilt


def _decoupled(M: np.ndarray) -> np.ndarray:
    """Copy of a 6x6 map with the transverse x-y coupling blocks zeroed.

    Leaves each plane's own 2x2 focusing (the diagonal blocks) and all dispersion /
    longitudinal terms intact, so the *unperturbed* (coupling-free) optics can be
    matched and propagated even on a lattice that contains skew elements. For a thin
    skew quad the diagonal blocks are the identity, so removing the coupling is exact;
    for a thick skew quad the retained diagonal block ``(F+D)/2`` is the plane's true
    coupling-free focusing.
    """
    M = M.copy()
    M[np.ix_([X, PX], [Y, PY])] = 0.0
    M[np.ix_([Y, PY], [X, PX])] = 0.0
    return M


def closest_tune_approach(lattice: Lattice) -> float:
    r"""Closest-tune-approach ``DeltaQ_min = |C^-|`` from the ring's skew quadrupoles.

    The difference-resonance coupling coefficient. When a ring is tuned toward the
    linear difference resonance ``Q_x = Q_y``, the two normal-mode tunes **cannot
    cross** — they repel, and the minimum gap they can reach is the modulus of the
    difference coupling coefficient

        C^- = (1 / 2 pi) sum_j (k1s l)_j sqrt(beta_x beta_y)_j
                              exp(i (mu_x - mu_y))_j ,

    summed over the skew-quadrupole sources ``j`` (a thick
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole` is trapezoid-sliced),
    with ``beta`` and the betatron phases taken from the **unperturbed** (coupling-off)
    optics at each source. The ``1/2 pi`` prefactor and the geometric mean
    ``sqrt(beta_x beta_y)`` are *derived* (not recalled) from the exact eigen-tune
    split of a single-skew-kick model — see ``docs/CONVENTIONS.md`` → *Betatron
    coupling*.

    ``|C^-|`` is the observable minimum split: on the resonance the normal-mode gap
    equals it exactly; a tune distance ``Delta = Q_x - Q_y`` away, the gap opens as
    the hyperbola ``sqrt(Delta^2 + |C^-|^2)``. It is validated two independent ways
    that cannot share an error — this closed form versus the exact
    :func:`normal_mode_tunes` eigenvalue gap — converging with an ``O((k1s l)^2)``
    residual as the coupling is taken to zero. Returns ``0.0`` for a lattice with no
    skew quadrupoles.

    **The sum walks element *types*, so a coupling source it does not know about is
    refused rather than silently summed as zero** (K2). A **rolled** element couples
    the planes without being a skew quadrupole — a rolled bend does it through the
    residual frame roll ``phi (1 - cos angle)``, a rolled normal quadrupole does it
    outright — and for such a lattice this function would otherwise return a
    reassuring ``0.0`` for a ring that is demonstrably coupled. The test is
    **measured**, not by type: an element whose own matrix has a nonzero transverse
    off-block and is not a skew quadrupole is one this sum cannot see.
    :func:`normal_mode_tunes` is the eigenvalue path, and it sees everything.
    """
    from .elements.skew_quadrupole import SkewQuadrupole, ThinSkewQuadrupole

    ref = lattice.ref
    for elem in lattice.elements:
        if isinstance(elem, (SkewQuadrupole, ThinSkewQuadrupole)):
            continue
        M = elem.matrix(ref)
        if M[np.ix_([X, PX], [Y, PY])].any() or M[np.ix_([Y, PY], [X, PX])].any():
            raise CoupledLatticeError(
                f"{type(elem).__name__} {elem.name!r} couples x and y without being a "
                f"skew quadrupole (roll={elem.roll}), and this sum walks element types: "
                "it would report DeltaQ_min = 0 for a ring that is coupled. Use "
                "normal_mode_tunes(), which diagonalises the one-turn map and sees "
                "every source"
            )
    # Unperturbed (coupling-off) one-turn and matched Twiss.
    decoupled = [_decoupled(elem.matrix(ref)) for elem in lattice.elements]
    one_turn = np.eye(6)
    for M in decoupled:
        one_turn = M @ one_turn
    tw0 = match_periodic(one_turn)  # coupling zeroed, so the guard passes
    bx, ax, mux = tw0.beta_x, tw0.alpha_x, 0.0
    by, ay, muy = tw0.beta_y, tw0.alpha_y, 0.0

    c_minus = 0.0 + 0.0j
    for elem, M in zip(lattice.elements, decoupled, strict=True):
        if isinstance(elem, ThinSkewQuadrupole):
            # beta/phase continuous across a thin kick — evaluate "at" the element.
            c_minus += elem.k1sl * math.sqrt(bx * by) * np.exp(1j * (mux - muy)) / (2.0 * math.pi)
        elif isinstance(elem, SkewQuadrupole) and elem.k1s != 0.0 and elem.length > 0.0:
            slices = 64
            ds = elem.length / slices
            sub = _decoupled(SkewQuadrupole(ds, elem.k1s).matrix(ref))
            cx, cy = _blocks(sub)
            # trapezoid the phasor k1s*sqrt(bx by)*e^{i(mux-muy)} across the body
            acc = 0.5 * math.sqrt(bx * by) * np.exp(1j * (mux - muy))
            for i in range(slices):
                bx, ax, dmux = _propagate_block(cx, bx, ax)
                by, ay, dmuy = _propagate_block(cy, by, ay)
                mux += dmux
                muy += dmuy
                w = 0.5 if i == slices - 1 else 1.0
                acc += w * math.sqrt(bx * by) * np.exp(1j * (mux - muy))
            c_minus += elem.k1s * acc * ds / (2.0 * math.pi)
            continue  # optics already advanced across the body
        # advance unperturbed beta/phase across this element
        cx, cy = _blocks(M)
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        mux += dmux
        muy += dmuy
    return float(abs(c_minus))


_INV_4PI = 1.0 / (4.0 * math.pi)


def _dipole_chroma_integrand(
    bx: float, ax: float, by: float, ay: float, dx: float, dpx: float, h: float, k1: float
) -> tuple[float, float]:
    r"""Per-length natural-chromaticity integrand inside a dipole body.

    ``(integrand_x, integrand_y)`` such that ``dQ'_u = (1/4pi) integrand_u ds``:

        integrand_x = -beta_x (k1 + h^2) + h (gamma_x D_x - 2 alpha_x D_px)
                      + 2 h k1 beta_x D_x,
        integrand_y = +beta_y k1        + gamma_y h D_x
                      -   h k1 beta_y D_x,

    with ``gamma_u = (1 + alpha_u^2)/beta_u``. Derived from the exact curvilinear
    Hamiltonian linearised about the dispersed closed orbit (``docs/CONVENTIONS.md``
    → *Dipole chromaticity*). The last term in each plane is the **curvature-sextupole
    feed-down**: a combined-function sector magnet must carry a Maxwell-forced
    3rd-order field term ``∝ h k1`` (without it ``div B != 0`` in the curved frame),
    which acts as a sextupole and feeds down to chromaticity at dispersion. The
    horizontal coefficient ``2 h k1`` and the vertical ``-h k1`` (note: NOT the
    symmetric ratio of an ordinary sextupole) are what make the combined-function
    result match xtrack and MAD-X; the vertical coefficient is an independent
    confirmation (Maxwell + the horizontal fix the field, the vertical then follows).
    """
    gamma_x = (1.0 + ax * ax) / bx
    gamma_y = (1.0 + ay * ay) / by
    integrand_x = -bx * (k1 + h * h) + h * (gamma_x * dx - 2.0 * ax * dpx) + 2.0 * h * k1 * bx * dx
    integrand_y = by * k1 + gamma_y * h * dx - h * k1 * by * dx
    return integrand_x, integrand_y


def natural_chromaticity(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Natural chromaticity ``(Q'_x, Q'_y) = (dQ_x/ddelta, dQ_y/ddelta)``.

    The lattice's inherent (sextupole-free) tune dependence on momentum, from the
    off-momentum weakening of every focusing element. Derived from the exact
    curvilinear Hamiltonian (see ``docs/CONVENTIONS.md`` → *Dipole chromaticity*)
    and expressed in the β-weighted form

        Q'_x = -(1/4pi) ∮ beta_x (k1 + h^2) ds
               + (1/4pi) ∮ h (gamma_x D_x - 2 alpha_x D_px) ds
               + (1/4pi) sum_edges beta_x h tan(e),
        Q'_y = +(1/4pi) ∮ beta_y k1 ds
               + (1/4pi) ∮ gamma_y h D_x ds
               - (1/4pi) sum_edges beta_y h tan(e),

    where ``h = 1/rho`` is the bending curvature, ``k1`` the (quadrupole or
    combined-function) gradient, and ``gamma_u = (1 + alpha_u^2)/beta_u``. The
    three groups of terms are:

    - **Gradient focusing** ``-(1/4pi)∮beta_x k1`` (``+`` for y) — the classic
      quadrupole natural chromaticity; both come out negative for an ordinary
      FODO. This is the only term on a straight (drift + quad) lattice.
    - **Dipole weak focusing + dispersion** ``h^2`` and the ``h(gamma D - 2 alpha
      D')`` corrections. Naively the ``h^2`` geometric focusing would dominate,
      but the dispersion term ``(1 + h D_x delta)`` in the curvilinear metric
      largely cancels it, so a pure sector bend contributes almost nothing — a
      *partial* fix (``h^2`` alone) is measurably **worse** than omitting bends.
      This whole group is xtrack-validated to ~1e-6 on bendy lattices.
    - **Combined-function curvature-sextupole feed-down** ``+2 h k1 beta_x D_x`` /
      ``-h k1 beta_y D_x`` — a Maxwell-forced 3rd-order field term of any
      combined-function *sector* magnet (without it ``div B != 0`` in the curved
      frame). It acts as a sextupole at dispersion. This is what makes the
      combined-function result match **xtrack and MAD-X** (both agree); it also
      completes the ``k1`` chromaticity of a strongly combined-function ring.
    - **Pole-face edges** ``+beta_x h tan(e)`` (``-`` for y) — a thin-kick
      contribution at each entrance/exit face, xtrack-validated to ~1e-8.

    See ``docs/CONVENTIONS.md`` → *Dipole chromaticity* for the derivation and the
    xtrack/MAD-X cross-checks.

    Thin quads are exact single-point contributions (``beta`` is continuous across
    a thin kick); thick quads and dipole bodies are integrated by ``slices``-fold
    trapezoidal sub-stepping, with the matched dispersion transported alongside
    ``beta``. The integer part of the tune is irrelevant to ``dQ/ddelta``, so no
    phase unwrapping is needed.
    """
    from .elements.dipole import Dipole, _edge_matrix
    from .elements.quadrupole import Quadrupole, ThinQuadrupole, _focusing_block

    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    xi_x = xi_y = 0.0

    def _advance(M: np.ndarray) -> None:
        """Transport beta/alpha (both planes) and the dispersion across map ``M``."""
        nonlocal bx, ax, by, ay, disp
        cx, cy = _blocks(M)
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)

    for elem in lattice.elements:
        if isinstance(elem, ThinQuadrupole):
            # beta is continuous across a thin kick, so the entrance beta is the
            # value "at" the quad; k1l is the signed integrated gradient.
            xi_x += -_INV_4PI * bx * elem.k1l
            xi_y += +_INV_4PI * by * elem.k1l
            _advance(elem.matrix(ref))
        elif isinstance(elem, Quadrupole) and elem.k1 != 0.0 and elem.length > 0.0:
            ds = elem.length / slices
            xb = _focusing_block(elem.k1, ds)  # x'' + k1 x = 0
            yb = _focusing_block(-elem.k1, ds)  # y'' - k1 y = 0
            sub4 = _transverse_4d(elem.__class__(ds, elem.k1).matrix(ref))
            int_bx = 0.5 * bx  # trapezoid: half-weight the entrance sample
            int_by = 0.5 * by
            for i in range(slices):
                bx, ax, _ = _propagate_block(xb, bx, ax)
                by, ay, _ = _propagate_block(yb, by, ay)
                disp = sub4 @ disp  # a straight quad adds no dispersive kick
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                int_bx += w * bx
                int_by += w * by
            xi_x += -_INV_4PI * elem.k1 * int_bx * ds
            xi_y += +_INV_4PI * elem.k1 * int_by * ds
        elif isinstance(elem, Dipole) and elem.length > 0.0 and (elem.angle != 0.0 or elem.k1):
            # Bends OR a straight combined-function magnet (angle=0, k1!=0, i.e. a
            # quadrupole-as-dipole): h=0 there, so the integrand reduces to -beta k1
            # and the edge terms vanish, recovering the pure quadrupole result.
            h = elem.curvature
            k1 = elem.k1
            # Entrance pole-face edge (thin kick, beta taken before the kick).
            t1 = h * math.tan(elem.e1)
            xi_x += _INV_4PI * bx * t1
            xi_y += -_INV_4PI * by * t1
            _advance(_edge_matrix(h, elem.e1))
            # Body: trapezoidal integral of the beta-form integrand, dispersion
            # transported through each sub-slice.
            ds = elem.length / slices
            sub = Dipole(ds, h * ds, k1=k1).matrix(ref)
            cx, cy = _blocks(sub)
            sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
            ix, iy = _dipole_chroma_integrand(bx, ax, by, ay, disp[0], disp[1], h, k1)
            acc_x, acc_y = 0.5 * ix, 0.5 * iy  # trapezoid: half-weight entrance
            for i in range(slices):
                bx, ax, _ = _propagate_block(cx, bx, ax)
                by, ay, _ = _propagate_block(cy, by, ay)
                disp = sub4 @ disp + subk
                ix, iy = _dipole_chroma_integrand(bx, ax, by, ay, disp[0], disp[1], h, k1)
                w = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                acc_x += w * ix
                acc_y += w * iy
            xi_x += _INV_4PI * acc_x * ds
            xi_y += _INV_4PI * acc_y * ds
            # Exit pole-face edge.
            t2 = h * math.tan(elem.e2)
            xi_x += _INV_4PI * bx * t2
            xi_y += -_INV_4PI * by * t2
            _advance(_edge_matrix(h, elem.e2))
        else:
            _advance(elem.matrix(ref))
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


def momentum_compaction(lattice: Lattice, slices: int = 64, method: str = "identity") -> float:
    r"""Momentum-compaction factor ``alpha_c`` of a periodic ``lattice``.

    A higher-momentum particle rides the dispersion orbit ``x = D_x delta`` and,
    where the orbit is curved, travels a longer (or shorter) path. The fractional
    circumference change per unit momentum deviation is the purely geometric
    integral over the ring:

        alpha_c = (1 / C) ∮ D_x(s) h(s) ds,   h(s) = 1/rho(s),  C = circumference.

    Only bending magnets contribute (``h = 0`` in drifts, quads, sextupoles), so a
    straight (dispersion-free) lattice has ``alpha_c = 0``. ``alpha_c`` carries
    **no** ``gamma0`` dependence — it is geometry only (the ``1/gamma0^2`` below
    cancels against the ``R56`` it is paired with).

    Two routes to the same number, selected by ``method``:

    ``"identity"`` (default)
        The exact symplecticity identity

            alpha_c = 1/gamma0^2 - (R51 D_x + R52 D_px + R56) / C,

        read off the **one-turn longitudinal row** on the matched dispersion orbit:
        over one turn at ``(x, px) = (D_x, D_px) delta`` the coordinate ``zeta``
        slips by ``(R51 D_x + R52 D_px + R56) delta``, which is
        ``(1/gamma0^2 - alpha_c) C delta``. Both ingredients (the one-turn matrix
        and the matched dispersion) are closed-form, so this is exact to machine
        precision — no quadrature error, and ``slices`` is ignored.

    ``"quadrature"``
        The path integral above, evaluated directly: the matched dispersion is
        transported along the lattice and inside each thick dipole ``D_x(s)`` is
        integrated by ``slices``-fold trapezoidal sub-stepping of the sub-bend map
        (``h`` is constant across a sector body). Converges onto the identity at
        ``O((h ds)^2)`` — ~1.6e-6 at the default 64 slices.

    The two routes touch **disjoint** matrix entries (the identity uses the
    longitudinal row, the integral uses the dispersion-generating ones), which is
    exactly why the quadrature is kept: it is the independent second route that
    keeps the default honest. ``tests/analytic/test_momentum_compaction.py`` holds
    them against each other and against a sympy re-derivation; the reference suite
    adds xtrack's ``momentum_compaction_factor`` and MAD-X's ``alfa``.
    """
    if method not in ("identity", "quadrature"):
        raise ValueError(f"method must be 'identity' or 'quadrature', got {method!r}")

    from .elements.dipole import Dipole

    tw0 = closed_twiss(lattice)
    if method == "identity":
        M = lattice.one_turn_matrix()
        slip = M[ZETA, X] * tw0.disp_x + M[ZETA, PX] * tw0.disp_px + M[ZETA, DELTA]
        return 1.0 / lattice.ref.gamma0**2 - slip / lattice.length

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

    Consumes :func:`momentum_compaction`'s default (exact identity) route, so ``eta``
    carries no quadrature error; ``slices`` is passed through and is therefore inert
    unless that default is overridden.
    """
    return momentum_compaction(lattice, slices) - 1.0 / lattice.ref.gamma0**2


def synchrotron_tune(lattice: Lattice, slices: int = 64) -> float:
    r"""Small-amplitude synchrotron tune ``Qs`` of a periodic ``lattice`` with RF.

    Longitudinal motion is a rotation in ``(zeta, delta)`` driven by two effects
    per turn: the **arc slip** ``Delta zeta = -eta C delta`` (path-length +
    velocity, via :func:`slip_factor`) and the **RF focusing** of the cavities,
    ``Delta delta = R65 zeta`` with ``R65 = -(q V k_rf cos phi_s)/(beta0^2 E0)``
    (see :class:`~accsim.elements.rfcavity.RFCavity`). The reduced one-turn
    synchrotron matrix (cavities lumped after the arc) is

        M_s = [[1, 0], [R65_tot, 1]] @ [[1, -eta C], [0, 1]],

    a symplectic 2x2 whose tune is ``cos(2 pi Qs) = 1/2 Tr(M_s) = 1 - R65_tot eta C / 2``.
    This reproduces the closed form

        Qs^2 = -(h eta q V cos phi_s) / (2 pi beta0^2 E0)     (small amplitude)

    to leading order (``k_rf C = 2 pi h``), and returns the exact ``arccos`` value.

    **Sourced from the slip factor, not the bare ``R56``.** On a dispersive ring
    the one-turn ``R56`` entry is *not* ``-eta C`` (it omits the ``R51 D_x +
    R52 D_px`` dispersion coupling); the arc's true longitudinal restoring uses
    ``eta``, which folds that coupling in. Building ``M_s`` from ``eta`` is what
    makes ``Qs`` correct when bends are present.

    Lumping all cavity slopes into a single thin kick is the standard smooth
    approximation; it is *exact* for a single cavity (the Stage-3 acceptance case).
    Raises :class:`UnstableLatticeError` if ``|1/2 Tr(M_s)| >= 1`` (no stable
    bucket, e.g. ``phi_s`` on the wrong side of transition, or above the
    synchrotron half-integer resonance).

    This is the textbook small-amplitude *formula*, not the exact machine tune: it
    omits the second-order synchro-betatron coupling that the full 6D one-turn map
    carries. accsim's own 6x6 one-turn map reproduces xtrack's ``tw.qs`` (the
    coupled eigen-tune) to ~1e-6; this lumped value differs from it at the
    coupling order (sub-percent on the Stage-3 test ring). See
    ``tests/reference/test_synchrotron_tune_xtrack.py``.

    ``slices`` reaches :func:`momentum_compaction` through :func:`slip_factor` and is
    therefore **inert** on its default (exact) route — raising it does not buy
    precision here.
    """
    from .elements.rfcavity import RFCavity

    cavities = [elem for elem in lattice.elements if isinstance(elem, RFCavity)]
    if not cavities:
        raise ValueError(
            "synchrotron_tune requires at least one RFCavity in the lattice; "
            "without RF there is no longitudinal focusing (Qs = 0)."
        )
    eta = slip_factor(lattice, slices)
    circumference = lattice.length
    r65_tot = sum(cav.slope(lattice.ref) for cav in cavities)
    half_trace = 1.0 - 0.5 * r65_tot * eta * circumference
    if abs(half_trace) >= 1.0:
        raise UnstableLatticeError(
            f"no stable RF bucket: 1/2 Tr(M_s) = {half_trace} (|.| >= 1). Check "
            "phi_s vs transition (phi_s=0 below, pi above) and the voltage."
        )
    return math.acos(half_trace) / (2.0 * math.pi)


def chromaticity(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Total first-order chromaticity ``(Q'_x, Q'_y)`` = natural + sextupole feed-down.

    Adds the sextupole feed-down term (:func:`_sextupole_feeddown`) to the full
    :func:`natural_chromaticity` (quadrupole gradients **and** the dipole
    weak-focusing / dispersion / edge terms). This is the quantity sextupoles exist
    to control: with the right ``k2`` at a dispersive location, the feed-down
    cancels the (negative) natural chromaticity.

    Since F2, :func:`natural_chromaticity` includes the full dipole contribution
    (weak-focusing, combined-function gradient with its curvature-sextupole
    feed-down, dispersion, and edges — all xtrack-validated), so this is the
    complete absolute total for a lattice of drifts, quads, dipoles and sextupoles.
    The sextupole feed-down term itself is pinned to a symbolic ``dQ/ddelta`` and
    cross-checked against xtrack via a with-minus-without-sextupole difference, in
    which every shared term (including the dipole chromaticity) cancels exactly (a
    sextupole's linear map is a drift, so adding it leaves beta, dispersion and the
    tunes untouched).

    **This is a design-orbit quantity, deliberately and permanently.** Every
    ingredient comes from :func:`propagate_twiss`, which uses each element's
    *on-axis* :meth:`~accsim.elements.element.Element.matrix`, so a machine whose
    closed orbit is displaced is evaluated with unperturbed ``beta`` and dispersion.
    On a steered lattice with live sextupoles that is wrong at the feed-down
    beta-beat level (I2 measures ~0.4% in ``beta_x`` for a 0.3 mm orbit at
    ``k2l = 20``; against xtrack, ~1.7e-3 in ``dqx`` for a 1.25 mm orbit).

    Use :func:`chromaticity_on_orbit` for the steered machine. This function keeps
    its design-orbit meaning rather than changing under existing callers, and a test
    pins the non-response. See ``docs/CONVENTIONS.md`` -> *Optics on the real
    (steered) orbit*.
    """
    nx, ny = natural_chromaticity(lattice, slices)
    fx, fy = _sextupole_feeddown(lattice, slices)
    return nx + fx, ny + fy


# ---------------------------------------------------------------------------
# J2: amplitude-dependent detuning (the octupole anharmonicity)
# ---------------------------------------------------------------------------

_INV_16PI = 1.0 / (16.0 * math.pi)


def amplitude_detuning(lattice: Lattice, slices: int = 64) -> np.ndarray:
    r"""First-order octupole anharmonicity ``dQ/dJ`` as a symmetric ``2x2`` [m^-1].

    Returns

        [[dQ_x/dJ_x, dQ_x/dJ_y],
         [dQ_y/dJ_x, dQ_y/dJ_y]]

    the rate at which each tune moves with each plane's action ``J`` [m rad], where
    a particle of action ``J_u`` reaches ``u_max = sqrt(2 J_u beta_u)``. Every other
    tune in this package is a property of the *machine*; this is the first quantity
    that makes the tune a property of the **particle** — which is what octupoles are
    installed for (Landau damping) and what limits how far a beam can be squeezed
    before it walks onto a resonance.

    The closed form, per octupole of integrated strength ``k3l`` at ``(beta_x,
    beta_y)``:

        dQ_x/dJ_x = + k3l beta_x^2 / (16 pi),
        dQ_y/dJ_y = + k3l beta_y^2 / (16 pi),
        dQ_x/dJ_y = dQ_y/dJ_x = - k3l beta_x beta_y / (8 pi),

    got by averaging the octupole potential ``V = k3l (x^4 - 6 x^2 y^2 + y^4)/24``
    over both betatron phases at fixed action and reading ``dQ_u = (1/2pi)
    d<V>/dJ_u``. The derivation is redone in sympy in
    ``tests/analytic/test_amplitude_detuning.py``, where the same ``dQ = (1/2pi)
    d<V>/dJ`` machinery is anchored on the quadrupole's known ``beta k1l/(4 pi)``.

    Two properties fall out of the derivation rather than being imposed, and both
    are gated: the matrix is **symmetric** (it is a second derivative of one
    averaged Hamiltonian), and the cross term is **-2x** the diagonal when
    ``beta_x = beta_y`` — a pure number, independent of ``k3l`` and of the optics.

    Thin octupoles are exact single-point contributions; a thick octupole is
    integrated by trapezoidal sub-slicing of ``beta^2`` across its body, whose
    linear map is a drift.

    **Scope.** First order in ``k3l`` and octupoles only. Sextupoles also detune,
    at *second* order in ``k2`` and through a different mechanism (the second-order
    term of perturbation theory, not the first), and they are **not** included here.
    That is no longer an unclaimed gap: :func:`sextupole_detuning` (O3) computes it,
    and :func:`total_detuning` adds the two. This function keeps its first-order
    octupole meaning rather than changing under existing callers — the two are
    different orders in different strengths, and fusing them into one number would
    hide that. A ring carrying both therefore detunes by more than this reports,
    measurably so once ``k2`` is large. Nor does this include the octupole's own
    second-order contribution, so it is the tangent at zero amplitude, not the tune
    at large amplitude.
    """
    from .elements.octupole import Octupole, ThinOctupole
    from .elements.quadrupole import _focusing_block

    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    dxx = dyy = dxy = 0.0
    for elem in lattice.elements:
        if isinstance(elem, ThinOctupole):
            dxx += +_INV_16PI * elem.k3l * bx * bx
            dyy += +_INV_16PI * elem.k3l * by * by
            dxy += -2.0 * _INV_16PI * elem.k3l * bx * by
        elif isinstance(elem, Octupole) and elem.k3 != 0.0 and elem.length > 0.0:
            ds = elem.length / slices
            db = _focusing_block(0.0, ds)  # octupole linear map is a drift
            int_xx, int_yy, int_xy = 0.5 * bx * bx, 0.5 * by * by, 0.5 * bx * by
            for i in range(slices):
                bx, ax, _ = _propagate_block(db, bx, ax)
                by, ay, _ = _propagate_block(db, by, ay)
                w = 0.5 if i == slices - 1 else 1.0  # half-weight at both ends
                int_xx += w * bx * bx
                int_yy += w * by * by
                int_xy += w * bx * by
            dxx += +_INV_16PI * elem.k3 * int_xx * ds
            dyy += +_INV_16PI * elem.k3 * int_yy * ds
            dxy += -2.0 * _INV_16PI * elem.k3 * int_xy * ds
            continue  # beta already advanced across the body
        cx, cy = _blocks(elem.matrix(lattice.ref))
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
    return np.array([[dxx, dxy], [dxy, dyy]])


# ---------------------------------------------------------------------------
# O3: the sextupole anharmonicity -- second order in k2 (J2's stated scope hole)
# ---------------------------------------------------------------------------

#: A resonance denominator this close to zero has no second-order normal form.
_RESONANT = 1e-12


def _sextupole_sites(
    lattice: Lattice, slices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(k2l, beta_x, beta_y, mu_x, mu_y)`` per thin sextupole and per thick slice.

    A thick body is the midpoint rule: ``slices`` thin kicks of ``k2 ds`` at the slice
    centres, with ``beta`` and the phase carried across by the body's linear map (a
    drift). Phases accumulate from the lattice entrance; nothing downstream depends on
    where that is (see :func:`sextupole_detuning`).
    """
    from .elements.quadrupole import _focusing_block
    from .elements.sextupole import Sextupole, ThinSextupole

    tw0 = closed_twiss(lattice)  # also the CoupledLatticeError guard
    bx, ax, mux = tw0.beta_x, tw0.alpha_x, 0.0
    by, ay, muy = tw0.beta_y, tw0.alpha_y, 0.0
    strength: list[float] = []
    site: list[tuple[float, float, float, float]] = []
    for elem in lattice.elements:
        if isinstance(elem, ThinSextupole):
            if elem.k2l != 0.0:
                strength.append(elem.k2l)
                site.append((bx, by, mux, muy))
        elif isinstance(elem, Sextupole) and elem.k2 != 0.0 and elem.length > 0.0:
            ds = elem.length / slices
            half = _focusing_block(0.0, 0.5 * ds)  # a sextupole's linear map is a drift
            for step in range(2 * slices):
                bx, ax, dmux = _propagate_block(half, bx, ax)
                by, ay, dmuy = _propagate_block(half, by, ay)
                mux, muy = mux + dmux, muy + dmuy
                if step % 2 == 0:  # the slice centre
                    strength.append(elem.k2 * ds)
                    site.append((bx, by, mux, muy))
            continue  # beta and phase are already across the body
        cx, cy = _blocks(elem.matrix(lattice.ref))
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        mux, muy = mux + dmux, muy + dmuy
    a = np.array(site, dtype=float).reshape(-1, 4)
    return np.array(strength, dtype=float), a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def sextupole_detuning(lattice: Lattice, slices: int = 32) -> np.ndarray:
    r"""Second-order sextupole anharmonicity ``dQ/dJ`` as a symmetric ``2x2`` [m^-1].

    Same shape and units as :func:`amplitude_detuning`

        [[dQ_x/dJ_x, dQ_x/dJ_y],
         [dQ_y/dJ_x, dQ_y/dJ_y]]

    and the thing that function's **Scope** paragraph says it does not compute. An
    octupole detunes at *first* order in its strength -- one phase average of one
    potential. A sextupole cannot: its first-order average vanishes (the potential is
    odd in the betatron phase), so the effect appears only at **second** order, where
    the ring's sextupoles act in *pairs*. That is why this is a double sum rather than
    a sum, and why it is quadratic in ``k2`` rather than linear.

    Writing ``S_i = k2l_i``, ``psi_x = |mu_xi - mu_xj|``, ``psi_y = |mu_yi - mu_yj|``,
    and ``C(m_x, m_y) = cos(m_x psi_x + m_y psi_y - pi Phi) / sin(pi Phi)`` with
    ``Phi = m_x Q_x + m_y Q_y``, the closed form is a sum over **all ordered pairs**
    ``(i, j)`` -- the diagonal ``i = j`` included, since a single sextupole detunes on
    its own -- of

        c_i = S_i beta_xi^(3/2),      d_i = S_i beta_xi^(1/2) beta_yi,

        dQ_x/dJ_x = -(1/64 pi) sum c_i c_j [3 C(1,0) + C(3,0)],
        dQ_x/dJ_y = +(1/16 pi) sum c_i d_j C(1,0)
                    + (1/32 pi) sum d_i d_j [C(1,-2) - C(1,2)],
        dQ_y/dJ_y = -(1/64 pi) sum d_i d_j [4 C(1,0) + C(1,2) + C(1,-2)].

    The four denominators are the lines a sextupole drives: ``Q_x`` and ``3 Q_x`` from
    the ``x^3`` part of its potential, ``Q_x +- 2 Q_y`` from the ``x y^2`` part. On any
    of them this raises :class:`ResonantLatticeError`; *near* one the answer genuinely
    diverges, and that divergence is returned rather than damped.

    **Where it comes from.** The second-order normal form of the one-turn map, redone
    from scratch in ``tests/analytic/test_sextupole_detuning.py``: each thin kick
    becomes a Lie generator, the generators are combined into one, the cubic part is
    removed by a canonical transformation, and the action-only part of what that
    leaves *is* this matrix. No coefficient here is quoted from anywhere. The machinery
    is anchored twice before it is pointed at a sextupole -- on the octupole, where it
    must reproduce :func:`amplitude_detuning`'s shipped first-order formula, and on
    **two thin quadrupoles**, where its second-order answer must equal the exact
    expansion of ``cos 2 pi Q = 1/2 Tr(M)`` of the real one-turn matrix. That second
    anchor is what pins the resonance denominator, the ``pi Q`` inside the cosine and
    the overall sign, and it is run in both beam orderings because only the
    ordering-symmetric form (hence the ``|mu_i - mu_j|``) can be right.

    Two properties fall out of the derivation rather than being imposed, and both are
    gated: the matrix is **symmetric** (it is a second derivative of one scalar normal
    form), and the answer does not depend on **where the turn is started** -- moving the
    observation point changes every ``mu`` but not the result, which is what the
    ``- pi Phi`` in each cosine is for. It is likewise unchanged by adding an integer to
    either tune.

    **Scope.** Second order in ``k2`` exactly, on the **design** orbit, for a
    transversely uncoupled lattice (:class:`CoupledLatticeError` otherwise). Skew
    sextupoles are ignored. Thick sextupoles are sub-sliced by the midpoint rule, which
    costs memory: the double sum is materialised, so it is ``O((N slices)^2)``.

    **The trap this walks past.** Do not compare this against a tracked (or PTC) tune
    shift without differencing out the sextupole-free ring first. accsim's
    :class:`~accsim.elements.drift.Drift` is *exact*, so ``x += L px/pz`` detunes all by
    itself with no magnet involved -- measured at ``0.127`` on a ring whose sextupole
    term is ``0.54``. This function reports exactly zero there, correctly: it is the
    sextupoles' contribution, not the ring's total anharmonicity.
    """
    k2l, bx, by, mux, muy = _sextupole_sites(lattice, slices)
    if k2l.size == 0:
        return np.zeros((2, 2))
    qx, qy = tunes(lattice)
    c = k2l * bx**1.5
    d = k2l * np.sqrt(bx) * by
    dmux = np.abs(mux[:, None] - mux[None, :])
    dmuy = np.abs(muy[:, None] - muy[None, :])

    def line(mx: int, my: int) -> np.ndarray:
        phi = math.pi * (mx * qx + my * qy)
        s = math.sin(phi)
        if abs(s) < _RESONANT:
            raise ResonantLatticeError(
                f"lattice sits on the {mx} Qx {my:+d} Qy sextupole resonance "
                f"(Qx = {qx:.9g}, Qy = {qy:.9g}): sin(pi Phi) = {s:.3g}"
            )
        return np.cos(mx * dmux + my * dmuy - phi) / s

    c10, c30, c12, c1m2 = line(1, 0), line(3, 0), line(1, 2), line(1, -2)
    cc, cd, dd = np.outer(c, c), np.outer(c, d), np.outer(d, d)
    dxx = -(cc * (3.0 * c10 + c30)).sum() / (64.0 * math.pi)
    dxy = (cd * c10).sum() / (16.0 * math.pi) + (dd * (c1m2 - c12)).sum() / (32.0 * math.pi)
    dyy = -(dd * (4.0 * c10 + c12 + c1m2)).sum() / (64.0 * math.pi)
    return np.array([[dxx, dxy], [dxy, dyy]])


def total_detuning(lattice: Lattice, slices: int = 64) -> np.ndarray:
    """``amplitude_detuning + sextupole_detuning``: every ``dQ/dJ`` this package claims.

    The octupoles' first-order term plus the sextupoles' second-order one, added
    **unadjusted** -- there is no fitted cross term between them, and MAD-X PTC agrees
    with the sum to nine digits, so at this order there is none to fit.

    ``slices`` applies to thick magnets in both, and defaults to
    :func:`amplitude_detuning`'s own ``64`` rather than :func:`sextupole_detuning`'s
    ``32`` **so that the octupole half of this number is exactly what calling
    ``amplitude_detuning(lattice)`` would give.** The lower default over there is a
    memory concession, not a physics one: the sextupole double sum is materialised, so
    it costs ``O((N slices)^2)``. Pass ``slices`` explicitly on a ring with many thick
    sextupoles.

    Still not the whole anharmonicity of a real ring: it is second order in ``k2``,
    first order in ``k3``, and linear in the action, and it excludes the kinematic
    detuning of the exact drift map (see :func:`sextupole_detuning`).
    """
    return amplitude_detuning(lattice, slices) + sextupole_detuning(lattice, slices)


# ---------------------------------------------------------------------------
# O4: first-order resonance driving terms -- the same normal form, read out
#     one order earlier
# ---------------------------------------------------------------------------

#: ``key -> (m_x, m_y, p_x, p_y, coefficient, source)``.  ``(m_x, m_y) = (j - k, l - m)``
#: is the monomial's charge, which fixes both the phase the term carries and the
#: resonance it is divided by; ``(p_x, p_y)`` are the powers of ``beta_x``/``beta_y``;
#: ``source`` is ``"sext"`` for a normal sextupole (strength ``k2l``) and ``"skew"`` for
#: a skew quadrupole (``k1sl``).  Every coefficient here is derived, not quoted -- see
#: :func:`resonance_driving_terms` and ``tests/analytic/test_resonance_driving_terms.py``.
_RDT_TERMS: dict[str, tuple[int, int, float, float, float, str]] = {
    "f3000": (3, 0, 1.5, 0.0, -1.0 / 48.0, "sext"),
    "f2100": (1, 0, 1.5, 0.0, -1.0 / 16.0, "sext"),
    "f1020": (1, 2, 0.5, 1.0, +1.0 / 16.0, "sext"),
    "f1011": (1, 0, 0.5, 1.0, +1.0 / 8.0, "sext"),
    "f1002": (1, -2, 0.5, 1.0, +1.0 / 16.0, "sext"),
    "f1010": (1, 1, 0.5, 0.5, +1.0 / 4.0, "skew"),
    "f1001": (1, -1, 0.5, 0.5, +1.0 / 4.0, "skew"),
}

_RdtSites = dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _rdt_sites(lattice: Lattice, slices: int) -> tuple[_RdtSites, float, float]:
    """``({"sext"|"skew": (strength, beta_x, beta_y, mu_x, mu_y)}, Q_x, Q_y)`` per source.

    Walked on the **coupling-off** optics, the way :func:`closest_tune_approach` does: a
    skew quadrupole makes :func:`closed_twiss` refuse outright, and first-order
    perturbation theory wants the unperturbed optics anyway. Thick bodies are the
    midpoint rule -- ``slices`` thin kicks at the slice centres, with beta and phase
    carried across by half-slices of the body's own decoupled linear map.
    """
    from .elements.quadrupole import _focusing_block
    from .elements.sextupole import Sextupole, ThinSextupole
    from .elements.skew_quadrupole import SkewQuadrupole, ThinSkewQuadrupole

    ref = lattice.ref
    for elem in lattice.elements:
        if isinstance(elem, (SkewQuadrupole, ThinSkewQuadrupole)):
            continue
        M = elem.matrix(ref)
        if M[np.ix_([X, PX], [Y, PY])].any() or M[np.ix_([Y, PY], [X, PX])].any():
            raise CoupledLatticeError(
                f"{type(elem).__name__} {elem.name!r} couples x and y without being a "
                f"skew quadrupole (roll={elem.roll}), and this sum walks element types: "
                "it would report f1001 = f1010 = 0 for a ring that is demonstrably "
                "coupled. The same measured guard, for the same reason, as "
                "closest_tune_approach"
            )
    decoupled = [_decoupled(elem.matrix(ref)) for elem in lattice.elements]
    one_turn = np.eye(6)
    for M in decoupled:
        one_turn = M @ one_turn
    tw0 = match_periodic(one_turn)
    bx, ax, mux = tw0.beta_x, tw0.alpha_x, 0.0
    by, ay, muy = tw0.beta_y, tw0.alpha_y, 0.0
    found: dict[str, list[tuple[float, float, float, float, float]]] = {"sext": [], "skew": []}
    for elem, M in zip(lattice.elements, decoupled, strict=True):
        thick: tuple[str, float, float] | None = None
        if isinstance(elem, ThinSextupole):
            if elem.k2l != 0.0:
                found["sext"].append((elem.k2l, bx, by, mux, muy))
        elif isinstance(elem, ThinSkewQuadrupole):
            if elem.k1sl != 0.0:
                found["skew"].append((elem.k1sl, bx, by, mux, muy))
        elif isinstance(elem, Sextupole) and elem.k2 != 0.0 and elem.length > 0.0:
            thick = ("sext", elem.k2, elem.length)
        elif isinstance(elem, SkewQuadrupole) and elem.k1s != 0.0 and elem.length > 0.0:
            thick = ("skew", elem.k1s, elem.length)
        if thick is not None:
            kind, strength, length = thick
            ds = length / slices
            if kind == "sext":
                hx = hy = _focusing_block(0.0, 0.5 * ds)  # a sextupole's own map is a drift
            else:
                hx, hy = _blocks(_decoupled(SkewQuadrupole(0.5 * ds, strength).matrix(ref)))
            for step in range(2 * slices):
                bx, ax, dmux = _propagate_block(hx, bx, ax)
                by, ay, dmuy = _propagate_block(hy, by, ay)
                mux, muy = mux + dmux, muy + dmuy
                if step % 2 == 0:  # the slice centre
                    found[kind].append((strength * ds, bx, by, mux, muy))
            continue  # optics already advanced across the body
        cx, cy = _blocks(M)
        bx, ax, dmux = _propagate_block(cx, bx, ax)
        by, ay, dmuy = _propagate_block(cy, by, ay)
        mux, muy = mux + dmux, muy + dmuy
    sites: _RdtSites = {}
    for kind, rows in found.items():
        a = np.array(rows, dtype=float).reshape(-1, 5)
        sites[kind] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4])
    return sites, mux / (2.0 * math.pi), muy / (2.0 * math.pi)


def resonance_driving_terms(lattice: Lattice, slices: int = 32) -> dict[str, complex]:
    r"""First-order resonance driving terms ``f_jklm`` at the lattice entrance.

    Returns ``{"f3000", "f2100", "f1020", "f1011", "f1002", "f1010", "f1001"}`` as
    complex numbers -- the five a **normal sextupole** drives and the two a **skew
    quadrupole** drives, which between them are every first-order term those two magnets
    produce: ``f_jklm = conj(f_kjml)``, so the ten sextupole monomials and the four
    skew-quadrupole ones are these seven and their conjugates.

    **What the number is.** :func:`sextupole_detuning` builds the normal form of the
    one-turn map and keeps the part that depends only on the actions -- the tune shift.
    An RDT is precisely what is *thrown away* there: the coefficient of the Lie generator
    that removes one non-action monomial. Writing the one-turn map as ``exp(:F:)``
    followed by the linear rotation, and expanding ``F`` in the resonance basis
    ``h_u = u_hat + i p_hat_u`` (``u_hat = u / sqrt(beta_u)``, so
    ``h_u = sqrt(2 J_u) e^{-i phi_u}``),

        F = sum_jklm  F_jklm  h_x^j conj(h_x)^k h_y^l conj(h_y)^m,

    the generator that normalises ``F`` at first order has coefficients

        f_jklm = F_jklm / (exp(-2 pi i [(j - k) Q_x + (l - m) Q_y]) - 1),

    and those are what this returns. Each source contributes, with
    ``E(m_x, m_y) = exp(-i (m_x mu_x + m_y mu_y))`` at the source and ``beta``, ``mu``
    the **unperturbed** optics there,

        sextupole, S = k2l                    skew quadrupole, K = k1sl
        F_3000 = -S bx^(3/2) E(3,0) / 48      F_1010 = +K sqrt(bx by) E(1, 1) / 4
        F_2100 = -S bx^(3/2) E(1,0) / 16      F_1001 = +K sqrt(bx by) E(1,-1) / 4
        F_1020 = +S sqrt(bx) by E(1, 2) / 16
        F_1011 = +S sqrt(bx) by E(1, 0) / 8
        F_1002 = +S sqrt(bx) by E(1,-2) / 16

    None of that is quoted from anywhere: it is read off the same Lie machinery O3 uses,
    re-derived and gated coefficient by coefficient as symbolic identities in
    ``tests/analytic/test_resonance_driving_terms.py``.

    **Which lines these are, and why that list is forced.** The charge ``(j-k, l-m)`` is
    both the phase the term carries and the resonance it is divided by, so the driven
    lines follow from the monomials rather than from memory: ``3 Q_x`` and ``Q_x`` from a
    sextupole's ``x^3``, ``Q_x`` and ``Q_x +- 2 Q_y`` from its ``x y^2``, and
    ``Q_x +- Q_y`` from a skew quadrupole's ``x y``. Sitting on one raises
    :class:`ResonantLatticeError`; approaching one makes that term diverge, which is
    physical and is returned.

    **The reference point matters -- an RDT is covariant, not invariant.** Unlike
    :func:`sextupole_detuning`, which is a property of the ring alone, this is a property
    of the ring *and* the point it is observed from. Moving the start forward past a set
    of sources, through phase advances ``(d_x, d_y)``, gives

        f_new = exp(+i (m_x d_x + m_y d_y)) * (f_old + F_crossed),

    with ``F_crossed`` the sum of the plain (undivided) ``F`` coefficients of just the
    sources stepped over. So each term rotates between sources and **jumps** at them, and
    ``|f|`` is not constant around the ring. Roll the element list to observe elsewhere.

    **Convention.** The basis is ``h_u = u_hat + i p_hat_u``, which is xtrack's and
    MAD-X's: on identical rings this function and xtrack's
    ``rdt_first_order_perturbation`` agree to round-off on all seven terms, phase
    included. The opposite basis ``h_u = u_hat - i p_hat_u`` -- the one O3's derivation is
    carried out in -- gives the complex **conjugate** of every term. That relation is
    measured, not assumed, and the physical arbiter is neither code: each term is a named
    sideband of the turn-by-turn spectrum, and its amplitude *and* phase are gated against
    these numbers by tracking.

    **Scope.** First order in the strengths; normal sextupoles and skew quadrupoles only;
    on the **design** orbit (no feed-down from a closed orbit or a misalignment); at the
    lattice entrance. Octupole terms (``f4000`` and friends), skew-sextupole terms and
    second-order RDTs are not computed. An octupole or a skew sextupole in the ring leaves
    these seven numbers correct -- it drives no line among them -- but contributes nothing
    to them either. An element that couples the planes *without* being a skew quadrupole
    (a rolled quadrupole, say) would corrupt ``f1001``/``f1010``, so it is refused with
    :class:`CoupledLatticeError` rather than silently summed as zero: the same measured
    guard, for the same reason, as :func:`closest_tune_approach`.
    """
    sites, qx, qy = _rdt_sites(lattice, slices)
    out: dict[str, complex] = {}
    for key, (mx, my, px, py, coef, kind) in _RDT_TERMS.items():
        den = np.exp(-2j * math.pi * (mx * qx + my * qy)) - 1.0
        if abs(den) < _RESONANT:
            raise ResonantLatticeError(
                f"lattice sits on the {mx} Qx {my:+d} Qy line that {key} is divided by "
                f"(Qx = {qx:.9g}, Qy = {qy:.9g}): |exp(-2 pi i Phi) - 1| = {abs(den):.3g}"
            )
        strength, bx, by, mux, muy = sites[kind]
        if strength.size == 0:
            out[key] = 0.0 + 0.0j
            continue
        term = coef * strength * bx**px * by**py * np.exp(-1j * (mx * mux + my * muy))
        out[key] = complex(term.sum() / den)
    return out


# ---------------------------------------------------------------------------
# I3: the same optics, evaluated on the real (steered) closed orbit
# ---------------------------------------------------------------------------


def _on_orbit_maps(lattice: Lattice, delta: float, step: float) -> list[np.ndarray]:
    """Per-element maps about the nonlinear closed orbit (local import breaks no cycle)."""
    from .orbit import linearised_element_maps

    return linearised_element_maps(lattice, delta=delta, step=step)


def closed_twiss_on_orbit(lattice: Lattice, *, delta: float = 0.0, step: float = 1e-7) -> Twiss:
    """Matched Twiss at the entrance, on the machine's **real** closed orbit.

    :func:`closed_twiss` with the one-turn map replaced by
    :func:`~accsim.orbit.linearised_one_turn_map`. Identical to it whenever the
    orbit is **on axis**; different by the feed-down beta-beat wherever an off-axis
    sextupole is live.

    "Nothing in the lattice is nonlinear" is no longer the other way for the two to
    coincide, and a :class:`~accsim.elements.drift.Drift` is why: its exact map is
    nonlinear, so on a steered orbit this reports a ``disp_y`` the design optics puts
    at exactly zero — the dispersion a transverse orbit *angle* makes. That is the
    point of the exact map, not a discrepancy to reconcile. A vertical steerer in an
    otherwise perfect ring is the clean case: design optics ``0``, this ``0.2590571``,
    xtrack ``0.2590571``.

    Raises :class:`CoupledLatticeError` if the *on-orbit* map is x-y coupled — a
    normal sextupole at ``y_co != 0`` is a skew quadrupole, so this happens on a
    vertically steered machine whose design optics is perfectly uncoupled. Use
    :func:`coupled_twiss_on_orbit` there.
    """
    from .orbit import linearised_one_turn_map

    return match_periodic(linearised_one_turn_map(lattice, delta=delta, step=step))


def propagate_twiss_on_orbit(
    lattice: Lattice, *, delta: float = 0.0, step: float = 1e-7
) -> list[Twiss]:
    r"""Twiss at every element boundary, on the machine's **real** closed orbit.

    The milestone's headline function. ``beta``, ``alpha`` and the dispersion are
    transported through the maps a particle near the *actual* orbit sees rather
    than the design matrices, so a steered machine with live sextupoles reports the
    beta-beat it really has. For a single thin sextupole that beat is the classic
    single-gradient form

        dbeta(s)/beta(s) = -k2l x_co beta(s_src) cos(2 |dpsi| - 2 pi Q) / (2 sin 2 pi Q)

    (``+`` in ``y``), first order in the orbit offset, with a second-order residual
    — both orders measured in the analytic suite.

    Thick sextupoles are handled without special-casing, because the maps come from
    differentiating each element's real ``track()``; only
    :func:`~accsim.orbit.linearised_lattice`, and so the chromaticity functions
    below, have to refuse them.
    """
    maps = _on_orbit_maps(lattice, delta, step)
    one_turn = np.eye(6)
    for m in maps:
        one_turn = m @ one_turn
    return propagate_twiss(lattice, match_periodic(one_turn), maps=maps)


def tunes_on_orbit(
    lattice: Lattice, *, delta: float = 0.0, step: float = 1e-7
) -> tuple[float, float]:
    """Full tunes ``(Qx, Qy)`` of the optics on the real closed orbit.

    :func:`tunes` about the steered orbit. Like it, this is the **accumulated**
    phase advance divided by ``2 pi``, so the integer part is carried. That is not
    cosmetic: :func:`chromaticity_on_orbit`'s independent gate central-differences
    this function in ``delta``, and a fractional-only tune read off the one-turn
    map with ``acos`` would be wrong by an integer whenever the two sample points
    straddled a half integer — a hazard removed rather than guarded.
    """
    end = propagate_twiss_on_orbit(lattice, delta=delta, step=step)[-1]
    return end.mu_x / (2.0 * math.pi), end.mu_y / (2.0 * math.pi)


def coupled_twiss_on_orbit(
    lattice: Lattice, *, delta: float = 0.0, step: float = 1e-7
) -> CoupledTwiss:
    """Edwards-Teng normal-mode optics on the real closed orbit.

    The vertically steered counterpart of :func:`closed_twiss_on_orbit`: a normal
    sextupole at ``y_co != 0`` feeds down a **skew** gradient ``k2l y_co``, so the
    machine the beam sees is x-y coupled even though its design map is exactly
    block-diagonal. G2's machinery, reached from the orbit.
    """
    from .orbit import linearised_one_turn_map

    return match_periodic_coupled(linearised_one_turn_map(lattice, delta=delta, step=step))


def propagate_coupled_twiss_on_orbit(
    lattice: Lattice, *, delta: float = 0.0, step: float = 1e-7
) -> list[CoupledTwiss]:
    """Normal-mode optics at every boundary, on the real closed orbit."""
    return propagate_coupled_twiss(lattice, maps=_on_orbit_maps(lattice, delta, step))


def natural_chromaticity_on_orbit(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    """:func:`natural_chromaticity` of the machine the beam on the real orbit sees.

    Evaluated on :func:`~accsim.orbit.linearised_lattice`, so it picks up both the
    beta-beat of the feed-down gradient *and* that gradient's own chromaticity —
    an off-axis sextupole is a quadrupole, and a quadrupole has natural
    chromaticity like any other.

    Exposed separately from :func:`chromaticity_on_orbit` because the difference of
    the two is the sextupole feed-down term at the beaten ``beta`` and dispersion,
    which is exactly the quantity a tracked, linearised-map measurement can reach
    independently — that is how the analytic suite gates this pair.
    """
    from .orbit import linearised_lattice

    return natural_chromaticity(linearised_lattice(lattice), slices)


def chromaticity_on_orbit(lattice: Lattice, slices: int = 64) -> tuple[float, float]:
    r"""Total first-order chromaticity of the machine on its **real** closed orbit.

    The orbit-aware counterpart of :func:`chromaticity`, which is and remains a
    design-orbit quantity. Both of its terms move when the machine is steered: the
    natural part because the feed-down gradient beats ``beta`` and dispersion *and*
    contributes its own gradient chromaticity, and the sextupole feed-down term
    because it is an integral over the beaten ``beta`` and ``D_x``.

    **Why this is not computed by tracking — a claim L1, L2 and L3 have retired.**
    It was once absolute: accsim's element maps carried no ``delta`` dependence at
    all, so linearising the tracked map about the off-momentum orbit measured the
    sextupole feed-down term and was *exactly* blind to the natural chromaticity,
    which accsim supplies analytically (F2). Three milestones changed that. The
    exact :class:`~accsim.elements.drift.Drift` (L1) made a drift a first-order
    chromatic element; the momentum-dependent
    :class:`~accsim.elements.quadrupole.Quadrupole` (L2) gave the thick quadrupole
    its own ``k1/(1+delta)``; and the exact :class:`~accsim.elements.dipole.Dipole`
    (L3) made the bend's weak focusing, dispersion and path length momentum-dependent
    together, because they are one circle. The tracked route now recovers the natural
    chromaticity **in full on a bendy ring too** — measured to ``1.6e-8`` relative on
    the analytic suite's arc, where it read 45%, then 58%.

    **The one place tracking still cannot follow is a *bending* combined-function
    magnet, and L4 changed what is missing rather than closing it.** L4 gave the curved
    quadrupole the expanded map (MAD-X's ``track_thick_cfd``, xtrack's
    ``mat-kick-mat``) including F2's curvature-sextupole feed-down, so the feed-down is
    now tracked; what that family drops instead is the curvilinear metric factor
    ``(1 + h x)`` of ``x' = px(1+hx)/pz``, which on the dispersed orbit **is** the
    ``h (gamma_x D_x - 2 alpha_x D_px)`` / ``gamma_y h D_x`` group of the integrand
    above. So tracking such a ring converges to this integral *minus* that group, and
    since that group is what largely cancels the geometric ``-beta_x h^2`` focusing, the
    difference is not small. A *straight* gradient magnet has ``h = 0``, both the
    curvature-sextupole and the metric group vanish identically, and tracking is
    complete.

    So this function stays built on the validated integrals over
    :func:`~accsim.orbit.linearised_lattice` — it is the deliverable, and it is what
    xtrack's exact bend models agree with. The tracked route is kept as the independent
    gate: on the whole of it for any ring of pure bends, and against the
    metric-group-removed integral where a *bending* gradient magnet is involved
    (``tests/analytic/test_curved_quadrupole.py``).

    One caveat that L3 introduced: on a **steered** machine the two answers separate
    at first order in the orbit (``2.05e-5`` at a ``4e-4`` kick, exactly zero without
    one), because this integral is taken over the design optics while tracking sees the
    machine the beam is in.

    Raises :class:`CoupledLatticeError` on a vertically steered machine (the
    equivalent lattice then carries a skew quadrupole, and the 2x2 Courant-Snyder
    integrals are not valid there), and :class:`NotImplementedError` for a thick
    sextupole — see :func:`~accsim.orbit.linearised_lattice` for both.
    """
    from .orbit import linearised_lattice

    return chromaticity(linearised_lattice(lattice), slices)


# ---------------------------------------------------------------------------
# M1: the optics off-momentum — chromatic functions and second-order chromaticity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChromaticTwiss:
    r"""How the linear optics at one point move when the momentum does.

    Every other Twiss object in this package describes the machine at **one**
    momentum. This one is its derivative: the rate at which ``beta`` and ``alpha``
    change with ``delta``, in the MAD8 normalisation both MAD-X and xtrack report.

    ``dbeta_u`` [m] and ``dalpha_u`` [1] are the raw derivatives
    ``dbeta_u/ddelta`` and ``dalpha_u/ddelta``. The other three are the
    conventional combinations (MAD8 physics manual section 6.3, and ``xtrack``'s
    ``bx_chrom`` / ``ax_chrom`` / ``wx_chrom``):

        b_u = (dbeta_u/ddelta) / beta_u,
        a_u = dalpha_u/ddelta - (dbeta_u/ddelta) alpha_u / beta_u,
        w_u = sqrt(a_u^2 + b_u^2).

    ``b_u`` is the *relative* beta-beat per unit momentum, which is what makes it
    comparable between machines and between points. ``w_u`` is the amplitude of the
    chromatic perturbation and is the quantity a lattice designer squeezes: a large
    ``w`` at an insertion means the off-momentum particles in the bunch are focused
    to a visibly different spot than the on-momentum ones.

    **Derivative with respect to** ``delta``, **not** ``pzeta``. The two differ by
    ``beta0`` factors and so agree on an ultra-relativistic ring to round-off; the
    choice is pinned on a low-``gamma0`` ring in the reference suite, where it is a
    real difference rather than a naming preference. See ``docs/CONVENTIONS.md`` ->
    *Chromatic functions*.
    """

    s: float
    dbeta_x: float
    dalpha_x: float
    dbeta_y: float
    dalpha_y: float
    a_x: float
    b_x: float
    w_x: float
    a_y: float
    b_y: float
    w_y: float


def _chromatic_step(delta: float) -> float:
    """Validate the momentum step shared by the two chromatic entry points."""
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0 (it is a step size), got {delta}")
    return float(delta)


def chromatic_functions(
    lattice: Lattice, *, delta: float = 1e-3, step: float = 1e-7
) -> list[ChromaticTwiss]:
    r"""``dbeta/ddelta``, ``dalpha/ddelta`` and the MAD8 ``a``/``b``/``w`` at every boundary.

    Central-differences :func:`propagate_twiss_on_orbit` at ``+delta`` and
    ``-delta``: the optics are re-matched **on the closed orbit at each momentum**,
    so every source of momentum dependence the package models is included at once —
    the thick quadrupole's ``k1/(1+delta)`` (L2), the exact drift and dipole (L1,
    L3), and a sextupole's feed-down at the dispersion orbit it sits on.

    Why finite differences rather than a perturbation integral: it is the method
    both references use, so a disagreement arbitrates the **maps** rather than the
    truncation order of two different expansions. That is B2's argument, and it is
    the reason this function is a thin layer over machinery that already exists.

    ``delta`` is the momentum step, not a tolerance. The error is ``O(delta^2)``
    from truncation and ``O(orbit noise / delta)`` from the closed-orbit solve, so
    the useful range is bounded at both ends; ``1e-3`` sits in the flat middle for
    the rings this package builds. The analytic suite gates the **order** (halving
    ``delta`` quarters the residual against the symbolic answer) rather than a value
    at one step.

    ``step`` is passed through to the Jacobian that builds each element's linearised
    map.

    Raises :class:`CoupledLatticeError` through :func:`propagate_twiss_on_orbit` if
    the on-orbit map is x-y coupled at either momentum — the Courant-Snyder ``beta``
    this differentiates is not defined there. The coupled analogue would be built on
    G2's Edwards-Teng optics and is not built here.
    """
    d = _chromatic_step(delta)
    plus = propagate_twiss_on_orbit(lattice, delta=+d, step=step)
    minus = propagate_twiss_on_orbit(lattice, delta=-d, step=step)
    centre = propagate_twiss_on_orbit(lattice, delta=0.0, step=step)

    out: list[ChromaticTwiss] = []
    for tp, tm, t0 in zip(plus, minus, centre, strict=True):
        dbx = (tp.beta_x - tm.beta_x) / (2.0 * d)
        dby = (tp.beta_y - tm.beta_y) / (2.0 * d)
        dax = (tp.alpha_x - tm.alpha_x) / (2.0 * d)
        day = (tp.alpha_y - tm.alpha_y) / (2.0 * d)
        b_x = dbx / t0.beta_x
        b_y = dby / t0.beta_y
        a_x = dax - dbx * t0.alpha_x / t0.beta_x
        a_y = day - dby * t0.alpha_y / t0.beta_y
        out.append(
            ChromaticTwiss(
                s=t0.s,
                dbeta_x=dbx,
                dalpha_x=dax,
                dbeta_y=dby,
                dalpha_y=day,
                a_x=a_x,
                b_x=b_x,
                w_x=math.hypot(a_x, b_x),
                a_y=a_y,
                b_y=b_y,
                w_y=math.hypot(a_y, b_y),
            )
        )
    return out


def second_order_chromaticity(
    lattice: Lattice, *, delta: float = 1e-3, step: float = 1e-7
) -> tuple[float, float]:
    r"""Second-order chromaticity ``(Q''_x, Q''_y) = d^2 Q / ddelta^2``.

    The curvature of the tune-versus-momentum curve, where :func:`chromaticity` is
    its slope. It is the quantity that decides how far off-momentum a particle can
    be before the *linear* chromaticity correction stops describing it — a ring
    whose sextupoles zero ``Q'`` still walks its tune onto a resonance at large
    ``delta`` if ``Q''`` is big.

    **Plain second difference of** :func:`tunes_on_orbit`, ``(Q(+d) - 2 Q(0) +
    Q(-d)) / d^2``, so it is ``d^2Q/ddelta^2`` and **not** half of it. The two
    conventions differ by exactly the factor a remembered formula gets wrong, and
    the analytic suite pins this one against a symbolic second derivative.

    :func:`tunes_on_orbit` carries the integer part of the tune, which this needs and
    an ``acos`` of the one-turn map would not supply: a second difference of
    fractional tunes is wrong by an integer whenever two of the three sample points
    straddle a half integer.

    **This is validated, and the disagreement M1 could not place was the drift model
    (M2).** On a **bend-free** ring it agrees with a symbolic closed form and with
    both references. On a ring **with bends** accsim, xtrack and MAD-X give three
    different answers (``0.7931``, ``0.7520``, ``0.7044`` on the analytic suite's arc)
    while agreeing on ``Q`` to ten digits and on ``Q'`` to seven — and the cause is
    that accsim's :class:`~accsim.elements.drift.Drift` is **exact**
    (``x += L px/pz``) where xtrack's default and MAD-X's TWISS drift are **paraxial**
    (``x += L px/(1+delta)``). The two coincide identically when the closed orbit is
    straight, which is why the bend-free control agreed; with bends the orbit carries
    ``px ~ D_px delta``, so they differ at ``O(delta^2)`` — landing on ``Q''`` while
    leaving ``Q`` and ``Q'`` untouched — and proportionally to the square of the
    bending angle.

    Set ``xt.Drift(model="exact")`` and xtrack reproduces this function to ``1e-8``
    relative in the vertical plane and to the two codes' own second-difference noise
    in the horizontal. On a five-element ring whose ``Q''`` is derived from lab-frame
    geometry at sixty digits, this function converges onto the **exact**-drift number
    at second order in ``delta``, and xtrack's default converges onto the *paraxial*
    one — so the two models are separately confirmed rather than merely reconciled.
    MAD-X sits within ``7e-4`` of the paraxial answer there, the small remainder being
    its own second-order TWISS maps; its drift model cannot be changed, so agreement
    with MAD-X on a bendy ring is unreachable by construction.

    See ``docs/CONVENTIONS.md`` -> *The drift model is what splits Q'' on a bendy
    ring*, ``tests/analytic/test_chromatic_arbiter.py``, and M2 in
    ``docs/ROADMAP.md``.

    ``delta`` and ``step`` are as in :func:`chromatic_functions`. A second difference
    divides by ``delta^2``, so closed-orbit noise enters as ``1/delta^2`` — twice as
    steeply as it does for a first derivative, which is why the default is looser
    here than a first-order chromaticity would want.
    """
    d = _chromatic_step(delta)
    qp = tunes_on_orbit(lattice, delta=+d, step=step)
    q0 = tunes_on_orbit(lattice, delta=0.0, step=step)
    qm = tunes_on_orbit(lattice, delta=-d, step=step)
    return (
        (qp[0] - 2.0 * q0[0] + qm[0]) / (d * d),
        (qp[1] - 2.0 * q0[1] + qm[1]) / (d * d),
    )


# ---------------------------------------------------------------------------
# M3: second-order dispersion — where the off-momentum orbit is, past the line
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondOrderDispersion:
    r"""The off-momentum closed orbit's Taylor expansion in ``delta``, to second order.

    Linear dispersion says the orbit moves *proportionally* to the momentum error.
    It does not: ``x_co(delta) = disp_x delta + 1/2 ddisp_x delta^2 + ...``, and this
    object carries both coefficients at one point ``s``.

    ``disp_x``/``disp_px``/``disp_y``/``disp_py`` [m, rad, m, rad] are
    ``d(x, px, y, py)/ddelta`` — the same quantity as :class:`Twiss`'s ``disp_*``,
    but *measured on the tracked orbit* rather than solved for from the linear maps.
    ``ddisp_*`` are the second derivatives ``d^2(x, px, y, py)/ddelta^2``, in
    ``[m, rad, m, rad]`` again.

    **Full second derivative, not half of it**, matching ``xtrack``'s ``ddx`` /
    ``ddpx`` / ``ddy`` / ``ddpy``. MAD-X's ``DDX`` is a *different* number — it is
    the coefficient of ``pt^2`` in an expansion in the **energy** deviation, so it
    is half of this after a change of variables; the exact relation is pinned in
    ``tests/reference/test_second_order_dispersion_madx.py`` and recorded in
    ``docs/CONVENTIONS.md`` -> *Second-order dispersion*.

    This is a property of the **orbit**, not of the optics about it, which is what
    makes it defined on a lattice where :func:`chromatic_functions` is not: an x-y
    coupled ring has no Courant-Snyder ``beta`` to differentiate, but it certainly
    has a closed orbit, and a skew quadrupole sitting at horizontal dispersion gives
    that orbit a *vertical* second-order dispersion.
    """

    s: float
    disp_x: float
    disp_px: float
    disp_y: float
    disp_py: float
    ddisp_x: float
    ddisp_px: float
    ddisp_y: float
    ddisp_py: float


def second_order_dispersion(
    lattice: Lattice,
    *,
    delta: float = 1e-3,
    tol: float = 1e-15,
    step: float = 1e-8,
) -> list[SecondOrderDispersion]:
    r"""``d^2(x, px, y, py)/ddelta^2`` (and the first derivative) at every boundary.

    Central-differences the **tracked** closed orbit
    (:func:`~accsim.orbit.propagate_orbit_nonlinear`) at ``+delta``, ``0`` and
    ``-delta``. Every source of curvature the package models is therefore included
    at once and none is put in by hand: the bend's own ``1/(1+delta)`` stiffness, the
    thick quadrupole's ``k1/(1+delta)`` (L2), the exact drift and sector bend (L1,
    L3), and a sextupole's feed-down at the dispersion orbit it sits on.

    **A linear-matrix machine has none of this.** Every ``Element.matrix()`` in this
    package is ``delta``-independent, so the *affine* closed orbit is exactly
    ``D delta`` and its second derivative is identically zero. What this function
    returns is, in full, the difference between the map a particle follows and the
    matrix used to describe it — which is why it is a cross-check of the exact maps
    rather than a re-reading of the linear ones.

    Returns ``len(lattice) + 1`` points, aligned with :func:`propagate_twiss`.

    ``delta`` is the momentum step, not a tolerance. Truncation is ``O(delta^2)``
    (the fourth derivative over twelve) and closed-orbit noise enters as
    ``1/delta^2``, so the useful range is bounded at both ends; the analytic suite
    gates the **order** — halving ``delta`` quarters the residual against a
    sixty-digit answer — rather than a value at one step.

    ``tol`` is tighter than :func:`~accsim.orbit.closed_orbit_nonlinear`'s own
    default **on purpose, and the difference is measurable**: a second difference
    divides by ``delta^2``, so that function's ``1e-14`` would land as ``~6e-9`` of
    noise in ``ddisp_x`` at the default step — a third of the truncation error, for
    nothing. At ``1e-15`` Newton's last step takes the orbit to ``~1e-19`` and the
    noise disappears under the truncation. ``step`` is the Jacobian step of the same
    solve.

    **On a ring that closes on the axis the drift model does not matter here, and that
    is the milestone's finding.** accsim's :class:`~accsim.elements.drift.Drift` is
    exact where xtrack's default and MAD-X's are paraxial, and M2 showed that splits
    ``Q''`` by 5% on a ring with bends. It does **not** split this on an unsteered ring:
    the exact drift exceeds the paraxial one by ``L px (px^2+py^2)/(2 (1+delta)^3)``, so
    with ``px = a + b delta`` on the closed orbit the ``delta^2`` part of the difference
    is ``3 a b^2`` — and the **on-momentum** orbit angle ``a`` is zero for any ring whose
    orbit runs down the axis. ``Q''`` is split anyway, because it differentiates the
    *Jacobian* about the orbit and ``d/dpx`` of the same term is ``O(b^2 delta^2)``: one
    order lower, and free of ``a``. So all three codes agree on this quantity to
    ``~1e-7`` where they disagreed by 5% on the other.

    **The condition is real, not a formality.** Steer the orbit off axis — a corrector,
    a misalignment (K1), an uncorrected error orbit — and the split returns, first order
    in ``a`` and second order in ``b``: a 10 mrad steerer on M2's minimal ring splits
    ``ddisp_x`` by ``6.8e-3`` relative and ``disp_x`` by ``4.1e-4``. On such a machine
    this function and a paraxial reference are measuring different things, exactly as
    they are for ``Q''``. See M3 in ``docs/ROADMAP.md`` and ``docs/CONVENTIONS.md`` ->
    *Second-order dispersion*.

    Raises :class:`~accsim.orbit.ClosedOrbitError` or
    :class:`~accsim.orbit.OrbitConvergenceError` through the orbit solve. Unlike
    :func:`chromatic_functions` it does **not** require an uncoupled lattice.
    """
    from .orbit import closed_orbit_nonlinear, propagate_orbit_nonlinear

    d = _chromatic_step(delta)

    def _orbit(value: float) -> np.ndarray:
        o0 = closed_orbit_nonlinear(lattice, delta=value, tol=tol, step=step)
        return np.array(propagate_orbit_nonlinear(lattice, o0, delta=value))

    plus, centre, minus = _orbit(+d), _orbit(0.0), _orbit(-d)
    first = (plus - minus) / (2.0 * d)
    second = (plus - 2.0 * centre + minus) / (d * d)

    out: list[SecondOrderDispersion] = []
    s = 0.0
    for i in range(len(lattice.elements) + 1):
        if i:
            s += lattice.elements[i - 1].length
        out.append(
            SecondOrderDispersion(s, *(float(v) for v in first[i]), *(float(v) for v in second[i]))
        )
    return out


# ======================================================================================
# Normalised coordinates: the linear normal form (axis O)
# ======================================================================================


class NormalFormError(ValueError):
    """Raised when a one-turn map has no linear normal form.

    The map is stable but **degenerate**: one mode's eigenvalue is repeated with no
    second eigenvector, so its symplectic norm ``Re(v) . S . Im(v)`` vanishes and there
    is no plane to rotate in. The case that actually occurs is an **RF-free ring** asked
    for its 6D form: ``zeta`` and ``delta`` are both eigenvalue-``1`` directions of the
    one-turn map (nothing restores the arrival time), which is the same degeneracy
    :func:`~accsim.orbit.closed_orbit_6d` refuses and the spin axis met three times
    before that. Use ``method="4d"`` for such a ring, or give it a cavity.
    """


def _unit_symplectic(dim: int) -> np.ndarray:
    """Block-diagonal ``[[0, 1], [-1, 0]]``, the form ``W`` must preserve."""
    S = np.zeros((dim, dim))
    for plane in range(dim // 2):
        S[2 * plane, 2 * plane + 1] = 1.0
        S[2 * plane + 1, 2 * plane] = -1.0
    return S


@dataclass(frozen=True)
class NormalForm:
    r"""The change of variables that turns the one-turn map into a rotation.

    ``M = W R W^-1`` with ``R = diag(Rot(2 pi Q_1), ...)`` block-diagonal in 2x2
    rotations, so in normalised coordinates ``u = W^-1 x`` a turn moves each mode around
    a circle of fixed radius. The radius squared over two is the mode's **action**
    (:func:`actions`), the invariant that replaces the per-plane Courant-Snyder one when
    the planes are coupled.

    **The parameterisation is a choice and it is recorded here** (and in
    ``docs/CONVENTIONS.md``): ``M = W R W^-1`` alone does not determine ``W`` — right
    multiplication by anything commuting with ``R`` preserves it, which is a per-plane
    scale *and* a per-plane rotation. Symplecticity fixes the scale; the phase is fixed
    by rotating each eigenvector until its own plane's **position** component is real and
    positive. That makes ``W[0, 1] = W[2, 3] = W[4, 5] = 0`` and makes the 2x2 diagonal
    blocks the Courant-Snyder matrix ``[[sqrt(beta), 0], [-alpha/sqrt(beta),
    1/sqrt(beta)]]`` — which is why :attr:`mode_beta` can be compared against
    :func:`closed_twiss` at all. It is xtrack's convention too, but the agreement with
    Stage 1's independently-derived ``beta``/``alpha`` is what justifies it here.

    Columns are ``[Re v_1, Im v_1, Re v_2, Im v_2, ...]``, normalised so that
    ``Re(v) . S . Im(v) = 1``. Modes are labelled by the plane each eigenvector lives in,
    so mode 1 is the ``x``-like one, mode 2 the ``y``-like one and (in 6D) mode 3 the
    longitudinal one — the same rule :func:`normal_mode_tunes` uses, and one that only
    becomes ambiguous exactly on a coupling resonance.
    """

    w: np.ndarray
    """The normalising matrix: ``x = W u`` takes normalised coordinates to lab ones."""
    w_inv: np.ndarray
    """``W^-1``: lab coordinates to normalised ones."""
    rotation: np.ndarray
    """``R``, block-diagonal 2x2 rotations by ``2 pi Q_i`` (``M = W R W^-1``)."""
    tunes: tuple[float, ...]
    """Fractional mode tunes in ``[0, 1)`` — two for ``4d``, three for ``6d``."""
    method: str
    """``"4d"`` (transverse block only) or ``"6d"`` (the full map)."""

    @property
    def dim(self) -> int:
        """``4`` or ``6`` — the size of the space that was normalised."""
        return int(self.w.shape[0])

    @property
    def mode_beta(self) -> tuple[float, ...]:
        """Each mode's beta function, ``W[2i, 2i]^2 + W[2i, 2i+1]^2``.

        For the transverse modes of an **uncoupled** lattice these are exactly
        :func:`closed_twiss`'s ``beta_x``/``beta_y``. In ``6d`` on a ring with RF they
        are **not**, and that is physics rather than disagreement — see
        :attr:`dispersion`.
        """
        return tuple(float(v) for v in np.diag(self.betas))

    @property
    def mode_alpha(self) -> tuple[float, ...]:
        """Each mode's alpha, ``-(W[2i,2i] W[2i+1,2i] + W[2i,2i+1] W[2i+1,2i+1])``."""
        return tuple(float(v) for v in np.diag(self.alphas))

    @property
    def dispersion(self) -> np.ndarray:
        """``(D_x, D_px, D_y, D_py)`` read off the longitudinal mode. ``6d`` only.

        This is the **dynamic** dispersion: the transverse excursion that accompanies a
        momentum *oscillating at the synchrotron tune*, not the matched
        :class:`Twiss` dispersion, which is the response to a momentum held fixed. The
        two differ at second order in ``Q_s`` and agree in the ``Q_s -> 0`` limit; on a
        strong-RF ring the difference is tens of percent and neither number is wrong.

        The formula is the one xtrack's ``twiss`` reports as ``dx``: the mode-3 columns
        with the ``zeta`` direction projected out.
        """
        if self.method != "6d":
            raise NormalFormError("dispersion needs the 6D normal form (method='6d')")
        return _dispersion_from_w(self.w)

    @property
    def betas(self) -> np.ndarray:
        """Mais-Ripken ``B[plane, mode]``; see :func:`_ripken_betas`. Constant along
        an element, so :class:`NormalFormPoint` is where it becomes interesting."""
        return _ripken_betas(self.w)

    @property
    def alphas(self) -> np.ndarray:
        """Mais-Ripken ``A[plane, mode]``."""
        return _ripken_alphas(self.w)

    @property
    def gammas(self) -> np.ndarray:
        """Mais-Ripken ``G[plane, mode]``."""
        return _ripken_gammas(self.w)

    @property
    def crab_dispersion(self) -> np.ndarray:
        """The crab dispersion at the entrance; see
        :attr:`NormalFormPoint.crab_dispersion`. ``6d`` only."""
        if self.method != "6d":
            raise NormalFormError("crab dispersion needs the 6D normal form (method='6d')")
        return _crab_dispersion_from_w(self.w)


def _paired_modes(eigvals: np.ndarray, eigvecs: np.ndarray) -> list[int]:
    """One eigenvector index per conjugate pair, ordered by the plane it lives in.

    Within a pair the representative is the one with **positive** symplectic norm, which
    is what fixes each mode's rotation sense (and so puts its tune in ``[0, 1)`` rather
    than in the ``acos``-ambiguous ``[0, 0.5]``). The pair-to-plane assignment maximises
    the total per-plane weight, so it is a permutation even when two modes prefer the
    same plane -- near a coupling resonance the labelling is arbitrary but it is never
    two-modes-one-plane.
    """
    dim = len(eigvals)
    S = _unit_symplectic(dim)
    remaining = list(range(dim))
    representatives: list[int] = []
    while remaining:
        i = remaining.pop(0)
        j = min(remaining, key=lambda k: abs(eigvals[i] - np.conj(eigvals[k])))
        remaining.remove(j)
        v = eigvecs[:, i]
        norm = float(np.real(v.real @ S @ v.imag))
        representatives.append(i if norm > 0.0 else j)
    weight = np.array(
        [
            [
                abs(eigvecs[2 * plane, m]) ** 2 + abs(eigvecs[2 * plane + 1, m]) ** 2
                for m in representatives
            ]
            for plane in range(dim // 2)
        ]
    )
    planes, order = linear_sum_assignment(weight, maximize=True)
    return [representatives[order[list(planes).index(plane)]] for plane in range(dim // 2)]


def normal_form(one_turn: np.ndarray, *, method: str = "6d", atol: float = 1e-6) -> NormalForm:
    r"""The linear normal form of a one-turn map: ``M = W R W^-1``.

    ``method="6d"`` normalises the full map — three modes, and the longitudinal one
    exists only if the ring has an RF cavity. ``method="4d"`` normalises the transverse
    ``(x, px, y, py)`` block alone, which is the right form for a ring without RF and the
    one whose modes are the Edwards-Teng modes of :func:`coupled_twiss`.

    Raises :class:`UnstableLatticeError` if any eigenvalue leaves the unit circle by more
    than ``atol`` (the motion grows without bound, so there is no rotation to conjugate
    to), and :class:`NormalFormError` if a mode is degenerate -- in practice, a ring with
    no cavity asked for its 6D form.

    See :class:`NormalForm` for the parameterisation, which is a *choice*: the identity
    ``M = W R W^-1`` and symplecticity together leave one free rotation angle per plane.
    """
    if method not in ("4d", "6d"):
        raise ValueError(f"method must be '4d' or '6d', got {method!r}")
    M = np.asarray(one_turn, dtype=float)
    if method == "4d" and M.shape == (4, 4):
        pass  # already the transverse block (what _transverse_4d would have returned)
    elif M.shape == (DIM, DIM):
        M = _transverse_4d(M) if method == "4d" else M
    else:
        raise ValueError(f"expected a {DIM}x{DIM} one-turn matrix, got shape {M.shape}")
    dim = M.shape[0]
    S = _unit_symplectic(dim)

    eigvals, eigvecs = np.linalg.eig(M)
    if not np.allclose(np.abs(eigvals), 1.0, atol=atol, rtol=0.0):
        raise UnstableLatticeError(
            f"lattice unstable: eigenvalue moduli {np.abs(eigvals)} are not all on the "
            "unit circle, so the map is not conjugate to a rotation."
        )
    modes = _paired_modes(eigvals, eigvecs)

    columns: list[np.ndarray] = []
    for plane, m in enumerate(modes):
        v = eigvecs[:, m] * np.exp(-1j * np.angle(eigvecs[2 * plane, m]))
        a, b = np.real(v), np.imag(v)
        norm_sq = float(a @ S @ b)
        if norm_sq <= _DEGENERATE_NORM * float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-300):
            raise NormalFormError(
                f"mode {plane + 1} is degenerate (symplectic norm {norm_sq:.3e}): the map "
                "has a repeated eigenvalue with no plane of rotation. A ring with no RF "
                "cavity has no 6D normal form -- use method='4d'."
            )
        scale = 1.0 / math.sqrt(norm_sq)
        columns += [a * scale, b * scale]

    w = np.array(columns).T
    rotation = np.zeros((dim, dim))
    tune_list: list[float] = []
    for plane, m in enumerate(modes):
        mu = float(np.angle(eigvals[m]))
        c, s = math.cos(mu), math.sin(mu)
        rotation[2 * plane : 2 * plane + 2, 2 * plane : 2 * plane + 2] = [[c, s], [-s, c]]
        tune_list.append((mu / (2.0 * math.pi)) % 1.0)
    return NormalForm(w, np.linalg.inv(w), rotation, tuple(tune_list), method)


def to_normalized(
    form: NormalForm | NormalFormPoint, state: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Lab coordinates to normalised ones, ``u = W^-1 x``.

    ``state`` may be a full 6D ``(x, px, y, py, zeta, delta)`` even for a ``4d`` form, in
    which case the transverse part is taken and a length-4 vector comes back. No
    emittance is involved: these are the coordinates in which a turn is a rotation, not
    the sigma-scaled ones. Divide by ``sqrt(emittance)`` per mode for those.
    """
    x = np.asarray(state, dtype=float)
    if x.shape == (DIM,) and form.dim == 4:
        x = x[_TRANSVERSE]
    if x.shape != (form.dim,):
        raise ValueError(f"expected a length-{form.dim} state (or a 6D one), got {x.shape}")
    return form.w_inv @ x


def from_normalized(
    form: NormalForm | NormalFormPoint, normalized: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Normalised coordinates back to lab ones, ``x = W u``.

    Returns a length-4 vector for a ``4d`` form and a length-6 one for ``6d`` -- the
    inverse of :func:`to_normalized` on the space that was actually normalised.
    """
    u = np.asarray(normalized, dtype=float)
    if u.shape != (form.dim,):
        raise ValueError(f"expected a length-{form.dim} normalised vector, got {u.shape}")
    return form.w @ u


def actions(
    form: NormalForm | NormalFormPoint, state: Sequence[float] | np.ndarray
) -> tuple[float, ...]:
    r"""The mode actions ``J_i = (u_i^2 + p_i^2)/2`` of a state.

    These are the invariants of the linear motion: a turn rotates each mode's normalised
    point about the origin, so ``J`` is unchanged to round-off however many turns are
    tracked. For an uncoupled lattice ``2 J_x`` is exactly the Courant-Snyder invariant
    ``(x^2 + (alpha x + beta px)^2)/beta``, and the average of ``J`` over a bunch is its
    emittance.
    """
    u = to_normalized(form, state)
    return tuple(float((u[2 * i] ** 2 + u[2 * i + 1] ** 2) / 2.0) for i in range(form.dim // 2))


# --------------------------------------------------------------------------------------
# The normal form along the ring (O2)
# --------------------------------------------------------------------------------------


def _ripken_betas(w: np.ndarray) -> np.ndarray:
    r"""``B[plane, mode]``: how much of each mode is carried in each plane's position.

    The Mais-Ripken generalisation of ``beta``. ``B[0, 0]`` is the ordinary ``beta_x``
    (mode 1 in ``x``) and ``B[1, 1]`` the ordinary ``beta_y``; the **off-diagonal**
    entries ``B[0, 1]`` and ``B[1, 0]`` -- xtrack's ``betx2`` and ``bety1`` -- are the
    cross-plane ones, exactly zero without coupling and the reason this matrix exists.
    """
    n = w.shape[0] // 2
    return np.array(
        [[w[2 * p, 2 * m] ** 2 + w[2 * p, 2 * m + 1] ** 2 for m in range(n)] for p in range(n)]
    )


def _ripken_alphas(w: np.ndarray) -> np.ndarray:
    """``A[plane, mode]``, the Mais-Ripken ``alpha`` (``alfx1``, ``alfx2``, ...)."""
    n = w.shape[0] // 2
    return np.array(
        [
            [
                -w[2 * p, 2 * m] * w[2 * p + 1, 2 * m]
                - w[2 * p, 2 * m + 1] * w[2 * p + 1, 2 * m + 1]
                for m in range(n)
            ]
            for p in range(n)
        ]
    )


def _ripken_gammas(w: np.ndarray) -> np.ndarray:
    """``G[plane, mode]``, the Mais-Ripken ``gamma`` -- the same read off the momentum row."""
    n = w.shape[0] // 2
    return np.array(
        [
            [w[2 * p + 1, 2 * m] ** 2 + w[2 * p + 1, 2 * m + 1] ** 2 for m in range(n)]
            for p in range(n)
        ]
    )


def _dispersion_from_w(w: np.ndarray) -> np.ndarray:
    """``(D_x, D_px, D_y, D_py)``: the transverse response *in phase with* ``delta``."""
    den = w[DELTA, 5] - w[DELTA, 4] * w[ZETA, 5] / w[ZETA, 4]
    return np.array([(w[i, 5] - w[i, 4] * w[ZETA, 5] / w[ZETA, 4]) / den for i in (X, PX, Y, PY)])


def _crab_dispersion_from_w(w: np.ndarray) -> np.ndarray:
    """``(dx_zeta, dpx_zeta, dy_zeta, dpy_zeta)``: the response in phase with ``zeta``.

    The same construction as :func:`_dispersion_from_w` with the roles of the two
    longitudinal coordinates exchanged -- ``delta`` projected out instead of ``zeta``.
    """
    den = w[ZETA, 4] - w[ZETA, 5] * w[DELTA, 4] / w[DELTA, 5]
    return np.array([(w[i, 4] - w[i, 5] * w[DELTA, 4] / w[DELTA, 5]) / den for i in (X, PX, Y, PY)])


@dataclass(frozen=True)
class NormalFormPoint:
    r"""The normal form at one point around the ring: ``W(s)``, and the phase to get there.

    :func:`propagate_normal_form` returns one of these per element boundary. ``W(s)`` is
    ``M(0 -> s) W(0)`` put back into :class:`NormalForm`'s phase convention, so it
    normalises the one-turn map **starting at s** -- a different matrix from the one at
    the entrance, conjugate to the same rotation ``R``.

    **Almost nothing here can see the re-phasing that produced it.** :attr:`betas`,
    :attr:`alphas`, :attr:`gammas`, :attr:`dispersion` and :attr:`crab_dispersion` are all
    invariant under ``W -> W diag(Rot, Rot, Rot)``: in each product the phase cancels
    between the two factors, and the dispersions are ratios taken inside a single
    eigenvector. Only two things are not blind -- the convention itself
    (``w[2p, 2p+1] = 0``) and :attr:`mu`. That is why the analytic gates for this
    milestone are so heavily weighted toward ``mu``.
    """

    s: float
    """Path length from the start of the lattice [m]."""
    w: np.ndarray
    """``W(s)``, in :class:`NormalForm`'s phase convention."""
    w_inv: np.ndarray
    """``W(s)^-1``: lab coordinates at ``s`` to normalised ones."""
    mu: tuple[float, ...]
    """Accumulated phase advance per mode [rad], continuous and starting at ``0``.

    Over one turn this is ``2 pi Q`` with ``Q`` the **full** integer-plus-fractional tune,
    not the fractional part the one-turn matrix gives. It is obtained by unwrapping the
    re-phasing angle, which is safe exactly while no single element advances the phase by
    more than ``pi``.
    """
    method: str
    """``"4d"`` or ``"6d"``, inherited from the :class:`NormalForm` this came from."""

    @property
    def dim(self) -> int:
        """``4`` or ``6``."""
        return int(self.w.shape[0])

    @property
    def betas(self) -> np.ndarray:
        """Mais-Ripken ``B[plane, mode]``; see :func:`_ripken_betas`."""
        return _ripken_betas(self.w)

    @property
    def alphas(self) -> np.ndarray:
        """Mais-Ripken ``A[plane, mode]``."""
        return _ripken_alphas(self.w)

    @property
    def gammas(self) -> np.ndarray:
        """Mais-Ripken ``G[plane, mode]``."""
        return _ripken_gammas(self.w)

    @property
    def mode_beta(self) -> tuple[float, ...]:
        """Each mode's beta in its own plane -- the diagonal of :attr:`betas`."""
        return tuple(float(v) for v in np.diag(self.betas))

    @property
    def mode_alpha(self) -> tuple[float, ...]:
        """Each mode's alpha in its own plane -- the diagonal of :attr:`alphas`."""
        return tuple(float(v) for v in np.diag(self.alphas))

    @property
    def dispersion(self) -> np.ndarray:
        """The **dynamic** dispersion at ``s``; see :attr:`NormalForm.dispersion`."""
        if self.method != "6d":
            raise NormalFormError("dispersion needs the 6D normal form (method='6d')")
        return _dispersion_from_w(self.w)

    @property
    def crab_dispersion(self) -> np.ndarray:
        r"""``(dx_zeta, dpx_zeta, dy_zeta, dpy_zeta)`` -- the orbit's dependence on arrival time.

        Where :attr:`dispersion` is the transverse excursion in phase with ``delta``, this
        is the part in phase with ``zeta``: the head and the tail of a bunch sitting at
        different ``x``. A ring with a crab cavity has it by construction, but an
        **ordinary** ring has it too, and small rather than zero for an interesting
        reason: the transverse response to a momentum oscillating at ``Q_s`` is driven
        off-resonance, so it *lags* the drive. The lag is first order in ``Q_s`` and the
        longitudinal mode's momentum content is another, so ``dx_zeta`` itself is
        **second** order -- and exactly zero on a ring with no bends, where the transverse
        rows never see ``delta`` at all.

        ``6d`` only.
        """
        if self.method != "6d":
            raise NormalFormError("crab dispersion needs the 6D normal form (method='6d')")
        return _crab_dispersion_from_w(self.w)


def closed_normal_form(lattice: Lattice, *, method: str = "6d") -> NormalForm:
    """The matched normal form at the entrance of a periodic ``lattice``.

    :func:`normal_form` of the one-turn matrix -- the normal-form counterpart of
    :func:`closed_twiss` and :func:`coupled_twiss`.
    """
    return normal_form(lattice.one_turn_matrix(), method=method)


def _rephase(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put ``a`` back into the phase convention; return ``(W, phi)``.

    Right-multiplying by ``diag(Rot(theta_p))`` commutes with ``R``, so it leaves
    ``a R a^-1`` alone and keeps ``a`` symplectic. ``theta_p = atan2(-b, a)`` with
    ``(a, b) = (A[2p, 2p], A[2p, 2p+1])`` is the unique choice that zeroes the second
    entry and leaves the first at ``+sqrt(a^2 + b^2)``. The angle removed, ``phi =
    -theta``, is the mode's phase advance from the start of the lattice, modulo ``2 pi``.
    """
    dim = a.shape[0]
    w = a.copy()
    phi = np.zeros(dim // 2)
    for p in range(dim // 2):
        c, s = float(a[2 * p, 2 * p]), float(a[2 * p, 2 * p + 1])
        r = math.hypot(c, s)
        if r <= _DEGENERATE_NORM * float(np.abs(a[:, 2 * p : 2 * p + 2]).max() + 1e-300):
            raise NormalFormError(
                f"mode {p + 1} carries no position in its own plane at this point "
                f"(|W[{2 * p}, {2 * p}:{2 * p + 2}]| = {r:.3e}): the phase convention has "
                "nothing to rotate onto the real axis."
            )
        phi[p] = math.atan2(s, c)
        cos_t, sin_t = c / r, -s / r
        block = a[:, 2 * p : 2 * p + 2]
        w[:, 2 * p] = block[:, 0] * cos_t - block[:, 1] * sin_t
        w[:, 2 * p + 1] = block[:, 0] * sin_t + block[:, 1] * cos_t
    return w, phi


def propagate_normal_form(
    lattice: Lattice, form0: NormalForm, *, maps: Sequence[np.ndarray] | None = None
) -> list[NormalFormPoint]:
    r"""The normal form at every element boundary, starting from ``form0``.

    Returns ``len(lattice) + 1`` points: the entrance, then the exit of each element in
    order -- the same shape and alignment as :func:`propagate_twiss`. The rule is

        W(s) = M(0 -> s) W(0) . D(s),

    with ``D(s)`` the per-mode rotation that puts the result back into
    :class:`NormalForm`'s phase convention. ``D`` commutes with ``R``, so every point
    normalises its own local one-turn map ``M(0->s) M M(0->s)^-1`` to the *same* rotation:
    the tunes belong to the ring, not to the point. The angle ``D`` removes, accumulated
    continuously, is :attr:`NormalFormPoint.mu`.

    ``form0`` decides the dimension: pass ``closed_normal_form(lattice, method="4d")`` for
    the transverse-only form (the right one for a ring with no cavity) or ``"6d"`` for the
    full one. The 4D transport is the transverse block of the running 6x6 transfer
    matrix, which for accsim's element set is also the product of the per-element blocks
    -- a property of the element set rather than of the algebra, and asserted as such in
    the analytic tests.

    ``maps`` substitutes the transport exactly as in :func:`propagate_twiss`: one 6x6 per
    element, in beam order, used in place of ``elem.matrix()``. ``form0`` must then have
    been built from the matching one-turn product; nothing here re-derives it.

    ``form0`` is used verbatim as the first point and is assumed to be in the convention
    already -- which everything :func:`normal_form` returns is. ``mu`` therefore starts at
    exactly zero, the way :func:`propagate_twiss` starts from ``twiss0.mu_x``.

    **Almost nothing this returns can see the re-phasing** -- see
    :class:`NormalFormPoint`. ``mu`` and the convention are the only witnesses, and no
    renormalisation of the eigenvectors is done along the way, so ``W(s)`` staying
    symplectic is a measurement rather than an assumption. (xtrack renormalises, which it
    needs because it also propagates through radiation maps, where the symplectic norm
    genuinely decays.)
    """
    if maps is not None and len(maps) != len(lattice.elements):
        raise ValueError(
            f"maps must have one matrix per element: got {len(maps)} for "
            f"{len(lattice.elements)} elements"
        )
    n_modes = form0.dim // 2
    transfer = np.eye(DIM)
    mu = np.zeros(n_modes)
    previous = np.zeros(n_modes)
    points = [NormalFormPoint(0.0, form0.w.copy(), form0.w_inv.copy(), tuple(mu), form0.method)]
    s = 0.0
    for i, elem in enumerate(lattice.elements):
        transfer = (elem.matrix(lattice.ref) if maps is None else maps[i]) @ transfer
        s += elem.length
        t = _transverse_4d(transfer) if form0.dim == 4 else transfer
        w, phi = _rephase(t @ form0.w)
        mu = mu + (phi - previous + math.pi) % (2.0 * math.pi) - math.pi
        previous = phi
        points.append(NormalFormPoint(s, w, np.linalg.inv(w), tuple(mu), form0.method))
    return points
