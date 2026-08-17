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

from .coords import DELTA, PX, PY, ZETA, X, Y
from .lattice import Lattice

_TRANSVERSE = [X, PX, Y, PY]  # the 4D transverse subspace (x, px, y, py)


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
    """
    from .elements.skew_quadrupole import SkewQuadrupole, ThinSkewQuadrupole

    ref = lattice.ref
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
    term of perturbation theory, not the first); no closed form for it is claimed
    anywhere in this package, and it is **not** included here. A ring carrying both
    will therefore detune by more than this function reports — measurably so once
    ``k2`` is large. Nor does this include the octupole's own second-order
    contribution, so it is the tangent at zero amplitude, not the tune at large
    amplitude.
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
    orbit is on axis or nothing in the lattice is nonlinear; different by the
    feed-down beta-beat wherever an off-axis sextupole is live.

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

    **Why this is not computed by tracking.** accsim's linear element maps carry no
    ``delta`` dependence — ``track()`` through a quadrupole is its ``matrix()`` at
    every momentum — so linearising the tracked map about the off-momentum orbit
    measures the sextupole feed-down term and is exactly blind to the natural
    chromaticity, which accsim supplies analytically (F2). Implementing this by
    tracking alone would silently drop that entire term. Instead the existing,
    validated integrals are run on :func:`~accsim.orbit.linearised_lattice`, and
    the tracked route is kept as the independent gate on the half it can see.

    Raises :class:`CoupledLatticeError` on a vertically steered machine (the
    equivalent lattice then carries a skew quadrupole, and the 2x2 Courant-Snyder
    integrals are not valid there), and :class:`NotImplementedError` for a thick
    sextupole — see :func:`~accsim.orbit.linearised_lattice` for both.
    """
    from .orbit import linearised_lattice

    return chromaticity(linearised_lattice(lattice), slices)
