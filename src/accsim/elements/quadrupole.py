"""Quadrupole: a linear focusing element (thick and thin lens forms)."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


def _focusing_block(g: float, L: float) -> np.ndarray:
    r"""2x2 transfer block for the 1D Hill equation ``u'' + g*u = 0`` over length ``L``.

    With ``u' = p`` (paraxial), the block acts on ``(u, p)``:

    - ``g > 0`` (focusing):    ``[[cos wL,   sin wL / w], [-w sin wL,  cos wL]]``
    - ``g < 0`` (defocusing):  ``[[cosh wL,  sinh wL / w], [ w sinh wL, cosh wL]]``
    - ``g = 0`` (drift):       ``[[1, L], [0, 1]]``

    where ``w = sqrt(|g|)``. The three cases join smoothly (``sin wL / w -> L`` as
    ``w -> 0``); writing them as one analytic family is what makes a single
    ``Quadrupole`` handle both planes and the ``k1 -> 0`` drift limit.
    """
    if g > 0.0:
        w = math.sqrt(g)
        c, s = math.cos(w * L), math.sin(w * L)
        return np.array([[c, s / w], [-w * s, c]])
    if g < 0.0:
        w = math.sqrt(-g)
        ch, sh = math.cosh(w * L), math.sinh(w * L)
        return np.array([[ch, sh / w], [w * sh, ch]])
    return np.array([[1.0, L], [0.0, 1.0]])


def _focusing_functions(g: np.ndarray | float, L: float) -> tuple[np.ndarray, np.ndarray]:
    r"""``(C, S)`` of :func:`_focusing_block`, **vectorised over a per-particle** ``g``.

    ``C = cos(wL)`` and ``S = sin(wL)/w`` (``w = sqrt(g)``), continued to
    ``cosh``/``sinh`` for ``g < 0`` and to ``(1, L)`` at ``g = 0`` — the same single
    analytic family :func:`_focusing_block` builds its 2x2 from, so
    ``_focusing_block(g, L) == [[C, S], [-g S, C]]``, asserted in the analytic suite.

    It exists because the *exact* map's focusing strength is ``k1 / (1 + delta)``,
    which is a **per-particle number**: a ``(6, n)`` bunch with a momentum spread has
    ``n`` different ``g`` values, and :func:`_focusing_block`'s ``math.cos`` and its
    ``if g > 0`` branch can serve only one. ``matrix()`` still uses the scalar form —
    it is evaluated at ``delta = 0`` and nowhere else.

    Both branches are evaluated and then selected, which costs a few flops and buys
    freedom from warnings; ``u = 0`` is special-cased so ``sin(u)/u`` is never formed.
    """
    g_arr = np.asarray(g, dtype=float)
    u = np.sqrt(np.abs(g_arr)) * L
    u_safe = np.where(u == 0.0, 1.0, u)
    trig = g_arr >= 0.0
    C = np.where(trig, np.cos(u), np.cosh(u))
    ratio = np.where(trig, np.sin(u_safe) / u_safe, np.sinh(u_safe) / u_safe)
    S = L * np.where(u == 0.0, 1.0, ratio)
    return C, S


def _path_lengthening(
    g: np.ndarray, u0: np.ndarray, up0: np.ndarray, L: float, C: np.ndarray, S: np.ndarray
) -> np.ndarray:
    r"""One plane's contribution to ``I = \int (u'^2 / 2) ds`` through a thick quad.

    The extra path a particle travels because it is not parallel to the axis. With
    ``u(s) = u0 C + up0 S`` the angle is ``u'(s) = A S + B C``, ``A = -g u0``,
    ``B = up0``, and the three elementary integrals over ``[0, L]`` are

        g * int S^2 = T,    int S C = S^2 / 2,    int C^2 = L - T,    T = (L - C S)/2

    (the first and third sum to ``L`` because ``C^2 + g S^2 = 1`` identically), giving

        I_u = (1/2) [ g u0^2 T  -  g u0 up0 S^2  +  up0^2 (L - T) ].

    Derived in ``tests/analytic/test_exact_quadrupole.py`` with sympy, not recalled.

    **Written without a single division by** ``g``. The textbook form (and MAD-X's,
    and xtrack's ``track_thick_cfd``) carries ``1/g`` in every term and needs a
    ``g == 0`` branch; substituting ``A`` and ``B`` back cancels all of them, leaving
    an expression that is *entire* in ``g``. So the weak-quadrupole limit is
    continuous rather than special-cased, and ``g -> 0`` collapses to ``up0^2 L / 2``
    — the drift's own path lengthening — with no branch to get wrong.
    """
    T = 0.5 * (L - C * S)
    return 0.5 * (g * u0 * u0 * T - g * u0 * up0 * S * S + up0 * up0 * (L - T))


def thick_quadrupole_map(
    state: np.ndarray, length: float, k1: float, ref: ReferenceParticle
) -> np.ndarray:
    r"""The thick normal quadrupole's momentum-dependent map (L2), as a free function.

    Shared by :class:`Quadrupole` and — through a 45 degree roll —
    :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole`, which is the same magnet
    turned. See :class:`Quadrupole` for the derivation and what the map is exact in.

    ``state`` is a ``(6,)`` vector or a ``(6, n)`` bunch; every quantity below is
    per-particle, which is the whole reason this is not a matrix multiply.
    """
    st = np.asarray(state, dtype=float)
    L = length
    if L == 0.0:
        return st.copy()

    delta = st[DELTA]
    one_plus = 1.0 + delta
    # The gradient a particle of momentum (1 + delta) actually feels. This single
    # division is the milestone: k1 is normalised to the *reference* rigidity, so an
    # off-momentum particle is focused less, and accsim's linear matrix never knew.
    Kx = k1 / one_plus
    Cx, Sx = _focusing_functions(Kx, L)
    Cy, Sy = _focusing_functions(-Kx, L)

    xp = st[PX] / one_plus  # the geometric angle dx/ds, paraxially
    yp = st[PY] / one_plus

    out = st.copy()
    out[X] = st[X] * Cx + xp * Sx
    out[PX] = (-Kx * st[X] * Sx + xp * Cx) * one_plus
    out[Y] = st[Y] * Cy + yp * Sy
    out[PY] = (+Kx * st[Y] * Sy + yp * Cy) * one_plus  # K_y = -K_x

    # zeta: the momentum term and the path-lengthening term, kept apart so that
    # neither is formed by subtracting two numbers of size L (see the class docstring).
    E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
    slip = L * delta * (2.0 + delta) / ref.gamma0**2 / (one_plus * (one_plus + E_over_E0))
    path = _path_lengthening(Kx, st[X], xp, L, Cx, Sx) + _path_lengthening(
        -Kx, st[Y], yp, L, Cy, Sy
    )
    out[ZETA] = st[ZETA] + slip - path * E_over_E0 / one_plus
    return out


class Quadrupole(Element):
    r"""A thick quadrupole of length ``L`` and normalised gradient ``k1`` [m^-2].

    Convention (MAD-X / Xsuite; recorded in ``docs/CONVENTIONS.md``):
    ``k1 = (1/B rho)(dB_y/dx)``. The linearised equations of motion are

        x'' + k1 x = 0,     y'' - k1 y = 0,

    so **``k1 > 0`` focuses in x and defocuses in y**. The transverse blocks are
    the closed-form solutions of these (cos/sin in the focusing plane, cosh/sinh
    in the defocusing plane); ``k1 = 0`` reduces exactly to a :class:`Drift`'s
    **matrix** — but not, quite, to its ``track``; see the warning below.

    Longitudinal: the reference orbit is straight, so the time-of-flight slip over
    length ``L`` is the same as a drift, ``R56 = L/gamma0^2`` (cross-checked
    against xtrack). A pure quadrupole has no dispersion (no curvature), so the
    transverse and longitudinal motion stay uncoupled at this linear order.

    The full 6x6 is symplectic by construction: it is ``exp(L*A)`` for the
    Hamiltonian generator ``A`` (pinned symbolically in the analytic tests).

    The momentum-dependent map (L2)
    -------------------------------
    Like the :class:`~accsim.elements.drift.Drift`, this element has **two maps**:
    :meth:`_matrix_body` above, which every optics function is built on, and
    :meth:`_track_body`, which is what a tracked particle actually follows. The
    first is the Jacobian of the second at the origin, and only there.

    ``k1`` is normalised to the **reference** rigidity, so a particle of momentum
    ``(1 + delta)`` is focused by ``k1 / (1 + delta)`` — it is stiffer and bends
    less in the same field. That single factor is the quadrupole's share of natural
    chromaticity, and until L2 accsim's tracked quadrupole did not have it: its
    ``track`` was its ``matrix`` at *every* momentum, which is the definition of a
    chromatically ideal magnet and is not what a magnet is. With
    ``K = k1/(1+delta)``, ``C = cos(sqrt(K) L)``, ``S = sin(sqrt(K) L)/sqrt(K)``
    (continued to ``cosh``/``sinh`` for ``K < 0``) and the geometric angles
    ``x' = px/(1+delta)``, ``y' = py/(1+delta)``:

        x    -> x C + x' S                    px -> (-K x S + x' C) (1 + delta)
        y    -> y Ch + y' Sh                  py -> (+K y Sh + y' Ch) (1 + delta)
        zeta -> zeta + L (1 - 1/rvv) - I / rvv,        rvv = beta / beta0

    where ``I`` is the path lengthening of :func:`_path_lengthening`, ``Ch``/``Sh``
    are the same functions at ``-K``, and the longitudinal line says exactly what it
    looks like: the particle's own speed against the extra distance its angle makes
    it travel. Setting ``delta = 0`` gives back :meth:`_matrix_body` in the
    transverse block, entry for entry — a statement asserted, not assumed.

    **What it is exact in, and what it is not.** This is the flow of

        H = p_zeta - (1 + delta) + (px^2 + py^2) / (2 (1 + delta))
            + (k1/2) (x^2 - y^2),

    the **paraxial** reduction of the exact Hamiltonian
    ``H = p_zeta - sqrt((1+delta)^2 - px^2 - py^2) + (k1/2)(x^2 - y^2)``, whose flow
    has no closed form at all — the square root and the quadratic potential do not
    commute, and every code either expands the root (MAD-X, and xtrack's default
    ``mat-kick-mat``, which this reproduces) or splits the Hamiltonian and
    integrates numerically. accsim takes the first, because a closed form is the
    only way :meth:`_matrix_body` can stay the exact Jacobian at the origin, which
    is what bounds this milestone's blast radius (see below).

    So the map is **exact in** ``delta`` — to all orders, which is what the
    chromaticity gates measure — and **paraxial in the angles**, dropping
    ``O(angle^3)`` relative. It is nonetheless *exactly* symplectic, being the exact
    flow of the approximate Hamiltonian rather than an approximate flow of the exact
    one, and that is the property worth having: a truncated-but-symplectic map is
    safe to iterate for a million turns, a more accurate non-symplectic one is not.
    Verified with :func:`~accsim.symplectic.is_symplectic_map_canonical`; note that
    plain :func:`~accsim.symplectic.is_symplectic_map` **rejects** it, exactly as it
    rejects the exact drift, because ``(zeta, delta)`` is not a canonical pair.

    ⚠️ **A zero-strength thick quadrupole is a drift in** ``matrix`` **but not,
    quite, in** ``track``. At ``k1 = 0`` this map is the *expanded* drift
    ``x += L px/(1+delta)``, where :class:`~accsim.elements.drift.Drift` is the
    *exact* ``x += L px/pz``. The two differ by ``O(angle^3)`` — relatively
    ``(px^2+py^2)/2``, the same gap that separates xtrack's two drift models. L2
    narrows that inconsistency from *first* order (the old linear map differed at
    ``O(px delta)``) but does not close it, and the ROADMAP's prediction that it
    would was wrong. Short-circuiting ``k1 == 0`` to the exact drift would close it
    only by making the map **discontinuous in** ``k1``, so it is asserted and
    documented instead — the residual is ``O(angle^3)`` *and independent of* ``k1``,
    which is what identifies it as the paraxial expansion rather than a bug.

    **On the design orbit nothing changes.** At ``delta = 0`` the transverse map is
    the linear matrix *identically*, not merely to first order, so a zero-momentum
    particle tracks bit-for-bit as it did before L2 and the only coordinate that
    moves at all is ``zeta`` (a quadrupole now lengthens an off-axis particle's
    path, as it must). Everything computed from :meth:`matrix` is untouched.

    **The** ``zeta`` **trap L1 warned about, and how it is avoided.** MAD-X and
    xtrack both evaluate ``dzeta = L - path/rvv``: two numbers of size ``L``,
    differenced to get a small one. Splitting it as ``L(1 - 1/rvv) - I/rvv`` and
    rationalising the first term through ``(1+delta) + E/E0`` — using
    ``(E/E0)^2 = 1 + beta0^2 delta (2 + delta)``, the drift's own identity — removes
    the cancellation, leaving two small quantities that are *added*. The first term
    is the drift's ``zeta`` map at zero angle, which is a free cross-check on the
    algebra.
    """

    def __init__(
        self,
        length: float,
        k1: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        self.k1 = float(k1)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        xb = _focusing_block(self.k1, L)  # x'' + k1 x = 0
        yb = _focusing_block(-self.k1, L)  # y'' - k1 y = 0
        M[np.ix_([X, PX], [X, PX])] = xb
        M[np.ix_([Y, PY], [Y, PY])] = yb
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """The momentum-dependent map — see the class docstring for the derivation.

        Vectorised over a trailing particle axis, so a ``(6,)`` state and a ``(6, n)``
        bunch with a momentum spread take the same path. A zero-length quadrupole is
        the identity, matching :meth:`_matrix_body`.
        """
        return thick_quadrupole_map(state, self.length, self.k1, ref)

    def normalized_field(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        r"""``(b_x, b_y) = (k1 y, k1 x)`` — zero on axis, linear off it.

        Normalised to ``(B rho)_0``. A quadrupole radiates only off axis, and
        ``|b|^2 = k1^2 (x^2 + y^2)`` depends on the radius alone — which is why the
        same magnet rolled (a :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole`)
        must radiate identically.
        """
        return self.k1 * np.asarray(y, dtype=float), self.k1 * np.asarray(x, dtype=float)

    def __repr__(self) -> str:
        return f"Quadrupole(length={self.length}, k1={self.k1}{self._repr_tail()})"


class ThinQuadrupole(Element):
    r"""A thin-lens quadrupole: a zero-length focusing kick of integrated strength ``k1l``.

    ``k1l = k1 * L`` [m^-1] is the integrated gradient, equal to the inverse focal
    length ``1/f``. The map is a pure momentum kick (no length, no longitudinal
    slip):

        px -> px - k1l * x      (focusing in x for k1l > 0)
        py -> py + k1l * y      (defocusing in y for k1l > 0)

    This is the ``L -> 0`` limit of :class:`Quadrupole` at fixed ``k1l`` and is
    symplectic (each plane's kick has unit determinant). It is the building block
    for the thin-lens FODO closed form used in the Stage 1 acceptance test.

    **This map is already exact, and there is nothing for L2 to do to it** — a fact
    worth stating, because it looks like an omission. ``px`` is the momentum
    normalised to ``P0``, and the kick a fixed field gives it,
    ``Delta px = -(q/P0) (dB_y/dx) x L = -k1l x``, carries **no** ``1/(1+delta)``:
    every particle's *momentum* changes by the same amount, and it is the *angle*
    ``px/pz`` that changes less for a stiffer particle. So a thin quadrupole is
    chromatically exact on its own, and the chromaticity of a thin-lens ring lives
    entirely in the exact drifts between the quads — which is why a drift +
    thin-quad ring already had 100% of its natural chromaticity after L1, while a
    drift + *thick*-quad ring had only 45%. The thick magnet is the one that had to
    carry ``k1/(1+delta)`` itself, because it drifts and focuses at the same time.

    **Displaced** (``dx``, ``dy``; K1), it is a quadrupole plus a
    :class:`~accsim.elements.corrector.Corrector` and *nothing else* — the one
    misalignment in the package with no higher terms at all, because a quadrupole's
    gradient is uniform. From ``px -> px - k1l (x - dx)``,

        theta_x = +k1l dx,      theta_y = -k1l dy,

    the same displacement sign giving opposite kick signs in the two planes (the
    ``py -> py + k1l y`` asymmetry). Both cross-derivatives of that kick vanish
    identically, so **no displacement of an unrolled quadrupole couples the
    planes** — only a roll can (K2).
    """

    def __init__(
        self,
        k1l: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy, roll=roll)
        self.k1l = float(k1l)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        M = np.eye(DIM)
        M[PX, X] = -self.k1l  # focusing in x for k1l > 0
        M[PY, Y] = self.k1l  # defocusing in y for k1l > 0
        return M

    @property
    def focal_length(self) -> float:
        """Focal length ``f = 1 / k1l`` [m] (positive ⇒ focusing in x)."""
        return 1.0 / self.k1l

    def __repr__(self) -> str:
        return f"ThinQuadrupole(k1l={self.k1l}{self._repr_tail()})"
