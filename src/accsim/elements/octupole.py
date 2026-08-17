"""Octupole: the first element whose *only* linear-optics effect is amplitude detuning.

A normal octupole applies the momentum kick

    Delta px = -1/6 k3l (x^3 - 3 x y^2),     Delta py = +1/6 k3l (3 x^2 y - y^3),

with integrated strength ``k3l = k3 * L`` [m^-3] and
``k3 = (1/B rho)(d^3 B_y/dx^3)`` [m^-4] (MAD-X / Xsuite convention). Like the
sextupole this is purely nonlinear — its Jacobian at ``(x, y) = 0`` is the
identity, so the **linear** 6x6 map is a drift (a thin octupole is the identity)
and ``beta``, dispersion and the tunes of the linear lattice are untouched.

Where the coefficients come from
--------------------------------
The same field expansion that fixes the quadrupole's ``k1`` and the sextupole's
``1/2``:

    B_y + i B_x = (B rho) * sum_n k_n (x + i y)^n / n!,

with the thin-lens kick ``Delta px = -(q/p0) int B_y ds``,
``Delta py = +(q/p0) int B_x ds``. The ``n = 3`` term is
``(B rho) k3 (x + i y)^3 / 6``, and ``(x + i y)^3 = (x^3 - 3 x y^2) + i (3 x^2 y
- y^3)`` gives exactly the two lines above — the ``1/6`` is ``1/3!``. That chain
is redone symbolically, from ``n = 1`` (the xtrack- and MAD-X-validated
``Quadrupole``) upward, in ``tests/analytic/test_octupole_kick.py``.

What an octupole is *for*: amplitude-dependent detuning
-------------------------------------------------------
Unlike the sextupole, whose detuning is second order in ``k2`` and carries no
closed form this package claims, the octupole shifts the tune at **first order**
in ``k3l``, by an amount proportional to the particle's action. Averaging the
kick's potential over the betatron phases gives the exact first-order
anharmonicity

    dQx/dJx = + k3l beta_x^2 / (16 pi),    dQy/dJy = + k3l beta_y^2 / (16 pi),
    dQx/dJy = dQy/dJx = - k3l beta_x beta_y / (8 pi),

derived in sympy and computed by :func:`accsim.twiss.amplitude_detuning`. The
off-diagonal term is ``-2x`` the diagonal at ``beta_x = beta_y`` and the matrix
is symmetric because it descends from a Hamiltonian — neither is imposed.

Feed-down, and what is deliberately not modelled
------------------------------------------------
About an orbit offset the cubic kick expands into a sextupole-like, a
quadrupole-like, a skew and a dipole term, exactly as the sextupole's quadratic
kick expands in :func:`accsim.orbit.linearised_element_maps`. That expansion is
**not** implemented: octupole feed-down is out of scope, and the linearising
helpers *raise* on an octupole rather than silently treating it as a drift.
At nonzero dispersion the same expansion in ``x = x_beta + D_x delta`` makes an
octupole a **second**-order chromatic element (its ``delta`` term is a sextupole,
not a gradient), so :func:`accsim.twiss.chromaticity` is correct at first order
and ``Q''`` is the blind spot; both are asserted in the suite.
"""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


def _drift_matrix(length: float, ref: ReferenceParticle) -> np.ndarray:
    """The linear drift map of ``length`` — the octupole's whole linear content."""
    M = np.eye(DIM)
    M[X, PX] = length  # R12 (x, px)
    M[Y, PY] = length  # R34 (y, py)
    M[ZETA, DELTA] = length / ref.gamma0**2  # R56, momentum-variable form
    return M


def _apply_kick(state: np.ndarray, k3l: float) -> np.ndarray:
    r"""Apply the thin octupole kick of integrated strength ``k3l``, mutating ``state``.

    Writes into the array it is handed and returns it — callers pass a fresh copy
    (or a temporary) so nothing owned by the caller is clobbered.

    ``Delta px = -1/6 k3l (x^3 - 3 x y^2)``, ``Delta py = +1/6 k3l (3 x^2 y - y^3)``.
    No ``1/(1 + delta)`` scaling: ``px`` is normalised to the *reference* momentum
    ``p0`` and the kick is an integrated field over that same ``p0``, exactly as for
    :class:`~accsim.elements.sextupole.ThinSextupole`. Works on a ``(6,)`` state or a
    ``(6, n)`` bunch alike.
    """
    x, y = state[X], state[Y]
    state[PX] -= k3l * (x * x * x - 3.0 * x * y * y) / 6.0
    state[PY] += k3l * (3.0 * x * x * y - y * y * y) / 6.0
    return state


class Octupole(Element):
    r"""A thick octupole of length ``L`` and normalised strength ``k3`` [m^-4].

    The linear transfer matrix is identical to a :class:`Drift` of length ``L``
    (including the longitudinal slip ``R56 = L/gamma0^2``): an octupole has no
    linear focusing and no curvature. ``k3`` drives the amplitude-dependent tune
    shift computed in :func:`accsim.twiss.amplitude_detuning` and the nonlinear
    tracking map below, and nothing else in the linear optics.

    Convention (MAD-X / Xsuite): ``k3 = (1/B rho)(d^3 B_y/dx^3)`` [m^-4]; the
    integrated strength is ``k3l = k3 * L``. The kick is
    ``Delta px = -1/6 k3 L (x^3 - 3 x y^2)``, ``Delta py = +1/6 k3 L (3 x^2 y - y^3)``.

    **Tracking is a drift-kick-drift integrator, and that is an approximation** —
    the same one :class:`~accsim.elements.sextupole.Sextupole` makes, with the same
    consequences. :meth:`track` splits the body into ``n_slices`` slices, each
    ``drift(L/2n) . kick(k3l/n) . drift(L/2n)``. Every factor is symplectic so the
    composition is symplectic *exactly*, but the real magnet kicks and drifts
    simultaneously: the Baker-Campbell-Hausdorff remainder is ``O(L^3)`` per slice
    at fixed ``k3``, hence ``O(1/n_slices^2)`` overall — a second-order integrator.
    Both scalings are measured in ``tests/analytic/test_octupole_kick.py``.

    At fixed *integrated* strength ``k3l`` the leading remainder is linear in ``L``,
    so shortening the body at fixed ``k3l`` closes the gap only at first order: a
    genuinely thin magnet is :class:`ThinOctupole`, not a short thick one.

    At ``k3 = 0`` the composition collapses to the linear drift map identically, for
    any ``n_slices``.
    """

    def __init__(
        self, length: float, k3: float, name: str | None = None, n_slices: int = 1
    ) -> None:
        super().__init__(length, name=name)
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.k3 = float(k3)
        self.n_slices = int(n_slices)

    @property
    def k3l(self) -> float:
        """Integrated strength ``k3l = k3 * L`` [m^-3]."""
        return self.k3 * self.length

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear map of an octupole is a drift: no focusing, no dispersion.
        return _drift_matrix(self.length, ref)

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # Drift-kick-drift, n_slices times. The split carries no constant part -- an
        # octupole's ``kick()`` is the inherited zero -- so at k3 = 0 the base class's
        # affine map is the right answer *and* keeps this override honest about the
        # (M, k) contract rather than quietly dropping k.
        if self.k3 == 0.0:
            return super().track(state, ref)
        n = self.n_slices
        half = _drift_matrix(0.5 * self.length / n, ref)
        k3l_slice = self.k3l / n
        out = np.array(state, dtype=float, copy=True)
        for _ in range(n):
            out = _apply_kick(half @ out, k3l_slice)
            out = half @ out
        return out

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        slices = f", n_slices={self.n_slices}" if self.n_slices != 1 else ""
        return f"Octupole(length={self.length}, k3={self.k3}{slices}{name})"


class ThinOctupole(Element):
    r"""A thin-lens octupole: a zero-length nonlinear kick of integrated strength ``k3l``.

    ``k3l = k3 * L`` [m^-3]. The kick is
    ``Delta px = -1/6 k3l (x^3 - 3 x y^2)``, ``Delta py = +1/6 k3l (3 x^2 y - y^3)``,
    applied exactly by :meth:`track`. Its **linear** map is the identity (a thin
    nonlinear kick has zero linear part *at the origin*), so beta, dispersion and the
    linear tunes do not depend on ``k3l`` at all — the tune an actual particle
    measures does, through its amplitude.

    **Exactly symplectic.** The kick is minus the gradient of

        V = k3l (x^4 - 6 x^2 y^2 + y^4) / 24,

    so its Jacobian satisfies ``M^T J M = J`` at every amplitude identically. That is
    a structural property of *any* gradient kick and therefore says nothing about
    whether the ``1/6`` is right — see the module docstring for what does.
    """

    def __init__(self, k3l: float, name: str | None = None) -> None:
        super().__init__(0.0, name=name)
        self.k3l = float(k3l)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Zero linear part: a thin octupole is the identity map at the origin.
        return np.eye(DIM)

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # The full nonlinear kick -- exact, not a linearisation. Only (px, py) move.
        return _apply_kick(np.array(state, dtype=float, copy=True), self.k3l)

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"ThinOctupole(k3l={self.k3l}{name})"
