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

Feed-down: a cubic kick reaches two orders below itself
-------------------------------------------------------
About an orbit offset ``(x_co, y_co)`` the cubic kick expands into **six** terms
(J3, derived in ``tests/analytic/test_octupole_feeddown.py`` and applied by
:func:`accsim.orbit.linearised_lattice`):

    dipole       theta_x = -1/6 k3l x_co (x_co^2 - 3 y_co^2)
                 theta_y = +1/6 k3l y_co (3 x_co^2 - y_co^2)
    normal quad  k1l_eff  = +1/2 k3l (x_co^2 - y_co^2)
    skew quad    k1sl_eff = +k3l x_co y_co
    normal sext  k2l_eff  = +k3l x_co
    skew sext    k2sl_eff = +k3l y_co
    octupole     unchanged

Where the sextupole's quadratic kick reached one order down (a gradient), this
reaches two — which is why the three effects appear at three *different* powers
of the orbit: the chromaticity moves as ``x_co`` (through ``k2l_eff`` at
dispersion), the tunes as ``x_co^2`` (through ``k1l_eff``), and the closed orbit
itself as ``x_co^3``. The skew pair vanishes on a flat orbit, and ``x = px = 0``
is an *exact* invariant subspace as well as ``y = py = 0``, so a purely vertical
bump does **not** steer the beam horizontally through an octupole — where through
a sextupole it does.

On the **design** orbit none of this exists, and that non-response is the
milestone's own reference point: expanding ``x = x_beta + D_x delta`` about
``x_co = 0`` makes an octupole a *second*-order chromatic element (its ``delta``
term is a sextupole, not a gradient), so :func:`accsim.twiss.chromaticity` is right
at first order there and ``Q''`` is the blind spot. Steering the machine is what
turns that into a first-order effect: expanding about ``x_co != 0`` instead gives a
``delta``-linear gradient ``k3l x_co D_x``, which is the ``Q'`` of the real machine
rather than a piece of ``Q''`` arriving late. ``Q''`` itself is uncomputed on
**both** orbits — the ``1/2 k3l D_x^2`` gradient, and the ``delta``-dependence of
``x_co`` — and remains out of scope.

A **thick** octupole is still refused by :func:`accsim.orbit.linearised_lattice`,
for the thick sextupole's reason: its offset varies across the body, so one
entrance-orbit split would carry an ``O(L^2)`` error.
:func:`accsim.orbit.linearised_element_maps` handles it, differentiating
``track()`` rather than walking element types.
"""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .drift import Drift
from .element import Element


def _drift_matrix(length: float, ref: ReferenceParticle) -> np.ndarray:
    """The linear drift map of ``length`` — the octupole's whole linear content.

    This is the *matrix* path only. Since P2 (ii) the tracking path drifts through
    :meth:`Drift._track_body` instead, which is the exact map this matrix is the
    origin Jacobian of; the two agree at first order and nowhere else.
    """
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

    **The gaps are the exact drift (P2 (ii)), not the linear one** — the same change
    :class:`~accsim.elements.sextupole.Sextupole` took, for the same reason. They were
    ``_drift_matrix`` until then, so a thick octupole carried a cruder drift than a bare
    :class:`~accsim.elements.drift.Drift` of the same length has since L1. The split now
    calls :meth:`Drift._track_body`, adding ``-L px delta``, ``-L py delta`` and
    ``-L (px^2 + py^2)/2`` — all bilinear or quadratic, hence beyond any 6x6. The last is
    the statement that a *kicked* particle takes a longer path: ``zeta`` responds to
    ``k3`` now, where before it was identically blind to it.

    Check symplecticity with
    :func:`~accsim.symplectic.is_symplectic_map_canonical`; plain
    :func:`~accsim.symplectic.is_symplectic_map` *rejects* this correct map, the
    ``(zeta, delta)`` caveat recorded for the drift itself.

    At ``k3 = 0`` the composition collapses to the drift identically, for any
    ``n_slices`` — the exact one, bit for bit, which is the ``k3 -> 0`` limit of the
    loop and not a separate model.
    """

    def __init__(
        self,
        length: float,
        k3: float,
        name: str | None = None,
        n_slices: int = 1,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.k3 = float(k3)
        self.n_slices = int(n_slices)

    @property
    def k3l(self) -> float:
        """Integrated strength ``k3l = k3 * L`` [m^-3]."""
        return self.k3 * self.length

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear map of an octupole is a drift: no focusing, no dispersion.
        return _drift_matrix(self.length, ref)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # Drift-kick-drift on the *exact* drift, n_slices times. The gaps delegate to
        # ``Drift._track_body`` rather than restating the geometry: that map's
        # longitudinal term is written in a deliberately rationalised form to avoid a
        # cancellation that would otherwise show up in finite-difference Jacobians, and
        # a second copy of it here is exactly how the two would drift apart.
        if self.k3 == 0.0:
            # The k3 -> 0 limit, taken in one step so that a zero-strength body is the
            # exact drift to the last bit rather than to the rounding of n composed
            # slices.
            out = Drift(self.length)._track_body(state, ref)
        else:
            n = self.n_slices
            half = Drift(0.5 * self.length / n)
            k3l_slice = self.k3l / n
            out = np.array(state, dtype=float, copy=True)
            for _ in range(n):
                out = _apply_kick(half._track_body(out, ref), k3l_slice)
                out = half._track_body(out, ref)
        # The split carries no constant part -- an octupole's ``kick()`` is the inherited
        # zero -- but a subclass may add one, and a ``_track_body`` override that drops
        # ``_kick_body`` breaks I1's affine contract. Added here, as the base class does.
        k = self._kick_body(ref)
        return out + (k if out.ndim == 1 else k[:, None])

    def __repr__(self) -> str:
        slices = f", n_slices={self.n_slices}" if self.n_slices != 1 else ""
        return f"Octupole(length={self.length}, k3={self.k3}{slices}{self._repr_tail()})"


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

    def __init__(
        self,
        k3l: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy, roll=roll)
        self.k3l = float(k3l)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        # Zero linear part: a thin octupole is the identity map at the origin.
        return np.eye(DIM)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # The full nonlinear kick -- exact, not a linearisation. Only (px, py) move.
        return _apply_kick(np.array(state, dtype=float, copy=True), self.k3l)

    def __repr__(self) -> str:
        return f"ThinOctupole(k3l={self.k3l}{self._repr_tail()})"
