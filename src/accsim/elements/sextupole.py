"""Sextupole: a nonlinear element whose *linear* map is a drift (thick and thin).

A normal sextupole applies the momentum kick

    Delta px = -1/2 k2l (x^2 - y^2),     Delta py = +k2l (x y),

with integrated strength ``k2l = k2 * L`` [m^-2] and
``k2 = (1/B rho)(d^2 B_y/dx^2)`` [m^-3] (MAD-X / Xsuite convention). This is
purely nonlinear: its Jacobian at the closed orbit ``(x, y) = 0`` is the
identity, so the **linear** 6x6 transfer matrix is just a drift (a thin
sextupole is the identity). Sextupoles therefore leave ``beta``, dispersion, and
the tunes of the linear lattice unchanged.

Where the coefficients come from
--------------------------------
Not from a potential chosen to reproduce them — from the field. The MAD-X /
Xsuite multipole expansion of a normal magnet is

    B_y + i B_x = (B rho) * sum_n k_n (x + i y)^n / n!,

whose ``n = 1`` term is the quadrupole this package already validates
(``B_y = B rho k1 x``, giving ``Delta px = -k1 L x`` and ``x'' + k1 x = 0``). Its
``n = 2`` term is ``(B rho) k2 (x + i y)^2 / 2``, and the thin-lens kick of a
transverse field is ``Delta px = -(q/p0) int B_y ds``, ``Delta py = +(q/p0) int
B_x ds``. Expanding ``(x + i y)^2 = (x^2 - y^2) + 2 i x y`` gives exactly the two
lines above — the ``1/2`` is the ``1/n!`` of the same expansion that fixes the
quadrupole's ``k1``. That derivation is redone symbolically, from ``n = 1``
upward, in ``tests/analytic/test_sextupole_kick.py``.

Feed-down: the linear shadow of the nonlinear kick
--------------------------------------------------
The physics that Stage 2 cares about ("chromaticity correction, linear effect")
enters through *feed-down*: at a point of nonzero dispersion ``x = x_beta +
D_x delta``, the quadratic kick expands to a ``delta``-dependent linear gradient
``k1_eff = k2 D_x delta``, which shifts the chromaticity. That first-order term
is computed in :func:`accsim.twiss.chromaticity`; it needs only ``k2``/``k2l``
and the matched dispersion, not a nonlinear tracking map.

Feed-down is also the **independent gate on the ``1/2``**. Linearising the
nonlinear :meth:`ThinSextupole.track` about an offset ``x0`` recovers a thin
quadrupole of integrated gradient ``k1l_eff = k2l x0``: any other coefficient in
the kick rescales that gradient by the same factor, and the resulting
chromaticity no longer matches the feed-down term that ``tests/reference/
test_sextupole_xtrack.py`` pins against xtrack's real tracking. Symplecticity
does *not* gate the coefficient — every gradient kick is symplectic whatever its
strength.

The gradient is only one of the terms. Expanding the kick about a full orbit
offset ``(x_co, y_co)`` also produces a **dipole** ``-1/2 k2l (x_co^2 - y_co^2)``
and a **skew** quadrupole ``k1sl_eff = k2l y_co``, so an off-axis sextupole steers
the beam and couples the planes as well as focusing it. Because the dipole term
depends on the orbit it displaces, the closed orbit of a machine with a live
sextupole is a *fixed point* rather than a linear solve — see
:func:`accsim.orbit.closed_orbit_nonlinear` and ``docs/CONVENTIONS.md`` ->
*Sextupole feed-down on a distorted orbit*.

The nonlinear map
-----------------
:meth:`ThinSextupole.track` applies the kick above exactly. It is a *gradient*
kick — ``(Delta px, Delta py) = -grad V`` for ``V = k2l (x^3/6 - x y^2 / 2)`` —
hence exactly symplectic at every amplitude, not merely to tolerance.
:meth:`Sextupole.track` (thick) composes drift-kick-drift, which is symplectic
exactly but only **second-order accurate in the length** (see the class
docstring). Amplitude-dependent tune, resonance driving and dynamic aperture are
all *out of scope* here: the map exists and is gated, but the package makes no
dynamic-aperture claim (``docs/ROADMAP.md`` -> *Out of scope*).
"""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


def _drift_matrix(length: float, ref: ReferenceParticle) -> np.ndarray:
    """The linear drift map of ``length`` — the sextupole's whole linear content."""
    M = np.eye(DIM)
    M[X, PX] = length  # R12 (x, px)
    M[Y, PY] = length  # R34 (y, py)
    M[ZETA, DELTA] = length / ref.gamma0**2  # R56, momentum-variable form
    return M


def _apply_skew_kick(state: np.ndarray, k2sl: float) -> np.ndarray:
    r"""Apply the thin **skew** sextupole kick ``k2sl``, mutating ``state``.

    ``Delta px = +k2sl (x y)``, ``Delta py = +1/2 k2sl (x^2 - y^2)`` — the normal
    kick's two components swapped and re-signed, which is what multiplying the
    strength by ``i`` in the field expansion does. Same ``p0`` normalisation and same
    ``(6,)``-or-``(6, n)`` handling as :func:`_apply_kick`.
    """
    x, y = state[X], state[Y]
    state[PX] += k2sl * (x * y)
    state[PY] += 0.5 * k2sl * (x * x - y * y)
    return state


def _apply_kick(state: np.ndarray, k2l: float) -> np.ndarray:
    r"""Apply the thin sextupole kick of integrated strength ``k2l``, mutating ``state``.

    Writes into the array it is handed and returns it — callers pass a fresh copy
    (or a temporary) so nothing owned by the caller is clobbered.


    ``Delta px = -1/2 k2l (x^2 - y^2)``, ``Delta py = +k2l (x y)``. No ``1/(1 +
    delta)`` scaling: ``px`` is normalised to the *reference* momentum ``p0``, and
    the kick is an integrated field over that same ``p0`` — the momentum deviation
    of the particle does not enter, exactly as for
    :class:`~accsim.elements.corrector.Corrector`. Works on a ``(6,)`` state or a
    ``(6, n)`` bunch alike.
    """
    x, y = state[X], state[Y]
    state[PX] -= 0.5 * k2l * (x * x - y * y)
    state[PY] += k2l * (x * y)
    return state


class Sextupole(Element):
    r"""A thick sextupole of length ``L`` and normalised strength ``k2`` [m^-3].

    The linear transfer matrix is identical to a :class:`Drift` of length ``L``
    (including the longitudinal slip ``R56 = L/gamma0^2``): a sextupole has no
    linear focusing and no curvature, so it neither bends the reference orbit nor
    couples the transverse planes at first order. ``k2`` drives the feed-down
    chromaticity computed in :func:`accsim.twiss.chromaticity` and the nonlinear
    tracking map below.

    Convention (MAD-X / Xsuite): ``k2 = (1/B rho)(d^2 B_y/dx^2)`` [m^-3]; the
    integrated strength is ``k2l = k2 * L``. The kick is
    ``Delta px = -1/2 k2 L (x^2 - y^2)``, ``Delta py = +k2 L (x y)``.

    **Tracking is a drift-kick-drift integrator, and that is an approximation.**
    :meth:`track` splits the body into ``n_slices`` slices, each
    ``drift(L/2n) . kick(k2l/n) . drift(L/2n)``. Each factor is symplectic, so the
    composition is symplectic *exactly* — long-term tracking neither damps nor
    blows up. What it is not is exact: the real magnet kicks and drifts
    *simultaneously* rather than in three separate steps, and the
    Baker-Campbell-Hausdorff remainder of the split is ``O(L^3)`` per slice at
    fixed ``k2`` (terms in both ``k2 L^3`` and ``k2^2 L^3``), hence
    ``O(1/n_slices^2)`` overall — a second-order integrator. Both scalings are
    measured in ``tests/analytic/test_sextupole_kick.py``.

    One caveat on reading that as a thin-lens limit: at fixed *integrated* strength
    ``k2l`` the ``k2^2 L^3`` term is ``k2l^2 L``, so shortening the body at fixed
    ``k2l`` closes the gap only linearly. A genuinely thin magnet is
    :class:`ThinSextupole`, not a short thick one.

    At ``k2 = 0`` the composition collapses to the linear drift map identically, for
    any ``n_slices``.
    """

    def __init__(
        self,
        length: float,
        k2: float,
        name: str | None = None,
        n_slices: int = 1,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy)
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.k2 = float(k2)
        self.n_slices = int(n_slices)

    @property
    def k2l(self) -> float:
        """Integrated strength ``k2l = k2 * L`` [m^-2]."""
        return self.k2 * self.length

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear map of a sextupole is a drift: no focusing, no dispersion.
        return _drift_matrix(self.length, ref)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # Drift-kick-drift, n_slices times (see the class docstring for the order
        # of accuracy this buys and what it costs). The split carries no constant
        # part -- a sextupole's ``kick()`` is the inherited zero -- so at k2 = 0 the
        # base class's affine map is the right answer *and* keeps this override
        # honest about the (M, k) contract rather than quietly dropping k.
        if self.k2 == 0.0:
            return super()._track_body(state, ref)
        n = self.n_slices
        half = _drift_matrix(0.5 * self.length / n, ref)
        k2l_slice = self.k2l / n
        out = np.array(state, dtype=float, copy=True)
        for _ in range(n):
            out = _apply_kick(half @ out, k2l_slice)
            out = half @ out
        return out

    def __repr__(self) -> str:
        slices = f", n_slices={self.n_slices}" if self.n_slices != 1 else ""
        return f"Sextupole(length={self.length}, k2={self.k2}{slices}{self._repr_tail()})"


class ThinSextupole(Element):
    r"""A thin-lens sextupole: a zero-length nonlinear kick of integrated strength ``k2l``.

    ``k2l = k2 * L`` [m^-2] is the integrated strength. The kick is
    ``Delta px = -1/2 k2l (x^2 - y^2)``, ``Delta py = +k2l (x y)``, applied exactly
    by :meth:`track`. Its **linear** map is the identity (a thin nonlinear kick has
    zero linear part *at the origin*), so it does not change ``beta``, dispersion,
    or the tunes; only the feed-down chromaticity at nonzero dispersion depends on
    ``k2l``.

    The identity :meth:`matrix` is therefore not an omission but the statement that
    the map's Jacobian at ``(x, y) = 0`` *is* the identity. Away from the origin the
    Jacobian is not — that difference is precisely feed-down, and is what makes a
    sextupole a chromaticity knob at dispersion and an orbit-dependent gradient
    error on a distorted orbit.

    **Exactly symplectic.** The kick is minus the gradient of
    ``V = k2l (x^3/6 - x y^2/2)``, so its Jacobian satisfies ``M^T J M = J`` at
    every amplitude identically — a structural property of *any* gradient kick, and
    therefore one that says nothing about whether the ``1/2`` is right (see the
    module docstring).
    """

    def __init__(
        self, k2l: float, name: str | None = None, *, dx: float = 0.0, dy: float = 0.0
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy)
        self.k2l = float(k2l)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Zero linear part: a thin sextupole is the identity map at the origin.
        return np.eye(DIM)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # The full nonlinear kick -- exact, not a linearisation. Only (px, py) move.
        return _apply_kick(np.array(state, dtype=float, copy=True), self.k2l)

    def __repr__(self) -> str:
        return f"ThinSextupole(k2l={self.k2l}{self._repr_tail()})"


class ThinSkewSextupole(Element):
    r"""A thin **skew** sextupole: a zero-length kick of integrated strength ``k2sl``.

    ``k2sl = k2s * L`` [m^-2]. The kick is

        Delta px = +k2sl (x y),      Delta py = +1/2 k2sl (x^2 - y^2),

    applied exactly by :meth:`track` — the normal sextupole's two components swapped
    and re-signed. In the field expansion
    ``B_y + i B_x = (B rho) sum_n (k_n + i k_ns) (x + i y)^n / n!`` that is the whole
    difference between the two families: the strength is multiplied by ``i``.
    Geometrically it is a normal :class:`ThinSextupole` **rolled by -30 degrees**
    about the beam axis (``pi / (2 (n + 1))`` for a ``2(n+1)``-pole), the sextupole's
    counterpart of the 45 degree roll that makes
    :class:`~accsim.elements.skew_quadrupole.ThinSkewQuadrupole` out of a normal quad.
    The angle is *solved for* rather than recalled in
    ``tests/analytic/test_skew_sextupole.py``, which also records that it is not
    unique (a sextupole field is unchanged by a third of a turn) and that ``+30``
    degrees gives exactly the opposite kick.

    Like every thin nonlinear kick its **linear** map is the identity, so beta,
    dispersion, the tunes and the linear coupling do not depend on ``k2sl``. It is
    exactly symplectic at any amplitude, being minus the gradient of
    ``V = -k2sl (3 x^2 y - y^3) / 6`` — a structural property of any gradient kick,
    which is why it pins nothing about the coefficient.

    **Where this element comes from, and what reads it.** It exists because J3's
    octupole feed-down produces one: an octupole at a *vertical* orbit offset
    ``y_co`` is a skew sextupole of strength ``k2sl = k3l y_co``
    (see :func:`accsim.orbit.linearised_lattice`), and dropping that term would have
    been the silent omission the octupole branch exists to avoid. Nothing else in the
    package reads it — :func:`accsim.twiss.chromaticity` sums normal sextupoles at
    ``D_x``, so the chromatic effect a skew sextupole actually has (a
    ``delta``-dependent *skew* gradient ``k1sl = k2sl D_x delta``, i.e. chromatic
    coupling) is **not modelled anywhere**, and is asserted as a non-response in the
    analytic suite rather than left to be discovered. There is deliberately no thick
    ``SkewSextupole``: nothing needs one yet, and the thick sextupole is already
    refused by ``linearised_lattice`` for its own ``O(L^2)`` reason.

    Because no accsim quantity responds to it, **its sign cannot be pinned by any
    analytic gate here** and is fixed by probe against xtrack
    (``ThinSkewSextupole(k2sl) == xt.Multipole(ksl=[0, 0, +k2sl])``), the same rule
    J1 and J2 followed for the normal sextupole and the octupole.
    """

    def __init__(
        self, k2sl: float, name: str | None = None, *, dx: float = 0.0, dy: float = 0.0
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy)
        self.k2sl = float(k2sl)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Zero linear part: a thin skew sextupole is the identity map at the origin.
        return np.eye(DIM)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        # The full nonlinear kick -- exact, not a linearisation. Only (px, py) move.
        return _apply_skew_kick(np.array(state, dtype=float, copy=True), self.k2sl)

    def __repr__(self) -> str:
        return f"ThinSkewSextupole(k2sl={self.k2sl}{self._repr_tail()})"
