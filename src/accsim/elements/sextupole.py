"""Sextupole: a nonlinear element whose *linear* map is a drift (thick and thin).

A normal sextupole applies the momentum kick

    Delta px = -1/2 k2l (x^2 - y^2),     Delta py = +k2l (x y),

with integrated strength ``k2l = k2 * L`` [m^-2] and
``k2 = (1/B rho)(d^2 B_y/dx^2)`` [m^-3] (MAD-X / Xsuite convention). This is
purely nonlinear: its Jacobian at the closed orbit ``(x, y) = 0`` is the
identity, so the **linear** 6x6 transfer matrix is just a drift (a thin
sextupole is the identity). Sextupoles therefore leave ``beta``, dispersion, and
the tunes of the linear lattice unchanged.

The physics that Stage 2 cares about ("chromaticity correction, linear effect")
enters through *feed-down*: at a point of nonzero dispersion ``x = x_beta +
D_x delta``, the quadratic kick expands to a ``delta``-dependent linear gradient
``k1_eff = k2 D_x delta``, which shifts the chromaticity. That first-order term
is computed in :func:`accsim.twiss.chromaticity`; it needs only ``k2``/``k2l``
and the matched dispersion, not a nonlinear tracking map. The full nonlinear kick
(amplitude-dependent tune, dynamic aperture) is explicitly out of Stage 2 scope
(see ``docs/ROADMAP.md``), so no nonlinear map is implemented here.
"""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


class Sextupole(Element):
    r"""A thick sextupole of length ``L`` and normalised strength ``k2`` [m^-3].

    The linear transfer matrix is identical to a :class:`Drift` of length ``L``
    (including the longitudinal slip ``R56 = L/gamma0^2``): a sextupole has no
    linear focusing and no curvature, so it neither bends the reference orbit nor
    couples the transverse planes at first order. ``k2`` is retained for the
    feed-down chromaticity computed in :func:`accsim.twiss.chromaticity`.

    Convention (MAD-X / Xsuite): ``k2 = (1/B rho)(d^2 B_y/dx^2)`` [m^-3]; the
    integrated strength is ``k2l = k2 * L``. The (non-linear) kick it represents is
    ``Delta px = -1/2 k2 L (x^2 - y^2)``, ``Delta py = +k2 L (x y)``.
    """

    def __init__(self, length: float, k2: float, name: str | None = None) -> None:
        super().__init__(length, name=name)
        self.k2 = float(k2)

    @property
    def k2l(self) -> float:
        """Integrated strength ``k2l = k2 * L`` [m^-2]."""
        return self.k2 * self.length

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear map of a sextupole is a drift: no focusing, no dispersion.
        L = self.length
        M = np.eye(DIM)
        M[X, PX] = L  # R12 (x, px)
        M[Y, PY] = L  # R34 (y, py)
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"Sextupole(length={self.length}, k2={self.k2}{name})"


class ThinSextupole(Element):
    r"""A thin-lens sextupole: a zero-length nonlinear kick of integrated strength ``k2l``.

    ``k2l = k2 * L`` [m^-2] is the integrated strength. The represented kick is
    ``Delta px = -1/2 k2l (x^2 - y^2)``, ``Delta py = +k2l (x y)``. Its **linear**
    map is the identity (a thin nonlinear kick has zero linear part), so it does
    not change ``beta``, dispersion, or the tunes; only the feed-down chromaticity
    at nonzero dispersion depends on ``k2l``.
    """

    def __init__(self, k2l: float, name: str | None = None) -> None:
        super().__init__(0.0, name=name)
        self.k2l = float(k2l)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Zero linear part: a thin sextupole is the identity map.
        return np.eye(DIM)

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"ThinSextupole(k2l={self.k2l}{name})"
