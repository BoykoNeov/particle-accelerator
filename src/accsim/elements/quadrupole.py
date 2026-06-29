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


class Quadrupole(Element):
    r"""A thick quadrupole of length ``L`` and normalised gradient ``k1`` [m^-2].

    Convention (MAD-X / Xsuite; recorded in ``docs/CONVENTIONS.md``):
    ``k1 = (1/B rho)(dB_y/dx)``. The linearised equations of motion are

        x'' + k1 x = 0,     y'' - k1 y = 0,

    so **``k1 > 0`` focuses in x and defocuses in y**. The transverse blocks are
    the closed-form solutions of these (cos/sin in the focusing plane, cosh/sinh
    in the defocusing plane); ``k1 = 0`` reduces exactly to a :class:`Drift`.

    Longitudinal: the reference orbit is straight, so the time-of-flight slip over
    length ``L`` is the same as a drift, ``R56 = L/gamma0^2`` (cross-checked
    against xtrack). A pure quadrupole has no dispersion (no curvature), so the
    transverse and longitudinal motion stay uncoupled at this linear order.

    The full 6x6 is symplectic by construction: it is ``exp(L*A)`` for the
    Hamiltonian generator ``A`` (pinned symbolically in the analytic tests).
    """

    def __init__(self, length: float, k1: float, name: str | None = None) -> None:
        super().__init__(length, name=name)
        self.k1 = float(k1)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        xb = _focusing_block(self.k1, L)  # x'' + k1 x = 0
        yb = _focusing_block(-self.k1, L)  # y'' - k1 y = 0
        M[np.ix_([X, PX], [X, PX])] = xb
        M[np.ix_([Y, PY], [Y, PY])] = yb
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"Quadrupole(length={self.length}, k1={self.k1}{name})"


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
    """

    def __init__(self, k1l: float, name: str | None = None) -> None:
        super().__init__(0.0, name=name)
        self.k1l = float(k1l)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        M = np.eye(DIM)
        M[PX, X] = -self.k1l  # focusing in x for k1l > 0
        M[PY, Y] = self.k1l  # defocusing in y for k1l > 0
        return M

    @property
    def focal_length(self) -> float:
        """Focal length ``f = 1 / k1l`` [m] (positive ⇒ focusing in x)."""
        return 1.0 / self.k1l

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"ThinQuadrupole(k1l={self.k1l}{name})"
