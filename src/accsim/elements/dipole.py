"""Dipole: a sector bending magnet (pure sector, no edge angles, no gradient)."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


class Dipole(Element):
    r"""A sector dipole of arc length ``L`` and bend angle ``theta`` [rad].

    Pure sector bend: the reference orbit curves with radius ``rho = L/theta``
    (curvature ``h = 1/rho = theta/L``); no quadrupole gradient and no pole-face
    (edge) angles. Bending is horizontal (the ``x`` plane).

    The linear 6x6 map is ``exp(L*A)`` of the sector-bend Hamiltonian generator
    (pinned symbolically and cross-checked entrywise against xtrack). Non-trivial
    entries, with ``C = cos theta``, ``S = sin theta``:

    - **Horizontal** (weak geometric focusing): ``R11 = R22 = C``,
      ``R12 = S/h = rho*S``, ``R21 = -h*S``.
    - **Dispersion** (coupling to ``delta``): ``R16 = (1-C)/h = rho*(1-C)``,
      ``R26 = S``. A higher-momentum particle bends less, so it is displaced
      outward (``R16 > 0``).
    - **Vertical**: a plain drift (``R34 = L``) — a pure sector bend has no
      vertical focusing.
    - **Longitudinal** (path-length / time-of-flight): ``R51 = -S``,
      ``R52 = (C-1)/h = -rho*(1-C) = -R16``, and
      ``R56 = rho*S - L + L/gamma0^2``. The ``R51``/``R52`` terms are exactly the
      symplectic partners of the dispersion (``R51 = R21*R16 - R11*R26``); ``R56``
      is the drift slip ``L/gamma0^2`` minus the extra arc the design orbit
      travels, ``rho*(theta - S)``.

    As ``theta -> 0`` every curvature term vanishes and the map reduces exactly to
    a :class:`Drift` of length ``L`` (``R56 -> L/gamma0^2``).
    """

    def __init__(self, length: float, angle: float, name: str | None = None) -> None:
        super().__init__(length, name=name)
        if length == 0.0 and angle != 0.0:
            raise ValueError("a finite bend angle requires a positive length")
        self.angle = float(angle)

    @property
    def curvature(self) -> float:
        """Curvature ``h = 1/rho = theta/L`` [m^-1] (0 for a straight dipole)."""
        return self.angle / self.length if self.length > 0.0 else 0.0

    @property
    def rho(self) -> float:
        """Bending radius ``rho = L/theta`` [m] (``inf`` for a straight dipole)."""
        return self.length / self.angle if self.angle != 0.0 else math.inf

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        theta = self.angle
        M = np.eye(DIM)

        # Straight limit: a zero-angle "bend" is just a drift.
        if theta == 0.0:
            M[X, PX] = L
            M[Y, PY] = L
            M[ZETA, DELTA] = L / ref.gamma0**2
            return M

        h = theta / L  # = 1/rho
        c, s = math.cos(theta), math.sin(theta)

        # Horizontal plane + dispersion.
        M[X, X] = c
        M[X, PX] = s / h
        M[X, DELTA] = (1.0 - c) / h
        M[PX, X] = -h * s
        M[PX, PX] = c
        M[PX, DELTA] = s
        # Vertical plane: drift.
        M[Y, PY] = L
        # Longitudinal: path-length coupling (symplectic partners of dispersion)
        # plus the drift-like slip reduced by the extra design-orbit arc length.
        M[ZETA, X] = -s
        M[ZETA, PX] = (c - 1.0) / h
        M[ZETA, DELTA] = s / h - L + L / ref.gamma0**2
        return M

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"Dipole(length={self.length}, angle={self.angle}{name})"
