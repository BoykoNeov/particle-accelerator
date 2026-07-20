"""Dipole: a sector bending magnet, optionally with pole-face (edge) angles."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


def _edge_matrix(h: float, e: float) -> np.ndarray:
    r"""Thin hard-edge pole-face focusing kick for edge angle ``e`` [rad].

    A pole face rotated by ``e`` (``e = 0`` is the sector face; ``e = theta/2``
    the symmetric rectangular face) acts as a thin quadrupole-like kick at the
    entrance/exit of the body. In the **hard-edge** limit (zero fringe extent,
    ``FINT = 0``) the linear map is the identity except:

        px -> px + h*tan(e) * x      (R21 = +h tan e)
        py -> py - h*tan(e) * y      (R43 = -h tan e)

    So a positive edge angle **defocuses horizontally and focuses vertically** --
    the sign is fixed by the geometry of the rotated face (the field the particle
    sees lengthens on the outside of the bend), not remembered. Each 2x2 block
    ``[[1, 0], [+-h tan e, 1]]`` has unit determinant, so the kick is symplectic.

    The fringe-field correction (``e -> e - psi`` in the *vertical* plane only,
    ``psi = h*g*fint*(1 + sin^2 e)/cos e``) is deliberately **not** applied here:
    this is the hard-edge map, the apples-to-apples match to MAD-X ``sbend`` with
    its default ``FINT = HGAP = 0``. Fringe is a separate, opt-in refinement.
    """
    E = np.eye(DIM)
    if h == 0.0 or e == 0.0:
        return E  # no bending or no rotation -> no edge focusing
    t = h * math.tan(e)
    E[PX, X] = t  # horizontal defocus for e > 0
    E[PY, Y] = -t  # vertical focus for e > 0
    return E


class Dipole(Element):
    r"""A sector dipole of arc length ``L`` and bend angle ``theta`` [rad].

    The reference orbit curves with radius ``rho = L/theta`` (curvature
    ``h = 1/rho = theta/L``); bending is horizontal (the ``x`` plane). Optional
    pole-face rotations ``e1`` (entrance) and ``e2`` (exit) add hard-edge
    focusing; with the default ``e1 = e2 = 0`` this is a **pure sector bend** and
    the map is byte-identical to the original sector dipole.

    The body's linear 6x6 map is ``exp(L*A)`` of the sector-bend Hamiltonian
    generator (pinned symbolically and cross-checked entrywise against xtrack).
    Non-trivial body entries, with ``C = cos theta``, ``S = sin theta``:

    - **Horizontal** (weak geometric focusing): ``R11 = R22 = C``,
      ``R12 = S/h = rho*S``, ``R21 = -h*S``.
    - **Dispersion** (coupling to ``delta``): ``R16 = (1-C)/h = rho*(1-C)``,
      ``R26 = S``. A higher-momentum particle bends less, so it is displaced
      outward (``R16 > 0``).
    - **Vertical**: a plain drift (``R34 = L``) -- a pure sector bend has no
      vertical focusing.
    - **Longitudinal** (path-length / time-of-flight): ``R51 = -S``,
      ``R52 = (C-1)/h = -rho*(1-C) = -R16``, and
      ``R56 = rho*S - L + L/gamma0^2``. The ``R51``/``R52`` terms are exactly the
      symplectic partners of the dispersion (``R51 = R21*R16 - R11*R26``); ``R56``
      is the drift slip ``L/gamma0^2`` minus the extra arc the design orbit
      travels, ``rho*(theta - S)``.

    **Edge angles.** The full map is ``Edge(e2) @ Body @ Edge(e1)`` -- the
    entrance edge acts first. Each edge is the hard-edge kick of
    :func:`_edge_matrix`. Two consequences worth naming, both exact in this linear
    hard-edge model:

    - **Rectangular bend** (``e1 = e2 = theta/2``): the two edges *exactly* cancel
      the body's horizontal weak focusing, leaving the horizontal block equal to a
      drift ``[[1, rho*sin theta], [0, 1]]`` (``R21 = 0`` to machine precision),
      while the vertical plane gets all its focusing from the edges
      (``R43 ~ -2 h tan(theta/2)``).
    - Edges are optics-active (they change beta, tune, chromaticity and dispersion
      through composition) but add no length and no direct longitudinal coupling.

    As ``theta -> 0`` every curvature term vanishes (and the edges too, since
    ``h -> 0``) and the map reduces exactly to a :class:`Drift` of length ``L``
    (``R56 -> L/gamma0^2``).
    """

    def __init__(
        self,
        length: float,
        angle: float,
        e1: float = 0.0,
        e2: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(length, name=name)
        if length == 0.0 and angle != 0.0:
            raise ValueError("a finite bend angle requires a positive length")
        self.angle = float(angle)
        self.e1 = float(e1)
        self.e2 = float(e2)

    @property
    def curvature(self) -> float:
        """Curvature ``h = 1/rho = theta/L`` [m^-1] (0 for a straight dipole)."""
        return self.angle / self.length if self.length > 0.0 else 0.0

    @property
    def rho(self) -> float:
        """Bending radius ``rho = L/theta`` [m] (``inf`` for a straight dipole)."""
        return self.length / self.angle if self.angle != 0.0 else math.inf

    def _body_matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """The bare sector-bend body (no edges)."""
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

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        body = self._body_matrix(ref)
        if self.e1 == 0.0 and self.e2 == 0.0:
            return body  # pure sector: byte-identical to the original map
        h = self.curvature
        # Entrance edge acts first: M = Edge(e2) @ Body @ Edge(e1).
        return _edge_matrix(h, self.e2) @ body @ _edge_matrix(h, self.e1)

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        edges = f", e1={self.e1}, e2={self.e2}" if (self.e1 or self.e2) else ""
        return f"Dipole(length={self.length}, angle={self.angle}{edges}{name})"
