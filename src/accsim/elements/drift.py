"""Drift: a field-free straight section."""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


class Drift(Element):
    r"""A field-free drift of length ``L``.

    Linear transfer matrix (derived symbolically in ``docs/CONVENTIONS.md`` and
    pinned by ``tests/analytic/test_drift.py``):

    - transverse:    ``x -> x + L*px``,  ``y -> y + L*py``  (R12 = R34 = L)
    - longitudinal:  ``zeta -> zeta + (L / gamma0^2) * delta``           (R56 = L/gamma0^2)

    The longitudinal coupling ``R56 = L/gamma0^2`` is the time-of-flight effect:
    a higher-momentum particle (``delta > 0``) is faster and arrives earlier, so
    ``zeta`` increases. It uses ``delta`` (momentum deviation), giving the
    ``1/gamma0^2`` coefficient; the energy-deviation convention would instead give
    ``L/(beta0^2 gamma0^2)``. As ``gamma0 -> inf`` the coupling vanishes — at
    ultrarelativistic energy all particles travel at ~c regardless of ``delta``.
    """

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        M[X, PX] = L
        M[Y, PY] = L
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M
