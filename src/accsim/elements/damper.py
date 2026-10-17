"""Damper: a lumped transverse damping, applied once a turn (W6)."""

from __future__ import annotations

import numpy as np

from ..coords import DIM, PX, PY
from ..reference import ReferenceParticle
from .element import Element

_PLANES = {"x": PX, "y": PY}


class Damper(Element):
    r"""A thin element that takes a fixed fraction of one plane's momentum every turn.

    On each pass of a turn loop it does

        p_u -> (1 - rate) p_u           (u = x or y, ``plane``),

    and nothing else. It is the lumped form of two real things with the same map:

    - **a ring's radiation damping, net of the RF.** A photon carries momentum along the
      particle's direction of motion, so a turn's radiation scales ``(P_x, P_y, P_s)``
      down together; the cavity gives back only the longitudinal part. What is left is
      ``px, py -> (1 - U0/E0) px, py`` once a turn (Sands). accsim's own radiation lives
      inside bends (:mod:`accsim.radiation_kick`), which W4's driven-response model does
      not admit, so this element is how a straight ring is given damping at all;
    - **an ideal transverse feedback kicker.** A kicker can only change the momentum,
      which is why the map does not touch ``x``.

    It is xtrack's ``LineSegmentMap(damping_rate_px=rate)`` convention
    (``damping_factors[1, 1] = 1 - damping_rate_px``, applied as ``px *= factor``).

    Over one turn the linear ring's determinant is ``1 - rate`` exactly, so its two
    eigenvalues have ``|lambda|^2 = 1 - rate``: the amplitude falls by ``sqrt(1 - rate)``
    a turn (an amplitude decrement ``Gamma = -ln(1 - rate) / 2``). Because it acts on
    ``p`` alone it also moves the tune where ``alpha != 0`` at the damper:

        cos mu_d = (2 cos mu - rate (cos mu - alpha sin mu)) / (2 sqrt(1 - rate)),

    ``-rate alpha / (4 pi)`` in tune to first order.

    **Like** :class:`~accsim.elements.acdipole.ACDipole`, **the turn loops apply it.** Its
    static maps — :meth:`matrix`, :meth:`kick`, :meth:`track` — are the identity, so every
    optics function, closed orbit, normal form and Taylor map sees the undamped machine
    (which is how a real ring's optics are quoted), and none of them can quietly return a
    number built from a one-turn map whose determinant is not 1. The turn loops of
    :class:`~accsim.tracking.Tracker` call :meth:`damp`; a single pass with ``turn=None``
    does not. Elements act in list order, so a damper placed after an AC dipole damps
    that turn's kick too.

    It damps toward the **design** orbit (``p -> 0``), not the closed orbit: on a ring
    whose closed orbit is not zero at the damper the tracked orbit settles off it, by
    ``O(rate)``. That is stated, not modelled.
    """

    def __init__(self, rate: float, plane: str = "x", name: str | None = None) -> None:
        super().__init__(0.0, name=name)
        if plane not in _PLANES:
            raise ValueError(f"plane must be 'x' or 'y', got {plane!r}")
        rate = float(rate)
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {rate!r}")
        self.rate = rate
        self.plane = plane

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        """The identity: the static map is the undamped machine's (see the class docstring)."""
        return np.eye(DIM)

    def damping_matrix(self) -> np.ndarray:
        """The ``(6, 6)`` linear map a turn loop applies: the identity with ``1 - rate``
        on the damped momentum's diagonal."""
        out = np.eye(DIM)
        out[_PLANES[self.plane], _PLANES[self.plane]] = 1.0 - self.rate
        return out

    def damp(self, state: np.ndarray) -> np.ndarray:
        """``state`` (``(6,)`` or ``(6, n)``) with the damped momentum times ``1 - rate``;
        a new array."""
        out = np.array(state, dtype=float, copy=True)
        out[_PLANES[self.plane]] *= 1.0 - self.rate
        return out

    def __repr__(self) -> str:
        return f"Damper(rate={self.rate}, plane={self.plane!r}{self._repr_tail()})"
