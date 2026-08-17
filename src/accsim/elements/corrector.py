"""Dipole corrector: the steering magnet that moves the closed orbit."""

from __future__ import annotations

import numpy as np

from ..coords import DIM, PX, PY
from ..reference import ReferenceParticle
from .element import Element


class Corrector(Element):
    r"""A thin dipole corrector (steerer): a constant angular kick.

        px -> px + kick_x,     py -> py + kick_y.

    ``kick_x`` / ``kick_y`` are deflection **angles** [rad] — an integrated field
    ``B*L`` divided by the magnetic rigidity ``B*rho``, so a corrector is specified
    by the angle it imparts to an on-momentum particle, exactly like a
    :class:`~accsim.elements.dipole.Dipole` is specified by its bend angle.

    **This is the one element whose action is not a matrix.** A kick of the same
    angle for every particle, independent of ``(x, px, y, py)``, is *inhomogeneous*
    — it cannot appear anywhere in a 6x6 acting on the state vector. It therefore
    lives in :meth:`kick`, while :meth:`matrix` is the identity. That identity is
    the physics, not a placeholder:

    - a constant kick **moves the closed orbit** but does not change the map
      *about* it, so ``beta``, the tunes, chromaticity and dispersion are untouched;
    - correspondingly the corrector is the only handle in the package that steers
      the orbit without touching the optics, which is exactly why real machines
      correct the orbit with dipoles and the optics with quadrupoles.

    Zero length: a steerer is short compared with a betatron wavelength, so it is
    modelled thin (the ``L -> 0`` limit at fixed ``B*L``). A finite-length steerer is
    ``Drift(L/2) + Corrector + Drift(L/2)`` to the same order.

    **Sign.** ``kick_x > 0`` increases ``px``. Related to a
    :class:`~accsim.elements.dipole.Dipole`, a positive ``kick_x`` is a bend-angle
    *deficit*: a weak-field dipole under-bends, leaving the particle with the
    positive ``px`` it would have kept had the magnet not been there. The
    convention is pinned empirically against xtrack in the reference suite rather
    than asserted from that argument — see ``docs/CONVENTIONS.md`` ->
    *Closed-orbit correction*.

    Longitudinal: none. The path-length change from a kicked trajectory is second
    order in the kick angle (``Delta L ~ L*theta^2/2``), below the linear order this
    element works at, so ``zeta`` and ``delta`` pass through untouched.

    **Displacing a corrector does nothing at all** (K1): its ``matrix`` is the
    identity, so the misalignment term ``(I - matrix) d`` of
    :meth:`~accsim.elements.element.Element.kick` is *exactly* zero, and its kick
    does not depend on where the particle is. A constant kick has no centre to
    miss — the same translation invariance a :class:`~accsim.elements.drift.Drift`
    has, and asserted at exact zero rather than to tolerance.
    """

    def __init__(
        self,
        kick_x: float = 0.0,
        kick_y: float = 0.0,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy, roll=roll)
        self.kick_x = float(kick_x)
        self.kick_y = float(kick_y)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        """The identity: a constant kick has no linear part (see the class docstring)."""
        return np.eye(DIM)

    def _kick_body(self, ref: ReferenceParticle) -> np.ndarray:
        k = np.zeros(DIM)
        k[PX] = self.kick_x
        k[PY] = self.kick_y
        return k

    def __repr__(self) -> str:
        return f"Corrector(kick_x={self.kick_x}, kick_y={self.kick_y}{self._repr_tail()})"
