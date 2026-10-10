"""AC dipole: a thin kick that oscillates from turn to turn (W1)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..coords import DIM, PX, PY
from ..reference import ReferenceParticle
from .element import Element

_PLANES = {"x": PX, "y": PY}


class ACDipole(Element):
    r"""A thin dipole whose kick oscillates at a drive tune near the betatron tune.

    On turn ``n`` (counted from 0, the first pass) it adds

        theta_n = kick * ramp(n) * sin(2 pi (tune * n + tune_rate * n^2 / 2 + lag))

    to ``px`` (``plane="x"``) or ``py`` (``plane="y"``). ``kick`` is the peak deflection
    angle [rad], an integrated field over the rigidity, exactly as a
    :class:`~accsim.elements.corrector.Corrector`'s — and like a corrector's it does not
    scale with ``1/(1 + delta)``: a field kick changes the momentum ``P_x`` by ``q B L``
    whatever the particle's momentum, and ``px`` is ``P_x / P0``. ``tune`` is the drive
    frequency in units of the revolution frequency (only its fractional part matters on
    a ring). ``lag`` is a phase **in turns**, as in xtrack and MAD-X — both of which
    multiply it by ``2 pi`` (xtrack's docstring says radians; its C does not).

    ``tune_rate`` [per turn] sweeps the drive (W5): the instantaneous drive tune on turn
    ``n`` is ``tune + tune_rate * n``, and the phase advances by
    ``2 pi (tune + tune_rate * (n + 1/2))`` from turn ``n`` to ``n + 1``. ``0`` (the
    default) is W1's fixed drive, bit for bit. Neither xtrack's ``ACDipole`` nor MAD-X's
    ``HACDIPOLE`` has a sweep; the fixed-drive optics (:func:`~accsim.twiss.driven_twiss`
    and its siblings) refuse a swept dipole.

    ``ramp = (r1, r2, r3, r4)`` [turns] is xtrack's trapezoid: off before ``r1``, a
    linear rise to full amplitude at ``r2``, flat until ``r3``, a linear fall to zero at
    ``r4``, off after. ``None`` (the default) is on at full amplitude from turn 0 for
    ever — the shape the steady-state identity needs.

    **The turn is not the element's to keep.** Its static maps — :meth:`matrix`,
    :meth:`kick`, :meth:`track` — are all the identity, so every optics function,
    closed orbit, normal form and Taylor map sees the undriven machine bit for bit. The
    per-turn kick lives in :meth:`drive_kick`, which the turn loops of
    :class:`~accsim.tracking.Tracker` apply, each passing its own turn index. A counter
    stored on the element would be advanced by the closed-orbit Newton solves and
    finite-difference Jacobians that call ``track`` many times a turn, and would hand
    each call a different kick.

    Thin, and with no misalignment arguments: a uniform kick has no centre to miss
    (the :class:`~accsim.elements.corrector.Corrector` argument). It changes no
    ``zeta``: the path lengthening of a kicked trajectory is second order in the kick.
    """

    def __init__(
        self,
        kick: float,
        tune: float,
        lag: float = 0.0,
        plane: str = "x",
        ramp: Sequence[int] | None = None,
        name: str | None = None,
        tune_rate: float = 0.0,
    ) -> None:
        super().__init__(0.0, name=name)
        if plane not in _PLANES:
            raise ValueError(f"plane must be 'x' or 'y', got {plane!r}")
        if ramp is not None:
            ramp = tuple(ramp)
            if (
                len(ramp) != 4
                or any(int(r) != r or r < 0 for r in ramp)
                or any(a > b for a, b in zip(ramp, ramp[1:], strict=False))
            ):
                raise ValueError(
                    "ramp must be four non-negative, non-decreasing integer turns "
                    f"(r1, r2, r3, r4), got {ramp!r}"
                )
            ramp = tuple(int(r) for r in ramp)
        self.amplitude = float(kick)
        self.tune = float(tune)
        self.tune_rate = float(tune_rate)
        self.lag = float(lag)
        self.plane = plane
        self.ramp: tuple[int, int, int, int] | None = ramp

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        """The identity: the static map is the undriven machine's (see the class docstring)."""
        return np.eye(DIM)

    def ramp_factor(self, turn: int) -> float:
        """The trapezoid's height on ``turn``, in ``[0, 1]`` — xtrack's piecewise form."""
        if self.ramp is None:
            return 1.0
        r1, r2, r3, r4 = self.ramp
        if turn < r1:
            return 0.0
        if turn < r2:
            return (turn - r1) / (r2 - r1)
        if turn < r3:
            return 1.0
        if turn < r4:
            return (r4 - turn) / (r4 - r3)
        return 0.0

    def drive_kick(self, turn: int, ref: ReferenceParticle) -> np.ndarray:
        """The ``(6,)`` kick added on pass ``turn`` (0 = the first), after the identity body."""
        out = np.zeros(DIM)
        sweep = 0.5 * self.tune_rate * turn * turn  # exactly 0.0 for a fixed drive
        phase = 2.0 * math.pi * (self.tune * turn + sweep + self.lag)
        out[_PLANES[self.plane]] = self.amplitude * self.ramp_factor(turn) * math.sin(phase)
        return out

    def __repr__(self) -> str:
        return (
            f"ACDipole(kick={self.amplitude}, tune={self.tune}, lag={self.lag}, "
            f"plane={self.plane!r}, ramp={self.ramp}, tune_rate={self.tune_rate}"
            f"{self._repr_tail()})"
        )
