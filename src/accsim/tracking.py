"""Particle / bunch state and a linear tracker.

Stage 0 tracks via the accumulated linear transfer matrix. Exact (symplectic,
possibly nonlinear) element maps for long-term tracking are added in later
stages; the API here (``Tracker.track`` / ``track_turns``) is the seam they will
plug into.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .coords import DELTA, DIM, PX, PY, ZETA, X, Y
from .lattice import Lattice


class Particle:
    """A single 6D phase-space state ``(x, px, y, py, zeta, delta)``."""

    __slots__ = ("state",)

    def __init__(
        self,
        x: float = 0.0,
        px: float = 0.0,
        y: float = 0.0,
        py: float = 0.0,
        zeta: float = 0.0,
        delta: float = 0.0,
    ) -> None:
        self.state = np.array([x, px, y, py, zeta, delta], dtype=float)

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> Particle:
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (DIM,):
            raise ValueError(f"expected a length-{DIM} state vector, got shape {arr.shape}")
        p = cls()
        p.state = arr.copy()
        return p

    # Named accessors onto the underlying vector.
    @property
    def x(self) -> float:
        return float(self.state[X])

    @property
    def px(self) -> float:
        return float(self.state[PX])

    @property
    def y(self) -> float:
        return float(self.state[Y])

    @property
    def py(self) -> float:
        return float(self.state[PY])

    @property
    def zeta(self) -> float:
        return float(self.state[ZETA])

    @property
    def delta(self) -> float:
        return float(self.state[DELTA])

    def __repr__(self) -> str:
        x, px, y, py, zeta, delta = self.state
        return f"Particle(x={x:g}, px={px:g}, y={y:g}, py={py:g}, zeta={zeta:g}, delta={delta:g})"


class Bunch:
    """A collection of N particles as a ``(6, N)`` array (one column per particle)."""

    __slots__ = ("states",)

    def __init__(self, states: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[0] != DIM:
            raise ValueError(f"expected a ({DIM}, N) array, got shape {states.shape}")
        self.states = states

    @property
    def n_particles(self) -> int:
        return self.states.shape[1]


class Tracker:
    """Pushes particles / bunches through a lattice using its linear map."""

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice

    def track(self, particle: Particle) -> Particle:
        """Track a single particle once through the lattice."""
        M = self.lattice.transfer_matrix()
        return Particle.from_array(M @ particle.state)

    def track_bunch(self, bunch: Bunch) -> Bunch:
        """Track every particle in a bunch once through the lattice."""
        M = self.lattice.transfer_matrix()
        return Bunch(M @ bunch.states)

    def track_turns(self, particle: Particle, n_turns: int) -> np.ndarray:
        """Track a particle for ``n_turns`` turns of the (closed) lattice.

        Returns an ``(n_turns + 1, 6)`` array of states including the initial one
        — the trajectory used by the long-term symplecticity smoke test.
        """
        if n_turns < 0:
            raise ValueError(f"n_turns must be >= 0, got {n_turns}")
        M = self.lattice.one_turn_matrix()
        history = np.empty((n_turns + 1, DIM))
        history[0] = particle.state
        s = particle.state.copy()
        for turn in range(1, n_turns + 1):
            s = M @ s
            history[turn] = s
        return history
