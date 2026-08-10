"""Particle / bunch state and a linear tracker.

Stage 0 tracks via the accumulated linear transfer matrix. Exact (symplectic,
possibly nonlinear) element maps for long-term tracking are added in later
stages; the API here (``Tracker.track`` / ``track_turns``) is the seam they will
plug into.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coords import DELTA, DIM, PX, PY, ZETA, X, Y
from .elements.aperture import Aperture
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


@dataclass
class LossResult:
    """Outcome of a loss-aware bunch track (see :meth:`Tracker.track_bunch_losses`).

    Per-particle bookkeeping, one entry per column of the input bunch:

    - ``states``       — ``(6, N)`` final states; a lost particle is **frozen** at
      its state on the turn it was lost (its columns stop updating thereafter).
    - ``alive``        — ``(N,)`` bool, ``True`` for particles that cleared every
      aperture on every turn.
    - ``loss_turn``    — ``(N,)`` int, the 0-based turn on which the particle was
      lost; ``-1`` for survivors.
    - ``loss_s``       — ``(N,)`` float, the **geometric** longitudinal position
      ``s`` [m] around the ring (in ``[0, C)``, independent of turn number) of the
      aperture that killed it; ``nan`` for survivors. This is the aperture's
      location, *not* the particle's ``zeta``.
    - ``loss_element`` — ``(N,)`` int, the index (into ``lattice.elements``) of the
      aperture that killed it; ``-1`` for survivors.
    """

    states: np.ndarray
    alive: np.ndarray
    loss_turn: np.ndarray
    loss_s: np.ndarray
    loss_element: np.ndarray

    @property
    def n_particles(self) -> int:
        return int(self.alive.size)

    @property
    def n_survived(self) -> int:
        return int(np.count_nonzero(self.alive))

    @property
    def n_lost(self) -> int:
        return self.n_particles - self.n_survived

    @property
    def transmission(self) -> float:
        """Surviving fraction ``n_survived / n_particles`` (the transmission)."""
        return self.n_survived / self.n_particles

    def loss_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate losses by longitudinal location, summed over all turns.

        Returns ``(s_locations, counts)`` sorted by ``s``: the distinct aperture
        positions [m] where particles were lost and how many died at each. This is
        the loss map — loss count vs. position around the ring.
        """
        lost = self.loss_turn >= 0
        s = np.round(self.loss_s[lost], 9)  # nm-level grouping of identical positions
        return np.unique(s, return_counts=True)


class Tracker:
    """Pushes particles / bunches through a lattice using its linear map."""

    def __init__(self, lattice: Lattice) -> None:
        self.lattice = lattice

    def track(self, particle: Particle, nonlinear: bool = False) -> Particle:
        """Track a single particle once through the lattice.

        ``nonlinear=False`` (default) uses the accumulated linear transfer matrix.
        ``nonlinear=True`` pushes the state element-by-element through each
        element's :meth:`~accsim.elements.element.Element.track`, so nonlinear
        maps act exactly: the RF cavity's ``sin`` kick and the sextupole's
        ``x^2 - y^2`` kick.

        **A lattice containing a sextupole is not a purely linear lattice.** The
        two paths agree to round-off only when every element's map *is* its
        matrix; with a sextupole (or an RF cavity) present, ``nonlinear=False``
        silently drops the nonlinear part — for a sextupole that means tracking
        its drift map and nothing else. That default is deliberate (every optics
        quantity in the package is built on the linear map, and the sextupole's
        linear map really is a drift), but it is a choice the caller has to make
        knowingly.
        """
        if not nonlinear:
            M, k = self.lattice.transfer_map()
            return Particle.from_array(M @ particle.state + k)
        return Particle.from_array(self._track_once(particle.state.copy()))

    def _track_once(self, state: np.ndarray) -> np.ndarray:
        """One element-by-element pass through the lattice (nonlinear)."""
        for elem in self.lattice.elements:
            state = elem.track(state, self.lattice.ref)
        return state

    def track_bunch(self, bunch: Bunch) -> Bunch:
        """Track every particle in a bunch once through the lattice."""
        M, k = self.lattice.transfer_map()
        return Bunch(M @ bunch.states + k[:, None])

    def track_bunch_losses(
        self, bunch: Bunch, n_turns: int = 1, nonlinear: bool = False
    ) -> LossResult:
        """Track a bunch with aperture loss accounting.

        Walks the lattice element-by-element for ``n_turns`` turns, accumulating
        the geometric ``s``. At each :class:`~accsim.elements.aperture.Aperture`
        the surviving particles are tested against its geometric predicate; a
        particle that fails is recorded (turn, ``s``, element index) and
        **frozen** — its state stops advancing and it is skipped on every later
        element and turn.

        ``nonlinear=False`` (default) acts with each element's affine 6x6, hoisted
        out of the turn loop. ``nonlinear=True`` routes every element through its
        :meth:`~accsim.elements.element.Element.track` instead, so a sextupole's
        (or RF cavity's) nonlinear map acts on the bunch. The flag exists because
        the default would otherwise *silently* linearise a sextupole in a
        loss-aware track — the answer would look fine and be the wrong physics.

        It is not a dynamic-aperture facility. Nonlinear tracking against apertures
        is the machinery DA studies are built from, but nothing in the package
        gates amplitude-dependent survival, and DA is out of scope
        (``docs/ROADMAP.md``); treat a loss count from a nonlinear track as
        illustrative unless you have gated it yourself.

        Returns a :class:`LossResult`. Loss location is the aperture's geometric
        ``s`` around the ring, not the particle's ``zeta``.
        """
        if n_turns < 1:
            raise ValueError(f"n_turns must be >= 1, got {n_turns}")
        ref = self.lattice.ref
        n = bunch.n_particles
        states = bunch.states.astype(float, copy=True)
        alive = np.ones(n, dtype=bool)
        loss_turn = np.full(n, -1, dtype=int)
        loss_s = np.full(n, np.nan)
        loss_element = np.full(n, -1, dtype=int)

        # Hoisted out of the turn loop: the lattice cannot change during a track,
        # and this inner loop runs n_turns * len(lattice) times (1e5 and up in the
        # long-term gates). The kick is stored as None when it is zero — which is
        # every element but a corrector — so the common case never broadcasts it.
        maps: list[tuple[np.ndarray, np.ndarray | None]] = []
        if not nonlinear:
            for elem in self.lattice.elements:
                k = elem.kick(ref)
                maps.append((elem.matrix(ref), k[:, None] if k.any() else None))

        for turn in range(n_turns):
            s = 0.0
            for ei, elem in enumerate(self.lattice.elements):
                if alive.any():
                    if nonlinear:
                        states[:, alive] = elem.track(states[:, alive], ref)
                    else:
                        M, k_col = maps[ei]
                        if k_col is None:
                            states[:, alive] = M @ states[:, alive]
                        else:
                            states[:, alive] = M @ states[:, alive] + k_col
                if isinstance(elem, Aperture):
                    inside = np.asarray(elem.survives(states), dtype=bool)
                    newly = alive & ~inside
                    if newly.any():
                        loss_turn[newly] = turn
                        loss_s[newly] = s
                        loss_element[newly] = ei
                        alive[newly] = False
                s += elem.length

        return LossResult(states, alive, loss_turn, loss_s, loss_element)

    def track_turns(self, particle: Particle, n_turns: int, nonlinear: bool = False) -> np.ndarray:
        """Track a particle for ``n_turns`` turns of the (closed) lattice.

        Returns an ``(n_turns + 1, 6)`` array of states including the initial one
        — the trajectory used by the long-term symplecticity smoke test.

        ``nonlinear=False`` (default) applies the one-turn matrix each turn (fast,
        exact for linear lattices). ``nonlinear=True`` pushes element-by-element so
        the RF cavity's ``sin`` kick and the sextupole's ``x^2 - y^2`` kick act
        exactly — the path for RF-bucket / separatrix long-term tracking, and the
        only path on which a sextupole does anything at all (see :meth:`track`).
        """
        if n_turns < 0:
            raise ValueError(f"n_turns must be >= 0, got {n_turns}")
        history = np.empty((n_turns + 1, DIM))
        history[0] = particle.state
        s = particle.state.copy()
        if nonlinear:
            for turn in range(1, n_turns + 1):
                s = self._track_once(s)
                history[turn] = s
        else:
            M, k = self.lattice.one_turn_map()
            for turn in range(1, n_turns + 1):
                s = M @ s + k
                history[turn] = s
        return history
