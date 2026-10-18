r"""Island chains: the free ring's period-``q`` orbits near a resonance (W9).

A nonlinear ring's tune moves with amplitude. Where it crosses a rational ``p/q`` the motion
does not simply rotate: the ``q``-turn map has a chain of fixed points, alternately stable
(the centres of **islands** a particle can be trapped in) and unstable (the corners of the
separatrix between them). W8 found that such a chain, the free ring's own six islands at
1/3, is what ends a damped autoresonant lock early. :func:`island_chain` finds a chain.

**How.** Newton on the ``q``-turn map, in one transverse plane, from a ring of seeds at
Courant-Snyder action ``0.9, 1, 1.1 x action`` about the nonlinear closed orbit. Every map
is the element-by-element :meth:`~accsim.elements.element.Element.track`, so the exact drift
and every multipole kick act. Newton reaches more than the chain, and three filters remove
what is not it:

- **An orbit shorter than ``q``.** Every point Newton reaches is completed into its whole
  orbit by the one-turn map. The closed orbit is a period-1 orbit, and a composite ``q``
  also catches the orbits of its divisors; both are dropped by their length, not by a cut
  in amplitude.
- **The wrong rotation.** On the 4-turn map a ``1/4`` orbit and a ``3/4`` one are both fixed
  points. An orbit is kept only if its per-turn phase advances, each taken in ``[0, 2 pi)``,
  add up to ``p`` turns over its ``q`` turns.
- **A duplicate.** Points within ``1e-7`` of the seed radius of one already found are the
  same point.

**What it returns.** Each :class:`IslandOrbit` carries its ``q`` points in turn order, their
actions and phases about the closed orbit (``h = sqrt(2 J) e^(-i phi)``, so ``phi =
atan2(-p_hat, u_hat)``), and the trace of the ``q``-turn Jacobian: ``|trace| < 2`` is
elliptic (stable), and then the libration is ``arccos(trace / 2) / q`` per turn. The
:class:`IslandChain` carries the island count ``m`` (the points on stable orbits) and the
pendulum half-width in tune ``omega / (pi m)``, which W8 gate 2 checked against a tracked
plateau.

**Refused.** A live :class:`~accsim.elements.acdipole.ACDipole` (the driven ring's orbits are
:func:`~accsim.twiss.driven_states`'), a live :class:`~accsim.elements.damper.Damper` (it
contracts area, so ``|trace| < 2`` no longer means elliptic), and a ring whose ``q``-turn map
leaks out of the plane (:class:`~accsim.twiss.CoupledLatticeError`: a skew quadrupole, a skew
sextupole, a vertical orbit through an octupole) or moves ``delta`` (a cavity off phase).

**Scope.** One uncoupled transverse plane, on momentum, a guess at the chain's action from
the caller. Not here: a 4D search for chains in a coupled ring, off-momentum chains, and a
search that finds the action itself (by scanning the tracked tune for plateaus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .coords import DELTA, DIM, PX, PY, X, Y
from .elements.acdipole import ACDipole
from .elements.damper import Damper
from .lattice import Lattice
from .orbit import closed_orbit_nonlinear
from .twiss import CoupledLatticeError, closed_twiss_on_orbit

_PLANES = {"x": (X, PX), "y": (Y, PY)}
_SEED_RINGS = (0.9, 1.0, 1.1)
_NEWTON_ITER = 60
_TOL = 1e-12  # accepted residual, relative to the seed radius sqrt(2 action)
_SAME = 1e-7  # two points closer than this (relative) are one
_LEAK = 1e-9  # off-plane motion above this (relative) is coupling


@dataclass(frozen=True)
class IslandOrbit:
    """One period-``q`` orbit of the free ring.

    ``points`` are ``(6,)`` states at the lattice start in turn order (``points[k + 1]`` is
    one turn after ``points[k]``). ``actions`` [m rad] and ``phases`` [rad] are each point's
    Courant-Snyder action and phase about the closed orbit. ``trace`` is the trace of the
    ``q``-turn Jacobian, the same at every point of the orbit.
    """

    points: tuple[np.ndarray, ...]
    actions: tuple[float, ...]
    phases: tuple[float, ...]
    trace: float

    @property
    def stable(self) -> bool:
        """Elliptic: a particle started near it librates about it."""
        return abs(self.trace) < 2.0

    @property
    def libration(self) -> float | None:
        """The libration ``arccos(trace / 2) / q`` [rad per turn] of a stable orbit."""
        if not self.stable:
            return None
        return math.acos(self.trace / 2.0) / len(self.points)


@dataclass(frozen=True)
class IslandChain:
    """The period-``q`` orbits :func:`island_chain` found, turning ``tune`` per turn."""

    tune: Fraction
    plane: str
    orbits: tuple[IslandOrbit, ...]

    @property
    def islands(self) -> int:
        """``m``: the number of islands, one per point on a stable orbit."""
        return sum(len(o.points) for o in self.orbits if o.stable)

    @property
    def libration(self) -> float | None:
        """``omega``: the stable orbits' mean libration [rad per turn]."""
        rates = [o.libration for o in self.orbits if o.stable]
        return float(np.mean(rates)) if rates else None

    @property
    def half_width(self) -> float | None:
        """The pendulum half-width in tune, ``omega / (pi m)``: how far either side of the
        resonance the chain holds a particle's tune. A pendulum's, so it assumes an island
        narrow against its distance from the closed orbit."""
        omega = self.libration
        return None if omega is None else omega / (math.pi * self.islands)


class _Plane:
    """The free ring seen in one plane: normalised coordinates about the closed orbit."""

    def __init__(self, lattice: Lattice, plane: str) -> None:
        self.lattice = lattice
        self.iu, self.ip = _PLANES[plane]
        self.base = np.zeros(DIM)
        self.base[:4] = closed_orbit_nonlinear(lattice)
        tw = closed_twiss_on_orbit(lattice)
        self.beta = float(tw.beta_x if plane == "x" else tw.beta_y)
        self.alpha = float(tw.alpha_x if plane == "x" else tw.alpha_y)
        self.off = [k for k in (X, PX, Y, PY, DELTA) if k not in (self.iu, self.ip)]

    def states(self, w: np.ndarray) -> np.ndarray:
        """``(6, n)`` states at normalised deviations ``w = (u_hat, p_hat)``, ``(2, n)``."""
        rb = math.sqrt(self.beta)
        s = np.repeat(self.base[:, None], w.shape[1], axis=1)
        s[self.iu] += rb * w[0]
        s[self.ip] += (w[1] - self.alpha * w[0]) / rb
        return s

    def normalised(self, s: np.ndarray) -> np.ndarray:
        rb = math.sqrt(self.beta)
        du, dp = s[self.iu] - self.base[self.iu], s[self.ip] - self.base[self.ip]
        return np.array([du / rb, (self.alpha * du + self.beta * dp) / rb])

    def turns(self, s: np.ndarray, n: int) -> np.ndarray:
        for _ in range(n):
            for e in self.lattice.elements:
                s = e.track(s, self.lattice.ref)
        return s

    def map(self, w: np.ndarray, n: int) -> np.ndarray:
        """``n`` turns of the free ring in normalised coordinates."""
        with np.errstate(all="ignore"):
            return self.normalised(self.turns(self.states(w), n))

    def leak(self, w: np.ndarray, n: int, radius: float) -> float:
        """The largest off-plane change over ``n`` turns, relative to the in-plane size."""
        with np.errstate(all="ignore"):
            out = self.turns(self.states(w), n)
        scale = np.array([radius * math.sqrt(self.beta), radius / math.sqrt(self.beta)])
        size = {X: scale[0], Y: scale[0], PX: scale[1], PY: scale[1], DELTA: scale[1]}
        worst = 0.0
        for k in self.off:
            d = np.abs(out[k] - self.base[k])
            d = d[np.isfinite(d)]
            if d.size:
                worst = max(worst, float(d.max()) / size[k])
        return worst


def _refuse_turn_elements(lattice: Lattice) -> None:
    for e in lattice.elements:
        if isinstance(e, ACDipole) and e.amplitude != 0.0:
            raise ValueError(
                "island_chain() is the free ring's: the lattice has a live ACDipole drive "
                "(amplitude != 0); driven orbits are driven_states()'"
            )
        if isinstance(e, Damper) and e.rate != 0.0:
            raise ValueError(
                "island_chain() is the free ring's: a live Damper contracts area, so "
                "|trace| < 2 would no longer mean a stable orbit"
            )


def _newton(ring: _Plane, w: np.ndarray, q: int, radius: float) -> np.ndarray:
    """Newton on the ``q``-turn map from every column of ``w`` at once, with a halving line
    search per column. Returns the converged columns, ``(2, k)``."""
    h = 1e-8 * radius
    pert = np.array([[0.0, h, -h, 0.0, 0.0], [0.0, 0.0, 0.0, h, -h]])
    alive = np.ones(w.shape[1], dtype=bool)
    for _ in range(_NEWTON_ITER):
        n = w.shape[1]
        cols = (w[:, :, None] + pert[:, None, :]).reshape(2, 5 * n)
        out = ring.map(cols, q).reshape(2, n, 5)
        alive &= np.all(np.isfinite(out), axis=(0, 2))
        F = out[:, :, 0] - w
        jac = np.stack(
            [(out[:, :, 1] - out[:, :, 2]) / (2 * h), (out[:, :, 3] - out[:, :, 4]) / (2 * h)],
            axis=-1,
        )  # (2, n, 2): jac[:, j, :] is column j's Jacobian
        f0 = np.where(alive, np.max(np.abs(F), axis=0), 0.0)
        todo = alive & (f0 > 1e-15 * radius)
        if not todo.any():
            break
        # (J - I) step = -F, column by column, by the 2x2 inverse (a singular J - I gives a
        # non-finite step, which the line search below refuses)
        a, b = jac[0, :, 0] - 1.0, jac[0, :, 1]
        c, d = jac[1, :, 0], jac[1, :, 1] - 1.0
        with np.errstate(all="ignore"):
            det = a * d - b * c
            step = np.array([-(d * F[0] - b * F[1]) / det, -(a * F[1] - c * F[0]) / det])
        step[:, ~todo] = 0.0
        t = np.where(todo, 1.0, 0.0)
        bad = todo.copy()
        while bad.any():
            trial = w[:, bad] + t[bad] * step[:, bad]
            r = np.max(np.abs(ring.map(trial, q) - trial), axis=0)
            ok = np.isfinite(r) & (r <= f0[bad])
            idx = np.nonzero(bad)[0]
            bad[idx[ok]] = False
            t[idx[~ok]] /= 2.0
            bad &= t > 1e-4
        w = w + t * step
    r = np.max(np.abs(ring.map(w, q) - w), axis=0)
    good = alive & np.isfinite(r) & (r <= _TOL * radius)
    return w[:, good]


def _orbit_of(ring: _Plane, w0: np.ndarray, q: int, radius: float) -> list[np.ndarray] | None:
    """The one-turn images of ``w0`` up to its return, or None if it is shorter than ``q``
    or does not return within ``q`` turns."""
    pts = [w0]
    for k in range(1, q + 1):
        nxt = ring.map(pts[-1][:, None], 1)[:, 0]
        if np.max(np.abs(nxt - w0)) < _SAME * radius:
            return pts if k == q else None
        pts.append(nxt)
    return None


def _trace(ring: _Plane, w0: np.ndarray, q: int, radius: float) -> float:
    h = 1e-8 * radius
    cols = w0[:, None] + np.array([[h, -h, 0.0, 0.0], [0.0, 0.0, h, -h]])
    out = ring.map(cols, q)
    return float((out[0, 0] - out[0, 1]) / (2 * h) + (out[1, 2] - out[1, 3]) / (2 * h))


def island_chain(
    lattice: Lattice,
    tune: Fraction | float,
    action: float,
    *,
    plane: str = "x",
    phases: int = 36,
) -> IslandChain:
    r"""The free ring's period-``q`` orbits near Courant-Snyder ``action``, turning ``tune``.

    ``tune`` is the resonance's rotation per turn, ``p/q`` (a float is read as the nearest
    fraction with denominator up to 1000; only its fractional part matters, so ``4/3`` and
    ``1/3`` are the same chain). ``action`` [m rad] is a guess at the chain's action about
    the closed orbit: seeds sit at ``0.9, 1, 1.1`` times it, ``phases`` of each. ``plane`` is
    ``"x"`` or ``"y"``. Returns an :class:`IslandChain`, whose ``orbits`` are empty if
    Newton reached no orbit of that rotation and length. See the module docstring for the
    filters and the refusals.
    """
    if plane not in _PLANES:
        raise ValueError(f"plane must be 'x' or 'y', got {plane!r}")
    if not action > 0.0:
        raise ValueError(f"action must be > 0, got {action!r}")
    if phases < 1:
        raise ValueError(f"phases must be >= 1, got {phases!r}")
    frac = (
        Fraction(tune) if isinstance(tune, Fraction) else Fraction(tune).limit_denominator(1000)
    ) % 1
    if frac == 0:
        raise ValueError(f"tune must not be an integer, got {tune!r}")
    p, q = frac.numerator, frac.denominator
    _refuse_turn_elements(lattice)
    ring = _Plane(lattice, plane)

    radius = math.sqrt(2.0 * action)
    phi = np.linspace(0.0, 2.0 * math.pi, phases, endpoint=False)
    rings = np.repeat(np.sqrt(2.0 * action * np.array(_SEED_RINGS)), phases)
    angles = np.tile(phi, len(_SEED_RINGS))
    seeds = np.array([rings * np.cos(angles), -rings * np.sin(angles)])
    if ring.leak(seeds, q, radius) > _LEAK:
        raise CoupledLatticeError(
            f"the {q}-turn map leaks out of the {plane!r} plane (or moves delta): the lattice "
            "is coupled at this amplitude, and a one-plane chain does not exist"
        )

    orbits: list[IslandOrbit] = []
    seen: list[np.ndarray] = []
    for w0 in _newton(ring, seeds, q, radius).T:
        if any(np.max(np.abs(w0 - s)) < _SAME * radius for s in seen):
            continue
        pts = _orbit_of(ring, w0, q, radius)
        if pts is None:
            seen.append(w0)
            continue
        seen.extend(pts)
        ph = [math.atan2(-w[1], w[0]) for w in pts]
        advance = sum((ph[(k + 1) % q] - ph[k]) % (2.0 * math.pi) for k in range(q))
        if round(advance / (2.0 * math.pi)) != p:
            continue
        if ring.leak(np.array(pts).T, q, radius) > _LEAK:
            raise CoupledLatticeError(
                f"the chain's {q}-turn map leaks out of the {plane!r} plane (or moves delta)"
            )
        states = ring.states(np.array(pts).T)
        orbits.append(
            IslandOrbit(
                points=tuple(states[:, k].copy() for k in range(q)),
                actions=tuple(0.5 * float(w @ w) for w in pts),
                phases=tuple(ph),
                trace=_trace(ring, w0, q, radius),
            )
        )
    return IslandChain(tune=frac, plane=plane, orbits=tuple(orbits))
