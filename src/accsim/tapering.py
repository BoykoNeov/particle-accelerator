r"""Tapering: the sawtooth momentum a radiating ring actually runs at (Q1).

A ring that radiates is not the machine its lattice file describes. The beam bleeds energy
continuously through the arcs and gets it back in one lump at the cavity, so the particle
sits at a **different momentum in every magnet**. That profile — ``delta`` as a function of
``s``, falling through each radiating element and jumping back at the RF — is a sawtooth
whose peak-to-peak height is exactly the fractional turn loss ``U_0 / (beta0^2 E_0)``.

This module computes that profile and nothing else. *Applying* it — scaling each magnet's
field by ``1 + delta(s)`` so that the beam sees the nominal machine again — is Q2, and it
needs a bending magnet whose field is separable from its geometry, which :class:`Dipole`
does not yet have (see ``docs/ROADMAP.md`` → Q2).

**Where the profile is centred is the physics; its height is bookkeeping.** The span is
fixed by conservation — the beam loses ``U_0`` per turn whatever else is wrong — so a
profile running ``0 → −U_0/E_0`` has *exactly the same span* as the right one while being
wrong by half the effect in every magnet. The offset is therefore a real choice, and this
module makes the same one xtrack does: ``delta0="zero_mean"`` puts the **length-weighted
mean of the profile at zero**, so the ring runs as far above the design momentum as below.

That choice is not arbitrary, and I4 is why. A radiating ring's 6D fixed point is *where
the sag is centred*: ``energy_loss_per_turn`` departs from the tracked loss at first order
about the design orbit and at **second** order about the closed one (I4's fitted exponents
0.999 and 2.003). xtrack **imposes** the zero mean as a convention; accsim's own
:func:`~accsim.orbit.closed_orbit_6d` **arrives** at it from periodicity. The two are
therefore an independent pair, which is why this function is deliberately built by forward
tracking and **not** from ``closed_orbit_6d`` — their agreement is a gate in
``tests/analytic/test_tapering.py`` rather than a dependency here.

**The cavity is stepped over, not tracked.** The profile is a property of the magnets: what
the beam's momentum does *between* refills. Tracking the cavity would fold the RF's own
restoring kick into the ramp and hide the sawtooth's teeth. xtrack does the same thing
(``XS_FLAG_KILL_CAVITY_KICK`` while it measures the ramp), for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coords import DELTA
from .elements.rfcavity import RFCavity
from .lattice import Lattice
from .orbit import closed_orbit

__all__ = ["TaperProfile", "taper_profile"]

#: Rounds of re-centring. The profile depends on its own starting momentum only through the
#: radiated power's ``E^2``, so the mean falls by a constant factor per round rather than
#: needing a solve — measured on I4's ring at ``4.5e-3`` per round, reaching round-off
#: (``1.6e-17``) by the sixth. The convergence *ratio* is a test, not a comment.
_ROUNDS = 6


@dataclass(frozen=True)
class TaperProfile:
    """The per-element momentum deviation of a radiating ring, and how it was centred.

    ``delta`` is one value per element — the **midpoint** of the element's entrance and
    exit momenta, which is what a magnet of finite length actually integrates and what
    both reference codes store (xtrack's ``delta_taper`` is the same average; MAD-X's
    ``pt`` column is the boundary value). ``boundary`` carries those boundaries, one more
    entry than there are elements, for a caller that wants the raw ramp.
    """

    delta: np.ndarray
    boundary: np.ndarray
    s_mid: np.ndarray
    lengths: np.ndarray
    delta_start: float
    loss_eV: float

    @property
    def span(self) -> float:
        """Peak-to-peak height of the sawtooth — the fractional turn loss."""
        return float(self.boundary.max() - self.boundary.min())

    @property
    def mean(self) -> float:
        """Length-weighted mean of the profile: ``0`` by construction under ``zero_mean``.

        Thin elements have no length and so no weight — they are places the profile is
        *read*, not places it is *spent*. Weighting them equally with a metre of dipole
        would make the answer depend on how finely the lattice happens to be written.
        """
        total = float(self.lengths.sum())
        if total == 0.0:
            return float(self.delta.mean()) if self.delta.size else 0.0
        return float(np.dot(self.delta, self.lengths) / total)


def _ramp(
    lattice: Lattice, orbit4: np.ndarray, delta_start: float
) -> tuple[np.ndarray, np.ndarray]:
    """One turn's momentum ramp: ``(boundary deltas, per-element midpoints)``.

    The cavity is stepped over; every other element is tracked with ``radiation="mean"``.
    """
    state = np.array([orbit4[0], orbit4[1], orbit4[2], orbit4[3], 0.0, delta_start], dtype=float)
    boundary = [float(state[DELTA])]
    for elem in lattice.elements:
        if not isinstance(elem, RFCavity):
            state = elem.track(state, lattice.ref, radiation="mean")
        boundary.append(float(state[DELTA]))
    edges = np.asarray(boundary, dtype=float)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def taper_profile(
    lattice: Lattice, *, delta0: float | str = "zero_mean", rounds: int = _ROUNDS
) -> TaperProfile:
    r"""The momentum profile ``delta(s)`` a radiating ``lattice`` runs at (Q1).

    ``delta0`` is the momentum the turn *starts* at: a number to pin it, or the default
    ``"zero_mean"`` to solve for the value that puts the profile's length-weighted mean at
    zero. Only the latter is a physical statement — see the module docstring.

    The trajectory is the transverse closed orbit at ``delta = 0`` (which is exactly the
    design orbit on a machine with no correctors and no misalignments), carried round once
    with ``radiation="mean"`` and the cavity stepped over.

    Raises :class:`ValueError` if ``delta0`` is neither a number nor ``"zero_mean"``.
    """
    if isinstance(delta0, str) and delta0 != "zero_mean":
        raise ValueError(f"delta0 must be a number or 'zero_mean', got {delta0!r}")

    orbit4 = closed_orbit(lattice)
    lengths = np.array([e.length for e in lattice.elements], dtype=float)
    ends = np.cumsum(lengths)
    s_mid = ends - 0.5 * lengths

    start = 0.0 if isinstance(delta0, str) else float(delta0)
    edges, mid = _ramp(lattice, orbit4, start)
    if isinstance(delta0, str):
        total = float(lengths.sum())
        for _ in range(max(1, rounds)):
            start -= float(np.dot(mid, lengths) / total) if total else float(mid.mean())
            edges, mid = _ramp(lattice, orbit4, start)

    ref = lattice.ref
    loss = float(edges[0] - edges[-1]) * ref.beta0**2 * ref.total_energy_eV
    return TaperProfile(
        delta=mid,
        boundary=edges,
        s_mid=s_mid,
        lengths=lengths,
        delta_start=start,
        loss_eV=loss,
    )
