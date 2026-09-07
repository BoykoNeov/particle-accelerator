r"""Tapering: the sawtooth momentum a radiating ring actually runs at (Q1).

A ring that radiates is not the machine its lattice file describes. The beam bleeds energy
continuously through the arcs and gets it back in one lump at the cavity, so the particle
sits at a **different momentum in every magnet**. That profile — ``delta`` as a function of
``s``, falling through each radiating element and jumping back at the RF — is a sawtooth
whose peak-to-peak height is exactly the fractional turn loss ``U_0 / (beta0^2 E_0)``.

This module computes that profile (:func:`taper_profile`, Q1) and applies it
(:func:`taper`, Q2) — scaling each magnet's field by ``1 + delta(s)`` so that the beam
sees the nominal machine again. Applying it needed a bending magnet whose field is
separable from its geometry, which is why it is its own milestone:
:class:`~accsim.elements.dipole.Dipole` now carries ``k0`` alongside ``angle``, and the
two stop being the same number.

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

import copy
from dataclasses import dataclass

import numpy as np

from .coords import DELTA
from .elements.aperture import AcceptanceElement
from .elements.beambeam import BeamBeam
from .elements.corrector import Corrector
from .elements.dipole import Dipole
from .elements.drift import Drift
from .elements.element import Element
from .elements.octupole import Octupole, ThinOctupole
from .elements.quadrupole import Quadrupole, ThinQuadrupole
from .elements.rfcavity import RFCavity
from .elements.sextupole import Sextupole, ThinSextupole, ThinSkewSextupole
from .elements.skew_quadrupole import SkewQuadrupole, ThinSkewQuadrupole
from .elements.solenoid import Solenoid
from .elements.wiggler import Wiggler
from .lattice import Lattice
from .orbit import closed_orbit, closed_orbit_6d

__all__ = ["TaperProfile", "taper", "taper_profile"]

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


#: Every powered strength in the package, by the element that owns it. A taper scales
#: **fields**, and a field is exactly what is on this list — so the table is this
#: milestone's definition of "magnet" rather than a convenience. :class:`Dipole` is here
#: for its *gradient* only; its dipole field ``k0`` is handled separately, because it is
#: the one strength that is not stored as a strength (see :func:`taper`).
_STRENGTHS: tuple[tuple[type, tuple[str, ...]], ...] = (
    (Dipole, ("k1",)),
    (Quadrupole, ("k1",)),
    (ThinQuadrupole, ("k1l",)),
    (SkewQuadrupole, ("k1s",)),
    (ThinSkewQuadrupole, ("k1sl",)),
    (Sextupole, ("k2",)),
    (ThinSextupole, ("k2l",)),
    (ThinSkewSextupole, ("k2sl",)),
    (Solenoid, ("ks",)),
    (Octupole, ("k3",)),
    (ThinOctupole, ("k3l",)),
    (Corrector, ("kick_x", "kick_y")),
    # A wiggler is a powered magnet like any other, and ``h0`` is its strength
    # (T2). Scaling it is the Q2 statement in the form this element takes: a
    # tapered wiggler **is** the design wiggler seen at a rescaled momentum, and
    # because its focusing is ``h0^2/2`` that focusing scales as the *square* of
    # the factor — the only entry in this table whose optics is not linear in it.
    (Wiggler, ("h0",)),
)

#: Elements a taper deliberately leaves alone, and the list is short because each entry is
#: a statement. A :class:`~accsim.elements.drift.Drift` and an
#: :class:`~accsim.elements.aperture.AcceptanceElement` have no field at all; an
#: :class:`~accsim.elements.rfcavity.RFCavity` is the thing *paying* for the loss, and its
#: voltage is set by that loss rather than by the local momentum; a
#: :class:`~accsim.elements.beambeam.BeamBeam` field belongs to the opposing bunch and no
#: power supply of this machine can scale it.
_UNPOWERED: tuple[type, ...] = (Drift, RFCavity, AcceptanceElement, BeamBeam)


def _scaled(element: Element, factor: float) -> Element:
    """A shallow copy of ``element`` with every powered strength times ``factor``.

    Shallow, for the reason :func:`~accsim.orbit._with_offset` is: elements hold scalars
    and (in the beam-beam case) arrays that are never written to.
    """
    if isinstance(element, _UNPOWERED):
        return element
    for kind, names in _STRENGTHS:
        if isinstance(element, kind):
            twin = copy.copy(element)
            for name in names:
                setattr(twin, name, getattr(twin, name) * factor)
            if isinstance(twin, Dipole):
                # The bend's own field. It is *not* ``angle``, and that is the whole
                # milestone: the geometry stays the ring's while the field follows the
                # beam. Scaling ``k0`` from its current value (rather than from the
                # curvature) is what lets a taper be applied to an already-tapered ring.
                twin.k0 *= factor
            return twin
    raise NotImplementedError(
        f"taper() does not know whether a {type(element).__name__} carries a field "
        f"({element.name!r}). Add it to _STRENGTHS if it does and to _UNPOWERED if it "
        "does not — silently leaving a magnet at its design strength would taper the "
        "ring for a machine it is not"
    )


def _orbit_ramp(lattice: Lattice) -> np.ndarray:
    """Per-element midpoint ``delta`` on the ring's own **radiating** 6D closed orbit.

    The profile a machine actually runs at once it is tapered, as opposed to the one
    :func:`taper_profile` forward-tracks on the design ring. The two are not the same
    object and the difference is what the rounds of :func:`taper` are for.
    """
    state = closed_orbit_6d(lattice, radiation="mean").copy()
    edges = [float(state[DELTA])]
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref, radiation="mean")
        edges.append(float(state[DELTA]))
    ramp = np.asarray(edges, dtype=float)
    return 0.5 * (ramp[:-1] + ramp[1:])


def _apply(lattice: Lattice, deltas: np.ndarray) -> Lattice:
    """``lattice`` with element ``i``'s field scaled by ``1 + deltas[i]``."""
    if len(deltas) != len(lattice.elements):
        raise ValueError(
            f"profile has {len(deltas)} entries for a lattice of "
            f"{len(lattice.elements)} elements — it belongs to a different ring"
        )
    return Lattice(
        [_scaled(e, 1.0 + float(d)) for e, d in zip(lattice.elements, deltas, strict=True)],
        ref=lattice.ref,
    )


def taper(lattice: Lattice, profile: TaperProfile | None = None, *, rounds: int = 3) -> Lattice:
    r"""The **tapered** ring: every magnet set to the momentum the beam has there (Q2).

    A radiating ring is not the machine its lattice file describes — the beam is at a
    different momentum in every magnet (:func:`taper_profile`), so every magnet is
    mis-set for the beam it is steering and the ring closes on a dispersion orbit
    millimetres wide (``7.09 mm`` on I4's 6.5 GeV ring). Tapering scales each magnet's
    field by ``1 + delta(s)`` so that a particle at ``delta(s)`` feels exactly the
    nominal magnet again. Returns a **new** lattice; the caller's is untouched, in the
    idiom ``orbit._with_offset`` and ``radiation._coupling_off_lattice`` already use.

    **It is a fixed point, not a formula, and that is this milestone's finding.** The
    first round uses ``profile`` (default :func:`taper_profile`), which is built on the
    *design* ring. Every further round re-derives the profile from the **tapered** ring's
    own radiating closed orbit and re-applies it from the design strengths — never
    compounding, since the ramp a taper must follow does not shrink when the ring is
    tapered (the machine radiates exactly as much as before; it is the *orbit* that goes
    away, not the sawtooth). Applying the same profile twice therefore tapers twice and
    is a bug, which is why the rounds always start from ``lattice``.

    **Each round removes one power of the sag**, measured over a factor 2.8 in span:

    ==========  ==========================  ==================================
    rounds      residual ``|x|`` on I4      the law
    ==========  ==========================  ==================================
    0           ``7.090e-3`` m              ``1.858 * span``
    1           ``6.828e-6`` m              ``0.4688 * span^2``
    2           ``5.506e-9`` m              ``0.104 * span^3``
    3           ``1.00e-10`` m              the closed-orbit solver's own floor
    ==========  ==========================  ==================================

    The default ``rounds=3`` is the floor, and it is where the reference code lands too:
    xtrack's ``compensate_radiation_energy_loss`` leaves ``9.70e-11`` on this same ring,
    which is this number to 3%. A fourth round does not improve on it. It is the *order*
    that is the test (J2's rule) and not any of those constants: a taper wrong by a
    coefficient leaves a residual first order in the span, and no number of rounds
    changes that.

    ``rounds=1`` is the one-shot: the design-ring profile applied once, with no closed
    orbit solved at all. It is the only mode available on a ring with **no RF cavity**,
    where :func:`~accsim.orbit.closed_orbit_6d` has no isolated fixed point to find. It is
    also the mode a deliberate break shows up in — the solve *repairs* a wrong profile,
    which is what makes it a solve.

    ``profile``'s ``delta`` is the per-element **midpoint** — what a magnet of finite
    length actually integrates, and what xtrack's ``delta_taper`` stores — so a magnet is
    set for the momentum at its own centre rather than at its entrance. Slicing the bends
    changes the answer by ``0.3%`` and no more (measured to 8 slices): the residual is
    *not* the magnets' length.

    **Why this is a milestone and not a loop over strengths.** For every magnet but one, a
    field is a number and scaling it is arithmetic. A
    :class:`~accsim.elements.dipole.Dipole` stores ``angle``, so its field and its
    curvature *are the same number* (``k0 = h = angle/L``) — and a tapered bend has
    ``k0 = h(1 + delta_t)`` with ``h`` **unchanged**, because the geometry belongs to the
    ring (the bends must still sum to ``2 pi``) while the field belongs to the beam.
    Splitting those two is what ``k0`` exists for.

    **The shortcut that does not work**, checked symbolically before this was written:
    evaluating the existing map at an effective momentum reproduces the drive term
    ``G = h - k0/(1 + delta)`` exactly but scales the weak-focusing ``h^2`` along with the
    gradient, leaving ``h^2 delta_t/(1 + delta)`` — ``1.2e-4`` relative on ``K_x``, a
    hundred times the gate. A bend's horizontal focusing is ``k0 h + k1``: a **product**
    of field and geometry, of which only one factor tapers.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    if profile is None:
        profile = taper_profile(lattice)
    out = _apply(lattice, profile.delta)
    for _ in range(rounds - 1):
        out = _apply(lattice, _orbit_ramp(out))
    return out
