r"""Q1 — the taper profile: the sawtooth momentum a radiating ring actually runs at.

A ring that radiates hands its beam back the turn's energy in one lump at the cavity, so
the momentum falls through every magnet and jumps once. The height of that sawtooth is
conservation and nothing else — it is ``U_0 / (beta0^2 E_0)`` whatever is wrong elsewhere —
so the *height* gates almost nothing. **Where the sawtooth is centred is the physics**, and
a profile running ``0 -> -U_0/E_0`` has exactly the same span as the right one while being
wrong by half the effect in every magnet. That is why the mean is a separate gate from the
span, and why it is asserted against round-off rather than with ``approx(rel=)``, which
P2 (i) showed is vacuous on a quantity whose true value is zero.

**The independent pair.** ``taper_profile`` is built by forward tracking with the cavity
stepped over. I4's :func:`~accsim.orbit.closed_orbit_6d` reaches the same object from the
opposite direction — a fixed point of the *whole* turn, cavity included — and neither is
used to build the other, so their agreement is a gate. It is a sharp one and it has a
closed form: the two centrings differ by **exactly half a magnet's share of the loss**,
``span / (2 N_bends)``, confirmed over a factor four in ``N_bends``; the *shape* residual
after that constant is removed falls off as ``1/N^2`` and is ``1.6e-6`` of the span on the
1.6 GeV rings of that sweep (``1.05e-4`` on I4's own 6.5 GeV one, first order in the sag).

The fixture is I4's ring throughout, deliberately. At 6.5 GeV it runs ``U_0/E_0 =
3.8161e-3``, and the milestone's first probe — an 8-cell 1 GeV toy at ``6.9e-5`` — was
abandoned because at that size a 30% coefficient error is invisible against a reference
gate, which is P2 (iv)'s lesson from the other side.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_closed_orbit_6d import ring  # noqa: E402  (sibling fixture, as I4's own tests do)

from accsim.coords import DELTA
from accsim.elements.rfcavity import RFCavity
from accsim.orbit import closed_orbit_6d
from accsim.radiation import energy_loss_per_turn
from accsim.tapering import taper_profile

#: The design-route integral is evaluated on the *design* orbit; the profile is tracked
#: along a trajectory that is centred in momentum but still oscillates transversely, which
#: is xtrack's launch convention too. The gap between them is this, and it is measured.
SPAN_EXCESS = 1.9176e-4


def _scale(lattice) -> float:
    """eV per unit ``delta``: ``beta0^2 E_0``, the same conversion I4's tests use."""
    return lattice.ref.beta0**2 * lattice.ref.total_energy_eV


def _six_d_ramp(lattice) -> np.ndarray:
    """The momentum at every element's midpoint on I4's 6D closed orbit.

    Deliberately built here, in the test, from ``closed_orbit_6d`` — the module under test
    never calls it, which is what makes the comparison below a gate instead of a tautology.
    """
    state = closed_orbit_6d(lattice, radiation="mean").copy()
    edges = [float(state[DELTA])]
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref, radiation="mean")
        edges.append(float(state[DELTA]))
    edges = np.asarray(edges)
    return 0.5 * (edges[:-1] + edges[1:])


def _not_cavity(lattice) -> np.ndarray:
    return np.array([not isinstance(e, RFCavity) for e in lattice.elements])


# ---------------------------------------------------------------------------
# Gate 1 — the height is the turn's energy loss, and the residual is named.
# ---------------------------------------------------------------------------


def test_the_span_is_the_turns_energy_loss() -> None:
    """The sawtooth's peak-to-peak height is ``U_0 / (beta0^2 E_0)``.

    ``energy_loss_per_turn`` is a design-route radiation integral — a different calculation
    entirely, sharing no code with the tracked ramp — so this is two arms meeting, not an
    identity. They meet to ``1.9e-4``, and that number is *pinned rather than tolerated*:
    it is the trajectory difference (design orbit vs the tracked one), the same first-order
    gap I4 measured as ``4.4e-3`` about the design orbit and ``1.7e-7`` about the closed one.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    expected = energy_loss_per_turn(lattice) / _scale(lattice)

    assert profile.span == pytest.approx(expected, rel=3e-4)
    assert profile.span / expected - 1.0 == pytest.approx(SPAN_EXCESS, rel=0.05)


def test_the_loss_the_profile_reports_is_the_span_in_electronvolts() -> None:
    """``loss_eV`` and ``span`` are the same statement in two units."""
    lattice, _ = ring()
    profile = taper_profile(lattice)
    assert profile.loss_eV == pytest.approx(profile.span * _scale(lattice), rel=1e-14)


# ---------------------------------------------------------------------------
# Gate 2 — the centring, which is the milestone.
# ---------------------------------------------------------------------------


def test_the_profile_is_centred_to_round_off() -> None:
    """The length-weighted mean is zero — asserted absolutely, against the span.

    ``approx(rel=)`` on a quantity whose true value is zero passes for any input, so the
    scale has to come from somewhere else. Here it is the span: the mean must be below
    ``1e-12`` of the sawtooth's own height. Measured: ``3e-17`` of it.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    assert abs(profile.mean) < 1e-12 * profile.span


def test_an_uncentred_profile_misses_by_half_the_span() -> None:
    """The deliberate break: ``delta0=0`` starts the ramp at the design momentum.

    Its mean is half the span low, which is the whole reason the mean is a separate gate.
    Its span is **not** quite identical, and the departure is the milestone's own physics
    rather than slack: a ramp that runs entirely below the design momentum radiates less,
    because the power goes as ``E^2``. First order that predicts ``-span``; the measured
    ``-1.20 span`` carries the trajectory's share on top, so it is pinned, not tolerated.
    """
    lattice, _ = ring()
    centred, raw = taper_profile(lattice), taper_profile(lattice, delta0=0.0)

    assert raw.span / centred.span - 1.0 == pytest.approx(-1.198 * centred.span, rel=0.05)
    assert raw.mean / raw.span == pytest.approx(-0.5, rel=2e-3)


def test_the_centring_converges_geometrically() -> None:
    """Each re-centring round divides the mean by a constant, and the constant is measured.

    The profile depends on its own starting momentum only through the radiated power's
    ``E^2``, so this is a contraction rather than a solve: no Newton, no Jacobian, and the
    ratio is a property of the ring. Gating the *ratio* rather than a final tolerance is
    J2's rule — an implementation that reached zero mean some other way would not have it.
    """
    lattice, _ = ring()
    means = [abs(taper_profile(lattice, rounds=r).mean) for r in (1, 2, 3, 4)]
    ratios = [b / a for a, b in zip(means[:-1], means[1:], strict=True)]

    assert all(r < 1e-2 for r in ratios)
    assert ratios[1] == pytest.approx(ratios[0], rel=0.2)


def test_the_starting_momentum_is_half_the_span_above_the_design():
    """A centred ramp that falls linearly starts half a span high — and this ring's does.

    The departure from exactly ``span/2`` is the lattice's structure: the momentum is spent
    in the bends and held flat across the quadrupoles, so the *length-weighted* centre is
    not the arithmetic one. It is a fraction of a percent here, which is the same
    ``1/(2 N_bends)`` discretisation that separates this profile from I4's below.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    assert profile.delta_start / profile.span == pytest.approx(0.5, rel=2e-3)


# ---------------------------------------------------------------------------
# Gate 3 — a magnet cannot hand energy back.
# ---------------------------------------------------------------------------


def test_the_momentum_falls_in_every_magnet_and_is_flat_everywhere_else() -> None:
    """Monotone through the radiating elements, and **exactly** flat across the thin ones.

    ``== 0.0`` rather than a tolerance: a zero-length element has no path to radiate over,
    so this is an identity of the model, not a numerical near-miss.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    step = np.diff(profile.boundary)

    assert np.all(step[profile.lengths > 0.0] < 0.0)
    assert np.max(np.abs(step[profile.lengths == 0.0])) == 0.0


def test_the_thin_elements_carry_no_weight_in_the_mean() -> None:
    """The mean is length-weighted, so it does not move when the lattice is re-written.

    Splitting a thin quadrupole into two halves at the same place changes the element list
    and changes nothing physical; an unweighted mean would move.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    assert profile.lengths[profile.lengths == 0.0].sum() == 0.0
    assert profile.mean == pytest.approx(
        float(np.dot(profile.delta, profile.lengths) / profile.lengths.sum()), rel=1e-14
    )


# ---------------------------------------------------------------------------
# Gate 4 — the uniform-ring closed form.
# ---------------------------------------------------------------------------


def test_the_sawtooth_is_a_straight_line_in_s_with_the_predicted_slope() -> None:
    """``d(delta)/ds = -U_0 / (beta0^2 E_0 C)`` on a ring whose bends are all alike.

    The departure from the straight line is not error: the momentum is spent only inside
    the bends, so the true profile is a staircase and the line is its envelope. That
    departure is ``7.0e-4`` of the span here and is bounded, not tolerated.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    thick = profile.lengths > 0.0

    slope, intercept = np.polyfit(profile.s_mid[thick], profile.delta[thick], 1)
    predicted = -energy_loss_per_turn(lattice) / _scale(lattice) / lattice.length
    assert slope == pytest.approx(predicted, rel=3e-4)

    residual = profile.delta[thick] - (slope * profile.s_mid[thick] + intercept)
    assert np.max(np.abs(residual)) / profile.span == pytest.approx(7.0e-4, rel=0.1)


# ---------------------------------------------------------------------------
# Gate 5 — the order gate: the span is the fourth power of the energy, over the first.
# ---------------------------------------------------------------------------


def test_the_span_scales_as_the_cube_of_the_energy() -> None:
    """``U_0 ~ E^4`` and the span is ``U_0/E``, so the exponent is **3**.

    Gated on the exponent rather than on a value, which is J2's rule: a uniformly
    mis-scaled profile would keep the exponent and a wrong power law would not, and only
    the second is the kind of error this can catch.
    """
    energies = np.array([1.625e9, 3.25e9, 6.5e9, 13.0e9])
    spans = [taper_profile(ring(energy=e, voltage=400.0e6)[0]).span for e in energies]

    exponent, _ = np.polyfit(np.log(energies), np.log(spans), 1)
    assert exponent == pytest.approx(3.0, abs=2e-3)


# ---------------------------------------------------------------------------
# Gate 6 — the independent pair: I4's 6D fixed point reaches the same profile.
# ---------------------------------------------------------------------------


def test_the_six_dimensional_closed_orbit_reaches_the_same_shape() -> None:
    """Two routes to one object, and the constant between them has a closed form.

    ``taper_profile`` tracks forward with the cavity stepped over and *imposes* zero mean.
    ``closed_orbit_6d`` solves the whole turn's fixed point, cavity included, and *arrives*
    at its own centring from periodicity. They differ by a constant — exactly half a
    magnet's share of the loss, ``span/(2 N_bends)`` — and after that constant is removed
    the shapes agree to ``1.05e-4`` of the span here. That residual is itself first order in
    the sag (``1.6e-6`` on the 1.6 GeV rings the sweep below uses, a factor 64 down for a
    factor 64 in span), which is why the sharp form of the claim lives in that sweep.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    keep = _not_cavity(lattice)
    weights = profile.lengths[keep]

    gap = _six_d_ramp(lattice)[keep] - profile.delta[keep]
    offset = float(np.dot(gap, weights) / weights.sum())
    bends = int(np.count_nonzero(profile.lengths[keep]))

    # The departure from an exact half-step is first order in the sag itself — measured
    # at ``5.4 x span``, so ``2.0e-2`` here at ``span = 3.8e-3`` and ``3.2e-4`` on the
    # 1.6 GeV rings the sweep below uses. The sharp ``1/N`` claim lives there, at the
    # energy where this correction is negligible; this one only has to see the half-step.
    assert offset / profile.span == pytest.approx(-1.0 / (2 * bends), rel=3e-2)
    assert np.max(np.abs(gap - offset)) / profile.span < 2e-4


@pytest.mark.parametrize("cells", [10, 20, 40])
def test_the_offset_from_the_six_dimensional_orbit_is_half_a_magnets_share(cells: int) -> None:
    """``span/(2 N_bends)`` over a factor four in the number of magnets.

    A constant offset that happened to be right on one ring would be a coincidence; one
    that tracks ``1/N`` is the discretisation it is claimed to be. The shape residual falls
    as ``1/N^2`` on the same sweep, which is the second half of the same statement.
    """
    lattice, _ = ring(cells=cells, energy=1.625e9, voltage=400.0e6, harmonic=cells)
    profile = taper_profile(lattice)
    keep = _not_cavity(lattice)
    weights = profile.lengths[keep]

    gap = _six_d_ramp(lattice)[keep] - profile.delta[keep]
    offset = float(np.dot(gap, weights) / weights.sum())
    bends = int(np.count_nonzero(profile.lengths[keep]))

    assert offset / profile.span == pytest.approx(-1.0 / (2 * bends), rel=2e-3)
    assert np.max(np.abs(gap - offset)) / profile.span < 1e-5


# ---------------------------------------------------------------------------
# Gate 7 — the sag orbit, which is the profile's consequence and Q2's target.
# ---------------------------------------------------------------------------


def test_the_profile_drives_a_millimetre_scale_orbit() -> None:
    """The point of the axis: a ``3.8e-3`` momentum sawtooth is a **7 mm** orbit.

    The excursion is the profile through the ring's dispersion, so it is large where no
    misalignment ever was, and it is what Q2 will remove. Both reference codes produce it
    independently (``7.0898e-3`` xtrack, ``7.0882e-3`` MAD-X); this is accsim's own.
    """
    lattice, _ = ring()
    state = closed_orbit_6d(lattice, radiation="mean").copy()
    excursion = [abs(float(state[0]))]
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref, radiation="mean")
        excursion.append(abs(float(state[0])))

    assert max(excursion) == pytest.approx(7.09e-3, rel=0.02)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_a_pinned_starting_momentum_is_honoured_exactly() -> None:
    """``delta0`` as a number is a pin, not a hint — no re-centring runs."""
    lattice, _ = ring()
    assert taper_profile(lattice, delta0=1e-3).delta_start == 1e-3


def test_an_unknown_delta0_keyword_is_refused() -> None:
    lattice, _ = ring()
    with pytest.raises(ValueError, match="zero_mean"):
        taper_profile(lattice, delta0="centre")


def test_the_number_of_entries_matches_the_lattice() -> None:
    lattice, _ = ring()
    profile = taper_profile(lattice)
    assert profile.delta.shape == (len(lattice.elements),)
    assert profile.boundary.shape == (len(lattice.elements) + 1,)
    assert profile.s_mid.shape == profile.delta.shape
    assert profile.s_mid[-1] == pytest.approx(lattice.length - 0.5 * profile.lengths[-1])


def test_the_ramp_starts_exactly_where_it_was_pinned() -> None:
    """A pinned ``delta0`` is the ramp's first boundary value, bit for bit.

    The boundary array is the raw ramp — no averaging, no centring — so this is the one
    place the contract can be checked with ``==`` rather than a tolerance.
    """
    lattice, _ = ring()
    assert taper_profile(lattice, delta0=0.0).boundary[0] == 0.0
    assert taper_profile(lattice, delta0=-2.5e-3).boundary[0] == -2.5e-3
