r"""Q2 — applying the taper: a bending magnet whose field is not its geometry.

Q1 computed the sawtooth a radiating ring runs at. This file is what happens when it is
*applied*: every magnet's field scaled by ``1 + delta(s)`` so the beam at ``delta(s)``
feels the nominal magnet again, and the millimetre-wide dispersion orbit the sag inflicts
goes away.

**The central object is one number splitting into two.** A :class:`~accsim.Dipole` stored
``angle``, so its field and its curvature *were the same number*, ``k0 = h = angle/L``. A
tapered bend has ``k0 = h(1 + delta_t)`` with ``h`` **unchanged** — the geometry belongs to
the ring (the bends must still sum to ``2 pi``) and the field belongs to the beam. Every
other magnet in the package stores a strength and tapering it is arithmetic; this one
needed surgery, on the element that also carries F2, L3, L4, P2 (i), P3 (a) and P3 (b).

**What made that surgery small is an exact symmetry, and it is worth stating because it
is the reason none of those six milestones moved.** Scale every field in a magnet by
``s = 1 + t`` and leave the geometry alone. Writing ``px = s P``, ``py = s Q`` and
``1 + delta = s (1 + delta')``, the bend Hamiltonian obeys

    H_tapered(x, px, y, py; delta)  =  s * H_nominal(x, P, y, Q; delta')

because ``s`` comes out of the square root and out of the whole vector potential together.
The ``s`` cancels in Hamilton's equations, so **a tapered magnet maps ``(x, px, y, py)``
exactly as the design magnet maps ``(x, P, y, Q)``** — not to leading order, exactly. So
the shipped map *is* the design map with rescaled momenta, and the exact circle, the
expanded combined-function body, the pole faces, the fringes and the wedges all come
through untouched.

The one row the symmetry does not carry is ``zeta``. The two magnets share a *trajectory*,
so they share its path length ``P``, but ``zeta = s - beta_0 c t`` turns that path into a
time using the particle's own speed and the design magnet was handed the wrong momentum
for it. ``dzeta = L - (beta_0/beta) P`` then fixes the row with no freedom left.

**Because the map is built from the symmetry, an internal check of the symmetry would be
vacuous.** The gate that is not is
:func:`test_the_tapered_bend_agrees_with_a_cartesian_lorentz_integration`: the Lorentz
force integrated in the laboratory frame, the exit face located by plane geometry, and the
arrival time from the path length — sharing no line of code and no convention with
``accsim``. It is run at ``gamma = 1.6`` as well as ``gamma = 21``, because the ``zeta``
correction is ``O(t/gamma^2)`` and on I4's 6.5 GeV electron ring that is ``1e-11`` of the
length: gating this milestone's ``zeta`` row on I4 would be Q1's abandoned 1 GeV probe all
over again, from the other side.

**At ring level the result is a law rather than a tolerance**, which is the only thing
that separates a correct taper from a mis-scaled one — a taper wrong by a coefficient
leaves a residual *first* order in the sag, and no amount of iterating removes it:

===========  ========================  ===========================
rounds       residual ``|x|`` on I4    the measured law
===========  ========================  ===========================
0            ``7.090e-3`` m            ``1.858 * span``
1            ``6.828e-6`` m            ``0.469 * span^2``
2            ``5.506e-9`` m            ``0.104 * span^3``
3            ``1.00e-10`` m            the closed-orbit solver's floor
===========  ========================  ===========================

Each round removes one power. That is what
:func:`test_each_round_of_tapering_removes_one_power_of_the_sag` gates, over a factor
eight in span, and the exponents are what is asserted (J2's rule) rather than the
constants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from test_closed_orbit_6d import ring  # noqa: E402  (sibling fixture, as Q1's own tests do)

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    closed_orbit,
    closed_twiss,
    energy_loss_per_turn,
    natural_chromaticity,
    taper,
    taper_profile,
    tunes,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y
from accsim.symplectic import unit_symplectic_matrix
from accsim.tapering import TaperProfile

PROTON_MASS_EV = 938.27208816e6
ELECTRON_MASS_EV = 0.51099895069e6

#: The energies I4's ring is swept over here. A factor 8 in the span, which is what turns
#: "the residual is small" into "the residual is second order in the sag".
ENERGIES = (3.25e9, 4.6e9, 6.5e9)

#: The number of bends in I4's ring — 20 cells of two.
N_BENDS = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def orbit_excursion(lattice: Lattice) -> float:
    """``max |x|`` over the element boundaries, on the ring's radiating closed orbit."""
    from accsim import closed_orbit_6d

    state = closed_orbit_6d(lattice, radiation="mean").copy()
    xs = [float(state[X])]
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref, radiation="mean")
        xs.append(float(state[X]))
    return float(np.max(np.abs(xs)))


def scaled_profile(profile: TaperProfile, factor: float) -> TaperProfile:
    """The same profile with every ``delta`` multiplied — the deliberate-break knob."""
    return TaperProfile(
        delta=profile.delta * factor,
        boundary=profile.boundary * factor,
        s_mid=profile.s_mid,
        lengths=profile.lengths,
        delta_start=profile.delta_start * factor,
        loss_eV=profile.loss_eV,
    )


def cartesian_bend(
    state: np.ndarray, length: float, h: float, k0: float, ref: ReferenceParticle
) -> np.ndarray:
    r"""``(x, px, y, py, dzeta)`` of a tapered bend, by direct Lorentz integration.

    The arbiter for the whole ``k0 != h`` map, and it is deliberately built from nothing
    the element uses. In the laboratory frame a uniform vertical field turns the momentum
    at ``dP/dsigma = -k0 * PZ/p`` with ``sigma`` the **path length**; the design arc of
    curvature ``h`` fixes where the exit face is (a plane through
    ``(-(1-cos)/h, sin/h)`` normal to the turned direction) and what its transverse axis
    is. ``zeta = s - beta_0 c t`` then follows from the path length alone,
    ``dzeta = L - (beta_0/beta) sigma``.

    The bend is toward **-x** in this frame: accsim's ``x`` is measured outward from the
    design arc's centre, which is the convention that makes ``R16 > 0``.
    """
    x0, px0, y0, py0, _, delta = (float(v) for v in state)
    p = 1.0 + delta

    def rhs(sigma: float, u: np.ndarray) -> list[float]:
        _, _, _, pxs, pys = u
        pz = math.sqrt(max(p * p - pxs * pxs - pys * pys, 0.0))
        return [pxs / p, pys / p, pz / p, -k0 * pz / p, 0.0]

    theta = h * length
    end = (-(1.0 - math.cos(theta)) / h, math.sin(theta) / h)
    tangent = (-math.sin(theta), math.cos(theta))
    face = (math.cos(theta), math.sin(theta))

    def hit_exit_face(sigma: float, u: np.ndarray) -> float:
        return (u[0] - end[0]) * tangent[0] + (u[2] - end[1]) * tangent[1]

    hit_exit_face.terminal = True
    hit_exit_face.direction = 1.0
    sol = solve_ivp(
        rhs,
        (0.0, 1.6 * length),
        [x0, y0, 0.0, px0, py0],
        events=hit_exit_face,
        rtol=1e-13,
        atol=1e-16,
    )
    sigma = float(sol.t_events[0][0])
    xs, ys, zs, pxs, pys = (float(v) for v in sol.y_events[0][0])
    pzs = math.sqrt(max(p * p - pxs * pxs - pys * pys, 0.0))
    energy = math.hypot(ref.momentum_eV * p, ref.mass_eV)
    beta0_over_beta = (energy / ref.total_energy_eV) / p
    return np.array(
        [
            (xs - end[0]) * face[0] + (zs - end[1]) * face[1],
            pxs * face[0] + pzs * face[1],
            ys,
            pys,
            length - beta0_over_beta * sigma,
        ]
    )


def tracked_jacobian(element: Dipole, base: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """Central-difference Jacobian of ``element.track`` at ``base``."""
    out = np.zeros((DIM, DIM))
    step = 1.0e-7
    for j in range(DIM):
        lo, hi = base.copy(), base.copy()
        hi[j] += step
        lo[j] -= step
        out[:, j] = (element.track(hi, ref) - element.track(lo, ref)) / (2.0 * step)
    return out


def canonical_symplectic_error(matrix: np.ndarray, delta: float, ref: ReferenceParticle) -> float:
    r"""``|M^T J M - J|`` with the longitudinal pair made canonically conjugate.

    ``(zeta, delta)`` is not a canonical pair — ``(zeta, p_zeta)`` is — and the two differ
    by the diagonal factor ``ddelta/dp_zeta``, which is exactly ``1`` at ``delta = 0`` and
    not at any other momentum. A tapered magnet's matrix lives at ``delta = t``, so the
    raw check is the wrong one for it; see
    :func:`test_the_raw_symplectic_check_fails_on_a_tapered_matrix_by_one_over_gamma_squared`.
    """
    unit = unit_symplectic_matrix()
    p0, e0, mass = ref.momentum_eV, ref.total_energy_eV, ref.mass_eV
    energy = math.hypot(p0 * (1.0 + delta), mass)
    scale = energy * ref.beta0**2 * e0 / (p0 * p0 * (1.0 + delta))
    change = np.eye(DIM)
    change[DELTA, DELTA] = scale
    conjugated = np.linalg.solve(change, matrix @ change)
    return float(np.abs(conjugated.T @ unit @ conjugated - unit).max())


@pytest.fixture(scope="module")
def residuals() -> dict[float, dict[str, float]]:
    """``{energy: {span, untapered, round1, round2}}`` — every closed-orbit solve here.

    Module-scoped because each entry is a 6D Newton solve on a hundred-element ring and
    four of the tests below read the same table.
    """
    table: dict[float, dict[str, float]] = {}
    for energy in ENERGIES:
        lattice, _ = ring(energy=energy)
        profile = taper_profile(lattice)
        table[energy] = {
            "span": profile.span,
            "untapered": orbit_excursion(lattice),
            "round1": orbit_excursion(taper(lattice, profile, rounds=1)),
            "round2": orbit_excursion(taper(lattice, profile, rounds=2)),
            "round3": orbit_excursion(taper(lattice, profile, rounds=3)),
        }
    return table


def power_law(spans: list[float], values: list[float]) -> float:
    """The fitted exponent of ``value ~ span^n``."""
    return float(np.polyfit(np.log(spans), np.log(values), 1)[0])


# ---------------------------------------------------------------------------
# The element: a field that is not the geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma_tag, energy", [("low", 1.5e9), ("high", 20.0e9)])
@pytest.mark.parametrize("taper_t", [0.0, 4.0e-3, -0.05])
@pytest.mark.parametrize(
    "state",
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0e-3, 2.0e-3, -1.5e-3, 8.0e-4, 0.0, 3.0e-3],
        [-4.0e-3, 1.0e-3, 2.0e-3, -2.0e-3, 0.0, -1.0e-2],
    ],
)
def test_the_tapered_bend_agrees_with_a_cartesian_lorentz_integration(
    gamma_tag: str, energy: float, taper_t: float, state: list[float]
) -> None:
    """The one gate here that shares nothing with the implementation.

    The shipped map is built from the tapering symmetry, so checking the symmetry against
    itself would prove nothing. This integrates the Lorentz force in the laboratory frame
    instead and finds the exit face by plane geometry — no curvilinear coordinates, no
    ``accsim`` convention, and the arrival time from the path length.

    Run at ``gamma = 1.6`` as well as ``gamma = 21`` **because of the last row**: the
    ``zeta`` correction a taper needs is ``O(t/gamma^2)``, invisible on an ultrarelativistic
    ring, and a fixture that cannot see it would pass a map with the row left out. The
    ``-5%`` taper is far outside anything a real ring needs, for the same reason.
    """
    ref = ReferenceParticle.from_total_energy(PROTON_MASS_EV, energy)
    h, length = 0.3, 1.0
    bend = Dipole(length, h * length, k0=h * (1.0 + taper_t))
    got = bend.track(np.array(state, dtype=float), ref)
    want = cartesian_bend(np.array(state, dtype=float), length, h, bend.k0, ref)
    assert got[X] == pytest.approx(want[0], abs=1.0e-12)
    assert got[PX] == pytest.approx(want[1], abs=1.0e-12)
    assert got[Y] == pytest.approx(want[2], abs=1.0e-12)
    assert got[PY] == pytest.approx(want[3], abs=1.0e-12)
    assert got[ZETA] == pytest.approx(want[4], abs=1.0e-12)


@pytest.mark.parametrize("k1", [0.0, 0.3])
def test_a_matched_particle_leaves_a_tapered_magnet_on_the_axis(k1: float) -> None:
    r"""The physical content of tapering, in one line.

    A magnet tapered for ``delta_t`` presents the *design* magnet to a particle at
    ``delta_t``: entering on the axis it leaves on the axis, with no deflection at all.
    An untapered magnet does not — the same particle is over-rigid for it and comes out
    displaced by the magnet's own dispersion, which is the ``7 mm`` this milestone exists
    to remove.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    h, t = 0.157, 4.0e-3
    tapered = Dipole(1.0, h, k1 * (1.0 + t), k0=h * (1.0 + t))
    matched = np.array([0.0, 0.0, 0.0, 0.0, 0.0, t])

    out = tapered.track(matched, ref)
    assert abs(out[X]) < 1.0e-16
    assert abs(out[PX]) < 1.0e-16

    design = Dipole(1.0, h, k1)
    stray = design.track(matched, ref)
    assert abs(stray[X]) > 1.0e-6  # the dispersion of a magnet set for the wrong beam


def test_the_matrix_is_the_tracked_jacobian_at_the_momentum_it_is_tapered_for() -> None:
    r"""``matrix()`` is the origin Jacobian of ``track()`` — at ``delta = t``, not ``0``.

    A tapered magnet's linear map is the map it presents to the beam it was tapered for,
    and that is a statement rather than a convenience: it is the expansion point at which
    the momentum rescaling is the identity on ``delta``. The departure at ``delta = 0`` is
    **first order in the taper** and is measured here rather than hidden — it is the
    magnet being genuinely mis-set for the design particle, which is the whole reason the
    taper exists.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    h, t = 0.157, 4.0e-3
    bend = Dipole(1.0, h, 0.3 * (1.0 + t), k0=h * (1.0 + t))
    matrix = bend.matrix(ref)

    at_taper = np.array([0.0, 0.0, 0.0, 0.0, 0.0, t])
    assert np.abs(tracked_jacobian(bend, at_taper, ref) - matrix).max() < 1.0e-10
    # and the affine model (matrix + kick) reproduces the tracked map there exactly
    assert np.abs(matrix @ at_taper + bend.kick(ref) - bend.track(at_taper, ref)).max() < 1.0e-15

    # The departure at delta = 0 is *first order in the taper* — the magnet being
    # genuinely mis-set for the design particle. Measured over a factor eight in t, where
    # a constant offset or a second-order term would both show up in the ratio.
    slopes = []
    for small in (1.0e-3, 8.0e-3):
        probe = Dipole(1.0, h, 0.3 * (1.0 + small), k0=h * (1.0 + small))
        gap = np.abs(tracked_jacobian(probe, np.zeros(DIM), ref) - probe.matrix(ref)).max()
        slopes.append(gap / small)
    assert slopes[0] == pytest.approx(slopes[1], rel=0.01)
    assert slopes[0] == pytest.approx(1.10, rel=0.02)


def test_the_matrix_is_symplectic_in_canonical_coordinates() -> None:
    """Exactly symplectic once ``delta`` is traded for its canonical partner.

    The similarity ``S M S^-1`` that builds a tapered matrix scales the whole symplectic
    form by one factor, so it preserves symplecticity for free — and would do so with the
    ``zeta`` row completely wrong. What this gate actually catches is the ``zeta`` row's
    clock ratio ``r``: see the deliberate break below.
    """
    for energy, mass in ((6.5e9, ELECTRON_MASS_EV), (1.5e9, PROTON_MASS_EV)):
        ref = ReferenceParticle.from_total_energy(mass, energy)
        h, t = 0.157, 4.0e-3
        for k1 in (0.0, 0.3):
            bend = Dipole(1.0, h, k1 * (1.0 + t), 0.04, -0.03, k0=h * (1.0 + t))
            assert canonical_symplectic_error(bend.matrix(ref), t, ref) < 1.0e-14


def test_the_raw_symplectic_check_fails_on_a_tapered_matrix_by_one_over_gamma_squared() -> None:
    r"""And the failure is the coordinates, not the map — so it is asserted, not fixed.

    ``(zeta, delta)`` is canonically conjugate only where ``ddelta/dp_zeta = 1``, which is
    ``delta = 0`` exactly. Every untapered element in this package is expanded there, so
    the raw check has always worked; a tapered magnet is expanded at ``delta = t`` and the
    raw residual is ``O(t/gamma^2)`` — invisible on I4's electron ring and four orders
    larger on a slow proton. If a future change made the raw check pass at ``gamma = 1.6``,
    the ``zeta`` row would have been flattened and this test would say so.
    """
    unit = unit_symplectic_matrix()
    h, t = 0.157, 4.0e-3
    seen = {}
    for energy, mass, tag in ((6.5e9, ELECTRON_MASS_EV, "fast"), (1.5e9, PROTON_MASS_EV, "slow")):
        ref = ReferenceParticle.from_total_energy(mass, energy)
        matrix = Dipole(1.0, h, 0.0, k0=h * (1.0 + t)).matrix(ref)
        seen[tag] = float(np.abs(matrix.T @ unit @ matrix - unit).max()) * ref.gamma0**2

    assert seen["fast"] == pytest.approx(seen["slow"], rel=0.05), (
        "the raw residual must scale as 1/gamma^2 — that is what makes it the coordinates"
    )
    assert seen["fast"] / t == pytest.approx(0.156, rel=0.05)


def test_the_effective_momentum_shortcut_is_refuted() -> None:
    r"""Tapering is **not** the untapered map read at a shifted momentum, and by how much.

    The shortcut reproduces the drive ``G = h - k0/(1 + delta)`` exactly, which is why it
    looks right, and then scales the weak focusing ``h^2`` along with the gradient. A
    bend's horizontal focusing is ``K_x = k0 h + k1`` — a *product* of field and geometry,
    of which only the field tapers — so the shortcut leaves ``h^2 t``, and the number is
    pinned here rather than argued.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    h, t, k1_nominal = 0.157, 4.0e-3, 0.3

    shipped = Dipole(1.0, h, k1_nominal * (1.0 + t), k0=h * (1.0 + t))
    shortcut = Dipole(1.0, h, k1_nominal * (1.0 + t))  # gradient tapered, field left at h

    focusing_true = shipped.k0 * h + shipped.k1
    focusing_shortcut = h * h + shortcut.k1
    assert focusing_true - focusing_shortcut == pytest.approx(h * h * t, rel=1.0e-12)
    assert focusing_true - focusing_shortcut == pytest.approx(9.86e-5, rel=0.01)

    # And what that costs, which is the part worth having: the magnet stops presenting the
    # design magnet to the beam it was tapered for. 0.3 mm across one metre of bend,
    # against the shipped magnet's 1e-17.
    matched = np.array([0.0, 0.0, 0.0, 0.0, 0.0, t])
    assert abs(shortcut.track(matched, ref)[X]) == pytest.approx(3.04e-4, rel=0.02)
    assert abs(shipped.track(matched, ref)[X]) < 1.0e-16


def test_an_untapered_dipole_is_bit_for_bit_the_magnet_it_was() -> None:
    """The guard axes B and N need: ``k0 == h`` must take the original path, exactly.

    Radiation and spin read ``normalized_field``, which now returns ``k0``. Nothing on
    those axes may move, so a dipole given its own curvature explicitly must be
    indistinguishable from one that was never given a ``k0`` at all — in the matrix, the
    kick, the tracked map and the field.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    plain = Dipole(1.0, 0.2, 0.3, 0.05, -0.02, fringe=True)
    explicit = Dipole(1.0, 0.2, 0.3, 0.05, -0.02, k0=0.2, fringe=True)
    assert not plain.is_tapered and not explicit.is_tapered
    assert plain.field_ratio == 1.0 and plain.taper == 0.0

    state = np.array([1.0e-3, 2.0e-3, -1.0e-3, 1.5e-3, 1.0e-4, 3.0e-3])
    assert np.array_equal(plain.matrix(ref), explicit.matrix(ref))
    assert np.array_equal(plain.kick(ref), explicit.kick(ref))
    assert np.array_equal(plain.track(state, ref), explicit.track(state, ref))
    assert np.array_equal(plain.kick(ref), np.zeros(DIM))
    assert plain.normalized_field(1.0e-3, 2.0e-3) == explicit.normalized_field(1.0e-3, 2.0e-3)


def test_the_field_a_tapered_magnet_radiates_by_is_its_own() -> None:
    r"""``normalized_field`` returns ``k0``, and this is the seam the symmetry misses.

    Radiation is applied in ``track`` *outside* the body map, so the taper reaches it only
    through this method. Were it left as the curvature, the tapered ring would be set for
    a loss it is not taking and the orbit gate below would be measuring nothing.
    """
    h, t = 0.157, 4.0e-3
    bend = Dipole(1.0, h, 0.3, k0=h * (1.0 + t))
    _, by = bend.normalized_field(0.0, 0.0)
    assert by == pytest.approx(h * (1.0 + t), rel=1.0e-15)
    assert by != pytest.approx(bend.curvature, rel=1.0e-9)


def test_a_field_without_a_geometry_is_refused() -> None:
    """``angle = 0`` with a field is a steering magnet, and the ratio ``k0/h`` is its ruin.

    The whole construction is the ratio of the field to the geometry it bends in. At
    ``h = 0`` there is no ratio, so rather than pick a branch the element says so.
    """
    with pytest.raises(NotImplementedError, match="steering magnet"):
        Dipole(1.0, 0.0, k0=0.1)
    Dipole(1.0, 0.0, 0.3)  # a straight gradient magnet is still fine


# ---------------------------------------------------------------------------
# The ring: the sag orbit, and what a taper does to it
# ---------------------------------------------------------------------------


def test_the_untapered_sag_orbit_is_first_order_in_the_span(
    residuals: dict[float, dict[str, float]],
) -> None:
    """``7.09 mm`` on I4, and it is ``1.86 * span`` at every energy.

    The anchor for everything below: the distortion a taper has to remove is *first* order
    in the sag, so anything second order in it is a residual rather than an error.
    """
    spans = [residuals[e]["span"] for e in ENERGIES]
    values = [residuals[e]["untapered"] for e in ENERGIES]
    assert power_law(spans, values) == pytest.approx(1.0, abs=0.01)
    for span, value in zip(spans, values, strict=True):
        assert value / span == pytest.approx(1.858, rel=0.01)
    assert residuals[6.5e9]["untapered"] == pytest.approx(7.090e-3, rel=1.0e-3)


def test_each_round_of_tapering_removes_one_power_of_the_sag(
    residuals: dict[float, dict[str, float]],
) -> None:
    r"""The milestone's sharpest gate, and it is an **exponent**, not a tolerance.

    A taper that is wrong by a coefficient — a stray ``1 + delta``, the ``h^2`` shortcut,
    the wrong sampling point along the magnet — leaves a residual *first* order in the
    sag, exactly like the distortion it was meant to remove, and no number of rounds
    changes that. What is measured instead is ``span^2`` after one round and ``span^3``
    after two, over a factor eight in span. J2's rule: gate the order.
    """
    spans = [residuals[e]["span"] for e in ENERGIES]
    assert power_law(spans, [residuals[e]["round1"] for e in ENERGIES]) == pytest.approx(
        2.0, abs=0.02
    )
    assert power_law(spans, [residuals[e]["round2"] for e in ENERGIES]) == pytest.approx(
        3.0, abs=0.08
    )
    for energy in ENERGIES:
        span = residuals[energy]["span"]
        # One round's constant is flat to four figures; two rounds' carries a visible
        # first-order-in-the-span correction of its own, which is why the exponent above
        # is the gate and these are the record.
        assert residuals[energy]["round1"] / span**2 == pytest.approx(0.4688, rel=0.005)
        assert residuals[energy]["round2"] / span**3 == pytest.approx(0.104, rel=0.06)


def test_the_taper_collapses_the_sag_orbit(
    residuals: dict[float, dict[str, float]],
) -> None:
    """The pre-committed gate: ``7.09 mm`` to below a micron on I4's own ring.

    One round reaches ``6.8 um`` and misses it; the second reaches ``5.5 nm`` and clears it
    by three orders; the third — the default — reaches ``1.0e-10`` m and stops, which is
    the closed-orbit solver's own floor and is where xtrack's own compensation lands
    (``9.70e-11``, cross-checked in ``tests/reference/``). That the *first* round misses is
    not a defect but the ``span^2`` law above.
    """
    at_i4 = residuals[6.5e9]
    assert at_i4["untapered"] > 7.0e-3
    assert at_i4["round1"] == pytest.approx(6.83e-6, rel=0.02)
    assert at_i4["round2"] < 1.0e-6, "the milestone's pre-committed micron"
    assert at_i4["round2"] == pytest.approx(5.51e-9, rel=0.05)
    assert at_i4["round3"] == pytest.approx(1.00e-10, rel=0.1)
    assert at_i4["untapered"] / at_i4["round3"] > 1.0e7


def test_a_one_percent_error_in_the_profile_leaves_one_percent_of_the_distortion() -> None:
    r"""The deliberate break, and it bites on the **one-shot** only.

    Scaling the profile by ``1.01`` mis-sets every magnet by ``1%`` of its own share of the
    sag, and the leftover orbit is ``1%`` of the untapered distortion — first order in the
    error, where the correct taper's residual is second order in the sag. It is asserted at
    ``rounds=1`` because the solve is *designed* to repair exactly this, which is the next
    test.
    """
    lattice, _ = ring()
    profile = taper_profile(lattice)
    untapered = orbit_excursion(lattice)
    broken = orbit_excursion(taper(lattice, scaled_profile(profile, 1.01), rounds=1))
    assert broken / untapered == pytest.approx(0.01, rel=0.05)


def test_the_solve_repairs_a_broken_profile() -> None:
    """Which is what makes it a solve rather than a formula.

    Round one is only a starting point: every round after it re-derives the profile from
    the tapered ring's own closed orbit, so a ``1%`` error in the profile it was handed
    survives one round and not two.
    """
    lattice, _ = ring()
    broken = scaled_profile(taper_profile(lattice), 1.01)
    assert orbit_excursion(taper(lattice, broken, rounds=1)) > 1.0e-5
    assert orbit_excursion(taper(lattice, broken, rounds=3)) < 1.0e-7


def test_applying_the_same_profile_twice_tapers_twice() -> None:
    r"""The trap the ``rounds`` argument exists to avoid, asserted so it stays avoided.

    Tapering does not reduce the sawtooth — the ring radiates exactly as much as before,
    and it is the *orbit* that goes away, not the ramp. So a caller who feeds the tapered
    ring back through :func:`taper` with a freshly computed profile applies the correction
    a second time and throws the entire correction away — the ring comes back to the
    distortion it started with, to a tenth of a percent.
    """
    lattice, _ = ring()
    once = taper(lattice, rounds=1)
    assert taper_profile(once).span == pytest.approx(taper_profile(lattice).span, rel=1.0e-3)

    twice = taper(once, rounds=1)
    untapered = orbit_excursion(lattice)
    assert orbit_excursion(once) < 1.0e-5
    # The whole correction is gone: the ring is as distorted as it started, having been
    # tapered once too often rather than not at all.
    assert orbit_excursion(twice) == pytest.approx(untapered, rel=0.01)
    assert orbit_excursion(twice) / orbit_excursion(once) > 1.0e3


def test_the_tapered_ring_still_radiates_the_same_energy() -> None:
    """Tapering removes a distortion, not a loss — the bends bend by the same angle.

    ``energy_loss_per_turn`` is the design-route integral over ``normalized_field``, so a
    ring whose fields all moved by ``delta(s)`` has every right to lose a different amount.
    It does not, to ``1e-5``, because the profile is centred: the magnets that are stronger
    are balanced by the ones that are weaker.
    """
    lattice, _ = ring()
    tapered = taper(lattice, rounds=1)
    assert energy_loss_per_turn(tapered) == pytest.approx(energy_loss_per_turn(lattice), rel=1.0e-4)


def test_slicing_the_bends_does_not_move_the_residual() -> None:
    r"""So the residual is not "one field per magnet over a range of momenta".

    Each magnet is tapered for the momentum at its own **midpoint**, and the beam's
    momentum falls across it, so the obvious suspect for the leftover orbit is the
    magnet's own length. Slicing every bend into four gives each slice its own field and
    changes the answer by under a percent — which is what sends the residual back to the
    ``span^2`` law instead.
    """

    def sliced_ring(slices: int) -> Lattice:
        ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
        cells, focal = 20, 2.5
        angle = 2.0 * math.pi / (2 * cells)
        elements: list = []
        for _ in range(cells):
            elements.append(ThinQuadrupole(0.5 / focal))
            elements += [Dipole(1.0 / slices, angle / slices) for _ in range(slices)]
            elements.append(ThinQuadrupole(-1.0 / focal))
            elements += [Dipole(1.0 / slices, angle / slices) for _ in range(slices)]
            elements.append(ThinQuadrupole(0.5 / focal))
        plain = Lattice(elements, ref=ref)
        cavity = RFCavity.from_harmonic(90.0e6, 20, plain.length, ref, phi_s=math.pi)
        return Lattice([*elements, cavity], ref=ref)

    one = orbit_excursion(taper(sliced_ring(1), rounds=1))
    four = orbit_excursion(taper(sliced_ring(4), rounds=1))
    assert four == pytest.approx(one, rel=0.01)


def test_the_geometry_belongs_to_the_ring_and_the_field_to_the_beam() -> None:
    r"""The milestone's central invariant, asserted on a real ring.

    Every bend's ``angle`` is untouched and they still sum to ``2 pi`` — a tapered machine
    is the *same machine*, bent the same way round — while the ``k0``s carry the sawtooth,
    spread across the ring by exactly the span. This is what a taper is allowed to change
    and what it is not, and it is checked here rather than in the reference suite because
    nothing in it needs xtrack and CI runs only the analytic tests.
    """
    lattice, _ = ring()
    tapered = taper(lattice)
    bends = [e for e in lattice.elements if isinstance(e, Dipole)]
    tapered_bends = [e for e in tapered.elements if isinstance(e, Dipole)]

    assert sum(e.angle for e in tapered_bends) == pytest.approx(2.0 * math.pi, rel=1.0e-12)
    assert all(t.angle == b.angle for t, b in zip(tapered_bends, bends, strict=True))
    assert all(t.curvature == b.curvature for t, b in zip(tapered_bends, bends, strict=True))
    assert any(t.k0 != b.k0 for t, b in zip(tapered_bends, bends, strict=True))
    assert np.ptp([t.taper for t in tapered_bends]) == pytest.approx(
        taper_profile(lattice).span, rel=0.03
    ), "the sawtooth, written into the magnets"


def test_the_design_optics_of_a_tapered_ring_are_still_the_design_optics() -> None:
    r"""What the *rest* of the package does when handed one, which nothing else asks.

    A tapered ring's one-turn matrix is a product of element matrices each expanded at its
    **own** ``delta = t_i``, and the ``t_i`` vary element to element, so the product is not
    the Jacobian of any single map. That could have moved the optics by the taper itself.
    It does not: the tunes move by ``2e-7`` against a taper spread of ``3.7e-3`` — *second*
    order — and ``beta`` and the natural chromaticity by ``1e-4`` relative. A tapered
    machine is the design machine, which is the point of tapering it.

    The one thing that is deliberately **not** design-like is the ``delta = 0`` closed orbit
    with radiation off: there the magnets really are mis-set, by their own taper, and the
    ring closes half a millimetre off axis. That is the same statement as the ``7 mm`` sag,
    read from the other side, and it is what :meth:`Dipole._kick_body` exists to carry.
    """
    lattice, _ = ring()
    tapered = taper(lattice)
    spread = float(np.ptp([e.taper for e in tapered.elements if isinstance(e, Dipole)]))

    design_q, tapered_q = tunes(lattice), tunes(tapered)
    assert abs(tapered_q[0] - design_q[0]) < spread**2 * 0.1
    assert abs(tapered_q[1] - design_q[1]) < spread**2 * 0.1

    design_tw, tapered_tw = closed_twiss(lattice), closed_twiss(tapered)
    assert tapered_tw.beta_x == pytest.approx(design_tw.beta_x, rel=spread)
    assert tapered_tw.beta_y == pytest.approx(design_tw.beta_y, rel=spread)

    design_chroma = natural_chromaticity(lattice)
    tapered_chroma = natural_chromaticity(tapered)
    assert tapered_chroma[0] == pytest.approx(design_chroma[0], rel=spread)
    assert tapered_chroma[1] == pytest.approx(design_chroma[1], rel=spread)

    assert np.abs(closed_orbit(lattice)).max() == 0.0
    assert np.abs(closed_orbit(tapered)).max() == pytest.approx(6.6e-4, rel=0.05)


# ---------------------------------------------------------------------------
# taper() itself
# ---------------------------------------------------------------------------


def test_a_zero_profile_returns_the_design_ring() -> None:
    """And returns a *new* lattice: the caller's magnets are never written to."""
    lattice, _ = ring()
    zero = TaperProfile(
        delta=np.zeros(len(lattice.elements)),
        boundary=np.zeros(len(lattice.elements) + 1),
        s_mid=np.zeros(len(lattice.elements)),
        lengths=np.array([e.length for e in lattice.elements]),
        delta_start=0.0,
        loss_eV=0.0,
    )
    out = taper(lattice, zero, rounds=1)
    assert out is not lattice
    for before, after in zip(lattice.elements, out.elements, strict=True):
        if isinstance(after, Dipole):
            assert after is not before
            assert after.k0 == before.k0 and after.k1 == before.k1
        elif isinstance(after, ThinQuadrupole):
            assert after.k1l == before.k1l
    assert np.array_equal(out.transfer_matrix(), lattice.transfer_matrix())


def test_taper_leaves_the_cavity_and_the_drifts_alone() -> None:
    """The unpowered list is a statement, and the cavity is the load-bearing entry.

    The RF is what *pays* for the loss; its voltage is set by that loss and not by the
    local momentum, so scaling it would change the very thing the taper is compensating.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    drift, quad = Drift(1.0), ThinQuadrupole(0.4)
    bend = Dipole(1.0, 0.157)
    plain = Lattice([quad, bend, drift], ref=ref)
    cavity = RFCavity.from_harmonic(90.0e6, 20, plain.length, ref, phi_s=math.pi)
    lattice = Lattice([quad, bend, drift, cavity], ref=ref)

    profile = TaperProfile(
        delta=np.array([0.1, 0.1, 0.1, 0.1]),
        boundary=np.zeros(5),
        s_mid=np.zeros(4),
        lengths=np.array([0.0, 1.0, 1.0, 0.0]),
        delta_start=0.0,
        loss_eV=0.0,
    )
    out = taper(lattice, profile, rounds=1)
    assert out.elements[0].k1l == pytest.approx(0.4 * 1.1)
    assert out.elements[1].k0 == pytest.approx(0.157 * 1.1)
    assert out.elements[1].curvature == pytest.approx(0.157), "the geometry does not move"
    assert out.elements[2] is drift
    assert out.elements[3] is cavity


def test_taper_refuses_an_element_it_cannot_classify() -> None:
    """Silence is the dangerous answer: an untapered magnet in a tapered ring is a bug."""

    class MysteryMagnet(Drift):
        pass

    # a subclass of Drift is unpowered by inheritance; a bare Element is not classified
    from accsim.elements.element import Element

    class Unknown(Element):
        def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:  # pragma: no cover
            return np.eye(DIM)

    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 6.5e9)
    lattice = Lattice([Unknown(1.0, name="mystery")], ref=ref)
    profile = TaperProfile(
        delta=np.array([0.01]),
        boundary=np.zeros(2),
        s_mid=np.zeros(1),
        lengths=np.array([1.0]),
        delta_start=0.0,
        loss_eV=0.0,
    )
    with pytest.raises(NotImplementedError, match="does not know whether"):
        taper(lattice, profile, rounds=1)
    assert isinstance(MysteryMagnet(1.0), Drift)


def test_taper_rejects_a_profile_from_a_different_ring() -> None:
    lattice, _ = ring()
    profile = taper_profile(lattice)
    short = TaperProfile(
        delta=profile.delta[:5],
        boundary=profile.boundary[:6],
        s_mid=profile.s_mid[:5],
        lengths=profile.lengths[:5],
        delta_start=0.0,
        loss_eV=0.0,
    )
    with pytest.raises(ValueError, match="different ring"):
        taper(lattice, short, rounds=1)
    with pytest.raises(ValueError, match="at least 1"):
        taper(lattice, profile, rounds=0)
