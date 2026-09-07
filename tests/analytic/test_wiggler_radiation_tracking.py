r"""Analytic gates for the wiggler radiating in **tracking** (T3) — the field gets an ``s``.

T2 gave the wiggler's period-averaged ``<h^2>`` and ``<|h|^3>`` to
:func:`~accsim.radiation.radiation_integrals`, so a ring with one in it damped correctly on
the *design* route while a tracked particle crossing the same magnet lost **nothing** — the
accessor ``radiation_kick`` samples has no ``s``, and a wiggler's field averages to exactly
zero over a period. T1 shipped that as a raise rather than a silent zero. T3 closes it.

Ordered by how much each can catch:

  * **The two odd moments, asserted as *orders*.** Under mid-point sampling at ``n`` samples
    per period the three moments radiation needs converge at three different rates, and the
    one the headline gate reads is the useless one: ``int kappa^2 ds`` (the mean loss) is
    **exact at every ``n >= 3``**, while ``int |kappa| ds`` (the photon count) converges as
    ``n^-2`` and ``int kappa^3 ds`` (the excitation variance, T2's ``I3``) as ``n^-4``. They
    carry ``h0^2``, ``h0`` and ``h0^3``, so a uniformly mis-scaled field — what every
    structural gate in this package is blind to — cannot hide in all three.
  * **The tracked loss is the integral's times the wiggle's own path lengthening**, exactly
    ``1 + theta^2/4``, and that factor is *the same number* T1 ships as the element's
    :meth:`~accsim.elements.element.Element.kick`. The design-route ``I2`` integrates along
    the reference ``s``; a particle radiates along the road it actually travels, and a
    wiggler is the only magnet in the package whose on-design road is longer than its
    length. Two milestones' worth of separate machinery landing on one number.
  * **T2's staircase separation, carried into tracking.** Against the same-``I2``,
    no-net-bend stack, the wiggler's tracked *variance* is higher by
    ``8 sqrt2/(3 pi) * (1 + theta^2/4)`` — T2's design-route ratio times the same path
    factor, because the staircase's own design orbit is exactly its length.
  * **Off momentum and off axis**, where every T1 and T2 gate is blind: the loss scales as
    ``(1 + delta)^2`` and grows as ``cosh^2(ky)``.
  * **The wiggle angle is the vector potential — and it moves the radiation by exactly
    zero on axis.** ``a_x = theta sin(ks) cosh(ky)`` makes ``px - a_x`` the sub-period orbit
    the shipped map does not carry, which is elegant and, for radiation, almost inert: the
    wiggle is horizontal and the field is vertical, so the velocity is *always* perpendicular
    to ``B`` and ``|B_perp|`` never sees the angle. It enters only through ``b_s``, which
    needs ``y != 0``. Gated at both ends — the bit-exact zero and the ``5.8e-07`` off axis.
  * **The refusals that lift, and the one that does not.** Radiation tracking and
    ``taper()`` work; ``normalized_field`` still raises and spin still refuses, which is T4.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import sympy as sp

from accsim import Dipole, Drift, Lattice, Quadrupole, ReferenceParticle, Wiggler, taper
from accsim.radiation_kick import (
    photon_energy_variance,
    photon_rate,
    radiation_constant_cgamma,
)
from accsim.tracking import Tracker

# The probe machine of T1 and T2: B0 = 1.5 T, lambda_w = 0.1 m, N = 10, E = 1 GeV.
PERIOD = 0.1
PERIODS = 10
H0 = 0.449689
LENGTH = PERIOD * PERIODS
ENERGY = 1e9
MASS0 = 0.51099895069e6
WAVENUMBER = 2.0 * math.pi / PERIOD
THETA = H0 / WAVENUMBER

#: The exact moments over the body, from ``<cos^2> = 1/2``, ``<|cos|> = 2/pi`` and
#: ``<|cos|^3> = 4/(3 pi)`` — T2 derived the last two in sympy; they are used here.
EXACT_MOMENTS = {
    1: 2.0 * H0 * LENGTH / math.pi,
    2: 0.5 * H0**2 * LENGTH,
    3: 4.0 * H0**3 * LENGTH / (3.0 * math.pi),
}

#: The wiggle is a longer road by exactly this, and it is T1's ``kick()`` on ``zeta``
#: divided by the length. Everything tracked carries it; nothing integrated does.
PATH_FACTOR = 1.0 + 0.25 * THETA**2


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(MASS0, ENERGY, 1.0)


@pytest.fixture
def wig() -> Wiggler:
    return Wiggler(PERIOD, H0, PERIODS, "w")


def _staircase() -> list[Dipole]:
    """T2's comparator: ``2*PERIODS`` hard-edge poles, same ``I2``, no net bend."""
    hs, half = H0 / math.sqrt(2.0), PERIOD / 2.0
    return [Dipole(half, (hs if j % 2 == 0 else -hs) * half) for j in range(2 * PERIODS)]


def _tracked_loss(elements: list, ref: ReferenceParticle, state: np.ndarray | None = None) -> float:
    """The energy [eV] one pass actually takes off the particle, ``radiation="mean"``."""
    st = np.zeros(6) if state is None else np.asarray(state, dtype=float)
    out = Tracker(Lattice(list(elements), ref)).track_once(st.copy(), radiation="mean")
    d0, d1 = float(st[5]), float(out[5])
    e0 = math.hypot(ref.momentum_eV * (1.0 + d0), MASS0)
    e1 = math.hypot(ref.momentum_eV * (1.0 + d1), MASS0)
    # Rationalised, for the reason L1 recorded for the drift: ``e0 - e1`` subtracts two
    # numbers of size 1e9 to produce one of size 1e3 and keeps only six digits of it,
    # which is a floor of 7e-11 -- above the gates below. This form keeps all of them.
    return ref.momentum_eV**2 * (d0 - d1) * (2.0 + d0 + d1) / (e0 + e1)


def _sampled_moment(w: Wiggler, power: int, y: float = 0.0) -> float:
    """``int |kappa|^p ds`` as the shipped sampling computes it — the accessors, not a copy."""
    s = w.radiation_sample_positions()
    _bx, by, _bs, _ax, _ay = w.field_at(s, 0.0, y)
    return float(np.sum(np.abs(by) ** power)) * (w.length / s.size)


# --- the discriminating gates: two odd moments, asserted as orders -----------------------
@pytest.mark.parametrize(
    ("power", "order", "why"),
    [
        (1, 2, "|cos| has a jump in its FIRST derivative"),
        (3, 4, "|cos|^3 ~ |u|^3 near its zero is C^2 — the jump is in the THIRD"),
    ],
)
def test_the_two_odd_moments_converge_at_their_own_measured_orders(
    power: int, order: int, why: str
) -> None:
    """Halving the step divides the error by ``2^order`` — the exponent, not a tolerance.

    These are the gates that can fail. The photon count (``power = 1``) sets how grainy the
    emission is; the excitation variance (``power = 3``) is T2's ``I3`` and sets the energy
    spread. They carry ``h0`` and ``h0^3`` where the mean loss carries ``h0^2``, so no
    single mis-scaling of the field passes all three — J1's lesson, in the form this
    milestone takes it, and J2's (gate on the order).
    """
    errs = []
    for n in (16, 32, 64, 128):
        got = _sampled_moment(Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=n), power)
        errs.append(abs(got / EXACT_MOMENTS[power] - 1.0))

    assert all(e > 0.0 for e in errs), f"{why}: an exact answer would make this vacuous"
    for coarse, fine in zip(errs, errs[1:], strict=False):
        # The measured ratios are 4.01/4.00/4.00 and 16.51/16.12/16.03; the band is wide
        # enough for the approach to the asymptote and far too narrow for the other order.
        assert coarse / fine == pytest.approx(2.0**order, rel=0.06), why
    # ...and the neighbouring order is emphatically excluded.
    assert not (errs[-2] / errs[-1] == pytest.approx(2.0 ** (order + 1), rel=0.06))


def test_the_mean_loss_is_exact_at_every_usable_sample_count_and_that_is_a_blindness() -> None:
    """``int cos^2 ds`` is integrated **exactly** by the mid-point rule at every ``n >= 3``.

    ``cos^2 = (1 + cos 2ks)/2`` and the uniform mid-point sum of ``cos 2ks`` over a whole
    period vanishes identically. So the obvious headline gate — "the tracked energy loss
    equals ``C_gamma E^4 I2 / 2 pi``" — passes the moment the machinery exists, at any
    resolution anyone would choose, and can therefore say nothing about whether the
    sampling is right. It is recorded here as a **blindness** next to the gate it makes
    vacuous, the way J1 recorded that the structural gates cannot see a kick coefficient.
    """
    exact = EXACT_MOMENTS[2]
    for n in (3, 4, 5, 8, 32, 256):
        got = _sampled_moment(Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=n), 2)
        assert got == pytest.approx(exact, rel=1e-15)


def test_the_sample_count_below_three_is_refused_because_the_mean_stops_being_exact() -> None:
    """Two samples per period is where the mid-point rule fails on ``cos^2`` itself."""
    with pytest.raises(ValueError, match="radiation_slices must be an integer >= 3"):
        Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=2)
    with pytest.raises(ValueError, match="radiation_slices must be an integer >= 3"):
        Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=8.5)  # type: ignore[arg-type]

    # ...and at n = 2 it is wrong by 100%: the two samples land on the zeros of cos.
    samples = (np.arange(2) + 0.5) / 2.0
    assert float(np.mean(np.cos(2 * math.pi * samples) ** 2)) == pytest.approx(0.0, abs=1e-30)


# --- the tracked loss, and the number two milestones share -------------------------------
def test_the_tracked_loss_is_the_integrals_times_the_wiggles_own_path_lengthening(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    """``U_tracked / U(I2) = 1 + theta^2/4``, and that factor is T1's ``kick()`` on zeta.

    ``I2 = int kappa^2 ds`` integrates along the **reference** coordinate. A particle
    radiates along the road it actually travels, and the wiggle is a longer road by
    ``L theta^2/4`` — the constant term T1 shipped as the only nonzero
    :meth:`~accsim.elements.element.Element.kick` an aligned, on-design element in this
    package has. So the design route and the tracked route differ by exactly that, on the
    design orbit, for the only element where they can.

    The pre-committed gate said these two agree at machine precision. They do not, they
    differ by ``1.2806e-05``, and the difference is not an error in either.
    """
    u_integral = radiation_constant_cgamma(ref) / (2.0 * math.pi) * ENERGY**4 * EXACT_MOMENTS[2]
    assert _tracked_loss([wig], ref) == pytest.approx(u_integral * PATH_FACTOR, rel=1e-13)

    # The factor is T1's constant, not a coefficient fitted here.
    assert PATH_FACTOR - 1.0 == pytest.approx(-wig.kick(ref)[4] / wig.length, rel=1e-15)
    assert PATH_FACTOR - 1.0 == pytest.approx(1.280574e-05, rel=1e-6)

    # A straight magnet has no such factor: a bend's on-design road IS its length.
    bend = Dipole(1.0, 0.3)
    u_bend = radiation_constant_cgamma(ref) / (2 * math.pi) * ENERGY**4 * 0.3**2 / 1.0
    assert _tracked_loss([bend], ref) == pytest.approx(u_bend, rel=1e-9)


def test_the_tracked_loss_does_not_move_with_the_sample_count(ref: ReferenceParticle) -> None:
    """The blindness above, seen end-to-end: 3 samples per period and 256 agree bit for bit."""
    losses = [
        _tracked_loss([Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=n)], ref)
        for n in (3, 8, 32, 256)
    ]
    for got in losses[1:]:
        # Not bit-identical, and the reason is not physics: summing 30 terms and summing
        # 2560 of them round differently. 8e-16 is that, and nothing else.
        assert got == pytest.approx(losses[0], rel=1e-14)


# --- T2's staircase separation, carried into tracking ------------------------------------
def _tracked_variance(elements: list, ref: ReferenceParticle) -> float:
    """The excitation variance ``sigma_U^2`` one pass accumulates, summed over samples."""
    total, st = 0.0, np.zeros(6)
    for el in elements:
        after = el.track(st, ref)
        pos = el.radiation_sample_positions()
        n = 1 if pos is None else pos.size
        s_at = 0.5 * el.length if pos is None else pos
        mid_y = 0.5 * (st[2] + after[2])
        _bx, by, _bs, _ax, _ay = el.field_at(s_at, 0.5 * (st[0] + after[0]), mid_y)
        kappa = np.abs(by)
        share = (el.length - (after[4] - st[4])) / n
        u = radiation_constant_cgamma(ref) / (2 * math.pi) * ENERGY**4 * kappa * kappa * share
        total += float(np.sum(photon_energy_variance(u, ENERGY, kappa, ref)))
        st = after
    return total


def test_the_staircase_separation_survives_into_the_tracked_variance(
    ref: ReferenceParticle,
) -> None:
    """``8 sqrt2/(3 pi)``, T2's design-route ratio, now measured through the tracking seam.

    A sinusoid spends more of its length near peak curvature than a square wave of the same
    mean square does, and the odd moment is what notices. T2 asserted that on
    ``radiation_integrals``; this asserts the same separation on the quantity a *tracked*
    particle's energy spread is built from, with the two routes sharing no code.

    The tracked ratio carries the path factor and the design-route one does not, for the
    reason the test above states — the staircase's own design orbit is exactly its length,
    so the whole of ``1 + theta^2/4`` lands on the wiggler's side. The pre-committed gate
    asked for the bare ``8 sqrt2/(3 pi)`` to ``1e-12``; that is wrong by ``1.3e-05``.
    """
    fine = Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=1024)
    ratio = _tracked_variance([fine], ref) / _tracked_variance(_staircase(), ref)

    assert ratio == pytest.approx(8 * math.sqrt(2) / (3 * math.pi) * PATH_FACTOR, rel=1e-9)
    assert ratio > 1.0  # the direction: the wiggler is the HIGH one
    assert ratio == pytest.approx(1.2004371, rel=1e-6)


def test_the_photon_count_separation_is_the_other_odd_moment(ref: ReferenceParticle) -> None:
    """``2 sqrt2 / pi = 0.9003``: the wiggler emits **fewer**, harder photons than the stack.

    The count goes as ``int |kappa| ds``, which is the *first* moment, so the same matched
    ``I2`` that leaves ``I3`` 20% high leaves the count 10% **low**. Two odd moments moving
    in opposite directions off one matched even one — which is why a mis-scaled field
    cannot satisfy them together.
    """

    def count(elements: list) -> float:
        total, st = 0.0, np.zeros(6)
        for el in elements:
            after = el.track(st, ref)
            pos = el.radiation_sample_positions()
            n = 1 if pos is None else pos.size
            _bx, by, _bs, _ax, _ay = el.field_at(0.5 * el.length if pos is None else pos, 0.0, 0.0)
            total += float(np.sum(photon_rate(ENERGY, np.abs(by), el.length / n, ref)))
            st = after
        return total

    fine = Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=1024)
    assert count([fine]) / count(_staircase()) == pytest.approx(
        2.0 * math.sqrt(2.0) / math.pi, rel=1e-5
    )
    assert count([fine]) < count(_staircase())  # fewer photons, and they are harder


# --- off momentum and off axis: where T1's and T2's gates cannot see ----------------------
def test_the_loss_scales_as_one_plus_delta_squared(ref: ReferenceParticle, wig: Wiggler) -> None:
    """``kappa ~ 1/(1+delta)`` and ``E^4 ~ (1+delta)^4`` compose to ``(1+delta)^2``.

    Every T1 and T2 gate sits at ``delta = 0``. This is the chromatic content of the
    radiation, and it is the *square* — a wiggler's focusing carries the second power of
    the rigidity too, and for the same reason: it is quadratic in a deflection that is
    linear in ``1/(1+delta)``.
    """
    on = _tracked_loss([wig], ref)
    for delta in (0.01, 0.05, -0.03):
        st = np.zeros(6)
        st[5] = delta
        got = _tracked_loss([wig], ref, st)
        assert got / on == pytest.approx((1.0 + delta) ** 2, rel=2e-4)


def test_the_loss_grows_off_axis_as_cosh_squared(ref: ReferenceParticle, wig: Wiggler) -> None:
    """``b_y = h0 cos(ks) cosh(ky)``: a wiggler's gap field is *stronger* away from the axis.

    A quadrupole's field grows off axis because that is its whole point; a bend's does not
    change at all. A wiggler's grows because Maxwell forces it to — the same ``cosh``/``sinh``
    pair that puts the focusing in the vertical plane. On axis this is invisible, which is
    where the entire T1/T2 gate list lives.
    """
    on = _tracked_loss([wig], ref)
    st = np.zeros(6)
    st[2] = 2e-3
    got = _tracked_loss([wig], ref, st)

    assert got > on
    assert got / on == pytest.approx(math.cosh(WAVENUMBER * 2e-3) ** 2, rel=6e-3)
    assert got / on == pytest.approx(1.015085, rel=1e-5)


# --- the vector potential: exact, elegant, and nearly inert for radiation -----------------
def test_the_vector_potential_is_the_whole_field_and_the_wiggle_angle(wig: Wiggler) -> None:
    """``a_x = theta sin(ks) cosh(ky)`` gives both field components, and ``px - a_x`` is x'.

    Symbolically: ``curl a`` with ``a = (a_x(y,s), 0, 0)`` is ``(0, da_x/ds, -da_x/dy)``, and
    those are exactly T1's ``b_y`` and ``b_s``. The consequence is the implementation: the
    state carries **canonical** momentum, the physics wants the **kinetic** one, and
    ``px - a_x = px - theta sin(ks)`` on axis is precisely the sub-period orbit
    ``x'(s) = -theta sin(ks)`` that the period-averaged map deliberately does not track. The
    wiggle lives in ``a``, which is *why* the shipped map holds ``px`` constant.
    """
    s, y, h0, k = sp.symbols("s y h0 k", real=True, positive=True)
    a_x = h0 / k * sp.sin(k * s) * sp.cosh(k * y)
    assert sp.simplify(sp.diff(a_x, s) - h0 * sp.cos(k * s) * sp.cosh(k * y)) == 0
    assert sp.simplify(-sp.diff(a_x, y) + h0 * sp.sin(k * s) * sp.sinh(k * y)) == 0

    # ...and the shipped accessor is that, numerically, with the field it implies.
    s_at = np.linspace(0.0, wig.length, 37)
    bx, by, bs, ax, ay = wig.field_at(s_at, 0.0, 1.5e-3)
    ks, ky = wig.wavenumber * s_at, wig.wavenumber * 1.5e-3
    assert np.allclose(bx, 0.0) and np.allclose(ay, 0.0)
    assert np.allclose(by, H0 * np.cos(ks) * np.cosh(ky), rtol=1e-15)
    assert np.allclose(bs, -H0 * np.sin(ks) * np.sinh(ky), rtol=1e-15)
    assert np.allclose(ax, THETA * np.sin(ks) * np.cosh(ky), rtol=1e-15)

    # The kinetic angle of a particle entering on axis IS the sub-period orbit.
    _bx, _by, _bs, ax0, _ay = wig.field_at(s_at, 0.0, 0.0)
    assert np.allclose(0.0 - ax0, -THETA * np.sin(ks), rtol=1e-15)


def test_the_wiggle_angle_moves_the_radiation_by_exactly_zero_on_axis(
    ref: ReferenceParticle,
) -> None:
    """Elegant, correct, and — for radiation — almost entirely inert. Stated, not hidden.

    The wiggle is horizontal and ``b_y`` is vertical, so the velocity is **always**
    perpendicular to the field in the plane of the motion: ``|B_perp|`` does not contain the
    wiggle angle at all. It enters only through ``b_s``, which vanishes on the mid-plane,
    and through ``i_z`` at ``O(theta^2)``. So deleting ``a`` changes the on-axis loss by not
    one bit, and moves it by ``5.8e-07`` at ``y = 2 mm``.

    That number is small and it is not round-off, which is what makes this a gate rather
    than a remark — and it is the *only* place the sub-period orbit reaches the radiation.
    Where the wiggle angle does matter is the spin precession axis, and that is T4.
    """

    class NoPotential(Wiggler):
        def field_at(self, s, x, y):  # type: ignore[no-untyped-def]
            bx, by, bs, ax, ay = super().field_at(s, x, y)
            return bx, by, bs, np.zeros_like(np.asarray(ax, dtype=float)), ay

    plain, bare = Wiggler(PERIOD, H0, PERIODS, "w"), NoPotential(PERIOD, H0, PERIODS, "w")
    for delta in (0.0, 0.05):
        st = np.zeros(6)
        st[5] = delta
        assert _tracked_loss([plain], ref, st) == _tracked_loss([bare], ref, st)

    st = np.zeros(6)
    st[2] = 2e-3
    with_a, without = _tracked_loss([plain], ref, st), _tracked_loss([bare], ref, st)
    assert (with_a - without) / with_a == pytest.approx(5.794e-07, rel=1e-3)


# --- the refusals: which lift, and which is left standing for T4 --------------------------
def test_radiation_tracking_through_a_wiggler_no_longer_refuses(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    """T1's and T2's refusal, lifted — and it is the milestone.

    This test replaces ``test_radiation_tracking_through_a_wiggler_refuses``, which T1 wrote
    to fail the day this landed. All three physical models now run, and the loss they take
    is the one the closed form predicts.
    """
    state = np.array([1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0])
    for model in ("mean", "quantum", "photons"):
        out = wig.track(state, ref, radiation=model, rng=np.random.default_rng(0))
        assert out[5] < 0.0  # it radiated

    assert wig.track(state, ref, radiation="off")[5] == 0.0  # ...and "off" still does not


def test_the_stochastic_models_land_on_the_mean_and_the_variance(ref: ReferenceParticle) -> None:
    """``"quantum"`` and ``"photons"`` reproduce the two moments the sampling computes.

    The graininess is a sum over the body now, not one draw against one field sample, so
    both the mean (the sum of the per-sample losses) and the variance (the sum of the
    per-sample variances) have to come back out of an actual bunch.
    """
    wig = Wiggler(PERIOD, H0, PERIODS, "w")
    predicted_u = _tracked_loss([wig], ref)
    predicted_sigma = math.sqrt(_tracked_variance([wig], ref))

    for model in ("quantum", "photons"):
        rng = np.random.default_rng(7)
        out = Tracker(Lattice([wig], ref)).track_once(np.zeros((6, 4000)), model, rng)
        delta = out[5]
        assert -float(delta.mean()) * ENERGY == pytest.approx(predicted_u, rel=0.05)
        assert float(delta.std()) * ENERGY == pytest.approx(predicted_sigma, rel=0.05)

    # ...and "mean" has no spread at all: every particle loses the same energy.
    out = Tracker(Lattice([wig], ref)).track_once(np.zeros((6, 64)), "mean")
    assert float(out[5].std()) == pytest.approx(0.0, abs=1e-20)


def test_tapering_a_ring_with_a_wiggler_now_works(ref: ReferenceParticle) -> None:
    """The last of ``taper()``'s two refusals, lifted — and it was this accessor.

    T2 decided a wiggler is a powered magnet and put it in ``tapering._STRENGTHS``, which
    removed one cause; the other was that ``taper()`` reaches
    :func:`~accsim.orbit.closed_orbit_6d` with ``radiation="mean"``, which tracks, which
    called the accessor that raised. Nothing else stood in the way.
    """
    from accsim import RFCavity

    ring = Lattice(
        [
            Quadrupole(0.5, 1.2, "qf"),
            Dipole(2.0, 0.3, name="b1"),
            Quadrupole(0.5, -1.2, "qd"),
            Drift(0.5),
            Wiggler(PERIOD, H0, PERIODS, "w"),
            Dipole(2.0, 0.3, name="b2"),
            RFCavity(4e6, 4e8, 0.0, "rf"),
        ],
        ref,
    )
    tapered = taper(ring)

    scaled = {e.name: e for e in tapered.elements}
    assert scaled["w"].h0 != H0  # the wiggler's own field followed the beam's momentum
    assert scaled["b1"].k0 != scaled["b2"].k0  # ...and the sag is not flat around the ring


def test_spin_through_a_wiggler_still_refuses_and_this_test_fails_the_day_t4_lands(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    """The one refusal T3 leaves standing, kept loud for the same reason T1 kept it.

    :mod:`accsim.spin` samples the field **once**, at the mid-point, exactly as radiation
    used to — so it would precess a spin through a field that averages to zero and report a
    polarisation built from nothing. Spin is not an integral along the path but a
    composition of rotations in which order matters, so the fix is not this milestone's
    sampling and the physics is different: that is T4.
    """
    with pytest.raises(NotImplementedError, match="spin through a wiggler is"):
        wig.normalized_field(0.0, 0.0)

    from accsim.reference import ELECTRON_ANOMALOUS_MOMENT

    spin_ref = ReferenceParticle.from_total_energy(
        MASS0, ENERGY, 1.0, anomalous_moment=ELECTRON_ANOMALOUS_MOMENT
    )
    state = np.array([1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0])
    with pytest.raises(NotImplementedError, match="spin through a wiggler is"):
        wig.track_with_spin(state, np.array([0.0, 1.0, 0.0]), spin_ref)


# --- nothing else moves ------------------------------------------------------------------
@pytest.mark.parametrize(
    "element",
    [
        Dipole(1.0, 0.3, name="b"),
        Dipole(1.0, 0.3, k1=0.4, name="bk"),
        Quadrupole(0.5, 1.2, "q"),
        Drift(0.7, "d"),
    ],
)
def test_the_new_hook_forwards_and_every_older_element_is_unchanged(element) -> None:  # type: ignore[no-untyped-def]
    """``field_at`` defaults to the three ``s``-independent accessors, and ignores ``s``.

    Gate 1: the arithmetic for every element built before T3 is the same arithmetic, which
    is what makes the whole of axis B bit-identical. An element with a constant body has
    the same field everywhere in it, so ``s`` is not merely unused — answering differently
    at two ``s`` would be the bug.
    """
    x, y = 1.3e-3, -0.7e-3
    assert element.radiation_sample_positions() is None

    bx, by = element.normalized_field(x, y)
    bs = element.longitudinal_field(x, y)
    ax, ay = element.normalized_vector_potential(x, y)
    for s in (0.0, 0.5 * element.length, element.length, 12.0):
        assert element.field_at(s, x, y) == (bx, by, bs, ax, ay)


def test_the_wigglers_map_and_integrals_are_untouched(ref: ReferenceParticle, wig: Wiggler) -> None:
    """T3 is a tracking mode. ``matrix()``, ``kick()`` and the integrals are T1's and T2's.

    The map stays the period average deliberately: a sub-period piece of it has no wiggle
    in it at all, because ``y'' = -h0^2 y sin^2(ks)`` is Mathieu-like and has no closed form
    on a partial period. T3 resolves the **sampling**, not the map — and that is why
    ``radiation_slices`` cannot move a single optics number.
    """
    from accsim import radiation_integrals

    coarse = Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=3)
    fine = Wiggler(PERIOD, H0, PERIODS, "w", radiation_slices=512)
    assert np.array_equal(coarse.matrix(ref), fine.matrix(ref))
    assert np.array_equal(coarse.kick(ref), fine.kick(ref))
    assert np.array_equal(coarse.matrix(ref), wig.matrix(ref))

    def integrals(w: Wiggler) -> dict:
        ring = Lattice([Quadrupole(0.5, 1.2), Dipole(2.0, 0.3), Quadrupole(0.5, -1.2), w], ref)
        return dataclasses.asdict(radiation_integrals(ring))

    assert integrals(coarse) == integrals(fine)


def test_the_photon_route_and_the_gaussian_route_agree_slice_by_slice(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    """Per **sample**, not merely in aggregate — the mismatch a 5% bunch gate cannot see.

    ``"quantum"`` draws one Gaussian for the whole element, of variance
    ``sum_i photon_energy_variance(u_i, ...)``. ``"photons"`` draws a Poisson count per
    sample and multiplies by that sample's own critical energy, giving
    ``sum_i u_c,i^2 rate_i <x^2>``. Those are the same number only if ``u_i``, ``kappa_i``
    and the path share ``l_i`` are the *same* per-slice quantities in both — and pairing a
    slice's ``u_c`` with the wrong slice's ``rate`` would still land the aggregate within a
    few percent, which is all a 4000-particle test can resolve. Since the tail is the only
    thing the photon model exists to get right (B5), it is checked term by term here.

    ``<x^2> = 11/27`` is the spectrum's second moment in units of the critical energy, which
    B3 derived from ``int K_{5/3}`` rather than recalling.
    """
    from accsim.radiation_kick import critical_photon_energy

    s = wig.radiation_sample_positions()
    _bx, by, _bs, _ax, _ay = wig.field_at(s, 0.0, 0.0)
    kappa = np.abs(by)
    share = wig.length * PATH_FACTOR / s.size  # the same l_path/n radiation_kick uses
    u = radiation_constant_cgamma(ref) / (2 * math.pi) * ENERGY**4 * kappa * kappa * share

    gaussian = photon_energy_variance(u, ENERGY, kappa, ref)
    u_c = critical_photon_energy(ENERGY, kappa, ref)
    compound = u_c**2 * photon_rate(ENERGY, kappa, share, ref) * (11.0 / 27.0)

    assert np.allclose(gaussian, compound, rtol=1e-12)  # every sample, not just the sum
    # ...and the check is not vacuous: the mid-points never land on a zero of the cosine,
    # but they still span a factor of ten in kappa, so u_c and rate each vary by that and
    # their product by a hundred. A mis-pairing of the two could not survive it.
    assert float(np.max(kappa)) / float(np.min(kappa)) == pytest.approx(10.15, rel=0.01)


def test_a_bunch_and_a_misaligned_magnet_go_through_the_sampled_path(
    ref: ReferenceParticle,
) -> None:
    """The sample axis broadcasts against a ``(6, n)`` bunch, and the body frame is the body's.

    The one piece of shape arithmetic T3 adds: the sample positions carry their own leading
    axis, so a column of them meets a row of particles. A single particle has no such row,
    which is why the two shapes are checked to give the *same* answer rather than merely to
    run — a ``(6, 1)`` bunch and a ``(6,)`` state are the same physics and must not differ.

    And a misaligned wiggler radiates according to where it really is: ``dy`` lifts the beam
    off the mid-plane, where ``b_y`` grows as ``cosh(ky)``, so it must radiate **more**.
    """
    wig = Wiggler(PERIOD, H0, PERIODS, "w")
    lattice = Lattice([Drift(0.2), wig, Dipole(1.0, 0.2)], ref)

    bunch = np.zeros((6, 5))
    bunch[0] = np.linspace(-1e-3, 1e-3, 5)
    bunch[2] = 5e-4
    for model, rng in (("mean", None), ("quantum", np.random.default_rng(3))):
        out = Tracker(lattice).track_once(bunch.copy(), radiation=model, rng=rng)
        assert out.shape == (6, 5)
        assert float(out[5].mean()) < 0.0
    # ...only the deterministic model is negative particle by particle: the Gaussian is
    # unclamped on purpose (B3's *The model can draw an energy gain*), and on a magnet this
    # short the fluctuation is comparable to the loss, so gains are common rather than rare.
    assert np.all(Tracker(lattice).track_once(bunch.copy(), radiation="mean")[5] < 0.0)

    # one particle, two shapes, one answer
    single = Tracker(lattice).track_once(bunch[:, 2].copy(), radiation="mean")
    column = Tracker(lattice).track_once(bunch[:, 2:3].copy(), radiation="mean")
    assert single == pytest.approx(column[:, 0], rel=0.0, abs=0.0)

    # off the mid-plane the gap field is stronger, so a lifted magnet radiates more
    aligned = _tracked_loss([Wiggler(PERIOD, H0, PERIODS, "a")], ref)
    lifted = _tracked_loss([Wiggler(PERIOD, H0, PERIODS, "b", dy=1e-3)], ref)
    assert lifted > aligned
    assert lifted / aligned == pytest.approx(math.cosh(WAVENUMBER * 1e-3) ** 2, rel=2e-3)
