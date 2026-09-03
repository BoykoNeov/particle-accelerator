r"""P2 (iii) acceptance: the RF cavity gives **energy**, not momentum.

Until this milestone :class:`~accsim.elements.rfcavity.RFCavity` added its kick straight
to ``delta``. The amplitude was right — ``q V [sin ...] / (beta0^2 E0)`` is exactly the
kick in ``p_zeta = (E - E0)/(beta0^2 E0)`` — but it was added to the wrong variable:
``delta`` is a *nonlinear* function of the energy, and the conversion was frozen at
``delta = 0``. The curvature it dropped is ``d^2 delta/d p_zeta^2 = -1/gamma0^2``.

**What this file gates, and what each gate cannot see.**

1. **The map, against a 60-digit evaluation of the exact statement.** The sharp gate:
   ``1e-16`` relative for the energy kick, ``1.6e-10`` for the old one. It is done in
   ``mpmath`` because the *readable* form of the same claim — "the energy went up by
   ``q V sin phi``" — cannot be measured to better than ``2e-9`` in double precision:
   ``Delta E`` is ``1.2e3`` eV on top of an ``E`` of ``1.9e10`` eV, so forming the
   difference costs seven digits before the map is even asked anything. That floor is
   measured below and stated, not worked around silently; the old map misses by
   ``2.5e-6``, a thousand times above it, so the readable form still discriminates.
2. **A zero kick is the identity bit for bit** — ``V = 0``, and the synchronous particle
   sitting inside a bunch. The increment form gives this with no special case; a
   ``delta -> p_zeta -> delta`` round trip would not (measured ``~1e-16`` relative on
   :func:`accsim.pzeta_from_delta`'s own round trip).
3. **First order is untouched** — the roadmap's gate for this milestone. Asserted on the
   *tracked* Jacobian and on a *tracked* synchrotron tune, never on
   :meth:`RFCavity.matrix`, which was not edited and so could only agree with itself.
4. **Canonical symplecticity, in both directions.** The old map is a shear in
   ``(zeta, delta)``, which is not a conjugate pair; the new one is a shear in
   ``(zeta, p_zeta)``, which is. The same physical residual ``1.58e-10`` therefore shows
   up on whichever map is being read in the wrong pair, and each checker calls the *other*
   map symplectic. Both directions are asserted, because a checker that only ever says
   "pass" gates nothing.
5. **The second-order coefficient**, ``T[delta, zeta, delta] = -R65/(2 gamma0^2)`` —
   PTC's named gap from P1 — derived here in sympy from
   ``(1+delta)^2 = 1 + 2 p_zeta + beta0^2 p_zeta^2`` rather than recalled, then read off
   the element's own differenced map.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    slip_factor,
    synchrotron_tune,
    taylor_expand,
)
from accsim.coords import DELTA, DIM, ZETA
from accsim.reference import CLIGHT
from accsim.symplectic import J6, from_canonical, to_canonical

MASS0 = 938.27208816e6
GAMMA0 = 20.0

#: The cavity used throughout: 1 MV at 3 MHz, off a zero crossing so that *both*
#: ``cos phi_s`` (the slope) and ``sin phi_s`` (the ``zeta^2`` curvature) are nonzero.
VOLTAGE, FREQ_HZ, PHI_S = 1.0e6, 3.0e6, 0.3

#: A generic longitudinal position: far enough up the wave that the kick is not tiny,
#: well inside the bucket. Every "the old map misses by ..." number below is at this zeta.
ZETA0 = 0.02


@pytest.fixture(scope="module")
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


@pytest.fixture(scope="module")
def cav() -> RFCavity:
    return RFCavity(VOLTAGE, FREQ_HZ, PHI_S)


class LinearisedCavity(RFCavity):
    """The pre-P2 (iii) cavity: the same kick, added straight to ``delta``.

    Kept as a working element rather than described in prose, so that every "the old map
    misses by ..." claim in this file is measured against the code that actually shipped,
    and so a whole *ring* can be tracked with it (:func:`test_the_tracked_synchrotron_tune_is_unchanged`).
    """

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        out = np.array(state, dtype=float, copy=True)
        out[DELTA] += self.energy_kick_pzeta(out[ZETA], ref)
        return out


@pytest.fixture(scope="module")
def old() -> LinearisedCavity:
    return LinearisedCavity(VOLTAGE, FREQ_HZ, PHI_S)


def _energy_eV(delta: float | np.ndarray, ref: ReferenceParticle) -> float | np.ndarray:
    """``E = sqrt(P^2 + m^2)`` in eV, from the momentum deviation."""
    return np.hypot(ref.momentum_eV * (1.0 + np.asarray(delta)), ref.mass_eV)


def _delta_out_exact(delta: float, dpzeta: float, ref: ReferenceParticle) -> mp.mpf:
    r"""The exact ``delta`` after an energy kick ``dpzeta``, at 60 significant digits.

    ``E' = E + beta0^2 E0 * Delta p_zeta`` and ``P'^2 = E'^2 - m^2``, written in units of
    ``p0 c`` so that only ``m/p0`` enters:

        ``1 + delta' = sqrt( (sqrt((1+delta)^2 + (m/p0)^2) + Delta ptau)^2 - (m/p0)^2 )``,
        ``Delta ptau = beta0 * Delta p_zeta``.

    **The kick is taken as a double, not recomputed here**, deliberately. This milestone
    changed the *conversion* from energy to ``delta``; the amplitude
    ``q V [sin(phi_s - k zeta) - sin phi_s]`` is untouched code, and evaluating it in
    double precision costs ``2.5e-14`` relative to a difference of two sines near
    ``sin(0.3)`` — a cancellation :mod:`accsim.taylor` already documents. Feeding that
    same double in isolates what changed from what did not; the amplitude itself is
    gated separately, in
    :func:`test_the_energy_gained_is_qV_sin_phi_and_this_is_the_floor_that_says_it`.
    """
    with mp.workdps(60):
        c = mp.mpf(ref.mass_eV) / mp.mpf(ref.momentum_eV)
        e_in = mp.sqrt((1 + mp.mpf(delta)) ** 2 + c**2)
        return mp.sqrt((e_in + mp.mpf(ref.beta0) * mp.mpf(dpzeta)) ** 2 - c**2) - 1


def _energy_gain_error(delta_in: float, delta_out: float, dpzeta: float, ref) -> float:
    """Relative error in the energy actually delivered, at 60 digits.

    ``(E_out - E_in)/(beta0^2 E0 Delta p_zeta) - 1``, with the energies formed in extended
    precision so that the ``1.2e3`` eV kick on a ``1.9e10`` eV particle does not lose
    seven digits to cancellation. Used for the old map, whose error is the thing being
    characterised and so must not be read through a floor.
    """
    with mp.workdps(60):
        c = mp.mpf(ref.mass_eV) / mp.mpf(ref.momentum_eV)

        def e_over_p0c(d: float) -> mp.mpf:
            return mp.sqrt((1 + mp.mpf(d)) ** 2 + c**2)

        got = e_over_p0c(delta_out) - e_over_p0c(delta_in)
        return float(got / (mp.mpf(ref.beta0) * mp.mpf(dpzeta)) - 1)


# ---------------------------------------------------------------------------
# 1. the map is the energy kick
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.0, 1e-4, 1e-3, 5e-3, -3e-3])
def test_delta_after_the_cavity_matches_the_exact_energy_kick(cav, old, ref, delta) -> None:
    """The sharp gate: ``delta'`` against 60 digits, ``1e-15`` relative, at every ``delta``.

    Must hold *off*-momentum, not only at ``delta = 0`` — that is the whole content of the
    milestone, and it is where the old map goes, measured, six orders worse.
    """
    state = np.array([1e-3, 2e-4, -5e-4, 1e-4, ZETA0, delta])
    want = _delta_out_exact(delta, cav.energy_kick_pzeta(ZETA0, ref), ref)
    out = cav.track(state, ref)
    assert abs(mp.mpf(out[DELTA]) / want - 1) < 1e-15
    # The control, on the same line: the shipped-until-now map, six orders worse.
    assert abs(mp.mpf(old.track(state, ref)[DELTA]) / want - 1) > 1e-11

    # Nothing else moves: px, py are normalised to the *reference* P0, which the thin
    # cavity does not change, and zeta is read, not written.
    assert np.array_equal(out[[0, 1, 2, 3, 4]], state[[0, 1, 2, 3, 4]])


def test_the_energy_gained_is_qV_sin_phi_and_this_is_the_floor_that_says_it(cav, old, ref) -> None:
    r"""The readable form of the gate above, with its own measurement floor stated.

    ``E_out - E_in = q V [sin(phi_s - k zeta) - sin phi_s]``. Forming that difference in
    double precision costs seven digits (``1.2e3`` eV out of ``1.9e10`` eV), so this can
    only be checked to ``~2e-9`` — measured, and asserted as a *bound on the check*, not
    presented as the map's accuracy. It still discriminates: the old map misses by
    ``2.5e-6`` at ``delta = 1e-3``, a thousand times above the floor.
    """
    delta = 1e-3
    state = np.array([0.0, 0.0, 0.0, 0.0, ZETA0, delta])
    k = cav.k_rf(ref)
    want = ref.charge * cav.voltage * (math.sin(PHI_S - k * ZETA0) - math.sin(PHI_S))

    new = _energy_eV(cav.track(state, ref)[DELTA], ref) - _energy_eV(delta, ref)
    was = _energy_eV(old.track(state, ref)[DELTA], ref) - _energy_eV(delta, ref)
    assert abs(new / want - 1.0) < 5e-9  # the floor of this arithmetic, not of the map
    assert abs(was / want - 1.0) > 1e-6  # the old map, a thousand times worse


def test_the_old_maps_energy_error_is_delta_over_gamma_squared(old, ref) -> None:
    r"""Where the old map's error comes from, as a closed form: ``delta/gamma0^2``.

    First order in the momentum deviation, not second — which is why this was worth a
    milestone rather than a footnote. Nothing here is a tolerance: the relative error in
    ``Delta E`` is compared to ``delta/gamma0^2`` at four amplitudes, and it is read at 60
    digits because at ``delta = 1e-4`` the error being characterised (``2.5e-7``) is only
    a hundred times the double-precision floor on the *check* (``1.8e-9``) — which showed
    up as a ``1.6e-2`` disagreement before the arithmetic was moved out of the way.
    """
    dpz = old.energy_kick_pzeta(ZETA0, ref)
    for delta in (1e-4, 1e-3, 5e-3, -3e-3):
        state = np.array([0.0, 0.0, 0.0, 0.0, ZETA0, delta])
        out = old.track(state, ref)[DELTA]
        rel = _energy_gain_error(delta, out, dpz, ref)
        assert rel == pytest.approx(delta / ref.gamma0**2, rel=1e-2)


def test_the_old_maps_energy_error_scales_as_one_over_gamma_squared(old) -> None:
    """``x4`` per halving of ``gamma0`` — the mechanism, not a number.

    This is what makes the milestone worth a session on a ``gamma0 = 20`` proton ring and
    invisible on a high-energy electron one, and it is the only claim about *size* here.
    """
    delta = 1e-3
    state = np.array([0.0, 0.0, 0.0, 0.0, ZETA0, delta])

    def rel_error(gamma0: float) -> float:
        r = ReferenceParticle.from_gamma(MASS0, gamma0)
        k = old.k_rf(r)
        want = r.charge * old.voltage * (math.sin(PHI_S - k * ZETA0) - math.sin(PHI_S))
        got = _energy_eV(old.track(state, r)[DELTA], r) - _energy_eV(delta, r)
        return abs(got / want - 1.0)

    hi, mid, lo = rel_error(40.0), rel_error(20.0), rel_error(10.0)
    assert mid / hi == pytest.approx(4.0, rel=1e-2)
    assert lo / mid == pytest.approx(4.0, rel=1e-2)


# ---------------------------------------------------------------------------
# 2. a zero kick is exactly the identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.0, 1e-6, 1e-3, 1e-2])
def test_a_vanishing_kick_is_the_identity_bit_for_bit(cav, ref, delta) -> None:
    """A kick of exactly zero must not perturb ``delta`` at all, at any momentum.

    Two ways for the kick to vanish, and **the first is the one that matters**: a *live*
    cavity at ``zeta = 0``, where the amplitude and ``E/(p0 c)`` are both nonzero and the
    whole conversion runs, and only ``sin(phi_s - 0) - sin(phi_s)`` is zero. A ``V = 0``
    cavity is the weaker case — it zeroes the amplitude before any of the arithmetic — and
    it is kept only as the degenerate end of the same claim.

    Writing the map as ``delta -> p_zeta -> delta`` would fail both by a ``~1e-16``
    relative residue (measured on :func:`accsim.pzeta_from_delta`'s round trip): far below
    any tolerance in this suite, but it would make every cavity a source of numerical noise
    on every turn of a ``1e5``-turn track, and it would put the 6D closed orbit (I4) off
    axis by that much for no physical reason. The increment form gives it with no branch:
    ``s = 0.0 * (2*E/(p0c) + 0.0)`` is exactly ``0.0`` however large the other factor is,
    and ``delta += 0.0`` is the identity.
    """
    synchronous = np.array([1e-3, 2e-4, -5e-4, 1e-4, 0.0, delta])
    assert np.array_equal(cav.track(synchronous, ref), synchronous)

    off_crest = np.array([1e-3, 2e-4, -5e-4, 1e-4, ZETA0, delta])
    assert np.array_equal(RFCavity(0.0, FREQ_HZ, PHI_S).track(off_crest, ref), off_crest)


def test_the_synchronous_particle_in_a_bunch_is_untouched(cav, ref) -> None:
    """``zeta = 0`` gives ``Delta p_zeta = 0`` identically, so ``delta`` does not move —
    even for a particle sitting in a bunch alongside others that *are* kicked.

    An ``if kick == 0`` short-circuit would cover the ``V = 0`` case above and miss this
    one; the increment form covers both because the branch is arithmetic, not control
    flow, and so it holds per-particle inside a vectorised bunch.
    """
    bunch = np.zeros((DIM, 4))
    bunch[DELTA] = [0.0, 1e-3, -2e-3, 5e-3]
    bunch[ZETA] = [0.0, 0.0, 0.01, -0.01]
    out = cav.track(bunch, ref)
    assert np.array_equal(out[DELTA, :2], bunch[DELTA, :2])  # both at zeta = 0
    assert np.all(out[DELTA, 2:] != bunch[DELTA, 2:])  # the others moved


def test_the_bunch_path_is_the_per_particle_path(cav, ref) -> None:
    """A ``(6, n)`` bunch and ``n`` separate ``(6,)`` tracks agree bit for bit."""
    rng = np.random.default_rng(11)
    bunch = rng.normal(scale=1e-3, size=(DIM, 8))
    bunch[ZETA] *= 10.0
    together = cav.track(bunch, ref)
    for j in range(bunch.shape[1]):
        assert np.array_equal(together[:, j], cav.track(bunch[:, j].copy(), ref))


@pytest.mark.parametrize("gamma0", [2.0, 5.0])
def test_you_cannot_take_more_than_the_kinetic_energy_and_the_map_says_so(gamma0) -> None:
    r"""The map has a **physical domain**, and its edge is a closed form.

    ``(1 + delta')^2 = (E/(p0 c) + Delta ptau)^2 - (m/p0)^2`` has no real root once the
    kick drops the total energy below ``m c^2``. On axis that edge is exactly

        ``Delta p_zeta = -(gamma0 - 1)/(beta0^2 gamma0)``,   i.e.   ``Delta E = -(gamma0 - 1) m c^2``

    — the whole kinetic energy, which is the only sentence needed to state it. Below the
    edge the map returns a real ``delta``; above it, ``nan``.

    **The old map had no domain and was not merely imprecise there.** At 99% of the
    kinetic energy removed it reports ``delta = -0.660`` (``gamma0 = 2``) where the exact
    answer is ``-0.918``: it claims the particle keeps a third of the design momentum
    after losing essentially all of its kinetic energy. At 101% it reports ``-0.673``, a
    perfectly finite number for a particle that no longer exists — and it would carry on
    past ``delta = -1``, a negative total momentum.

    This is not a pathology of the fix; it is the fix telling the truth about where the
    model stops. `tests/analytic/test_moving_bucket.py` meets it on a runaway trajectory
    and handles it with the same idiom L3's exact bend already needed.
    """
    ref = ReferenceParticle.from_gamma(MASS0, gamma0)
    kinetic_eV = (gamma0 - 1.0) * MASS0
    edge = -(gamma0 - 1.0) / (ref.beta0**2 * gamma0)

    def probe(fraction: float) -> tuple[float, float]:
        """Track on axis at the phase where the kick is exactly ``-fraction`` of kinetic."""
        c = RFCavity(fraction * kinetic_eV, FREQ_HZ, 0.0)
        state = np.zeros(DIM)
        state[ZETA] = math.pi / (2.0 * c.k_rf(ref))  # sin(-k zeta) = -1: the full amplitude
        kick = c.energy_kick_pzeta(state[ZETA], ref)
        assert kick == pytest.approx(fraction * edge, rel=1e-12)  # the fixture is on target
        return float(c.track(state, ref)[DELTA]), float(state[DELTA] + kick)

    inside_new, inside_old = probe(0.99)
    assert math.isfinite(inside_new) and inside_new > -1.0
    assert inside_new == pytest.approx(-0.918146 if gamma0 == 2.0 else -0.941690, rel=1e-5)
    assert inside_old == pytest.approx(0.99 * edge, rel=1e-12)  # the old map: just the kick
    assert inside_old - inside_new > 0.1  # not a small correction: a qualitatively wrong delta

    outside_new, outside_old = probe(1.01)
    assert math.isnan(outside_new)  # no real root: the particle would be below rest
    assert math.isfinite(outside_old)  # the old map reported a number regardless


def test_reversing_the_voltage_undoes_the_kick(cav, ref) -> None:
    """``+V`` then ``-V`` at the same ``zeta`` is the identity to ``5e-18``.

    The map is invertible in closed form; the old one was invertible only to the order it
    was linearised at. ``zeta`` never changes, so the second cavity sees the same phase.
    """
    back = RFCavity(-VOLTAGE, FREQ_HZ, PHI_S)
    state = np.array([1e-3, 2e-4, -5e-4, 1e-4, 0.03, 4e-3])
    assert np.max(np.abs(back.track(cav.track(state, ref), ref) - state)) < 5e-18


# ---------------------------------------------------------------------------
# 3. first order is untouched — the roadmap's gate for this milestone
# ---------------------------------------------------------------------------


def test_the_tracked_jacobian_is_still_the_shipped_matrix(cav, ref) -> None:
    """``R65`` and ``R66`` are what they were, read from *tracking*.

    Asserting ``matrix()`` against itself would prove only that ``_matrix_body`` was not
    edited. ``d delta/d p_zeta = 1`` at the origin and the synchronous particle sits at
    ``Delta p_zeta = 0``, so both entries survive the change — but that is a claim about
    the *tracking* map, so it is the tracking map that is differenced here.
    """
    R = jacobian(lambda s: cav.track(s, ref), np.zeros(DIM), step=1e-6)
    assert np.max(np.abs(R - cav.matrix(ref))) < 1e-14
    assert R[DELTA, ZETA] == pytest.approx(cav.slope(ref), rel=1e-9)
    assert R[DELTA, DELTA] == pytest.approx(1.0, abs=1e-14)


def _bunched_ring(cavity_cls, voltage: float, gamma0: float = GAMMA0) -> Lattice:
    """Dispersive arc FODO (``eta > 0``, above transition) plus one cavity at harmonic 8.

    Above transition and ``q > 0``, so ``phi_s = pi`` is the stable phase.
    """
    r = ReferenceParticle.from_gamma(MASS0, gamma0)
    arc = [
        ThinQuadrupole(0.25),
        Dipole(1.0, 0.05),
        ThinQuadrupole(-0.5),
        Dipole(1.0, 0.05),
        ThinQuadrupole(0.25),
    ]
    circumference = sum(e.length for e in arc)
    freq = 8 * r.beta0 * CLIGHT / circumference
    return Lattice([*arc, cavity_cls(voltage, freq, math.pi)], r)


def _tracked_qs(lat: Lattice, delta0: float, turns: int = 8192) -> float:
    """Longitudinal tune from an interpolated FFT peak of the tracked ``zeta``."""
    state = np.zeros(DIM)
    state[DELTA] = delta0
    hist = np.empty(turns)
    for n in range(turns):
        for elem in lat.elements:
            state = elem.track(state, lat.ref)
        hist[n] = state[ZETA]
    spec = np.abs(np.fft.rfft((hist - hist.mean()) * np.hanning(turns)))
    i = int(np.argmax(spec[1:])) + 1
    a, b, c = spec[i - 1], spec[i], spec[i + 1]
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) / turns


def test_the_tracked_synchrotron_tune_is_unchanged(ref) -> None:
    r"""The roadmap's "``Qs`` unchanged at first order", made non-vacuous.

    :func:`accsim.synchrotron_tune` reads :meth:`RFCavity.slope`, which this milestone did
    not touch, so asserting on it would restate
    :func:`test_the_tracked_jacobian_is_still_the_shipped_matrix`. What *could* have moved
    is the tracked motion, so the whole ring is tracked twice — once with the energy kick,
    once with :class:`LinearisedCavity` — and the two tunes compared.

    Measured over 8192 turns (~13 synchrotron periods): ``2.1e-9`` relative at a ``1e-3``
    launch and ``3.9e-7`` at ``5e-3``. Both are far below the ``4e-6`` at which the
    tracked tune departs from the small-amplitude closed form in the first place, which is
    the number that says the change is invisible at first order: whatever moved is smaller
    than the approximation ``Qs`` is defined by.
    """
    exact = _bunched_ring(RFCavity, 1.0e6)
    crude = _bunched_ring(LinearisedCavity, 1.0e6)

    # The small-amplitude closed form, and how far the tracked tune sits from it anyway.
    eta = slip_factor(Lattice(exact.elements[:-1], exact.ref))
    qs2 = -(8 * eta * exact.ref.charge * 1.0e6 * math.cos(math.pi)) / (
        2 * math.pi * exact.ref.beta0**2 * exact.ref.total_energy_eV
    )
    assert synchrotron_tune(exact) == pytest.approx(math.sqrt(qs2), rel=1e-5)
    assert synchrotron_tune(exact) == synchrotron_tune(crude)  # slope() is untouched

    small = _tracked_qs(exact, 1e-3)
    assert small == pytest.approx(math.sqrt(qs2), rel=1e-2)  # the pendulum, tracked
    assert small / _tracked_qs(crude, 1e-3) - 1 == pytest.approx(0.0, abs=1e-8)

    # It is not identically zero — it grows with amplitude, as a second-order effect must.
    big = abs(_tracked_qs(exact, 5e-3) / _tracked_qs(crude, 5e-3) - 1)
    assert 1e-8 < big < 1e-5


# ---------------------------------------------------------------------------
# 4. canonical symplecticity, in both directions
# ---------------------------------------------------------------------------


def _canonical_residual(map_fn, state: np.ndarray, ref: ReferenceParticle) -> float:
    """``max |M^T J M - J|`` with the longitudinal pair changed to ``(zeta, p_zeta)``.

    The number behind :func:`accsim.is_symplectic_map_canonical`'s boolean, so the tests
    below can gate a *scaling* rather than a threshold. ``step = 1e-4`` because the
    residual of an exactly symplectic map is pure differencing round-off, which falls as
    the step *grows*: measured on this cavity at ``zeta = 0.02``, ``5.1e-15`` at ``1e-4``
    against ``1.1e-12`` at ``1e-7``. The old map's residual is physical and does not move
    with the step at all (``1.58e-10`` at every step in that sweep) — which is itself the
    cleanest evidence that the two numbers are different in kind.
    """

    def conjugated(canonical: np.ndarray) -> np.ndarray:
        return to_canonical(map_fn(from_canonical(canonical, ref)), ref)

    M = jacobian(conjugated, to_canonical(state, ref), step=1e-4)
    return float(np.max(np.abs(M.T @ J6 @ M - J6)))


def test_the_energy_kick_is_symplectic_in_zeta_pzeta_and_the_old_one_is_not(cav, old, ref) -> None:
    r"""A kick in ``p_zeta`` that depends on ``zeta`` alone is a shear in a *conjugate*
    pair, so it is exactly symplectic at any amplitude. The old map was a shear in
    ``(zeta, delta)``, which is not a conjugate pair.

    Measured at ``zeta = 0.02``: ``5.1e-15`` for the energy kick against ``1.58e-10`` for
    the old one — four orders, a separation rather than a margin.
    """
    state = np.array([1e-3, 2e-4, -5e-4, 1e-4, ZETA0, 5e-3])
    assert is_symplectic_map_canonical(
        lambda s: cav.track(s, ref), state, ref, step=1e-4, atol=1e-13
    )
    assert not is_symplectic_map_canonical(
        lambda s: old.track(s, ref), state, ref, step=1e-4, atol=1e-13
    )
    assert _canonical_residual(lambda s: cav.track(s, ref), state, ref) < 1e-13
    assert _canonical_residual(lambda s: old.track(s, ref), state, ref) > 1e-11


def test_the_zeta_delta_check_says_the_opposite_and_both_are_right(cav, old, ref) -> None:
    r""":mod:`accsim.symplectic`'s documented caveat, on this element — and the symmetry.

    ``is_symplectic_map`` tests ``(zeta, delta)``, so it calls the map that gets the energy
    wrong symplectic and the map that gets it right not. The residual it reports for the
    energy kick is ``1.583e-10`` — *the same number* the canonical check reports for the
    old map, because in both cases it is the one physical quantity here, the failure of
    ``delta`` to be conjugate to ``zeta``, seen from the two sides.

    Asserted rather than noted: a reader who reaches for the wrong checker gets a green
    suite and a worse cavity, and the equality of the two residuals is the fact that says
    the disagreement is a change of variables and not a bug in either map.
    """
    state = np.array([1e-3, 2e-4, -5e-4, 1e-4, ZETA0, 5e-3])
    assert is_symplectic_map(lambda s: old.track(s, ref), state, step=1e-4, atol=1e-13)
    assert not is_symplectic_map(lambda s: cav.track(s, ref), state, step=1e-4, atol=1e-13)

    def zeta_delta_residual(map_fn) -> float:
        M = jacobian(map_fn, state, step=1e-4)
        return float(np.max(np.abs(M.T @ J6 @ M - J6)))

    theirs = zeta_delta_residual(lambda s: cav.track(s, ref))
    ours = _canonical_residual(lambda s: old.track(s, ref), state, ref)
    assert theirs == pytest.approx(ours, rel=1e-3)
    assert theirs == pytest.approx(1.583e-10, rel=1e-2)


def test_the_old_maps_canonical_residual_is_the_size_of_the_energy_kick(old, ref) -> None:
    r"""It scales as ``V * zeta``, and **not** with ``delta`` — which is the trap.

    ``d p_zeta'/d p_zeta - 1 ~ Delta p_zeta/gamma0^2`` for the old map, so the residual is
    set by *how big the kick is*, not by how far off-momentum the particle is. A test that
    probed the two maps at ``zeta = 0`` would find them identical — both exact, both the
    identity — and conclude nothing; one that varied ``delta`` looking for a trend would
    find none and might read that as the map being fine. Both are pinned here.
    """

    def residual(zeta: float, volts: float, delta: float = 5e-3) -> float:
        c = LinearisedCavity(volts, FREQ_HZ, PHI_S)
        s = np.array([0.0, 0.0, 0.0, 0.0, zeta, delta])
        return _canonical_residual(lambda st: c.track(st, ref), s, ref)

    base = residual(0.1, VOLTAGE)
    assert residual(0.5, VOLTAGE) == pytest.approx(5 * base, rel=2e-2)  # linear in zeta
    assert residual(0.1, VOLTAGE / 2) == pytest.approx(base / 2, rel=2e-2)  # and in V
    # Flat in delta: a factor of ten in the momentum deviation moves it by under 10%.
    assert residual(0.1, VOLTAGE, delta=5e-4) == pytest.approx(base, rel=0.1)
    # And at zeta = 0 there is no kick at all, so the two maps are the same map.
    at_rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5e-3])
    assert np.array_equal(
        RFCavity(VOLTAGE, FREQ_HZ, PHI_S).track(at_rest, ref), old.track(at_rest, ref)
    )


# ---------------------------------------------------------------------------
# 5. the second-order coefficient, derived rather than recalled
# ---------------------------------------------------------------------------


def test_delta_as_a_function_of_pzeta_has_curvature_minus_one_over_gamma_squared() -> None:
    r"""``delta = p_zeta - p_zeta^2/(2 gamma0^2) + O(p_zeta^3)``, from sympy.

    The whole milestone in one line. ``p_zeta = (E - E0)/(beta0^2 E0)`` gives
    ``E = E0 (1 + beta0^2 p_zeta)``, and ``P^2 = E^2 - m^2`` with ``P0 = beta0 E0`` gives

        ``(1 + delta)^2 = 1 + 2 p_zeta + beta0^2 p_zeta^2``

    exactly. Its curvature at the origin is ``beta0^2 - 1 = -1/gamma0^2``, and that
    factor — nothing else — is what the old cavity dropped.
    """
    pz, b0, E0 = sp.symbols("pz beta0 E0", positive=True)
    P0 = b0 * E0
    E = E0 * (1 + b0**2 * pz)
    # m^2 = E0^2 - P0^2, so (1+delta)^2 = P^2/P0^2 comes out in terms of beta0 alone.
    one_plus_delta_sq = sp.simplify((E**2 - (E0**2 - P0**2)) / P0**2)
    assert sp.simplify(one_plus_delta_sq - (1 + 2 * pz + b0**2 * pz**2)) == 0

    delta = sp.sqrt(one_plus_delta_sq) - 1
    series = sp.series(delta, pz, 0, 3).removeO()
    assert sp.simplify(sp.expand(series) - (pz - (1 - b0**2) * pz**2 / 2)) == 0

    g0 = sp.symbols("gamma0", positive=True)
    curvature = sp.diff(delta, pz, 2).subs(pz, 0).subs(b0, sp.sqrt(1 - 1 / g0**2))
    assert sp.simplify(curvature + 1 / g0**2) == 0  # 1 - beta0^2 = 1/gamma0^2


def test_the_second_order_map_gains_exactly_minus_r65_over_two_gamma_squared(cav, ref) -> None:
    r"""``T[delta, zeta, delta] = T[delta, delta, zeta] = -R65/(2 gamma0^2)``.

    PTC's named gap from P1, ``+4.02e-9`` on this cavity. Derived here as the whole
    composition — ``delta' = psi(Z(delta) + g(zeta))``, expanded to second order in both
    variables — and then read off the element's own differenced map. The factor of two is
    P1's symmetric storage convention: ``T`` holds half the mixed partial.
    """
    d, z, b0, A, k, ph = sp.symbols("d z beta0 A k phi", positive=True)
    c = sp.sqrt(1 - b0**2) / b0  # m/p0 = 1/(beta0 gamma0)
    Z = (sp.sqrt((1 + d) ** 2 + c**2) - 1 / b0) / b0  # delta -> p_zeta
    g = A * (sp.sin(ph - k * z) - sp.sin(ph))  # the kick, in p_zeta
    pz = Z + g
    delta_out = sp.sqrt(1 + 2 * pz + b0**2 * pz**2) - 1  # p_zeta -> delta

    quad = sp.series(sp.series(delta_out, d, 0, 3).removeO(), z, 0, 3).removeO()
    r65 = sp.simplify(sp.diff(quad, z).subs([(d, 0), (z, 0)]))
    cross = sp.simplify(sp.diff(quad, z, 1, d, 1).subs([(d, 0), (z, 0)]))
    assert sp.simplify(r65 + A * k * sp.cos(ph)) == 0  # the shipped R65, recovered
    gamma0 = 1 / sp.sqrt(1 - b0**2)
    assert sp.simplify(cross / 2 + r65 / (2 * gamma0**2)) == 0

    # ... and the shipped element reproduces it.
    want = -cav.slope(ref) / (2 * ref.gamma0**2)
    step = np.array([1e-3] * 4 + [1e-2, 1e-3])
    m = taylor_expand(lambda s: cav.track(s, ref), np.zeros(DIM), step=step)
    assert abs(m.T[DELTA, ZETA, DELTA] / want - 1) < 1e-6
    assert m.T[DELTA, ZETA, DELTA] == pytest.approx(m.T[DELTA, DELTA, ZETA], rel=1e-12)
    assert abs(want) > 4e-9  # not a round-off statement: 4000x the 1e-12 floor


def test_the_zeta_squared_curvature_gains_its_own_much_smaller_term(cav, ref) -> None:
    r"""``T[delta, zeta, zeta] = -A k^2 sin(phi_s)/2 - R65^2/(2 gamma0^2)``.

    The second, far smaller consequence: the same ``d^2 delta/d p_zeta^2`` acting on the
    *square* of the linear part of the kick. It is ``4.1e-7`` of the leading curvature
    here — below every tolerance in the reference suite, and above the ``5e-9`` relative
    gate ``tests/analytic/test_second_order_map.py`` carries on this entry, which is why
    that test's expected value had to move with this milestone.
    """
    amp = ref.charge * cav.voltage / (ref.beta0**2 * ref.total_energy_eV)
    leading = -amp * cav.k_rf(ref) ** 2 * math.sin(PHI_S) / 2
    extra = -(cav.slope(ref) ** 2) / (2 * ref.gamma0**2)
    step = np.array([1e-3] * 4 + [1e-2, 1e-3])
    m = taylor_expand(lambda s: cav.track(s, ref), np.zeros(DIM), step=step)
    assert abs(m.T[DELTA, ZETA, ZETA] / (leading + extra) - 1) < 5e-9
    # The new piece is real but tiny, and this states how tiny rather than hiding it.
    assert abs(extra / leading) == pytest.approx(4.1e-7, rel=0.1)
