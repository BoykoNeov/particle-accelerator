r"""Analytic gates — quantum excitation, and the equilibrium it holds (B3).

B2 made a tracked particle lose the energy it radiates, so the simulation *exhibits*
damping. Taken alone that map is a lie about the physics: it damps every amplitude to
zero, and a real beam does not shrink to a point. Light comes in photons; a particle
crossing a magnet emits a countable number of them at random moments with random
energies, and that graininess is a random walk that pushes back against the damping.
Where the two balance is the equilibrium beam.

The whole milestone is one number said twice. ``accsim.radiation`` has computed
``eps_x = C_q gamma^2 I5 / (J_x I2)`` and ``sigma_delta^2 = C_q gamma^2 I3 / (J_z I2)``
since Stage 7 — closed forms riding the Twiss functions, touching no particle. B3 puts
a random kick in the tracking and asks whether a bunch left alone *arrives* there. The
two routes share exactly one thing, the constant ``C_q``, and nothing else.

The gates, ordered so a wrong variance cannot hide behind a right mean:

  1. **The spectrum's own moments.** ``<u> = 8/(15 sqrt3) u_c`` and ``<u^2> = 11/27
     u_c^2`` are integrated out of ``int_x^inf K_{5/3}`` here, not quoted, and
     ``n_gamma <u>`` is checked to be the ``U`` B2 already ships — the consistency
     bridge between the ``alpha, hbar c`` system and the ``C_gamma`` system.
  2. **The variance, derived symbolically** *from* those moments:
     ``sigma_U^2 = n_gamma <u^2> = 2 C_q E gamma^2 kappa U``, symbolic difference
     exactly ``0``. Deriving it from the collapsed form instead would only check the
     algebra, not the moments.
  3. **The equilibrium, derived symbolically** from that variance and the damping,
     landing on the shipped closed form exactly — with the synchrotron phase-averaging
     ``1/2`` written down, because dropping it gives exactly ``2x`` and nothing else in
     the suite has that signature.
  4. **What the code injects.** The noise direction is ``(0, px, 0, py, 0, 1+delta)``,
     one common factor as in B2; its magnitude is the advertised variance; and the
     one-turn covariance built from per-element probes reproduces the covariance
     300,000 tracked particles actually show.
  5. **The equilibrium without statistics.** ``Sigma = M Sigma M^T + D`` — the discrete
     Lyapunov equation for the tracked map's own noise — is solved exactly and its
     eigen-emittances compared to the design route. This is the sharp gate; tracking to
     equilibrium is not, because it is statistics-limited by construction.
  6. **Both departures have named owners, measured as laws.** The two routes do not
     agree to round-off, and the reasons are (a) B2's one-kick-per-element lumping,
     which owns an ``eps_x`` offset and to which ``sigma_delta`` is blind, and (b) the
     **finite synchrotron tune** — the closed forms are the smooth-ring limit, and the
     departure is a function of ``Q_s`` *alone*, to 4 parts in 100,000 across a ring
     with 256x the radiated power.
  7. **The horizontal excitation is dispersion, not recoil.** Deleting the photon's
     direct transverse kick from the injected noise moves ``eps_x`` by 4e-6. Stated as
     a pre-commitment: this is what makes ``eps_x`` a gate on ``I5`` and not on the
     kick's transverse arm.
  8. **There is no vertical excitation at all** — exactly zero, not small. The real
     floor comes from the ``1/gamma`` photon opening angle, which this model omits.
  9. **It actually settles there**, from a point *and* from three times equilibrium,
     against a stated statistical budget rather than a loosened tolerance.
 10. **The model can draw an energy gain**, ~1% of the time, and is deliberately not
     clamped: clamping would bias the mean and the variance by the very percent the
     equilibrium is being gated to.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import kv
from scipy.stats import norm

from accsim import (
    Dipole,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
)
from accsim.radiation import (
    damping_times,
    energy_loss_per_turn,
    equilibrium_emittance,
    equilibrium_energy_spread,
    quantum_constant_cq,
)
from accsim.radiation_kick import (
    HBAR_C_EV_M,
    RADIATION_MODELS,
    STOCHASTIC_MODELS,
    photon_energy_variance,
    radiation_kick,
)
from accsim.reference import CLIGHT
from accsim.symplectic import unit_symplectic_matrix

ELECTRON_MASS_EV = 0.51099895069e6
L_BEND = 1.0

# Three rings, because the milestone needs three different things.
#
# COLD: the sharp gate. 20 cells, 5 GeV, and the RF turned down to just over U0 so the
# synchrotron tune is Q_s = 0.024 -- see gate 6, the closed forms are the Q_s -> 0
# limit. Sliced 8x so B2's lumping is converged too. Here the two routes agree to 0.11%.
# WARM/HOT: the same geometry with the RF turned up, purely to move Q_s.
# SETTLE: 6.5 GeV, where tau_x is 846 turns instead of 1858, which is what makes an
# explicit settle-to-equilibrium measurement affordable. Its Q_s is large and its
# lumping unconverged -- neither matters, because the tracked bunch is compared to the
# Lyapunov solution of the *same* map, not to the closed form.
COLD = {"cells": 20, "focal": 2.5, "energy": 5.0e9, "voltage": 9.0e6, "harmonic": 20}
SETTLE = {"cells": 20, "focal": 2.5, "energy": 6.5e9, "voltage": 90.0e6, "harmonic": 20}


def _ring(
    cells: int,
    focal: float,
    energy: float,
    voltage: float,
    harmonic: int,
    rf: bool = True,
    slices: int = 1,
) -> Lattice:
    """Isomagnetic FODO ring, total bend ``2 pi``, with the RF that replaces ``U0``.

    The same shape B2's gates use, so the two milestones are measured on one machine.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy)
    angle = 2.0 * math.pi / (2 * cells)
    bend = [Dipole(L_BEND / slices, angle / slices) for _ in range(slices)]
    cell = [
        ThinQuadrupole(0.5 / focal),
        *bend,
        ThinQuadrupole(-1.0 / focal),
        *bend,
        ThinQuadrupole(0.5 / focal),
    ]
    elements = list(cell) * cells
    lat = Lattice(elements, ref=ref)
    if not rf:
        return lat
    cavity = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=math.pi)
    return Lattice([*elements, cavity], ref=ref)


# ---------------------------------------------------------------------------
# Measurement machinery. Not physics -- the physics is what it is applied to.
#
# The one piece worth reading is _OnePhoton: a stand-in for numpy's Generator that
# returns a chosen number of standard deviations on one nominated draw and zero on every
# other. It turns the stochastic map into a differentiable one, so each element's noise
# can be isolated exactly instead of estimated from samples. It exercises the shipped
# code path end to end -- the variance formula included -- and adds no statistics.
# ---------------------------------------------------------------------------
class _OnePhoton:
    """``rng`` double: ``amp`` standard deviations on draw ``k``, nothing on the rest."""

    def __init__(self, k: int, amp: float) -> None:
        self.k, self.amp, self.i = k, amp, 0

    def normal(self, loc: float, scale: np.ndarray | float) -> np.ndarray | float:
        drawn = self.amp * scale if self.i == self.k else 0.0 * np.asarray(scale)
        self.i += 1
        return drawn + loc


def _one_turn_jacobian(tracker: Tracker, state: np.ndarray, step: float = 1e-7) -> np.ndarray:
    """Jacobian of the turn with radiation ``"mean"`` -- the *deterministic* map.

    Finite-differencing the stochastic map would return noise, and it would not fail
    loudly; the linearisation the equilibrium is built on is the mean map's.
    """
    jac = np.empty((6, 6))
    for i in range(6):
        plus, minus = state.copy(), state.copy()
        plus[i] += step
        minus[i] -= step
        jac[:, i] = (
            tracker.track_once(plus, radiation="mean") - tracker.track_once(minus, radiation="mean")
        ) / (2.0 * step)
    return jac


def _equilibrium_orbit(tracker: Tracker) -> np.ndarray:
    """Newton on ``track_once(s) = s`` with radiation ``"mean"`` (B2's routine).

    Newton, not tracking: the horizontal damping time is thousands of turns, so a
    "converged" orbit reached by tracking is still drifting. And ``"mean"``, not
    ``"quantum"``: Newton on a stochastic map does not converge at all.
    """
    state = np.zeros(6)
    for _ in range(60):
        residual = tracker.track_once(state, radiation="mean") - state
        if np.max(np.abs(residual)) < 1e-14:
            break
        state = state - np.linalg.solve(_one_turn_jacobian(tracker, state) - np.eye(6), residual)
    return state


def _envelope(
    lattice: Lattice, orbit: np.ndarray, step: float = 1e-7, amp: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(M, columns, injected, tails)`` for the ring at its equilibrium orbit.

    ``injected[i]`` is the noise element ``i`` adds per standard deviation, in the state
    right after it; ``tails[i]`` propagates that to the end of the turn; ``columns[i]``
    is the product, and ``M`` is the one-turn Jacobian. Built from per-element
    Jacobians rather than one probe per full turn, which is the difference between
    ``O(n)`` and ``O(n^2)`` and matters on a sliced ring.
    """
    ref = lattice.ref
    elements = list(lattice.elements)
    entry, state = [], orbit.copy()
    for elem in elements:
        entry.append(state.copy())
        state = elem.track(state, ref, radiation="mean")

    jacobians, injected = [], []
    for elem, at in zip(elements, entry, strict=True):
        block = np.empty((6, 6))
        for i in range(6):
            plus, minus = at.copy(), at.copy()
            plus[i] += step
            minus[i] -= step
            block[:, i] = (
                elem.track(plus, ref, radiation="mean") - elem.track(minus, ref, radiation="mean")
            ) / (2.0 * step)
        jacobians.append(block)
        up = elem.track(at.copy(), ref, radiation="quantum", rng=_OnePhoton(0, amp))
        down = elem.track(at.copy(), ref, radiation="quantum", rng=_OnePhoton(0, -amp))
        injected.append((up - down) / (2.0 * amp))

    propagator = np.eye(6)
    columns, tails = [], []
    for i in range(len(elements) - 1, -1, -1):
        tails.append(propagator.copy())
        columns.append(propagator @ injected[i])
        propagator = propagator @ jacobians[i]
    return propagator, np.array(columns[::-1]).T, np.array(injected), np.array(tails[::-1])


def _equilibrium_sigma(lattice: Lattice) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The equilibrium 6x6 covariance, exactly: ``Sigma = M Sigma M^T + D``.

    ``D = sum_i c_i c_i^T`` is the one-turn noise covariance and ``M`` the damped map,
    so this is the fixed point of "diffusion in, damping out" with no tracking and no
    statistics in it anywhere. Returns ``(Sigma, M, injected, tails)``.
    """
    orbit = _equilibrium_orbit(Tracker(lattice))
    m, columns, injected, tails = _envelope(lattice, orbit)
    return solve_discrete_lyapunov(m, columns @ columns.T), m, injected, tails


def _mode_emittances(sigma: np.ndarray) -> tuple[float, float, float]:
    """``(eps_x, eps_y, eps_z)`` from the eigenvalues of ``Sigma S``.

    The invariant route, and the only honest one here: ``sigma_x^2`` is
    ``eps_x beta_x + (D_x sigma_delta)^2`` and on these rings the dispersive term is a
    third of the total, so dividing by ``beta_x`` would report an emittance ~2x too
    large. Modes are labelled by the plane their eigenvector lives in, not by size --
    the longitudinal emittance is the *largest* here, so sorting would mislabel it.
    """
    values, vectors = np.linalg.eig(sigma @ unit_symplectic_matrix())
    found: dict[int, float] = {}
    for value, vector in zip(values, vectors.T, strict=True):
        if value.imag < 0:
            continue  # one of each conjugate pair
        weight = np.abs(vector) ** 2
        plane = int(
            np.argmax([weight[0] + weight[1], weight[2] + weight[3], weight[4] + weight[5]])
        )
        found[plane] = max(found.get(plane, 0.0), abs(value))
    return found.get(0, 0.0), found.get(1, 0.0), found.get(2, 0.0)


def _synchrotron_tune(m: np.ndarray) -> float:
    """``Q_s`` from the tracked one-turn map -- the smallest phase advance of the six."""
    return float(np.sort(np.abs(np.angle(np.linalg.eigvals(m))))[0] / (2.0 * math.pi))


def _tau_turns(lat: Lattice) -> tuple[float, float, float]:
    """``damping_times`` in turns, ascending ``(tau_z, tau_y, tau_x)``."""
    t0 = lat.length / (lat.ref.beta0 * CLIGHT)
    return tuple(sorted(t / t0 for t in damping_times(lat)))  # type: ignore[return-value]


_CACHE: dict[tuple, tuple] = {}


def _cached_sigma(**params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Lattice]:
    """``_equilibrium_sigma`` memoised: several gates share the same expensive rings."""
    key = tuple(sorted(params.items()))
    if key not in _CACHE:
        lat = _ring(**params)
        _CACHE[key] = (*_equilibrium_sigma(lat), lat)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Gate 1 — the synchrotron spectrum's own moments, integrated not quoted.
# ---------------------------------------------------------------------------
def _spectrum_moment(mu: float) -> float:
    r"""``int_0^inf x^mu F(x)/x dx`` with ``F(x) = x int_x^inf K_{5/3}(t) dt``.

    Swapping the order of integration once (exact, no identity assumed) collapses the
    double integral to ``int_0^inf K_{5/3}(t) t^(mu+1)/(mu+1) dt``, which is one
    quadrature. ``mu = 0`` weights photon *number*, ``mu = 1`` photon *energy* (the
    radiated power) and ``mu = 2`` its square.
    """
    return quad(lambda t: kv(5.0 / 3.0, t) * t ** (mu + 1.0) / (mu + 1.0), 0.0, np.inf, limit=200)[
        0
    ]


def test_the_photon_spectrum_moments_are_the_bessel_integrals_own() -> None:
    """``<u>/u_c = 8/(15 sqrt3)`` and ``<u^2>/u_c^2 = 11/27``, from ``K_{5/3}``.

    These two numbers are the entire content of the variance coefficient; quoting them
    from a textbook would make the symbolic gate below a check of arithmetic only.
    """
    number, power, square = (_spectrum_moment(mu) for mu in (0.0, 1.0, 2.0))
    assert power / number == pytest.approx(8.0 / (15.0 * math.sqrt(3.0)), rel=1e-9)
    assert square / number == pytest.approx(11.0 / 27.0, rel=1e-9)
    # and the normalisation the photon rate is built on
    assert power == pytest.approx(8.0 * math.pi / (9.0 * math.sqrt(3.0)), rel=1e-12)


def test_the_hbar_c_the_package_uses_agrees_with_the_reference_arms() -> None:
    """``hbar c`` is *not* a named owner of any residual on this axis.

    B2 left two owners on the xtrack cross-check, one of them xtrack's pre-2019 electron
    charge. ``C_q`` brings in a second constant, so it is worth knowing before the
    reference arm runs that this one is not a third: xtrack hardcodes
    ``1.973269804593025e-7`` and the package rounds it, agreeing to 5e-11.
    """
    assert HBAR_C_EV_M == pytest.approx(1.973269804593025e-07, rel=1e-9)


# ---------------------------------------------------------------------------
# Gate 2 — the variance, derived symbolically FROM those moments.
# ---------------------------------------------------------------------------
def _symbolic_photon_algebra() -> dict[str, sp.Expr]:
    """The two constant systems and the one bridge between them, as sympy expressions.

    The photon picture speaks ``alpha`` and ``hbar c``; the radiation integrals speak
    ``C_gamma`` and ``r_e``. They meet at ``r_e m c^2 = alpha hbar c`` and nowhere else,
    so that substitution is the only thing the gates below are allowed to use.
    """
    alpha, hbar_c, gamma, kappa, ell, mc2, r_e = sp.symbols(
        "alpha hbar_c gamma kappa ell mc2 r_e", positive=True
    )
    sq3 = sp.sqrt(3)
    return {
        "n_gamma": sp.Rational(5, 2) / sq3 * alpha * gamma * kappa * ell,
        "u_c": sp.Rational(3, 2) * hbar_c * gamma**3 * kappa,
        "C_gamma": 4 * sp.pi * r_e / (3 * mc2**3),
        "C_q": sp.Rational(55, 32) / sq3 * hbar_c / mc2,
        "E": gamma * mc2,
        "gamma": gamma,
        "kappa": kappa,
        "ell": ell,
        "bridge": {alpha: r_e * mc2 / hbar_c},
    }


def test_the_photon_rate_and_the_mean_photon_energy_reproduce_the_shipped_loss() -> None:
    """``n_gamma <u> == (C_gamma / 2 pi) E^4 kappa^2 l`` — symbolically exactly zero.

    The bridge gate. B2's mean loss and B3's photon picture are written in different
    constants; if they disagreed, a variance built on the photon side would be scaled
    wrong in a way no equilibrium measurement could separate from a wrong ``C_q``.
    """
    s = _symbolic_photon_algebra()
    mean_u = sp.Rational(8, 15) / sp.sqrt(3) * s["u_c"]
    shipped = s["C_gamma"] / (2 * sp.pi) * s["E"] ** 4 * s["kappa"] ** 2 * s["ell"]
    assert sp.simplify((s["n_gamma"] * mean_u - shipped).subs(s["bridge"])) == 0


def test_the_variance_coefficient_is_derived_from_the_photon_moments() -> None:
    """``n_gamma <u^2> == 2 C_q E gamma^2 kappa U`` — symbolically exactly zero.

    This is the shipped :func:`photon_energy_variance` in closed form, and it is derived
    here from the *moments* (gate 1) rather than from itself. Emission is a compound
    Poisson process, so the variance of the total is ``rate x <u^2>`` — the mean photon
    energy does not enter, which is why a model that got ``<u>`` right and ``<u^2>``
    wrong would pass every mean-loss gate B2 has.
    """
    s = _symbolic_photon_algebra()
    mean_u2 = sp.Rational(11, 27) * s["u_c"] ** 2
    shipped_u = s["C_gamma"] / (2 * sp.pi) * s["E"] ** 4 * s["kappa"] ** 2 * s["ell"]
    collapsed = 2 * s["C_q"] * s["E"] * s["gamma"] ** 2 * s["kappa"] * shipped_u
    assert sp.simplify((s["n_gamma"] * mean_u2 - collapsed).subs(s["bridge"])) == 0


# ---------------------------------------------------------------------------
# Gate 3 — the equilibrium, derived from that variance and the damping.
# ---------------------------------------------------------------------------
def _symbolic_equilibrium(phase_factor: sp.Expr) -> sp.Expr:
    r"""Solve ``2 <d^2> / tau_z = phase_factor * sum<u^2> / E^2`` for ``<d^2>``.

    Excitation in, damping out, per turn, with ``1/tau_z = J_z U0 / (2E)`` in the
    amplitude convention :func:`accsim.radiation.damping_times` uses. Summing
    ``sigma_U^2 ∝ kappa^3 l`` around the ring turns it into ``I3``, and ``U0`` carries
    the ``I2``. ``phase_factor`` is the synchrotron phase averaging — see below.
    """
    s = _symbolic_photon_algebra()
    i2, i3, jz = sp.symbols("I2 I3 J_z", positive=True)
    sd2 = sp.Symbol("sigma_d2", positive=True)
    prefactor = s["C_gamma"] / (2 * sp.pi) * s["E"] ** 4
    sum_u2 = 2 * s["C_q"] * s["E"] * s["gamma"] ** 2 * prefactor * i3
    u0 = prefactor * i2
    solved = sp.solve(
        sp.Eq(2 * sd2 * jz * u0 / (2 * s["E"]), phase_factor * sum_u2 / s["E"] ** 2), sd2
    )[0]
    return sp.simplify(solved - s["C_q"] * s["gamma"] ** 2 * i3 / (jz * i2)), sp.simplify(solved)


def test_the_variance_and_the_damping_give_the_shipped_equilibrium_energy_spread() -> None:
    """``sigma_delta^2 = C_q gamma^2 I3 / (J_z I2)``, symbolic difference exactly zero.

    The milestone in one line: the graininess B3 injects and the equilibrium Stage 7
    predicted are the same statement, and the only constant they share is ``C_q``.
    """
    difference, _ = _symbolic_equilibrium(sp.Rational(1, 2))
    assert difference == 0


def test_dropping_the_synchrotron_phase_averaging_doubles_the_answer_exactly() -> None:
    r"""The ``1/2`` is load-bearing, and its failure signature is a clean factor 2.

    The photons kick ``delta`` and nothing else, but ``delta`` is only one coordinate of
    a synchrotron oscillation: a kick at a random phase adds ``<Delta d^2>`` to the
    *invariant* ``a^2``, and ``<delta^2> = <a^2>/2``. Forget it and the predicted spread
    is ``sqrt(2)`` too wide -- a plausible-looking error that no scaling gate can see,
    since it is energy-, geometry- and lattice-independent. Pinned as exactly 2 here so
    it cannot be absorbed into a tolerance anywhere else.
    """
    _, with_half = _symbolic_equilibrium(sp.Rational(1, 2))
    _, without = _symbolic_equilibrium(sp.Integer(1))
    assert sp.simplify(without / with_half) == 2


def test_the_shipped_variance_helper_is_that_expression_for_any_species() -> None:
    """:func:`photon_energy_variance` numerically equals ``2 C_q E gamma^2 kappa U``.

    Run for an electron and a proton: ``C_q ∝ 1/m`` while ``U ∝ 1/m^4``, so a species
    mix-up between the two constants would not cancel.
    """
    for mass, energy in ((ELECTRON_MASS_EV, 3.0e9), (938.272e6, 7.0e12)):
        ref = ReferenceParticle.from_total_energy(mass, energy)
        for kappa, u in ((0.4, 1.0e5), (0.02, 3.3e2)):
            gamma = ref.total_energy_eV / mass
            expected = 2.0 * quantum_constant_cq(ref) * ref.total_energy_eV * gamma**2 * kappa * u
            assert photon_energy_variance(u, ref.total_energy_eV, kappa, ref) == pytest.approx(
                expected, rel=1e-14
            )


# ---------------------------------------------------------------------------
# Gate 4 — what the code actually injects.
# ---------------------------------------------------------------------------
def test_the_quantum_kick_reduces_to_the_mean_one_when_the_draw_is_zero() -> None:
    """A zero draw must reproduce ``"mean"`` bit for bit — the mean is not perturbed."""
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.array([1.0e-3, 2.0e-4, -5.0e-4, 3.0e-4, 1.0e-3, 7.0e-4])
    plain = elem.track(state.copy(), ref, radiation="mean")
    drawn = elem.track(state.copy(), ref, radiation="quantum", rng=_OnePhoton(-1, 0.0))
    assert np.array_equal(plain, drawn)


def test_the_quantum_kick_averages_to_the_mean_kick() -> None:
    """Over many draws the quantum loss has the classical loss as its mean.

    The Gaussian is unbiased by construction, so the only thing this can catch is a
    coefficient accidentally added to the mean rather than to the spread — but that is
    exactly what an implementation that reused the wrong variable would do.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.array([1.0e-3, 2.0e-4, -5.0e-4, 3.0e-4, 1.0e-3, 7.0e-4])
    n = 200_000
    rng = np.random.default_rng(4242)
    bunch = np.repeat(state[:, None], n, axis=1)
    drawn = elem.track(bunch, ref, radiation="quantum", rng=rng)
    plain = elem.track(state.copy(), ref, radiation="mean")
    spread = drawn[5].std(ddof=1)
    # the mean of n draws is within a few sigma/sqrt(n) of the deterministic answer
    assert abs(drawn[5].mean() - plain[5]) < 4.0 * spread / math.sqrt(n)


def test_the_injected_noise_is_one_common_factor_on_px_py_and_one_plus_delta() -> None:
    """The photon leaves along the direction of motion — in the *fluctuation* too.

    B2 established that the mean kick scales ``(px, py, 1+delta)`` by one factor. The
    random part is a fluctuation in the same factor, so the injected noise vector must
    be exactly proportional to ``(0, px, 0, py, 0, 1+delta)``. Checked as the ratio of
    the ``px`` response to the ``delta`` response, at every radiating element of a ring.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    orbit = _equilibrium_orbit(Tracker(lat))
    _, _, injected, _ = _envelope(lat, orbit)
    state = orbit.copy()
    checked = 0
    for elem, noise in zip(lat.elements, injected, strict=True):
        after = elem.track(state, ref, radiation="mean")
        if np.abs(noise).max() > 0.0:
            expected = after[1] / (1.0 + after[5])
            assert noise[1] / noise[5] == pytest.approx(expected, rel=1e-6)
            checked += 1
        state = after
    assert checked == 40  # every bend, and nothing else


def test_the_variance_the_code_injects_is_the_variance_it_advertises() -> None:
    """One element, tracked: the measured spread in ``delta`` is the advertised one.

    Independent of the equilibrium machinery, and the only gate that ties the shipped
    number to a *drawn* sample. ``Delta delta = -U (1+delta)/(beta^2 E)`` to first
    order, so the spread in ``delta`` is ``sigma_U`` scaled by that same factor, which
    is read off the mean kick rather than recomputed.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.zeros(6)
    plain = elem.track(state.copy(), ref, radiation="mean")

    # The mean loss the element took, reconstructed from the energy it lost -- not from
    # the formula under test.
    def energy(delta: float) -> float:
        p = ref.momentum_eV * (1.0 + delta)
        return math.sqrt(p * p + ref.mass_eV**2)

    u = energy(state[5]) - energy(plain[5])
    kappa = 2.0 * math.pi / (2 * COLD["cells"]) / L_BEND  # 1/rho on the design orbit
    sigma_u = math.sqrt(photon_energy_variance(u, energy(state[5]), kappa, ref))
    # dDelta / dU from the mean kick itself
    slope = (plain[5] - state[5]) / u

    n = 400_000
    rng = np.random.default_rng(99)
    drawn = elem.track(np.repeat(state[:, None], n, axis=1), ref, radiation="quantum", rng=rng)
    measured = drawn[5].std(ddof=1)
    assert measured == pytest.approx(abs(slope) * sigma_u, rel=5.0 / math.sqrt(2.0 * n))


def test_the_one_turn_noise_matches_what_three_hundred_thousand_particles_show() -> None:
    """The probe-built ``D`` is the covariance the code really injects over a turn.

    Everything downstream is built on ``D``, and ``D`` is built with a stand-in
    generator; this is the gate that says the stand-in and ``numpy``'s Generator drive
    the same map. The naive check -- summing each element's variance -- is *wrong* by
    24% on this ring, because a kick injected early is partly rotated into ``zeta``
    before the turn ends; propagating each element's noise to the observation point is
    what ``D`` does and what makes the comparison meaningful.
    """
    lat = _ring(**COLD)
    tracker = Tracker(lat)
    orbit = _equilibrium_orbit(tracker)
    _, columns, _, _ = _envelope(lat, orbit)
    predicted = columns @ columns.T

    n = 300_000
    rng = np.random.default_rng(2024)
    tracked = tracker.track_once(np.repeat(orbit[:, None], n, axis=1), "quantum", rng)
    centre = tracker.track_once(orbit.copy(), "mean")
    deviation = tracked - centre[:, None]
    measured = deviation @ deviation.T / n

    budget = math.sqrt(2.0 / n)  # relative error of a variance from n samples
    for i in (0, 1, 4, 5):
        assert measured[i, i] == pytest.approx(predicted[i, i], rel=4.0 * budget)


# ---------------------------------------------------------------------------
# Gate 5 — the equilibrium, exactly, with no statistics in it.
# ---------------------------------------------------------------------------
def test_the_equilibrium_energy_spread_is_the_design_route_closed_form() -> None:
    """A bunch's energy spread, from the tracked map's own noise, lands on Stage 7's.

    Two routes with nothing in common but ``C_q``: one integrates ``I3`` along the Twiss
    functions, the other solves ``Sigma = M Sigma M^T + D`` for the map that a tracked
    particle actually sees. They agree to 0.11% on a ring chosen so both of the
    departures gated below are small.
    """
    sigma, _, _, _, lat = _cached_sigma(slices=8, **COLD)
    assert math.sqrt(sigma[5, 5]) == pytest.approx(equilibrium_energy_spread(lat), rel=2.0e-3)


def test_the_equilibrium_horizontal_emittance_is_the_design_route_closed_form() -> None:
    """And the same for ``eps_x = C_q gamma^2 I5 / (J_x I2)``.

    The harder of the two: ``I5`` is the dispersion invariant ``curly-H``, so this is
    the only gate in the package that reaches ``I5`` from a tracked particle at all.
    """
    sigma, _, _, _, lat = _cached_sigma(slices=8, **COLD)
    eps_x, _, _ = _mode_emittances(sigma)
    assert eps_x == pytest.approx(equilibrium_emittance(lat), rel=2.0e-3)


def test_the_longitudinal_and_horizontal_agree_at_the_same_level() -> None:
    """Both departures are ~0.1%, and in the *same* direction — one cause, not two."""
    sigma, _, _, _, lat = _cached_sigma(slices=8, **COLD)
    eps_x, _, _ = _mode_emittances(sigma)
    ratio_delta = math.sqrt(sigma[5, 5]) / equilibrium_energy_spread(lat)
    ratio_eps = eps_x / equilibrium_emittance(lat)
    assert ratio_delta > 1.0 and ratio_eps > 1.0
    assert abs(ratio_delta - ratio_eps) < 2.0e-4


# ---------------------------------------------------------------------------
# Gate 6 — the two departures, with their owners, measured as laws.
# ---------------------------------------------------------------------------
def test_the_departure_is_a_function_of_the_synchrotron_tune_and_nothing_else() -> None:
    r"""Two rings with 256x the radiated power and the same ``Q_s`` depart identically.

    The closed forms are the **smooth-ring** result: they assume the synchrotron phase
    barely advances while the turn's photons are emitted. Solving the discrete map does
    not assume that, so the two part company at finite ``Q_s``. The claim is that ``Q_s``
    is the *whole* story, and this is the sharp form of it: 1.25 GeV at 30 MV and 5 GeV
    at 120 MV have the same ``Q_s`` to 0.14% -- while ``U0 ∝ E^4`` differs by 256x, the
    equilibrium spread by 4x and the emittance by 16x -- and their departures agree to 4
    parts in 100,000. Nothing but ``Q_s`` could do that.
    """
    ratios, tunes = [], []
    for energy, voltage in ((1.25e9, 30.0e6), (5.0e9, 120.0e6)):
        params = {**COLD, "energy": energy, "voltage": voltage}
        sigma, m, _, _, lat = _cached_sigma(**params)
        ratios.append(math.sqrt(sigma[5, 5]) / equilibrium_energy_spread(lat))
        tunes.append(_synchrotron_tune(m))
    assert tunes[0] == pytest.approx(tunes[1], rel=2e-3)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-4)
    assert ratios[0] > 1.07  # and the departure is large here, so this is not vacuous


def test_the_departure_vanishes_as_the_square_of_the_synchrotron_tune() -> None:
    r"""``sigma_delta^2 / closed form = 1 + c (2 pi Q_s)^2`` — the order, not a tolerance.

    ``c`` is asserted to be the same number across a factor 2.8 in ``Q_s``, which is what
    makes ``Q_s`` an *owner* rather than a fudge: a wrong variance would shift the whole
    curve, not bend it. It also justifies the ring the sharp gates run on -- at
    ``Q_s = 0.024`` the correction is 0.1% and cannot be hiding anything larger.
    """
    coefficients = []
    for voltage in (9.0e6, 12.0e6, 20.0e6):
        sigma, m, _, _, lat = _cached_sigma(slices=4, **{**COLD, "voltage": voltage})
        excess = sigma[5, 5] / equilibrium_energy_spread(lat) ** 2 - 1.0
        coefficients.append(excess / (2.0 * math.pi * _synchrotron_tune(m)) ** 2)
    assert min(coefficients) > 0.0
    assert max(coefficients) / min(coefficients) < 1.03


def test_the_lumping_owns_the_emittance_offset_and_the_energy_spread_is_blind_to_it() -> None:
    r"""B2's one-kick-per-element is the *other* owner, and it lands on one plane only.

    A single kick per element evaluates the loss at the element's entry energy; B2
    asserted that the mean converges as ``(N-1)/N`` under slicing. Here it shows up as a
    ~0.6% offset in ``eps_x`` that slicing removes, while ``sigma_delta`` moves by 3e-5
    and is effectively blind to it. Two owners, two different signatures: this one dies
    under slicing and is ``Q_s``-independent, the other survives slicing and scales as
    ``Q_s^2``. Neither could be mistaken for the other, or for a wrong ``C_q``.
    """
    ratios_eps, ratios_delta = [], []
    for slices in (1, 4, 8):
        sigma, _, _, _, lat = _cached_sigma(slices=slices, **COLD)
        eps_x, _, _ = _mode_emittances(sigma)
        ratios_eps.append(eps_x / equilibrium_emittance(lat))
        ratios_delta.append(math.sqrt(sigma[5, 5]) / equilibrium_energy_spread(lat))
    assert ratios_eps[0] < 0.996  # unsliced: ~0.6% low
    assert abs(ratios_eps[2] - 1.0) < 2.0e-3  # sliced: converged onto the closed form
    assert abs(ratios_eps[2] - ratios_eps[1]) < abs(ratios_eps[1] - ratios_eps[0])
    assert max(ratios_delta) - min(ratios_delta) < 1.0e-4  # the other plane never moved


# ---------------------------------------------------------------------------
# Gate 7 — pre-commitment: what the emittance gate is, and is not, sensitive to.
# ---------------------------------------------------------------------------
def test_the_horizontal_excitation_is_dispersion_and_not_photon_recoil() -> None:
    r"""Delete the photon's direct transverse kick and ``eps_x`` moves by 4e-6.

    Stated as a limit on what gate 5 proves. A photon carries away transverse momentum
    as well as energy, so the noise has a ``px`` component — but it is smaller than the
    ``delta`` component by exactly ``px/(1+delta) ~ 2e-4``, and its effect on a
    *variance* is that squared. What actually excites the horizontal plane is the energy
    kick meeting the dispersion: the off-momentum closed orbit moves, and the betatron
    amplitude jumps by ``D_x Delta delta``. That is the ``curly-H`` in ``I5``.

    So the emittance gate is a gate on ``I5`` and on ``C_q``, and it is **blind** to the
    transverse arm of the kick to six figures. The gate that is not blind to it is B2's
    vertical damping time, which is the whole reason that one exists.
    """
    sigma, m, injected, tails, _ = _cached_sigma(slices=8, **COLD)
    without = injected.copy()
    without[:, 1] = 0.0  # no direct px recoil
    without[:, 3] = 0.0  # no direct py recoil
    columns = np.array([tails[i] @ without[i] for i in range(len(without))]).T
    stripped = solve_discrete_lyapunov(m, columns @ columns.T)
    full_eps, _, _ = _mode_emittances(sigma)
    stripped_eps, _, _ = _mode_emittances(stripped)
    assert stripped_eps == pytest.approx(full_eps, rel=1e-5)


# ---------------------------------------------------------------------------
# Gate 8 — there is no vertical excitation at all.
# ---------------------------------------------------------------------------
def test_the_vertical_plane_receives_exactly_zero_noise() -> None:
    r"""Not small — exactly zero, bit for bit, and that is a stated model boundary.

    The photons leave along the direction of motion, so the model gives them no opening
    angle; on a flat lattice ``py = 0`` on the closed orbit and the injected noise has an
    identically zero vertical component. The real floor is the ``1/gamma`` opening angle,
    ``eps_y = (13/55) C_q <beta_y/|rho|^3> / (J_y I2)``, which this model omits by
    construction — the same flat-lattice boundary
    :func:`accsim.radiation.equilibrium_emittance` records from the design side, and the
    one :func:`accsim.radiation.equilibrium_emittances_coupled` fills from coupling.
    """
    sigma, _, injected, _, _ = _cached_sigma(slices=8, **COLD)
    assert np.array_equal(injected[:, 2], np.zeros(len(injected)))
    assert np.array_equal(injected[:, 3], np.zeros(len(injected)))
    _, eps_y, _ = _mode_emittances(sigma)
    assert eps_y == 0.0


def test_the_equilibrium_covariance_is_therefore_singular() -> None:
    """The consequence, said out loud: the equilibrium distribution is 4-dimensional.

    A Cholesky factor of ``Sigma`` does not exist. Any code that samples the equilibrium
    bunch must use an eigen square root — including the settling gate below, which is
    how this was found.
    """
    sigma, _, _, _, _ = _cached_sigma(slices=8, **COLD)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(sigma)
    assert np.array_equal(sigma[2:4, :], np.zeros((2, 6)))  # the vertical block is empty
    assert np.linalg.matrix_rank(sigma, tol=1e-12 * np.abs(sigma).max()) == 4


# ---------------------------------------------------------------------------
# Gate 9 — and it actually settles there, from any starting distribution.
# ---------------------------------------------------------------------------
def test_a_tracked_bunch_settles_on_the_predicted_equilibrium_from_any_start() -> None:
    r"""The roadmap's pre-committed gate: track it and see.

    Two bunches of 300, tracked *together* so they share one map: one starting as a
    point (zero emittance, damping **up** — which only quantum excitation can do) and
    one at three times the equilibrium covariance (damping **down**). Three damping
    times to equilibrate, then the 6x6 covariance averaged over two more.

    The comparison is against the Lyapunov solution of the same map, not against the
    closed form, so this gate measures *settling* and is not confounded with the ``Q_s``
    and lumping owners of gate 6 — which is what lets the ring be chosen hot enough
    (``tau_x = 846`` turns) to afford at all.

    **The budget, stated rather than tuned.** The action decorrelates in ``tau/2``, so
    averaging ``2 tau`` over 300 particles is ~1200 independent samples: 2.0% on a width,
    4.1% on a variance. The gate is 3 sigma of that.
    """
    lat = _ring(**SETTLE)
    tracker = Tracker(lat)
    sigma, _, _, _ = _equilibrium_sigma(lat)
    tau = int(round(_tau_turns(lat)[2]))
    assert 700 < tau < 1000  # the ring is the one the budget below was computed for

    n_each, rng = 300, np.random.default_rng(11)
    values, vectors = np.linalg.eigh(sigma)  # not Cholesky: eps_y = 0, Sigma is singular
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    orbit = _equilibrium_orbit(tracker)
    bunch = np.concatenate(
        [
            np.repeat(orbit[:, None], n_each, axis=1),
            orbit[:, None] + math.sqrt(3.0) * (root @ rng.standard_normal((6, n_each))),
        ],
        axis=1,
    )

    settle, average = 3 * tau, 2 * tau
    accumulated = np.zeros((2, 6, 6))
    for turn in range(settle + average):
        bunch = tracker.track_once(bunch, "quantum", rng)
        if turn >= settle:
            for j, part in enumerate((slice(0, n_each), slice(n_each, 2 * n_each))):
                centred = bunch[:, part] - bunch[:, part].mean(axis=1, keepdims=True)
                accumulated[j] += centred @ centred.T / n_each
    assert np.isfinite(bunch).all()  # nothing left the RF bucket

    independent = n_each * average / (tau / 2.0)
    budget = 1.0 / math.sqrt(2.0 * independent)  # relative error on a width
    predicted_eps, _, _ = _mode_emittances(sigma)
    for j in range(2):
        measured = accumulated[j] / average
        eps_x, eps_y, _ = _mode_emittances(measured)
        assert math.sqrt(measured[5, 5]) == pytest.approx(math.sqrt(sigma[5, 5]), rel=3.0 * budget)
        assert eps_x == pytest.approx(predicted_eps, rel=6.0 * budget)
        assert eps_y < 1.0e-12 * eps_x  # and the vertical never lit up


def test_a_vertically_displaced_bunch_damps_to_nothing_rather_than_to_a_floor() -> None:
    r"""The other half of the ``eps_y = 0`` claim, and the one that is not vacuous.

    "Stays at zero from zero" is bit-for-bit true but weak. Start the bunch *off* the
    vertical axis and the noise on ``py`` is **multiplicative** — it is
    ``py (f - <f>)``, proportional to ``py`` itself — so it perturbs the damping rate and
    leaves the fixed point at zero. The vertical therefore keeps damping through the
    equilibrium instead of stopping at it. Measured over two damping times, against the
    horizontal equilibrium the same photons hold up.
    """
    lat = _ring(**SETTLE)
    tracker = Tracker(lat)
    tau_y = int(round(_tau_turns(lat)[1]))
    orbit = _equilibrium_orbit(tracker)
    rng = np.random.default_rng(5)
    n = 120
    bunch = np.repeat(orbit[:, None], n, axis=1)
    bunch[2] += 1.0e-3
    start = float(np.mean(bunch[2] ** 2))
    for _ in range(2 * tau_y):
        bunch = tracker.track_once(bunch, "quantum", rng)
    assert float(np.mean(bunch[2] ** 2)) < start * math.exp(-3.0)


# ---------------------------------------------------------------------------
# Gate 10 — the model can draw an energy gain, and is deliberately not clamped.
# ---------------------------------------------------------------------------
def test_the_loss_goes_negative_about_as_often_as_the_gaussian_says() -> None:
    r"""~1% of draws are an energy *gain*, and that is a boundary, not a bug.

    With ``n_gamma ~ 24`` photons per magnet the relative fluctuation is
    ``sqrt(4.30/n_gamma) ~ 0.42``, putting ``u < 0`` at 2.4 sigma. Clamping at zero
    would bias the mean **and** the variance by ~1%, which is the size of the
    equilibrium gates above; an unclamped Gaussian keeps both exact. The gate is that
    the measured negative fraction is the normal one — i.e. that nothing clamps.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.zeros(6)
    plain = elem.track(state.copy(), ref, radiation="mean")
    n = 200_000
    rng = np.random.default_rng(31337)
    drawn = elem.track(np.repeat(state[:, None], n, axis=1), ref, radiation="quantum", rng=rng)
    # delta *rose* above the classical answer => less was radiated; above the entry
    # value => a net gain.
    gained = float(np.mean(drawn[5] > state[5]))
    ratio = (plain[5] - state[5]) / drawn[5].std(ddof=1)  # -(mean loss)/sigma, in sigmas
    assert gained == pytest.approx(norm.cdf(ratio), abs=4.0 / math.sqrt(n))
    assert 0.005 < gained < 0.05  # not a tail event, which is why clamping would bias


def test_the_photon_count_per_magnet_is_the_textbook_rate() -> None:
    r"""``n_gamma = (5 / (2 sqrt3)) alpha gamma`` per radian, recovered from the ratio.

    The Gaussian never counts photons, so the count is implicit — but it is recoverable:
    ``(sigma_U/U)^2 = <u^2>/(<u>^2 n_gamma)``, and ``<u^2>/<u>^2 = (11/27)(15 sqrt3/8)^2
    = 4.297``. Checked against the rate computed from ``alpha`` and the bend angle,
    which is the number the "is a Gaussian legitimate here" argument rests on.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    theta = 2.0 * math.pi / (2 * COLD["cells"])
    kappa = theta / L_BEND
    energy = ref.total_energy_eV
    u = energy_loss_per_turn(lat) / 40.0
    implied = 4.297 / (photon_energy_variance(u, energy, kappa, ref) / u**2)
    alpha = ref.classical_radius_m * ref.mass_eV / HBAR_C_EV_M
    expected = 2.5 / math.sqrt(3.0) * alpha * ref.gamma0 * theta
    assert implied == pytest.approx(expected, rel=2e-3)
    assert 14.0 < implied < 20.0  # ~16 photons here, so sqrt(4.297/16) = 0.52 spread


def test_the_variance_is_slicing_invariant_while_the_mean_converges() -> None:
    r"""``U ∝ kappa^2 l`` but ``sigma_U^2 ∝ kappa^3 l`` — a structural gate on the power.

    B2 asserted that slicing converges the *mean* as ``(N-1)/N``. The variance behaves
    differently and that difference is a check on the ``kappa`` power: ``N`` independent
    slices each of ``1/N`` the variance sum to the same total, so a total that moved
    under slicing would mean the wrong power of ``kappa``. Measured here as the injected
    ``D`` before any propagation, so the damping cannot mask it.
    """
    totals = []
    for slices in (1, 2, 8):
        lat = _ring(slices=slices, **COLD)
        orbit = _equilibrium_orbit(Tracker(lat))
        _, _, injected, _ = _envelope(lat, orbit)
        totals.append(float(np.sum(injected[:, 5] ** 2)))
    assert totals[2] == pytest.approx(totals[0], rel=3e-3)
    assert totals[1] == pytest.approx(totals[0], rel=3e-3)


# ---------------------------------------------------------------------------
# Gate 11 — reproducibility, the API, and the degenerate cases.
# ---------------------------------------------------------------------------
def test_the_same_seed_reproduces_bit_for_bit_and_a_different_seed_does_not() -> None:
    """A stochastic track is still an experiment that must be repeatable."""
    lat = _ring(**COLD)
    tracker = Tracker(lat)
    state = np.array([1e-4, 0.0, 1e-4, 0.0, 0.0, 0.0])
    first = tracker.track_once(state.copy(), "quantum", np.random.default_rng(7))
    again = tracker.track_once(state.copy(), "quantum", np.random.default_rng(7))
    other = tracker.track_once(state.copy(), "quantum", np.random.default_rng(8))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_quantum_radiation_without_a_generator_raises_at_every_entry_point() -> None:
    """No hidden global seeding: an unseeded stochastic track is not reproducible.

    Raising rather than defaulting is the same choice B2 made for ``radiation`` without
    ``nonlinear``: silently handing back an answer that looks fine is the failure mode
    this package spends its gates defending against.
    """
    lat = _ring(**COLD)
    tracker = Tracker(lat)
    particle = Particle.from_array(np.zeros(6))
    assert "quantum" in STOCHASTIC_MODELS
    with pytest.raises(ValueError, match="rng"):
        tracker.track(particle, nonlinear=True, radiation="quantum")
    with pytest.raises(ValueError, match="rng"):
        tracker.track_turns(particle, 1, nonlinear=True, radiation="quantum")
    with pytest.raises(ValueError, match="rng"):
        tracker.track_once(np.zeros(6), "quantum")
    with pytest.raises(ValueError, match="rng"):
        elem = next(e for e in lat.elements if isinstance(e, Dipole))
        elem.track(np.zeros(6), lat.ref, radiation="quantum")


def test_a_ring_with_no_bends_has_no_excitation_at_all() -> None:
    """No field, no photons: the drift-and-quadrupole ring injects identically zero."""
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, 3.0e9)
    lat = Lattice([ThinQuadrupole(0.25), ThinQuadrupole(-0.25)], ref=ref)
    tracker = Tracker(lat)
    rng = np.random.default_rng(1)
    state = np.array([1e-3, 1e-4, -1e-3, 2e-4, 0.0, 1e-4])
    assert np.array_equal(
        tracker.track_once(state.copy(), "quantum", rng), tracker.track_once(state.copy(), "mean")
    )


def test_the_quantum_kick_is_distributionally_the_same_for_a_bunch_and_a_particle() -> None:
    """B2's bit-for-bit bunch/particle gate has no quantum analogue, so gate the moments.

    A bunch draws ``n`` numbers where a particle draws one, so the streams differ by
    construction. What must not differ is the distribution they are drawn from.
    """
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.array([5e-4, 1e-4, 0.0, 0.0, 0.0, 2e-4])
    n = 60_000
    rng = np.random.default_rng(1234)
    as_bunch = elem.track(np.repeat(state[:, None], n, axis=1), ref, radiation="quantum", rng=rng)
    rng = np.random.default_rng(4321)
    one_by_one = np.array(
        [elem.track(state.copy(), ref, radiation="quantum", rng=rng)[5] for _ in range(4000)]
    )
    assert one_by_one.mean() == pytest.approx(
        as_bunch[5].mean(), abs=5.0 * as_bunch[5].std() / 60.0
    )
    assert one_by_one.std(ddof=1) == pytest.approx(as_bunch[5].std(ddof=1), rel=0.06)


def test_the_kick_helper_and_the_element_path_agree_under_the_same_draws() -> None:
    """:func:`radiation_kick` is the seam; the element must not carry a second copy."""
    lat = _ring(**COLD)
    ref = lat.ref
    elem = next(e for e in lat.elements if isinstance(e, Dipole))
    state = np.array([1e-3, 2e-4, -5e-4, 3e-4, 1e-3, 7e-4])
    body = elem._track_body(state, ref)
    direct = radiation_kick(elem, state, body.copy(), ref, "quantum", _OnePhoton(0, 1.5))
    through = elem.track(state.copy(), ref, radiation="quantum", rng=_OnePhoton(0, 1.5))
    assert np.array_equal(direct, through)


def test_track_bunch_and_track_bunch_losses_take_the_generator_too() -> None:
    """The bunch entry points radiate like the single-particle path, and demand an rng."""
    from accsim.tracking import Bunch

    lat = _ring(**COLD)
    tracker = Tracker(lat)
    states = np.tile(np.array([[1e-4], [0.0], [1e-4], [0.0], [0.0], [1e-4]]), (1, 5))
    with pytest.raises(ValueError, match="rng"):
        tracker.track_bunch(Bunch(states.copy()), nonlinear=True, radiation="quantum")
    with pytest.raises(ValueError, match="rng"):
        tracker.track_bunch_losses(Bunch(states.copy()), 1, nonlinear=True, radiation="quantum")
    out = tracker.track_bunch(
        Bunch(states.copy()), nonlinear=True, radiation="quantum", rng=np.random.default_rng(3)
    )
    expected = tracker.track_once(states.copy(), "quantum", np.random.default_rng(3))
    assert np.array_equal(out.states, expected)


def test_quantum_is_advertised_but_the_wrong_map_is_not() -> None:
    """``"quantum"`` joins the offered models; ``"mean_delta_only"`` stays hidden."""
    assert "quantum" in RADIATION_MODELS
    lat = _ring(**COLD)
    with pytest.raises(ValueError, match="quantum"):
        Tracker(lat).track_once(np.zeros(6), "quantom", np.random.default_rng(1))


def test_radiation_with_quantum_is_still_not_symplectic() -> None:
    """Dissipation plus diffusion: the map must fail the check, and does.

    Asserted rather than worked around, exactly as B2 asserts it for the mean kick.
    """
    from accsim.symplectic import is_symplectic_map

    lat = _ring(**COLD)
    tracker = Tracker(lat)
    rng = np.random.default_rng(2)
    assert not is_symplectic_map(
        lambda s: tracker.track_once(s, "quantum", rng), np.zeros(6), atol=1e-9
    )
