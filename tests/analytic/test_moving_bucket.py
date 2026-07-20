"""Analytic checks for the *moving* (accelerating) RF bucket.

For ``sin phi_s != 0`` the ``-zeta sin phi_s`` tilt in ``U`` makes the bucket
asymmetric about ``zeta = 0`` and shrinks it. The gates, layered so a wrong
fixed-point choice and a wrong height formula cannot cancel:

1. **Height** against a closed form **derived symbolically here from accsim's own
   ``H``** — never against another call of the same code::

       delta_max(phi_s)^2 / delta_max(stationary)^2 = cos(psi) - (pi/2 - psi) sin psi,
       psi = asin |sin phi_s|

   on all **four** branches (proton/electron x below/above transition). The
   electron ones matter: the bounding unstable fixed point there is *not* the
   ``k_rf zeta_u = 2 phi_s - pi`` member the stationary-only code assumed, and a
   negative control asserts that naive choice is measurably wrong.
2. **Separatrix** is the level set through ``zeta_u``, spanning to the *far turning
   point* — a transcendental root, genuinely asymmetric, and narrower than the
   stationary bucket.
3. **Bounded/unbounded tracking** straddling that true far turning point, over
   many synchrotron periods — the closed-form-free leg.
4. The ``phi_s``-on-the-unstable-branch ``ValueError``, for both ``qV > 0`` and
   ``qV < 0``.

Bucket **area** is deliberately not provided: it is a non-elementary integral, and
the folklore ``(1 - sin phi_s)/(1 + sin phi_s)`` is itself an approximation, so
there is nothing to gate it against exactly. See ``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    CLIGHT,
    DELTA,
    ZETA,
    Dipole,
    Drift,
    Lattice,
    Particle,
    ReferenceParticle,
    RFCavity,
    ThinQuadrupole,
    Tracker,
    longitudinal_hamiltonian,
    rf_bucket_height,
    separatrix,
    slip_factor,
    synchronous_phase,
)

PROTON_MASS = 938.27208816e6
ELECTRON_MASS = 0.51099895e6
HARMONIC = 10
VOLTAGE = 2.0e6


def _proton(gamma: float = 5.0) -> ReferenceParticle:
    return ReferenceParticle.from_gamma(PROTON_MASS, gamma, charge=1.0)


def _electron(gamma: float = 5.0) -> ReferenceParticle:
    """A light ``qV < 0`` reference. ``gamma`` matches the proton's so the two arcs
    keep the same ``eta``; the electron branches then differ *only* in ``sign(qV)``,
    which is precisely the thing the fixed-point selection keys on."""
    return ReferenceParticle.from_gamma(ELECTRON_MASS, gamma, charge=-1.0)


def _below_arc() -> list:
    """Bend-free cell: alpha_c = 0 => eta = -1/gamma0^2 < 0 (below transition)."""
    return [Drift(1.0), ThinQuadrupole(0.3), Drift(1.0), ThinQuadrupole(-0.3)]


def _above_arc() -> list:
    """Dispersive arc: alpha_c > 1/gamma0^2 => eta > 0 (above transition)."""
    return [
        ThinQuadrupole(0.5 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(-1.0 / 2.5),
        Dipole(1.0, 0.15),
        ThinQuadrupole(0.5 / 2.5),
    ]


def _voltage_for(ref: ReferenceParticle) -> float:
    """Keep ``qV / (beta0^2 E0)`` modest on both refs.

    The gates below are ratios and so are voltage-independent, but the *tracking*
    one is not: it needs ``delta_max`` small enough for linear optics to hold and
    ``Qs`` large enough that 1e4 turns spans many synchrotron periods. The electron
    rest mass is ~1800x smaller, so it needs a correspondingly smaller voltage —
    both are asserted directly in the tracking test rather than assumed here.
    """
    return VOLTAGE if ref.charge > 0 else 1.0e3


def _lattice(
    ref: ReferenceParticle, arc: list, phi_s: float, voltage: float | None = None
) -> Lattice:
    circumference = sum(e.length for e in arc)
    freq = HARMONIC * ref.beta0 * CLIGHT / circumference
    return Lattice(
        [*arc, RFCavity(_voltage_for(ref) if voltage is None else voltage, freq, phi_s)], ref
    )


def _branch_phi_s(
    ref: ReferenceParticle, arc: list, s: float, voltage: float | None = None
) -> float:
    """The stable ``phi_s`` with ``|sin phi_s| = s`` on this lattice's branch.

    The energy gain is **positive** on every branch — the RF supplies energy, which
    is the physical case (an electron store ring replenishing ``U0``). With
    ``qV < 0`` that forces ``sin phi_s < 0``, which is exactly why the bounding
    unstable fixed point moves off the hardcoded ``(2 phi_s - pi)/k_rf``.
    """
    v = _voltage_for(ref) if voltage is None else voltage
    above = slip_factor(_lattice(ref, arc, 0.0, v)) > 0.0
    gain = s * abs(ref.charge * v)
    return synchronous_phase(v, gain, above_transition=above, charge=ref.charge)


# (label, ref factory, arc factory) — the four branches.
BRANCHES = [
    ("proton below", _proton, _below_arc),
    ("proton above", _proton, _above_arc),
    ("electron below", _electron, _below_arc),
    ("electron above", _electron, _above_arc),
]
SIN_PHI_S = [0.019, 0.2, 0.5, 0.8]


def _symbolic_height_ratio():
    """``f(psi)`` derived from accsim's own ``U``, not a remembered constant.

    Returns a callable ``f(psi)``. The derivation mirrors the module docstring of
    ``accsim.longitudinal``: unstable fixed points of ``dU/dzeta``, the bucket is
    the inner one, and the ratio is taken against the branch's stationary bucket.
    """
    sp = pytest.importorskip("sympy")
    A, k, eta, C, zeta, phi = sp.symbols("A k eta C zeta phi", real=True)

    def U(z, p):
        return -A * (sp.cos(p - k * z) / k - z * sp.sin(p))

    # the unstable fixed points: dU/dzeta = 0 away from k zeta = 2 pi n
    dU = sp.simplify(sp.diff(U(zeta, phi), zeta))
    assert sp.simplify(dU.subs(zeta, 0)) == 0  # zeta = 0 is the stable centre
    assert sp.simplify(dU.subs(zeta, (2 * phi - sp.pi) / k)) == 0  # an unstable one

    psi = sp.Symbol("psi", positive=True)
    # Below-transition proton branch: phi_s = psi, bounding FP k zeta_u = 2 psi - pi,
    # stationary reference phi_s = 0 (k zeta_u = -pi). Both use the same U.
    d2 = lambda zu, p: 2 * (U(0, p) - U(zu, p)) / (eta * C)  # noqa: E731
    ratio = sp.simplify(d2((2 * psi - sp.pi) / k, psi) / d2(-sp.pi / k, 0))
    f = sp.cos(psi) - (sp.pi / 2 - psi) * sp.sin(psi)
    assert sp.simplify(sp.expand(ratio - f)) == 0, f"symbolic ratio != f(psi): {ratio}"
    assert f.subs(psi, 0) == 1 and sp.simplify(f.subs(psi, sp.pi / 2)) == 0
    return sp.lambdify(psi, f, "math")


@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_height_ratio_matches_symbolic_closed_form(label, make_ref, make_arc) -> None:
    """The one gate that has teeth: code vs. an independently derived expression."""
    f = _symbolic_height_ratio()
    ref, arc = make_ref(), make_arc()
    stationary = rf_bucket_height(_lattice(ref, arc, _branch_phi_s(ref, arc, 0.0)))
    for s in SIN_PHI_S:
        phi_s = _branch_phi_s(ref, arc, s)
        assert abs(math.sin(phi_s)) == pytest.approx(s, rel=1e-12)
        moving = rf_bucket_height(_lattice(ref, arc, phi_s))
        assert (moving / stationary) ** 2 == pytest.approx(f(math.asin(s)), rel=1e-12), label


@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_moving_bucket_is_smaller_and_collapses_at_zero_gain(label, make_ref, make_arc) -> None:
    ref, arc = make_ref(), make_arc()
    stationary = rf_bucket_height(_lattice(ref, arc, _branch_phi_s(ref, arc, 0.0)))
    heights = [rf_bucket_height(_lattice(ref, arc, _branch_phi_s(ref, arc, s))) for s in SIN_PHI_S]
    assert all(h < stationary for h in heights), label
    assert heights == sorted(heights, reverse=True), label  # monotonically shrinking
    # exact collapse to the Stage-3 stationary bucket at zero gain
    assert rf_bucket_height(_lattice(ref, arc, _branch_phi_s(ref, arc, 0.0))) == stationary


def test_naive_fixed_point_is_wrong_for_negative_qv() -> None:
    """Negative control for the third fix: ``k zeta_u = 2 phi_s - pi`` is NOT general.

    On both ``qV < 0`` (electron) branches the bucket is bounded by the *other*
    adjacent unstable point. Using the hardcoded member would return a height that
    is too large — assert it genuinely differs, so the general selection cannot be
    reverted without this failing.
    """
    from accsim.longitudinal import _adjacent_unstable_zetas, _bucket_bounds, _effective_cavity

    f = _symbolic_height_ratio()
    for label, make_ref, make_arc in BRANCHES:
        ref, arc = make_ref(), make_arc()
        phi_s = _branch_phi_s(ref, arc, 0.5)
        lat = _lattice(ref, arc, phi_s)
        cav = _effective_cavity(lat)
        eta, C = slip_factor(lat), lat.length
        zeta_u, _, delta2 = _bucket_bounds(cav, eta, C)
        naive = (2.0 * phi_s - math.pi) / cav.k_rf

        # the naive point is always *an* unstable fixed point ...
        assert any(math.isclose(naive, z, rel_tol=1e-12) for z in _adjacent_unstable_zetas(cav)), (
            f"{label}: naive zeta_u is not even adjacent"
        )
        # ... but only on the qV > 0 branches is it the bounding one.
        agrees = math.isclose(naive, zeta_u, rel_tol=1e-12)
        assert agrees == (ref.charge > 0), f"{label}: naive-vs-bounding mismatch"

        stationary = rf_bucket_height(_lattice(ref, arc, _branch_phi_s(ref, arc, 0.0)))
        assert delta2 / stationary**2 == pytest.approx(f(math.asin(0.5)), rel=1e-12)
        if not agrees:
            ham = longitudinal_hamiltonian(lat)
            naive_d2 = 2.0 * (ham(0.0, 0.0) - ham(naive, 0.0)) / (eta * C)
            assert naive_d2 > delta2 * 1.05, f"{label}: naive choice is not detectably wrong"


# --- separatrix: asymmetric, and its own level set ------------------------------
@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_separatrix_is_an_asymmetric_level_set(label, make_ref, make_arc) -> None:
    ref, arc = make_ref(), make_arc()
    lat = _lattice(ref, arc, _branch_phi_s(ref, arc, 0.5))
    ham = longitudinal_hamiltonian(lat)
    zeta, delta = separatrix(lat, n_points=301)

    h_vals = np.array([ham(z, d) for z, d in zip(zeta, delta, strict=True)])
    h_ref = ham(float(zeta[0]), 0.0)  # a tip
    assert np.allclose(h_vals, h_ref, rtol=0, atol=1e-9 * abs(h_ref)), label

    # the peak of the curve is the bucket height, attained at the centre zeta = 0
    assert np.max(np.abs(delta)) == pytest.approx(rf_bucket_height(lat), rel=1e-4), label
    assert abs(zeta[np.argmax(np.abs(delta))]) < 0.02 * np.ptp(zeta), label

    # genuinely asymmetric: the two tips are NOT mirror images about zeta = 0
    lo, hi = float(np.min(zeta)), float(np.max(zeta))
    assert not math.isclose(hi, -lo, rel_tol=0.05), f"{label}: separatrix still symmetric"
    # ... and narrower than the stationary bucket's full RF wavelength
    k_rf = lat.elements[-1].k_rf(ref)
    assert hi - lo < 2.0 * math.pi / k_rf, label


@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_separatrix_recovers_symmetry_at_zero_gain(label, make_ref, make_arc) -> None:
    ref, arc = make_ref(), make_arc()
    lat = _lattice(ref, arc, _branch_phi_s(ref, arc, 0.0))
    zeta, _ = separatrix(lat, n_points=101)
    lo, hi = float(np.min(zeta)), float(np.max(zeta))
    assert hi == pytest.approx(-lo, rel=1e-12), label
    assert hi - lo == pytest.approx(2.0 * math.pi / lat.elements[-1].k_rf(ref), rel=1e-12), label


# --- the unstable branch must raise, on BOTH signs of qV ------------------------
@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_unstable_branch_has_no_bucket(label, make_ref, make_arc) -> None:
    """The other root of ``sin phi_s = s`` gives the same energy gain and no bucket."""
    from accsim import energy_gain_per_turn

    ref, arc = make_ref(), make_arc()
    stable_phi = _branch_phi_s(ref, arc, 0.5)
    unstable_phi = math.pi - stable_phi  # same sin, opposite sign of cos
    assert math.sin(unstable_phi) == pytest.approx(math.sin(stable_phi), rel=1e-12)

    stable_lat, unstable_lat = _lattice(ref, arc, stable_phi), _lattice(ref, arc, unstable_phi)
    assert energy_gain_per_turn(unstable_lat) == pytest.approx(
        energy_gain_per_turn(stable_lat), rel=1e-12
    ), f"{label}: the two roots must be indistinguishable by energy gain"

    assert rf_bucket_height(stable_lat) > 0.0
    with pytest.raises(ValueError, match="no stable RF bucket"):
        rf_bucket_height(unstable_lat)
    with pytest.raises(ValueError, match="no stable RF bucket"):
        separatrix(unstable_lat)


# --- tracking: the closed-form-free leg -----------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(("label", "make_ref", "make_arc"), BRANCHES, ids=[b[0] for b in BRANCHES])
def test_bounded_inside_unbounded_outside_moving_bucket(label, make_ref, make_arc) -> None:
    """Straddle the height at the centre; the escape test uses the TRUE far tip."""
    from accsim import synchrotron_tune

    ref, arc = make_ref(), make_arc()
    lat = _lattice(ref, arc, _branch_phi_s(ref, arc, 0.4))
    dmax = rf_bucket_height(lat)
    zeta, _ = separatrix(lat, n_points=401)
    lo, hi = float(np.min(zeta)), float(np.max(zeta))
    width = hi - lo

    turns = 10_000
    # Self-guard: an under-resolved run would "confine" an outside particle simply
    # by not tracking long enough for it to leave, and a huge delta_max would break
    # the linear optics the arc is built from. Assert both, don't assume them.
    periods = float(synchrotron_tune(lat)) * turns
    assert periods > 20.0, f"{label}: only {periods:.2f} synchrotron periods tracked"
    assert dmax < 0.05, f"{label}: delta_max = {dmax:.3g} too large for linear optics"

    inside = Tracker(lat).track_turns(Particle(zeta=0.0, delta=0.85 * dmax), 10_000, nonlinear=True)
    z_in = inside[:, ZETA]
    assert z_in.min() > lo - 0.05 * width, label  # librates within the asymmetric bucket
    assert z_in.max() < hi + 0.05 * width, label
    assert np.max(np.abs(inside[:, DELTA])) < 1.05 * dmax, label

    outside = Tracker(lat).track_turns(Particle(zeta=0.0, delta=1.2 * dmax), 10_000, nonlinear=True)
    assert np.max(np.abs(outside[:, ZETA])) > 20.0 * width, label  # rotates, runs away


@pytest.mark.slow
def test_asymmetric_bucket_is_not_bounded_by_the_mirror_point() -> None:
    """The far tip is a transcendental root, not ``-zeta_u``: tracking must show it.

    A particle launched between ``-zeta_u`` and the true far tip on the wide side
    is still *inside* the bucket, which the old mirror-symmetric bound would have
    called outside (or vice versa on the narrow side).
    """
    ref, arc = _proton(), _below_arc()
    lat = _lattice(ref, arc, _branch_phi_s(ref, arc, 0.6))
    zeta, _ = separatrix(lat, n_points=801)
    lo, hi = float(np.min(zeta)), float(np.max(zeta))
    zeta_u = lo if abs(lo) > abs(hi) else hi
    mirror = -zeta_u
    far = hi if zeta_u == lo else lo
    assert abs(far) < abs(mirror), "the far tip must be inside the mirror point"

    # midway between the true far tip and the (wrong) mirror point, on the axis
    probe = 0.5 * (far + mirror)
    traj = Tracker(lat).track_turns(Particle(zeta=probe, delta=0.0), 10_000, nonlinear=True)
    assert np.max(np.abs(traj[:, ZETA])) > 20.0 * (hi - lo), (
        "a point beyond the true far tip must be unbounded — the mirror point is not the bound"
    )
