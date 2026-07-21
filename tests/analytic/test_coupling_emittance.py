r"""Analytic gates — vertical (mode-2) equilibrium emittance from betatron coupling.

The horizontal quantum excitation of a flat radiating ring (which alone sets
``eps_x``) is shared between the two coupled normal modes when a skew quadrupole
couples ``x`` and ``y``. :func:`equilibrium_emittances_coupled` returns the eigen-mode
emittances

    G = sqrt(Delta^2 + |C^-|^2),
    eps_1 = eps_x0 (G + Delta) / (2 G),   eps_2 = eps_x0 (G - Delta) / (2 G),

with ``eps_x0`` the coupling-off horizontal emittance, ``Delta`` the decoupled
difference-resonance detuning, and ``|C^-|`` the coupling strength
(:func:`closest_tune_approach`). ``eps_2`` (the smaller, y-like mode) is the vertical
emittance from coupling. The sharing coefficient is **not** a symbolic closed form —
it is pinned against xtrack's radiation-envelope eigen-emittances in
``tests/reference/test_coupling_emittance_xtrack.py``. Here are the gates the analytic
suite can own without the envelope:

  1. **Sum conservation** ``eps_1 + eps_2 == eps_x0`` (exact) — coupling redistributes
     the horizontal excitation, adds none. The structural gate.
  2. **No coupling → no vertical emittance** ``(eps_x0, 0)`` exactly.
  3. **Mode gap ties the emittance split** ``g == Delta * eps_x0 / (eps_1 - eps_2)`` with
     ``g`` the coupled normal-mode tune split from :func:`normal_mode_tunes`
     (eigenvalues of the 4x4) — an independent path from the ``closest_tune_approach``
     phasor sum in the formula. Written on ``G`` (not the ratio), so it does not amplify
     the ~1e-3 gap agreement the way ``eps_2/eps_1`` would.
  4. **Far off resonance** ``eps_2/eps_1 -> |C^-|^2/(4 Delta^2)`` — the ``1/4`` asymptote
     that separates the correct **eigen-mode** sharing from the projected ``1/2`` and
     raw ``1/1`` forms (the roadmap's pre-committed ``|C^-|^2/(|C^-|^2+Delta^2)`` was the
     ``1/1``, 4x too large).
  5. **Near resonance → equal sharing** ``eps_2/eps_1 -> 1`` as ``|C^-| >> Delta``.
  6. **Monotonic** — ``eps_2`` grows, ``eps_1`` shrinks, with the skew strength.
  7. **Symbolic** — the ``1/4`` asymptote coefficient by series (not a remembered
     constant), plus the ``(G ∓ Delta)/2G == cos^2/sin^2 phi`` half-angle identity.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    closest_tune_approach,
    equilibrium_emittance,
    equilibrium_emittances_coupled,
    normal_mode_tunes,
    tunes,
)

ELECTRON_MASS_EV = 0.51099895069e6


def _ring(
    kd: float,
    n_cells: int = 48,
    l_bend: float = 2.0,
    ldrift: float = 0.4,
    kf: float = 0.32,
    energy_eV: float = 1.0e9,
) -> Lattice:
    """A weak separated-function electron FODO ring (radiating: dipoles present).

    The ``kf``/``kd`` asymmetry sets the decoupled tune split (the detuning
    ``Delta``); weak bends keep ``J_x ≈ 1`` so the equal-damping sharing model is clean.
    """
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, energy_eV)
    angle = 2.0 * math.pi / (2 * n_cells)
    cell = [
        ThinQuadrupole(0.5 * kf),
        Drift(ldrift),
        Dipole(l_bend, angle),
        Drift(ldrift),
        ThinQuadrupole(kd),
        Drift(ldrift),
        Dipole(l_bend, angle),
        Drift(ldrift),
        ThinQuadrupole(0.5 * kf),
    ]
    return Lattice(cell * n_cells, ref=ref)


def _dist_int(x: float) -> float:
    return abs(x - round(x))


def _delta(lat0: Lattice) -> float:
    qx, qy = tunes(lat0)
    return _dist_int(qx - qy)


# a near-resonance ring (small Delta) and an off-resonance ring (larger Delta); both
# uncoupled + stable, thin skew added per test.
NEAR = _ring(kd=-0.27)  # Delta ~ 0.005
OFF = _ring(kd=-0.32)  # Delta ~ 0.10


def _with_skew(base: Lattice, k1sl: float) -> Lattice:
    return Lattice([ThinSkewQuadrupole(k1sl), *base.elements], ref=base.ref)


def test_rings_stable_and_split() -> None:
    """Sanity: both bare rings are uncoupled-stable with the intended detunings."""
    d_near, d_off = _delta(NEAR), _delta(OFF)
    assert d_near < 0.02 < d_off  # NEAR near the resonance, OFF away from it
    assert equilibrium_emittance(OFF) > 0.0 and equilibrium_emittance(NEAR) > 0.0


def test_sum_conserved() -> None:
    """eps_1 + eps_2 == eps_x0 exactly — coupling shares, never creates, excitation."""
    eps_x0 = equilibrium_emittance(OFF)
    for k1sl in (0.01, 0.03, 0.06):
        e1, e2 = equilibrium_emittances_coupled(_with_skew(OFF, k1sl))
        assert e1 + e2 == pytest.approx(eps_x0, rel=1e-12)
        assert e1 >= e2 > 0.0  # mode 1 is the larger (x-like) mode off resonance


def test_no_coupling_zero_vertical() -> None:
    """With no skew, all excitation stays horizontal: (eps_x0, 0) exactly."""
    e1, e2 = equilibrium_emittances_coupled(OFF)
    assert e1 == pytest.approx(equilibrium_emittance(OFF), rel=1e-12)
    assert e2 == 0.0


def test_mode_gap_ties_emittance_split() -> None:
    """g == Delta * eps_x0 / (eps_1 - eps_2), g the eigenvalue mode-tune split.

    ``G = eps_x0 Delta / (eps_1 - eps_2)`` is an identity of the closed form; asserting
    the *independent* eigen gap ``g`` (:func:`normal_mode_tunes`, eigenvalues of the
    4x4) equals it ties the sharing to the coupled mode geometry — the emittance
    analogue of G1's "closed form == eigen gap". Written on ``G`` (not the ratio) so
    the gap agreement is not amplified. Weak coupling only, where the leading-order
    ``closest_tune_approach`` |C^-| tracks the exact eigen gap to <1% (off resonance it
    drifts further — that is a G1 property, not this milestone's).
    """
    for base in (NEAR, OFF):
        eps_x0 = equilibrium_emittance(base)
        d = _delta(base)
        for k1sl in (0.006, 0.012):
            lat = _with_skew(base, k1sl)
            e1, e2 = equilibrium_emittances_coupled(lat)
            q1, q2 = normal_mode_tunes(lat)
            g = _dist_int(q1 - q2)
            assert g == pytest.approx(d * eps_x0 / (e1 - e2), rel=1e-2)


def test_far_off_resonance_asymptote() -> None:
    """eps_2/eps_1 -> |C^-|^2 / (4 Delta^2) when |C^-| << Delta (the eigen-mode form)."""
    d = _delta(OFF)
    k1sl = 0.006  # weak: |C^-| well below Delta
    lat = _with_skew(OFF, k1sl)
    e1, e2 = equilibrium_emittances_coupled(lat)
    cminus = closest_tune_approach(lat)
    assert cminus < 0.2 * d  # in the asymptotic regime
    assert e2 / e1 == pytest.approx(cminus**2 / (4.0 * d**2), rel=2e-2)
    # decisively NOT the projected |C^-|^2/(2 Delta^2) or raw |C^-|^2/Delta^2 forms
    assert e2 / e1 < 0.7 * (cminus**2 / (2.0 * d**2))


def test_near_resonance_approaches_equal_sharing() -> None:
    """As |C^-| >> Delta the modes equalise: eps_2/eps_1 -> 1 (near the resonance)."""
    d = _delta(NEAR)
    lat = _with_skew(NEAR, 0.05)  # |C^-| several x Delta
    cminus = closest_tune_approach(lat)
    assert cminus > 5.0 * d
    e1, e2 = equilibrium_emittances_coupled(lat)
    assert e2 / e1 > 0.6  # substantial equalisation


def test_monotonic_in_coupling() -> None:
    """Stronger skew -> more vertical, less horizontal emittance."""
    prev2, prev1 = -1.0, 1e9
    for k1sl in (0.005, 0.01, 0.02, 0.04):
        e1, e2 = equilibrium_emittances_coupled(_with_skew(OFF, k1sl))
        assert e2 > prev2 and e1 < prev1
        prev2, prev1 = e2, e1


def test_far_off_resonance_coefficient_is_a_quarter_symbolic() -> None:
    """Symbolically pin eps_2/eps_1 -> |C^-|^2/(4 Delta^2) (the eigen-mode ``1/4``).

    The ``1/4`` separates the correct eigen-mode sharing from the projected ``1/2`` and
    raw ``1/1`` forms. Series, not a remembered constant. Plus the half-angle identity
    the shipped ``(G ∓ Delta)/2G`` coefficients rest on.
    """
    d, c = sp.symbols("Delta C", positive=True)
    g = sp.sqrt(d**2 + c**2)
    lead = sp.series((g - d) / (g + d), c, 0, 3).removeO()
    assert sp.simplify(lead - c**2 / (4 * d**2)) == 0
    cos2phi = d / g  # tan(2 phi) = C/Delta, Delta>0 -> cos(2 phi) = Delta/G
    assert sp.simplify((g + d) / (2 * g) - (1 + cos2phi) / 2) == 0
    assert sp.simplify((g - d) / (2 * g) - (1 - cos2phi) / 2) == 0


def test_returns_plain_floats() -> None:
    e1, e2 = equilibrium_emittances_coupled(_with_skew(OFF, 0.02))
    assert type(e1) is float and type(e2) is float
