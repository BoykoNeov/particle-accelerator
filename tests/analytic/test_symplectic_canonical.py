"""The canonical longitudinal variable, and symplecticity tested in it.

accsim's longitudinal pair ``(zeta, delta)`` is not canonically conjugate — the
partner of ``zeta = s - beta0 c t`` is the energy-like ``p_zeta``, not the momentum
deviation ``delta``. For every **linear** element that never matters: a linear drift
is three independent shear blocks and a shear is symplectic in whatever pair it acts
on. It starts mattering the moment a map is exact in ``delta``, which is what
:func:`~accsim.symplectic.is_symplectic_map_canonical` is for.

The maps here are written **locally**, not taken from ``accsim.elements``: the
subject is the *checker*, so it needs a map already known to be symplectic and a
map already known not to be, independent of whatever the element library does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    delta_from_pzeta,
    is_symplectic,
    is_symplectic_map,
    is_symplectic_map_canonical,
    jacobian,
    pzeta_from_delta,
)
from accsim.coords import DELTA, DIM, PX, PY
from accsim.symplectic import J6, from_canonical, to_canonical

# --------------------------------------------------------------------------
# Local reference maps: an exact drift, and the tempting half-fix
# --------------------------------------------------------------------------
L_REF = 2.0


def _exact_drift(state: np.ndarray, ref: ReferenceParticle, length: float = L_REF) -> np.ndarray:
    """The Hamiltonian-exact field-free map. Symplectic by construction."""
    out = np.asarray(state, dtype=float).copy()
    px, py, delta = out[PX], out[PY], out[DELTA]
    one_plus = 1.0 + delta
    pz = math.sqrt(one_plus * one_plus - px * px - py * py)
    P = ref.momentum_eV * one_plus
    beta_p = P / math.hypot(P, ref.mass_eV)
    out[0] += length * px / pz
    out[2] += length * py / pz
    out[4] += length * (1.0 - (ref.beta0 / beta_p) * one_plus / pz)
    return out


def _half_fixed_drift(
    state: np.ndarray, ref: ReferenceParticle, length: float = L_REF
) -> np.ndarray:
    """Transverse motion exact, ``zeta`` left linear — the plausible wrong answer.

    This is the map you get by "adding the ``1/pz``" to the transverse coordinates
    and stopping. It is **not** symplectic, and the whole point of the canonical
    check is that it says so.
    """
    out = np.asarray(state, dtype=float).copy()
    px, py, delta = out[PX], out[PY], out[DELTA]
    pz = math.sqrt((1.0 + delta) ** 2 - px * px - py * py)
    out[0] += length * px / pz
    out[2] += length * py / pz
    out[4] += length / ref.gamma0**2 * delta
    return out


def _state(amp: float) -> np.ndarray:
    """One off-axis state. Angles must be nonzero, or every map here coincides."""
    return np.array([amp, amp, -amp, 0.7 * amp, amp, amp])


# --------------------------------------------------------------------------
# 1. The coordinate change itself
# --------------------------------------------------------------------------


def test_pzeta_matches_its_definition(proton_gamma5: ReferenceParticle) -> None:
    """``p_zeta = (E - E0) / (beta0^2 E0)``, against a direct evaluation.

    The implementation avoids subtracting two nearly-equal energies; this computes
    it the naive way at a ``delta`` large enough that the naive way is still
    accurate, so the two arithmetics are independent.
    """
    ref = proton_gamma5
    E0, m, P0 = ref.total_energy_eV, ref.mass_eV, ref.momentum_eV
    for delta in (1.0e-3, 1.0e-2, 0.1, -0.05):
        E = math.hypot(P0 * (1.0 + delta), m)
        expected = (E - E0) / (ref.beta0**2 * E0)
        assert pzeta_from_delta(delta, ref) == pytest.approx(expected, rel=1e-12)


def test_pzeta_equals_delta_to_first_order_and_differs_at_second() -> None:
    r"""``p_zeta = delta + delta^2 / (2 gamma0^2) + O(delta^3)``, coefficient pinned.

    The two longitudinal variables agree to **first** order at *every* energy, not
    only ultrarelativistically — ``dE/E = beta^2 dp/p`` makes the leading
    coefficient exactly 1. Expanding
    ``p_zeta = (E - E0)/(beta0^2 E0)`` with ``E = E0 sqrt(1 + beta0^2 (2 delta + delta^2))``
    gives ``delta + (delta^2/2)(1 - beta0^2) = delta + delta^2/(2 gamma0^2)``.

    That is why the distinction hides so well: it is second order *and* suppressed
    by ``1/gamma0^2``, so it vanishes twice over in the ultrarelativistic limit this
    package mostly lives in. Asserting the coefficient — rather than "the two are
    close" — is what makes this a check on the variable and not on a tolerance.
    """
    from accsim import PROTON_MASS_EV

    for gamma0 in (2.0, 5.0, 50.0):
        ref = ReferenceParticle.from_gamma(PROTON_MASS_EV, gamma0)
        for delta in (1.0e-4, 1.0e-3):
            excess = pzeta_from_delta(delta, ref) - delta
            assert excess / delta**2 == pytest.approx(1.0 / (2.0 * gamma0**2), rel=1e-3)

    # Ultrarelativistically the second-order term is gone too, so the two coincide.
    ultra = ReferenceParticle.from_gamma(PROTON_MASS_EV, 1.0e6)
    assert pzeta_from_delta(1.0e-3, ultra) == pytest.approx(1.0e-3, rel=1e-12)


def test_the_coordinate_change_round_trips(proton_gamma5: ReferenceParticle) -> None:
    """``delta -> p_zeta -> delta`` to full double precision, over many decades.

    Both directions are written cancellation-free, so a small ``delta`` must not
    lose digits — that is asserted here rather than assumed, because a lossy
    round-trip would show up in the canonical check as a spurious residual and be
    mistaken for a non-symplectic map.
    """
    ref = proton_gamma5
    for delta in (1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 0.05, -0.05, -1.0e-9):
        back = delta_from_pzeta(pzeta_from_delta(delta, ref), ref)
        assert back == pytest.approx(delta, rel=1e-13, abs=1e-24)


def test_zero_maps_to_zero(proton_gamma5: ReferenceParticle) -> None:
    """The reference particle is the origin in both variables, exactly."""
    assert pzeta_from_delta(0.0, proton_gamma5) == 0.0
    assert delta_from_pzeta(0.0, proton_gamma5) == 0.0


def test_the_state_transforms_touch_only_the_last_coordinate(
    proton_gamma5: ReferenceParticle,
) -> None:
    """A change of longitudinal variable is not licence to disturb the other five."""
    st = _state(1.0e-3)
    can = to_canonical(st, proton_gamma5)
    np.testing.assert_array_equal(can[:DELTA], st[:DELTA])
    assert can[DELTA] != st[DELTA]
    np.testing.assert_allclose(from_canonical(can, proton_gamma5), st, rtol=1e-13, atol=0.0)


# --------------------------------------------------------------------------
# 2. Linear maps: the two checks agree, because the distinction is invisible there
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "element",
    [Drift(1.7), Quadrupole(0.3, 1.2), Quadrupole(0.3, -1.2), Sextupole(0.4, 0.0)],
    ids=["drift", "quad_f", "quad_d", "thick_sext_k2_zero"],
)
def test_a_linear_matrix_passes_in_both_coordinate_systems(
    element, proton_gamma5: ReferenceParticle
) -> None:
    """A linear matrix is symplectic in ``(zeta, delta)`` *and* in ``(zeta, p_zeta)``.

    The existing gates all live in the first, and this says adding the second breaks
    none of them: a shear in ``(zeta, delta)`` is still a shear after the variable
    change, only with a different coefficient.

    The map under test is each element's ``matrix``, applied as a map — deliberately
    not its ``track``. A :class:`~accsim.elements.drift.Drift`'s ``track`` is the
    **exact** map, which is symplectic but fails the ``(zeta, delta)`` check; that is
    the subject of the next section, and conflating the two here would turn this test
    into a statement about which elements happen to be linear today.
    """
    ref = proton_gamma5
    M = element.matrix(ref)
    assert is_symplectic(M)
    st = _state(1.0e-3)
    assert is_symplectic_map(lambda s: M @ s, st)
    assert is_symplectic_map_canonical(lambda s: M @ s, st, ref)


def test_the_canonical_check_is_not_vacuously_true(proton_gamma5: ReferenceParticle) -> None:
    """A map that is not symplectic in *any* variables is still rejected.

    Without this, "everything passes" could mean the coordinate change had quietly
    turned the test into a tautology. A drift with the wrong ``R34`` is the control.
    """
    ref = proton_gamma5

    def broken(state: np.ndarray) -> np.ndarray:
        out = np.asarray(state, dtype=float).copy()
        out[0] += L_REF * out[PX]
        out[2] += 1.5 * L_REF * out[PY]  # wrong: no longer a unit shear
        out[PY] *= 1.2  # ...and the pair no longer has unit determinant
        return out

    st = _state(1.0e-3)
    assert not is_symplectic_map(broken, st)
    assert not is_symplectic_map_canonical(broken, st, ref)


# --------------------------------------------------------------------------
# 3. The exact map: what each check says, and why only one of them may be used
# --------------------------------------------------------------------------


def test_the_exact_map_is_symplectic_only_in_the_canonical_variables(
    proton_gamma5: ReferenceParticle,
) -> None:
    """The headline: ``(zeta, delta)`` rejects a map that *is* symplectic.

    The exact drift is the flow of a Hamiltonian, so it is symplectic — and the
    canonical check confirms it at amplitudes spanning four decades, while the
    ``(zeta, delta)`` check rejects it at all but the smallest. That is the whole
    reason :func:`is_symplectic_map_canonical` exists, and asserting the *rejection*
    is as important as asserting the acceptance: it is what stops someone
    "repairing" an exact map until the wrong check goes green.
    """
    ref = proton_gamma5
    for amp in (1.0e-3, 1.0e-2, 5.0e-2):
        st = _state(amp)
        assert is_symplectic_map_canonical(lambda s: _exact_drift(s, ref), st, ref)
        assert not is_symplectic_map(lambda s: _exact_drift(s, ref), st)


def test_the_zeta_delta_residual_is_the_coordinates_not_the_map(
    proton_gamma5: ReferenceParticle,
) -> None:
    r"""*Where* and *how fast* the ``(zeta, delta)`` check fails, measured.

    If the residual were a defect in the map it would sit anywhere and scale however
    it liked. Being a consequence of the coordinate choice, it is confined to the two
    ``(p_transverse, delta)`` entries — the exact map's transverse angles depend on
    ``delta`` in a way ``delta`` is not the conjugate momentum for — and it is
    **second order** in the amplitude. Measured on ``Drift(2.0)`` at ``gamma0 = 5``:
    ``7.7e-8`` at amplitude ``1e-3`` and ``7.7e-6`` at ``1e-2``.

    Asserting the location and the order is a far sharper statement than a tolerance,
    and it is what distinguishes this from a bug.
    """
    ref = proton_gamma5

    def residual(amp: float) -> np.ndarray:
        M = jacobian(lambda s: _exact_drift(s, ref), _state(amp), step=1e-7)
        return M.T @ J6 @ M - J6

    # Location: only (px, delta) and (py, delta) and their antisymmetric partners.
    R = residual(1.0e-2)
    allowed = np.zeros((DIM, DIM), dtype=bool)
    for i, j in ((PX, DELTA), (PY, DELTA)):
        allowed[i, j] = allowed[j, i] = True
    assert np.max(np.abs(R[allowed])) > 1.0e-6
    assert np.max(np.abs(R[~allowed])) < 1.0e-9

    # Order: a decade in amplitude is two decades in the residual.
    big, small = np.max(np.abs(residual(1.0e-2))), np.max(np.abs(residual(1.0e-3)))
    assert big / small == pytest.approx(100.0, rel=0.05)


def test_the_half_fix_is_caught_at_first_order(proton_gamma5: ReferenceParticle) -> None:
    r"""The gate's real job: reject an exact transverse map with ``zeta`` left linear.

    This is the plausible wrong implementation — add ``1/pz`` to ``x`` and ``y``, leave
    the longitudinal map alone — and it is *not* symplectic, because the transverse
    coordinates' dependence on momentum and the path length's dependence on angle are
    canonical partners: you cannot have one without the other.

    It fails at **first** order in the amplitude (``2.0e-4`` at amplitude ``1e-4``,
    against exactly ``0`` for the correct map), so the separation is not a matter of
    tolerance. Note that :func:`is_symplectic_map` rejects the correct map *and* this
    one, which is precisely why it cannot be used to choose between them.
    """
    ref = proton_gamma5
    for amp in (1.0e-4, 1.0e-3, 1.0e-2):
        st = _state(amp)
        assert not is_symplectic_map_canonical(lambda s: _half_fixed_drift(s, ref), st, ref)
        assert is_symplectic_map_canonical(lambda s: _exact_drift(s, ref), st, ref)

    # First order, not second: ten times the amplitude, ten times the residual.
    def residual(amp: float) -> float:
        can = to_canonical(_state(amp), ref)
        M = jacobian(
            lambda s: to_canonical(_half_fixed_drift(from_canonical(s, ref), ref), ref),
            can,
            step=1e-7,
        )
        return float(np.max(np.abs(M.T @ J6 @ M - J6)))

    assert residual(1.0e-3) / residual(1.0e-4) == pytest.approx(10.0, rel=0.05)


def test_a_bad_state_shape_is_rejected(proton_gamma5: ReferenceParticle) -> None:
    """A 4D orbit vector is a common slip; it must not be silently padded."""
    with pytest.raises(ValueError, match="length-6"):
        is_symplectic_map_canonical(lambda s: s, np.zeros(4), proton_gamma5)
