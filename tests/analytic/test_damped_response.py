"""The damped driven response (W6 part b): W4's steady states with a damper in the ring.

A :class:`~accsim.elements.damper.Damper` takes a fixed fraction of the swing every turn,
so a steady state needs the drive to put it back. In the driven frame (W2, W4) the kick
is ``g u + k p`` at the dipole, and the ring with that substituted is a free oscillation
at the drive tune: its determinant is 1, which fixes ``k = 1 / prod(1 - rate) - 1``
exactly, and its trace fixes the tune ``q(g)``. The detuning enters by first-order
eigenvalue perturbation, with eigenvectors in place of Twiss (the ring has
non-symplectic pieces), plus one term W4 never needed: ``p / u`` at the dipole moves
with the eigenvector, and the kick's size depends on it through ``k``.

Damping creates a second fold. Undamped, W4's large state runs on for ever; damped, it
ends where the drive can no longer pay for its damping, and a sweep back drops off it.
:func:`accsim.driven_jump_tunes` gives both folds as drive tunes.

The truth is the exact driven map: period-``q`` orbits at rational drive tunes, found by
Newton on the ``q``-turn map with the damping applied (``tests/_w4_driven_orbits.py``).

Fixture: W4's split-tune FODO, ``k3l = -2000`` at the dipole, a damper right after it
(``Gamma = 0.003``, a damping time of 333 turns) unless stated.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from _w4_driven_orbits import IDX, LAG, REF, _acd, _dipole_amplitude, _periodic_orbit, _ring
from accsim import (
    PX,
    Damper,
    Lattice,
    driven_fold_kick,
    driven_fold_tune,
    driven_jump_tunes,
    driven_states,
)
from accsim.twiss import _DampedFolds, _DampedResponseCurve, _response_curve, _states_on

K3L = -2000.0
GAMMA = 0.003
NU13 = Fraction(13, 40)
NU17 = Fraction(17, 50)  # both exact folds are pinned here
# The damper right after the dipole, and before elements 7 and 15 (alpha differs at each).
PLACES = {"after": IDX + 1, "e7": 7, "e15": 15}


def _rate(gamma: float) -> float:
    return 1.0 - math.exp(-2.0 * gamma)


def _lat(
    kick: float,
    nu: float,
    *,
    k3l: float = K3L,
    where: str = "after",
    gamma: float = GAMMA,
    plane: str = "x",
) -> Lattice:
    els = list(_ring(_acd(kick, float(nu)), k3l).elements)  # [..., dipole (IDX), octupole, ...]
    els.insert(PLACES[where], Damper(_rate(gamma), plane=plane))
    return Lattice(els, REF)


def _block(e) -> np.ndarray:
    return (e.damping_matrix() if isinstance(e, Damper) else e.matrix(REF))[:2, :2]


def _exact(lat: Lattice, curve, g: float, nu: Fraction, shift: complex = 0j):
    """The exact period-q orbit seeded at the curve's state ``g`` (its swing moved by
    ``shift``): ``(complex swing at the dipole, q-turn Jacobian)``, in-phase + i
    quadrature, or ``(None, None)``."""
    d = curve._damped(np.array([g]))
    xhat = curve.sign * curve.kick * complex(d["rho"][0]) + shift
    ratio = complex(d["ratio"][0])  # p / u at the dipole entrance, on the drive's phasor
    e2 = np.exp(2j * math.pi * LAG)
    zd = np.array([(xhat * e2).imag, (xhat * ratio * e2).imag])
    A = np.eye(2)
    for e in lat.elements[:IDX]:
        A = _block(e) @ A
    z, _, J = _periodic_orbit(lat, np.linalg.solve(A, zd), nu.denominator)
    if z is None:
        return None, None
    return complex(*_dipole_amplitude(lat, z, nu.denominator, float(nu))), J


def _swing(state) -> complex:
    return complex(state.amplitude, state.quadrature)


# --- 5. the driven frame with damping -----------------------------------------------------
def test_k_is_rate_over_one_minus_rate_and_the_steady_state_is_the_eigenvector() -> None:
    """The exact linear damped steady state (the periodic solution of the affine 40-turn
    map, the damper inside it), demodulated at every element boundary: its kick at the
    dipole is ``g u + k p`` with ``g`` real and equal to the curve's W2 gradient, and
    ``p / u`` everywhere is the substituted ring's eigenvector carried round."""
    kick, nu, n = 1e-5, NU13, NU13.denominator
    lat = _lat(kick, nu, k3l=0.0)
    curve = _response_curve(lat, "test")
    assert isinstance(curve, _DampedResponseCurve)
    rate = _rate(GAMMA)
    assert curve.k == pytest.approx(rate / (1.0 - rate), rel=1e-12, abs=0.0)

    maps = [_block(e) for e in lat.elements]
    acd = lat.elements[IDX]

    def period(z0: np.ndarray, record: bool = False):
        z, us, ps, ks = z0.copy(), [], [], []
        for turn in range(n):
            row_u, row_p = [], []
            for j, m in enumerate(maps):
                row_u.append(z[0])
                row_p.append(z[1])
                z = m @ z
                if j == IDX:
                    k_n = acd.drive_kick(turn, REF)[PX]
                    ks.append(k_n)
                    z = z + np.array([0.0, k_n])
            us.append(row_u)
            ps.append(row_p)
        return (z, np.array(us).T, np.array(ps).T, np.array(ks)) if record else z

    b = period(np.zeros(2))
    A = np.column_stack([period(np.eye(2)[i]) - b for i in range(2)])
    z0 = np.linalg.solve(np.eye(2) - A, b)
    z_end, U, P, K = period(z0, record=True)
    np.testing.assert_allclose(z_end, z0, rtol=1e-12, atol=0.0)

    e = np.exp(-2j * math.pi * (float(nu) * np.arange(n) + LAG))
    u_hat = 2j / n * U @ e  # u_n = Im(u_hat e^{i phi_n}) at each element entrance
    p_hat = 2j / n * P @ e
    k_hat = 2j / n * K @ e
    g = (k_hat - curve.k * p_hat[IDX]) / u_hat[IDX]
    assert abs(g.imag) < 1e-12 * abs(g)
    assert g.real == pytest.approx(curve.g_w2, rel=1e-12)

    S = np.array([[1.0, 0.0], [curve.g_w2, 1.0 + curve.k]])
    w, V = np.linalg.eig(curve.R @ S)
    v = V[:, int(np.argmin(np.abs(w - np.exp(2j * math.pi * float(nu)))))]
    walk, ratio = S @ v, {IDX: v[1] / v[0]}
    for j in [*range(IDX + 1, len(maps)), *range(IDX)]:
        ratio[j] = walk[1] / walk[0]
        walk = maps[j] @ walk
    want = p_hat / u_hat
    got = np.array([ratio[j] for j in range(len(maps))])
    assert np.max(np.abs(got - want)) < 1e-12 * np.max(np.abs(want))


# --- 6. undamped is W4 --------------------------------------------------------------------
# Returned by the pre-W6 code (commit 3bc145a) on these rings; the refactor must not move
# a bit.
_W4_FROZEN = {
    (1e-5, 0.325): (
        [
            (-0.015984370327830306, True),
            (0.00015231094161633895, True),
            (0.015839065595919164, False),
        ],
        0.00040755290022517787,
        0.342466997292209,
    ),
    (2e-5, 0.33): (
        [
            (-0.013872629198612025, True),
            (0.00041651285961946514, True),
            (0.013469822967032726, False),
        ],
        0.0002555038475194897,
        0.341493294892095,
    ),
}


@pytest.mark.parametrize("kick,nu", list(_W4_FROZEN))
def test_without_a_damper_the_three_functions_are_w4_bit_for_bit(kick: float, nu: float) -> None:
    states, fold_kick, fold_tune = _W4_FROZEN[(kick, nu)]
    lat = _ring(_acd(kick, nu), K3L)
    got = driven_states(lat)["x"]
    assert [(s.amplitude, s.stable) for s in got] == states
    assert all(s.quadrature == 0.0 for s in got)
    assert driven_fold_kick(lat)["x"] == fold_kick
    assert driven_fold_tune(lat)["x"] == fold_tune
    # A damper on the other plane cannot reach this one: accepted, and changes nothing.
    els = list(lat.elements)
    els.insert(IDX + 1, Damper(_rate(GAMMA), plane="y"))
    other = Lattice(els, REF)
    assert [(s.amplitude, s.stable) for s in driven_states(other)["x"]] == states
    assert driven_fold_kick(other)["x"] == fold_kick


@pytest.mark.parametrize("kick,nu", list(_W4_FROZEN))
def test_the_damped_machinery_at_rate_zero_is_w4(kick: float, nu: float) -> None:
    states, fold_kick, fold_tune = _W4_FROZEN[(kick, nu)]
    lat = _lat(kick, nu, gamma=0.0)
    assert isinstance(_response_curve(lat, "test"), _DampedResponseCurve)
    got = driven_states(lat)["x"]
    assert [s.stable for s in got] == [st for _, st in states]
    for s, (amp, _) in zip(got, states, strict=True):
        assert s.amplitude == pytest.approx(amp, rel=1e-12)
        assert abs(s.quadrature) < 1e-12 * abs(amp)
    assert driven_fold_kick(lat)["x"] == pytest.approx(fold_kick, rel=1e-12)
    assert driven_fold_tune(lat)["x"] == pytest.approx(fold_tune, rel=1e-12)
    with pytest.raises(ValueError, match="no damping"):
        driven_jump_tunes(lat)  # rate 0 damps nothing: the large state never ends


def test_driven_jump_tunes_needs_damping() -> None:
    with pytest.raises(ValueError, match="no damping"):
        driven_jump_tunes(_ring(_acd(2e-5, 0.34), K3L))


# --- 7. the order gate --------------------------------------------------------------------
class _NoEigenvectorTerm(_DampedResponseCurve):
    eigenvector_term = False


def _small_state_misses(where: str, gamma: float, kicks, control: bool):
    """Per kick: ``(miss / |exact|, miss / nonlinear correction)`` of the small state."""
    out = []
    for kick in kicks:
        lat = _lat(kick, NU13, where=where, gamma=gamma)
        curve = (_NoEigenvectorTerm if control else _DampedResponseCurve)(lat, "test")
        state = _small(curve)[1]
        truth = _DampedResponseCurve(lat, "test")  # the seed is the full model's either way
        ex, _ = _exact(lat, truth, _small(truth)[0], NU13)
        assert ex is not None
        linear = kick * complex(curve._damped(np.array([curve.g_w2]))["rho"][0])
        miss = abs(_swing(state) - ex)
        out.append((miss / abs(ex), miss / abs(ex - linear)))
    return out


def _small(curve):
    """``(g, state)`` of the smallest swing on ``curve``."""
    return min(_states_on(curve), key=lambda gs: abs(_swing(gs[1])))


KICKS7 = (3.2e-5, 1.6e-5, 8e-6)


@pytest.mark.parametrize("where", list(PLACES))
def test_the_small_state_miss_falls_x16_per_halving_of_the_kick(where: str) -> None:
    """Second order in the nonlinear correction, so x16 per halving of the kick (W4's
    driven-frame order). Measured 16.03-16.10 at the three damper positions."""
    misses = [m for m, _ in _small_state_misses(where, GAMMA, KICKS7, control=False)]
    ratios = [a / b for a, b in zip(misses, misses[1:], strict=False)]
    assert all(15.0 < r < 17.0 for r in ratios), ratios


@pytest.mark.parametrize("where", list(PLACES))
def test_control_without_the_eigenvector_term_the_miss_falls_only_x4(where: str) -> None:
    """Without ``p / u``'s first-order move the miss is a fixed fraction of the nonlinear
    correction: x4 per halving (first order), the fraction constant across kicks."""
    rows = _small_state_misses(where, GAMMA, KICKS7, control=True)
    misses = [m for m, _ in rows]
    ratios = [a / b for a, b in zip(misses, misses[1:], strict=False)]
    assert all(3.9 < r < 4.1 for r in ratios), ratios
    fractions = [f for _, f in rows]
    assert max(fractions) / min(fractions) < 1.05, fractions


def test_control_the_missing_term_is_proportional_to_gamma() -> None:
    """The fraction the control misses by halves with the damping (measured 2.00)."""
    kicks = (1.6e-5,)
    ((_, f_full),) = _small_state_misses("after", GAMMA, kicks, control=True)
    ((_, f_half),) = _small_state_misses("after", GAMMA / 2.0, kicks, control=True)
    assert 1.95 < f_full / f_half < 2.05, f_full / f_half


# --- 9 (and 8). the folds -----------------------------------------------------------------
def _exact_fold(predicted_kick: float, g_fold: float, fracs) -> float:
    """W4's method: the squared gap between the two exact orbits that merge at the fold,
    at two kicks just inside it, extrapolated linearly to zero gap.

    Near a fold the predicted pair sits off the exact one by more than the pair's own
    size, and Newton seeded at the predicted saddle can slide onto the sink (it did, at
    1.004 times the large end): a zero gap, and a fold 0.4% off. So the sink is found
    first, and the saddle seeded at its prediction moved by the sink's own miss; the two
    must then be different orbits, one a sink and one a saddle."""
    gaps = []
    for f in fracs:
        kick = f * predicted_kick
        lat = _lat(kick, NU17)
        curve = _DampedResponseCurve(lat, "test")
        pair = sorted(_states_on(curve), key=lambda gs: abs(gs[0] - g_fold))[:2]
        (g_sink, sink), (g_saddle, _) = sorted(pair, key=lambda gs: not gs[1].stable)
        ex_sink, J_sink = _exact(lat, curve, g_sink, NU17)
        ex_saddle, J_saddle = _exact(lat, curve, g_saddle, NU17, shift=ex_sink - _swing(sink))
        assert np.abs(np.linalg.eigvals(J_sink)).max() < 1.0
        assert np.abs(np.linalg.eigvals(J_saddle)).max() > 1.0
        gaps.append((kick, abs(ex_sink - ex_saddle) ** 2))
    (k1, s1), (k2, s2) = gaps
    return k1 + s1 * (k1 - k2) / (s2 - s1)


@pytest.fixture(scope="module")
def folds() -> dict[str, float]:
    """Predicted and exact fold kicks at 17/50."""
    curve = _DampedResponseCurve(_lat(1e-5, NU17), "test")
    g_small, g_large = curve.fold_pair()
    small, large = (float(curve.kick_of(g)[0]) for g in (g_small, g_large))
    return {
        "small": small,
        "large": large,
        "small_exact": _exact_fold(small, g_small, (0.996, 0.998)),
        "large_exact": _exact_fold(large, g_large, (1.002, 1.004)),
    }


def test_the_folds_match_the_exact_ones(folds: dict[str, float]) -> None:
    """Measured: small fold 5.8e-4 low, large end 1.6e-3 low."""
    assert driven_fold_kick(_lat(1e-5, NU17))["x"] == folds["small"]
    assert abs(folds["small"] / folds["small_exact"] - 1.0) < 2e-3
    assert abs(folds["large"] / folds["large_exact"] - 1.0) < 3e-3


def test_driven_jump_tunes_finds_the_exact_folds_at_17_over_50(folds: dict[str, float]) -> None:
    """At the exact fold kicks the corresponding jump tune is 17/50 (measured 1.5e-6 and
    1.2e-5 away); the small one is driven_fold_tune's."""
    up = driven_jump_tunes(_lat(folds["small_exact"], NU17))["x"]
    down = driven_jump_tunes(_lat(folds["large_exact"], NU17))["x"]
    assert abs(up[0] - float(NU17)) < 5e-5
    assert abs(down[1] - float(NU17)) < 5e-5
    lat = _lat(folds["small_exact"], NU17)
    assert driven_fold_tune(lat)["x"] == up[0]
    # The loop: the sweep up jumps nearer the tune than the sweep down drops.
    tune = _DampedFolds(lat, "test").q_d % 1.0
    assert up[1] < up[0] < tune


@pytest.mark.parametrize("which", ["small", "large"])
def test_the_stable_flags_match_the_exact_multipliers(folds: dict[str, float], which: str) -> None:
    """2% inside each fold, all three states: a sink has both 50-turn multipliers inside
    the unit circle, the middle state one outside."""
    kick = folds[f"{which}_exact"] * (0.98 if which == "small" else 1.02)
    lat = _lat(kick, NU17)
    curve = _DampedResponseCurve(lat, "test")
    states = _states_on(curve)
    assert len(states) == 3
    assert sorted(s.stable for _, s in states) == [False, True, True]
    for g, state in states:
        _, J = _exact(lat, curve, g, NU17)
        mult = np.abs(np.linalg.eigvals(J))
        if state.stable:
            assert mult.max() < 1.0, mult
        else:
            assert mult.max() > 1.0 > mult.min(), mult


# --- 10. the onset of bistability ---------------------------------------------------------
def test_sqrt3_from_the_averaged_cubic() -> None:
    """``n ((D + A n)^2 + G^2) = eps^2`` (``n = |c|^2``, ``D = 2 pi (Q_d - nu)``, ``A = 2 pi
    a``, ``G = Gamma``) first has three roots where its first and second ``n``-derivatives
    vanish together: ``|D| = sqrt(3) G``."""
    n, D, A = sp.symbols("n D A", real=True)
    G = sp.symbols("G", positive=True)
    f = n * ((D + A * n) ** 2 + G**2)
    sols = sp.solve([sp.diff(f, n), sp.diff(f, n, 2)], [D, n], dict=True)
    assert {sp.simplify(abs(s[D]) / G) for s in sols} == {sp.sqrt(3)}


def test_the_onset_approaches_sqrt3_gamma_over_2pi() -> None:
    """The model's onset (where the curve first has both folds) lies beyond the averaged
    cubic's by an excess first order in Gamma: under 0.3%, halving with it (measured
    2.05e-3, 1.03e-3, 5.2e-4)."""
    excess = []
    for gamma in (0.004, 0.002, 0.001):
        dist = _DampedFolds(_lat(1e-5, NU17, gamma=gamma), "test").onset()
        excess.append(dist / (math.sqrt(3.0) * gamma / (2.0 * math.pi)) - 1.0)
    assert all(0.0 < e < 3e-3 for e in excess), excess
    ratios = [a / b for a, b in zip(excess, excess[1:], strict=False)]
    assert all(1.9 < r < 2.1 for r in ratios), ratios


def test_below_the_onset_there_is_no_jump() -> None:
    f = _DampedFolds(_lat(1e-5, NU17), "test")
    cusp = f.kicks(f.onset())[0]
    assert driven_jump_tunes(_lat(0.5 * cusp, NU17))["x"] is None
    assert driven_fold_tune(_lat(0.5 * cusp, NU17))["x"] is None


# --- the quadrature's sign, with the drive written in the other half ----------------------
def test_the_quadrature_sign_holds_with_the_drive_at_one_minus_nu() -> None:
    """A drive at 33/50 is the same sequence of kicks as one at 17/50 run backwards in
    phase, and the model's eigenvector phasor is then the conjugate of the drive's. The
    in-phase part cannot see that; the quadrature flips sign if it is mishandled."""
    nu = Fraction(33, 50)
    kick = 1e-5  # below the large state's end: one state
    lat = _lat(kick, nu)
    curve = _DampedResponseCurve(lat, "test")
    assert curve.folded
    ((g, state),) = _states_on(curve)
    ex, _ = _exact(lat, curve, g, nu)
    assert abs(state.quadrature) > 0.05 * abs(state.amplitude)  # a real quadrature
    assert abs(_swing(state) - ex) < 1e-2 * abs(ex)
    assert np.sign(state.quadrature) == np.sign(ex.imag)
