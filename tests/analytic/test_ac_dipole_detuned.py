"""The amplitude-dependent driven response (W4): Duffing's cubic, in the driven frame.

Swung hard, a particle sees its own tune move with its amplitude — an octupole's
first-order ``dQ/dJ`` and the exact drift's kinematic one — so the swing feeds back on the
distance to resonance that sets it. The steady state becomes a root of a cubic: one when
the detuning pushes the tune away from the drive, three when it pulls it toward it (a small
in-phase swing, stable; a large antiphase one, stable; one between them, unstable), and a
**fold kick** past which the small one is gone.

The truth needs no arbiter. With a rational drive tune ``p/q`` every steady state is a
period-``q`` orbit of the exact driven map, found here by Newton on the ``q``-turn map in
``(u, p_u)`` (the other plane at zero is invariant: the octupole's kick in it is
proportional to it). Its stability is the type of the ``q``-turn Jacobian. Against that:

- the driven-frame cubic of :func:`accsim.driven_states` misses at **second** order in
  the nonlinear correction (halving the kick divides the miss by 16), while the same
  cubic written with the natural optics misses at **first** order (x4) — W2's beta beat;
- the large states, whose detuning must cancel ``Q - nu``, miss at first order in it;
- the fold, the count of states on each side of it, their stability types, and the side
  of the resonance that has them all agree with the exact map.

The fixture is W1's thin FODO with the focusing split (``Qx = 1.344``, ``Qy = 1.041``): on
the equal-tune ring the driven ``x`` motion pumps ``y`` parametrically at ``2 nu``, which is
asserted as a named limitation and not modelled.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from _w4_driven_orbits import (
    HIGH_BETA,
    IDX,
    LAG,
    NU,
    REF,
    _acd,
    _dipole_amplitude,
    _exact_state,
    _periodic_orbit,
    _q_turns,
    _ring,
    _seed,
)
from accsim import (
    PX,
    PY,
    ACDipole,
    Dipole,
    Drift,
    Lattice,
    Particle,
    Quadrupole,
    ThinOctupole,
    ThinSextupole,
    Tracker,
    X,
    Y,
    driven_amplitude,
    driven_fold_kick,
    driven_states,
)
from accsim import twiss as tw_mod
from accsim.tune import tracked_tunes
from accsim.twiss import (
    ResonantLatticeError,
    amplitude_detuning,
    closed_twiss,
    match_periodic,
    propagate_twiss,
    tunes,
)


def _small(lat: Lattice) -> float:
    """The in-phase state nearest W1's amplitude."""
    ((plane, states),) = driven_states(lat).items()
    w1 = driven_amplitude(lat)[plane]
    return min((s.amplitude for s in states), key=lambda a: abs(a - w1))


# --- 1. the drift's detuning, derived and tracked -----------------------------------------
def _phase_average_detuning(potential, coords) -> tuple[sp.Expr, sp.Expr, tuple]:
    """``(dQ_x/dJ_x, dQ_x/dJ_y)`` of a perturbation ``H(x, px, y, py)``: ``(1/2pi)
    d^2 <H> / dJ dJ`` over both betatron phases (J2's machinery)."""
    jx, jy, phx, phy = sp.symbols("J_x J_y phi_x phi_y", positive=True)
    bx, by, ax, ay = sp.symbols("beta_x beta_y alpha_x alpha_y", positive=True)
    x = sp.sqrt(2 * jx * bx) * sp.cos(phx)
    px = -sp.sqrt(2 * jx / bx) * (sp.sin(phx) + ax * sp.cos(phx))
    y = sp.sqrt(2 * jy * by) * sp.cos(phy)
    py = -sp.sqrt(2 * jy / by) * (sp.sin(phy) + ay * sp.cos(phy))
    h = sp.expand(potential.subs(dict(zip(coords, (x, px, y, py), strict=True))))
    avg = sp.integrate(sp.integrate(h, (phx, 0, 2 * sp.pi)), (phy, 0, 2 * sp.pi)) / (4 * sp.pi**2)
    direct = sp.simplify(sp.diff(avg, jx, 2) / (2 * sp.pi))
    cross = sp.simplify(sp.diff(avg, jx, jy) / (2 * sp.pi))
    gx, gy = (1 + ax**2) / bx, (1 + ay**2) / by
    return direct, cross, (bx, by, gx, gy)


def test_the_drift_detuning_is_derived() -> None:
    """The exact drift ``-sqrt(1 - p^2)`` carries ``+ (px^2 + py^2)^2 / 8``; phase-averaged
    it detunes by ``3 L gamma_x^2 / (16 pi)`` (direct) and ``L gamma_x gamma_y / (8 pi)``
    (cross). The machinery is anchored on the octupole's ``k3l beta^2 / (16 pi)`` (J2)."""
    x, px, y, py, length, k3l = sp.symbols("x p_x y p_y L k_3l")
    coords = (x, px, y, py)
    p2 = px**2 + py**2
    # the expansion of the exact drift itself, not recalled
    series = sp.series(-sp.sqrt(1 - sp.Symbol("w")), sp.Symbol("w"), 0, 3).removeO()
    assert sp.expand(series - (-1 + sp.Symbol("w") / 2 + sp.Symbol("w") ** 2 / 8)) == 0
    direct, cross, (bx, by, gx, gy) = _phase_average_detuning(length * p2**2 / 8, coords)
    assert sp.simplify(direct - 3 * length * gx**2 / (16 * sp.pi)) == 0
    assert sp.simplify(cross - length * gx * gy / (8 * sp.pi)) == 0
    assert float(3 / (16 * sp.pi)) == pytest.approx(tw_mod._DRIFT_DETUNING, rel=1e-15)
    # anchor: the same machinery on the octupole gives J2's shipped coefficients
    oct_pot = k3l * (x**4 - 6 * x**2 * y**2 + y**4) / 24
    d_oct, c_oct, _ = _phase_average_detuning(oct_pot, coords)
    assert sp.simplify(d_oct - k3l * bx**2 / (16 * sp.pi)) == 0
    assert sp.simplify(c_oct + k3l * bx * by / (8 * sp.pi)) == 0


def test_the_drift_detuning_is_what_a_free_particle_does() -> None:
    """On the bare ring (no octupole) the tracked free tune moves by ``a J`` with ``a`` the
    drift sum: the shift falls x4 per halving of amplitude, the residual x16."""
    bare = _ring()
    tw = propagate_twiss(bare, closed_twiss(bare))
    a = sum(
        tw_mod._DRIFT_DETUNING * e.length * t.gamma_x**2
        for e, t in zip(bare.elements, tw, strict=False)
        if isinstance(e, Drift)
    )
    q0 = tracked_tunes(bare, 4096, x0=1e-7, y0=1e-9, nonlinear=True)[0]
    sig, res = [], []
    for x0 in (0.02, 0.01, 0.005):
        shift = tracked_tunes(bare, 4096, x0=x0, y0=1e-9, nonlinear=True)[0] - q0
        sig.append(shift)
        res.append(shift - a * tw[0].gamma_x * x0 * x0 / 2)  # px = 0: J = gamma x0^2 / 2
    for i in (0, 1):
        assert 3.9 < sig[i] / sig[i + 1] < 4.1
        assert 15.0 < res[i] / res[i + 1] < 17.0


# --- 2. the small state starts as W1's ------------------------------------------------------
@pytest.mark.parametrize("k3l", [-2000.0, 2000.0])
def test_the_small_state_is_w1s_at_zero_kick(k3l: float) -> None:
    """Its correction from W1's closed form is the detuning's: x4 per halving of the kick,
    with the sign of ``k3l`` (pulled toward the drive, it swings more)."""
    corr = []
    for kick in (4e-5, 2e-5, 1e-5):
        lat = _ring(_acd(kick), k3l)
        corr.append(_small(lat) / driven_amplitude(lat)["x"] - 1.0)
    assert all(math.copysign(1.0, c) == -math.copysign(1.0, k3l) for c in corr)
    assert 3.9 < corr[0] / corr[1] < 4.1
    assert 3.9 < corr[1] / corr[2] < 4.1


# --- 3. the order gate, with its two controls ---------------------------------------------
def _natural_cubic(lat: Lattice, k3l: float, oct_at: int | None, u_near: float) -> float:
    """CONTROL: ``Q + a J`` with the natural ``beta`` and ``a`` put into W1's closed form."""
    acd = next(e for e in lat.elements if isinstance(e, ACDipole))
    bare = _ring(None, k3l, oct_at)
    tw = propagate_twiss(bare, closed_twiss(bare))
    ioct = next(i for i, e in enumerate(bare.elements) if isinstance(e, ThinOctupole))
    a = amplitude_detuning(bare)[0, 0]
    a += sum(
        tw_mod._DRIFT_DETUNING * e.length * t.gamma_x**2
        for e, t in zip(bare.elements, tw, strict=False)
        if isinstance(e, Drift)
    )
    beta, mu = tw[IDX].beta_x, tw[-1].mu_x
    assert tw[ioct].beta_x > 0.0
    c = math.cos(2 * math.pi * acd.tune)

    def f(u: float) -> float:
        mu_j = mu + 2 * math.pi * a * u * u / (2 * beta)
        return u * 2 * (c - math.cos(mu_j)) - acd.amplitude * beta * math.sin(mu_j)

    lo, hi = 0.7 * u_near, 1.3 * u_near
    assert f(lo) * f(hi) < 0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if f(lo) * f(m) <= 0 else (m, hi)
    return 0.5 * (lo + hi)


def _wrong_beta_cubic(lat: Lattice, k3l: float, u_near: float) -> float:
    """CONTROL: the driven-frame cubic, but reading the octupole's beta at the dipole."""
    acd = next(e for e in lat.elements if isinstance(e, ACDipole))
    iacd = next(i for i, e in enumerate(lat.elements) if isinstance(e, ACDipole))
    base = [e.matrix(lat.ref) for e in lat.elements]
    bare = _ring()
    tb = propagate_twiss(bare, closed_twiss(bare))
    beta, mu, nu = tb[IDX].beta_x, tb[-1].mu_x, acd.tune

    def f(u: float) -> float:
        g = acd.amplitude / u
        maps = [m.copy() for m in base]
        maps[iacd][PX, X] += g
        M = np.eye(6)
        for m in maps:
            M = m @ M
        tws = propagate_twiss(lat, match_periodic(M), maps=maps)
        bd = tws[iacd].beta_x
        a = k3l * bd * bd / (16 * math.pi)  # the wrong beta
        a += sum(
            tw_mod._DRIFT_DETUNING * e.length * t.gamma_x**2
            for e, t in zip(lat.elements, tws, strict=False)
            if isinstance(e, Drift)
        )
        j = u * u / (2 * bd)
        return math.cos(2 * math.pi * (nu - a * j)) - math.cos(mu) - g * beta * math.sin(mu) / 2

    lo, hi = 0.7 * u_near, 1.3 * u_near
    assert f(lo) * f(hi) < 0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if f(lo) * f(m) <= 0 else (m, hi)
    return 0.5 * (lo + hi)


FIXTURES = [(-2000.0, None), (2000.0, None), (-500.0, HIGH_BETA)]


@pytest.mark.parametrize(("k3l", "oct_at"), FIXTURES)
def test_the_driven_frame_cubic_misses_at_second_order(k3l: float, oct_at) -> None:
    """``driven_states`` against the exact period-40 orbit: the relative miss falls x16 per
    halving of the kick (the absolute one x32: the amplitude halves too).
    The natural-optics cubic falls only x4 — its coefficient is wrong by W2's beta beat."""
    miss, nat, wrong = [], [], []
    for kick in (4e-5, 2e-5, 1e-5):
        lat = _ring(_acd(kick), k3l, oct_at)
        pred = _small(lat)
        u, quad, J = _exact_state(lat, pred)
        assert abs(quad) < 1e-12 * abs(u)  # a periodic orbit in phase with the drive
        assert abs(0.5 * np.trace(J)) < 1.0  # elliptic, as the small state must be
        miss.append((u - pred) / u)  # relative: the order the gate is stated in
        nat.append((u - _natural_cubic(lat, k3l, oct_at, pred)) / u)
        if oct_at is not None:
            wrong.append((u - _wrong_beta_cubic(lat, k3l, pred)) / u)
    assert 15.0 < miss[1] / miss[2] < 17.0
    assert 3.5 < nat[1] / nat[2] < 4.5
    if wrong:  # blind where the octupole sits at the dipole; caught here
        assert 3.5 < wrong[1] / wrong[2] < 4.5


# --- 4 and 5. the count, the fold, the stability type -------------------------------------
def test_three_states_below_the_fold_and_one_above() -> None:
    """At 0.98 ``F_c`` each predicted state seeds a distinct exact orbit within 1% of it,
    of the predicted stability type; at 1.02 ``F_c`` only the antiphase one is left and
    nothing small and in phase exists. The exact fold, where the exact small and middle
    orbits merge, is within 0.2% of the prediction (measured 1.0014 ``F_c``)."""
    k3l = -2000.0
    fc = driven_fold_kick(_ring(_acd(1e-5), k3l))["x"]
    assert fc is not None
    lat = _ring(_acd(0.98 * fc), k3l)
    states = driven_states(lat)["x"]
    assert [s.stable for s in states] == [True, True, False]
    assert states[0].amplitude < 0.0 < states[1].amplitude < states[2].amplitude
    found = []
    for st in states:
        u, _, J = _exact_state(lat, st.amplitude)
        assert u is not None
        assert abs(u - st.amplitude) < 0.01 * abs(st.amplitude)
        assert (abs(0.5 * np.trace(J)) < 1.0) == st.stable  # gate 5: the Floquet type
        found.append(u)
    assert min(abs(a - b) for i, a in enumerate(found) for b in found[i + 1 :]) > 1e-3
    small, middle = states[1].amplitude, states[2].amplitude

    above = _ring(_acd(1.02 * fc), k3l)
    (only,) = driven_states(above)["x"]
    assert only.amplitude < 0.0 and only.stable
    # Weak on its own (it passes if Newton merely stalls); the fold claim is carried by
    # the extrapolation below, which measures where the exact orbits actually merge.
    z, _, _ = _periodic_orbit(above, _seed(above, small, 1.02 * fc, NU), 40, it=20)
    if z is not None:  # whatever Newton found from the small state's seed, it is not small
        u, _ = _dipole_amplitude(above, z, 40, NU)
        assert not (0.0 < u < middle)

    # The exact fold: the small and middle orbits merge there, and near it the square of
    # their gap falls linearly with the kick (a saddle-node). Extrapolate it to zero.
    gaps = []
    for f in (0.996, 0.998):
        lat_f = _ring(_acd(f * fc), k3l)
        _, sm, md = driven_states(lat_f)["x"]
        u_sm, _, _ = _exact_state(lat_f, sm.amplitude)
        u_md, _, _ = _exact_state(lat_f, md.amplitude)
        assert 0.0 < u_sm < u_md
        gaps.append((u_md - u_sm) ** 2)
    f_exact = 0.998 + gaps[1] * (0.998 - 0.996) / (gaps[0] - gaps[1])
    assert abs(f_exact - 1.0) < 2e-3


# --- 6. the direction ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("k3l", "side", "three"),
    [(-2000.0, 1, True), (2000.0, -1, True), (2000.0, 1, False), (-2000.0, -1, False)],
)
def test_three_states_only_when_the_detuning_pulls_toward_the_drive(k3l, side, three) -> None:
    """``side = +1``: the drive below the tune. Pulling toward it (``k3l < 0`` below,
    ``> 0`` above) gives a fold; pushing away gives one state at every kick."""
    q = tunes(_ring())[0] % 1.0
    nu = q - side * 0.0191
    fold = driven_fold_kick(_ring(_acd(1e-5, nu), k3l))["x"]
    if three:
        assert fold is not None
        assert len(driven_states(_ring(_acd(0.5 * fold, nu), k3l))["x"]) == 3
        assert len(driven_states(_ring(_acd(1.5 * fold, nu), k3l))["x"]) == 1
    else:
        assert fold is None
        mirrored = driven_fold_kick(_ring(_acd(1e-5, nu), -k3l))["x"]
        for f in (0.1, 1.0, 10.0):
            (only,) = driven_states(_ring(_acd(f * mirrored, nu), k3l))["x"]
            assert only.stable
            assert math.copysign(1.0, only.amplitude) == side  # W1's phase, kept


# --- 7. the large states, first order in Q - nu ------------------------------------------
def test_the_large_states_miss_at_first_order_in_the_distance_to_resonance() -> None:
    """Their detuning must cancel ``Q - nu``, so their miss cannot shrink with the kick;
    it shrinks with ``Q - nu``. Halving it twice at ``kick = 0.5 F_c``: ratios in 2-3,
    falling (measured 2.85, 2.47)."""
    k3l = -2000.0

    def kf_for(delta: float) -> float:
        lo, hi = 0.2, 0.26
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if tunes(_ring(kf=m))[0] % 1.0 - NU < delta else (lo, m)
        return 0.5 * (lo + hi)

    miss = []
    for delta in (0.0191, 0.00955, 0.004775):
        kf = kf_for(delta)
        fc = driven_fold_kick(_ring(_acd(1e-5), k3l, kf=kf))["x"]
        lat = _ring(_acd(0.5 * fc), k3l, kf=kf)
        anti = driven_states(lat)["x"][0]
        assert anti.amplitude < 0.0 and anti.stable
        u, _, J = _exact_state(lat, anti.amplitude)
        assert abs(0.5 * np.trace(J)) < 1.0
        miss.append(abs(u - anti.amplitude) / abs(u))
    r1, r2 = miss[0] / miss[1], miss[1] / miss[2]
    assert 2.0 < r2 < r1 < 3.0


# --- 8. a ramp ----------------------------------------------------------------------------
def test_a_slow_ramp_lands_on_the_small_state_and_past_the_fold_leaves_it() -> None:
    """Ramped up over 1500 turns to 0.7 ``F_c`` the particle swings at the small state's
    amplitude; ramped to 1.3 ``F_c`` it is thrown past the middle state."""
    k3l = -2000.0
    fc = driven_fold_kick(_ring(_acd(1e-5), k3l))["x"]
    ramp = (0, 1500, 10**9, 10**9)
    n = 1500 + 800

    def tracked(kick: float) -> np.ndarray:
        lat = _ring(_acd(kick, ramp=ramp), k3l, first=True)  # row n = x at the dipole
        with np.errstate(invalid="ignore", over="ignore"):
            return Tracker(lat).track_turns(Particle(), n, nonlinear=True)[:n, X]

    below = tracked(0.7 * fc)
    turns = np.arange(n - 800, n)
    u = 2 / 800 * float(np.sum(below[-800:] * np.sin(2 * np.pi * (NU * turns + LAG))))
    expected = _small(_ring(_acd(0.7 * fc), k3l))
    assert u == pytest.approx(expected, rel=1e-3)

    middle = driven_states(_ring(_acd(0.98 * fc), k3l))["x"][2].amplitude
    above = tracked(1.3 * fc)
    assert not np.all(np.isfinite(above)) or np.nanmax(np.abs(above)) > middle


# --- 9. the other plane -------------------------------------------------------------------
def test_the_vertical_drive_is_the_same_cubic() -> None:
    """Gate 3's first fixture driven in ``y`` near ``Qy`` (``nu = 1/40``): x16. The kicks
    are the same fractions of the plane's own fold as gate 3's (0.1 to 0.025 ``F_c``): the
    vertical fold is 19x lower (2.1e-5), so gate 3's absolute kicks would sit on it."""
    nu = 1 / 40
    assert 0.0 < tunes(_ring())[1] % 1.0 - nu < 0.02  # below Qy, as in x
    fc = driven_fold_kick(_ring(_acd(1e-6, nu, plane="y"), -2000.0))["y"]
    miss = []
    for kick in (0.1 * fc, 0.05 * fc, 0.025 * fc):
        lat = _ring(_acd(kick, nu, plane="y"), -2000.0)
        ((_, states),) = driven_states(lat).items()
        w1 = driven_amplitude(lat)["y"]
        pred = min((s.amplitude for s in states), key=lambda a: abs(a - w1))
        u, _, _ = _exact_state(lat, pred, nu, plane="y")
        miss.append((u - pred) / u)
    assert 15.0 < miss[1] / miss[2] < 17.0


# --- 10. the named limitation: the undriven plane is pumped --------------------------------
def test_on_an_equal_tune_ring_the_undriven_plane_is_pumped() -> None:
    """``Qx = Qy``: the driven ``x`` orbit modulates the octupole's vertical gradient at
    ``2 nu`` with ``Qy`` near ``nu`` — a parametric drive. ``y = 0`` stays a solution, but
    the vertical Floquet block leaves the unit circle. ``driven_states`` does not see it."""
    nu = 0.275  # = 11/40, below Qx = Qy = 1.2894
    lat = _ring(_acd(3e-4, nu), 2000.0, kf=0.25, kd=-0.25)
    qx, qy = tunes(_ring(kf=0.25, kd=-0.25))
    assert qx == pytest.approx(qy, abs=1e-12)
    (state,) = driven_states(lat)["x"]
    assert state.stable  # its own plane: fine
    z, _, _ = _periodic_orbit(lat, _seed(lat, state.amplitude, 3e-4, nu), 40)
    h = 1e-9
    cols = np.zeros((6, 4))
    cols[X, :], cols[PX, :] = z[0], z[1]
    cols[Y, 0], cols[Y, 1], cols[PY, 2], cols[PY, 3] = h, -h, h, -h
    out = _q_turns(lat, cols, 40)
    Jy = np.column_stack(
        (
            (out[[Y, PY], 0] - out[[Y, PY], 1]) / (2 * h),
            (out[[Y, PY], 2] - out[[Y, PY], 3]) / (2 * h),
        )
    )
    assert abs(0.5 * np.trace(Jy)) > 1.0


# --- 11. refusals -------------------------------------------------------------------------
def test_a_state_past_the_validity_cut_is_refused_not_dropped() -> None:
    """Driven hard enough, a branch of the response runs into the validity cut before it
    reaches the kick: a state exists where first-order detuning cannot see it. Refused
    (measured: 100x the fold on the pushing ring, 20x on the pulling one) — never a
    shorter tuple that looks complete."""
    fc = driven_fold_kick(_ring(_acd(1e-5), -2000.0))["x"]
    assert len(driven_states(_ring(_acd(30 * fc), 2000.0))["x"]) == 1
    assert len(driven_states(_ring(_acd(10 * fc), -2000.0))["x"]) == 1
    for k3l, f in ((2000.0, 100.0), (-2000.0, 20.0)):
        with pytest.raises(ValueError, match="first-order detuning"):
            driven_states(_ring(_acd(f * fc), k3l))


@pytest.mark.parametrize(
    "intruder",
    [Dipole(1.0, 0.01), ThinSextupole(1.0), Quadrupole(0.5, 0.1), ThinOctupole(100.0, dx=1e-3)],
)
def test_out_of_scope_rings_are_refused(intruder) -> None:
    lat = _ring(_acd(1e-5), -2000.0)
    lat = Lattice([*lat.elements, intruder], REF)
    with pytest.raises(NotImplementedError):
        driven_states(lat)
    with pytest.raises(NotImplementedError):
        driven_fold_kick(lat)


def test_two_planes_zero_kick_and_a_resonant_drive_are_refused() -> None:
    base = _ring(None, -2000.0).elements
    two = Lattice([_acd(1e-5), _acd(1e-5, 0.02, plane="y"), *base], REF)
    with pytest.raises(NotImplementedError):
        driven_states(two)
    with pytest.raises(ValueError):
        driven_states(_ring(_acd(0.0), -2000.0))
    with pytest.raises(ResonantLatticeError):
        driven_states(_ring(_acd(1e-5, 0.5), -2000.0))
