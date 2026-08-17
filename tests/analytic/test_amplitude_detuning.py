r"""J2 (part 2) — amplitude-dependent detuning, the thing an octupole is *for*.

Every tune in this package so far belongs to the machine. This one belongs to the
**particle**: with an octupole in the ring, how fast a particle goes round in phase
depends on how big its oscillation is. The closed form gated here is

    dQ_x/dJ_x = + k3l beta_x^2 / (16 pi),
    dQ_y/dJ_y = + k3l beta_y^2 / (16 pi),
    dQ_x/dJ_y = dQ_y/dJ_x = - k3l beta_x beta_y / (8 pi),

computed by :func:`accsim.twiss.amplitude_detuning`. It is exact at **first** order
in ``k3l`` and in the action.

**Why this is the octupole's gate and not the sextupole's.** A sextupole detunes
too, but only through *second*-order perturbation theory: the effect is quadratic in
``k2`` and no closed form for it is claimed anywhere here. The octupole's term is
linear in ``k3l`` and falls straight out of a single phase average, which is what
makes a real analytic gate possible — and it is why the detuning ring below carries
**no sextupoles at all**. Their contribution would not vanish as ``k3l -> 0`` and
would contaminate the fit at a level this formula never predicts;
:func:`test_sextupole_detuning_is_real_and_deliberately_not_claimed` measures that
background instead of pretending it is absent.

**What makes it non-circular.** The averaging machinery ``dQ_u = (1/2pi)
d<V>/dJ_u`` is anchored first on the **quadrupole**, where it must reproduce the
independently known ``beta k1l/(4 pi)`` — checked both symbolically and against
accsim's own matrix tunes. Only then is the same machinery pointed at the octupole
potential. The potential itself descends from the field expansion anchored in
``test_octupole_kick.py``, so the ``1/6`` is never assumed here: a deliberately
mis-scaled octupole is caught by the tracked measurement as a clean factor of 6.

**How the tracked comparison is made honest.** Tracking sees all orders in ``k3l``
and in the action, while the formula is first order in both, so a single tolerance
at a single amplitude would swallow exactly the coefficient error the gate exists to
catch. The gate is therefore an **order** gate: over four halvings of the amplitude
the measured detuning falls by 4 (it is linear in the action, which falls by 4)
while the residual against the closed form falls by 16 (it is quadratic). Measured
2026-08-17 at ``N_TURNS = 1024``: signal ratios 4.095, 4.021, 4.005; residual ratios
17.83, 16.39, 16.09 — identical at 2048 turns, so they are physics rather than
sampling noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Drift,
    Lattice,
    Octupole,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    amplitude_detuning,
    tracked_tunes,
    tunes,
)
from accsim.twiss import closed_twiss, propagate_twiss

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0

# The working point. Chosen (by scan) to sit far from every resonance an octupole
# itself drives -- 4Qx, 4Qy, 2Qx +- 2Qy -- and far from the tunes NAFF reads badly
# (0, 1/2, 1). Nearest such line is 0.137 away. Sitting near one would read as a
# coefficient error rather than as the resonance it is.
KF, KD = 1.25, -1.50
K3L = 5.0e4  # m^-3, integrated
N_TURNS = 1024


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _ring(ref: ReferenceParticle, mid: list | None = None, n_cells: int = 3) -> Lattice:
    """A sextupole-free FODO ring, palindromic about ``s = 0`` and about the midpoint.

    The palindrome is load-bearing twice over: it puts ``alpha = 0`` at the launch
    point (so the action of a particle released with ``px = py = 0`` is exactly
    ``x0^2/(2 beta)``, with no hidden ``1 + alpha^2``), and it centres ``mid`` in the
    cell. Both are asserted rather than assumed, in
    :func:`test_launch_point_has_zero_alpha`.
    """
    half = [Drift(0.5), Quadrupole(0.3, KF), Drift(0.5), Quadrupole(0.3, KD), Drift(0.5)]
    return Lattice([*half, *(mid or []), *half[::-1]] * n_cells, ref)


def _octupole_ring(ref: ReferenceParticle, k3l: float = K3L) -> Lattice:
    return _ring(ref, [ThinOctupole(k3l)])


def _actions(tw, x0: float, y0: float) -> tuple[float, float]:
    """Courant-Snyder actions of a particle released at ``(x0, y0)`` with ``px = py = 0``.

    ``J = (gamma u^2 + 2 alpha u u' + beta u'^2)/2`` with ``u' = 0`` leaves
    ``J = (1 + alpha^2) u^2 / (2 beta)``. The ``1 + alpha^2`` is written out rather
    than dropped: it is exactly 1 here only because the ring is a palindrome.
    """
    jx = (1.0 + tw.alpha_x**2) * x0**2 / (2.0 * tw.beta_x)
    jy = (1.0 + tw.alpha_y**2) * y0**2 / (2.0 * tw.beta_y)
    return jx, jy


def _measured_detuning(
    lat_on: Lattice, lat_off: Lattice, x0: float, y0: float, n_turns: int = N_TURNS
) -> tuple[float, float]:
    """Tune shift caused by the octupole, at the launch amplitude ``(x0, y0)``.

    Taken as a **difference** against the same ring with the octupole removed, tracked
    at the *same* amplitude — not against the matrix tunes. NAFF has its own
    ``O(1/n_turns)`` bias, and differencing two measurements made the same way cancels
    it; comparing against an exact linear tune would leave that bias in the answer at
    the level the detuning itself lives.
    """
    qx1, qy1 = tracked_tunes(lat_on, n_turns, x0=x0, y0=y0, nonlinear=True)
    qx0, qy0 = tracked_tunes(lat_off, n_turns, x0=x0, y0=y0, nonlinear=True)
    return qx1 - qx0, qy1 - qy0


# --------------------------------------------------------------------------
# 1. The averaging machinery, anchored on the quadrupole
# --------------------------------------------------------------------------


def _averaged_tune_shift(V: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    r"""``(dQ_x, dQ_y)`` from a thin perturbing potential ``V(x, y)``, at fixed action.

    First-order perturbation theory in action-angle variables:
    ``x = sqrt(2 J_x beta_x) cos(phi_x)``, ``y = sqrt(2 J_y beta_y) cos(phi_y)``, the
    potential is averaged over both betatron phases, and the tune shift is
    ``dQ_u = (1/2 pi) d<V>/dJ_u``. Written once, evaluated on two different ``V``.
    """
    jx, jy, bx, by, phx, phy = sp.symbols("J_x J_y beta_x beta_y phi_x phi_y", positive=True)
    x = sp.sqrt(2 * jx * bx) * sp.cos(phx)
    y = sp.sqrt(2 * jy * by) * sp.cos(phy)
    Vs = V.subs({sp.Symbol("x", real=True): x, sp.Symbol("y", real=True): y})
    mean = sp.integrate(
        sp.integrate(sp.expand_trig(sp.expand(Vs)), (phx, 0, 2 * sp.pi)), (phy, 0, 2 * sp.pi)
    )
    mean = sp.simplify(mean / (2 * sp.pi) ** 2)
    return sp.simplify(sp.diff(mean, jx) / (2 * sp.pi)), sp.simplify(
        sp.diff(mean, jy) / (2 * sp.pi)
    )


def test_averaging_machinery_reproduces_the_quadrupole_tune_shift(ref: ReferenceParticle) -> None:
    """The anchor: on ``V = k1l x^2/2`` the machinery must give ``dQ_x = beta_x k1l/(4 pi)``.

    That result is not taken on trust either — it is confirmed against accsim's own
    matrix tunes by adding a weak thin quadrupole to a real ring. So before the
    averaging is pointed at the octupole it has already had to reproduce a number the
    package computes by a completely different (linear-algebra) route.
    """
    x, k1l = sp.symbols("x k1l", real=True)
    bx = sp.Symbol("beta_x", positive=True)
    dqx, dqy = _averaged_tune_shift(k1l * x**2 / 2)
    assert sp.simplify(dqx - bx * k1l / (4 * sp.pi)) == 0
    assert sp.simplify(dqy) == 0  # a normal quad in x does not move Q_y through this term

    k1l_val = 1e-5  # weak: first order must dominate
    lat_perturbed = _ring(ref, [ThinQuadrupole(k1l_val)])
    tab = propagate_twiss(lat_perturbed, closed_twiss(lat_perturbed))
    sum_bx = sum(
        tab[i].beta_x for i, e in enumerate(lat_perturbed.elements) if isinstance(e, ThinQuadrupole)
    )
    sum_by = sum(
        tab[i].beta_y for i, e in enumerate(lat_perturbed.elements) if isinstance(e, ThinQuadrupole)
    )
    qx0, qy0 = tunes(_ring(ref))
    qx1, qy1 = tunes(lat_perturbed)
    assert qx1 - qx0 == pytest.approx(sum_bx * k1l_val / (4 * math.pi), rel=2e-4)
    assert qy1 - qy0 == pytest.approx(-sum_by * k1l_val / (4 * math.pi), rel=2e-4)


def test_octupole_anharmonicity_follows_from_the_same_averaging(ref: ReferenceParticle) -> None:
    """The same machinery on the octupole potential gives what ``amplitude_detuning`` computes.

    ``V = k3l (x^4 - 6 x^2 y^2 + y^4)/24`` is the potential gated in
    ``test_octupole_kick.py`` as minus the integral of the field-derived kick — so the
    ``1/6`` in the kick and the ``1/24`` here are the same statement, and neither is
    invented at this point.
    """
    x, y, k3l = sp.symbols("x y k3l", real=True)
    jx, jy = sp.symbols("J_x J_y", positive=True)
    bx, by = sp.symbols("beta_x beta_y", positive=True)

    dqx, dqy = _averaged_tune_shift(k3l * (x**4 - 6 * x**2 * y**2 + y**4) / 24)
    assert sp.simplify(dqx - (k3l * (bx**2 * jx - 2 * bx * by * jy) / (16 * sp.pi))) == 0
    assert sp.simplify(dqy - (k3l * (by**2 * jy - 2 * bx * by * jx) / (16 * sp.pi))) == 0

    # ...and that is exactly the matrix the package returns, at the real beta values.
    lat = _octupole_ring(ref)
    tab = propagate_twiss(lat, closed_twiss(lat))
    idx = [i for i, e in enumerate(lat.elements) if isinstance(e, ThinOctupole)]
    assert len(idx) == 3  # three cells, one octupole each
    A = amplitude_detuning(lat)
    exp = np.zeros((2, 2))
    for i in idx:
        b_x, b_y = tab[i].beta_x, tab[i].beta_y
        exp += K3L / (16 * math.pi) * np.array([[b_x**2, -2 * b_x * b_y], [-2 * b_x * b_y, b_y**2]])
    assert np.allclose(A, exp, rtol=1e-12, atol=0.0)


def test_matrix_is_symmetric_and_the_cross_term_ratio_is_minus_two(
    ref: ReferenceParticle,
) -> None:
    """Two properties that fall out of the derivation rather than being imposed.

    Symmetry (``dQ_x/dJ_y == dQ_y/dJ_x``) is the statement that both come from one
    averaged Hamiltonian. And the *ratio* of the cross term to the diagonal is
    ``-2 beta_y/beta_x``, carrying no ``k3l`` at all — hence exactly ``-2`` when
    ``beta_x = beta_y``.

    The ratio is the right form, not ``-2 sqrt(dQx/dJx . dQy/dJy)``: that square root
    is negative for **either** sign of ``k3l`` (the product of the two diagonals is
    positive both ways) while the true cross term flips with ``k3l``. A defocusing
    octupole is entirely ordinary — Landau octupoles are run at both polarities — so
    the negative strength is checked here rather than assumed away, and the squared
    identity ``A01^2 == 4 A00 A11`` is the sign-free version of the same statement.
    """
    for k3l in (K3L, -K3L):
        lat = _octupole_ring(ref, k3l)
        tab = propagate_twiss(lat, closed_twiss(lat))
        i = next(i for i, e in enumerate(lat.elements) if isinstance(e, ThinOctupole))
        A = amplitude_detuning(lat)

        assert A[0, 1] == A[1, 0]  # bit for bit
        assert A[0, 1] / A[0, 0] == pytest.approx(-2.0 * tab[i].beta_y / tab[i].beta_x, rel=1e-12)
        assert A[1, 0] / A[1, 1] == pytest.approx(-2.0 * tab[i].beta_x / tab[i].beta_y, rel=1e-12)
        assert A[0, 1] ** 2 == pytest.approx(4.0 * A[0, 0] * A[1, 1], rel=1e-12)
        # The diagonal takes the sign of k3l and the cross term takes the opposite one.
        assert math.copysign(1.0, A[0, 0]) == math.copysign(1.0, k3l)
        assert math.copysign(1.0, A[1, 1]) == math.copysign(1.0, k3l)
        assert math.copysign(1.0, A[0, 1]) == math.copysign(1.0, -k3l)


def test_detuning_is_quadratic_in_beta(ref: ReferenceParticle) -> None:
    """``dQ_x/dJ_x`` scales as ``beta_x^2`` — checked by moving the octupole, not by algebra.

    Placed at the cell centre (high ``beta_y``) and at the ring start (high
    ``beta_x``), the diagonal terms swap by the square of the beta ratio.
    """
    lat_mid = _octupole_ring(ref)
    tab = propagate_twiss(lat_mid, closed_twiss(lat_mid))
    i_mid = next(i for i, e in enumerate(lat_mid.elements) if isinstance(e, ThinOctupole))
    b_mid_x, b_mid_y = tab[i_mid].beta_x, tab[i_mid].beta_y

    half = [Drift(0.5), Quadrupole(0.3, KF), Drift(0.5), Quadrupole(0.3, KD), Drift(0.5)]
    cell = [*half, *half[::-1]]
    lat_start = Lattice([ThinOctupole(K3L), *cell] * 3, ref)
    tw0 = closed_twiss(lat_start)

    A_mid, A_start = amplitude_detuning(lat_mid), amplitude_detuning(lat_start)
    assert A_start[0, 0] / A_mid[0, 0] == pytest.approx((tw0.beta_x / b_mid_x) ** 2, rel=1e-10)
    assert A_start[1, 1] / A_mid[1, 1] == pytest.approx((tw0.beta_y / b_mid_y) ** 2, rel=1e-10)
    assert tw0.beta_x > b_mid_x and tw0.beta_y < b_mid_y  # the two placements really differ


def test_zero_strength_detunes_by_exactly_nothing(ref: ReferenceParticle) -> None:
    """No octupole, no first-order detuning — and the tracked tunes agree."""
    assert np.array_equal(amplitude_detuning(_octupole_ring(ref, 0.0)), np.zeros((2, 2)))
    lat = _ring(ref)
    small = tracked_tunes(lat, N_TURNS, x0=1e-4, y0=1e-4, nonlinear=True)
    large = tracked_tunes(lat, N_TURNS, x0=8e-4, y0=8e-4, nonlinear=True)
    assert small[0] == pytest.approx(large[0], abs=1e-12)
    assert small[1] == pytest.approx(large[1], abs=1e-12)


# --------------------------------------------------------------------------
# 2. The tracked gate: the order in the action, not a single tolerance
# --------------------------------------------------------------------------


def test_launch_point_has_zero_alpha(ref: ReferenceParticle) -> None:
    """The palindrome puts ``alpha = 0`` at ``s = 0``, so the action has no ``1 + alpha^2``.

    Asserted rather than assumed: the whole tracked comparison converts a launch
    amplitude into an action, and a silent ``1 + alpha^2`` would rescale every
    measured slope.
    """
    tw = closed_twiss(_octupole_ring(ref))
    assert tw.alpha_x == pytest.approx(0.0, abs=1e-12)
    assert tw.alpha_y == pytest.approx(0.0, abs=1e-12)


def test_tracked_detuning_matches_the_closed_form_to_first_order(ref: ReferenceParticle) -> None:
    """The headline. Signal falls by 4 per halving of amplitude; residual falls by 16.

    Linear in the action against quadratic — the two orders are what separate "the
    coefficient is right" from "the tolerance was generous". A wrong coefficient
    leaves a residual that is *linear* in the action, so it would show up as a
    residual ratio of 4, not 16.
    """
    lat_on, lat_off = _octupole_ring(ref), _ring(ref)
    tw = closed_twiss(lat_on)
    A = amplitude_detuning(lat_on)

    signals_x, signals_y, residuals_x, residuals_y = [], [], [], []
    for amp in (8e-4, 4e-4, 2e-4, 1e-4):
        jx, jy = _actions(tw, amp, amp)
        dqx, dqy = _measured_detuning(lat_on, lat_off, amp, amp)
        pred_x, pred_y = A[0, 0] * jx + A[0, 1] * jy, A[1, 0] * jx + A[1, 1] * jy
        signals_x.append(dqx)
        signals_y.append(dqy)
        residuals_x.append(dqx - pred_x)
        residuals_y.append(dqy - pred_y)

    # The two planes move in opposite directions here (the cross term dominates Q_x):
    # a sign error in the off-diagonal would not survive this.
    assert signals_x[0] < 0.0 < signals_y[0]

    for name, sig, res in (("x", signals_x, residuals_x), ("y", signals_y, residuals_y)):
        sig_ratios = [sig[i] / sig[i + 1] for i in range(3)]
        res_ratios = [res[i] / res[i + 1] for i in range(3)]
        assert all(r == pytest.approx(4.0, rel=0.05) for r in sig_ratios), f"{name}: {sig_ratios}"
        assert all(r > 12.0 for r in res_ratios), f"{name}: {res_ratios}"
        assert res_ratios[-1] == pytest.approx(16.0, rel=0.15), f"{name}: {res_ratios}"
        # ...and at the smallest amplitude the first-order form is simply right.
        assert abs(res[-1]) < 3e-3 * abs(sig[-1])


def test_cross_terms_measured_by_a_two_dimensional_fit(ref: ReferenceParticle) -> None:
    """Both columns of the matrix, separately — and the symmetry, measured not imposed.

    ``tracked_tunes`` refuses a zero launch amplitude in either plane (a plane at rest
    has no tune), so one action can never be isolated. Fitting the plane
    ``Q_u = Q_u0 + a J_x + b J_y`` over a small grid recovers the four entries anyway,
    and makes ``dQ_x/dJ_y == dQ_y/dJ_x`` an experimental result.
    """
    lat_on, lat_off = _octupole_ring(ref), _ring(ref)
    tw = closed_twiss(lat_on)
    A = amplitude_detuning(lat_on)

    rows, obs_x, obs_y = [], [], []
    for x0 in (1.0e-4, 1.5e-4, 2.0e-4):
        for y0 in (1.0e-4, 1.5e-4, 2.0e-4):
            jx, jy = _actions(tw, x0, y0)
            dqx, dqy = _measured_detuning(lat_on, lat_off, x0, y0)
            rows.append([jx, jy])
            obs_x.append(dqx)
            obs_y.append(dqy)
    M = np.array(rows)
    fit_x = np.linalg.lstsq(M, np.array(obs_x), rcond=None)[0]
    fit_y = np.linalg.lstsq(M, np.array(obs_y), rcond=None)[0]

    assert fit_x[0] == pytest.approx(A[0, 0], rel=0.02)
    assert fit_x[1] == pytest.approx(A[0, 1], rel=0.02)
    assert fit_y[0] == pytest.approx(A[1, 0], rel=0.02)
    assert fit_y[1] == pytest.approx(A[1, 1], rel=0.02)
    # Symmetry, from two independent fits to two independent measurements.
    assert fit_x[1] == pytest.approx(fit_y[0], rel=0.02)


class _MisScaledOctupole(ThinOctupole):
    """A ``ThinOctupole`` whose kick carries ``1`` where the field expansion says ``1/6``.

    Still a valid magnetic field (curl-free), still exactly symplectic, still with an
    identity Jacobian at the origin — every structural check in
    ``test_octupole_kick.py`` passes on it. ``amplitude_detuning`` sees a
    ``ThinOctupole`` and predicts the *correct* coefficient, so the tracked
    measurement must miss it by exactly 6.
    """

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        out = np.array(state, dtype=float, copy=True)
        x, y = out[0], out[2]
        out[1] -= self.k3l * (x**3 - 3.0 * x * y**2)
        out[3] += self.k3l * (3.0 * x**2 * y - y**3)
        return out


def test_mis_scaled_octupole_is_caught_as_a_factor_of_six(ref: ReferenceParticle) -> None:
    """The gate has teeth. This is the bug no structural check in J2 can see."""
    lat_bad = _ring(ref, [_MisScaledOctupole(K3L)])
    lat_off = _ring(ref)
    tw = closed_twiss(lat_bad)
    A = amplitude_detuning(lat_bad)  # predicts the *correct* 1/6 coefficient

    amp = 1e-4
    jx, jy = _actions(tw, amp, amp)
    dqx, dqy = _measured_detuning(lat_bad, lat_off, amp, amp)
    pred_x, pred_y = A[0, 0] * jx + A[0, 1] * jy, A[1, 0] * jx + A[1, 1] * jy
    assert dqx / pred_x == pytest.approx(6.0, rel=0.02)
    assert dqy / pred_y == pytest.approx(6.0, rel=0.02)


# --------------------------------------------------------------------------
# 3. Honest non-claims and scope, gated rather than documented
# --------------------------------------------------------------------------


def test_sextupole_detuning_is_real_and_deliberately_not_claimed(ref: ReferenceParticle) -> None:
    """A sextupole detunes too — quadratically in ``k2``, and this function reports zero.

    Second-order perturbation theory: the effect is *linear in the action* like the
    octupole's, so it cannot be told apart from it by an amplitude scan — it is told
    apart by its strength scaling, measured here as ``k2^2``. That is exactly why the
    detuning ring carries no sextupoles: this background does not vanish as
    ``k3l -> 0``, and no closed form for it is claimed anywhere in the package.
    """
    lat_off = _ring(ref)
    assert np.array_equal(amplitude_detuning(_ring(ref, [ThinSextupole(3.0)])), np.zeros((2, 2)))

    amp = 4e-4
    shifts = []
    for k2l in (3.0, 6.0):
        dqx, dqy = _measured_detuning(_ring(ref, [ThinSextupole(k2l)]), lat_off, amp, amp)
        shifts.append((dqx, dqy))
    assert abs(shifts[0][0]) > 1e-6  # it is real, not round-off
    assert shifts[1][0] / shifts[0][0] == pytest.approx(4.0, rel=0.05)  # k2^2, not k2
    assert shifts[1][1] / shifts[0][1] == pytest.approx(4.0, rel=0.05)


def test_thick_octupole_approaches_the_thin_one_quadratically(ref: ReferenceParticle) -> None:
    """A thick octupole integrates ``beta^2`` across its body, and that is not ``beta(0)^2``.

    At fixed integrated strength the gap to the thin element closes as ``L^2``, because
    ``beta`` is quadratic about the (centred) body. The thin element is the ``L -> 0``
    limit of the thick one, which is the consistency statement worth having.
    """
    A_thin = amplitude_detuning(_octupole_ring(ref))
    gaps = []
    for L in (0.4, 0.2, 0.1):
        half = [
            Drift(0.5),
            Quadrupole(0.3, KF),
            Drift(0.5),
            Quadrupole(0.3, KD),
            Drift(0.5 - L / 2),
        ]
        lat = Lattice([*half, Octupole(L, K3L / L), *half[::-1]] * 3, ref)
        gaps.append(abs(amplitude_detuning(lat)[0, 0] - A_thin[0, 0]))
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    for r in ratios:
        assert r == pytest.approx(4.0, rel=0.1), f"ratios {ratios} — not O(L^2)"


def test_linear_tracking_shows_no_detuning_at_all(ref: ReferenceParticle) -> None:
    """``nonlinear=False`` linearises the octupole away, so every amplitude has one tune.

    The documented hazard, stated in the currency of this milestone: a detuning study
    run on the default tracking path measures exactly zero, convincingly.
    """
    lat = _octupole_ring(ref)
    small = tracked_tunes(lat, N_TURNS, x0=1e-4, y0=1e-4, nonlinear=False)
    large = tracked_tunes(lat, N_TURNS, x0=8e-4, y0=8e-4, nonlinear=False)
    assert small[0] == pytest.approx(large[0], abs=1e-12)
    assert small[1] == pytest.approx(large[1], abs=1e-12)
    assert amplitude_detuning(lat)[0, 0] != 0.0  # ...while the closed form says otherwise
