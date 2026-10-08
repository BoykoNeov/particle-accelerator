"""Driven optics (W2): the beam an AC dipole swings sees a ring with one more quadrupole.

W1 found that in steady state the dipole's kick on turn ``n`` is in phase with the beam's
own position there: ``theta_n = g x_n`` with ``g = kick / x_hat`` real. A kick
proportional to ``x`` is a thin gradient, so the driven motion is a *free* oscillation of
the ring with one extra thin gradient at the dipole — one that acts in the driven plane
only (it is not a Maxwellian quadrupole) and pulls the tune exactly onto the drive tune
``nu``. Its strength is Miyamoto's effective gradient (xtrack's ``eff_grad``),

    g = 2 (cos 2 pi nu - cos 2 pi Q) / (beta sin 2 pi Q),        p_u -> p_u + g u,

independent of the drive's amplitude, lag and ramp. The driven beta function and phase are
that ring's Twiss — what an optics measurement made with an AC dipole actually reads, and
why it must be corrected back to the natural optics. Miyamoto et al., PRST-AB 11, 084002
(2008), with ``lambda = sin pi(nu - Q) / sin pi(nu + Q)`` and ``phi`` the natural phase
downstream of the dipole:

    beta_d / beta = (1 + lambda^2 - 2 lambda cos(2 phi - 2 pi Q)) / (1 - lambda^2),
    tan(psi_d - pi nu) = (1 + lambda)/(1 - lambda) tan(phi - pi Q).

All of it is derived below with sympy, by two routes for ``g``. The sharp gate needs no
arbiter: W1's exact steady-state solution, carried to every element boundary, *is* an
eigen-solution of the driven ring, so its ``p_hat / x_hat = (i - alpha_d) / beta_d`` and
its phase reproduce the driven Twiss pointwise, both sides of the dipole.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    PROTON_MASS_EV,
    PX,
    PY,
    ACDipole,
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    X,
    Y,
)
from accsim.twiss import (
    ResonantLatticeError,
    closed_twiss,
    driven_gradient,
    driven_twiss,
    match_periodic,
    propagate_twiss,
)

REF = ReferenceParticle.from_gamma(PROTON_MASS_EV, 10.0)
LD, NCELL = 5.0, 6
IDX = 2  # after the first drift (or bend), where alpha != 0
OFFSET = 0.012  # |Q - nu|, as in W1
KICK, LAG = 1.0e-5, 0.13
# thin FODO: 77.4 deg/cell (Q = 1.29); 101.6 deg/cell puts frac(Q) = 0.69 above the half
K1L = {"fodo": 0.25, "high": 0.31, "bent": 0.25}
ANGLE = 2.0 * math.pi / (2 * NCELL)  # "bent": every drift is a sector bend, so D_x != 0
# measured floors over the 12 ring x plane x side cases: beta 4e-15, alpha 3e-15,
# phase 6e-15, action 8e-15
GATE = 1e-12


def _ring(kind: str, *acds: tuple[int, ACDipole]) -> Lattice:
    k1l = K1L[kind]
    els: list = []
    for _ in range(NCELL):
        body = Dipole(LD, ANGLE) if kind == "bent" else Drift(LD)
        body2 = Dipole(LD, ANGLE) if kind == "bent" else Drift(LD)
        els += [ThinQuadrupole(k1l), body, ThinQuadrupole(-k1l), body2]
    for idx, acd in sorted(acds, key=lambda t: -t[0]):
        els.insert(idx, acd)
    return Lattice(els, REF)


def _natural(kind: str) -> list:
    lat = _ring(kind)
    return propagate_twiss(lat, closed_twiss(lat))


def _q(kind: str, plane: str) -> float:
    end = _natural(kind)[-1]
    return (end.mu_x if plane == "x" else end.mu_y) / (2.0 * math.pi)


def _nu(kind: str, plane: str, side: int) -> float:
    """Drive tune ``frac(Q) - side * OFFSET``: ``side = +1`` below the tune, ``-1`` above."""
    return _q(kind, plane) % 1.0 - side * OFFSET


def _drive(kind: str, plane: str = "x", side: int = 1, **kw) -> ACDipole:
    kw.setdefault("kick", KICK)
    kw.setdefault("lag", LAG)
    return ACDipole(tune=_nu(kind, plane, side), plane=plane, **kw)


def _uv(plane: str) -> tuple[int, int]:
    return (X, PX) if plane == "x" else (Y, PY)


def _steady_state(lat: Lattice, idx: int, acd: ACDipole) -> np.ndarray:
    """W1's exact steady state as complex 6-vectors at every element boundary.

    Row 0 is the lattice start, row ``i + 1`` the exit of element ``i``; the dipole's kick
    ``b c`` is added at its exit. No driven-optics code is used: this is the linear solve.
    """
    mats = [e.matrix(lat.ref) for e in lat.elements]
    A = np.eye(6)
    for M in mats[:idx]:
        A = M @ A
    B = np.eye(6)
    for M in mats[idx + 1 :]:
        B = M @ B
    lam = np.exp(2j * np.pi * acd.tune)
    b = np.zeros(6)
    b[_uv(acd.plane)[1]] = 1.0
    c = acd.amplitude * np.exp(2j * np.pi * acd.lag)
    z = np.linalg.solve(lam * np.eye(6) - B @ A, B @ b * c)
    out = [z]
    for i, M in enumerate(mats):
        z = M @ z
        if i == idx:
            z = z + b * c
        out.append(z)
    return np.array(out)


def _driven_fields(tw: list, plane: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane == "x":
        return (
            np.array([t.beta_x for t in tw]),
            np.array([t.alpha_x for t in tw]),
            np.array([t.mu_x for t in tw]),
        )
    return (
        np.array([t.beta_y for t in tw]),
        np.array([t.alpha_y for t in tw]),
        np.array([t.mu_y for t in tw]),
    )


def _identity_miss(zs: np.ndarray, tw: list, plane: str) -> dict[str, float]:
    """How far a Twiss list is from the steady state's own optics, per quantity."""
    u, pu = _uv(plane)
    r = zs[:, pu] / zs[:, u]  # (i - alpha_d) / beta_d for an eigen-solution at e^{+2 pi i nu}
    beta, alpha, mu = _driven_fields(tw, plane)
    arg = np.angle(zs[:, u])
    amp2 = np.abs(zs[:, u]) ** 2 / beta  # the driven action (times 2): constant
    return {
        "beta": float(np.max(np.abs(1.0 / r.imag - beta) / beta)),
        "alpha": float(np.max(np.abs(-r.real / r.imag - alpha) / (1.0 + np.abs(alpha)))),
        "phase": float(np.max(np.abs(np.exp(1j * (arg - arg[0])) - np.exp(1j * (mu - mu[0]))))),
        "action": float(np.ptp(amp2) / np.mean(amp2)),
    }


# --- the derivations ---------------------------------------------------------------------
def _cs(mu, al, be):
    ga = (1 + al**2) / be
    return sp.Matrix(
        [
            [sp.cos(mu) + al * sp.sin(mu), be * sp.sin(mu)],
            [-ga * sp.sin(mu), sp.cos(mu) - al * sp.sin(mu)],
        ]
    )


def _rot(a):
    return sp.Matrix([[sp.cos(a), sp.sin(a)], [-sp.sin(a), sp.cos(a)]])


def test_the_gradient_is_derived_by_two_independent_routes() -> None:
    """(i) the trace: a thin ``p += g u`` at the dipole puts the tune on ``nu``;
    (ii) W1's steady state: ``kick / x_hat``. Both are xtrack's ``eff_grad``."""
    mu, th, al, be, g = sp.symbols("mu theta alpha beta g", real=True, positive=True)
    M0 = _cs(mu, al, be)  # one turn from the dipole back to it, natural
    eff = 2 * (sp.cos(th) - sp.cos(mu)) / (be * sp.sin(mu))
    kick = sp.Matrix([[1, 0], [g, 1]])
    (trace_route,) = sp.solve(sp.Eq((kick * M0).trace(), 2 * sp.cos(th)), g)
    assert sp.simplify(trace_route - eff) == 0
    lam = sp.exp(sp.I * th)
    xhat = ((lam * sp.eye(2) - M0).inv() * M0 * sp.Matrix([0, 1]))[0]  # per unit kick
    assert sp.simplify((1 / xhat - eff).rewrite(sp.exp)) == 0


def test_miyamotos_beta_and_phase_are_derived_not_remembered() -> None:
    """Floquet coordinates (natural beta = 1, alpha = 0): the driven ring is ``R(phi) G
    R(2 pi Q - phi)`` with ``G`` the normalised gradient ``g beta``."""
    mu, th, phi = sp.symbols("mu theta phi", real=True, positive=True)
    G = sp.Matrix([[1, 0], [2 * (sp.cos(th) - sp.cos(mu)) / sp.sin(mu), 1]])
    lam = sp.sin((th - mu) / 2) / sp.sin((th + mu) / 2)
    # beta_d / beta at natural phase phi downstream of the dipole
    beta_d = (_rot(phi) * G * _rot(mu - phi))[0, 1] / sp.sin(th)
    miyamoto = (1 + lam**2 - 2 * lam * sp.cos(2 * phi - mu)) / (1 - lam**2)
    assert sp.simplify((beta_d - miyamoto).rewrite(sp.exp)) == 0
    # the driven phase advance from the dipole's exit to phi
    M1 = G * _rot(mu)
    b0 = M1[0, 1] / sp.sin(th)
    a0 = (M1[0, 0] - M1[1, 1]) / (2 * sp.sin(th))
    C = _rot(phi)
    tan_psi = C[0, 1] / (b0 * C[0, 0] - a0 * C[0, 1])
    t1, t2 = sp.tan(th / 2), (1 + lam) / (1 - lam) * sp.tan(phi - mu / 2)
    assert sp.simplify((tan_psi - (t1 + t2) / (1 - t1 * t2)).rewrite(sp.exp)) == 0


# --- the gradient --------------------------------------------------------------------------
@pytest.mark.parametrize("plane", ["x", "y"])
@pytest.mark.parametrize("side", [1, -1])
def test_the_gradient_is_the_kick_over_the_steady_state_amplitude(plane: str, side: int) -> None:
    """Numerically, W1's solve: ``c / x_hat`` is real (in phase) and equals ``g``."""
    acd = _drive("fodo", plane, side)
    lat = _ring("fodo", (IDX, acd))
    zs = _steady_state(lat, IDX, acd)
    xhat = zs[IDX, _uv(plane)[0]]  # at the dipole's entrance: before its kick
    ratio = acd.amplitude * np.exp(2j * np.pi * acd.lag) / xhat
    g = driven_gradient(lat)[plane]
    assert abs(ratio.imag) < 1e-12 * abs(ratio.real)  # measured ~1e-16
    assert ratio.real == pytest.approx(g, rel=1e-12)
    # below the tune it defocuses (pushes the tune down onto nu), above it focuses
    assert np.sign(g) == side


def test_the_gradient_ignores_amplitude_lag_and_ramp() -> None:
    a = _ring("fodo", (IDX, _drive("fodo", kick=1e-5, lag=0.0)))
    b = _ring("fodo", (IDX, _drive("fodo", kick=3e-4, lag=0.37, ramp=(0, 100, 200, 300))))
    assert driven_gradient(a) == driven_gradient(b)
    np.testing.assert_array_equal(
        [t.beta_x for t in driven_twiss(a)], [t.beta_x for t in driven_twiss(b)]
    )


# --- the identity: the steady state IS the driven ring's eigen-solution ---------------------
@pytest.mark.parametrize("kind", ["fodo", "high", "bent"])
@pytest.mark.parametrize("plane", ["x", "y"])
@pytest.mark.parametrize("side", [1, -1])
def test_the_steady_state_carries_the_driven_optics_everywhere(
    kind: str, plane: str, side: int
) -> None:
    """``beta_d``, ``alpha_d``, phase and a constant action, at every boundary."""
    acd = _drive(kind, plane, side)
    lat = _ring(kind, (IDX, acd))
    miss = _identity_miss(_steady_state(lat, IDX, acd), driven_twiss(lat), plane)
    for name, value in miss.items():
        assert value < GATE, (name, value)


def test_alpha_jumps_across_the_dipole_and_the_steady_state_sees_it() -> None:
    """The thin gradient kinks the driven beta: ``alpha_d`` jumps by ``-g beta_d``."""
    acd = _drive("fodo")
    lat = _ring("fodo", (IDX, acd))
    tw = driven_twiss(lat)
    g = driven_gradient(lat)["x"]
    before, after = tw[IDX], tw[IDX + 1]  # the dipole's entrance and exit
    assert after.beta_x == pytest.approx(before.beta_x, rel=1e-14)
    assert after.alpha_x - before.alpha_x == pytest.approx(-g * before.beta_x, rel=1e-10)
    zs = _steady_state(lat, IDX, acd)
    r_in, r_out = zs[IDX, PX] / zs[IDX, X], zs[IDX + 1, PX] / zs[IDX + 1, X]
    assert -r_in.real / r_in.imag == pytest.approx(before.alpha_x, rel=1e-10)
    assert -r_out.real / r_out.imag == pytest.approx(after.alpha_x, rel=1e-10)


@pytest.mark.parametrize("control", ["natural", "flipped"])
def test_the_identity_rejects_the_natural_optics_and_a_flipped_gradient(control: str) -> None:
    acd = _drive("fodo")
    lat = _ring("fodo", (IDX, acd))
    zs = _steady_state(lat, IDX, acd)
    if control == "natural":
        tw = propagate_twiss(lat, closed_twiss(lat))
    else:
        g = driven_gradient(lat)["x"]
        maps = [e.matrix(lat.ref) for e in lat.elements]
        maps[IDX] = np.eye(6)
        maps[IDX][PX, X] = -g
        one = np.eye(6)
        for M in maps:
            one = M @ one
        tw = propagate_twiss(lat, match_periodic(one), maps=maps)
    miss = _identity_miss(zs, tw, "x")
    # measured: natural 8.0% beta, 9.6% phase; flipped 17% beta, 19% phase
    assert miss["beta"] > 0.03, miss  # the beat: 2 lambda / (1 - lambda^2) ~ 8% here
    assert miss["phase"] > 0.03, miss


# --- the closed forms, on the ring ---------------------------------------------------------
@pytest.mark.parametrize("kind", ["fodo", "high"])
@pytest.mark.parametrize("side", [1, -1])
def test_miyamotos_closed_forms_hold_around_the_ring(kind: str, side: int) -> None:
    acd = _drive(kind, "x", side)
    lat = _ring(kind, (IDX, acd))
    nat = _natural(kind)
    nat.insert(IDX + 1, nat[IDX])  # the identity dipole: its exit is its entrance
    drv = driven_twiss(lat)
    Q, nu = _q(kind, "x"), acd.tune
    lam = math.sin(math.pi * (nu - Q)) / math.sin(math.pi * (nu + Q))
    two_pi_q = 2.0 * math.pi * Q
    mu_dip = nat[IDX].mu_x
    for i, (n, d) in enumerate(zip(nat, drv, strict=True)):
        # natural phase downstream of the dipole's exit, in (0, 2 pi Q]
        phi = n.mu_x - mu_dip + (two_pi_q if i <= IDX else 0.0)
        ratio = (1 + lam**2 - 2 * lam * math.cos(2 * phi - two_pi_q)) / (1 - lam**2)
        assert d.beta_x / n.beta_x == pytest.approx(ratio, rel=1e-11), i
        psi = d.mu_x - drv[IDX + 1].mu_x + (drv[-1].mu_x if i <= IDX else 0.0)
        want = math.atan((1 + lam) / (1 - lam) * math.tan(phi - math.pi * Q))
        gap = (psi - math.pi * nu - want) % math.pi  # tan is pi-periodic
        assert min(gap, math.pi - gap) < 1e-11, i
    assert 0.03 < 2 * abs(lam) / (1 - lam**2) < 0.2  # a beat a measurement would see


# --- the tune, the other plane, the dispersion ---------------------------------------------
@pytest.mark.parametrize("kind", ["fodo", "high", "bent"])
@pytest.mark.parametrize("side", [1, -1])
def test_the_driven_tune_is_the_drive_tune_with_the_natural_integer(kind: str, side: int) -> None:
    acd = _drive(kind, "x", side)
    end = driven_twiss(_ring(kind, (IDX, acd)))[-1]
    Q = _q(kind, "x")
    assert end.mu_x / (2.0 * math.pi) == pytest.approx(math.floor(Q) + acd.tune, abs=1e-12)


def test_a_drive_at_one_minus_nu_is_the_same_drive() -> None:
    """``sin(2 pi (1 - nu) n + l) = -sin(2 pi nu n - l)``: the same kicks, the same optics."""
    acd = _drive("fodo")
    twin = ACDipole(kick=KICK, tune=1.0 - acd.tune, lag=-LAG, plane="x")
    a = driven_twiss(_ring("fodo", (IDX, acd)))
    b = driven_twiss(_ring("fodo", (IDX, twin)))
    for p, q in zip(a, b, strict=True):
        assert (q.beta_x, q.alpha_x, q.mu_x) == pytest.approx(
            (p.beta_x, p.alpha_x, p.mu_x), rel=1e-12, abs=1e-12
        )


def test_the_undriven_plane_and_the_dispersion_are_the_natural_ones() -> None:
    """Bit for bit: the drive's frequency is not zero, so the static closed orbit (and its
    momentum derivative) is untouched — a matched 'dispersion' of the substituted ring
    would be a number with no physical meaning."""
    acd = _drive("bent", "x")
    lat = _ring("bent", (IDX, acd))
    drv = driven_twiss(lat)
    nat = propagate_twiss(lat, closed_twiss(lat))
    assert max(abs(t.disp_x) for t in nat) > 0.5  # the bends really do disperse
    for d, n in zip(drv, nat, strict=True):
        assert (d.s, d.beta_y, d.alpha_y, d.mu_y) == (n.s, n.beta_y, n.alpha_y, n.mu_y)
        assert (d.disp_x, d.disp_px, d.disp_y, d.disp_py) == (
            n.disp_x,
            n.disp_px,
            n.disp_y,
            n.disp_py,
        )


def test_one_dipole_per_plane_drives_each_plane_independently() -> None:
    """An uncoupled ring superposes: each plane's driven optics is its own dipole's."""
    ax, ay = _drive("fodo", "x"), _drive("fodo", "y", side=-1)
    jy = IDX + 4  # in the original element list; with ax inserted first it lands at jy + 1
    both = driven_twiss(_ring("fodo", (IDX, ax), (jy, ay)))
    x_only = driven_twiss(_ring("fodo", (IDX, ax)))
    y_only = driven_twiss(_ring("fodo", (jy, ay)))
    # drop the other dipole's exit point (it is an identity there) and compare
    bx = [t for i, t in enumerate(both) if i != jy + 2]
    by = [t for i, t in enumerate(both) if i != IDX + 1]
    for a, b in zip(bx, x_only, strict=True):
        assert (a.beta_x, a.alpha_x, a.mu_x) == pytest.approx(
            (b.beta_x, b.alpha_x, b.mu_x), rel=1e-13, abs=1e-13
        )
    for a, b in zip(by, y_only, strict=True):
        assert (a.beta_y, a.alpha_y, a.mu_y) == pytest.approx(
            (b.beta_y, b.alpha_y, b.mu_y), rel=1e-13, abs=1e-13
        )
    assert by[-1].mu_y != pytest.approx(_natural("fodo")[-1].mu_y)  # y really is driven


# --- refusals -----------------------------------------------------------------------------
def test_refusals() -> None:
    with pytest.raises(ValueError, match="no ACDipole"):
        driven_twiss(_ring("fodo"))
    two = _ring("fodo", (IDX, _drive("fodo")), (IDX + 4, _drive("fodo", side=-1)))
    with pytest.raises(ValueError, match="one ACDipole per plane"):
        driven_twiss(two)
    fq = _q("fodo", "x") % 1.0
    for nu in (fq, 1.0 - fq, 3.0 + fq):  # on the tune: x_hat diverges, g = 0
        with pytest.raises(ResonantLatticeError, match="resonance"):
            driven_twiss(_ring("fodo", (IDX, ACDipole(KICK, nu))))
    for nu in (0.0, 0.5, 2.0):  # the driven ring would sit on an integer or half-integer
        with pytest.raises(ResonantLatticeError, match="integer"):
            driven_twiss(_ring("fodo", (IDX, ACDipole(KICK, nu))))
