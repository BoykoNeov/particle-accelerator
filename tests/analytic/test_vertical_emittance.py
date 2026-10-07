r"""V2 — vertical emittance from design vertical dispersion.

V1 let a bend leave the horizontal plane. A ring that bends vertically has **design**
vertical dispersion ``D_y``, and wherever the beam radiates a photon, the energy kick
meets that dispersion and kicks the vertical betatron amplitude. The equilibrium is the
textbook balance, one plane over:

    eps_y = C_q gamma^2 I5y / (J_y I2),      I5y = ∮ |h|^3 H_y ds,
    J_y   = 1 - I4y / I2,                    I4y = ∮ D_y g_y ds,

with ``H_y = gamma_y D_y^2 + 2 alpha_y D_y D_y' + beta_y D_y'^2`` and ``g`` the gradient of
the radiated power across the orbit. A bend's ``g`` points along its **own** curvature,
``g = h (h^2 + 2 k1) e_bend`` (the ``-h^2 tan e`` face term likewise), so a tilted bend
feeds the horizontal and vertical damping by ``cos t`` and ``sin t`` of one number.

The gates, in order of how much they would catch:

1. **Nothing moves on a flat ring** — every shipped number to the bit, against a frozen
   copy of the pre-V2 sum.
2. **The rolled-ring identity, with a gradient.** A ring turned over by ``pi/2`` is the same
   machine; its vertical integrals must be the flat ring's horizontal ones. At ``k1 = 0`` the
   gradient term vanishes, so this gate runs with ``k1 != 0`` — that is what pins the sign of
   ``2 k1`` in the vertical plane, which xtrack's integral route gets wrong (see the
   reference file).
3. **The ``H_y`` invariant.** Outside a vertical bend ``D_y`` obeys the homogeneous betatron
   equation, so ``H_y`` is its Courant-Snyder invariant: constant around the ring, and equal
   to ``|d_n|^2 / (4 sin^2 pi Q_y)`` for the dispersion ``d`` the dogleg generates. Every
   horizontal bend's share of ``I5y`` is that number times ``|h|^3 L`` — a closed form that
   shares nothing with the transport sum.
4. **``I1 = alpha_c C`` with the vertical half in it** — against the MAD-X-validated
   compaction identity route.
5. **The tracked equilibrium** (B3's Lyapunov solve on the radiating, tilted ring): a route
   through the tracking map, not the optics, departing as ``(2 pi Q_s)^2`` as B3 established.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_design_tilt as v1  # noqa: E402  (sibling fixtures: the dogleg ring)
import test_radiation_quantum as b3  # noqa: E402  (sibling machinery: the Lyapunov solve)
from scipy.integrate import quad

from accsim import Dipole, Drift, Lattice, Quadrupole, RFCavity
from accsim.coords import DELTA, PY, Y
from accsim.elements.dipole import _edge_matrix
from accsim.radiation import (
    _curly_h,
    damping_partition_numbers,
    damping_times,
    equilibrium_emittance,
    equilibrium_emittances_coupled,
    equilibrium_energy_spread,
    equilibrium_vertical_emittance,
    polarization_integrals,
    quantum_constant_cq,
    radiation_integrals,
)
from accsim.twiss import (
    CoupledLatticeError,
    _blocks,
    _dispersive_kick,
    _propagate_block,
    _transverse_4d,
    closed_twiss,
    momentum_compaction,
    propagate_twiss,
    tunes,
)

PI2 = math.pi / 2


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def _cf_cell(tilt: float, k1: float, e: float) -> list:
    """A combined-function FODO cell with pole faces — every term of ``I4`` live.

    ``k1 = -0.05`` is a defocusing gradient (the cell is unstable for ``k1 > 0``).
    ``tilt`` turns the whole cell with its frame: bends by ``tilt``, quadrupoles by ``roll``.
    """
    return [
        Quadrupole(0.5, 0.30, roll=tilt),
        Drift(0.6),
        Dipole(1.5, v1.ANG, k1=k1, e1=e, e2=e, tilt=tilt),
        Drift(0.6),
        Quadrupole(0.5, -0.30, roll=tilt),
        Drift(0.6),
        Dipole(1.5, v1.ANG, k1=k1, e1=e, e2=e, tilt=tilt),
        Drift(0.6),
    ]


def cf_ring(tilt: float = 0.0, k1: float = -0.05, e: float = 0.1) -> Lattice:
    return Lattice([x for _ in range(v1.NCELL) for x in _cf_cell(tilt, k1, e)], ref=v1._ref())


def _sliced(lat: Lattice, n: int) -> Lattice:
    """Every bend cut into ``n`` equal pieces (tilt kept) — the same machine, for tracking."""
    out = []
    for e in lat.elements:
        if isinstance(e, Dipole) and e.angle != 0.0:
            assert e.k1 == 0.0 and e.e1 == 0.0 and e.e2 == 0.0  # pieces would need faces split
            out += [Dipole(e.length / n, e.angle / n, tilt=e.tilt) for _ in range(n)]
        else:
            out.append(e)
    return Lattice(out, ref=lat.ref)


def _pre_v2_integrals(lattice: Lattice, slices: int = 64) -> tuple[float, ...]:
    """The shipped Dipole branch of ``radiation_integrals`` exactly as it stood before V2.

    Frozen here verbatim (minus the wiggler branch, which V2 does not touch) so that "a flat
    ring moves by no bit" is a comparison against code, not against captured floats that a
    different platform could round differently.
    """
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    i1 = i2 = i3 = i4 = i5 = 0.0
    for elem in lattice.elements:
        M = elem.matrix(lattice.ref)
        if isinstance(elem, Dipole) and elem.angle != 0.0 and elem.length > 0.0:
            h = elem.curvature
            k1 = elem.k1
            ds = elem.length / slices
            sub = Dipole(ds, h * ds, k1=k1).matrix(lattice.ref)
            sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
            xblock = _blocks(sub)[0]
            dx_entrance = disp[0]
            if elem.e1 != 0.0:
                ent = _edge_matrix(h, elem.e1)
                disp = _transverse_4d(ent) @ disp
                bx, ax, _ = _propagate_block(_blocks(ent)[0], bx, ax)
            acc_dx = 0.5 * disp[0]
            acc_h = 0.5 * _curly_h(bx, ax, disp[0], disp[1])
            for i in range(slices):
                disp = sub4 @ disp + subk
                bx, ax, _ = _propagate_block(xblock, bx, ax)
                w = 0.5 if i == slices - 1 else 1.0
                acc_dx += w * disp[0]
                acc_h += w * _curly_h(bx, ax, disp[0], disp[1])
            int_dx = acc_dx * ds
            dx_exit = disp[0]
            if elem.e2 != 0.0:
                ext = _edge_matrix(h, elem.e2)
                disp = _transverse_4d(ext) @ disp
                bx, ax, _ = _propagate_block(_blocks(ext)[0], bx, ax)
            i1 += h * int_dx
            i2 += h * h * elem.length
            i3 += abs(h) ** 3 * elem.length
            i4 += h * (h * h + 2.0 * k1) * int_dx
            i4 -= h * h * (dx_entrance * math.tan(elem.e1) + dx_exit * math.tan(elem.e2))
            i5 += abs(h) ** 3 * acc_h * ds
            continue
        bx, ax, _ = _propagate_block(_blocks(M)[0], bx, ax)
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)
    return i1, i2, i3, i4, i5


# ---------------------------------------------------------------------------
# Gate 1 — a flat ring moves by no bit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ring", [v1.flat_ring, cf_ring], ids=["sector", "combined+faces"])
def test_a_flat_ring_moves_by_no_bit(ring) -> None:
    lat = ring()
    ri = radiation_integrals(lat)
    assert (ri.i1, ri.i2, ri.i3, ri.i4, ri.i5) == _pre_v2_integrals(lat)
    assert ri.i4y == 0.0 and ri.i5y == 0.0  # exactly: D_y is exactly zero on a flat ring
    jx, jy, jz = damping_partition_numbers(lat)
    assert jy == 1.0
    assert (jx, jz) == (1.0 - ri.i4 / ri.i2, 2.0 + ri.i4 / ri.i2)
    assert equilibrium_vertical_emittance(lat) == 0.0


# ---------------------------------------------------------------------------
# Gate 2 — the machine turned over is the same machine.
# ---------------------------------------------------------------------------
def test_the_rolled_ring_swaps_the_planes_with_a_gradient() -> None:
    """``tilt = pi/2`` on every element: the flat ring standing on its side.

    Run with a gradient and pole faces, because at ``k1 = 0`` the ``2 k1`` term — whose sign
    in the vertical plane is the one thing a rotation could get wrong — is identically zero.
    """
    flat, rolled = radiation_integrals(cf_ring()), radiation_integrals(cf_ring(tilt=PI2))
    assert rolled.i4y == pytest.approx(flat.i4, rel=1e-13, abs=0)
    assert rolled.i5y == pytest.approx(flat.i5, rel=1e-13, abs=0)
    assert rolled.i1 == pytest.approx(flat.i1, rel=1e-13, abs=0)
    assert (rolled.i2, rolled.i3) == (flat.i2, flat.i3)  # pure geometry, untouched
    # The horizontal plane of the rolled ring is now dispersion-free; what is left is
    # cos(pi/2) ~ 6e-17 in floating point, squared.
    assert abs(rolled.i4) < 1e-14 * flat.i4 and abs(rolled.i5) < 1e-14 * flat.i5
    # The gradient really is load-bearing here. Every bend shares h and k1, so the body
    # term is (h^2 + 2 k1) * I1 and the gradient's share is exactly 2 k1 I1; flipping its
    # sign would move I4y by twice that, which is most of I4 itself.
    gradient_share = 2.0 * -0.05 * flat.i1
    assert abs(2.0 * gradient_share) > 0.5 * abs(flat.i4)


def test_the_rolled_ring_swaps_the_damping_and_the_emittance() -> None:
    jx, jy, jz = damping_partition_numbers(cf_ring())
    rx, ry, rz = damping_partition_numbers(cf_ring(tilt=PI2))
    assert ry == pytest.approx(jx, rel=1e-12, abs=0)
    assert rx == pytest.approx(1.0, rel=1e-14, abs=0)
    assert rz == pytest.approx(jz, rel=1e-13, abs=0)
    assert equilibrium_vertical_emittance(cf_ring(tilt=PI2)) == pytest.approx(
        equilibrium_emittance(cf_ring()), rel=1e-12, abs=0
    )
    assert equilibrium_energy_spread(cf_ring(tilt=PI2)) == pytest.approx(
        equilibrium_energy_spread(cf_ring()), rel=1e-13, abs=0
    )
    tx, ty, tz = damping_times(cf_ring())
    ux, uy, uz = damping_times(cf_ring(tilt=PI2))
    assert (uy, uz) == (pytest.approx(tx, rel=1e-12, abs=0), pytest.approx(tz, rel=1e-12, abs=0))


def test_a_half_turn_is_the_flat_ring() -> None:
    """``tilt = pi`` reverses every bend *and* the dispersion with it, so ``D h`` keeps its
    sign: the integrals are the flat ring's. This is the gate a projection by ``|cos t|``
    (or ``cos^2 t``) fails — both pass gate 2, and both flip ``I4`` and ``I1`` here."""
    flat, half = radiation_integrals(cf_ring()), radiation_integrals(cf_ring(tilt=math.pi))
    assert half.i4 == pytest.approx(flat.i4, rel=1e-12, abs=0)
    assert half.i5 == pytest.approx(flat.i5, rel=1e-12, abs=0)
    assert half.i1 == pytest.approx(flat.i1, rel=1e-12, abs=0)
    assert abs(half.i4y) < 1e-13 * flat.i4


# ---------------------------------------------------------------------------
# Gate 3 — the H_y invariant, a closed form outside the transport.
# ---------------------------------------------------------------------------
def _dogleg_invariant() -> tuple[float, int, int]:
    """``(H_y, i_vb1, i_vb2)``: the closed-form invariant of the dogleg ring's ``D_y``.

    ``d`` is the dispersion the dogleg makes from nothing, read off its own composite map;
    the periodic ``D = (I - M_y)^-1 d`` has, in normalised coordinates, ``|D_n| =
    |d_n| / (2 |sin(pi Q_y)|)`` because ``I - R(mu)`` is ``2 sin(mu/2)`` times a rotation.
    """
    lat = v1.dogleg_ring()
    names = [e.name for e in lat.elements]
    i1, i2 = names.index("vb1"), names.index("vb2")
    dog = np.eye(6)
    for e in lat.elements[i1 : i2 + 1]:
        dog = e.matrix(lat.ref) @ dog
    d0, d1 = dog[Y, DELTA], dog[PY, DELTA]
    at = propagate_twiss(lat, closed_twiss(lat))[i2 + 1]  # the exit of vb2
    gy = (1.0 + at.alpha_y**2) / at.beta_y
    dn2 = gy * d0 * d0 + 2.0 * at.alpha_y * d0 * d1 + at.beta_y * d1 * d1
    qy = tunes(lat)[1]
    return dn2 / (4.0 * math.sin(math.pi * qy) ** 2), i1, i2


def _vertical_bend_share(lat: Lattice, index: int) -> float:
    """``∫ |h|^3 H_y ds`` across one ``tilt = pi/2`` sector bend, from its analytic orbit.

    In the bend's frame its own ``x`` is the lab ``y`` (``s_rotation(pi/2)``), so ``D_y``
    obeys ``D'' + h^2 D = h``: ``D(s) = D0 cos + D0'/h sin + (1 - cos)/h``. ``beta_y``
    rides the same weak focusing. Integrated by adaptive quadrature — no slicing anywhere.
    """
    elem = lat.elements[index]
    assert elem.tilt == PI2 and elem.k1 == 0.0
    tw = propagate_twiss(lat, closed_twiss(lat))[index]
    h = elem.curvature
    b0, a0, d0, p0 = tw.beta_y, tw.alpha_y, tw.disp_y, tw.disp_py
    g0 = (1.0 + a0 * a0) / b0

    def integrand(s: float) -> float:
        c, sn = math.cos(h * s), math.sin(h * s)
        m11, m12, m21, m22 = c, sn / h, -h * sn, c
        beta = m11 * m11 * b0 - 2 * m11 * m12 * a0 + m12 * m12 * g0
        alpha = -m11 * m21 * b0 + (m11 * m22 + m12 * m21) * a0 - m12 * m22 * g0
        d = d0 * c + p0 * sn / h + (1.0 - c) / h
        dp = -d0 * h * sn + p0 * c + sn
        return abs(h) ** 3 * _curly_h(beta, alpha, d, dp)

    return quad(integrand, 0.0, elem.length, epsabs=0, epsrel=1e-13)[0]


def test_vertical_excitation_is_the_invariant_times_the_geometry() -> None:
    lat = v1.dogleg_ring()
    hy, i1, i2 = _dogleg_invariant()
    horizontal = sum(
        abs(e.curvature) ** 3 * e.length
        for e in lat.elements
        if isinstance(e, Dipole) and e.angle != 0.0 and e.tilt == 0.0
    )
    exact = hy * horizontal + _vertical_bend_share(lat, i1) + _vertical_bend_share(lat, i2)
    # The horizontal bends carry nearly all of it: 99.64%. (Pre-committed as 99.99% from
    # |h|^3 alone; the dogleg's bends carry more because D_y' is large inside them.)
    assert 0.99 < hy * horizontal / exact < 0.999
    # The trapezoid is exact where H_y is constant, so its error lives in the two vertical
    # bends alone and falls as 1/n^2 — gate the law, then the converged number.
    # Measured: 4.9e-7 relative at 64 slices, 3e-8 at 256.
    err = [radiation_integrals(lat, slices=n).i5y - exact for n in (8, 16, 32, 64)]
    assert err[0] / err[1] == pytest.approx(4.0, rel=2e-2)
    assert err[1] / err[2] == pytest.approx(4.0, rel=2e-2)
    assert err[2] / err[3] == pytest.approx(4.0, rel=2e-2)
    # Richardson: strip the 1/n^2 term and what is left is the closed form, to 1e-10.
    richardson = (4.0 * err[3] - err[2]) / 3.0
    assert abs(richardson) < 1e-10 * exact


def test_the_invariant_is_what_the_formula_says() -> None:
    """The closed form ``|d_n|^2 / (4 sin^2 pi Q_y)`` against ``H_y`` evaluated from the
    matched optics at a point far from the dogleg — the ring's entrance."""
    lat = v1.dogleg_ring()
    hy, _, _ = _dogleg_invariant()
    tw = closed_twiss(lat)
    assert _curly_h(tw.beta_y, tw.alpha_y, tw.disp_y, tw.disp_py) == pytest.approx(
        hy, rel=1e-12, abs=0
    )


# ---------------------------------------------------------------------------
# Gate 4 — I1 is the momentum compaction, vertical half included.
# ---------------------------------------------------------------------------
def test_i1_is_the_momentum_compaction_with_its_vertical_half() -> None:
    """Against the identity route (the one-turn map's ``zeta`` row — no slicing, and the
    route MAD-X's ``alfa`` meets at ``1e-10``). The trapezoid's error is ``1/n^2`` — measured
    ``5.5e-7`` relative at 64 slices and ``3.5e-8`` at 256, a factor 16, identical on the
    flat ring — so the law is the gate."""
    lat = v1.dogleg_ring()
    exact = momentum_compaction(lat) * lat.length
    err = [radiation_integrals(lat, slices=n).i1 / exact - 1.0 for n in (64, 128, 256)]
    assert err[0] / err[1] == pytest.approx(4.0, rel=1e-2)
    assert err[1] / err[2] == pytest.approx(4.0, rel=1e-2)
    # Discrimination: the vertical half is 3.9e-5 of the whole — 1000x the 256-slice error.
    vertical = _vertical_i1(lat, 256) / exact
    assert abs(vertical) > 500 * abs(err[2])


def _vertical_i1(lat: Lattice, slices: int) -> float:
    """``∮ h_y D_y ds`` alone: the two vertical bends' share of ``I1``, by the same
    transport. Used only to show the vertical half is far outside the tolerance."""
    tw = propagate_twiss(lat, closed_twiss(lat))
    total = 0.0
    for i, e in enumerate(lat.elements):
        if isinstance(e, Dipole) and e.tilt == PI2:
            ds = e.length / slices
            sub = Dipole(ds, e.angle / slices, tilt=PI2).matrix(lat.ref)
            disp = np.array([tw[i].disp_x, tw[i].disp_px, tw[i].disp_y, tw[i].disp_py])
            acc = 0.5 * disp[2]
            for k in range(slices):
                disp = _transverse_4d(sub) @ disp + _dispersive_kick(sub)
                acc += (0.5 if k == slices - 1 else 1.0) * disp[2]
            total += e.curvature * acc * ds  # sin(pi/2) = 1
    return total


# ---------------------------------------------------------------------------
# Gate 5 — the damping bookkeeping closes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ring", [v1.dogleg_ring, lambda: cf_ring(tilt=PI2)], ids=["dogleg", "rolled"]
)
def test_robinson_holds_with_vertical_bending(ring) -> None:
    lat = ring()
    assert sum(damping_partition_numbers(lat)) == pytest.approx(4.0, rel=1e-14, abs=0)
    ri = radiation_integrals(lat)
    cq, g2 = quantum_constant_cq(lat.ref), lat.ref.gamma0**2
    assert equilibrium_vertical_emittance(lat) == pytest.approx(
        cq * g2 * ri.i5y / ((1.0 - ri.i4y / ri.i2) * ri.i2), rel=1e-15, abs=0
    )


def test_the_dogleg_moves_the_vertical_partition_off_one() -> None:
    """``J_y`` is no longer exactly 1 — by ``-I4y/I2 ~ 6e-7`` here, small because the dogleg's
    bends are weak, and with the sign of ``∮ D_y h_y^3``."""
    lat = v1.dogleg_ring()
    ri = radiation_integrals(lat)
    jy = damping_partition_numbers(lat)[1]
    assert jy != 1.0
    assert jy - 1.0 == pytest.approx(-ri.i4y / ri.i2, rel=1e-6, abs=0)
    assert 1e-7 < abs(jy - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Gate 6 — the tracked equilibrium: B3's Lyapunov solve on the radiating tilted ring.
# ---------------------------------------------------------------------------
def _dogleg_with_rf(
    voltage: float, harmonic: int = 10, slices: int = 8, energy: float = 3.0e9
) -> Lattice:
    """The dogleg ring at ``energy``, every bend cut in ``slices`` (B2's lumping), plus RF.

    An electron above transition is stationary at ``phi_s = 0`` (CONVENTIONS ->
    *Synchronous phase branch*); ``pi`` would put it on the unstable root.
    """
    ref = v1._ref() if energy == 3.0e9 else _ref_at(energy)
    lat = _sliced(Lattice(list(v1.dogleg_ring().elements), ref=ref), slices)
    cav = RFCavity.from_harmonic(voltage, harmonic, lat.length, ref, phi_s=0.0)
    return Lattice([*lat.elements, cav], ref=ref)


def _ref_at(energy: float):
    from accsim import ReferenceParticle

    return ReferenceParticle.from_total_energy(v1.ELECTRON_MASS_EV, energy, charge=-1.0)


#: RF "excess" sqrt(V^2 - U0^2) at 3 GeV [V]: three synchrotron tunes, ~0.010 to ~0.024.
#: Scaled with the energy, so every energy runs at the same three Q_s.
_EXCESS = tuple(math.sqrt(v * v - 3.7550e6**2) for v in (3.78e6, 3.9e6, 4.5e6))


def _departure_law(energy: float) -> tuple[float, float, float]:
    """``(U0/E, slope, intercept)`` of the tracked ``eps_y / I5y-form - 1`` against
    ``(2 pi Q_s)^2``, from B3's Lyapunov solve of the radiating tilted ring at three RF
    voltages. A route through the *tracking* map that shares no code with the integrals."""
    from accsim.radiation import energy_loss_per_turn

    base = Lattice(list(v1.dogleg_ring().elements), ref=_ref_at(energy))
    eps, u0 = equilibrium_vertical_emittance(base), energy_loss_per_turn(base)
    xs, ds = [], []
    for k in _EXCESS:
        volt = math.sqrt((k * energy / 3.0e9) ** 2 + u0 * u0)
        sigma, m, _, _ = b3._equilibrium_sigma(_dogleg_with_rf(volt, energy=energy))
        xs.append((2.0 * math.pi * b3._synchrotron_tune(m)) ** 2)
        ds.append(b3._mode_emittances(sigma)[1] / eps - 1.0)
    slope, intercept = np.polyfit(xs, ds, 1)
    assert (
        max(abs(d - (slope * x + intercept)) for x, d in zip(xs, ds, strict=True)) < 2e-5
    )  # it IS a line
    return u0 / energy, float(slope), float(intercept)


def test_the_tracked_equilibrium_lands_on_the_integral_once_both_owners_vanish() -> None:
    """The tracked ``eps_y`` departs from the closed form through two owners, each with its
    own law, and lands on it where both vanish:

    - **the finite synchrotron tune** (B3): ``c (2 pi Q_s)^2``, ``c ~ 0.41`` — the same slope
      MAD-X ``EMIT`` and xtrack's radiation twiss show on this ring — and energy-independent;
    - **the energy sag**: the closed orbit runs ``U0/E`` low-to-high around the ring, and the
      intercept at ``Q_s = 0`` is ``b * U0/E`` with ``b ~ -1`` (``-0.937`` at 8 slices,
      ``-0.970`` at 16, ``-0.987`` at 32: it is the sag, not the lumping, and it converges).
      ``taper()`` removes 77% of it — the sag acting on the optics; the rest is the radiation's
      own ``(1 + delta)`` dependence, not localised further.

    With ``U0/E -> 0`` and ``Q_s -> 0`` the residual is ``+1.0e-5`` / ``-1.4e-5`` / ``-2.0e-5``
    at 8 / 16 / 32 slices: the tracked route and the integral agree to ``2e-5``.
    """
    (s_lo, c_lo, a_lo), (s_hi, c_hi, a_hi) = _departure_law(1.5e9), _departure_law(3.0e9)
    assert s_hi / s_lo == pytest.approx(8.0, rel=1e-6)  # U0/E ~ E^3
    assert c_lo == pytest.approx(c_hi, rel=2e-3)  # the Q_s^2 owner is energy-independent
    assert 0.35 < c_hi < 0.45
    sag = (a_hi - a_lo) / (s_hi - s_lo)
    assert -1.05 < sag < -0.9
    floor = a_lo - sag * s_lo
    assert abs(floor) < 3e-5  # vs 1.2e-3 from the sag and 1.6e-3 from Q_s at the low end


# ---------------------------------------------------------------------------
# Scope — what V2 still refuses, each written to fail the day it lifts.
# ---------------------------------------------------------------------------
def test_a_tilt_that_is_not_a_quarter_turn_couples_and_is_refused() -> None:
    """A sector bend's weak focusing is a horizontal lens; tilted by anything but a multiple
    of ``pi/2`` it is a skew one, so the ring is coupled and the per-plane integrals do not
    exist. ``closed_twiss`` refuses it before the sum runs."""
    lat = Lattice(
        [*v1._cell(), Dipole(0.3, v1.VA, tilt=0.3), *[e for _ in range(3) for e in v1._cell()]],
        ref=v1._ref(),
    )
    with pytest.raises(CoupledLatticeError):
        radiation_integrals(lat)


def test_the_coupling_sharing_model_refuses_vertical_bending() -> None:
    """``equilibrium_emittances_coupled`` shares the *horizontal* excitation between modes and
    would report ``eps_2 = 0`` on the dogleg ring, which has no coupling but a real ``eps_y``."""
    with pytest.raises(NotImplementedError, match="tilt"):
        equilibrium_emittances_coupled(v1.dogleg_ring())


def test_the_polarization_integrals_still_refuse_a_tilted_bend() -> None:
    with pytest.raises(NotImplementedError, match="tilt"):
        polarization_integrals(v1.dogleg_ring())
