r"""Analytic gates for I4 with a combined-function gradient and pole-face edges.

The general damping-partition integral is
``I4 = ∮ D_x h (h^2 + 2 k1) ds - Σ_faces D_x h^2 tan(e)``. The body ``2 k1`` term
is pinned here by a **closed-form** case; the edge term (no clean closed form) is
pinned against MAD-X in ``tests/reference/test_radiation_edges_madx.py`` and by the
within-baseline ``I1 == alpha_c * C`` identity, which the edge transport must also
satisfy.

**The closed-form anchor is the smooth constant-gradient ring.** A ring of
identical combined-function bends (constant ``h`` and ``k1``) has an *exact*
periodic dispersion ``D_x = h / K_x``, ``D_x' = 0`` (``K_x = h^2 + k1``) -- it is a
genuine fixed point of the combined-function map (``R21 D_x + R26 = 0`` identically),
not a smooth-limit approximation, so it holds to machine precision for any number
of segments. That makes every integral closed-form:

    I2 = h^2 C,   I4 = h^2 (h^2 + 2 k1) C / K_x,
    I4 / I2 = (h^2 + 2 k1) / K_x,   J_x = 1 - I4/I2 = -k1 / K_x = n/(1-n)

with the field index ``n = -k1 / h^2`` (vertical stability needs ``k1 < 0``, i.e.
``0 < n < 1``). This directly pins the ``2 k1`` coefficient: a wrong ``h^2 + k1``
(coefficient 1 instead of 2) would give ``I4 = I2`` exactly and ``J_x = 0`` for
*every* ``n`` -- so ``I4 != I2`` already refutes it.
"""

from __future__ import annotations

import math

import pytest

from accsim import Dipole, Drift, Lattice, Quadrupole, ReferenceParticle
from accsim.radiation import damping_partition_numbers, radiation_integrals
from accsim.twiss import momentum_compaction

C_RING = 100.0
E_TOTAL = 2.0e9


def _smooth_combined_ring(n: float, segments: int = 60) -> tuple[Lattice, float, float]:
    """A constant-gradient (weak-focusing) ring of ``segments`` identical bends.

    Field index ``n`` sets ``k1 = -n h^2`` (so ``0 < n < 1`` is stable in both
    planes). Returns ``(ring, h, k1)``.
    """
    from accsim import ELECTRON_MASS_EV

    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, E_TOTAL, charge=-1.0)
    h = 2.0 * math.pi / C_RING
    k1 = -n * h * h
    elems = [Dipole(C_RING / segments, 2.0 * math.pi / segments, k1=k1) for _ in range(segments)]
    return Lattice(elems, ref=ref), h, k1


@pytest.mark.parametrize("n", [0.3, 0.6])
def test_smooth_ring_dispersion_is_constant(n: float) -> None:
    """The exact fixed-point dispersion ``D_x = h/K_x``, ``D_x' = 0``."""
    from accsim.twiss import closed_twiss

    ring, h, k1 = _smooth_combined_ring(n)
    tw = closed_twiss(ring)
    assert tw.disp_x == pytest.approx(h / (h * h + k1), rel=1e-10)
    assert tw.disp_px == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [0.3, 0.6])
def test_smooth_ring_I4_closed_form(n: float) -> None:
    """I4 == h^2 (h^2 + 2 k1) C / K_x -- pins the body 2*k1 coefficient exactly."""
    ring, h, k1 = _smooth_combined_ring(n)
    ri = radiation_integrals(ring)
    kx = h * h + k1
    assert ri.i2 == pytest.approx(h * h * C_RING, rel=1e-10)
    assert ri.i4 == pytest.approx(h * h * (h * h + 2.0 * k1) * C_RING / kx, rel=1e-9)


@pytest.mark.parametrize("n", [0.3, 0.6])
def test_smooth_ring_damping_partition_closed_form(n: float) -> None:
    """J_x == n/(1-n); the wrong coefficient-1 map would give I4 == I2 and J_x == 0."""
    ring, h, k1 = _smooth_combined_ring(n)
    ri = radiation_integrals(ring)
    jx, jy, jz = damping_partition_numbers(ring)
    assert jx == pytest.approx(n / (1.0 - n), rel=1e-9)
    # Refute coefficient 1: it forces I4 == I2 exactly (J_x == 0) for every n.
    assert ri.i4 != pytest.approx(ri.i2, rel=1e-3)
    assert abs(jx) > 0.1  # comfortably away from the coefficient-1 value of 0
    assert jx + jy + jz == pytest.approx(4.0, abs=1e-12)  # Robinson still exact


def _electron_fodo(k1_bend: float = 0.0, edge: float = 0.0) -> Lattice:
    """A stable separated-function FODO whose bends may carry a gradient / edges."""
    from accsim import ELECTRON_MASS_EV

    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, E_TOTAL, charge=-1.0)
    lq, kq, lb, th, ld = 0.3, 1.2, 1.0, 0.3927, 0.5
    return Lattice(
        [
            Quadrupole(lq, +kq),
            Drift(ld),
            Dipole(lb, th, k1=k1_bend, e1=edge, e2=edge),
            Drift(ld),
            Quadrupole(lq, -kq),
            Drift(ld),
            Dipole(lb, th, k1=k1_bend, e1=edge, e2=edge),
            Drift(ld),
        ],
        ref=ref,
    )


def test_I2_unchanged_by_gradient_and_edges() -> None:
    """I2 = ∮ h^2 ds is pure geometry -- gradient and edges leave it untouched."""
    base = radiation_integrals(_electron_fodo()).i2
    assert radiation_integrals(_electron_fodo(k1_bend=0.05)).i2 == pytest.approx(base, rel=1e-12)
    assert radiation_integrals(_electron_fodo(edge=0.19635)).i2 == pytest.approx(base, rel=1e-12)


def test_I1_equals_alpha_c_C_with_gradient_and_edges() -> None:
    """The within-baseline identity holds through combined-function + edge transport.

    ``I1 = ∮ D_x h ds`` must equal the exact ``alpha_c * C`` (identity method, from
    the one-turn matrix) -- this validates the dispersion transport that also feeds
    ``I4``/``I5``, independent of any external code.
    """
    for lat in (_electron_fodo(k1_bend=0.05), _electron_fodo(edge=0.19635)):
        ri = radiation_integrals(lat, slices=256)
        assert ri.i1 == pytest.approx(momentum_compaction(lat) * lat.length, rel=1e-5)


def test_robinson_exact_for_combined_and_edge_lattice() -> None:
    """J_x + J_y + J_z == 4 for a lattice with both a gradient and edges."""
    jx, jy, jz = damping_partition_numbers(_electron_fodo(k1_bend=0.05, edge=0.05))
    assert jx + jy + jz == pytest.approx(4.0, abs=1e-12)


def _combined_fodo(k1_bend: float) -> Lattice:
    """A stable combined-function FODO: the bends themselves focus/defocus."""
    from accsim import ELECTRON_MASS_EV

    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, E_TOTAL, charge=-1.0)
    lb, th, ld = 1.0, 0.3, 0.4
    return Lattice(
        [
            Dipole(lb, th, k1=+k1_bend),
            Drift(ld),
            Dipole(lb, th, k1=-k1_bend),
            Drift(ld),
        ],
        ref=ref,
    )


def test_strong_gradient_drives_horizontal_anti_damping() -> None:
    """A strong combined-function gradient can push J_x < 0 -- a sector bend cannot.

    Horizontal anti-damping (J_x < 0, so the horizontal amplitude *grows*) is a
    genuine combined-function signature: the 2*k1 term dominates I4, pushing
    I4/I2 > 1. A pure-sector ring keeps J_x close to +1. The value is trusted
    because I1 == alpha_c * C holds to machine precision on the same lattice.
    """
    lat = _combined_fodo(0.3)
    jx, jy, jz = damping_partition_numbers(lat)
    assert jx < 0.0  # horizontal anti-damping
    assert jx + jy + jz == pytest.approx(4.0, abs=1e-12)  # Robinson still exact
    ri = radiation_integrals(lat, slices=256)
    assert ri.i1 == pytest.approx(momentum_compaction(lat) * lat.length, rel=1e-6)
    # Contrast: a stable pure-sector (separated-function) ring damps normally, J_x > 0.
    jx0, _, _ = damping_partition_numbers(_electron_fodo())
    assert jx0 > 0.2  # positive: horizontal amplitude damps, no sign flip


def test_edges_shift_I4_and_reduce_to_no_edge() -> None:
    """Rectangular edges move I4 (they contribute -D_x h^2 tan(e)); e=0 does not."""
    th = 0.3927
    no_edge = radiation_integrals(_electron_fodo()).i4
    with_edge = radiation_integrals(_electron_fodo(edge=th / 2)).i4
    assert with_edge != pytest.approx(no_edge, rel=1e-3)  # the face term is live
    # e1 = e2 = 0 explicitly reproduces the pure-sector value (regression).
    assert radiation_integrals(_electron_fodo(edge=0.0)).i4 == pytest.approx(no_edge, rel=1e-12)
