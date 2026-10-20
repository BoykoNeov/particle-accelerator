"""Momentum compaction, slip factor and ``Q_s`` on a design-coupled ring.

Change 1 of *coupled optics in the editor's main view* (``docs/ROADMAP.md``). All three
used to reach :func:`~accsim.twiss.closed_twiss` and so refused any ring whose one-turn
map couples ``x`` and ``y``. The dispersion they need is the 4D matched solve
``D = (I - M4)^-1 d``, which does not care about coupling, so they now accept such a
ring; these are the pre-committed gates that say the number is right.

The fixtures are six thin-lens FODO cells (``test_momentum_compaction.py``'s arc cell)
with a thin quadrupole that splits the tunes, coupled either by one thin skew
quadrupole placed where the horizontal dispersion is large (it leaks into ``D_y``), or
by a solenoid. The xtrack cross-check of the same ring is
``tests/reference/test_coupled_compaction_xtrack.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    DELTA,
    PX,
    PY,
    ZETA,
    Dipole,
    Lattice,
    ReferenceParticle,
    RFCavity,
    Solenoid,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    UnstableLatticeError,
    X,
    Y,
    closed_twiss,
    momentum_compaction,
    normal_form,
    slip_factor,
    synchrotron_tune,
    transition_gamma,
)

MASS0 = 938.27208816e6
GAMMA0 = 5.0
F_FOCAL = 2.5
L_BEND, ANGLE = 1.0, 0.15
N_CELLS = 6
SPLIT = -0.03  # thin quadrupole: moves the ring off Q_x + Q_y = 1 and off Q_x = Q_y
K1SL = 0.05


def _ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


def _cell() -> list:
    return [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]


def _elements(k1sl: float = K1SL, n_cells: int = N_CELLS) -> list:
    """The ring; the skew quadrupole sits after the first bend of cell 2 (D_x = 2.66)."""
    out: list = []
    for i in range(n_cells):
        cell = _cell()
        if i == 1:
            cell = cell[:2] + [ThinSkewQuadrupole(k1sl)] + cell[2:]
        out += cell
    return [*out, ThinQuadrupole(SPLIT)]


def skew_ring(k1sl: float = K1SL, n_cells: int = N_CELLS) -> Lattice:
    return Lattice(_elements(k1sl, n_cells), _ref())


def solenoid_ring() -> Lattice:
    """The same ring with no skew quadrupole and a solenoid before element 8."""
    els = _elements(0.0)
    return Lattice([*els[:7], Solenoid(0.5, 0.2), *els[7:]], _ref())


def cavity_ring(voltage: float) -> Lattice:
    """The skew ring with one cavity; ``eta > 0`` (above transition), so ``phi_s = pi``."""
    lat = skew_ring()
    cav = RFCavity.from_harmonic(voltage, 10, lat.length, lat.ref, phi_s=math.pi)
    return Lattice([*lat.elements, cav], lat.ref)


def _dispersion(lat: Lattice) -> np.ndarray:
    """The 4D matched dispersion, solved here rather than read from the package."""
    M = lat.one_turn_matrix()
    m4 = M[:4, :4]
    return np.linalg.solve(np.eye(4) - m4, M[:4, DELTA])


def _vertical_share(lat: Lattice) -> float:
    """The vertical pair's part of alpha_c: ``-(R53 D_y + R54 D_py) / C``."""
    M, D = lat.one_turn_matrix(), _dispersion(lat)
    return -(M[ZETA, Y] * D[2] + M[ZETA, PY] * D[3]) / lat.length


FIXTURES = {"skew": skew_ring, "solenoid": solenoid_ring}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_fixtures_are_coupled_and_stable(name: str) -> None:
    """Guard: both rings really are coupled (closed_twiss refuses them) and stable."""
    lat = FIXTURES[name]()
    M = lat.one_turn_matrix()
    off = max(np.abs(M[:2, 2:4]).max(), np.abs(M[2:4, :2]).max())
    assert off > 0.1
    with pytest.raises(ValueError, match="coupled"):
        closed_twiss(lat)
    nf = normal_form(M, method="4d")
    assert abs(nf.tunes[0] - nf.tunes[1]) > 0.05


# ---------------------------------------------------------------------- gate 1
@pytest.mark.parametrize("name", list(FIXTURES))
def test_identity_and_quadrature_agree_on_a_coupled_ring(name: str) -> None:
    """Gate 1: the two disjoint routes meet, and the quadrature's miss is 1/n^2."""
    lat = FIXTURES[name]()
    exact = momentum_compaction(lat)
    miss = [momentum_compaction(lat, method="quadrature", slices=n) - exact for n in (32, 64, 128)]
    assert 3.9 < miss[0] / miss[1] < 4.1
    assert 3.9 < miss[1] / miss[2] < 4.1
    assert abs(miss[2]) < 1e-7


# ---------------------------------------------------------------------- gate 2
@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_vertical_pair_is_load_bearing(name: str) -> None:
    """Gate 2: leaving R53 D_y + R54 D_py out would miss gate 1 by orders of magnitude."""
    lat = FIXTURES[name]()
    floor = abs(
        momentum_compaction(lat, method="quadrature", slices=128) - momentum_compaction(lat)
    )
    assert abs(_vertical_share(lat)) > 1e3 * floor


def test_the_vertical_share_matches_the_measured_size() -> None:
    """Gate 2's numbers, so a fixture change that shrinks the pair is noticed."""
    assert _vertical_share(skew_ring()) == pytest.approx(1.711e-3, rel=1e-3)
    assert _vertical_share(solenoid_ring()) == pytest.approx(4.638e-4, rel=1e-3)


# ---------------------------------------------------------------------- gate 3
def test_alpha_c_is_second_order_in_the_skew_strength() -> None:
    """Gate 3: the coupled branch meets the uncoupled ring continuously, at k1sl^2."""
    base = momentum_compaction(skew_ring(0.0))
    coef = [(momentum_compaction(skew_ring(k)) - base) / k**2 for k in (1e-2, 1e-4)]
    assert coef[0] == pytest.approx(0.44008, rel=1e-4)
    assert coef[1] == pytest.approx(coef[0], rel=1e-4)


def test_the_uncoupled_ring_keeps_its_old_number_bit_for_bit() -> None:
    """Gate 3: k1sl = 0 is uncoupled and goes through closed_twiss exactly as before."""
    lat = skew_ring(0.0)
    M, tw = lat.one_turn_matrix(), closed_twiss(lat)
    slip = M[ZETA, X] * tw.disp_x + M[ZETA, PX] * tw.disp_px + M[ZETA, DELTA]
    slip += M[ZETA, Y] * tw.disp_y + M[ZETA, PY] * tw.disp_py
    assert momentum_compaction(lat) == 1.0 / lat.ref.gamma0**2 - slip / lat.length


def test_slip_factor_and_transition_gamma_follow() -> None:
    """eta and gamma_t are alpha_c's consumers; they accept the coupled ring with it."""
    lat = skew_ring()
    a = momentum_compaction(lat)
    assert slip_factor(lat) == pytest.approx(a - 1.0 / GAMMA0**2, rel=1e-14)
    assert transition_gamma(lat) == pytest.approx(1.0 / math.sqrt(a), rel=1e-14)


# ---------------------------------------------------------------------- gate 5
def test_synchrotron_tune_meets_the_6d_eigen_tune_at_the_coupling_order() -> None:
    """Gate 5: the lumped Q_s on the coupled eta misses the 6D tune as Q_s^2, no floor."""
    rel = []
    for v in (2e5, 5e4, 1.25e4):
        lat = cavity_ring(v)
        q3 = normal_form(lat.one_turn_matrix(), method="6d").tunes[2]
        q3 = min(q3, 1.0 - q3)
        rel.append(synchrotron_tune(lat) / q3 - 1.0)
    assert rel[0] == pytest.approx(-1.185e-5, rel=1e-2)
    assert 3.9 < rel[0] / rel[1] < 4.1
    assert 3.9 < rel[1] / rel[2] < 4.1


# ---------------------------------------------------------------------- gate 6
def test_an_unstable_coupled_ring_is_refused() -> None:
    """Gate 6: seven cells sit on the sum resonance; the coupled branch must not solve D."""
    lat = skew_ring(n_cells=7)
    with pytest.raises(UnstableLatticeError):
        momentum_compaction(lat)
    with pytest.raises(UnstableLatticeError):
        momentum_compaction(lat, method="quadrature")


def test_quadrature_still_refuses_a_rolled_bend() -> None:
    """Gate 6: the rolled bend's refusal is unchanged by the coupled branch."""
    els = _elements(0.0)
    els[1] = Dipole(L_BEND, ANGLE, roll=0.01)
    with pytest.raises(NotImplementedError, match="roll"):
        momentum_compaction(Lattice(els, _ref()), method="quadrature")
