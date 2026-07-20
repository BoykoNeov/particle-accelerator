"""Pin the combined-function I4 body coefficient (2*k1) against MAD-X (D3 second reference).

Marked ``reference``: skips when cpymad is absent.

**Why this test is load-bearing.** The smooth-ring analytic gate in
``tests/analytic/test_radiation_combined.py`` asserts ``J_x == n/(1-n)`` -- but that
right-hand side is *derived from the same* ``I4 = ∮ D_x h (h^2 + 2 k1) ds`` formula
the code implements, so it validates the *integration* (via the independently-known
``D_x = h/K_x``) but **not** the physics coefficient: a shared error (say the true
term were ``k1`` or ``3 k1``) would move both sides together and pass. The ``2`` is
the one genuinely new physics constant in this milestone, so it needs an external
anchor -- this test.

MAD-X's integral-method ``synch_4`` is that anchor. For a combined-function ring it
is only ~1% accurate (its ``synch_*`` disagree with MAD-X's *own* ``alfa`` at that
level -- asserted below, so the tolerance is *explained by MAD-X's treatment*, not
padded), but that easily discriminates the coefficient: ``2 k1`` lands ~1% from
``synch_4`` while ``1 k1`` would force ``I4 = I2 = synch_2``, which is ~9% away --
~10x worse. So this passes for the shipped coefficient and fails hard for the
wrong one.
"""

from __future__ import annotations

import pytest
from _madx import madx_session

from accsim import Dipole, Drift, Lattice, Quadrupole, ReferenceParticle
from accsim.radiation import radiation_integrals

pytestmark = pytest.mark.reference

MASS0 = 0.51099895e6  # electron, eV
ENERGY_GEV = 2.0
LQ, KQ, LB, TH, LD, KB = 0.3, 1.2, 1.0, 0.3927, 0.5, 0.05


def _accsim_i4_and_i2() -> tuple[float, float]:
    ref = ReferenceParticle.from_total_energy(MASS0, ENERGY_GEV * 1e9, charge=-1.0)
    ring = Lattice(
        [
            Quadrupole(LQ, +KQ),
            Drift(LD),
            Dipole(LB, TH, k1=KB),
            Drift(LD),
            Quadrupole(LQ, -KQ),
            Drift(LD),
            Dipole(LB, TH, k1=KB),
            Drift(LD),
        ],
        ref=ref,
    )
    ri = radiation_integrals(ring, slices=256)
    return ri.i4, ri.i2


def _madx_synch() -> dict[str, float]:
    circ = 2 * LQ + 4 * LD + 2 * LB
    with madx_session() as m:
        m.input(f"""
            beam, particle=electron, energy={ENERGY_GEV};
            qf: quadrupole, l={LQ}, k1={KQ};
            qd: quadrupole, l={LQ}, k1={-KQ};
            b:  sbend, l={LB}, angle={TH}, k1={KB};
            cell: sequence, l={circ}, refer=entry;
              qf, at=0.0;
              b,  at={LQ + LD};
              qd, at={LQ + 2 * LD + LB};
              b,  at={2 * LQ + 3 * LD + LB};
            endsequence;
            use, sequence=cell;
            twiss, chrom;
        """)
        t = m.table.summ
        return {
            "synch_1": t["synch_1"][0],
            "synch_2": t["synch_2"][0],
            "synch_4": t["synch_4"][0],
            "alfa_C": t["alfa"][0] * circ,
        }


def test_combined_I4_coefficient_matches_madx() -> None:
    i4, i2 = _accsim_i4_and_i2()
    md = _madx_synch()

    # accsim's 2*k1 I4 matches MAD-X's integral-method synch_4 to ~1%.
    assert i4 == pytest.approx(md["synch_4"], rel=1.5e-2)

    # The tolerance is MAD-X's, not padding: for a combined-function ring MAD-X's own
    # synch_1 disagrees with its alfa*C (= I1 by definition) at ~0.8%.
    madx_self_inconsistency = abs(md["synch_1"] / md["alfa_C"] - 1.0)
    assert 1e-3 < madx_self_inconsistency < 2e-2

    # Decisive discrimination: the wrong coefficient 1 would force I4 == I2 == synch_2,
    # which is ~9% from synch_4 -- an order of magnitude outside the tolerance above.
    assert abs(i4 / md["synch_4"] - 1.0) < 2e-2  # shipped coefficient (2): ~0.8%
    assert abs(md["synch_2"] / md["synch_4"] - 1.0) > 5e-2  # coefficient 1 (I4=I2): ~9%
    assert i2 == pytest.approx(md["synch_2"], rel=1e-6)  # (i2 really is synch_2)
