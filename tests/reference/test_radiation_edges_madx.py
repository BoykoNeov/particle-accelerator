"""Cross-check the I4 pole-face (edge) term against MAD-X's synch_4 (D3 second reference).

Marked ``reference``: skips when cpymad is absent.

``I4`` gains a ``-D_x h^2 tan(e)`` contribution at each dipole pole face. There is
no clean closed form for it (``D_x`` at the face is a global lattice quantity), so
this is the pinning gate for the edge term's sign and coefficient, against MAD-X's
own integral-method ``synch_4``.

**Why MAD-X is a valid anchor here (but not for the combined-function body term).**
For a **pure-sector-with-edges** ring MAD-X is self-consistent: its ``synch_1``
equals its own ``alfa * C`` (``I1`` by definition) to machine precision, so its
synch integrals are the trustworthy integral-method values. For a
**combined-function** ring MAD-X's ``synch_*`` instead disagree with its own
``alfa`` at the ~1% level (a MAD-X treatment characteristic, like xtrack's
damped-map eigenanalysis) -- so the ``2 k1`` body coefficient is pinned by the
closed-form smooth-ring gate in ``tests/analytic/test_radiation_combined.py``, not
here. This test asserts MAD-X's self-consistency explicitly, so the anchor's
validity is checked rather than assumed.
"""

from __future__ import annotations

import pytest
from _madx import madx_session

from accsim import Dipole, Drift, Lattice, Quadrupole, ReferenceParticle
from accsim.radiation import radiation_integrals
from accsim.twiss import momentum_compaction

pytestmark = pytest.mark.reference

MASS0 = 0.51099895e6  # electron, eV
ENERGY_GEV = 2.0
LQ, KQ, LB, TH, LD = 0.3, 1.2, 1.0, 0.3927, 0.5


def _accsim_ring(edge: float) -> Lattice:
    ref = ReferenceParticle.from_total_energy(MASS0, ENERGY_GEV * 1e9, charge=-1.0)
    return Lattice(
        [
            Quadrupole(LQ, +KQ),
            Drift(LD),
            Dipole(LB, TH, e1=edge, e2=edge),
            Drift(LD),
            Quadrupole(LQ, -KQ),
            Drift(LD),
            Dipole(LB, TH, e1=edge, e2=edge),
            Drift(LD),
        ],
        ref=ref,
    )


def _madx_synch(edge: float) -> dict[str, float]:
    circ = 2 * LQ + 4 * LD + 2 * LB
    with madx_session() as m:
        m.input(f"""
            beam, particle=electron, energy={ENERGY_GEV};
            qf: quadrupole, l={LQ}, k1={KQ};
            qd: quadrupole, l={LQ}, k1={-KQ};
            b:  sbend, l={LB}, angle={TH}, e1={edge}, e2={edge};
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
            "synch_4": t["synch_4"][0],
            "alfa_C": t["alfa"][0] * circ,
        }


def test_edge_term_matches_madx_synch4() -> None:
    edge = TH / 2  # rectangular bend: e1 = e2 = theta/2
    ri = radiation_integrals(_accsim_ring(edge), slices=256)
    md = _madx_synch(edge)

    # MAD-X must be self-consistent here (its synch integrals are integral-method,
    # not eigenanalysis) for the anchor to be valid.
    assert md["synch_1"] == pytest.approx(md["alfa_C"], rel=1e-9)

    # The edge term: accsim's I4 matches MAD-X's synch_4 (both carry -D_x h^2 tan e).
    assert ri.i4 == pytest.approx(md["synch_4"], rel=2e-3, abs=1e-6)


def test_edge_term_has_teeth() -> None:
    """The face term is what makes I4 negative here; without it I4 is positive.

    A flipped edge sign (or a dropped face term) would leave I4 near the
    pure-sector value, so the sign flip against MAD-X's negative synch_4 is the
    discriminating check.
    """
    edge = TH / 2
    i4_edge = radiation_integrals(_accsim_ring(edge), slices=256).i4
    i4_sector = radiation_integrals(_accsim_ring(0.0), slices=256).i4
    assert i4_sector > 0.0  # pure sector: I4 = ∮ D_x h^3 ds > 0
    assert i4_edge < 0.0  # rectangular edges drive it negative
    # accsim I1 still equals the exact alpha_c * C through the edge transport.
    ri = radiation_integrals(_accsim_ring(edge), slices=256)
    lat = _accsim_ring(edge)
    assert ri.i1 == pytest.approx(momentum_compaction(lat) * lat.length, rel=1e-5)
