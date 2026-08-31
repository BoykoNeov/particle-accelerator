"""Throwaway probe: PTC's cubic rows on a bumped octupole ring."""

from __future__ import annotations

import math

import pytest
from _madx import madx_session

from accsim import (
    Corrector,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    ThinOctupole,
    resonance_driving_terms_on_orbit,
    tunes,
)
from accsim.orbit import closed_orbit_nonlinear

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0
ENERGY_GEV = 10.0
KF, KD = 0.8225, -0.90
OCTS = {5.5: 400.0, 8.4: -280.0, 11.6: 180.0}
KEYS = ("f3000", "f2100", "f1020", "f1011", "f1002")


def _idx(key):
    return tuple(int(c) for c in key[1:])


def _w(key):
    return math.prod(math.factorial(i) for i in _idx(key))


def _madx_ring(kick=0.0, scale=1.0):
    elems = [(0.25 + 3.0 * c, f"qf{c}: qf") for c in range(4)]
    elems += [(1.75 + 3.0 * c, f"qd{c}: qd") for c in range(4)]
    elems += [
        (pos, f"oc{i}: multipole, knl={{0,0,0,{w * scale:.12g}}}")
        for i, (pos, w) in enumerate(OCTS.items())
    ]
    elems += [(0.0, f"kck: hkicker, kick={kick:.12g}")]
    body = "\n".join(f"      {d}, at={pos};" for pos, d in sorted(elems))
    return f"""
    beam, particle=proton, energy={ENERGY_GEV}, sequence=ring;
    qf: quadrupole, l=0.5, k1= {KF};
    qd: quadrupole, l=0.5, k1= {KD};
    ring: sequence, l=12.0, refer=centre;
{body}
    endsequence;
    """


def _accsim_ring(kick=0.0, scale=1.0):
    els: list = [Corrector(kick_x=kick)]
    s = 0.0
    for _ in range(4):
        for k in (KF, KD):
            els.append(Quadrupole(0.5, k))
            s += 0.5
            done = 0.0
            for p in sorted(q for q in OCTS if s < q < s + 1.0):
                els.append(Drift(p - s - done))
                els.append(ThinOctupole(OCTS[p] * scale))
                done = p - s
            els.append(Drift(1.0 - done))
            s += 1.0
    assert sum(isinstance(e, ThinOctupole) for e in els) == len(OCTS)
    return Lattice(els, ReferenceParticle.from_gamma(MASS0, GAMMA0))


def _keys3():
    return [
        (a, b, c, d)
        for a in range(4)
        for b in range(4)
        for c in range(4)
        for d in range(4)
        if a + b + c + d == 3
    ]


def _ptc(sequence, order=4):
    sel = "\n".join(f"      select_ptc_normal, gnfu={a},{b},{c},{d};" for a, b, c, d in _keys3())
    with madx_session() as madx:
        madx.input(sequence)
        madx.use(sequence="ring")
        madx.twiss(sequence="ring")
        madx.input(f"""
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst=5, exact=true;
          select_ptc_normal, q1=0, q2=0;
{sel}
        ptc_normal, closed_orbit, normal, icase=4, no={order};
        ptc_end;
        """)
        t = madx.table.normal_results
        out: dict = {}
        for n, o1, o2, o3, o4, v in zip(
            t.name, t.order1, t.order2, t.order3, t.order4, t.value, strict=True
        ):
            out.setdefault(str(n).strip(), {})[(int(o1), int(o2), int(o3), int(o4))] = float(v)
    return out


def _c(gnf, key):
    return complex(gnf["gnfc"][_idx(key)], gnf["gnfs"][_idx(key)]) / _w(key)


def test_probe():
    KICK = 2.0e-4
    print()
    lat = _accsim_ring(kick=KICK)
    print("accsim tunes", tunes(lat), "orbit x0", closed_orbit_nonlinear(lat)[0])
    with madx_session() as madx:
        madx.input(_madx_ring(kick=KICK))
        madx.use(sequence="ring")
        tw = madx.twiss(sequence="ring")
        print("madx   tunes", float(madx.table.summ.q1[0]), float(madx.table.summ.q2[0]),
              "orbit x0", float(tw.x[0]))
    bumped = _ptc(_madx_ring(kick=KICK))
    bare = _ptc(_madx_ring(kick=KICK, scale=0.0))
    flat = _ptc(_madx_ring(kick=0.0))
    mine = resonance_driving_terms_on_orbit(lat)
    print(f"{'key':7} {'accsim':>28} {'ptc(bump)-ptc(bare)':>30} {'ratio':>22}")
    for k in KEYS:
        d = _c(bumped, k) - _c(bare, k)
        r = d / mine[k] if abs(mine[k]) > 1e-12 else float("nan")
        print(f"{k:7} {mine[k]:>28.10e} {d:>30.10e} {r:>22.8f}")
    print("\nbare (octupoles off, bumped) and flat (octupoles on, no bump):")
    for k in KEYS:
        print(f"  {k:7} bare={_c(bare, k):.6e}  flat={_c(flat, k):.6e}  bumped={_c(bumped, k):.6e}")
    raise AssertionError("probe")
