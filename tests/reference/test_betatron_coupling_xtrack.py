"""Cross-check betatron coupling (skew quad + normal-mode tunes + |C^-|) vs xtrack.

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed) when
xtrack's JIT C-kernel compilation is unavailable (the clang-cl fix-up in
``conftest.py`` / ``_xtrack_jit`` makes it build on this Windows toolchain).

Two things are pinned against the reference tracker that the analytic suite cannot
self-check:

  * the **sign/magnitude** of the skew gradient ``k1s`` (accsim's ``SkewQuadrupole``
    vs xtrack's ``Quadrupole(k1s=...)``), the empirical anchor the roll-identity
    analytic gate deliberately leaves open;
  * the coupled **normal-mode tunes** and the **closest tune approach** ``|C^-|``,
    against xtrack's coupled 4D Twiss on a symmetric (on-resonance) FODO ring.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    closest_tune_approach,
    normal_mode_tunes,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

_T = [0, 1, 2, 3]  # transverse (x, px, y, py)
MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 5.0


def _skip_if_no_jit(line: xt.Line, ref: xt.Particles):
    try:
        line.build_tracker()
        return line
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_skew_quadrupole_coupling_matches_xtrack_sign() -> None:
    """The x-y **coupling** entries match xtrack (sign + first-order magnitude).

    Honest, localised disagreement (D3-style): accsim's ``SkewQuadrupole`` is the
    *exact* hard-edge 45-deg roll of a normal quad -- its diagonal blocks carry the
    ``(F+D)/2`` focusing (order ``k1s^2``), and MAD-X reproduces the whole 4x4 to
    ~2e-16 (see ``test_betatron_coupling_madx.py``). xtrack's ``Quadrupole(k1s)`` is
    a **first-order-in-k1s** model: its diagonal is a pure drift and only the linear
    coupling is kept. So against xtrack we pin the coupling entries (the sign anchor
    the analytic roll-identity gate leaves open) and *document* the model gap, rather
    than loosen a tolerance over it.
    """
    L, k1s = 0.5, 0.3
    ref_xt = xt.Particles(mass0=MASS0, q0=-1, gamma0=GAMMA0)
    line = xt.Line(elements=[xt.Quadrupole(length=L, k1s=k1s)])
    line.particle_ref = ref_xt
    _skip_if_no_jit(line, ref_xt)
    R_xt = np.asarray(line.get_R_matrix(particle_on_co=ref_xt.copy())["R_matrix"])

    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0)
    R_us = SkewQuadrupole(L, k1s).matrix(ref)

    # the direct momentum-kick coupling R[px,y] = R[py,x] = k1s*L is linear in k1s
    # and matches to ~1e-5 in both codes -- this is the sign/magnitude anchor.
    assert R_us[1, 2] == pytest.approx(R_xt[1, 2], rel=1e-4)  # R[px,y]
    assert R_us[3, 0] == pytest.approx(R_xt[3, 0], rel=1e-4)  # R[py,x]
    assert R_us[1, 2] == pytest.approx(k1s * L, rel=1e-3) and R_us[1, 2] * R_xt[1, 2] > 0.0
    # the documented model gap: xtrack's diagonal is drift-like, accsim's is not,
    # and the position-coupling R[x,py] differs at O(k1s L^2) for the same reason.
    assert R_xt[1, 0] == pytest.approx(0.0, abs=1e-9)  # xtrack R[px,x] ~ 0 (drift diag)
    assert abs(R_us[1, 0]) > 1e-4  # accsim keeps the O(k1s^2) focusing (MAD-X confirms)


def _fodo_elements(kq: float, n_cells: int, ldrift: float, lq: float):
    """Return (accsim_elements, xtrack_elements) for a symmetric FODO ring (Qx = Qy)."""
    acc, xtk = [], []
    for _ in range(n_cells):
        acc += [Quadrupole(lq, kq), Drift(ldrift), Quadrupole(lq, -kq), Drift(ldrift)]
        xtk += [
            xt.Quadrupole(length=lq, k1=kq),
            xt.Drift(length=ldrift),
            xt.Quadrupole(length=lq, k1=-kq),
            xt.Drift(length=ldrift),
        ]
    return acc, xtk


def test_coupled_mode_tunes_and_cminus_match_xtrack() -> None:
    """Symmetric FODO (Qx=Qy) + a skew: normal-mode tunes and |C^-| vs xtrack 4D Twiss."""
    kq, n_cells, ldrift, lq = 1.2, 4, 0.7, 0.3
    L_sk, k1s = 0.2, 0.02
    acc, xtk = _fodo_elements(kq, n_cells, ldrift, lq)
    # insert the same thick skew quad at the midpoint of both rings
    mid = len(acc) // 2
    mid_xt = len(xtk) // 2
    acc = acc[:mid] + [SkewQuadrupole(L_sk, k1s)] + acc[mid:]
    xtk = xtk[:mid_xt] + [xt.Quadrupole(length=L_sk, k1s=k1s)] + xtk[mid_xt:]

    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0)
    lat = Lattice(acc, ref)

    ref_xt = xt.Particles(mass0=MASS0, q0=-1, gamma0=GAMMA0)
    line = xt.Line(elements=xtk)
    line.particle_ref = ref_xt
    _skip_if_no_jit(line, ref_xt)
    tw = line.twiss(method="4d")

    q1, q2 = normal_mode_tunes(lat)
    # xtrack reports the two mode tunes as qx, qy (fractional) on a coupled 4D twiss
    qx_xt, qy_xt = tw.qx % 1.0, tw.qy % 1.0
    got = sorted([q1, q2])
    exp = sorted([qx_xt, qy_xt])
    np.testing.assert_allclose(got, exp, atol=1e-4)

    # on resonance (symmetric FODO), the mode-tune gap == |C^-|
    cmin = closest_tune_approach(lat)
    gap_xt = abs(qx_xt - qy_xt)
    assert gap_xt == pytest.approx(cmin, rel=3e-2)
