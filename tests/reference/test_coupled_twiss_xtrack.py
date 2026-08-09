"""Cross-check the Edwards-Teng coupled Twiss (G2) against xtrack's coupled 4D Twiss.

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed) when
xtrack's JIT C-kernel compilation is unavailable (see ``conftest.py`` / ``_xtrack_jit``).

**What is compared.** accsim returns the Edwards-Teng normal-mode parameters
``(gamma_c, C, beta_1, beta_2)``; xtrack reports **Ripken** betas ``betx1, betx2,
bety1, bety2`` — the contribution of each mode to each plane's beam size per unit
mode emittance. They are the same information in different coordinates. Writing the
sigma matrix ``Sigma = V diag(e1 B1, e2 B2) V^T`` with
``V = [[gamma_c I, C], [-adj(C), gamma_c I]]`` and reading off the ``xx``/``yy``
entries gives the dictionary

    betx1 = gamma_c^2 beta_1            bety2 = gamma_c^2 beta_2
    betx2 = (C B_2 C^T)_00              bety1 = (adj(C) B_1 adj(C)^T)_00

so this test validates ``gamma_c`` and the **whole coupling matrix** ``C`` — not just
the mode betas — against an independent implementation. ``betx2``/``bety1`` are
second order in the coupling and vanish uncoupled, so they are the sharpest check on
``C``.

**The one honest disagreement, and why it is not an error.** xtrack's
``Quadrupole(k1s)`` is a *first-order-in-k1s* model (its diagonal block is a pure
drift), while accsim's ``SkewQuadrupole`` is the exact 45-degree roll that MAD-X
reproduces to ~2e-16 — the model gap documented for G1. The residual it leaves here
must therefore scale as ``k1s^2``, and
:func:`test_model_gap_scales_quadratically_in_k1s` asserts exactly that: a 4x
stronger skew multiplies the deviation by ~16. A genuine error in the decomposition
would not obey that law.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    CoupledTwiss,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    SkewQuadrupole,
    coupled_twiss,
    normal_mode_tunes,
    propagate_coupled_twiss,
)
from accsim.twiss import _adj2

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
GAMMA0 = 5.0
KQ, N_CELLS, LDRIFT, LQ = 1.2, 4, 0.7, 0.3
L_SKEW = 0.2
L_SPLIT, K_SPLIT = 0.01, 5.0  # a weak quad that pushes the ring off the difference resonance


def _build(k1s: float):
    """The same off-resonance FODO ring with one thick skew quad, in both codes."""
    acc, xtk = [], []
    for _ in range(N_CELLS):
        acc += [Quadrupole(LQ, KQ), Drift(LDRIFT), Quadrupole(LQ, -KQ), Drift(LDRIFT)]
        xtk += [
            xt.Quadrupole(length=LQ, k1=KQ),
            xt.Drift(length=LDRIFT),
            xt.Quadrupole(length=LQ, k1=-KQ),
            xt.Drift(length=LDRIFT),
        ]
    mid = len(acc) // 2
    acc = acc[:mid] + [SkewQuadrupole(L_SKEW, k1s)] + acc[mid:] + [Quadrupole(L_SPLIT, K_SPLIT)]
    xtk = (
        xtk[:mid]
        + [xt.Quadrupole(length=L_SKEW, k1s=k1s)]
        + xtk[mid:]
        + [xt.Quadrupole(length=L_SPLIT, k1=K_SPLIT)]
    )
    return acc, xtk


def _twiss_xtrack(xtk):
    line = xt.Line(elements=xtk)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=-1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line.twiss(method="4d")


def _ripken(ct: CoupledTwiss) -> tuple[float, float, float, float]:
    """``(betx1, betx2, bety1, bety2)`` from the Edwards-Teng parameters."""
    b1 = np.array([[ct.beta_1, -ct.alpha_1], [-ct.alpha_1, ct.gamma_1]])
    b2 = np.array([[ct.beta_2, -ct.alpha_2], [-ct.alpha_2, ct.gamma_2]])
    C, cbar = ct.c_matrix, _adj2(ct.c_matrix)
    g2 = ct.gamma_c**2
    return (
        g2 * ct.beta_1,
        float((C @ b2 @ C.T)[0, 0]),
        float((cbar @ b1 @ cbar.T)[0, 0]),
        g2 * ct.beta_2,
    )


def test_ripken_betas_match_xtrack() -> None:
    """All four Ripken betas agree with xtrack — so gamma_c and C are both right."""
    k1s = 0.02
    acc, xtk = _build(k1s)
    lat = Lattice(acc, ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0))
    tw = _twiss_xtrack(xtk)

    bx1, bx2, by1, by2 = _ripken(coupled_twiss(lat))
    # the mode betas: limited only by xtrack's O(k1s^2) skew model
    assert bx1 == pytest.approx(tw.betx1[0], rel=1e-5)
    assert by2 == pytest.approx(tw.bety2[0], rel=1e-5)
    # the cross terms: pure coupling, O(|C|^2), zero for an uncoupled ring
    assert bx2 == pytest.approx(tw.betx2[0], rel=5e-4)
    assert by1 == pytest.approx(tw.bety1[0], rel=5e-4)
    assert bx2 > 0.0 and by1 > 0.0

    # and the mode tunes, from the eigen route (independent of the ET decomposition)
    q1, q2 = normal_mode_tunes(lat)
    assert q1 == pytest.approx(tw.qx % 1.0, abs=1e-5)
    assert q2 == pytest.approx(tw.qy % 1.0, abs=1e-5)


def test_ripken_betas_match_around_the_whole_ring() -> None:
    """Not just at the start: every element boundary agrees, so propagation is pinned."""
    k1s = 0.02
    acc, xtk = _build(k1s)
    lat = Lattice(acc, ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0))
    tw = _twiss_xtrack(xtk)
    pts = propagate_coupled_twiss(lat)
    assert len(pts) - 1 == len(tw.betx1) - 1  # accsim adds the ring-end point
    worst_mode = worst_cross = 0.0
    for i, ct in enumerate(pts[:-1]):
        bx1, bx2, by1, by2 = _ripken(ct)
        worst_mode = max(worst_mode, abs(bx1 / tw.betx1[i] - 1.0), abs(by2 / tw.bety2[i] - 1.0))
        worst_cross = max(worst_cross, abs(bx2 / tw.betx2[i] - 1.0), abs(by1 / tw.bety1[i] - 1.0))
    assert worst_mode < 1e-5
    assert worst_cross < 5e-4


def test_model_gap_scales_quadratically_in_k1s() -> None:
    """The residual vs xtrack is its first-order skew model, not an accsim error.

    xtrack drops the ``O(k1s^2)`` focusing that the exact roll keeps (MAD-X agrees with
    accsim to ~2e-16 on the element matrix), so the ``betx1`` deviation must grow like
    ``k1s^2``: quadrupling ``k1s`` must multiply it by ~16. A wrong prefactor or a sign
    slip in the decomposition would leave a residual that does *not* follow that law.
    """
    devs = {}
    for k1s in (0.005, 0.02):
        acc, xtk = _build(k1s)
        lat = Lattice(acc, ReferenceParticle.from_gamma(MASS0, GAMMA0, charge=-1.0))
        tw = _twiss_xtrack(xtk)
        bx1, _bx2, _by1, _by2 = _ripken(coupled_twiss(lat))
        devs[k1s] = abs(bx1 / tw.betx1[0] - 1.0)
    assert devs[0.005] < 1e-6  # already negligible at weak coupling
    assert devs[0.02] / devs[0.005] == pytest.approx(16.0, rel=0.1)
