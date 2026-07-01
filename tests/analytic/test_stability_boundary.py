"""Stage 2 acceptance: the stability boundary ``|Tr M| < 2`` == the analytic
phase-advance limit.

For the symmetric thin-lens FODO cell (full-quad focal length ``f``, half-cell
drift ``L``)

    QF/2(f_lens=2f) - Drift(L) - QD(f_lens=-f) - Drift(L) - QF/2(f_lens=2f)

the one-turn x-block has ``cos mu = 1 - L^2/(2 f^2)`` (verified symbolically
below). The cell is stable iff ``|1/2 Tr M| < 1``, i.e. ``cos mu > -1``, i.e.
``f > L/2``. The single *reachable* boundary is the over-focusing edge
``f_crit = L/2``, where ``cos mu = -1`` and the phase advance per cell reaches its
analytic limit ``mu = pi``. (The ``cos mu = +1`` edge is merely the no-focusing
``f -> inf`` limit, not an instability, so a symmetric FODO has one boundary, not
two.) Both transverse planes hit the boundary together because the symmetric cell
has ``mu_x = mu_y``.

Anti-circularity: ``is_stable`` *is* ``|1/2 Tr| < 1``, so a check of ``is_stable``
against a re-computed ``1/2 Tr`` from the same matrix would be tautological. The
independent anchor is instead the **symbolically-derived** critical focal length
``f_crit`` (solving ``Tr M = -2`` in sympy, touching no accsim code); the accsim
thin-kick + drift chain must then reproduce it. The phase-advance side is
cross-checked through :func:`tunes`, whose ``atan2`` continuous-accumulation path
is independent of the raw ``1/2 Tr``.
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    UnstableLatticeError,
    closed_twiss,
    is_stable,
    tunes,
)
from accsim.coords import PX, PY, X, Y

L_HALF = 1.0  # half-cell drift [m]


def _thin_fodo(f: float, ll: float = L_HALF) -> list:
    """Symmetric thin FODO, focusing quads split half-and-half at the ends."""
    return [
        ThinQuadrupole(0.5 / f, name="qf/2"),
        Drift(ll, name="d1"),
        ThinQuadrupole(-1.0 / f, name="qd"),
        Drift(ll, name="d2"),
        ThinQuadrupole(0.5 / f, name="qf/2"),
    ]


@pytest.fixture(scope="module")
def f_crit() -> float:
    """Critical focal length from ``Tr(one-turn x-block) = -2``, solved in sympy.

    Independent of accsim: builds the thin 2x2 matrices by hand and solves the
    stability-boundary equation symbolically. Also pins the closed form
    ``cos mu = 1 - L^2/(2 f^2)`` so the boundary is anchored to derived physics,
    not a remembered coefficient.
    """
    sp = pytest.importorskip("sympy")
    f, ll = sp.symbols("f L", positive=True)
    qfh = sp.Matrix([[1, 0], [-1 / (2 * f), 1]])  # half-F, focusing in x
    qd = sp.Matrix([[1, 0], [1 / f, 1]])  # full-D, defocusing in x
    drift = sp.Matrix([[1, ll], [0, 1]])
    m = qfh * drift * qd * drift * qfh  # one-turn x-block from the F centre
    # Closed form of the half-trace (cos mu), re-derived, not assumed.
    assert sp.simplify(m.trace() / 2 - (1 - ll**2 / (2 * f**2))) == 0
    # Tr = -2 (cos mu = -1, mu = pi) is the over-focusing stability boundary.
    roots = [r for r in sp.solve(sp.Eq(m.trace(), -2), f) if r.subs(ll, L_HALF) > 0]
    assert len(roots) == 1
    return float(roots[0].subs(ll, L_HALF))


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts: optics is energy-independent, so any reference works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _half_traces(lat: Lattice) -> tuple[float, float]:
    """``(1/2 Tr x-block, 1/2 Tr y-block)`` of the one-turn map."""
    m = lat.one_turn_matrix()
    return 0.5 * (m[X, X] + m[PX, PX]), 0.5 * (m[Y, Y] + m[PY, PY])


def test_critical_focal_length_is_half_the_drift(f_crit: float) -> None:
    # The symbolic boundary sits at f_crit = L/2 (the classic FODO result).
    assert f_crit == pytest.approx(L_HALF / 2.0)


def test_trace_hits_minus_two_at_the_symbolic_critical_strength(
    ref: ReferenceParticle, f_crit: float
) -> None:
    # The accsim thin-kick + drift chain must land 1/2 Tr on -1 (|Tr| = 2) at the
    # independently-derived f_crit -- and in BOTH planes together, since the
    # symmetric cell has mu_x = mu_y. This is the non-trivial content: that
    # composing the element maps reproduces the predicted boundary strength.
    hx, hy = _half_traces(Lattice(_thin_fodo(f_crit), ref))
    assert hx == pytest.approx(-1.0)
    assert hy == pytest.approx(-1.0)


def test_is_stable_flips_across_the_boundary(ref: ReferenceParticle, f_crit: float) -> None:
    stable = Lattice(_thin_fodo(f_crit * 1.01), ref)  # f > L/2: under-focused
    unstable = Lattice(_thin_fodo(f_crit * 0.99), ref)  # f < L/2: over-focused

    assert is_stable(stable.one_turn_matrix())
    assert not is_stable(unstable.one_turn_matrix())

    # Beyond the boundary there is no real matched beta -- matching must refuse
    # rather than return a complex Twiss.
    closed_twiss(stable)  # does not raise
    with pytest.raises(UnstableLatticeError):
        closed_twiss(unstable)


def test_stability_region_matches_analytic_phase_limit(
    ref: ReferenceParticle, f_crit: float
) -> None:
    # Sweep the focal length across the boundary; the code's |1/2 Tr| < 1 gate
    # must agree, at every point, with the analytic phase-advance criterion
    # sin(mu/2) = L/(2 f) < 1  <=>  mu < pi  <=>  f > L/2. Two independent
    # definitions of "stable" (matrix trace vs. hand phase formula) coincide.
    for f in (0.30, 0.40, 0.49, 0.51, 0.60, 1.00, 2.00):
        analytic_stable = (L_HALF / (2.0 * f)) < 1.0  # sin(mu/2) < 1
        code_stable = is_stable(Lattice(_thin_fodo(f), ref).one_turn_matrix())
        assert code_stable == analytic_stable, f"disagreement at f={f}"
        assert analytic_stable == (f > f_crit)


def test_boundary_is_the_phase_advance_limit_pi(ref: ReferenceParticle, f_crit: float) -> None:
    # The independent atan2 phase-accumulation path (tunes) must send the per-cell
    # phase advance to its analytic limit mu = pi (Q = 1/2) as f -> f_crit+, i.e.
    # the trace-space boundary |Tr| = 2 and the phase-space limit mu = pi are the
    # same boundary, reached by two different code routes.
    #
    # Build a cell at a target mu just inside the boundary via sin(mu/2) = L/(2f).
    # Stay off the boundary itself (beta_max ~ 1/sin mu diverges there) -- mu =
    # 0.9 pi is ample. NB: f = L/(2 sin(mu/2)) maps mu and 2*pi - mu to the SAME
    # f, so this parametrisation can only reach the stable range (0, pi); the
    # unstable side is reached by lowering f below f_crit (test above), never by
    # pushing this target past pi.
    mu_target = 0.9 * math.pi
    f = L_HALF / (2.0 * math.sin(mu_target / 2.0))
    assert f > f_crit  # a target mu < pi must land on the stable side
    qx, qy = tunes(Lattice(_thin_fodo(f), ref))
    assert qx == pytest.approx(mu_target / (2.0 * math.pi))
    assert qy == pytest.approx(mu_target / (2.0 * math.pi))
