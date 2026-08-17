"""Cross-check the Drift transfer matrix against xtrack (Xsuite core tracker).

Marked ``reference``: skipped when xtrack is not installed, and skipped (not
failed) when xtrack's just-in-time C-kernel compilation is unavailable. On this
Windows toolchain the JIT compile is enabled by the ``_xtrack_jit`` fix-up
(applied in ``tests/reference/conftest.py``), which routes the build through
clang-cl — see ``docs/CONVENTIONS.md`` for the full diagnosis. On machines
without clang-cl the fix-up is a no-op and this cross-check skips gracefully.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, ZETA, Drift, ReferenceParticle  # noqa: F401

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")


def _xtrack_drift_rmatrix(length: float, mass0_eV: float, q0: float, gamma0: float) -> np.ndarray:
    """Full one-turn 6x6 R-matrix of a single xtrack Drift, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=mass0_eV, q0=q0, gamma0=gamma0)
    line = xt.Line(elements=[xt.Drift(length=length)])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_drift_matrix_matches_xtrack() -> None:
    L = 2.0
    mass0 = 0.51099895069e6  # electron, eV
    gamma0 = 2.0  # non-ultrarelativistic so R56 = L/gamma0^2 is unambiguous

    R_xt = _xtrack_drift_rmatrix(L, mass0, q0=-1, gamma0=gamma0)
    ref = ReferenceParticle.from_gamma(mass0, gamma0, charge=-1.0)
    R_us = Drift(L).matrix(ref)

    # Gold check: every entry of the 6x6 map agrees with xtrack, not just the two
    # non-trivial couplings. This pins the whole drift convention (R12 = R34 = L,
    # the identity structure elsewhere) against the reference tracker.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # The physics that distinguishes our convention: R56 = L/gamma0^2 (momentum
    # variable delta), NOT L/(beta0^2 gamma0^2) (energy variable). For gamma0 = 2
    # those are 0.5 vs 0.667 — xtrack confirms the momentum-variable value, and
    # its +sign (delta > 0 arrives earlier, so zeta increases).
    assert R_us[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-9)
    assert R_xt[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-6)


# --------------------------------------------------------------------------
# The exact map (L1) — and the model xtrack has to be told to use
# --------------------------------------------------------------------------

MASS0 = 938.27208816e6
GAMMA0 = 5.0
L_EXACT = 2.0

# Deliberately large angles. At px = py = 0 the exact map, xtrack's "expanded" default
# and accsim's linear matrix all coincide, so an on-axis comparison would pass whatever
# was implemented — this is the same reason the analytic gate uses large angles.
_STATES = [
    np.array([1.0e-3, 1.0e-2, -5.0e-4, 7.0e-3, 2.0e-3, 1.0e-3]),
    np.array([0.0, 5.0e-2, 0.0, -3.0e-2, 0.0, 1.0e-2]),
]


def _xtrack_tracked(model: str) -> list[np.ndarray]:
    """Track every state in ``_STATES`` through one ``xt.Drift`` of the given model."""
    line = xt.Line(elements=[xt.Drift(length=L_EXACT, model=model)])
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    out = []
    for st in _STATES:
        p = xt.Particles(
            mass0=MASS0,
            q0=1,
            gamma0=GAMMA0,
            x=st[0],
            px=st[1],
            y=st[2],
            py=st[3],
            zeta=st[4],
            delta=st[5],
        )
        line.track(p)
        out.append(np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]]))
    return out


def test_the_exact_drift_map_matches_xtracks_exact_model() -> None:
    r"""accsim's ``Drift.track`` **is** ``xt.Drift(model="exact")``, to machine precision.

    Two independent implementations of the same Hamiltonian flow, agreeing to ``4.4e-16``
    on all six coordinates — including ``zeta``, which accsim evaluates in a rationalised
    form (``L (delta(2+delta)/gamma0^2 - px^2 - py^2) / (pz (pz + E/E0))``) precisely so
    that nothing cancels. That the two arithmetics land on the same bits is the check
    that the rearrangement is algebra and not an approximation.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    for st, want in zip(_STATES, _xtrack_tracked("exact"), strict=True):
        got = Drift(L_EXACT).track(st, ref)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)


def test_xtracks_default_drift_is_the_expanded_model_not_the_exact_one() -> None:
    r"""**The trap this cross-check exists to close.** ``xt.Drift()`` is *not* exact.

    Its ``model`` defaults to ``"adaptive"``, which resolves to ``"expanded"``
    (``xtrack/beam_elements/elements_src/drift.h``): ``x += L px / (1 + delta)``, paraxial
    in the angles, the MAD-X convention. accsim implements ``x += L px / pz``. Comparing
    the two produces an ``O(angle^3)`` discrepancy — ``1.5e-6`` here at ``px = 1e-2`` and
    ``1.7e-4`` at ``px = 5e-2`` — which reads exactly like a sign or convention bug and is
    nothing of the kind.

    So this asserts the disagreement, deliberately: any reference test that builds a
    line with drifts and compares off-axis must pass ``model="exact"``, and this is the
    test that says why. It also makes the *analytic* choice non-vacuous by exhibiting the
    other candidate at a distance no tolerance would absorb.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    exact, expanded = _xtrack_tracked("exact"), _xtrack_tracked("expanded")
    for st, want_exact, want_expanded in zip(_STATES, exact, expanded, strict=True):
        got = Drift(L_EXACT).track(st, ref)
        # accsim is the exact model...
        assert np.max(np.abs(got - want_exact)) < 1e-15
        # ...and demonstrably not the expanded one, by orders.
        assert np.max(np.abs(got - want_expanded)) > 1e-6
        # The two xtrack models differ by (px^2 + py^2)/2 relatively, which is what
        # pins the discrepancy on the transverse denominator and not on zeta.
        angle_sq = float(st[1] ** 2 + st[3] ** 2)
        relative = abs(want_exact[0] - want_expanded[0]) / abs(want_exact[0] - st[0])
        assert relative == pytest.approx(0.5 * angle_sq, rel=0.05)


def test_the_linear_matrix_is_the_exact_maps_slope_at_the_origin() -> None:
    """The R-matrix cross-check above is model-independent, and this says why.

    ``get_R_matrix`` linearises about the reference particle, where ``px = py = 0`` and
    the exact map's Jacobian *is* the linear matrix. So both xtrack models give the same
    R-matrix and `test_drift_matrix_matches_xtrack` needed no change when the exact map
    landed — which is also why every design-optics cross-check in this directory is
    untouched by L1.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    on_axis = np.array([1.0e-3, 0.0, -5.0e-4, 0.0, 2.0e-3, 0.0])
    exact = Drift(L_EXACT).track(on_axis, ref)
    linear = Drift(L_EXACT).matrix(ref) @ on_axis
    # Transverse: bit-for-bit, because every new term carries a factor of px or py.
    np.testing.assert_array_equal(exact[[0, 1, 2, 3, 5]], linear[[0, 1, 2, 3, 5]])
