"""Cross-check the Dipole (pure sector bend) 6x6 against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The xtrack ``Bend`` is configured as a *pure sector* — edges disabled
(``edge_entry/exit_active = 0``) and ``k1 = 0`` — so its R-matrix is the bare
sector map with no edge focusing or gradient, apples-to-apples with
:class:`accsim.elements.dipole.Dipole`.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, ZETA, Dipole, ReferenceParticle, X

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 5.0  # non-ultrarelativistic so the L/gamma0^2 slip is sizeable
LENGTH = 2.0
ANGLE = 0.2


def _xtrack_bend_rmatrix() -> np.ndarray:
    """6x6 R-matrix of a pure sector xtrack Bend, or skip if the JIT can't build."""
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k1=0.0)
    bend.edge_entry_active = 0
    bend.edge_exit_active = 0
    line = xt.Line(elements=[bend])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_dipole_matrix_matches_xtrack() -> None:
    R_xt = _xtrack_bend_rmatrix()
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    R_us = Dipole(LENGTH, ANGLE).matrix(ref)

    # Whole 6x6: horizontal focusing, dispersion, vertical drift, and the
    # longitudinal path-length row all agree with the reference tracker.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # Convention pins stated against the reference:
    assert R_us[X, DELTA] > 0.0 and R_xt[X, DELTA] > 0.0  # dispersion outward
    # R56 carries the momentum-variable slip L/gamma0^2 minus the design-orbit arc,
    # NOT an energy-variable value — xtrack confirms the exact number.
    assert R_us[ZETA, DELTA] == pytest.approx(R_xt[ZETA, DELTA], rel=1e-6)
    assert R_us[PX, DELTA] == pytest.approx(R_xt[PX, DELTA], rel=1e-6)


# --------------------------------------------------------------------------
# The exact map (L3) — and the model family a bend belongs to
# --------------------------------------------------------------------------

L_EXACT = 1.0
ANGLE_EXACT = 0.3

# Large angle **and** large delta. A bend is the one element that is exact in both,
# so both doors have to be closed: a map that expanded the square root would agree
# at small angle, and one that mishandled the rigidity would agree at delta = 0.
_STATES = [
    np.array([3.0e-3, 8.0e-3, -2.0e-3, 5.0e-3, 1.0e-3, 2.0e-3]),
    np.array([2.0e-3, -5.0e-3, 1.5e-3, 4.0e-3, 0.0, 5.0e-2]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.0e-2]),
]


def _xtrack_tracked_bend(model: str) -> list[np.ndarray]:
    """Track every state in ``_STATES`` through one ``xt.Bend`` of the given model."""
    bend = xt.Bend(length=L_EXACT, angle=ANGLE_EXACT, k1=0.0, model=model)
    bend.edge_entry_active = 0  # accsim's default is a pure sector bend, no edges
    bend.edge_exit_active = 0
    line = xt.Line(elements=[bend])
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


def test_the_exact_bend_map_matches_xtracks_thick_bend() -> None:
    r"""accsim's ``Dipole.track`` **is** ``xt.Bend(model="bend-kick-bend")``, to 1e-15.

    Two independent implementations of the same circle, agreeing on all six coordinates.
    ``bend-kick-bend`` is the model whose thick step is the closed-form uniform-field
    solution (``track_thick_bend.h``); with ``k1 = 0`` and no multipoles the kick between
    the halves is empty, so that model *is* the exact map and the integrator setting
    cannot matter. That makes it the right arm for an exactness claim, where the default
    ``rot-kick-rot`` would fold in its own Yoshida splitting.

    ``zeta`` is included, and it is the sharp part: accsim evaluates it as
    ``L(1 - 1/rvv) - (delta L/(1+delta) + D/h) E/E0`` precisely so that nothing of size
    ``L`` is subtracted from anything else, where xtrack writes the cancelling
    ``length - delta_ell/rvv``. That two such different arithmetics land on the same bits
    is the check that the rearrangement is algebra and not an approximation.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    for st, want in zip(_STATES, _xtrack_tracked_bend("bend-kick-bend"), strict=True):
        got = Dipole(L_EXACT, ANGLE_EXACT).track(st, ref)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)


def test_the_expanded_bend_model_is_a_different_map_and_the_gate_separates_them() -> None:
    r"""**The trap, closed deliberately.** ``mat-kick-mat`` is not the exact bend.

    xtrack carries both families for a bend, as it does for a drift:

    - ``bend-kick-bend`` / the default ``rot-kick-rot``: the geometry is kept exactly
      and the multipoles are kicks. accsim implements this one.
    - ``mat-kick-mat`` (xtrack's own name for it is "expanded"): MAD-X's
      ``track_thick_cfd``, which expands the square root and is **paraxial in the
      angles** — the family a combined-function bend is stuck in, because there the
      exact flow has no closed form at all.

    They differ here by ``1e-5``, which is far above this gate's ``1e-15`` and reads
    like a bug if the model is not named. So the disagreement is asserted rather than
    discovered: any reference comparison involving an off-axis bend must say which
    family it means. It is also the size of what accsim's *combined-function* bend is
    still missing, since that is the branch L3 left on the linear map.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    exact = _xtrack_tracked_bend("bend-kick-bend")
    expanded = _xtrack_tracked_bend("mat-kick-mat")

    worst = max(float(np.abs(a - b).max()) for a, b in zip(exact, expanded, strict=True))
    assert worst > 1.0e-6, "the two model families must be distinguishable at these amplitudes"

    # accsim is on the exact side of that split, by four orders.
    for st, want in zip(_STATES, exact, strict=True):
        got = Dipole(L_EXACT, ANGLE_EXACT).track(st, ref)
        assert np.abs(got - want).max() < 1.0e-15


def test_the_default_bend_model_agrees_too_up_to_its_own_integrator() -> None:
    """The default ``rot-kick-rot`` is the same physics, integrated rather than closed.

    Worth pinning because every *ring* cross-check in the suite builds default bends:
    what those comparisons are entitled to is agreement at the splitting error, not at
    machine precision. ``rot-kick-rot`` keeps the curvature in a polar drift and puts the
    weak-focusing term ``-h k0 x`` into the kicks, integrated with Yoshida-4 — so even at
    ``k1 = 0`` there is something to split, and it does not vanish.

    The three models make a clean hierarchy, and stating it as one is the point of this
    test: **1.9e-16** for the closed form, **7.3e-10** for the same physics integrated,
    **1.4e-5** for the expanded family. The middle number is what a default-bend ring
    comparison is entitled to; reading the first there would be luck and the third would
    be a different model.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    worst = 0.0
    for st, want in zip(_STATES, _xtrack_tracked_bend("rot-kick-rot"), strict=True):
        got = Dipole(L_EXACT, ANGLE_EXACT).track(st, ref)
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst == pytest.approx(7.26e-10, rel=5e-2)
