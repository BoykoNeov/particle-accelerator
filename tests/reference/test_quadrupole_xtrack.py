"""Cross-check the Quadrupole transfer matrix against xtrack.

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed)
when xtrack's JIT C-kernel compilation is unavailable. See
``tests/reference/test_drift_xtrack.py`` and ``docs/CONVENTIONS.md`` for the
toolchain story.

The purpose beyond entrywise agreement: pin two convention choices against the
reference tracker — (1) ``k1 > 0`` focuses in x / defocuses in y, and (2) the
longitudinal slip ``R56 = L/gamma0^2`` is carried *inside* the thick quad (some
codes slice it differently), the gotcha that motivated this check.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import DELTA, PX, PY, ZETA, Quadrupole, ReferenceParticle, X, Y

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")


def _xtrack_quad_rmatrix(
    length: float, k1: float, mass0_eV: float, q0: float, gamma0: float
) -> np.ndarray:
    """One-turn 6x6 R-matrix of a single xtrack Quadrupole, or skip if JIT can't build."""
    ref = xt.Particles(mass0=mass0_eV, q0=q0, gamma0=gamma0)
    line = xt.Line(elements=[xt.Quadrupole(length=length, k1=k1)])
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def test_quadrupole_matrix_matches_xtrack() -> None:
    L, k1 = 0.5, 1.2
    mass0 = 0.51099895069e6  # electron, eV
    gamma0 = 2.0  # non-ultrarelativistic so R56 = L/gamma0^2 is unambiguous

    R_xt = _xtrack_quad_rmatrix(L, k1, mass0, q0=-1, gamma0=gamma0)
    ref = ReferenceParticle.from_gamma(mass0, gamma0, charge=-1.0)
    R_us = Quadrupole(L, k1).matrix(ref)

    # Whole 6x6 agrees: focusing structure, both planes, and the longitudinal block.
    np.testing.assert_allclose(R_us, R_xt, rtol=1e-6, atol=1e-9)

    # Convention pins, stated explicitly against the reference:
    assert R_us[PX, X] < 0.0 and R_xt[PX, X] < 0.0  # k1>0 focuses x (R21 < 0)
    assert R_us[PY, Y] > 0.0 and R_xt[PY, Y] > 0.0  # k1>0 defocuses y (R43 > 0)
    assert R_xt[ZETA, DELTA] == pytest.approx(L / gamma0**2, rel=1e-6)  # slip carried here


# --------------------------------------------------------------------------
# The momentum-dependent map (L2) — and the model family it belongs to
# --------------------------------------------------------------------------

MASS0 = 938.27208816e6
GAMMA0 = 5.0
L_EXACT = 0.7
K1_EXACT = 1.2

# Deliberately large **delta**, not large angles. This map is exact in delta and
# paraxial in the angles, so momentum is the axis along which the candidate maps
# separate — the mirror image of the drift's cross-check in test_drift_xtrack.py,
# where the discriminating axis was the angle. At delta = 0 accsim's map is its own
# linear matrix, so an on-momentum comparison would pass whatever was implemented.
_STATES = [
    np.array([1.0e-3, 1.0e-4, -5.0e-4, 7.0e-5, 2.0e-3, 1.0e-3]),
    np.array([2.0e-3, -3.0e-4, 1.5e-3, 2.0e-4, 0.0, 3.0e-2]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0e-2]),
]


def _xtrack_tracked(model: str) -> list[np.ndarray]:
    """Track every state in ``_STATES`` through one ``xt.Quadrupole`` of the given model."""
    line = xt.Line(elements=[xt.Quadrupole(length=L_EXACT, k1=K1_EXACT, model=model)])
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


def test_the_momentum_dependent_map_matches_xtracks_mat_kick_mat() -> None:
    r"""accsim's ``Quadrupole.track`` **is** ``xt.Quadrupole(model="mat-kick-mat")``.

    Two independent implementations of the same closed-form flow — the exact solution
    of the paraxial Hamiltonian, MAD-X's thick combined-function map — agreeing to
    ``1.1e-16`` on all six coordinates, ``zeta`` included.

    ``zeta`` is the coordinate that makes this a check and not a formality. xtrack (and
    MAD-X) evaluate it as ``L - path/rvv``, differencing two numbers of size ``L``;
    accsim splits it into ``L(1 - 1/rvv)``, rationalised through ``(1+delta) + E/E0``,
    minus the path-lengthening integral, and evaluates the integral itself in a form
    with no ``1/K`` in it at all. That the two arithmetics land on the same bits is the
    evidence that the rearrangement is algebra rather than an approximation.

    ``model`` is passed explicitly even though ``mat-kick-mat`` is xtrack's default for
    a quadrupole (``QUADRUPOLE_DEFAULT_MODEL`` in ``default_magnet_config.h``), because
    the drift taught this suite that a default is not a specification — see
    ``test_drift_xtrack.py``, where xtrack's default drift turned out **not** to be the
    exact one.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    for st, want in zip(_STATES, _xtrack_tracked("mat-kick-mat"), strict=True):
        got = Quadrupole(L_EXACT, K1_EXACT).track(st, ref)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)


def test_accsim_is_not_the_splitting_family_and_the_difference_is_orders() -> None:
    r"""**The choice, made non-vacuous.** A sliced quadrupole is a different map.

    There is no closed form for the *exact* quadrupole Hamiltonian — the square root
    and the quadratic potential do not commute — so every code picks one of two
    families: solve the **paraxial** Hamiltonian exactly (this map), or split the exact
    one and integrate numerically (xtrack's ``drift-kick-drift-*``, PTC's ``exact``).
    Both are symplectic; they are different maps, and they disagree here by ``8e-5`` to
    ``1.5e-4`` against the ``1e-16`` asserted above.

    accsim takes the closed form because ``matrix()`` has to remain the exact Jacobian
    of ``track()`` at the origin — a sliced map's origin Jacobian is the *sliced*
    approximation to the cos/sin block, not the block — and that identity is what every
    design-optics gate in the package rests on. Recording the size of the alternative
    is what keeps that a choice rather than an accident: it is far too large for any
    tolerance to absorb, so nothing here could be quietly satisfied by the other family.

    The trade is stated where it belongs, on
    :class:`~accsim.elements.quadrupole.Quadrupole`: exact in ``delta``, paraxial in
    the angles, dropping ``O(angle^3)``.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    sliced = _xtrack_tracked("drift-kick-drift-exact")
    for st, want_sliced in zip(_STATES[:2], sliced[:2], strict=True):
        got = Quadrupole(L_EXACT, K1_EXACT).track(st, ref)
        assert np.max(np.abs(got - want_sliced)) > 1e-5

    # On axis the two families coincide exactly — there is nothing to slice — which is
    # why the states above carry a transverse amplitude and the third one is excluded.
    on_axis = _STATES[2]
    assert np.max(np.abs(Quadrupole(L_EXACT, K1_EXACT).track(on_axis, ref) - sliced[2])) < 1e-15


def test_the_linear_matrix_is_the_maps_slope_at_the_reference_particle() -> None:
    """Why the R-matrix cross-check above is model-independent — and untouched by L2.

    ``get_R_matrix`` linearises about the reference particle, where ``delta = 0`` and
    accsim's map *is* its linear matrix. So every xtrack quadrupole model returns the
    same R-matrix, `test_quadrupole_matrix_matches_xtrack` needed no change when the
    momentum-dependent map landed, and neither did any design-optics cross-check in
    this directory.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    on_momentum = np.array([1.0e-3, 1.0e-4, -5.0e-4, 7.0e-5, 2.0e-3, 0.0])
    exact = Quadrupole(L_EXACT, K1_EXACT).track(on_momentum, ref)
    linear = Quadrupole(L_EXACT, K1_EXACT).matrix(ref) @ on_momentum
    # Transverse to the last bit; zeta differs by the path lengthening, which no matrix
    # in these coordinates can carry.
    np.testing.assert_allclose(exact[[X, PX, Y, PY]], linear[[X, PX, Y, PY]], rtol=1e-15, atol=0.0)
    assert exact[ZETA] != linear[ZETA]
