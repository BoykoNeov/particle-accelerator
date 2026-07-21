r"""Cross-check the coupling vertical emittance ``eps_2`` against xtrack (G1 ε_y).

Marked ``reference``: skipped when xtrack is absent, and skipped (not failed) when
xtrack's JIT C-kernel compilation is unavailable (the clang-cl fix-up in
``conftest.py`` / ``_xtrack_jit``).

:func:`equilibrium_emittances_coupled` predicts the two eigen-mode equilibrium
emittances under betatron coupling from a *leading-order* sharing model (decoupled
detuning ``Delta`` + coupling strength ``|C^-|`` + the coupling-off ``eps_x0``). The
external anchor is xtrack's **radiation-envelope eigen-emittances** ``eq_gemitt_x`` /
``eq_gemitt_y`` (a full Sigma-matrix eigenanalysis, an independent method), on a
**weak-bend** electron ring near the difference resonance with a single thin skew.

**Why a weak-bend ring** (the scope conditions, stated in
``equilibrium_emittances_coupled`` and ``docs/CONVENTIONS.md``). Two things both grow
with bend strength and must be kept small for the sharing model to be clean:

  * **The base ``eps_x0`` method difference.** ``eps_x0`` is an integral-formula
    emittance (:func:`equilibrium_emittance`); xtrack's ``eq_gemitt_*`` is a damped-map
    **envelope** eigenanalysis. These already diverge at the Stage-7 level — ~3-4% on a
    weak ring, but **~3x** on a 3x-stronger-bend ring (verified: uncoupled
    ``eps_x0/eq_gemitt_x ≈ 0.36`` there). Since ``eps_2`` is a fraction of ``eps_x0``,
    the weak ring is what makes the *absolute* comparison meaningful.
  * **Skew-induced vertical dispersion** — a skew at finite ``D_x`` couples horizontal
    dispersion into a *vertical* one, a second ``eps_y`` source the sharing model does
    not carry (option B / out of scope). Also ``h``-weighted, so also small here.

On this weak ring (``h`` ~ 0.011 m^-1, ``J_x ≈ 0.997``) both are ≲ few %, so the
absolute ``eps_1``/``eps_2`` agree with the envelope to ~1-3%. The **convention-invariant**
check is the sharing *ratio* ``eps_2/eps_1`` (independent of the ``eps_x0`` normalisation),
and the test additionally **refutes the roadmap's originally pre-committed**
``eps_y/eps_x = |C^-|^2/(|C^-|^2 + Delta^2)`` form, ~3-4x too large here (the projected/raw
coefficient, not the eigen-mode ``1/4`` one).
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    Dipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    ThinSkewQuadrupole,
    closest_tune_approach,
    equilibrium_emittances_coupled,
    tunes,
)
from accsim.reference import CLIGHT

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
ENERGY_EV = 1.0e9
N_CELLS = 96
L_BEND = 3.0  # weak bends: h = angle/L_bend ~ 0.011 m^-1
LDRIFT = 0.4
KF = 0.32
KD = -0.245  # near the difference resonance (Delta ~ 0.007)
ANGLE = 2.0 * math.pi / (2 * N_CELLS)


def _dist_int(x: float) -> float:
    return abs(x - round(x))


def _accsim_cell() -> list:
    return [
        ThinQuadrupole(0.5 * KF),
        Drift(LDRIFT),
        Dipole(L_BEND, ANGLE),
        Drift(LDRIFT),
        ThinQuadrupole(KD),
        Drift(LDRIFT),
        Dipole(L_BEND, ANGLE),
        Drift(LDRIFT),
        ThinQuadrupole(0.5 * KF),
    ]


def _xtrack_line(k1sl: float) -> xt.Line:
    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    cell = [
        quad(0.5 * KF),
        xt.Drift(length=LDRIFT),
        bend(),
        xt.Drift(length=LDRIFT),
        quad(KD),
        xt.Drift(length=LDRIFT),
        bend(),
        xt.Drift(length=LDRIFT),
        quad(0.5 * KF),
    ]
    elems = [xt.Multipole(ksl=[0.0, k1sl], length=0.0)]  # thin skew coupling source
    for _ in range(N_CELLS):
        elems += cell
    f_rev = CLIGHT / (N_CELLS * (2 * L_BEND + 4 * LDRIFT))
    elems.append(xt.Cavity(voltage=5.0e6, frequency=400 * f_rev, lag=180.0))
    line = xt.Line(elements=elems)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=-1, gamma0=ENERGY_EV / MASS0)
    return line


def _twiss_or_skip(line: xt.Line):
    try:
        line.build_tracker()
        line.configure_radiation(model="mean")
        return line.twiss(radiation_analysis=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack radiation twiss unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("k1sl", [0.004, 0.008])
def test_coupled_eigen_emittances_match_xtrack(k1sl: float) -> None:
    """eps_1, eps_2 match xtrack's envelope eq_gemitt_x/y to a few percent, and the
    roadmap's pre-committed |C^-|^2/(|C^-|^2+Delta^2) form is decisively refuted."""
    ref = ReferenceParticle.from_total_energy(MASS0, ENERGY_EV)
    ring = _accsim_cell() * N_CELLS
    lat = Lattice([ThinSkewQuadrupole(k1sl), *ring], ref=ref)

    e1, e2 = equilibrium_emittances_coupled(lat)

    tw = _twiss_or_skip(_xtrack_line(k1sl))
    gx, gy = float(tw.eq_gemitt_x), float(tw.eq_gemitt_y)

    # both eigen-mode emittances match the envelope eigenanalysis (uniform ~1-3%
    # integral-vs-envelope underestimate, like the Stage-7 eps_x cross-check).
    assert e1 == pytest.approx(gx, rel=4e-2)
    assert e2 == pytest.approx(gy, rel=5e-2)
    # the sharing ratio itself (convention-invariant to the eps_x0 normalisation).
    assert e2 / e1 == pytest.approx(gy / gx, rel=6e-2)

    # discrimination guard: the roadmap's originally-committed F form is ~3-4x the truth,
    # so a test that merely "matched xtrack loosely" could not be satisfied by it.
    qx, qy = tunes(Lattice(ring, ref=ref))
    delta = _dist_int(qx - qy)
    cminus = closest_tune_approach(lat)
    roadmap_ratio = cminus**2 / (cminus**2 + delta**2)
    assert roadmap_ratio > 2.0 * (gy / gx)  # roadmap overpredicts by >2x — refuted
