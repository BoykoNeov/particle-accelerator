r"""Cross-check the radiation integrals / damping against xtrack (Stage 7).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

The analytic suite pins accsim's radiation quantities *internally* — Robinson
``J_x+J_y+J_z=4`` (exact), the isomagnetic ``I2``/``I3`` closed forms, ``I1 ==
alpha_c*C``, and the ``eps_x ∝ gamma^2`` / ``sigma_delta ∝ gamma`` scaling. This is
the external anchor: the *same* isomagnetic electron ring is built in xtrack, run
through its radiation twiss (``radiation_analysis=True``, ``model='mean'``), and the
lattice-level quantities compared.

**Two tiers of agreement, and why.** accsim and xtrack agree to ~1e-6 on the
quantities that don't depend on the bend-body radiation convention, and to ~1-3% on
the ones that do:

- **Tight (convention-invariant): energy loss ``U0`` and the vertical damping time
  ``tau_y = 2 E T0 / U0``** (``J_y = 1`` in both codes). These pin ``I2``, ``C_gamma``,
  and the ``2 E T0`` assembly against an external code. xtrack's own
  ``momentum_compaction_factor`` also matches accsim's ``I1`` to ~1e-7 (checked in
  ``test_momentum_compaction_xtrack.py``), so the dispersion transport is identical.
- **Looser (different method): partition numbers ``J_x``/``J_z`` (~1-2%) and the
  equilibrium emittance ``eps_x`` (~3-4%).** accsim computes these from the standard
  **radiation-integral formulae** (``J = 1 -/+ I4/I2``, ``eps_x = C_q gamma^2 I5/(J_x
  I2)``). xtrack's ``radiation_analysis`` instead computes them from the **damped
  one-turn-map eigen/envelope analysis** — it does *not* expose radiation integrals at
  all (``tw`` has no ``rad_int*`` attributes). The two methods coincide in the weak-
  bending limit but differ at the 1-3% level in a ring this strong (``I4/I2 ~ 0.38``,
  ~5x a normal ring). This is **not** an accsim error: accsim's ``I5`` is independently
  pinned within the baseline against ``propagate_twiss`` (xtrack-validated Twiss) to
  1e-6, ``I4 = h^2 alpha_c C`` to 1e-10, and ``I1``/``I2`` match xtrack to 1e-6 — so the
  integrals are right; the residual is integral-formula vs exact-eigenanalysis. The
  cross-check confirms magnitude, sign, and scaling; the absolute is pinned by the
  analytic gates, not this tolerance.
"""

from __future__ import annotations

import math

import pytest

from accsim import (
    Dipole,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    damping_partition_numbers,
    damping_times,
    energy_loss_per_turn,
    equilibrium_emittance,
)
from accsim.reference import CLIGHT

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

ELECTRON_MASS_EV = 0.51099895069e6
N_CELLS = 20
F_FOCAL = 2.5
L_BEND = 1.0
ANGLE = 2.0 * math.pi / (2 * N_CELLS)  # total bend 2*pi
ENERGY_EV = 1.0e9  # 1 GeV electron


def _accsim_ring() -> Lattice:
    ref = ReferenceParticle.from_total_energy(ELECTRON_MASS_EV, ENERGY_EV)
    cell = [
        ThinQuadrupole(0.5 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(-1.0 / F_FOCAL),
        Dipole(L_BEND, ANGLE),
        ThinQuadrupole(0.5 / F_FOCAL),
    ]
    return Lattice(cell * N_CELLS, ref=ref)


def _xtrack_radiation_twiss():
    def quad(k1l: float):
        return xt.Multipole(knl=[0.0, k1l], length=0.0)

    def bend():
        b = xt.Bend(length=L_BEND, angle=ANGLE, k1=0.0)
        b.edge_entry_active = 0
        b.edge_exit_active = 0
        return b

    cell = [quad(0.5 / F_FOCAL), bend(), quad(-1.0 / F_FOCAL), bend(), quad(0.5 / F_FOCAL)]
    elems = cell * N_CELLS
    # A cavity is needed for the 6D (radiation) closed solution; voltage well above
    # U0 gives a comfortable bucket, frequency a harmonic of the revolution frequency.
    f_rev = CLIGHT / (N_CELLS * 2 * L_BEND)
    elems.append(xt.Cavity(voltage=2.0e6, frequency=100 * f_rev, lag=180.0))
    line = xt.Line(elements=elems)
    line.particle_ref = xt.Particles(
        mass0=ELECTRON_MASS_EV, q0=-1, gamma0=ENERGY_EV / ELECTRON_MASS_EV
    )
    try:
        line.build_tracker()
        line.configure_radiation(model="mean")
        return line.twiss(radiation_analysis=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack radiation twiss unavailable: {type(exc).__name__}: {exc}")


def test_radiation_matches_xtrack() -> None:
    tw = _xtrack_radiation_twiss()
    ring = _accsim_ring()

    e = ring.ref.total_energy_eV
    t0 = ring.length / (ring.ref.beta0 * CLIGHT)

    # --- Tight tier: convention-invariant quantities. ---
    u0 = energy_loss_per_turn(ring)
    assert u0 == pytest.approx(tw.eneloss_turn, rel=1e-4)  # I2 + C_gamma, external

    # xtrack reports damping *rates* (1/s); tau = 1/rate. tau_y uses J_y = 1 (both),
    # so tau_y = 2 E T0 / U0 is convention-invariant.
    tau_x, tau_y, tau_z = damping_times(ring)
    xt_tau = [1.0 / a for a in tw.damping_constants_s]
    assert tau_y == pytest.approx(2.0 * e * t0 / u0, rel=1e-10)  # accsim self-consistent
    assert tau_y == pytest.approx(xt_tau[1], rel=2e-3)  # and matches xtrack

    # --- Looser tier: integral-formulae vs xtrack's damped-map eigenanalysis (docstring). ---
    jx, jy, jz = damping_partition_numbers(ring)
    xt_jx, xt_jy, xt_jz = tw.partition_numbers
    assert jy == pytest.approx(xt_jy, rel=1e-4)  # both exactly 1
    assert jx == pytest.approx(xt_jx, abs=0.01)  # ~1% (method difference, strong ring)
    assert jz == pytest.approx(xt_jz, abs=0.01)
    assert tau_x == pytest.approx(xt_tau[0], rel=2e-2)
    assert tau_z == pytest.approx(xt_tau[2], rel=2e-2)

    # Equilibrium geometric horizontal emittance — right magnitude/scaling; the ~3-4%
    # residual is the integral-formula vs eigen-analysis method difference (see docstring).
    eps_x = equilibrium_emittance(ring)
    assert eps_x == pytest.approx(tw.eq_gemitt_x, rel=4e-2)
