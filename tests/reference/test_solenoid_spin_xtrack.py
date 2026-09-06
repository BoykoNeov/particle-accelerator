r"""Cross-check spin and radiation in a solenoid (S2) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**This file is unusual for this project: its headline result is that the reference code is
wrong, and the disagreement is asserted as a mechanism rather than measured against a
tolerance.** That is not a position taken lightly, so the evidence is laid out first.

xtrack's ``magnet_spin`` (``track_magnet_radiation.h``) builds the direction of motion from
``px + ax``. Its **own solenoid body** computes the kinetic momentum as ``pk1 = px + sk*y``
(that is ``px - ax``, ``track_magnet_drift.h:427``), and its **own radiation path** uses
``mean_kin_px`` built from ``px - ax`` (``track_magnet.h:97,108``). One code, two formulas
for one quantity, and only the spin one disagrees with the definition of kinetic momentum.
Every element validated before S2 has ``a = 0``, so the two had never disagreed and nothing in
either code caught it.

Four independent things say the spin one is the error:

1. **The disagreement is flat under slicing.** ``n = 1, 2, 4`` slices of the same solenoid
   leave the residual at ``2.883823e-03`` without a digit moving. A quadrature or sampling
   difference shrinks with refinement; a model difference does not.
2. **Its two orders say it is first order in** ``a``. Linear in the transverse amplitude,
   quadratic in ``ks`` — i.e. ``ks (ks y)``, the shape of a vector-potential error and not
   of a coefficient, a length or a quadrature. It does not, on its own, separate the *sign*
   flip from the fact that xtrack samples the momentum at the magnet's **exit** where accsim
   samples the traversal mean; that second difference is the same leading order and about
   13% of the size. Point 4 is what separates them.
3. **accsim fed xtrack's own formula reproduces xtrack**, to ``5.2e-6`` — and what is left
   over is **second** order in the amplitude where the disagreement it replaced is first.
   The swap turns a model difference into an ordinary quadrature one, so the disagreement
   is entirely in that one expression.
4. **The arbiter-free leg settles the sign without either code.** S1's validated map says
   what the trajectory does; central-differencing it gives the true tangent, and the
   kinetic momentum reproduces it to ``6.3e-11`` while xtrack's misses by ``2|a|``. That is
   in ``tests/analytic/test_solenoid_spin.py`` and is the reason this file can afford to
   disagree with its only arbiter.

**One arbiter, and it is the disputed one.** MAD-X has no spin tracking; N1-N5 all ran on
xtrack alone, so this is axis N's established condition rather than a deficiency opened
here. The *radiation* half is the clean one — xtrack's radiation path uses the correct
momentum — and it is gated as an agreement in the usual way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import ReferenceParticle, Solenoid
from accsim.coords import DELTA, PX, PY, ZETA, X, Y
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G
from accsim.reference import ELECTRON_MASS_EV as MASS0
from accsim.spin import precession_vector, rotate

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

ENERGY = 5e9
P0C = math.sqrt(ENERGY**2 - MASS0**2)
LENGTH = 0.9
KS = 0.6

SPIN0 = np.array([0.2, 0.9, math.sqrt(1.0 - 0.04 - 0.81)])
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 2.0e-4, 0.0, 0.0])


def _ref(g: float = G) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(MASS0, ENERGY, charge=-1.0, anomalous_moment=g)


_LINES: dict[tuple, object] = {}


def _line(ks: float, length: float, slices: int = 1, radiation: str | None = None):
    """A solenoid line, cached: every ``xt.Line`` build JIT-compiles a fresh C kernel.

    ``xt.UniformSolenoid`` is the element (``xt.Solenoid`` is deprecated in favour of it,
    and S1 checked that both give the same map). Its docstring is explicit that radiation
    and spin happen in the **body** only, with nothing in the fringe — which is the same
    hard-edge convention S1 shipped, so the two are comparable element for element.
    """
    key = (ks, length, slices, radiation)
    if key not in _LINES:
        elements = [xt.UniformSolenoid(length=length / slices, ks=ks) for _ in range(slices)]
        line = xt.Line(elements=elements, element_names=[f"s{i}" for i in range(slices)])
        line.particle_ref = xt.Particles(mass0=MASS0, p0c=P0C, q0=-1)
        if radiation is None:
            # Without this the kernel is built with spin off and track() leaves the spin
            # exactly unchanged -- no error, no warning. N1's trap, unchanged.
            line.configure_spin("auto")
        try:
            line.build_tracker()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
        if radiation is not None:
            line.configure_radiation(model=radiation)
        _LINES[key] = line
    return _LINES[key]


def _xtrack(ks, length, state, slices=1, radiation=None):
    line = _line(ks, length, slices, radiation)
    p = xt.Particles(
        mass0=MASS0,
        p0c=P0C,
        q0=-1,
        anomalous_magnetic_moment=G,
        x=state[X],
        px=state[PX],
        y=state[Y],
        py=state[PY],
        zeta=state[ZETA],
        delta=state[DELTA],
        spin_x=SPIN0[0],
        spin_y=SPIN0[1],
        spin_z=SPIN0[2],
    )
    line.track(p)
    return (
        np.array([float(p.spin_x[0]), float(p.spin_y[0]), float(p.spin_z[0])]),
        np.array(
            [
                float(p.x[0]),
                float(p.px[0]),
                float(p.y[0]),
                float(p.py[0]),
                float(p.zeta[0]),
                float(p.delta[0]),
            ]
        ),
    )


def _accsim_with_xtracks_own_momentum(ks: float, length: float, state: np.ndarray):
    r"""accsim's map, fed ``px + ax`` — a deliberate reimplementation of xtrack's formula.

    Everything else is the shipped code: the same ``precession_vector``, the same
    ``rotate``, the same path length. Only the two lines that build the momentum are
    xtrack's. If this reproduces xtrack and the shipped version does not, the disagreement
    is in that expression and nowhere else — which is the claim, and it is why the
    reimplementation is worth having in a file otherwise built on shipped calls.
    """
    sol = Solenoid(length, ks)
    ref = _ref()
    before = np.asarray(state, dtype=float)
    after = sol.track(before.copy(), ref)

    # xtrack's magnet_spin: the EXIT canonical momentum, PLUS the vector potential
    ax_out, ay_out = (float(v) for v in sol.normalized_vector_potential(after[X], after[Y]))
    px, py = after[PX] + ax_out, after[PY] + ay_out

    mid_x, mid_y = 0.5 * (before[X] + after[X]), 0.5 * (before[Y] + after[Y])
    bs = sol.longitudinal_field(mid_x, mid_y)
    omega = precession_vector(0.0, 0.0, px, py, after[DELTA], ref, bs=bs)

    p_ev = ref.momentum_eV * (1.0 + after[DELTA])
    energy = math.hypot(p_ev, MASS0)
    l_path = (p_ev / energy) / ref.beta0 * (length - (after[ZETA] - before[ZETA]))
    return rotate(SPIN0, omega, l_path)


# ============================== the orbit, first =========================================


def test_the_orbit_still_agrees_so_the_spin_comparison_is_about_the_spin() -> None:
    """S1's agreement, restated on this file's own line before anything is concluded.

    A spin disagreement is only interesting if the two codes are tracking the same
    trajectory. They are: the 6D states agree at S1's measured paraxial floor, so nothing
    below can be blamed on the orbit.
    """
    _, xt_state = _xtrack(KS, LENGTH, STATE)
    ac_state = Solenoid(LENGTH, KS).track(STATE.copy(), _ref())
    np.testing.assert_allclose(ac_state, xt_state, rtol=0, atol=1e-9)


def test_xtrack_precesses_spin_in_a_solenoid_at_all() -> None:
    """The premise of the whole file, and it is not obvious.

    A solenoid has **no transverse field**, and both codes reach an element's field through
    an accessor that a naive implementation would let return zero. Before any disagreement
    can be characterised, the reference has to be doing something: it rotates the spin by
    ``|dS| ~ 0.44`` and keeps it a unit vector exactly.
    """
    xt_spin, _ = _xtrack(KS, LENGTH, STATE)
    assert float(np.abs(xt_spin - SPIN0).max()) > 0.4
    assert float(np.linalg.norm(xt_spin)) == pytest.approx(1.0, rel=0, abs=1e-15)


# ================== the disagreement, asserted as a mechanism ============================


def test_the_disagreement_with_xtrack_does_not_converge_away_under_slicing() -> None:
    """Evidence 1: refinement does not touch it, so it is a model difference.

    Slicing is the standard way to tell a quadrature difference from a model difference,
    and this project has used it on every axis since B2. A midpoint-rule gap shrinks as
    ``1/N^2``; a wrong formula does not shrink at all. Here the residual is the **same
    number** at 1, 2 and 4 slices — a relative spread below ``1e-9`` — while the quantity
    itself is ``2.9e-3``.
    """
    ac_spin = Solenoid(LENGTH, KS).track_with_spin(STATE.copy(), SPIN0.copy(), _ref())[1]
    residuals = []
    for slices in (1, 2, 4):
        xt_spin, _ = _xtrack(KS, LENGTH, STATE, slices=slices)
        residuals.append(float(np.abs(ac_spin - xt_spin).max()))

    assert residuals[0] == pytest.approx(2.883823e-03, rel=1e-3)
    for finer in residuals[1:]:
        assert finer == pytest.approx(residuals[0], rel=1e-9)


def test_the_disagreement_is_linear_in_the_amplitude_and_quadratic_in_ks() -> None:
    r"""Evidence 2: the exponents say the error is first order in the vector potential.

    ``a = (-ks y/2, +ks x/2)`` is first order in the transverse amplitude and first order in
    ``ks``; it enters the precession multiplied by the field, which is another power of
    ``ks``. So an error proportional to ``a`` shows up **linear in the amplitude and
    quadratic in ``ks``**, where a wrong coefficient would be zeroth order in the amplitude
    and a quadrature gap would not survive the slicing test above. J2's rule: gate the
    exponents.

    **The amplitude exponent is exact and the ``ks`` one is not, and the reason is
    measured.** xtrack's ``magnet_spin`` samples the momentum at the magnet's **exit**;
    accsim samples the traversal **mean**. Their difference is therefore ``2a`` *plus* an
    endpoint-sampling term of the same leading order and about 13% of the size, whose own
    higher-order corrections differ — so the ratio oscillates about 4 at working strengths
    (measured 4.44 then 3.60 over ``ks = 0.6, 0.3, 0.15``) and settles on it only as
    ``ks -> 0`` (3.955, 3.978, **3.989** at ``ks = 0.02, 0.01, 0.005``). The band below
    excludes 2 and 8 comfortably, which is what the exponent has to do here; the clean
    asymptotic ``4.00`` is gated in ``tests/analytic/test_solenoid_spin.py``, where the two
    candidate models can be swept to arbitrarily small ``ks`` without the reference's own
    quadrature gap coming up underneath the signal.
    """
    ref = _ref()
    by_amplitude = []
    for f in (1.0, 0.5, 0.25):
        state = STATE * np.array([f, f, f, f, 1.0, 1.0])
        xt_spin, _ = _xtrack(KS, LENGTH, state)
        ac_spin = Solenoid(LENGTH, KS).track_with_spin(state.copy(), SPIN0.copy(), ref)[1]
        by_amplitude.append(float(np.abs(ac_spin - xt_spin).max()))
    for coarse, fine in zip(by_amplitude, by_amplitude[1:], strict=False):
        assert coarse / fine == pytest.approx(2.0, rel=0.02)

    by_strength = []
    for ks in (0.6, 0.3, 0.15):
        xt_spin, _ = _xtrack(ks, LENGTH, STATE)
        ac_spin = Solenoid(LENGTH, ks).track_with_spin(STATE.copy(), SPIN0.copy(), ref)[1]
        by_strength.append(float(np.abs(ac_spin - xt_spin).max()))
    for coarse, fine in zip(by_strength, by_strength[1:], strict=False):
        assert 3.4 < coarse / fine < 4.6  # quadratic; 2 and 8 are both far outside


def test_accsim_fed_xtracks_own_momentum_reproduces_xtrack() -> None:
    """Evidence 3, and the one that localises the disagreement to a single expression.

    Swap ``p - a`` for ``p + a`` — xtrack's ``magnet_spin`` — and change nothing else, and
    the two codes agree to ``5.2e-6``: three orders below the ``2.9e-3`` the shipped model
    differs by, and what is left is the ordinary midpoint-versus-back-derived-field gap N1
    already records for every element. There is no second disagreement hiding behind the
    first.
    """
    residuals = []
    for f in (1.0, 0.5, 0.25):
        state = STATE * np.array([f, f, f, f, 1.0, 1.0])
        xt_spin, _ = _xtrack(KS, LENGTH, state)
        residuals.append(
            float(np.abs(_accsim_with_xtracks_own_momentum(KS, LENGTH, state) - xt_spin).max())
        )

    assert residuals[0] == pytest.approx(5.235832e-06, rel=1e-3)
    shipped = Solenoid(LENGTH, KS).track_with_spin(STATE.copy(), SPIN0.copy(), _ref())[1]
    xt_spin, _ = _xtrack(KS, LENGTH, STATE)
    assert float(np.abs(shipped - xt_spin).max()) > 100.0 * residuals[0]

    # ...and what is left over is SECOND order in the amplitude, where the disagreement it
    # replaced is first. That is the difference between a quadrature gap and a model one,
    # measured on the same sweep rather than argued.
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(4.0, rel=0.02)


# ========================= radiation: the clean half =====================================


def test_radiation_in_a_solenoid_agrees_with_xtrack() -> None:
    """The half with an undisputed arbiter, because xtrack uses the right momentum here.

    ``track_magnet.h`` builds ``old_kin_px``/``new_kin_px`` from ``px - ax`` for the
    radiation path — the *correct* kinetic momentum, the one its spin path does not use. So
    where the spin comparison above is a disagreement asserted by mechanism, this is an
    ordinary agreement, and the two together are what make the claim about ``magnet_spin``
    a claim about ``magnet_spin`` rather than about accsim.

    The energy lost is compared relatively, because the absolute number is tiny: an
    off-axis particle in a solenoid radiates only through the transverse velocity its own
    vector potential gives it.
    """
    ref = _ref()
    for f in (1.0, 0.5):
        state = STATE * np.array([f, f, f, f, 1.0, 1.0])
        _, xt_state = _xtrack(KS, LENGTH, state, radiation="mean")
        ac_state = Solenoid(LENGTH, KS).track(state.copy(), ref, radiation="mean")
        xt_loss = state[DELTA] - xt_state[DELTA]
        ac_loss = state[DELTA] - ac_state[DELTA]
        assert xt_loss > 0.0
        # measured 3.2e-7 and 1.3e-6 relative; the gap is the midpoint sampling N1 and B2
        # already record for every element, and it is the same size here.
        assert ac_loss == pytest.approx(xt_loss, rel=3e-6)


def test_an_on_axis_particle_radiates_exactly_zero_in_both_codes() -> None:
    """The exact-zero half, agreed by both codes rather than only asserted by one.

    A particle travelling along the field feels no perpendicular field, so the loss is not
    small but **exactly** zero. It is also the one statement a silent ``(0, 0)`` field would
    have got right for the wrong reason, which is why the off-axis agreement above is the
    discriminating half and this is the control.
    """
    zero = np.zeros(6)
    _, xt_state = _xtrack(KS, LENGTH, zero, radiation="mean")
    ac_state = Solenoid(LENGTH, KS).track(zero.copy(), _ref(), radiation="mean")
    assert float(xt_state[DELTA]) == 0.0
    assert float(ac_state[DELTA]) == 0.0
