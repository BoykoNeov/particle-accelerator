r"""P2 (iii) against xtrack: the cavity, by **tracking**, to the last bit.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**Why this leg is the sharp one.** PTC's ``maptable``
(``tests/reference/test_second_order_map_madx.py``) sees the cavity only through a
second-order *expansion* of a whole ring, at the ``1e-8`` floor that composing 36 thick
elements imposes. xtrack applies the same physics element by element in the same
coordinates, so what is compared here is the raw tracked state:

    ``track_rf.h``: ``LocalParticle_add_to_energy(part, q V sin(phase - k tau), 1)``
    ``local_particle_custom_api.h``: ``ptau += dE/p0c``; ``delta`` is then recomputed
    from ``ptau``, and ``pz_only = 1`` leaves ``px``/``py`` alone.

That is, entry for entry, the map :meth:`accsim.RFCavity._track_body` applies since
P2 (iii). ``x``, ``px``, ``y``, ``py`` and ``zeta`` come back **bit-identical** and
``delta`` agrees to ``2.6e-16``, against the ``8.3e-13`` the pre-P2 (iii) momentum kick
misses by — and that miss is gated on its *scaling* in ``delta``, not on a number.

**The ``2.6e-16`` is xtrack's floor, not accsim's.** xtrack reconstructs
``delta = sqrt(ptau^2 + 2 ptau/beta0 + 1) - 1``, and subtracting 1 from a number near 1
costs ``ulp(1) ~ 2.2e-16`` however small ``delta`` is. accsim's increment form never forms
that difference, so the residual here wanders between ``6e-18`` and ``2.6e-16`` with no
trend in ``delta`` — which is what a rounding floor looks like, as against the old map's
clean factor of ten per decade of ``delta``.

**The two codes reference the synchronous phase differently, and it is not this
milestone's doing.** accsim applies ``sin(phi_s - k zeta) - sin(phi_s)``; xtrack applies
the bare ``sin(phase - k tau)`` with no offset. accsim's convention is the kick *measured
relative to a ramping reference*, so a synchronous particle stays at ``delta = 0`` and the
``q V sin(phi_s)`` per turn is the reference's business (Stage 5's ``accelerate``);
xtrack's is the kick in the lab. They therefore differ by a constant ``q V sin(phi_s)`` of
**energy** unless ``sin phi_s = 0`` — which is why the agreement gates below run on a
stationary bucket, and why the offset gets a test of its own rather than a tolerance.

**Two more traps, both live.** xtrack's ``q`` is ``fabs(q0) * charge_ratio``, the
*absolute* charge, where accsim's is signed — so for a negative particle the
correspondence is ``phase = phi_s + pi`` (``test_spin_sidebands_xtrack.py``), asserted
below on an electron so this file does not quietly depend on the proton case. And
``XTRACK_CAVITY_PRESERVE_ANGLE``, were it defined in the build, would rescale
``px``/``py`` at the kick; the bit-identical ``px``/``py`` assertion is what catches that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import ReferenceParticle, RFCavity
from accsim.coords import DELTA, ZETA

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
ELECTRON_MASS0, ELECTRON_GAMMA0 = 0.51099895e6, 4000.0
VOLTAGE, FREQ_HZ = 1.0e6, 3.0e6

#: Stationary bucket. ``sin phi_s = 0`` is the condition under which accsim's
#: reference-relative kick and xtrack's lab kick are the *same* kick — see the module
#: docstring. The conversion this milestone changed is driven by ``cos phi_s``, so it is
#: fully exercised here; only the constant offset is switched off.
PHI_S = 0.0

#: The phase used to gate that offset, where both sines are nonzero.
PHI_S_OFFSET = 0.3

#: A generic probe: every coordinate nonzero, ``zeta`` far enough up the wave that the
#: kick is not tiny, and ``delta`` set per test.
STATE = np.array([2.0e-3, 1.0e-4, -1.5e-3, 5.0e-5, 2.0e-2, 5.0e-3])

#: Every ``xt.Line`` build JIT-compiles a fresh C kernel — xobjects names each module
#: ``uuid4().hex``, so nothing is ever cached between builds (``docs/CONVENTIONS.md`` ->
#: *Test-suite cost*), at ~12 s apiece. Four distinct cavities are needed in this file and
#: the parametrised test would otherwise rebuild the same one five times, so they are
#: cached here by their construction arguments.
_LINES: dict[tuple[float, float, float, int], object] = {}


def _tracker(phase: float, mass0: float, gamma0: float, q0: int):
    key = (phase, mass0, gamma0, q0)
    if key not in _LINES:
        line = xt.Line(elements=[xt.Cavity(voltage=VOLTAGE, frequency=FREQ_HZ, phase=phase)])
        line.particle_ref = xt.Particles(mass0=mass0, q0=q0, gamma0=gamma0)
        try:
            line.build_tracker()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
        _LINES[key] = line
    return _LINES[key]


def _track_xtrack(
    state: np.ndarray,
    phase: float = PHI_S,
    mass0: float = MASS0,
    gamma0: float = GAMMA0,
    q0: int = 1,
) -> np.ndarray:
    """Track ``state`` through a one-element xtrack line, returning the 6D result."""
    line = _tracker(phase, mass0, gamma0, q0)
    p = xt.Particles(
        mass0=mass0,
        q0=q0,
        gamma0=gamma0,
        x=state[0],
        px=state[1],
        y=state[2],
        py=state[3],
        zeta=state[4],
        delta=state[5],
    )
    line.track(p)
    return np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]])


def _linearised(cav: RFCavity, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
    """The pre-P2 (iii) map: the same kick, added straight to ``delta``."""
    out = np.array(state, dtype=float, copy=True)
    out[DELTA] += cav.energy_kick_pzeta(out[ZETA], ref)
    return out


def _energy_eV(delta: float, ref: ReferenceParticle) -> float:
    return math.hypot(ref.momentum_eV * (1.0 + delta), ref.mass_eV)


@pytest.fixture(scope="module")
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


@pytest.fixture(scope="module")
def cav() -> RFCavity:
    return RFCavity(VOLTAGE, FREQ_HZ, PHI_S)


@pytest.mark.parametrize("delta", [0.0, 1e-4, 1e-3, 5e-3, -3e-3])
def test_the_cavity_is_xtracks_cavity_to_the_last_bit(cav, ref, delta) -> None:
    """accsim's cavity and xtrack's, on the same state, at five momenta.

    A thin element, so the two codes share no drift model that could disagree — which is
    what makes bit-identical the right expectation for five of the six coordinates rather
    than an aspiration. ``delta`` carries xtrack's own ``sqrt(...) - 1`` rounding, so it is
    gated at ``1e-15`` absolute (measured ``6e-18``-``2.6e-16``); the control below is what
    establishes that this is a floor and not a shared error.
    """
    state = STATE.copy()
    state[DELTA] = delta
    ours = cav.track(state, ref)
    theirs = _track_xtrack(state)

    # px, py must be untouched by both -- they are normalised to the reference P0, which a
    # thin cavity does not move. This is the assertion that would catch a build with
    # XTRACK_CAVITY_PRESERVE_ANGLE defined, where xtrack rescales them instead.
    transverse = [0, 1, 2, 3, ZETA]
    assert np.array_equal(ours[transverse], state[transverse])
    assert np.array_equal(theirs[transverse], state[transverse])
    assert abs(ours[DELTA] - theirs[DELTA]) < 1e-15


def test_the_pre_p2_momentum_kick_missed_xtrack_linearly_in_delta(cav, ref) -> None:
    r"""The control: the agreement above is a result, not two codes sharing a floor.

    The map accsim shipped until P2 (iii) misses xtrack's tracked ``delta`` by an amount
    **linear in ``delta``** — ``1.7e-14``, ``1.7e-13``, ``8.3e-13`` at ``1e-4``, ``1e-3``,
    ``5e-3``, a clean factor of ten per decade, which is ``d^2 delta/d p_zeta^2 =
    -1/gamma0^2`` read straight off. The largest is 3200x the ``2.6e-16`` the energy kick
    reaches, and the *shape* is the discriminator: a rounding floor has no trend in
    ``delta``, and this has nothing but.
    """

    def miss(delta: float) -> float:
        state = STATE.copy()
        state[DELTA] = delta
        return abs(_linearised(cav, state, ref)[DELTA] - _track_xtrack(state)[DELTA])

    small, mid, big = miss(1e-4), miss(1e-3), miss(5e-3)
    assert big > 1e-13  # decisively above xtrack's own 2.6e-16 rounding
    assert mid / small == pytest.approx(10.0, rel=0.05)
    assert big / mid == pytest.approx(5.0, rel=0.05)
    # ... and at delta = 0 the conversion has nothing left to get wrong, so the old map
    # falls back to xtrack's floor rather than to a smaller version of the same error.
    assert miss(0.0) < 1e-15


def test_the_two_codes_reference_the_synchronous_phase_differently(ref) -> None:
    r"""The ``- sin(phi_s)`` offset, gated in **energy** as exactly ``q V sin(phi_s)``.

    This is a Stage-3 convention, not something P2 (iii) introduced, and it is the reason
    every gate above runs at ``phi_s = 0``. accsim's kick is measured relative to a
    *ramping reference*, xtrack's is the lab kick, so the two differ by the per-turn energy
    the reference absorbs — a constant, independent of ``delta`` and of ``zeta``.

    Asserting it in energy rather than in ``delta`` is the point, and it is asserted as a
    *comparison of spreads* rather than against a remembered number: across
    ``delta = 0 ... 5e-3`` the offset read in energy is constant to ``4e-13``, while the
    same offset read in ``delta`` varies by ``delta_max/gamma0^2`` — the milestone's own
    coefficient, `1.25e-5` here, ten million times the energy spread. Reading a frame
    convention in the wrong variable would have made a clean constant look like a drift.
    """
    cav = RFCavity(VOLTAGE, FREQ_HZ, PHI_S_OFFSET)
    want = ref.charge * VOLTAGE * math.sin(PHI_S_OFFSET)
    deltas = (0.0, 1e-3, 5e-3)
    in_energy, in_delta = [], []
    for delta in deltas:
        state = STATE.copy()
        state[DELTA] = delta
        ours = cav.track(state, ref)[DELTA]
        theirs = _track_xtrack(state, phase=PHI_S_OFFSET)[DELTA]
        in_energy.append(_energy_eV(theirs, ref) - _energy_eV(ours, ref))
        in_delta.append(theirs - ours)
        assert in_energy[-1] == pytest.approx(want, rel=1e-11)

    def spread(values: list[float]) -> float:
        return (max(values) - min(values)) / abs(values[0])

    assert spread(in_energy) < 1e-11  # a constant, as a frame offset must be
    assert spread(in_delta) == pytest.approx(max(deltas) / ref.gamma0**2, rel=0.05)
    assert spread(in_delta) > 1e6 * spread(in_energy)


def test_the_correspondence_still_needs_half_a_turn_of_phase_for_an_electron() -> None:
    r"""The charge-sign trap, re-asserted on the new map.

    xtrack's ``q`` is ``fabs(q0) * charge_ratio``; accsim's is ``ref.charge``, signed. So
    for an electron the two cavities are exact negatives at the same ``phase``, and the
    correspondence is ``phase = phi_s + pi``. P2 (iii) did not touch this — the amplitude
    is what is unchanged and the conversion is what moved — but a file that gated the
    cavity only on a proton would not notice if it had.
    """
    ref = ReferenceParticle.from_gamma(ELECTRON_MASS0, ELECTRON_GAMMA0, charge=-1.0)
    ours = RFCavity(VOLTAGE, FREQ_HZ, PHI_S).track(STATE, ref)

    args = (ELECTRON_MASS0, ELECTRON_GAMMA0, -1)
    shifted = _track_xtrack(STATE, PHI_S + math.pi, *args)
    assert np.max(np.abs(ours - shifted)) < 1e-15

    naive = _track_xtrack(STATE, PHI_S, *args)
    # Not a small discrepancy: the kick is the opposite sign, so the two differ by twice it.
    assert naive[DELTA] - STATE[DELTA] == pytest.approx(-(ours[DELTA] - STATE[DELTA]), rel=1e-6)
