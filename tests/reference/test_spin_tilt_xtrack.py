r"""V3 — spin through a tilted bend, against xtrack (the one arbiter).

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

MAD-X has no spin, so xtrack is the only outside code — which is why the sharpest gates in
``tests/analytic/test_spin_tilt.py`` need no arbiter at all. What this file adds is the
outside **sense**: xtrack's ``rot_s_rad`` reaches the spin through ``SRotation``, which turns
``spin_x``/``spin_y`` with exactly the formula it turns ``x``/``y`` with
(``track_srotation.h``), and V1 already pinned that coordinate sense to accsim's.

Every silent switch N1-N4 found is set: ``configure_spin``, an explicit anomalous moment, an
explicit ``q0``, ``Drift(model="exact")``, and ``model="bend-kick-bend"`` with one uniform
kick.

**The element: round-off in the bend's own plane, xtrack's typo across it.** N1 found
xtrack's ``direction_of_motion`` writes ``sqrt(1 - ix*ix + iy*iy)``, a spin error third
order in ``py`` and exactly zero in ``px``. Through a tilted bend those are the bend's
**own** ``py`` and ``px`` — so a particle moving in the bend's plane agrees to round-off at
every tilt, and one moving across it shows the ``py^3`` law. That the typo follows the
tilted frame is itself the evidence that xtrack turns the spin with the magnet.

**The ring: the one-turn rotation agrees everywhere, xtrack's own n_0 search does not.**
Tracked through xtrack, the rotator ring's one-turn spin rotation matches accsim's to
``1e-14`` at every rotator strength tried. ``twiss(spin=True)`` agrees along the ring at a
30-degree lean and **fails** (``LinAlgError``) at 60 degrees: its fixed-point search varies
``s_x`` and ``s_z`` inside independent ``(-1, 1)`` boxes and sets ``s_y = sqrt(1 - s_x^2 -
s_z^2)`` (``twiss.py``, ``_find_spin_fixed_point``), and on this ring it steps where that is
negative. Measured: fine at 15 and 30 degrees, failing at 45, 54, 60, 72 and 82. The onset
between 30 and 45 was not swept.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from accsim import Dipole
from accsim.elements.drift import Drift
from accsim.elements.quadrupole import Quadrupole
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G
from accsim.reference import ELECTRON_MASS_EV as MASS0
from accsim.spin import closed_spin_solution, propagate_spin_solution, spin_axis_and_tune

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "analytic"))

from test_spin_tilt import TILTS, electron, g_gamma, rotator_ring  # noqa: E402

ENERGY = 3.0e9
P0C = math.sqrt(ENERGY**2 - MASS0**2)
LENGTH = 1.0
ANGLE = 0.2
SPIN0 = np.array([0.2, 0.9, math.sqrt(1.0 - 0.04 - 0.81)])


def _particles(state=None, spin=SPIN0) -> xt.Particles:
    state = np.zeros(6) if state is None else state
    return xt.Particles(
        mass0=MASS0,
        q0=-1.0,
        p0c=P0C,
        x=state[0],
        px=state[1],
        y=state[2],
        py=state[3],
        zeta=state[4],
        delta=state[5],
        anomalous_magnetic_moment=G,
        spin_x=spin[0],
        spin_y=spin[1],
        spin_z=spin[2],
    )


def _built(line: xt.Line) -> xt.Line:
    line.particle_ref = xt.Particles(mass0=MASS0, q0=-1.0, p0c=P0C, anomalous_magnetic_moment=G)
    # Without this the kernel is compiled with spin off and track() is a no-op on it.
    line.configure_spin("auto")
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    for name in line.element_names:
        if isinstance(line[name], xt.Bend | xt.Quadrupole):
            line[name].integrator = "uniform"
            line[name].num_multipole_kicks = 1
    return line


def _track(line: xt.Line, state: np.ndarray, spin: np.ndarray = SPIN0):
    p = _particles(state, spin)
    line.track(p)
    return (
        np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]]),
        np.array([p.spin_x[0], p.spin_y[0], p.spin_z[0]]),
    )


@pytest.fixture(scope="module")
def bend_line():
    """One tilted bend; each test sets ``rot_s_rad`` (no recompile needed)."""
    bend = xt.Bend(length=LENGTH, angle=ANGLE, k0=ANGLE / LENGTH, model="bend-kick-bend")
    return _built(xt.Line(elements=[bend], element_names=["e"]))


# --- one tilted bend ---------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.0, *TILTS])
def test_in_the_bends_own_plane_the_codes_agree_to_round_off(bend_line, tilt: float):
    """Orbit and spin, for a particle whose transverse motion lies in the bend's plane.

    In the bend's frame its ``py`` is zero, so xtrack's typo has nothing to act on and N1's
    round-off agreement must survive the tilt. The opposite sense is a different axis and is
    missed by order one (where the tilt is not its own opposite).
    """
    bend_line["e"].rot_s_rad = tilt
    c, s = math.cos(tilt), math.sin(tilt)
    ref = electron()
    for amplitude in (0.0, 1e-3, 2e-3):
        state = np.array([1e-3 * c, amplitude * c, 1e-3 * s, amplitude * s, 0.0, 5e-4])
        xt_state, xt_spin = _track(bend_line, state)
        ac_state, ac_spin = Dipole(LENGTH, ANGLE, tilt=tilt).track_with_spin(state, SPIN0, ref)
        assert np.abs(ac_state - xt_state).max() < 1e-14
        assert np.abs(ac_spin - xt_spin).max() < 1e-14
        if abs(s) > 1e-12:
            _, opposite = Dipole(LENGTH, ANGLE, tilt=-tilt).track_with_spin(state, SPIN0, ref)
            assert np.abs(opposite - xt_spin).max() > 0.1


@pytest.mark.parametrize("tilt", [0.0, *TILTS])
def test_across_the_bends_plane_xtracks_typo_follows_the_tilted_frame(bend_line, tilt: float):
    """N1's ``py^3`` law, in the bend's **own** ``py``, at every tilt.

    The motion is set perpendicular to the bend's plane, so its body-frame ``py`` is the
    whole amplitude and its body ``px`` is zero. A factor 8 per doubling identifies the
    mechanism; it could not appear in the bend's frame unless xtrack turned the spin into
    that frame along with the coordinates.
    """
    bend_line["e"].rot_s_rad = tilt
    c, s = math.cos(tilt), math.sin(tilt)
    ref = electron()
    residuals, orbit = [], []
    for amplitude in (5e-4, 1e-3, 2e-3, 4e-3):
        state = np.array([0.0, -amplitude * s, 0.0, amplitude * c, 0.0, 0.0])
        xt_state, xt_spin = _track(bend_line, state)
        ac_state, ac_spin = Dipole(LENGTH, ANGLE, tilt=tilt).track_with_spin(state, SPIN0, ref)
        residuals.append(float(np.abs(xt_spin - ac_spin).max()))
        orbit.append(float(np.abs(xt_state - ac_state).max()))

    assert max(orbit) < 1e-14  # the orbit is not implicated
    for small, large in zip(residuals, residuals[1:], strict=False):
        assert large / small == pytest.approx(8.0, rel=0.02)


# --- the rotator ring ----------------------------------------------------------------------


def _twin(element):
    if isinstance(element, Drift):
        return xt.Drift(length=element.length, model="exact")
    if isinstance(element, Quadrupole):
        return xt.Quadrupole(length=element.length, k1=element.k1, rot_s_rad=element.roll)
    if isinstance(element, Dipole):
        return xt.Bend(
            length=element.length,
            angle=element.angle,
            k0=element.angle / element.length,
            model="bend-kick-bend",
            rot_s_rad=element.tilt,
        )
    raise AssertionError(f"no xtrack twin wired up for {type(element).__name__}")


@pytest.fixture(scope="module")
def ring_line():
    """The rotator ring; :func:`_set_rotator` retunes its two vertical bends in place."""
    lattice = rotator_ring(math.pi / 6)
    names = [f"e{i}" for i in range(len(lattice.elements))]
    return _built(xt.Line(elements=[_twin(e) for e in lattice.elements], element_names=names))


def _set_rotator(line: xt.Line, phi: float):
    lattice = rotator_ring(phi)
    for i in (1, len(lattice.elements) - 2):
        line[f"e{i}"].angle = lattice.elements[i].angle
        line[f"e{i}"].k0 = lattice.elements[i].angle / lattice.elements[i].length
    return lattice


@pytest.mark.parametrize("frac", [1 / 12, 1 / 6, 1 / 4, 1 / 3, 0.4])
def test_the_one_turn_spin_rotation_matches_xtracks_tracking(ring_line, frac: float):
    """accsim's exact 3x3 against three basis spins tracked once round in xtrack.

    The design orbit is closed in both codes, so no differencing and no fixed-point search
    is involved on either side: both matrices are exact up to round-off.
    """
    lattice = _set_rotator(ring_line, frac * math.pi)
    columns = [_track(ring_line, np.zeros(6), e)[1] for e in np.eye(3)]
    theirs = np.array(columns).T
    ours = closed_spin_solution(lattice).one_turn_matrix

    assert np.abs(ours - theirs).max() < 1e-12
    n0, nu = spin_axis_and_tune(theirs)
    phi = frac * math.pi
    assert np.abs(n0 - np.array([0.0, math.cos(phi), -math.sin(phi)])).max() < 1e-13
    assert nu == pytest.approx(g_gamma(lattice.ref) % 1.0, abs=1e-12)


def test_n0_agrees_with_xtracks_twiss_along_the_ring_at_thirty_degrees(ring_line):
    """``propagate_spin_solution`` against ``twiss(spin=True)``, element by element."""
    lattice = _set_rotator(ring_line, math.pi / 6)
    tw = ring_line.twiss(method="4d", spin=True)
    ours = np.array(propagate_spin_solution(lattice))
    theirs = np.array([tw.spin_x, tw.spin_y, tw.spin_z]).T
    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() < 1e-12


def test_xtracks_n0_search_fails_where_n0_leans_sixty_degrees(ring_line):
    """Recorded, not worked around: the arbiter's search, not either code's physics.

    The one-turn rotation above agrees at this strength, and accsim's ``n_0`` is the closed
    form, so the failure is in ``_find_spin_fixed_point``'s parametrisation. Written to fail
    the day xtrack's search handles a strongly leaning ``n_0`` — then compare it here too.
    """
    _set_rotator(ring_line, math.pi / 3)
    with pytest.raises(np.linalg.LinAlgError):
        ring_line.twiss(method="4d", spin=True)
