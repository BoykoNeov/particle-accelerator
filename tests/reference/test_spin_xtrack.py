r"""Cross-check the tracked Thomas-BMT spin precession (N1) against xtrack.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**Three things have to be set up before the comparison means anything**, and each of them
silently returns a plausible wrong answer if it is not:

1. ``line.configure_spin("auto")``. Without it xtrack's kernel is compiled with spin
   **off** and ``track()`` returns the spin **exactly unchanged** — no error, no warning.
   A comparison written without it measures nothing, and would read as "accsim invented a
   precession xtrack does not have".
2. ``anomalous_magnetic_moment`` on the particle. ``xt.Particles`` defaults it to ``0``,
   which is not "spin physics off" but the *cyclotron* rotation — a spin tune of exactly
   zero and a tracked spin answering a different question. That is M2's trap by another
   name: the reference's **default configuration** is not the physics being checked.
3. ``model="bend-kick-bend"`` on the bend, with ``integrator="uniform"`` and
   ``num_multipole_kicks=1`` (B2's argument, unchanged). xtrack's *default* bend
   integration is a fourth-order splitting whose one-kick design orbit is not quite the
   axis — see :func:`test_the_default_bend_splitting_moves_the_orbit_and_the_spin_follows`.

With all three set, the picture is sharper than N1 predicted. The milestone expected an
``O(L^2)`` gap everywhere, on the grounds that xtrack does not evaluate an analytic field
at all (``magnet_estimate_field`` back-derives ``B`` from the trajectory's curvature)
while accsim samples its own field at the traversal mid-point. What is actually there:

- on a **bend**, and on a quadrupole with only **one** transverse plane populated, the
  two agree to **round-off** at every slicing — because the precession axis does not move,
  so the rotations commute and only the scalar ``int b ds`` survives, and both codes'
  quadratures of that scalar are the same number;
- with **both** planes populated the axis turns, the rotations stop commuting, and the two
  lumped maps converge to each other as ``1/N^3`` — a factor 8 per doubling, gated as that
  order rather than as a tolerance.

The gap is **non-commutativity**, not the field model, and the single-plane exactness is
what proves it.

**What is left is one genuine defect in xtrack, and it is asserted rather than dodged.**
``direction_of_motion`` (``track_magnet_radiation.h:22``) computes
``sqrt(1 - ix*ix + iy*iy)`` — a ``+`` where a ``-`` belongs — so the vector it returns is
not a unit vector. The error enters the spin through ``b_par`` and is therefore **third**
order in ``py`` (one power from ``b . i``, two from the botched normalisation) and
**exactly zero** in ``px``. Both are gated, because the pair of exponents identifies the
mechanism where a tolerance would only record a size.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import Dipole, ReferenceParticle
from accsim.elements.quadrupole import Quadrupole
from accsim.reference import ELECTRON_ANOMALOUS_MOMENT as G
from accsim.reference import ELECTRON_MASS_EV as MASS0

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

ENERGY = 5e9
P0C = math.sqrt(ENERGY**2 - MASS0**2)
LENGTH = 1.0
ANGLE = 0.2
K1 = 1.4

SPIN0 = np.array([0.2, 0.9, math.sqrt(1.0 - 0.04 - 0.81)])


def _ref(g: float = G) -> ReferenceParticle:
    return ReferenceParticle.from_total_energy(MASS0, ENERGY, anomalous_moment=g)


def _line(element, *, spin: bool = True, kicks: int = 1):
    """A one-magnet line set up so that its integration matches accsim's single kick."""
    line = xt.Line(elements=[element], element_names=["e"])
    line.particle_ref = xt.Particles(mass0=MASS0, p0c=P0C)
    if spin:
        # Without this the kernel is built with spin off and track() is a no-op on it.
        line.configure_spin("auto")
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    line["e"].integrator = "uniform"
    line["e"].num_multipole_kicks = kicks
    return line


def _xtrack(line, state: np.ndarray, spin: np.ndarray, g: float = G):
    p = xt.Particles(
        mass0=MASS0,
        p0c=P0C,
        x=state[0],
        px=state[1],
        y=state[2],
        py=state[3],
        zeta=state[4],
        delta=state[5],
        anomalous_magnetic_moment=g,
        spin_x=spin[0],
        spin_y=spin[1],
        spin_z=spin[2],
    )
    line.track(p)
    return (
        np.array([p.x[0], p.px[0], p.y[0], p.py[0], p.zeta[0], p.delta[0]]),
        np.array([p.spin_x[0], p.spin_y[0], p.spin_z[0]]),
    )


@pytest.fixture(scope="module")
def bend_line():
    """A sector bend whose map is exact, so the comparison is about the spin."""
    return _line(xt.Bend(length=LENGTH, angle=ANGLE, k0=ANGLE / LENGTH, model="bend-kick-bend"))


@pytest.fixture(scope="module")
def quad_line():
    return _line(xt.Quadrupole(length=LENGTH, k1=K1))


# --- the two codes, once the comparison is set up ----------------------------------


@pytest.mark.parametrize(
    "state",
    [
        np.zeros(6),
        np.array([2e-3, 1e-3, 0.0, 0.0, 0.0, 0.0]),
        np.array([-1e-3, 5e-4, 2e-3, 0.0, 0.0, 1e-3]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2e-3]),
    ],
)
def test_a_bend_matches_xtrack_to_round_off_once_its_map_is_exact(bend_line, state):
    """Spin and orbit both, on a bend, at ``py = 0``.

    ``py`` is held at zero throughout because that is the one variable xtrack's
    ``direction_of_motion`` typo rides on; it gets its own test below, where the
    disagreement is asserted with its **order** rather than removed.
    """
    xt_state, xt_spin = _xtrack(bend_line, state, SPIN0)
    ac_state, ac_spin = Dipole(LENGTH, ANGLE).track_with_spin(state, SPIN0, _ref())
    np.testing.assert_allclose(ac_state, xt_state, atol=1e-14)
    np.testing.assert_allclose(ac_spin, xt_spin, atol=1e-13)


def test_a_bend_reproduces_the_closed_form_where_xtracks_default_does_not(bend_line):
    r"""On the design orbit both codes land on ``-G gamma theta``, the closed form.

    This is the three-way statement the milestone is after: accsim, xtrack's exact bend
    map, and a number derived from the Thomas-BMT equation rather than from either code.
    """
    expected = -G * _ref().gamma0 * ANGLE
    _, xt_spin = _xtrack(bend_line, np.zeros(6), np.array([1.0, 0.0, 0.0]))
    _, ac_spin = Dipole(LENGTH, ANGLE).track_with_spin(
        np.zeros(6), np.array([1.0, 0.0, 0.0]), _ref()
    )
    for spin in (ac_spin, xt_spin):
        angle = math.atan2(-spin[2], spin[0])
        # remainder(), not %: the two differ by a full turn here and a modulo would
        # report 2 pi rather than 0.
        assert math.remainder(angle - expected, 2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)


def _quadrupole_spin_gap(quad_line, state: np.ndarray, slices: int) -> tuple[float, float]:
    """``(spin gap, orbit gap)`` between the codes for a quadrupole cut ``slices`` ways.

    The two slice *differently* — accsim by chaining shorter elements, xtrack by
    ``num_multipole_kicks`` inside one — so agreement across a sweep is a stronger check
    than agreement at any single value.
    """
    quad_line["e"].num_multipole_kicks = slices
    xt_state, xt_spin = _xtrack(quad_line, state, SPIN0)
    ac_state, ac_spin = state.copy(), SPIN0.copy()
    for _ in range(slices):
        ac_state, ac_spin = Quadrupole(LENGTH / slices, K1).track_with_spin(
            ac_state, ac_spin, _ref()
        )
    return (
        float(np.max(np.abs(ac_spin - xt_spin))),
        float(np.max(np.abs(ac_state - xt_state))),
    )


@pytest.mark.parametrize(
    ("label", "state"),
    [
        ("horizontal", np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 5e-4])),
        ("vertical", np.array([0.0, 0.0, -1.5e-3, 0.0, 0.0, 5e-4])),
    ],
)
@pytest.mark.parametrize("slices", [1, 4])
def test_a_quadrupole_matches_xtrack_exactly_in_a_single_plane(quad_line, label, state, slices):
    r"""With only one transverse plane populated the two codes agree to **round-off**.

    Which is more than either construction promises on its own: accsim samples its
    **analytic** field at the traversal mid-point, xtrack back-derives ``B`` from the
    trajectory's own curvature (``magnet_estimate_field``), and those are different
    recipes. They coincide here for a reason that is derivable rather than lucky.

    In a single plane the precession **axis does not move**. With ``y = py = 0`` the
    field is ``b = (0, k1 x, 0)`` and ``b . i = b_y i_y = 0``, so ``Omega`` points along
    ``-y`` for the whole traversal; with ``x = px = 0`` it points along ``-x``. Every
    rotation therefore commutes with every other, only the **scalar** ``int b ds``
    survives, and both codes' quadratures of that scalar are the same number.

    That is exactly why it stops being true when both planes are populated — see the
    next test, where the axis turns and the order drops to third.
    """
    spin_gap, orbit_gap = _quadrupole_spin_gap(quad_line, state, slices)
    assert orbit_gap < 1e-14
    assert spin_gap < 1e-14


def test_a_quadrupole_in_both_planes_converges_at_third_order(quad_line):
    r"""Populate both planes and the axis turns, so the two lumped maps part company.

    ``Omega`` now has an ``x`` and a ``y`` component whose *ratio* changes along the
    path, so the rotations no longer commute and the answer depends on how the field is
    resolved along the way — which is where accsim's analytic mid-point sample and
    xtrack's curvature back-derivation are genuinely different recipes. They converge to
    each other as ``1/N^3``: a factor **8** per doubling, asserted as that order rather
    than as a tolerance at one slicing (B4's argument).

    The orbit stays identical throughout, which is what makes this a statement about the
    spin and not about the map.
    """
    state = np.array([1e-3, 0.0, -1.5e-3, 0.0, 0.0, 5e-4])
    gaps = [_quadrupole_spin_gap(quad_line, state, n) for n in (1, 2, 4)]
    assert max(orbit for _, orbit in gaps) < 1e-14
    spins = [spin for spin, _ in gaps]
    for coarse, fine in zip(spins, spins[1:], strict=False):
        assert coarse / fine == pytest.approx(8.0, rel=0.05)


# --- the traps, asserted so they cannot be re-set silently -------------------------


def test_xtrack_leaves_the_spin_exactly_unchanged_unless_spin_is_configured():
    """The first trap, and the one that would have made this whole file vacuous.

    ``configure_spin`` is what puts spin into the compiled kernel. Without it the
    tracked spin comes back **bit for bit** as it went in — for a magnet that certainly
    precesses it, with no error raised. Asserted with ``==``, because "unchanged" here
    is exact and the point is that it is silent.
    """
    line = _line(
        xt.Bend(length=LENGTH, angle=ANGLE, k0=ANGLE / LENGTH, model="bend-kick-bend"),
        spin=False,
    )
    _, xt_spin = _xtrack(line, np.zeros(6), SPIN0)
    np.testing.assert_array_equal(xt_spin, SPIN0)


def test_xtracks_anomalous_moment_defaults_to_zero(bend_line):
    """The second trap: the default is the Dirac particle, not the electron.

    Left unset, ``xt.Particles`` gives ``G = 0``, so the spin merely follows the
    momentum and a flat ring's spin tune is exactly zero. On the design orbit that means
    the spin comes back **unrotated** — which looks like a working simulation and is a
    different question from the one being asked. accsim refuses to guess instead
    (``accsim.spin.anomalous_moment`` raises on an unset moment).
    """
    _, xt_spin = _xtrack(bend_line, np.zeros(6), SPIN0, g=0.0)
    np.testing.assert_allclose(xt_spin, SPIN0, atol=1e-15)


@pytest.mark.parametrize("angle", [0.1, 0.2, 0.4])
def test_the_default_bend_splitting_moves_the_orbit_and_the_spin_follows(angle):
    r"""xtrack's *default* bend is a fourth-order splitting, and its spin is honest.

    An apparent ``1.4e-5`` spin disagreement on a bend looks alarming and is not a spin
    disagreement at all. With ``G = 0`` a sector bend must leave a design-orbit spin
    exactly alone, and ``model="bend-kick-bend"`` does — but the default integration
    leaves the *design particle itself* slightly off axis, and the spin then correctly
    follows that slightly wrong momentum. The residual is the **orbit's**:

    - it equals ``theta`` times the orbit residual, to three digits;
    - it is ``O(theta^5)`` (a factor 32 per doubling of the bend angle);
    - it is ``O(N^-4)`` in the number of kicks (a factor 16 per doubling) — the
      signature of a fourth-order splitting;
    - and it is independent of the element length and of the beam energy.

    Localise before deriving (M2's lesson): the exponents name the mechanism, and no
    tolerance on the spin alone could have distinguished "xtrack's spin is wrong" from
    "xtrack's orbit is approximate and its spin is right".
    """
    line = _line(xt.Bend(length=LENGTH, angle=angle, k0=angle / LENGTH))  # default model
    spin_residual, orbit_residual = [], []
    for kicks in (1, 2, 4):
        line["e"].num_multipole_kicks = kicks
        state, spin = _xtrack(line, np.zeros(6), np.array([1.0, 0.0, 0.0]), g=0.0)
        spin_residual.append(abs(math.atan2(-spin[2], spin[0])))
        orbit_residual.append(float(np.max(np.abs(state))))

    # the spin is doing exactly what the orbit does
    assert spin_residual[0] / orbit_residual[0] == pytest.approx(angle, rel=1e-3)
    # ...and both are a fourth-order splitting's remainder
    for coarse, fine in zip(spin_residual, spin_residual[1:], strict=False):
        assert coarse / fine == pytest.approx(16.0, rel=0.05)
    # accsim, meanwhile, is exactly zero here
    _, ac_spin = Dipole(LENGTH, angle).track_with_spin(
        np.zeros(6), np.array([1.0, 0.0, 0.0]), _ref(g=0.0)
    )
    assert abs(math.atan2(-ac_spin[2], ac_spin[0])) < 1e-15


def test_the_default_bend_splittings_spin_residual_is_fifth_order_in_the_angle():
    """The other exponent, which pins it to the bend angle rather than to the field."""
    residuals = []
    for angle in (0.05, 0.1, 0.2, 0.4):
        line = _line(xt.Bend(length=LENGTH, angle=angle, k0=angle / LENGTH))
        _, spin = _xtrack(line, np.zeros(6), np.array([1.0, 0.0, 0.0]), g=0.0)
        residuals.append(abs(math.atan2(-spin[2], spin[0])))
    for coarse, fine in zip(residuals[1:], residuals[:-1], strict=False):
        assert coarse / fine == pytest.approx(32.0, rel=0.05)


# --- xtrack's non-unit direction of motion ----------------------------------------


def test_the_unit_vector_typo_is_third_order_in_py(bend_line):
    r"""``sqrt(1 - ix*ix + iy*iy)`` costs xtrack a spin error growing as ``py^3``.

    The mechanism, and why that exponent and not another: the returned vector is long by
    ``iy^2``, and it multiplies ``b . i``, which for a bend is ``b_y iy`` and so already
    carries one power of ``py``. One from the projection, two from the normalisation.

    The orbit is untouched — the typo lives only in the radiation/spin helper — so this
    is a pure spin statement, and it is asserted rather than avoided by tracking at
    ``py = 0``: a disagreement with a *derived* exponent is a finding, and a disagreement
    hidden by a choice of test point is a gap.
    """
    residuals, orbit = [], []
    for py in (5e-4, 1e-3, 2e-3, 4e-3):
        state = np.array([0.0, 0.0, 0.0, py, 0.0, 0.0])
        xt_state, xt_spin = _xtrack(bend_line, state, SPIN0)
        ac_state, ac_spin = Dipole(LENGTH, ANGLE).track_with_spin(state, SPIN0, _ref())
        residuals.append(float(np.max(np.abs(xt_spin - ac_spin))))
        orbit.append(float(np.max(np.abs(xt_state - ac_state))))

    assert max(orbit) < 1e-14  # the orbit is not implicated
    for small, large in zip(residuals, residuals[1:], strict=False):
        assert large / small == pytest.approx(8.0, rel=0.02)  # py^3


def test_the_unit_vector_typo_is_exactly_absent_in_px(bend_line):
    """And the other half of the mechanism: with ``py = 0`` there is nothing to spoil.

    ``b . i`` for a bend is ``b_y iy``, so a purely horizontal angle never reaches the
    botched component at all. Together with the ``py^3`` law above this identifies the
    line of C, which a single tolerance could not.
    """
    for px in (1e-3, 4e-3):
        state = np.array([0.0, px, 0.0, 0.0, 0.0, 0.0])
        _, xt_spin = _xtrack(bend_line, state, SPIN0)
        _, ac_spin = Dipole(LENGTH, ANGLE).track_with_spin(state, SPIN0, _ref())
        assert float(np.max(np.abs(xt_spin - ac_spin))) < 1e-14
