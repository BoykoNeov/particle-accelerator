"""I1 cross-check: xtrack's own closed-orbit search, and the corrector **sign**.

The analytic suite gates the magnitude and the shape of the closed orbit — the
``sqrt(beta_k beta)/(2 sin pi Q)`` form, superposition, the resonance — but it
cannot gate the *sign convention* of a corrector, because every reference it
could use is one accsim also derives. A self-confirming sign gate is precisely
the trap G1 fell into (its vertical-emittance coefficient was pre-committed and
wrong, and xtrack corrected it), so the sign lives here and only here.

**Established empirically, not recalled** (probe run 2026-08-10, xtrack via
clang-cl):

    accsim  Corrector(kick_x = +k)   ==   xt.Multipole(knl = [-k])
    accsim  Corrector(kick_y = +k)   ==   xt.Multipole(ksl = [+k])

The asymmetry is real and is the MAD-X multipole convention: ``knl[0]`` is the
*normal* dipole component and carries the bend sign (``px -= knl[0]``), while
``ksl[0]`` is the skew one (``py += ksl[0]``). Anyone tempted to "fix" the minus
sign should read :func:`test_the_horizontal_sign_gate_is_not_vacuous`, which
asserts the other choice is decisively wrong rather than merely different.

What this reaches that the analytic suite cannot:

- the sign, as above;
- **an independent closed-orbit search.** accsim solves ``(I - M4) x = k4`` in
  closed form; xtrack finds the closed orbit by iterating a real tracker. Two
  different algorithms agreeing to 1.9e-15 m on a 1 mm orbit is a statement about
  the affine map, not about linear algebra;
- **the correction actually steers the machine.** The last test hands xtrack the
  kicks ``correct_orbit`` chose and asks it, independently, where the beam now
  goes.

Thick quadrupoles are used deliberately (not the analytic suite's thin ones), so
the element maps being composed are the non-trivial ones.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import (
    Corrector,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    correct_orbit,
    propagate_orbit,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
N_CELLS = 3
KICK = 2.0e-4  # steerer angle [rad]

# Measured 2026-08-10: the two codes' closed orbits agree to 1.9e-15 m absolute
# (1.6e-12 relative) on a ~1 mm orbit. The floor is xtrack's *iterative*
# closed-orbit search, not accsim's closed-form solve — accsim's own residual is
# exact — so this is a tolerance on the reference, with ~50x headroom.
ORBIT_ATOL = 1e-13


def _accsim_cell() -> list:
    return [Quadrupole(LQ, K1), Drift(LD), Quadrupole(LQ, -K1), Drift(LD)]


def _xtrack_cell() -> list:
    return [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Drift(length=LD),
    ]


def _xtrack_twiss(correctors: list):
    """Twiss a ring of ``N_CELLS`` cells followed by the given thin ``correctors``."""
    line = xt.Line(elements=_xtrack_cell() * N_CELLS + correctors)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def _accsim_orbit(correctors: list) -> tuple[np.ndarray, np.ndarray]:
    """accsim's closed orbit ``(x, y)`` at every boundary of the same ring."""
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    lat = Lattice(_accsim_cell() * N_CELLS + correctors, ref)
    table = propagate_orbit(lat)
    return np.array([o[0] for o in table]), np.array([o[2] for o in table])


def test_horizontal_closed_orbit_matches_xtrack() -> None:
    """Sign, magnitude and shape at every boundary, against an independent search.

    ``knl=[-KICK]`` is the empirically established equivalent of
    ``Corrector(kick_x=+KICK)``. Both codes report the orbit at the same 14
    points; agreement is at the level of the closed-orbit searches themselves.
    """
    x_acc, _ = _accsim_orbit([Corrector(kick_x=KICK)])
    tw = _xtrack_twiss([xt.Multipole(knl=[-KICK])])
    x_xt = np.array(tw.x)
    assert x_acc.shape == x_xt.shape
    assert np.abs(x_acc).max() > 5e-4  # a real orbit, ~1 mm, not round-off
    assert np.allclose(x_acc, x_xt, rtol=0, atol=ORBIT_ATOL)


def test_vertical_closed_orbit_matches_xtrack() -> None:
    """The vertical sign is the *opposite* convention, and that is not a typo.

    ``ksl[0]`` is the skew dipole component: ``Corrector(kick_y=+k)`` matches
    ``ksl=[+k]``, where the horizontal matched ``knl=[-k]``.
    """
    _, y_acc = _accsim_orbit([Corrector(kick_y=KICK)])
    tw = _xtrack_twiss([xt.Multipole(knl=[0.0], ksl=[KICK])])
    y_xt = np.array(tw.y)
    assert np.abs(y_acc).max() > 5e-4
    assert np.allclose(y_acc, y_xt, rtol=0, atol=ORBIT_ATOL)


def test_the_horizontal_sign_gate_is_not_vacuous() -> None:
    """The other sign choice is decisively wrong, not merely a different phase.

    Guards the gate above: if the orbit happened to be (anti)symmetric, matching
    would not have distinguished the conventions. It does — the wrong sign gives
    exactly the negated orbit, a ~2 mm error.
    """
    x_acc, _ = _accsim_orbit([Corrector(kick_x=KICK)])
    x_wrong = np.array(_xtrack_twiss([xt.Multipole(knl=[+KICK])]).x)
    assert np.allclose(x_acc, -x_wrong, rtol=0, atol=ORBIT_ATOL)
    assert np.abs(x_acc - x_wrong).max() > 1e-3


def test_both_planes_at_once_stay_decoupled_in_xtrack_too() -> None:
    """One corrector kicking both planes: xtrack sees the same two orbits.

    Confirms the plane bookkeeping (which the analytic suite can only check
    against itself) with an outside code, and that neither plane leaks.
    """
    x_acc, y_acc = _accsim_orbit([Corrector(kick_x=KICK, kick_y=-1.4e-4)])
    tw = _xtrack_twiss([xt.Multipole(knl=[-KICK], ksl=[-1.4e-4])])
    assert np.allclose(x_acc, np.array(tw.x), rtol=0, atol=ORBIT_ATOL)
    assert np.allclose(y_acc, np.array(tw.y), rtol=0, atol=ORBIT_ATOL)


def test_xtrack_confirms_the_corrected_machine_is_steered() -> None:
    """The milestone's point, checked by an outside code end to end.

    accsim is given a steering error and two correctors downstream of it, and
    solves for the kicks that null the orbit outside the resulting bump. xtrack
    is then handed those kicks — not the request, the *answer* — and asked where
    the beam goes. It must find the orbit corrected outside the bump and still
    displaced inside it, since no corrector can undo what happened upstream.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    err = Corrector(kick_x=KICK, name="err")
    ca, cb = Corrector(name="ca"), Corrector(name="cb")
    # error | 1 cell | ca | 1 cell | cb | 1 cell
    elems = [err, *_accsim_cell(), ca, *_accsim_cell(), cb, *_accsim_cell()]
    lat = Lattice(elems, ref)

    n = len(lat)
    ib = next(i for i, e in enumerate(lat.elements) if e is cb)
    outside = list(range(ib + 1, n + 1))  # strictly downstream of the last corrector
    inside = list(range(1, ib + 1))
    res = correct_orbit(lat, [ca, cb], outside, "x")
    assert res.rms_before > 1e-4
    assert res.rms_after < 1e-15  # accsim's own view first

    xline = [
        xt.Multipole(knl=[-KICK]),
        *_xtrack_cell(),
        xt.Multipole(knl=[-ca.kick_x]),
        *_xtrack_cell(),
        xt.Multipole(knl=[-cb.kick_x]),
        *_xtrack_cell(),
    ]
    line = xt.Line(elements=xline)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        tw = line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")

    x_xt = np.array(tw.x)
    assert np.abs(x_xt[outside]).max() < ORBIT_ATOL  # steered flat, per xtrack
    assert np.abs(x_xt[inside]).max() > 1e-4  # and the bump is still there
    x_acc = np.array([o[0] for o in propagate_orbit(lat)])
    assert np.allclose(x_acc, x_xt, rtol=0, atol=ORBIT_ATOL)
