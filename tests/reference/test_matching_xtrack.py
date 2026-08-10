"""H1 cross-check: hand the *matched* lattice to xtrack and ask it for the tunes.

This deliberately does **not** compare accsim's matcher against xtrack's matcher —
that would compare two optimisers and tell us nothing about the physics. Instead
accsim solves for the strengths, and xtrack is asked, independently, what optics
those strengths actually produce. The matcher is right only if an outside code
agrees the machine now has the requested tunes and chromaticity.

What this reaches that the analytic suite cannot:

- **The integer part of the tune.** accsim accumulates phase advance through
  ``propagate_twiss``; xtrack twisses a real tracker. A matcher that hit the right
  *fractional* tune on the wrong integer would pass every analytic test in
  ``test_matching.py`` and fail here.
- **The optics model itself, not accsim's self-consistency.** The analytic gates
  prove the matcher solves accsim's own equations; only an outside code can say
  those equations describe the machine.

The ring is built as ``cell * N_CELLS``, which repeats the same element *objects*,
so one knob drives all ``N_CELLS`` placements and the response matrices must sum
every occurrence — the aliasing case, exercised end to end. Note this gate is not
what *catches* an aliasing bug: ``match_tunes`` uses an exact residual, so a
Jacobian off by a factor of ``N_CELLS`` still converges (just slower), and
``match_chromaticity`` would be caught by its own post-solve residual check. The
value here is the independent confirmation of where the machine ended up.

Measured residuals (2026-08-10, xtrack via clang-cl): tunes ``4.0e-10`` /
``1.1e-9``, chromaticity ``2.4e-3`` / ``3.6e-4``. The tune gate is the linear
optics both codes share, already cross-checked at the ~1e-6 level in Stage 1; the
chromaticity gate is looser for the reason ``test_sextupole_xtrack`` documents —
accsim's first-order ``beta·D_x`` feed-down against xtrack's real nonlinear kick
and finite-``delta`` chromaticity step. Both gates leave ~4x headroom.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import pytest

from accsim import (
    Dipole,
    Drift,
    Knob,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    chromaticity,
    match_chromaticity,
    match_tunes,
    tunes,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
LQ = 0.3  # quad length [m]
K1 = 1.2  # starting quad gradient [m^-2]
LD = 1.0  # drift length [m]
LB = 1.0  # dipole length [m]
ANG = 0.12  # dipole bend angle [rad] -> nonzero dispersion
LS = 0.2  # sextupole length [m]
N_CELLS = 3

Q_TARGET = (0.7400, 0.5900)  # full tunes, integer part included
XI_TARGET = (1.0, 1.0)  # slightly positive: head-tail stable, and not a lucky zero


def _matched_accsim() -> tuple[Lattice, dict[str, float]]:
    """Match the tunes, then the chromaticity; return the lattice and its strengths.

    ``cell * N_CELLS`` repeats the same objects, so each knob drives every
    placement of its element — the aliasing case the response matrices must sum.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    qf = Quadrupole(LQ, K1, name="qf")
    qd = Quadrupole(LQ, -K1, name="qd")
    sf = Sextupole(LS, 0.0, name="sf")
    sd = Sextupole(LS, 0.0, name="sd")
    cell = [qf, Drift(LD), sf, Dipole(LB, ANG), qd, sd, Dipole(LB, ANG), Drift(LD)]
    lat = Lattice(cell * N_CELLS, ref)

    match_tunes(lat, Q_TARGET, (Knob([qf]), Knob([qd], weights=[-1.0])))
    match_chromaticity(lat, XI_TARGET, (Knob([sf]), Knob([sd], weights=[-1.0])))

    # accsim's own view must be exact before xtrack is asked anything.
    assert tunes(lat) == pytest.approx(Q_TARGET, abs=1e-11)
    assert chromaticity(lat) == pytest.approx(XI_TARGET, abs=1e-11)
    return lat, {"k1f": qf.k1, "k1d": qd.k1, "k2f": sf.k2, "k2d": sd.k2}


def _xtrack_twiss(s: dict[str, float]):
    cell = [
        xt.Quadrupole(length=LQ, k1=s["k1f"]),
        xt.Drift(length=LD),
        xt.Sextupole(length=LS, k2=s["k2f"]),
        xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
        xt.Quadrupole(length=LQ, k1=s["k1d"]),
        xt.Sextupole(length=LS, k2=s["k2d"]),
        xt.Bend(length=LB, angle=ANG, k0=ANG / LB),
        xt.Drift(length=LD),
    ]
    line = xt.Line(elements=cell * N_CELLS)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        return line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")


def test_matched_tunes_confirmed_by_xtrack() -> None:
    """xtrack twisses the matched strengths and reports the requested tunes.

    Full tunes, so the integer part is part of the assertion. Observed ``4.0e-10``
    / ``1.1e-9`` — essentially exact agreement, since both codes are evaluating
    the same linear optics of the same thick elements.
    """
    _, strengths = _matched_accsim()
    tw = _xtrack_twiss(strengths)
    assert tw.qx == pytest.approx(Q_TARGET[0], abs=1e-8)
    assert tw.qy == pytest.approx(Q_TARGET[1], abs=1e-8)


def test_matched_chromaticity_confirmed_by_xtrack() -> None:
    """... and the requested chromaticity, from xtrack's own finite-delta derivative.

    Looser than the tune gate for the reason ``test_sextupole_xtrack`` documents:
    accsim's first-order ``beta·D_x`` feed-down vs. xtrack's real nonlinear
    sextupole kick and finite-``delta`` chromaticity step. Observed ``2.4e-3`` /
    ``3.6e-4`` against a natural chromaticity of ``(-0.797, -0.734)`` that the
    sextupoles pull up to ``+1``, i.e. a correction of ~1.8 delivered to ~1.3e-3
    relative — the expected size of that model difference, not a matching error.
    """
    _, strengths = _matched_accsim()
    tw = _xtrack_twiss(strengths)
    assert tw.dqx == pytest.approx(XI_TARGET[0], abs=1e-2)
    assert tw.dqy == pytest.approx(XI_TARGET[1], abs=1e-2)


def test_matching_moved_the_machine_somewhere_real() -> None:
    """Guard against a vacuous pass: the match must actually have changed things.

    If the starting lattice already sat on the targets, both gates above would
    pass without the matcher doing anything.
    """
    _, strengths = _matched_accsim()
    assert strengths["k1f"] != pytest.approx(K1, rel=1e-3)
    assert strengths["k2f"] != pytest.approx(0.0, abs=1e-3)
    assert strengths["k2d"] != pytest.approx(0.0, abs=1e-3)
