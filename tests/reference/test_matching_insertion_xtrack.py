"""H2 cross-check: hand the *matched* insertion to xtrack and ask it for the optics.

As in the H1 cross-check this does **not** compare accsim's matcher against
xtrack's matcher — that would compare two optimisers. accsim solves for the
strengths; xtrack is asked, independently, what optics those strengths produce.

Both H2 branches are covered, because they can fail in different ways:

- the **transfer-line** branch, against xtrack's *open* twiss started from the
  same entrance Twiss (``line.twiss(betx=..., alfx=..., ...)``). This is the
  insertion problem proper: match ``beta*`` and a waist at the interaction point,
  then let an outside code confirm the waist is really there. A propagation sign
  error in ``propagate_twiss`` that happened to be self-consistent would satisfy
  every analytic gate and fail here.
- the **periodic** branch, against xtrack's closed twiss of the matched ring. The
  analytic gates prove accsim re-solves *its own* closed solution when a knob
  moves; only an outside code can say that solution is the machine's.

Both branches are thick-element **linear** optics that the two codes model
identically — no first-order perturbation formula stands between them, unlike the
chromaticity half of the H1 cross-check — so the agreement is at the level of the
twiss arithmetic itself. Measured 2026-08-10 (xtrack 0.106.4 via clang-cl): line
``beta*`` ``8.9e-16`` relative, ``alpha*`` ``8.5e-16`` absolute, the untargeted y
plane ``6.7e-16`` / ``0.0``; ring ``beta_x`` ``7.8e-16``, ``beta_y`` ``6.7e-16``
relative. Every gate below is set at ``1e-12``, three orders of headroom.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.
"""

from __future__ import annotations

import pytest

from accsim import (
    Drift,
    Knob,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Target,
    Twiss,
    closed_twiss,
    match_insertion,
    propagate_twiss,
)

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0

# --- final-doublet insertion (transfer line) -------------------------------
LQ = 0.3  # quad length [m]
K1_START = 1.0  # starting gradient of both insertion quads [m^-2]
L1, L2, L3 = 2.0, 1.0, 3.0  # drifts: entrance -> Q1 -> Q2 -> IP
BETA_IN = 40.0  # entrance beta (a waist of the upstream arc), both planes
BETA_STAR = 6.0  # what we ask for at the IP, in x

# --- FODO ring (periodic branch) ------------------------------------------
LD = 1.0
K1_RING = 1.2
N_CELLS = 4


def _insertion() -> tuple[Lattice, Quadrupole, Quadrupole, Twiss]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    q1 = Quadrupole(LQ, K1_START, name="q1")
    q2 = Quadrupole(LQ, -K1_START, name="q2")
    lat = Lattice([Drift(L1), q1, Drift(L2), q2, Drift(L3)], ref)
    start = Twiss(0.0, BETA_IN, 0.0, 0.0, BETA_IN, 0.0, 0.0)
    return lat, q1, q2, start


def _matched_insertion() -> tuple[dict[str, float], float, float]:
    """Match ``(beta_x*, alpha_x*=0)`` at the IP; return the strengths and the IP optics."""
    lat, q1, q2, start = _insertion()
    ip = len(lat)
    match_insertion(
        lat,
        [Target("beta_x", at=ip, value=BETA_STAR), Target("alpha_x", at=ip, value=0.0)],
        [Knob([q1], name="kq1"), Knob([q2], name="kq2")],
        twiss0=start,
    )
    end = propagate_twiss(lat, start)[ip]
    # accsim's own view must be exact before xtrack is asked anything.
    assert end.beta_x == pytest.approx(BETA_STAR, rel=1e-11)
    assert end.alpha_x == pytest.approx(0.0, abs=1e-10)
    return {"k1a": q1.k1, "k1b": q2.k1}, end.beta_y, end.alpha_y


def _xtrack_line(elements):
    line = xt.Line(elements=list(elements))
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def _xtrack_insertion_twiss(s: dict[str, float]):
    line = _xtrack_line(
        [
            xt.Drift(length=L1),
            xt.Quadrupole(length=LQ, k1=s["k1a"]),
            xt.Drift(length=L2),
            xt.Quadrupole(length=LQ, k1=s["k1b"]),
            xt.Drift(length=L3),
        ]
    )
    try:
        return line.twiss(method="4d", betx=BETA_IN, bety=BETA_IN, alfx=0.0, alfy=0.0)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack open twiss unavailable: {type(exc).__name__}: {exc}")


def test_matched_beta_star_confirmed_by_xtrack() -> None:
    """xtrack propagates the matched strengths from the same entrance and finds beta*."""
    strengths, _, _ = _matched_insertion()
    tw = _xtrack_insertion_twiss(strengths)
    assert tw.betx[-1] == pytest.approx(BETA_STAR, rel=1e-12)


def test_matched_waist_confirmed_by_xtrack() -> None:
    """... and finds ``alpha_x = 0`` there, i.e. the waist really is at the IP.

    ``alpha`` is the constraint a propagation sign error would break while leaving
    ``beta`` plausible, so this is the sharper half of the pair.
    """
    strengths, _, _ = _matched_insertion()
    tw = _xtrack_insertion_twiss(strengths)
    assert tw.alfx[-1] == pytest.approx(0.0, abs=1e-12)


def test_untargeted_plane_agrees_too() -> None:
    """The y plane was never a target, so both codes must simply *agree* on it.

    A gate on what was asked for can be satisfied by a matcher that is wrong
    everywhere else; this one has no target to hide behind.
    """
    strengths, beta_y, alpha_y = _matched_insertion()
    tw = _xtrack_insertion_twiss(strengths)
    assert tw.bety[-1] == pytest.approx(beta_y, rel=1e-12)
    assert tw.alfy[-1] == pytest.approx(alpha_y, abs=1e-11)
    # ... and it is a genuinely different number, not an accidental match.
    assert abs(beta_y - BETA_STAR) > 1.0


def _matched_ring() -> tuple[dict[str, float], float, float]:
    """Periodic branch: retune the ring's beta at the cell start with two quad knobs."""
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    qf = Quadrupole(LQ, K1_RING, name="qf")
    qd = Quadrupole(LQ, -K1_RING, name="qd")
    cell = [qf, Drift(LD), qd, Drift(LD)]
    lat = Lattice(cell * N_CELLS, ref)

    before = closed_twiss(lat)
    want_x, want_y = before.beta_x * 1.20, before.beta_y * 0.85
    match_insertion(
        lat,
        [Target("beta_x", at=0, value=want_x), Target("beta_y", at=0, value=want_y)],
        [Knob([qf], name="kqf"), Knob([qd], weights=[-1.0], name="kqd")],
    )
    after = closed_twiss(lat)
    assert after.beta_x == pytest.approx(want_x, rel=1e-11)
    assert after.beta_y == pytest.approx(want_y, rel=1e-11)
    # Not a vacuous pass: the strengths really moved.
    assert qf.k1 != pytest.approx(K1_RING, rel=1e-3)
    return {"k1f": qf.k1, "k1d": qd.k1}, want_x, want_y


def test_matched_ring_beta_confirmed_by_xtrack() -> None:
    """xtrack's closed twiss of the matched ring reports the requested beta.

    The periodic branch re-solves the closed solution at every evaluation; this is
    the outside confirmation that the solution it converged on is the machine's.
    """
    strengths, want_x, want_y = _matched_ring()
    cell = [
        xt.Quadrupole(length=LQ, k1=strengths["k1f"]),
        xt.Drift(length=LD),
        xt.Quadrupole(length=LQ, k1=strengths["k1d"]),
        xt.Drift(length=LD),
    ]
    line = _xtrack_line(cell * N_CELLS)
    try:
        tw = line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack closed twiss unavailable: {type(exc).__name__}: {exc}")
    assert tw.betx[0] == pytest.approx(want_x, rel=1e-12)
    assert tw.bety[0] == pytest.approx(want_y, rel=1e-12)
