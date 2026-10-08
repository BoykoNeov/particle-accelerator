"""Cross-check the AC dipole's driven optics against xtrack's ``twiss_mode`` (W2).

Marked ``reference``; skips when xtrack or its JIT is unavailable.

xtrack's ``ACDipole(twiss_mode=True)`` replaces the drive by Miyamoto's thin gradient,
``p_u += eff_grad * u`` in the driven plane only, given the natural ``beta`` at the dipole
and the natural tune; its ``twiss`` then returns the driven optics. accsim's
:func:`~accsim.twiss.driven_twiss` makes the same substitution, so this leg checks the
*transcription* — the strength, its sign, the plane, where it sits — not Miyamoto's claim
itself. That claim (the steady state is the substituted ring's eigen-solution) is gated
without an arbiter in ``tests/analytic/test_ac_dipole_driven_optics.py``; the tracking leg
is W1's, which holds accsim's driven tracking to xtrack's turn by turn.

One JIT build: the dipole's ``freq``, ``plane``, ``beta_at_acdipole`` and ``natural_q`` are
data, reset between cases without recompiling. xtrack's ``mux`` is in units of ``2 pi``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    PROTON_MASS_EV,
    ACDipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinQuadrupole,
    driven_gradient,
    driven_twiss,
)
from accsim.twiss import closed_twiss, propagate_twiss

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

GAMMA0, K1L, LD, NCELL, IDX = 10.0, 0.25, 5.0, 6, 2
OFFSET = 0.012
# relative; measured over the four plane x side cases: beta 2.0e-15, alpha 1.8e-15,
# mu 4.3e-16, tune 2.2e-16 — a linear ring, so xtrack's finite-difference twiss is exact
GATE = 1e-12


def _accsim(acd: ACDipole | None) -> Lattice:
    els = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(K1L), Drift(LD), ThinQuadrupole(-K1L), Drift(LD)]
    els.insert(IDX, acd if acd is not None else ACDipole(0.0, 0.3))
    return Lattice(els, ReferenceParticle.from_gamma(PROTON_MASS_EV, GAMMA0))


@pytest.fixture(scope="module")
def line():
    els, names = [], []
    for i in range(NCELL):
        els += [
            xt.Multipole(knl=[0, K1L]),
            xt.Drift(length=LD),
            xt.Multipole(knl=[0, -K1L]),
            xt.Drift(length=LD),
        ]
        names += [f"qf{i}", f"da{i}", f"qd{i}", f"db{i}"]
    els.insert(IDX, xt.ACDipole(freq=0.3, plane="h", twiss_mode=True))
    names.insert(IDX, "acd")
    out = xt.Line(elements=els, element_names=names)
    out.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1.0, gamma0=GAMMA0)
    try:
        out.build_tracker()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return out


@pytest.mark.parametrize("plane", ["x", "y"])
@pytest.mark.parametrize("side", [1, -1])
def test_the_driven_optics_matches_xtracks_twiss_mode(line, plane: str, side: int) -> None:
    natural_lat = _accsim(None)
    nat = propagate_twiss(natural_lat, closed_twiss(natural_lat))
    end = nat[-1]
    q = (end.mu_x if plane == "x" else end.mu_y) / (2.0 * math.pi)
    beta = nat[IDX].beta_x if plane == "x" else nat[IDX].beta_y
    nu = q % 1.0 - side * OFFSET
    lat = _accsim(ACDipole(kick=1e-5, tune=nu, plane=plane))
    ours = driven_twiss(lat)
    g = driven_gradient(lat)[plane]

    acd = line.element_dict["acd"]  # the element itself: the view reads "v" as a variable
    acd.plane = "h" if plane == "x" else "v"
    acd.natural_q = q
    acd.beta_at_acdipole = beta
    acd.freq = nu
    assert acd.eff_grad == pytest.approx(g, rel=1e-13)  # same sign; measured 6.2e-15

    tw = line.twiss(method="4d")
    u = "x" if plane == "x" else "y"
    xb, xa, xm = (np.asarray(tw[f"{k}{u}"]) for k in ("bet", "alf", "mu"))
    assert xm[-1] == pytest.approx(tw[f"q{u}"], rel=1e-12)  # mu is in units of 2 pi
    assert tw[f"q{u}"] == pytest.approx(math.floor(q) + nu, abs=GATE)
    beta_d = np.array([t.beta_x if plane == "x" else t.beta_y for t in ours])
    alpha_d = np.array([t.alpha_x if plane == "x" else t.alpha_y for t in ours])
    mu_d = np.array([t.mu_x if plane == "x" else t.mu_y for t in ours])
    np.testing.assert_allclose(xb, beta_d, rtol=GATE)
    np.testing.assert_allclose(xa, alpha_d, rtol=0.0, atol=GATE * np.max(np.abs(alpha_d)))
    np.testing.assert_allclose(2.0 * math.pi * xm, mu_d, rtol=0.0, atol=GATE * mu_d[-1])
    # and the drive really moved the optics: the natural beta misses by the beat
    beta_n = np.array(
        [t.beta_x if plane == "x" else t.beta_y for t in propagate_twiss(lat, closed_twiss(lat))]
    )
    assert np.max(np.abs(xb / beta_n - 1.0)) > 0.03
