"""Stage 2 acceptance cross-check: natural chromaticity vs. xtrack.

xtrack derives ``Q'`` by tracking real particles at different ``delta`` through the
full nonlinear maps, so it is genuinely independent of accsim's first-order
β-weighted formula — exactly the reference check the roadmap wants for a quantity
where a hand-derived coefficient error would otherwise go unseen.

The same thick-quad FODO ring is built in both codes (thin lenses are avoided
because xtrack's ``Quadrupole`` is unambiguous and already pinned entrywise). One
convention guard is asserted up front: xtrack's ``tw.dqx`` must be ``dQx/ddelta``
(un-normalised), confirmed by finite-differencing xtrack's *own* tunes at
``delta = +/- h`` — that is precisely where a hidden factor (a stray ``Q`` or
``2pi``) would live.

Marked ``reference``: skips (not fails) when xtrack or its JIT compiler is
unavailable — see ``docs/CONVENTIONS.md``.
"""

from __future__ import annotations

import pytest

from accsim import Drift, Lattice, Quadrupole, ReferenceParticle, natural_chromaticity

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

LQ = 0.3  # quad length [m]
K1 = 1.2  # quad gradient [m^-2]
LD = 1.0  # drift length [m]
MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0
N_CELLS = 4


def _accsim_chromaticity() -> tuple[float, float]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [Quadrupole(LQ, K1), Drift(LD), Quadrupole(LQ, -K1), Drift(LD)]
    return natural_chromaticity(Lattice(cell * N_CELLS, ref))


def _xtrack_line():
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    cell = [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Drift(length=LD),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Drift(length=LD),
    ]
    line = xt.Line(elements=cell * N_CELLS)
    line.particle_ref = ref
    try:
        line.build_tracker()
        line.twiss(method="4d")  # probe: raises here if the JIT can't build
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


def test_chromaticity_convention_is_dq_ddelta() -> None:
    # Guard the convention before trusting tw.dqx: finite-difference xtrack's OWN
    # tunes at delta = +/- h and confirm it reproduces tw.dqx. This pins that
    # dqx == dQx/ddelta (not dQ/Q, not divided by 2pi) — the factor a chromaticity
    # bug hides behind.
    line = _xtrack_line()
    tw0 = line.twiss(method="4d")
    h = 1e-5
    twp = line.twiss(method="4d", delta0=h)
    twm = line.twiss(method="4d", delta0=-h)
    fd_x = (twp.qx - twm.qx) / (2.0 * h)
    fd_y = (twp.qy - twm.qy) / (2.0 * h)
    assert fd_x == pytest.approx(tw0.dqx, rel=1e-3)
    assert fd_y == pytest.approx(tw0.dqy, rel=1e-3)


def test_natural_chromaticity_matches_xtrack() -> None:
    line = _xtrack_line()
    tw = line.twiss(method="4d")

    xi_x, xi_y = _accsim_chromaticity()

    # The Stage 2 acceptance gate: natural chromaticity agrees with xtrack.
    # Tolerance covers accsim's trapezoidal slicing and xtrack's own finite step.
    assert xi_x == pytest.approx(tw.dqx, rel=1e-4, abs=1e-6)
    assert xi_y == pytest.approx(tw.dqy, rel=1e-4, abs=1e-6)
    assert xi_x < 0.0 and xi_y < 0.0
