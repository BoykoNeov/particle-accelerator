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

import math

import pytest

from accsim import Dipole, Drift, Lattice, Quadrupole, ReferenceParticle, natural_chromaticity

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


# --------------------------------------------------------------------------- #
# F2: the DIPOLE contribution — weak focusing, dispersion, pole-face edges, AND
# the combined-function gradient with its Maxwell-forced curvature-sextupole
# feed-down. The quad+drift ring above has no bends, so it never exercised these
# terms. Three lattices do: a bendy sector FODO, the same with pole-face edges,
# and an alternating-gradient combined-function ring (k1 in the dipole). All match
# xtrack's real-tracking dqx/dqy — the combined-function case in particular, whose
# curvature term also matches MAD-X. See docs/CONVENTIONS.md -> Dipole chromaticity.
# --------------------------------------------------------------------------- #
N_BENDCELL = 8
BEND_L = 1.0
BEND_ANG = 2.0 * math.pi / (2 * N_BENDCELL)  # two bends per cell -> full ring

# Combined-function AG ring (dipoles ARE the focusing; strong k1 exposes the
# curvature-sextupole term that a weakly-combined lattice would hide).
N_AGCELL = 12
AG_L = 1.5
AG_LD = 0.5
AG_ANG = 2.0 * math.pi / (2 * N_AGCELL)
AG_KF = 0.3


def _bendy_accsim_chromaticity(edge: float) -> tuple[float, float]:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [
        Quadrupole(LQ, K1),
        Dipole(BEND_L, BEND_ANG, e1=edge, e2=edge),
        Quadrupole(LQ, -K1),
        Dipole(BEND_L, BEND_ANG, e1=edge, e2=edge),
    ]
    return natural_chromaticity(Lattice(cell * N_BENDCELL, ref), slices=200)


def _bendy_xtrack_line(edge: float):
    ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    active = 1 if edge else 0
    bend = {
        "length": BEND_L,
        "angle": BEND_ANG,
        "k0": BEND_ANG / BEND_L,
        "k1": 0.0,
        "model": "rot-kick-rot",
        "edge_entry_active": active,
        "edge_exit_active": active,
        "edge_entry_angle": edge,
        "edge_exit_angle": edge,
    }
    cell = [
        xt.Quadrupole(length=LQ, k1=K1),
        xt.Bend(**bend),
        xt.Quadrupole(length=LQ, k1=-K1),
        xt.Bend(**bend),
    ]
    line = xt.Line(elements=cell * N_BENDCELL)
    line.particle_ref = ref
    try:
        line.build_tracker()
        line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return line


@pytest.mark.parametrize("edge", [0.0, 0.15], ids=["sector", "with-edges"])
def test_dipole_chromaticity_matches_xtrack(edge: float) -> None:
    # F2 acceptance: the dipole weak-focusing + dispersion (+ pole-face edge)
    # chromaticity agrees with xtrack's real-tracking dqx/dqy on a bendy ring.
    line = _bendy_xtrack_line(edge)
    tw = line.twiss(method="4d")
    xi_x, xi_y = _bendy_accsim_chromaticity(edge)
    assert xi_x == pytest.approx(tw.dqx, rel=1e-4, abs=1e-5)
    assert xi_y == pytest.approx(tw.dqy, rel=1e-4, abs=1e-5)


def test_combined_function_chromaticity_matches_xtrack() -> None:
    # F2 acceptance: the combined-function dipole gradient chromaticity — including
    # the Maxwell-forced curvature-sextupole feed-down (+2 h k1 beta_x Dx /
    # -h k1 beta_y Dx) — agrees with xtrack on a strongly combined-function AG ring.
    # Without that term accsim would give dqx ~ -0.72 instead of xtrack's +0.62.
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    cell = [
        Dipole(AG_L, AG_ANG, k1=+AG_KF),
        Drift(AG_LD),
        Dipole(AG_L, AG_ANG, k1=-AG_KF),
        Drift(AG_LD),
    ]
    lat = Lattice(cell * N_AGCELL, ref)

    bp = {
        "length": AG_L,
        "angle": AG_ANG,
        "k0": AG_ANG / AG_L,
        "model": "rot-kick-rot",
        "edge_entry_active": 0,
        "edge_exit_active": 0,
    }
    xcell = [
        xt.Bend(k1=+AG_KF, **bp),
        xt.Drift(length=AG_LD),
        xt.Bend(k1=-AG_KF, **bp),
        xt.Drift(length=AG_LD),
    ]
    line = xt.Line(elements=xcell * N_AGCELL)
    line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
    try:
        line.build_tracker()
        line.twiss(method="4d")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    tw = line.twiss(method="4d")
    xi_x, xi_y = natural_chromaticity(lat, slices=300)
    assert xi_x == pytest.approx(tw.dqx, rel=1e-3, abs=1e-3)
    assert xi_y == pytest.approx(tw.dqy, rel=1e-3, abs=1e-3)
