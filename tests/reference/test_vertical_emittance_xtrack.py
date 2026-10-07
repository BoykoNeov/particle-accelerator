r"""V2 — vertical emittance from design ``D_y`` against xtrack, which arbitrates it twice.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

xtrack computes the vertical equilibrium by **two** methods, and V2's finding is that only
one of them is right everywhere:

- its **integral route** (``twiss(radiation_integrals=True)``: ``rad_int_i4y``,
  ``rad_int_i5y``, ``rad_int_eq_gemitt_y``) is accsim's method, element by element. On the
  dogleg ring — tilted bends without a gradient — it lands on accsim to its own slicing
  error, which falls as ``1/n^2``. On a tilted **gradient** bend it is wrong: it writes
  ``i4y = D_y (kappa0_y kappa^2 - 2 k1 kappa_y)`` with ``k1`` the bend's *own* gradient,
  where the ``-`` belongs to a lab-frame normal gradient. A ring turned over by ``pi/2``
  then gets ``J_y = -0.59`` against the flat ring's ``J_x = +0.26`` — a negative
  emittance. The error is exactly ``-4 k1 I1`` (gated).
- its **eigen route** (``twiss(radiation_analysis=True)``: ``partition_numbers``,
  ``eq_gemitt_y``) solves the radiating one-turn map, respects the rotation exactly, and
  departs from the integral as ``c (2 pi Q_s)^2`` with the same ``c`` accsim's own tracked
  equilibrium shows (``test_vertical_emittance.py`` gate 6).

**Cost.** Four ``xt.Line`` builds (12-70 s apiece — ``docs/CONVENTIONS.md`` -> *Test-suite
cost*), all module-scoped.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

import test_vertical_emittance as v2  # noqa: E402
from test_design_tilt import ANG, ELECTRON_MASS_EV, NCELL, VA, dogleg_ring  # noqa: E402

from accsim.radiation import (  # noqa: E402
    damping_partition_numbers,
    equilibrium_vertical_emittance,
    radiation_integrals,
)

PI2 = math.pi / 2
CLIGHT = 299792458.0
K1B = -0.05  # the combined-function gradient (the ring is unstable for k1 > 0)


def _line(
    t: float = 0.0, k1b: float = 0.0, dogleg: bool = False, slices: int = 40, rf: bool = False
):
    """The V1 ring in xtrack — optionally with the dogleg, rolled by ``t``, gradient ``k1b``
    in its bends — every bend sliced ``slices`` times (thick), and optionally a cavity."""
    els, nms = [], []
    for i in range(NCELL):
        for tag, e in (
            (f"qf{i}", xt.Quadrupole(length=0.5, k1=0.30, rot_s_rad=t)),
            (f"d{i}a", xt.Drift(length=0.6)),
            (f"b{i}a", xt.Bend(length=1.5, angle=ANG, k0_from_h=True, k1=k1b, rot_s_rad=t)),
            (f"d{i}b", xt.Drift(length=0.6)),
            (f"qd{i}", xt.Quadrupole(length=0.5, k1=-0.30, rot_s_rad=t)),
            (f"d{i}c", xt.Drift(length=0.6)),
            (f"b{i}b", xt.Bend(length=1.5, angle=ANG, k0_from_h=True, k1=k1b, rot_s_rad=t)),
            (f"d{i}d", xt.Drift(length=0.6)),
        ):
            els.append(e)
            nms.append(tag)
        if dogleg and i == 0:
            els += [
                xt.Bend(length=0.3, angle=VA, k0_from_h=True, rot_s_rad=PI2),
                xt.Drift(length=1.0),
                xt.Bend(length=0.3, angle=-VA, k0_from_h=True, rot_s_rad=PI2),
            ]
            nms += ["vb1", "dd", "vb2"]
    if rf:
        length = sum(e.length for e in els)
        els.append(xt.Cavity(voltage=3.9e6, frequency=10 * CLIGHT / length, lag=180.0))
        nms.append("rf")
    line = xt.Line(elements=els, element_names=nms)
    line.particle_ref = xt.Particles(mass0=ELECTRON_MASS_EV, q0=-1.0, energy0=3.0e9)
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Uniform(slices, mode="thick"), element_type=xt.Bend),
        ]
    )
    line.build_tracker()
    return line


_INT = ("rad_int_i1x", "rad_int_i1y", "rad_int_i2", "rad_int_i4x", "rad_int_i4y",
        "rad_int_i5x", "rad_int_i5y", "rad_int_eq_gemitt_y")  # fmt: skip


def _integrals(line) -> dict[str, float]:
    tw = line.twiss(method="4d", radiation_integrals=True)
    return {k: float(getattr(tw, k)) for k in _INT}


#: The RF voltages of the eigen scan [V], harmonic 10: Q_s ~ 0.010, 0.015, 0.024.
_SCAN = (3.78e6, 3.9e6, 4.5e6)


@pytest.fixture(scope="module")
def dogleg_xt():
    """``(integrals at 40 slices, integrals at 80, eigen scan [(Q_s, eps_y)])``.

    The integrals come from cavity-free lines: with the cavity in the line, xtrack's 4D
    ``radiation_integrals`` moved ``eq_gemitt_y`` by ``~6e-7`` — larger than its own 80-slice
    error — so the eigen scan gets a line of its own.
    """
    coarse = _integrals(_line(dogleg=True, slices=40))
    fine = _integrals(_line(dogleg=True, slices=80))
    line = _line(dogleg=True, slices=80, rf=True)
    line.configure_radiation(model="mean")
    scan = []
    for volt in _SCAN:
        line["rf"].voltage = volt
        tw = line.twiss(radiation_analysis=True)
        scan.append((abs(float(tw.qs)), float(tw.eq_gemitt_y)))
    return coarse, fine, scan


@pytest.fixture(scope="module")
def gradient_xt():
    """The combined-function ring flat and turned over: ``{t: (integrals, eigen J)}``."""
    out = {}
    for t in (0.0, PI2):
        line = _line(t=t, k1b=K1B, slices=40, rf=True)
        ints = _integrals(line)
        line["rf"].voltage = 8.0e6
        line.configure_radiation(model="mean")
        out[t] = (ints, np.asarray(line.twiss(radiation_analysis=True).partition_numbers))
    return out


# ---------------------------------------------------------------------------
# The integral route, on tilted bends without a gradient: accsim's method, element by element.
# ---------------------------------------------------------------------------
def _accsim_limit() -> dict[str, float]:
    """accsim's integrals with the trapezoid's ``1/n^2`` stripped (Richardson, 128/256).

    At 256 slices accsim's ``i4y`` still carries ``6.9e-7`` — 11% of xtrack's 80-slice error
    — which is enough to bend xtrack's law; the extrapolated ``i5y`` lands on gate 3's closed
    form to ``1e-15``."""
    from accsim.radiation import quantum_constant_cq

    lat = dogleg_ring()
    a, b = radiation_integrals(lat, slices=128), radiation_integrals(lat, slices=256)
    i4y, i5y = (4.0 * b.i4y - a.i4y) / 3.0, (4.0 * b.i5y - a.i5y) / 3.0
    cq = quantum_constant_cq(lat.ref) * lat.ref.gamma0**2
    return {"i4y": i4y, "i5y": i5y, "eps_y": cq * i5y / (b.i2 - i4y)}


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [("i5y", "rad_int_i5y"), ("i4y", "rad_int_i4y"), ("eps_y", "rad_int_eq_gemitt_y")],
)
def test_the_dogleg_integrals_meet_xtrack_at_its_slicing_law(dogleg_xt, ours, theirs) -> None:
    """xtrack's error against accsim's converged value is its own slicing's, and falls by 4
    from 40 to 80 slices. Measured at 80: ``i5y`` ``3.2e-7``, ``eps_y`` ``3.2e-7``, ``i4y``
    ``6e-6`` (``i4y`` is a signed sum over two opposite vertical bends, so it is small and
    its relative error large). The two codes' ``C_q gamma^2`` agree to ``5e-11``."""
    coarse, fine, _ = dogleg_xt
    value = _accsim_limit()[ours]
    e40, e80 = coarse[theirs] / value - 1.0, fine[theirs] / value - 1.0
    assert e40 / e80 == pytest.approx(4.0, rel=0.1)
    assert abs(e80) < 1e-5


def test_the_vertical_half_of_i1_has_xtracks_sign(dogleg_xt) -> None:
    """``I1`` is the total ``∮ h . D``; xtrack splits it. The vertical half is a pure V2
    quantity — two vertical bends of opposite sign, in a ``D_y`` they make themselves. (The
    horizontal half is not compared: xtrack's ``i1x`` sits ``5e-5`` from its own ring's
    ``alpha_c C`` at 80 slices, where accsim's ``I1`` is gated against the identity route.)"""
    _, fine, _ = dogleg_xt
    vertical = v2._vertical_i1(dogleg_ring(), 256)
    assert vertical == pytest.approx(fine["rad_int_i1y"], rel=1e-5, abs=0)
    assert vertical < 0.0


# ---------------------------------------------------------------------------
# A tilted gradient bend: xtrack's integral route breaks the rotation, its eigen route does not.
# ---------------------------------------------------------------------------
def test_xtracks_eigen_route_respects_the_rotation(gradient_xt) -> None:
    jx_flat = gradient_xt[0.0][1][0]
    j_rolled = gradient_xt[PI2][1]
    assert j_rolled[1] == pytest.approx(jx_flat, rel=1e-5)
    assert j_rolled[0] == pytest.approx(1.0, abs=1e-5)


def test_accsim_turns_xtracks_flat_integral_over(gradient_xt) -> None:
    """The flat ring is where xtrack's integral route is right (no tilt), and it is accsim's
    method; the rolled ring's ``J_y`` in accsim must be that ``J_x``."""
    flat = gradient_xt[0.0][0]
    jx_xtrack_flat = 1.0 - flat["rad_int_i4x"] / flat["rad_int_i2"]
    jy_rolled = damping_partition_numbers(v2.cf_ring(tilt=PI2, k1=K1B, e=0.0))[1]
    jx_flat = damping_partition_numbers(v2.cf_ring(k1=K1B, e=0.0))[0]
    assert jx_flat == pytest.approx(jx_xtrack_flat, rel=2e-5)  # xtrack's 40-slice error
    assert jy_rolled == pytest.approx(jx_flat, rel=1e-12)


def test_xtracks_integral_route_flips_the_gradient_in_the_vertical_plane(gradient_xt) -> None:
    """The localisation, as an identity: xtrack's rolled ``i4y`` is the flat ``i4x`` with the
    gradient term reversed — ``(h^2 - 2 k1) I1`` for ``(h^2 + 2 k1) I1`` — so it differs by
    exactly ``-4 k1 I1``. Written to fail the day xtrack fixes it."""
    flat, rolled = gradient_xt[0.0][0], gradient_xt[PI2][0]
    predicted = flat["rad_int_i4x"] - 4.0 * K1B * flat["rad_int_i1x"]
    assert rolled["rad_int_i4y"] == pytest.approx(predicted, rel=1e-7)
    assert rolled["rad_int_i4y"] / flat["rad_int_i4x"] > 2.0  # not a rounding difference
    assert 1.0 - rolled["rad_int_i4y"] / rolled["rad_int_i2"] < 0.0  # J_y < 0: unphysical


# ---------------------------------------------------------------------------
# The eigen route on the dogleg: the finite-Q_s departure has accsim's slope.
# ---------------------------------------------------------------------------
def test_xtracks_eigen_emittance_departs_with_accsims_tracked_slope(dogleg_xt) -> None:
    """xtrack's eigen ``eq_gemitt_y`` and accsim's tracked Lyapunov ``eps_y`` both solve the
    radiating one-turn map; both depart from the integral as ``c (2 pi Q_s)^2`` and the two
    ``c`` must agree. (The intercepts are not compared: each carries its code's energy-sag
    and thick-bend bookkeeping — CONVENTIONS -> *Vertical emittance*.)"""
    _, _, scan = dogleg_xt
    eps = equilibrium_vertical_emittance(dogleg_ring())
    xs = [(2 * math.pi * q) ** 2 for q, _ in scan]
    ds = [e / eps - 1.0 for _, e in scan]
    slope_xt, _ = np.polyfit(xs, ds, 1)
    _, slope_accsim, _ = v2._departure_law(3.0e9)
    assert slope_xt == pytest.approx(slope_accsim, rel=3e-2)
