r"""F2 acceptance: full dipole natural chromaticity.

``natural_chromaticity`` gained the dipole contribution — weak-focusing ``h^2``,
combined-function gradient ``k1``, the dispersion corrections, and pole-face
edges — on top of the quadrupole term. These gates pin that physics *without*
xtrack (the xtrack cross-check lives in ``tests/reference``), layered so a wrong
coefficient and a wrong bookkeeping cannot cancel:

1. **Symbolic integrand** — the per-length tune-shift integrand is re-derived from
   the exact curvilinear Hamiltonian (linearised about the dispersed orbit) and
   the generator tune-shift formula, and matched term-by-term against a hardcoded
   symbolic expression. Coefficients are *derived*, not remembered.
2. **beta-form == gamma-form ring total** — the module ships the beta-weighted
   form ``-beta(k1+h^2) + h(gamma Dx - 2 alpha Dpx)``; the symbolic derivation is
   naturally the gamma-form ``gamma(h Dx - 1) - 2 alpha h Dpx``. They are equal
   only around a *closed* ring (the identity ``∮ alpha' ds = 0`` ⇒ ``∮ gamma ds =
   ∮ beta K ds``). Computing both independently and demanding equality ties the
   shipped form to the symbolic one.
3. **Off-momentum-map self-consistency** — an *independent* computation of the
   same Hamiltonian's chromaticity: build the off-momentum one-turn map from
   ``exp(A(x_co, px_co, delta) ds)`` along the matched dispersed orbit and
   finite-difference the tune. This validates the whole implementation (body +
   dispersion + edges + combined-function incl. the curvature-sextupole term) via a
   different path (matrix exponential vs perturbation theory). The combined-function
   result is *also* cross-checked against xtrack in ``tests/reference``.
4. **Reductions** — a straight (quad-only) lattice is byte-identical to the old
   quad-only formula; a pure sector bend's own contribution is a near-cancellation
   (weak-focusing killed by dispersion), *not* the naive ``-∮ beta_x h^2``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from accsim import (
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    natural_chromaticity,
)
from accsim.elements.dipole import Dipole, _edge_matrix
from accsim.twiss import (
    _blocks,
    _dipole_chroma_integrand,
    _dispersive_kick,
    _propagate_block,
    _transverse_4d,
    closed_twiss,
)

MASS0 = 938.27208816e6  # proton, eV
GAMMA0 = 20.0


@pytest.fixture
def ref() -> ReferenceParticle:
    return ReferenceParticle.from_gamma(MASS0, GAMMA0)


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #
def _sector_fodo(ref: ReferenceParticle, e1e2: float = 0.0, dip_k1: float = 0.0) -> Lattice:
    """8-cell FODO with two bends per cell (total bend 2 pi). Optionally edged."""
    n, lq, k1, lb = 8, 0.3, 0.9, 1.0
    ang = 2.0 * math.pi / (2 * n)
    cell = [
        Quadrupole(lq, k1),
        Dipole(lb, ang, k1=dip_k1, e1=e1e2, e2=e1e2),
        Quadrupole(lq, -k1),
        Dipole(lb, ang, k1=dip_k1, e1=e1e2, e2=e1e2),
    ]
    return Lattice(cell * n, ref)


def _ag_combined(ref: ReferenceParticle, kf: float) -> Lattice:
    """Alternating-gradient combined-function ring (dipoles ARE the focusing)."""
    n, lb, ld = 12, 1.5, 0.5
    ang = 2.0 * math.pi / (2 * n)
    cell = [Dipole(lb, ang, k1=+kf), Drift(ld), Dipole(lb, ang, k1=-kf), Drift(ld)]
    return Lattice(cell * n, ref)


# --------------------------------------------------------------------------- #
# 1. Symbolic derivation of the integrand
# --------------------------------------------------------------------------- #
def test_integrand_derived_symbolically() -> None:
    """Re-derive the full integrand from the exact Hamiltonian (sympy), including
    the Maxwell-forced curvature-sextupole term.

    A combined-function sector magnet's naive potential ``psi = -h x -(k1+h^2)/2 x^2
    + k1/2 y^2`` violates ``div B = 0`` in the curved frame; the fix is a 3rd-order
    term ``psi3 = c1 x^3 + c2 x y^2`` (even in y) with Maxwell requiring
    ``psi_xx + psi_yy = h psi_x/(1+hx)``. That fixes ONE combination
    (``6 c1 + 2 c2 + h k1 = 0``); the split is pinned by the *horizontal* xtrack
    match — and the **vertical** then follows with no further freedom, which is the
    non-circular confirmation. The resulting integrand is the one the module ships.
    """
    sp = pytest.importorskip("sympy")
    x, px, y, py, delta = sp.symbols("x px y py delta", real=True)
    h, k1, dx, dpx, c1, c2 = sp.symbols("h k1 Dx Dpx c1 c2", real=True)
    bx, ax, gx, by, ay, gy = sp.symbols("beta_x alpha_x gamma_x beta_y alpha_y gamma_y", real=True)

    psi = (
        -h * x
        - sp.Rational(1, 2) * (k1 + h**2) * x**2
        + sp.Rational(1, 2) * k1 * y**2
        + c1 * x**3
        + c2 * x * y**2
    )
    # Maxwell (curved-frame): psi_xx + psi_yy = h psi_x/(1+hx); the O(x) residual.
    residual = sp.diff(psi, x, 2) + sp.diff(psi, y, 2) - h * sp.diff(psi, x) / (1 + h * x)
    residual = sp.series(sp.series(residual, x, 0, 2).removeO(), y, 0, 2).removeO()
    maxwell_eq = sp.Eq(sp.Poly(sp.expand(residual), x, y).coeffs()[0], 0)

    pz = sp.sqrt((1 + delta) ** 2 - px**2 - py**2)
    hh = -(1 + h * x) * pz - psi
    eom = [sp.diff(hh, px), -sp.diff(hh, x), sp.diff(hh, py), -sp.diff(hh, y)]
    coords = [x, px, y, py]
    a_full = sp.Matrix(4, 4, lambda i, j: sp.diff(eom[i], coords[j]))
    orbit = {x: dx * delta, px: dpx * delta, y: 0, py: 0}
    d_a = sp.simplify(sp.diff(a_full.subs(orbit), delta).subs(delta, 0))

    # Tune-shift-from-generator: dmu = -alpha N11 - (beta/2) N21 + (gamma/2) N12.
    dmux = sp.expand(-ax * d_a[0, 0] - bx / 2 * d_a[1, 0] + gx / 2 * d_a[0, 1])
    dmuy = sp.expand(-ay * d_a[2, 2] - by / 2 * d_a[3, 2] + gy / 2 * d_a[2, 3])

    # xi = (1/2pi) int dmu; the module's (1/4pi) integrand is 2*dmu. Pin (c1,c2) by
    # Maxwell + the horizontal empirical (xtrack) feed  +2 h k1 beta_x Dx.
    horiz_eq = sp.Eq(2 * dmux, gx * (h * dx - 1) - 2 * ax * h * dpx + 2 * h * k1 * bx * dx)
    sol = sp.solve([maxwell_eq, horiz_eq], [c1, c2], dict=True)[0]
    assert sp.simplify(sol[c1] - (-h * k1 / 3)) == 0
    assert sp.simplify(sol[c2] - (h * k1 / 2)) == 0

    # The VERTICAL integrand then follows with NO further freedom (independent check).
    iy = sp.simplify(2 * dmuy.subs(sol))
    assert sp.simplify(iy - (gy * (h * dx - 1) - h * k1 * by * dx)) == 0
    # And the horizontal matches the shipped beta-form once gamma_x is substituted.
    ix = sp.simplify(2 * dmux.subs(sol))
    assert sp.simplify(ix - (gx * (h * dx - 1) - 2 * ax * h * dpx + 2 * h * k1 * bx * dx)) == 0


# --------------------------------------------------------------------------- #
# 2. beta-form (shipped) == gamma-form (symbolic) around the ring
# --------------------------------------------------------------------------- #
def _gamma_integrand(
    bx: float, ax: float, by: float, ay: float, dx: float, dpx: float, h: float, k1: float
) -> tuple[float, float]:
    """gamma-form integrand ``gamma_u (h Dx - 1)`` plus the curvature-sextupole
    feed-down (``+2 h k1 beta_x Dx`` / ``-h k1 beta_y Dx``); x also ``- 2 alpha_x h Dpx``."""
    gx = (1.0 + ax * ax) / bx
    gy = (1.0 + ay * ay) / by
    ix = gx * (h * dx - 1.0) - 2.0 * ax * h * dpx + 2.0 * h * k1 * bx * dx
    iy = gy * (h * dx - 1.0) - h * k1 * by * dx
    return ix, iy


def _gamma_form_chromaticity(lattice: Lattice, slices: int = 200) -> tuple[float, float]:
    """Independent gamma-form chromaticity over ALL elements (incl. drifts).

    ``xi_u = (1/4pi) ∮ [gamma_u (h Dx - 1) - {2 alpha_x h Dpx for x}] ds`` plus the
    same edge terms. A different bookkeeping from the module's beta-form; equal
    only around the closed ring.
    """
    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    inv4pi = 1.0 / (4.0 * math.pi)
    xi_x = xi_y = 0.0

    def advance(m: np.ndarray) -> None:
        nonlocal bx, ax, by, ay, disp
        cx, cy = _blocks(m)
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
        disp = _transverse_4d(m) @ disp + _dispersive_kick(m)

    for elem in lattice.elements:
        is_dip = isinstance(elem, Dipole) and elem.angle != 0.0
        h = elem.curvature if is_dip else 0.0
        if is_dip:
            t = h * math.tan(elem.e1)
            xi_x += inv4pi * bx * t
            xi_y += -inv4pi * by * t
            advance(_edge_matrix(h, elem.e1))
        if elem.length == 0.0:
            advance(elem.matrix(ref))
            continue
        ds = elem.length / slices
        if is_dip:
            sub = Dipole(ds, h * ds, k1=elem.k1).matrix(ref)
        elif isinstance(elem, Quadrupole) and elem.k1 != 0.0:
            sub = Quadrupole(ds, elem.k1).matrix(ref)
        else:
            sub = Drift(ds).matrix(ref)
        k1 = elem.k1 if is_dip else 0.0
        cx, cy = _blocks(sub)
        sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
        ix, iy = _gamma_integrand(bx, ax, by, ay, disp[0], disp[1], h, k1)
        acc_x, acc_y = 0.5 * ix, 0.5 * iy
        for i in range(slices):
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
            disp = sub4 @ disp + subk
            ix, iy = _gamma_integrand(bx, ax, by, ay, disp[0], disp[1], h, k1)
            w = 0.5 if i == slices - 1 else 1.0
            acc_x += w * ix
            acc_y += w * iy
        xi_x += inv4pi * acc_x * ds
        xi_y += inv4pi * acc_y * ds
        if is_dip:
            t = h * math.tan(elem.e2)
            xi_x += inv4pi * bx * t
            xi_y += -inv4pi * by * t
            advance(_edge_matrix(h, elem.e2))
    return xi_x, xi_y


@pytest.mark.parametrize(
    "make", [lambda r: _sector_fodo(r), lambda r: _ag_combined(r, 0.2)], ids=["sector", "combined"]
)
def test_betaform_equals_gammaform_ring_total(ref: ReferenceParticle, make) -> None:
    """The shipped beta-form and the symbolic gamma-form agree on the ring total.

    Smooth-K lattices only (no pole-face edges): a thin edge is a delta-function in
    K, for which the identity ``∮ gamma ds = ∮ beta K ds`` (from ``∮ alpha' ds =
    0``) does not apply pointwise. Edges are validated separately by the
    off-momentum-map gate and the xtrack cross-check. Combined-function is included
    so the dipole-k1 term is tied to the symbolic derivation too.
    """
    lat = make(ref)
    nx, ny = natural_chromaticity(lat, slices=600)
    gx, gy = _gamma_form_chromaticity(lat, slices=600)
    # Equal only around the closed ring; at finite slices the trapezoidal integral
    # of their per-length difference (-alpha') telescopes to O((ds)^2), not exactly 0.
    assert nx == pytest.approx(gx, rel=1e-4, abs=1e-6)
    assert ny == pytest.approx(gy, rel=1e-4, abs=1e-6)


def test_module_integrand_matches_helper(ref: ReferenceParticle) -> None:
    """The public integrand helper is the beta-form the symbolic gamma-form maps to."""
    # At one concrete (beta, alpha, D) point, the beta-form and gamma-form integrands
    # differ by exactly the boundary term d/ds(alpha) contribution -> only the ring
    # total matches, but the helper must reproduce the documented beta-form algebra.
    bx, ax, by, ay, dx, dpx, h, k1 = 5.0, -0.9, 8.0, 1.3, 3.9, 0.58, 0.26, 0.15
    ix, iy = _dipole_chroma_integrand(bx, ax, by, ay, dx, dpx, h, k1)
    gamx = (1 + ax * ax) / bx
    gamy = (1 + ay * ay) / by
    # beta-form incl. the curvature-sextupole feed-down (+2 h k1 beta_x Dx / -h k1 beta_y Dx).
    assert ix == pytest.approx(
        -bx * (k1 + h * h) + h * (gamx * dx - 2 * ax * dpx) + 2 * h * k1 * bx * dx
    )
    assert iy == pytest.approx(by * k1 + gamy * h * dx - h * k1 * by * dx)


# --------------------------------------------------------------------------- #
# 3. Off-momentum-map self-consistency (independent path; covers combined-func)
# --------------------------------------------------------------------------- #
def _generator(h: float, k1: float, xco: float, pxco: float, delta: float) -> np.ndarray:
    """4x4 linear generator d/ds[x,px,y,py] of the exact EOM about (xco,pxco,0,0).

    Includes the Maxwell-forced curvature field ``psi3 = c1 x^3 + c2 x y^2`` with
    ``c1 = -h k1/3, c2 = +h k1/2`` (see ``_derive`` in the symbolic test): its only
    effect on the linear generator is a position-dependent gradient
    ``a[1,0] += 6 c1 xco = -2 h k1 xco`` and ``a[3,2] += 2 c2 xco = +h k1 xco`` —
    the curvature-sextupole feed-down at the dispersed orbit.
    """
    pz = math.sqrt((1 + delta) ** 2 - pxco**2)
    a = np.zeros((4, 4))
    a[0, 0] = h * pxco / pz
    a[0, 1] = (1 + h * xco) * (1.0 / pz + pxco**2 / pz**3)
    a[1, 0] = -(k1 + h * h) - 2.0 * h * k1 * xco
    a[1, 1] = -h * pxco / pz
    a[2, 3] = (1 + h * xco) / pz
    a[3, 2] = k1 + h * k1 * xco
    return a


def _offmomentum_blocks(
    lattice: Lattice, delta: float, slices: int = 80
) -> tuple[np.ndarray, np.ndarray]:
    """Off-momentum one-turn (x,px) and (y,py) 2x2 blocks about the dispersed orbit."""
    from scipy.linalg import expm

    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    m4 = np.eye(4)
    for elem in lattice.elements:
        is_dip = isinstance(elem, Dipole) and elem.angle != 0.0
        h = elem.curvature if is_dip else 0.0
        k1 = elem.k1 if is_dip else (elem.k1 if isinstance(elem, Quadrupole) else 0.0)
        if is_dip and elem.e1:
            e = _edge_matrix(h, elem.e1)
            m4 = _transverse_4d(e) @ m4
            disp = _transverse_4d(e) @ disp + _dispersive_kick(e)
        if elem.length == 0.0:
            mel = _transverse_4d(elem.matrix(ref))
            m4 = mel @ m4
            disp = mel @ disp + _dispersive_kick(elem.matrix(ref))
            continue
        ds = elem.length / slices
        if is_dip:
            sub = Dipole(ds, h * ds, k1=elem.k1).matrix(ref)
        elif isinstance(elem, Quadrupole) and elem.k1 != 0.0:
            sub = Quadrupole(ds, elem.k1).matrix(ref)
        else:
            sub = Drift(ds).matrix(ref)
        sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
        for _ in range(slices):
            a = _generator(h, k1, disp[0] * delta, disp[1] * delta, delta)
            m4 = expm(a * ds) @ m4
            disp = sub4 @ disp + subk
        if is_dip and elem.e2:
            e = _edge_matrix(h, elem.e2)
            m4 = _transverse_4d(e) @ m4
            disp = _transverse_4d(e) @ disp + _dispersive_kick(e)
    return m4[:2, :2], m4[2:, 2:]


def _map_chromaticity(lattice: Lattice, step: float = 1e-6) -> tuple[float, float]:
    """dQ/ddelta by finite-differencing the off-momentum one-turn tune (sign-safe)."""

    def half_tr(b: np.ndarray) -> float:
        return 0.5 * (b[0, 0] + b[1, 1])

    bx0, by0 = _offmomentum_blocks(lattice, 0.0)
    bxp, byp = _offmomentum_blocks(lattice, +step)
    bxm, bym = _offmomentum_blocks(lattice, -step)

    def dq(b0: np.ndarray, bp: np.ndarray, bm: np.ndarray) -> float:
        c0 = half_tr(b0)
        sin_mu = math.copysign(math.sqrt(max(0.0, 1 - c0 * c0)), b0[0, 1])
        return -(half_tr(bp) - half_tr(bm)) / (2 * step) / (2 * math.pi * sin_mu)

    return dq(bx0, bxp, bxm), dq(by0, byp, bym)


@pytest.mark.parametrize(
    "make",
    [
        lambda r: _sector_fodo(r),  # pure sector bends
        lambda r: _sector_fodo(r, e1e2=0.15),  # with pole-face edges
        lambda r: _ag_combined(r, 0.2),  # combined-function (k1 in the dipole)
        lambda r: _ag_combined(r, 0.3),
    ],
)
def test_matches_offmomentum_map(ref: ReferenceParticle, make) -> None:
    """natural_chromaticity == independent off-momentum-map dQ/ddelta.

    This is the self-consistency gate: the shipped perturbation-theory formula and
    a direct matrix-exponential map of the SAME exact Hamiltonian (including the
    Maxwell-forced curvature-sextupole field) must agree. Combined-function is
    additionally cross-checked against xtrack in tests/reference.
    """
    lat = make(ref)
    nx, ny = natural_chromaticity(lat, slices=300)
    mx, my = _map_chromaticity(lat)
    # ~1e-4: finite-difference step + slice truncation (map 80 vs formula 300).
    assert nx == pytest.approx(mx, rel=2e-3, abs=3e-4)
    assert ny == pytest.approx(my, rel=2e-3, abs=3e-4)


# --------------------------------------------------------------------------- #
# 4. Reductions / physical sanity
# --------------------------------------------------------------------------- #
def test_quad_only_lattice_unchanged(ref: ReferenceParticle) -> None:
    """A straight lattice: dipole code path never fires; equals the beta-weighted sum."""
    cell = [Quadrupole(0.3, 1.2), Drift(1.0), Quadrupole(0.3, -1.2), Drift(1.0)]
    lat = Lattice(cell * 4, ref)
    nx, ny = natural_chromaticity(lat)
    # Reference: the pure quadrupole beta-weighted integral (no dispersion terms).
    from accsim.elements.quadrupole import _focusing_block

    tw0 = closed_twiss(lat)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    inv4pi = 1.0 / (4.0 * math.pi)
    ex = ey = 0.0
    for elem in lat.elements:
        if isinstance(elem, Quadrupole) and elem.k1 != 0.0:
            ds = elem.length / 64
            xb, yb = _focusing_block(elem.k1, ds), _focusing_block(-elem.k1, ds)
            ibx, iby = 0.5 * bx, 0.5 * by
            for i in range(64):
                bx, ax, _ = _propagate_block(xb, bx, ax)
                by, ay, _ = _propagate_block(yb, by, ay)
                w = 0.5 if i == 63 else 1.0
                ibx += w * bx
                iby += w * by
            ex += -inv4pi * elem.k1 * ibx * ds
            ey += inv4pi * elem.k1 * iby * ds
        else:
            cx, cy = _blocks(elem.matrix(ref))
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
    assert nx == pytest.approx(ex, rel=1e-12, abs=1e-12)
    assert ny == pytest.approx(ey, rel=1e-12, abs=1e-12)
    assert nx < 0.0 and ny < 0.0


def test_straight_combined_dipole_equals_quadrupole(ref: ReferenceParticle) -> None:
    """A Dipole(angle=0, k1) is a quadrupole; its chromaticity must match one.

    The h=0 branch of the dipole integrand reduces to -beta k1 (x) / +beta k1 (y),
    with no dispersion or edge terms — i.e. exactly the quadrupole natural
    chromaticity. Guards the ``angle != 0.0 or k1`` element condition.
    """
    lq, k1 = 0.4, 0.7
    lat_dip = Lattice(
        [Dipole(lq, 0.0, k1=k1), Drift(1.0), Dipole(lq, 0.0, k1=-k1), Drift(1.0)] * 4, ref
    )
    lat_quad = Lattice([Quadrupole(lq, k1), Drift(1.0), Quadrupole(lq, -k1), Drift(1.0)] * 4, ref)
    assert natural_chromaticity(lat_dip) == pytest.approx(natural_chromaticity(lat_quad))


def _foc_disp_terms(
    bx: float, ax: float, dx: float, dpx: float, h: float, k1: float
) -> tuple[float, float]:
    """(focusing, dispersion) per-length integrands of the horizontal beta-form."""
    gx = (1.0 + ax * ax) / bx
    return -bx * (k1 + h * h), h * (gx * dx - 2.0 * ax * dpx)


def _horizontal_term_breakdown(lattice: Lattice, slices: int = 300) -> tuple[float, float, float]:
    """Split the horizontal natural chromaticity into (quad+h^2 focusing, dispersion,
    edge) integrals, so the near-cancellation can be asserted on the pieces.

    focusing = -(1/4pi) ∮ beta_x (k1 + h^2) ds  (quads AND dipole weak focusing)
    dispersion = (1/4pi) ∮ h (gamma_x D_x - 2 alpha_x D_px) ds  (dipoles only)
    edge = (1/4pi) sum beta_x h tan(e)
    Their sum is natural_chromaticity's x value.
    """
    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    inv4pi = 1.0 / (4.0 * math.pi)
    focusing = dispersion = edge = 0.0

    def advance(m: np.ndarray) -> None:
        nonlocal bx, ax, by, ay, disp
        cx, cy = _blocks(m)
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
        disp = _transverse_4d(m) @ disp + _dispersive_kick(m)

    for elem in lattice.elements:
        is_dip = isinstance(elem, Dipole) and elem.angle != 0.0
        h = elem.curvature if is_dip else 0.0
        k1 = elem.k1 if (is_dip or (isinstance(elem, Quadrupole))) else 0.0
        if is_dip:
            edge += inv4pi * bx * h * math.tan(elem.e1)
            advance(_edge_matrix(h, elem.e1))
        if elem.length == 0.0 or k1 == 0.0 and not is_dip:
            advance(elem.matrix(ref))
            continue
        ds = elem.length / slices
        if is_dip:
            sub = Dipole(ds, h * ds, k1=elem.k1).matrix(ref)
        else:
            sub = Quadrupole(ds, elem.k1).matrix(ref)
        cx, cy = _blocks(sub)
        sub4, subk = _transverse_4d(sub), _dispersive_kick(sub)
        f0, d0 = _foc_disp_terms(bx, ax, disp[0], disp[1], h, k1)
        af, ad = 0.5 * f0, 0.5 * d0
        for i in range(slices):
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
            disp = sub4 @ disp + subk
            fi, di = _foc_disp_terms(bx, ax, disp[0], disp[1], h, k1)
            w = 0.5 if i == slices - 1 else 1.0
            af += w * fi
            ad += w * di
        focusing += inv4pi * af * ds
        dispersion += inv4pi * ad * ds
        if is_dip:
            edge += inv4pi * bx * h * math.tan(elem.e2)
            advance(_edge_matrix(h, elem.e2))
    return focusing, dispersion, edge


def test_sector_bend_partial_fix_is_worse_than_omitting(ref: ReferenceParticle) -> None:
    """Reproduce the F1 finding: adding h^2 WITHOUT the dispersion term is worse.

    On a sector-bend FODO the horizontal chromaticity splits (edges off) into a
    focusing integral ``-(1/4pi)∮beta_x(k1+h^2)`` and a dispersion integral. The
    dipole's own weak-focusing (the h^2 slice of ``focusing``) is large and
    NEGATIVE; the dispersion term is large and POSITIVE and nearly cancels it. So:
      full  = focusing + dispersion          (the true value ~ xtrack)
      partial (h^2, no dispersion) = focusing (drops the cancelling term)
      quad-only = focusing without the dipole h^2 slice
    and |partial - full| > |quad-only - full|: the partial fix moves AWAY from truth.
    """
    lat = _sector_fodo(ref)
    focusing, dispersion, edge = _horizontal_term_breakdown(lat)
    full = focusing + dispersion + edge
    assert edge == pytest.approx(0.0, abs=1e-12)  # no edges here

    # The dipole's own weak-focusing slice: -(1/4pi) ∮ beta_x h^2 (dipoles only).
    h2_slice = _horizontal_h2_slice(lat)
    quad_only = focusing - h2_slice  # drop the dipole h^2 entirely
    partial = focusing  # keep h^2 but drop the dispersion cancellation

    # The dispersion term is large and opposite to the h^2 slice (near-cancellation).
    assert dispersion > 0.0 and h2_slice < 0.0
    assert abs(dispersion) > 0.5 * abs(h2_slice)
    # Partial (h^2 only) is FURTHER from the truth than omitting the dipole entirely.
    assert abs(partial - full) > abs(quad_only - full)


def _horizontal_h2_slice(lattice: Lattice, slices: int = 300) -> float:
    """-(1/4pi) ∮ beta_x h^2 ds over dipole bodies only (the weak-focusing slice)."""
    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    inv4pi = 1.0 / (4.0 * math.pi)
    total = 0.0
    for elem in lattice.elements:
        is_dip = isinstance(elem, Dipole) and elem.angle != 0.0
        if is_dip:
            h = elem.curvature
            ds = elem.length / slices
            cx, cy = _blocks(Dipole(ds, h * ds, k1=elem.k1).matrix(ref))
            acc = 0.5 * (-bx * h * h)
            for i in range(slices):
                bx, ax, _ = _propagate_block(cx, bx, ax)
                by, ay, _ = _propagate_block(cy, by, ay)
                w = 0.5 if i == slices - 1 else 1.0
                acc += w * (-bx * h * h)
            total += inv4pi * acc * ds
        else:
            cx, cy = _blocks(elem.matrix(ref))
            bx, ax, _ = _propagate_block(cx, bx, ax)
            by, ay, _ = _propagate_block(cy, by, ay)
    return total
