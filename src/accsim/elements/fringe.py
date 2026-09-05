r"""The **gradient** pole face: the hard-edge multipole fringe, and the quadrupole wedge.

P2 (i) gave a face its *dipole* fringe (:func:`~accsim.elements.dipole.hard_edge_fringe_map`)
and P3 (a) gave it its *rotation* (:func:`~accsim.elements.dipole.wedge_map`). Both stop at
the dipole component of the body field. A face terminates whatever multipole the body
carries, and this module is the ``k1`` half of that -- the piece a plain
:class:`~accsim.elements.quadrupole.Quadrupole` needs and the piece that made
``Dipole(fringe=True, k1=...)`` refuse until P3 (b).

Two maps live here, and they are **different physics** despite sharing a face:

- :func:`multipole_fringe_map` -- the gradient's *termination*. A **cubic** point
  transformation, so it moves nothing at first or second order: neither ``matrix``, nor
  the tunes, nor the second-order map of :mod:`accsim.taylor` can see it. It exists at
  every face, rotated or not.
- :func:`quad_wedge_map` -- the gradient's *wedge*, the sliver of body gradient between a
  **rotated** face plane and the sector plane. **Quadratic**, so it does move the
  second-order map; it vanishes identically at zero face angle.
"""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle

__all__ = ["multipole_fringe_map", "quad_wedge_map"]


def multipole_fringe_map(
    state: np.ndarray, k1: float, ref: ReferenceParticle, *, exit_face: bool = False
) -> np.ndarray:
    r"""The **hard-edge gradient fringe** at one face — exactly symplectic, cubic.

    Where the dipole fringe of :func:`~accsim.elements.dipole.hard_edge_fringe_map` is
    what the *bending* field's termination does, this is what the **gradient's**
    termination does. With ``kappa = ±k1/(1 + delta)`` (``+`` entering, ``-`` leaving)
    the map is the point transformation

        x -> x + kappa (x^3 + 3 x y^2)/12,      y -> y - kappa (3 x^2 y + y^3)/12

    with the momenta carried by the **inverse transpose** of its Jacobian and a matching
    arrival-time term. That is the ``min_order = 1`` case of MAD-NG's (and xtrack's)
    ``MultFringe``, which :class:`~accsim.elements.quadrupole.Quadrupole` reaches through
    ``edge_entry_active`` and :class:`~accsim.elements.dipole.Dipole` through
    ``edge_*_model="full"``.

    Where the ``1/12`` comes from
    -----------------------------
    It is **derived, not transcribed** — ``tests/analytic/test_multipole_fringe.py``
    does it symbolically, and the derivation is the only gate with teeth here (see
    *What cannot see this map* below). Two steps:

    1. **Maxwell fixes the coefficient.** A quadrupole with a longitudinal profile
       ``g(s)`` is not ``g(s)`` times the 2D field: ``grad^2 psi = 0`` forces a
       correction. For a 2D harmonic of degree ``N``, ``grad_perp^2 (r^2 u) =
       4(N + 1) u``, so ``psi = g u - g'' r^2 u / (4(N+1))`` and a quadrupole
       (``u = -k1 x y``, ``N = 2``) gets ``-1/12``. **That is the 12.**
    2. **The Lorentz force turns it into a map.** ``curl B = 0`` puts a longitudinal
       ``b_s = -g' u`` in the fringe, and the ``g''`` term above puts an ``O(1/l)``
       transverse field there. Integrating ``dp_x/ds = y' b_s - b_y`` along the
       unperturbed line, subtracting the hard-edge model's own integral, and letting the
       fringe extent ``l -> 0`` leaves exactly the four displacements above — the
       ``g'`` and ``g''`` moments (``int g' = 1``, ``int s g'' = -1``) are what survive,
       which is why the answer is **profile-independent**.

    That derivation also fixes the **entrance** sign independently of any reference code,
    and the exit's by ``int g' = -1`` when the field switches off instead of on.

    Symplecticity, exactly
    ----------------------
    The map is the type-2 generating function

        F2 = P . g(q, p_tau) + t p_tau',      g = q - f(q)/(1 + delta),

    a point transformation in ``(x, y)`` whose parameter depends on ``p_tau``. Reading
    ``p = (dg/dq)^T P`` gives the inverse-transpose momentum rule; reading
    ``T = dF2/dp_tau`` gives the ``zeta`` term, ``(E/E0) (P.f)/(1+delta)^3``. So it is
    symplectic **at any amplitude**, not to the order it is written — the same property
    the dipole fringe's generating function has, and the reason a transcription slip
    shows up as a broken Poisson bracket.

    What cannot see this map
    ------------------------
    Every entry is **third** order in the coordinates. Its Jacobian at the origin is the
    identity, so ``matrix``, every tune, beta, dispersion and chromaticity, *and the whole
    second-order map of* :mod:`accsim.taylor` are untouched — MAD-X's ``sectormap`` and
    PTC at ``no = 2`` are blind to it. It is the one gap of the four P1 found that P1
    could not have found. Worse, the map is exactly **linear in ``k1``** and exactly
    **cubic in the amplitude**, so a uniform mis-scale of the coefficient passes the
    ``k1`` scaling, the amplitude order *and* symplecticity. That is J1's lesson: the
    structural gates are all blind to the number, and only the derivation above and a
    reference code pin it.

    ``k1 = 0`` returns the input bit for bit. ``state`` is a ``(6,)`` vector or a
    ``(6, n)`` bunch.
    """
    st = np.asarray(state, dtype=float)
    if k1 == 0.0:
        return st.copy()

    x, y = st[X], st[Y]
    rpp = 1.0 / (1.0 + st[DELTA])
    kappa = -k1 if exit_face else k1

    x2, y2, xy = x * x, y * y, x * y
    # f = (fx, fy) is the displacement *before* the 1/(1+delta): x -> x - fx/(1+delta).
    # Note it is not a gradient field -- d fx/dy = -d fy/dx -- so no scalar potential in
    # (x, y) generates it; the generating function is the F2 above, in the momenta.
    fx = -kappa * (x2 * x + 3.0 * x * y2) / 12.0
    fy = kappa * (3.0 * x2 * y + y2 * y) / 12.0
    # The four partials of f. fxx = -fyy and fxy = -fyx, which is that same antisymmetry.
    fxx = -kappa * (x2 + y2) * 0.25
    fxy = -kappa * xy * 0.5
    fyx = kappa * xy * 0.5
    fyy = kappa * (x2 + y2) * 0.25

    # J = [[a, c], [b, d]] is the Jacobian of the (x, y) point transformation; the momenta
    # take J^-T, which is what makes the pair canonical.
    a = 1.0 - fxx * rpp
    b = -fyx * rpp
    c = -fxy * rpp
    d = 1.0 - fyy * rpp
    det = a * d - b * c

    new_px = (d * st[PX] - b * st[PY]) / det
    new_py = (a * st[PY] - c * st[PX]) / det

    # zeta = beta0 * tau with tau conjugate to p_tau; dg/dp_tau = f (E/E0)/(beta0 (1+delta)^3),
    # so zeta picks up (E/E0)/(1+delta)^3 times P.f -- the same (E/E0) factor, formed the
    # same way, as hard_edge_fringe_map and wedge_map.
    e_over_e0 = np.hypot(ref.momentum_eV * (1.0 + st[DELTA]), ref.mass_eV) / ref.total_energy_eV

    out = st.copy()
    out[X] = x - fx * rpp
    out[Y] = y - fy * rpp
    out[PX] = new_px
    out[PY] = new_py
    out[ZETA] = st[ZETA] + e_over_e0 * rpp * rpp * rpp * (new_px * fx + new_py * fy)
    return out


def quad_wedge_map(state: np.ndarray, theta: float, k1: float) -> np.ndarray:
    r"""The **gradient's wedge** at a rotated face — the ``k1`` analogue of the dipole wedge.

    :func:`~accsim.elements.dipole.wedge_map` integrates the sliver of *bending* field
    between the rotated face plane and the sector plane. A combined-function magnet's
    gradient lives in that same sliver, and this is its share:

        px -> px + k1 theta (y^2/2 - x^2),      py -> py + k1 theta x y

    with ``theta = -e`` at both faces (the wedge is a slice of the body's own field, which
    does **not** switch sign at the exit — the rule P3 (a) established for the dipole
    wedge, and xtrack applies it here too by passing ``knorm[1]`` un-negated).

    **It is** :func:`~accsim.elements.dipole.curvature_sextupole_kick` **with a different
    length, and that is not a coincidence.** Both are the quadrupole potential integrated
    over a path length that varies *linearly in* ``x``: for the curvature sextupole the
    ``(1 + h x)`` metric factor over the body's length ``L``, giving ``h k1 L``; for the
    wedge the geometric sliver ``-x tan(e) ~ x theta``, giving ``k1 theta``. Since a
    bend's ``theta = h L``, the two integrated strengths are the same expression. The
    ``2:-1`` ratio between the ``x^2`` and ``y^2/2`` terms is Maxwell's, in both. The
    equality is *asserted* in ``tests/analytic/test_multipole_fringe.py`` rather than
    assumed, and this map is kept separate so the two are not confused.

    **Unlike** :func:`multipole_fringe_map` **this is quadratic**, so it does move the
    second-order map — which is a gain, not a cost: it hands MAD-X's ``sectormap`` back
    as a sharp arbiter on this half of the face, where it is vacuous on the other.

    ``theta = 0`` or ``k1 = 0`` returns the input bit for bit. ``state`` is a ``(6,)``
    vector or a ``(6, n)`` bunch.
    """
    st = np.asarray(state, dtype=float)
    strength = k1 * theta
    if strength == 0.0:
        return st.copy()
    x, y = st[X], st[Y]
    out = st.copy()
    out[PX] = st[PX] + strength * (y * y * 0.5 - x * x)
    out[PY] = st[PY] + strength * x * y
    return out
