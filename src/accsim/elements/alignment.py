r"""Alignment geometry: what it means for an element to sit somewhere else (K1, K2).

Every element's own map is written in the element's **own** frame. A misaligned
element needs two extra maps around it — one to get from the lattice frame into the
magnet's frame at the entrance, one to get back out at the exit — and those two are
what this module builds.

For a **translation** (K1) and for a **roll of a straight element** (K2) the pair is
trivially inverse: step in, step back out. That is why K1 needed nothing from here at
all, and why a rolled quadrupole is exactly the conjugation ``R(-phi) M R(+phi)``.

A rolled **bending** magnet is the case where they are *not* inverse, and it is the
whole of K2. A bend carries the reference frame around with it, so rolling the magnet
puts its exit face somewhere the lattice does not expect: displaced, pitched and
yawed, with only part of the entrance roll left over. What comes back out is a
general **rigid motion**, built by :func:`frame_change`.

Conventions
-----------
Rigid motions are 4x4 homogeneous matrices ``[[R, t], [0, 1]]`` acting on
**geometric** column vectors ``(x, y, s)`` — the frame's own axis order, *not* the
phase-space order. ``R``'s columns are the new frame's axes expressed in the old
frame, and ``t`` is the new frame's origin in the old frame, so ``M`` maps a point's
new-frame coordinates to its old-frame ones.

The sense of every rotation is pinned against xtrack (``rot_s_rad_no_frame``,
``SRotation``) in ``tests/reference/test_roll_xtrack.py`` rather than argued here —
see ``docs/CONVENTIONS.md`` -> *Misalignments — the roll*.
"""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle

__all__ = ["arc_motion", "frame_change", "roll_motion", "s_rotation"]


def s_rotation(phi: float) -> np.ndarray:
    r"""The 6x6 map of a coordinate frame rotated by ``phi`` about the ``s`` axis.

    This is the **passive** rotation — the particle does not move, the axes do:

        x -> +cos(phi) x + sin(phi) y,      y -> -sin(phi) x + cos(phi) y

    and the same on ``(px, py)``. Byte-identical to xtrack's ``SRotation(angle=phi)``
    (which is what a positive ``roll`` is pinned against), and exactly the identity
    at ``phi = 0`` — a design element must not pay for the machinery.

    Longitudinal coordinates are untouched: a rotation about ``s`` moves no point
    along ``s``, so there is no path-length change to account for. That is precisely
    what stops being true for the exit of a rolled *bend* (:func:`frame_change`).
    """
    M = np.eye(DIM)
    if phi == 0.0:
        return M
    c, s = math.cos(phi), math.sin(phi)
    M[X, X] = M[Y, Y] = M[PX, PX] = M[PY, PY] = c
    M[X, Y] = M[PX, PY] = s
    M[Y, X] = M[PY, PX] = -s
    return M


def roll_motion(phi: float) -> np.ndarray:
    """Rigid motion of a frame rolled by ``phi`` about its own ``s`` axis, ``(4, 4)``.

    The counterpart of :func:`s_rotation` on the *geometry* side: this rotates the
    frame, where :func:`s_rotation` transforms coordinates in it, so the two are
    transposes of one another in the transverse block.
    """
    c, s = math.cos(phi), math.sin(phi)
    M = np.eye(4)
    M[0, 0] = M[1, 1] = c
    M[0, 1] = -s
    M[1, 0] = s
    return M


def arc_motion(angle: float, rho: float) -> np.ndarray:
    r"""Rigid motion of a horizontal bend's design arc, entry frame -> exit frame.

    The reference orbit turns by ``angle`` in the ``x``-``s`` plane on a radius
    ``rho``, curving toward **negative** ``x`` (accsim's and MAD-X's sign: a positive
    bend angle deflects the beam so that a *higher*-momentum particle, bending less,
    ends up at positive ``x`` — which is the ``R16 > 0`` of
    :class:`~accsim.elements.dipole.Dipole`). So the exit frame's origin sits at

        (x, y, s) = (rho (cos angle - 1), 0, rho sin angle)

    with its ``s`` axis turned by ``angle`` about ``y``. Pinned against xtrack's own
    arc transport, which is what its curved-misalignment header composes.
    """
    c, s = math.cos(angle), math.sin(angle)
    M = np.eye(4)
    M[0, 0] = c
    M[0, 2] = -s
    M[2, 0] = s
    M[2, 2] = c
    M[0, 3] = rho * (c - 1.0)
    M[2, 3] = rho * s
    return M


def frame_change(motion: np.ndarray, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
    r"""Affine 6D map ``(M, k)`` for coordinates given in one frame, wanted in another.

    ``motion`` is the rigid motion ``[[R, t], [0, 1]]`` that takes the **new** frame's
    coordinates to the **old** frame's — i.e. the new frame sits at ``t`` with axes
    ``R``. The particle arrives described in the new frame, at that frame's ``s = 0``
    plane, and must come out described in the old frame at *its* ``s = 0`` plane.

    The exact map is three steps, and the third is the one a pure rotation never
    needs:

    1. **Move the point and the momentum**: ``r = R (x, y, 0) + t`` and
       ``p = R (px, py, pz)``, with ``pz = sqrt((1 + delta)^2 - px^2 - py^2)``.
    2. **Drift back to the plane** ``s = 0`` of the old frame, a straight line in a
       field-free region, by ``ds = -r_s``.
    3. **Charge the time, not the distance.** A frame change advances no *design*
       length, so ``s`` in ``zeta = s - beta0 c t`` does not move while ``t`` does.
       That is the sign trap of the whole construction: the correction is
       ``-ds (1 + delta) / (rvv pz)`` with **no** compensating ``+ds``, where a real
       drift of the same length would have one.

    Returned is its **affine linearisation about the origin**, derived in sympy
    (``tests/analytic/test_roll.py``) rather than by hand, which is exact for
    accsim's linear elements and is what :meth:`~accsim.elements.element.Element.matrix`
    and :meth:`~accsim.elements.element.Element.kick` compose. Writing
    ``a = t_s / R_ss`` (the drift-back length, negated), the whole result is

        k      = (t_x - a R_xs,  R_xs,  t_y - a R_ys,  R_ys,  a,  0)
        M_pos  = R_2x2 - outer(R_[xy]s, R_s[xy]) / R_ss        (the transverse block)
        M_ang  = R_2x2                                          (angles just rotate)
        M[.,px/py]  = -a * M_pos                                (the drift back)
        M[.,delta]  = R_[xy]s                                   (angles, via pz)
        M[zeta,.]   = R_s[xy] / R_ss  and  M[zeta,delta] = -a / gamma0^2

    Two limits worth naming, both asserted: ``motion = identity`` gives exactly
    ``(I, 0)``, and a pure translation along ``s`` gives exactly a
    :class:`~accsim.elements.drift.Drift` of length ``-t_s`` **plus** a constant
    ``zeta`` shift of ``+t_s`` — the drift a design element would have been charged
    for, refunded.
    """
    R = np.asarray(motion, dtype=float)[:3, :3]
    t = np.asarray(motion, dtype=float)[:3, 3]
    rss = R[2, 2]
    if rss <= 0.0:
        raise ValueError(
            f"a frame change must not turn the beam by 90 degrees or more (R_ss = {rss}); "
            "the particle would never reach the new frame's transverse plane"
        )
    a = t[2] / rss  # the drift back to the old frame's s = 0 plane is -a long

    # Transverse position/angle blocks. Row/column order is (x, y) here, mapped onto
    # accsim's (x, px, y, py) at the end.
    rot = R[:2, :2]  # how the transverse axes themselves turn
    fwd = R[:2, 2]  # the new frame's s axis, seen transversely: the constant angles
    down = R[2, :2] / rss  # how a transverse offset reads as a longitudinal one
    pos = rot - np.outer(fwd, down)

    M = np.eye(DIM)
    for i, ci in enumerate((X, Y)):
        for j, cj in enumerate((X, Y)):
            M[ci, cj] = pos[i, j]
            M[ci, cj + 1] = -a * pos[i, j]  # +1: X -> PX, Y -> PY
            M[ci + 1, cj + 1] = rot[i, j]
        M[ci + 1, DELTA] = fwd[i]
        M[ZETA, ci] = down[i]
        M[ZETA, ci + 1] = -a * down[i]
    M[ZETA, DELTA] = -a / ref.gamma0**2

    k = np.zeros(DIM)
    k[X] = t[0] - a * fwd[0]
    k[Y] = t[1] - a * fwd[1]
    k[PX] = fwd[0]
    k[PY] = fwd[1]
    k[ZETA] = a
    return M, k
