r"""Geometry: where the machine actually sits, in laboratory coordinates (R1).

Every optics quantity in this package lives in the **beam** frame — ``x`` is a displacement
from a reference curve that is assumed to exist and to close. Nothing here had ever asked
where that curve *goes*. This module does: given a :class:`~accsim.lattice.Lattice`, it walks
the reference curve through the laboratory and reports the position and orientation of the
beam frame at every element boundary.

It is **geometry, not dynamics**: no particle is tracked, no map is evaluated, and the only
things read off an element are its ``length`` and — for a
:class:`~accsim.elements.dipole.Dipole` — its ``angle`` and its design ``tilt``. The angle is
doing work. Since
Q2 a bend's *field* (``k0``) and its *geometry* (``h = angle/L``) are separate numbers, and a
tapered ring has ``k0 != h`` in every magnet. **The survey follows the geometry.** A survey
that reached for ``k0`` would rescale the whole ring while still closing exactly, which is
why ``survey(taper(lattice)) == survey(lattice)`` is a gate in
``tests/analytic/test_survey.py`` rather than a remark here. Both reference codes are blind
to the field the same way, and that was measured before this module was written.

The frame
---------
Right-handed laboratory axes ``(X, Y, Z)``. The walk starts at the origin with the beam
pointing along ``+Z``, and ``W`` is the 3x3 rotation carrying local beam coordinates
``(x, y, s)`` into the laboratory. With the machine in the horizontal plane that is a single
yaw ``theta`` about ``Y``,

    W = [[ cos theta, 0, sin theta],
         [         0, 1,         0],
         [-sin theta, 0, cos theta]],

and a bend of angle ``a`` advances the frame by the arc's chord and turns it by ``-a``:

    dv_local = rho (cos a - 1, 0, sin a),     rho = L / a,      theta -> theta - a.

**A positive bend angle moves the machine towards negative ``X``.** That sign is not a
choice made here — it is what both `xtrack` and MAD-X report, agreeing with each other to
``1e-16`` on a 90-degree bend, and this module is pinned against both.

Nothing divides by the curvature. Writing ``1 - cos a = 2 sin^2(a/2)``, the chord is

    dv_x = -L (a/2) sinc(a/2)^2,      dv_z = L sinc(a),

which is exact at ``a = 0`` (a straight element advances by ``(0, 0, L)``) with no branch and
no cancellation for a weak bend — the same device
:func:`~accsim.elements.dipole.exact_sector_bend_map` uses, for the same reason.

**Rows are element boundaries: ``N + 1`` of them, not ``N``, and the names are kept beside
them rather than on them.** This is the module's one design decision, and it comes from a
measurement. Both reference codes walk the *same* ``N + 1`` boundaries — what differs is
which element each row is **named** after: `xtrack`'s ``line.survey()`` names a boundary
after the element it *begins* (and calls the last one ``_end_point``), MAD-X's ``SURVEY``
names it after the element it *ends* (and prepends ``$start``, appending a duplicate
``$end``). The rows themselves agree entry for entry. Adopting either naming would bake in a
half-element offset that the other code then appears to disagree about, so this table carries
``N + 1`` poses and ``N`` names side by side, and each reference leg is a direct comparison.

**Out of the plane: the design tilt (V1).** Until V1 the survey was planar by construction,
because no element could bend out of the horizontal plane. A bend with a design ``tilt``
(MAD-X ``TILT``, xtrack's plain ``rot_s_rad``) bends in its own plane, turned by ``tilt``
about the beam axis, so the walk becomes a genuine 3D composition. With ``T = R_z(+tilt)``
the step and the turn are the planar ones seen through ``T``,

    dv_local = T . rho (cos a - 1, 0, sin a),      W -> W . T R_y(-a) T^T,

and ``tilt = +pi/2`` on a positive angle turns the machine **down** (``Y`` decreasing) — the
sense both reference codes report, agreeing with each other and with this walk to ``1e-15``.
``roll`` still does nothing here: it is a *misalignment* (K2 — the magnet turns, the frame
does not), and a misalignment moves the magnet, not the curve.

``theta``, ``phi`` and ``psi`` are then read off ``W`` in MAD-X's order,
``W = R_y(theta) R_x(-phi) R_z(psi)`` — the inverse xtrack's ``get_angles_from_w`` uses —
and unwrapped along the ring. **A machine with no tilted element takes the old planar path,
bit for bit**: ``theta`` accumulated as ``-sum(angle)`` and ``phi = psi = 0`` exactly, so
nothing R1 measured has moved.

``theta`` is **not wrapped**: a full turn reports ``-2 pi``, not ``0``, which is what both
reference codes do.

The module is ``accsim.geometry`` and the function is ``survey`` — deliberately not the same
word. ``survey`` is the domain term both reference codes use (`xtrack`'s ``line.survey()``,
MAD-X's ``SURVEY``) and is re-exported at the package top level, so naming the *module* the
same would have made ``accsim.survey`` resolve to the function and shadow the submodule. No
other module in this package is shadowed that way and this one does not become the first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lattice import Lattice

__all__ = ["SurveyTable", "survey"]

TWO_PI = 2.0 * math.pi


def _sinc(u: float) -> float:
    """``sin(u)/u``, continued to ``1`` at ``u = 0``.

    The scalar twin of :func:`accsim.elements.dipole._sinc`, and it is here for the reason
    that docstring gives: writing the arc through it removes every division by the curvature,
    so a straight element is the ``a -> 0`` limit of the same expression rather than a
    special case. ``sin(u)/u`` is accurate for small ``u`` — ``sin(u)`` rounds to ``u`` —
    so only the exact zero needs continuing.
    """
    return 1.0 if u == 0.0 else math.sin(u) / u


def _yaw(theta: float) -> np.ndarray:
    """The rotation carrying local ``(x, y, s)`` into the laboratory for a yaw ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _tilt(tilt: float) -> np.ndarray:
    """``R_z(tilt)``: a bend's own axes, turned by its design ``tilt`` about ``s``.

    Positive ``tilt`` carries the bend's own ``x`` axis towards local ``+y`` — the same sense
    as :func:`~accsim.elements.alignment.s_rotation` on the phase-space coordinates, which is
    what ties the survey to the map (``tests/analytic/test_design_tilt.py``: the dispersion
    points away from the way the survey turns, for every tilt).
    """
    c, s = math.cos(tilt), math.sin(tilt)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _angles(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(theta, phi, psi)`` from ``(N+1, 3, 3)`` frames, ``W = R_y(theta) R_x(-phi) R_z(psi)``.

    The same inverse xtrack applies (``get_angles_from_w``) and MAD-X tabulates, unwrapped
    row to row so a full turn reads ``-2 pi`` as on the planar path.
    """
    theta = np.arctan2(W[:, 0, 2], W[:, 2, 2])
    psi = np.arctan2(W[:, 1, 0], W[:, 1, 1])
    phi = np.arctan2(W[:, 1, 2], W[:, 1, 1] / np.cos(psi))
    return np.unwrap(theta), np.unwrap(phi), np.unwrap(psi)


def _step(length: float, angle: float) -> np.ndarray:
    r"""The local displacement across a body of length ``length`` bending by ``angle``.

    ``rho (cos a - 1, 0, sin a)`` with ``rho = length/angle``, written so that neither
    component ever divides by the angle:

        dv_x = -L (a/2) sinc(a/2)^2      [= -rho (1 - cos a), the sagitta's parent]
        dv_z = L sinc(a)                 [= rho sin a, the chord along the entry direction]

    At ``angle = 0`` this is exactly ``(0, 0, L)``.
    """
    half = 0.5 * angle
    return np.array([-length * half * _sinc(half) ** 2, 0.0, length * _sinc(angle)])


@dataclass(frozen=True)
class SurveyTable:
    """The reference curve's pose at every element boundary.

    Every array has ``N + 1`` entries for a lattice of ``N`` elements: row ``i`` is the pose
    at the **entrance** of element ``i``, and the last row is the exit of the last element.
    Both reference codes walk these same ``N + 1`` boundaries and differ only in which
    element each row is *named* after, which is why the names live in :attr:`names` beside
    the rows rather than on them — see the module docstring.

    ``Y``, ``phi`` and ``psi`` are identically zero on a machine with no design tilt — exactly,
    not to round-off (see the module docstring).
    """

    #: Arc length along the reference curve [m], ``0`` at the first row.
    s: np.ndarray
    #: Laboratory position [m]. ``Y`` is identically zero on a machine with no tilt.
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    #: Orientation [rad], ``W = R_y(theta) R_x(-phi) R_z(psi)``. ``theta`` is the yaw about
    #: ``Y`` and accumulates **unwrapped**; ``phi`` (pitch) and ``psi`` (roll) are identically
    #: zero on a machine with no tilt.
    theta: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    #: ``(N+1, 3, 3)`` rotations carrying local ``(x, y, s)`` into the laboratory.
    W: np.ndarray
    #: The element names, in lattice order — ``N`` of them, one fewer than the rows.
    names: tuple[str, ...]

    def __len__(self) -> int:
        """The number of **rows**, ``N + 1``."""
        return int(self.s.size)

    @property
    def positions(self) -> np.ndarray:
        """``(N+1, 3)`` laboratory positions, one row per boundary."""
        return np.column_stack((self.X, self.Y, self.Z))

    @property
    def closure_gap(self) -> float:
        """Distance [m] between the last boundary and the first.

        **A near-vacuous gate, and worth saying so.** For a ring this is zero when the bend
        angles sum to a multiple of ``2 pi`` — a property of the *lattice*, not of this
        walker: any implementation that accumulates rotations in any consistent order closes.
        What closure does not see is where the curve went in between, which is why the sharp
        tests are the per-element chord and sagitta.
        """
        return float(np.linalg.norm(self.positions[-1] - self.positions[0]))

    @property
    def closure_angle(self) -> float:
        """How far the final frame misses the initial one [rad], into ``[0, pi]``.

        On a planar machine ``theta`` is the whole orientation and it is unwrapped, so a
        closed ring ends at ``-2 pi k``; this is the distance from that, i.e. ``|theta[-1]|``
        reduced modulo ``2 pi``. Out of the plane the yaw alone is not the frame, so the
        answer is the angle of the relative rotation ``W[0]^T W[-1]`` — computed through
        ``atan2`` rather than ``arccos``, which loses half its digits near closure.
        """
        if not (np.any(self.phi) or np.any(self.psi)):
            turned = float(self.theta[-1] - self.theta[0])
            return float(abs((turned + math.pi) % TWO_PI - math.pi))
        rel = self.W[0].T @ self.W[-1]
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        return float(math.atan2(0.5 * float(np.linalg.norm(axis)), 0.5 * (np.trace(rel) - 1.0)))

    @property
    def total_angle(self) -> float:
        """The total bend angle turned through [rad] — ``-(theta[-1] - theta[0])``.

        On a machine with tilted bends this is the net **yaw**, not the sum of the bend
        angles: a vertical bend turns the frame without moving ``theta``.
        """
        return float(-(self.theta[-1] - self.theta[0]))


def survey(lattice: Lattice) -> SurveyTable:
    """Walk ``lattice``'s reference curve through the laboratory.

    Starts at the origin pointing along ``+Z`` and returns the pose at every element
    boundary (``N + 1`` rows — see :class:`SurveyTable`).

    Only ``length`` and, for a bending element, ``angle`` and ``tilt`` are read. **Not**
    ``k0`` (Q2's field, which a tapered ring changes in every magnet), not ``e1``/``e2`` (the
    pole faces bound the field, not the reference curve), and not ``dx``/``dy``/``roll`` (a
    misalignment moves the magnet away from the design curve, it does not move the curve).
    Each of those is a test.
    """
    elements = lattice.elements
    n = len(elements)
    tilts = [float(getattr(elem, "tilt", 0.0)) for elem in elements]

    s = np.zeros(n + 1)
    pos = np.zeros((n + 1, 3))
    theta = np.zeros(n + 1)
    W = np.zeros((n + 1, 3, 3))
    W[0] = np.eye(3)

    for i, elem in enumerate(elements):
        length = float(elem.length)
        # ``angle`` is the geometry, and only a bending element has one. Reading it by name
        # rather than by type is deliberate: any future element that bends the reference
        # curve announces it the same way Dipole does.
        angle = float(getattr(elem, "angle", 0.0))
        # The step is expressed in the frame the element is **entered** in, ``W[i]``, and
        # never in the one it leaves in. Two wrong ways to write this line survive a
        # symmetric ring — using ``W[i+1]``, and transposing ``W`` (equivalently flipping
        # the sign of ``theta``) — because the corners of a regular polygon are the same set
        # whichever way you visit them. Both fail on the second row of a ring whose bends
        # and drifts are unequal, which is the fixture that gates them.
        step, turn = _step(length, angle), _yaw(-angle)
        if tilts[i] != 0.0:
            # The bend acts in its own plane, turned by the tilt: the planar step and turn
            # seen through ``T``. Skipped at zero tilt so a flat ring is bit-for-bit R1's.
            T = _tilt(tilts[i])
            step, turn = T @ step, T @ turn @ T.T
        pos[i + 1] = pos[i] + W[i] @ step
        # Composed on the right, so the turn acts in the frame the beam is already in. A
        # planar machine cannot see the order (rotations about a common axis commute); the
        # tilted line in tests/analytic/test_design_tilt.py can, and gates it.
        W[i + 1] = W[i] @ turn
        theta[i + 1] = theta[i] - angle
        s[i + 1] = s[i] + length

    if any(tilts):
        theta, phi, psi = _angles(W)
    else:
        phi, psi = np.zeros(n + 1), np.zeros(n + 1)

    return SurveyTable(
        s=s,
        X=pos[:, 0],
        Y=pos[:, 1],
        Z=pos[:, 2],
        theta=theta,
        phi=phi,
        psi=psi,
        W=W,
        names=tuple(elem.name for elem in elements),
    )
