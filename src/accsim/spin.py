r"""The Thomas-BMT spin precession a tracked particle takes through a magnet (N1).

Every quantity on axes A-M is a property of a particle's *position and momentum*. A
charged particle also carries a **spin**, and this module is the map that rotates it as
the particle crosses a magnet. It is the whole of axis N1: no new state vector, no new
field model, and — because a spin does not act back on the orbit — not one existing
number in the package moves.

**The physics, in one paragraph.** A spin in a magnetic field precesses at the
Thomas-BMT rate. Per unit of path length, in the element's own frame,

    ``dS/ds = Omega x S``,
    ``Omega = -(1/(1 + delta)) [ (1 + G gamma) b_perp + (1 + G) b_par ]``   [rad/m]

where ``b = B/(B rho)_0`` is the element's field in the package's usual normalisation
(:meth:`~accsim.elements.element.Element.normalized_field` — the *same* thing the
radiation kick asks a magnet for, so no element grows a second field model),
``b_par``/``b_perp`` are its components along and across the direction of motion,
``G = (g - 2)/2`` is the species' anomalous magnetic moment
(:attr:`~accsim.reference.ReferenceParticle.anomalous_moment`) and ``gamma`` is the
**particle's** own Lorentz factor, not the reference one.

**Why that normalisation makes the charge disappear.** The textbook form divides by the
particle's magnetic rigidity ``B rho = p/q``, so it carries ``q``; the package's ``b``
already *is* ``B q / p_0`` (that is what makes ``k1`` mean what it means), and the two
factors of ``q`` cancel exactly. An electron ring and a proton ring are therefore the
same expression, and a sign error in the charge cannot hide here.

**The frame rotation, which is half the physics of a bend.** ``Omega`` is written in the
curvilinear frame, and inside a bend that frame *turns*. Composing the two is what makes
a dipole's net effect the famous one: on the design orbit ``b = (0, h, 0)`` is constant,
the BMT rotation is ``-(1 + G gamma) theta`` about ``y``, the frame contributes
``+theta``, and what is left is

    **a rotation by** ``-G gamma theta`` **about** ``y`` — exactly, with no quadrature error.

Summed over a flat ring's ``2 pi`` of bending that is the spin tune ``nu_0 = G gamma``.
This module applies the frame rotation in two halves, one either side of the BMT
rotation, which is both the midpoint rule and what xtrack does — so the two are
comparable element by element rather than only turn by turn.

**Why the spin tune is a control and not a gate.** ``nu_0 = G gamma`` depends only on the
bends summing to ``2 pi`` and on the beam energy. A version of this module whose
transverse coefficient were mis-scaled, or whose quadrupole contribution were missing
altogether, reproduces it *exactly* — on the design orbit a quadrupole has no field at
all. That blindness is asserted in ``tests/analytic/test_spin.py`` rather than hoped
against; the gates that discriminate are:

- **``G = 0`` locks the spin to the direction of motion.** With no anomalous moment the
  BMT rotation *is* the cyclotron rotation, so a spin started along ``p`` stays along
  ``p``. It is exact where the field is (a drift, a sector bend on the design orbit) and
  converges at second order in the slice length where it is not; a mis-scaled
  coefficient, a wrong sign, or a missing frame rotation all leave a residual that does
  not converge at all.
- **A quadrupole at a vertical offset** rotates the spin about ``x`` by
  ``-(1 + G gamma) k1 int y ds / (1 + delta)``, which is what pins the ``(1 + G gamma)``
  factor itself — the ``G = 0`` identity cannot, since the factor is ``1`` there by
  construction.

**Scope and costs.**

- *Thin elements do not precess*, and — unlike radiation — that is an approximation
  rather than a limit. A thin magnet's radiated energy really does vanish with its
  length (``U ~ kappa^2 L``); its *integrated* field does not, so a thin quadrupole's
  true spin rotation is finite and dropping it is a real omission. It is dropped anyway
  because **xtrack's thin** ``Multipole`` **does not rotate spin either** (spin lives
  only in its ``track_magnet`` family), so building it would mean inventing a model with
  no arbiter, which is L5's reason and the one trade this project's validation strategy
  does not make. The cost is precise: a thin-lens ring has **no** vertical spin dynamics,
  so every gate on this axis is built from thick magnets.
- *The field is integrated by the midpoint rule*, exactly as
  :mod:`accsim.radiation_kick` integrates it — one evaluation at the mean of the entry
  and exit coordinates, over the element's own path length. That is exact for a sector
  bend on the design orbit (the field is constant) and second-order accurate otherwise.
  xtrack does *not* evaluate an analytic field at all: its ``magnet_estimate_field``
  back-derives ``B`` from the trajectory's curvature, so the two codes differ at
  ``O(L^2)`` by construction and the reference gate is a **convergence order**, not a
  tolerance.
- *A rolled bend is refused.* A roll of a straight element is a conjugation and the spin
  simply rides through it; for a *bending* magnet the exit face moves (K2's rigid-body
  geometry) and the frame rotation is no longer a rotation about ``y`` in the lattice
  frame. Rather than apply a wrong one, :func:`spin_precession` raises.
- *Not a phase-space map.* A rotation of a unit vector has nothing to do with
  symplecticity, and ``matrix()``/``kick()`` are untouched. The invariant that bounds
  axis L — ``matrix()`` is the exact origin Jacobian of ``track()`` — is unaffected.
- *Radiation and precession are evaluated on the **same** un-radiated traversal, and
  that is a choice.* With ``radiation`` on, ``Element._track_impl`` hands both this
  function and :func:`accsim.radiation_kick.radiation_kick` the *same* ``before``/
  ``after`` pair, so the spin is rotated using the ``delta`` the particle had **before**
  the loss, not after. That is deliberate and consistent rather than an oversight: the
  radiation kick itself evaluates the loss at that same ``delta`` (it is one lumped kick
  standing in for a continuous drag along the element), so both routes describe the same
  traversal and neither sees the other's end state. The cost is one factor of
  ``U/E ~ 3e-7`` per magnet in the precession rate — the same over-counting B2 already
  records for the loss, converging as ``(N-1)/N`` under slicing. It is stated here
  because **N3 is the milestone that consumes exactly this combination**: a radiating,
  precessing particle is what Sokolov-Ternov polarization is about, and an undocumented
  ordering is the kind of thing M1 was caught by.

**The closed solution (N2), which is the second half of this module.** Everything above
is a *map*. Given a ring, the questions that follow are the two the closed orbit and the
tune answer for position and momentum: which spin direction comes back to itself after a
turn (``n_0``, :func:`closed_spin_solution`), and how fast a spin that is *not* along it
winds around it (``nu_0``, the spin tune). They are reached the same way I1 reached the
closed orbit -- as the fixed point of the one-turn map -- with one simplification and one
trap.

The simplification: the one-turn spin map is a **rotation**, and a rotation is linear in
the spin, so :func:`spin_one_turn_matrix` gets the whole 3x3 exactly by carrying the three
Cartesian basis vectors around once. There is no Newton iteration and no differencing
step; ``n_0`` is then the eigenvector of eigenvalue ``1`` and ``nu_0`` the rotation angle
about it. The trap is that on a flat, unsteered ring the answer is ``n_0 = y`` **bit for
bit**, whatever the coefficients of the map are -- so the closed solution is as blind a
control as the spin tune was in N1 until the ring is given a *vertical* closed orbit. What
breaks the degeneracy, what the tilt then measures, and why it is an **integer** spin tune
(not ``k +- Q_y``) that it resonates at, are set out in :func:`closed_spin_solution`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .coords import DELTA, DIM, PX, PY, ZETA, X, Y
from .reference import ReferenceParticle

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from .elements.element import Element
    from .lattice import Lattice

__all__ = [
    "SPIN_DIM",
    "ClosedSpinSolution",
    "SpinSolutionError",
    "anomalous_moment",
    "along_direction_of_motion",
    "closed_spin_solution",
    "direction_of_motion",
    "precession_vector",
    "propagate_spin_solution",
    "rotate",
    "rotate_about_s",
    "spin_axis_and_tune",
    "spin_one_turn_matrix",
    "spin_precession",
    "spin_tune",
]

#: A spin is a unit 3-vector ``(S_x, S_y, S_z)`` in the local curvilinear frame — the
#: same axes the 6D state's ``(x, y, s)`` use. It is carried *alongside* the 6D state
#: rather than appended to it, because it neither influences it nor is part of it.
SPIN_DIM: int = 3


def anomalous_moment(ref: ReferenceParticle) -> float:
    """``G = (g - 2)/2`` for ``ref``, raising if it was never set.

    The one place the ``None`` default of
    :attr:`~accsim.reference.ReferenceParticle.anomalous_moment` is turned into an
    error. It exists because the alternative — a numeric default of ``0`` — is what
    ``xt.Particles`` does, and a zero anomalous moment is not "no spin physics": it is
    the *cyclotron* rotation, a spin tune of exactly zero, and a plausible-looking
    tracked spin that is answering a different question. ``0.0`` set **explicitly** is
    accepted, and is the Dirac-particle limit the sharpest analytic gate runs in.
    """
    g = ref.anomalous_moment
    if g is None:
        raise ValueError(
            "this ReferenceParticle has no anomalous_moment set, and spin precession "
            "cannot be computed without one. Pass anomalous_moment=... to the "
            "constructor (accsim.reference.ELECTRON_ANOMALOUS_MOMENT / "
            "PROTON_ANOMALOUS_MOMENT), or 0.0 for the Dirac limit. It is deliberately "
            "not defaulted: a silent 0 gives the cyclotron rotation and a zero spin "
            "tune without erroring."
        )
    return float(g)


def direction_of_motion(
    px: np.ndarray | float, py: np.ndarray | float, delta: np.ndarray | float
) -> np.ndarray:
    r"""The unit vector along the particle's momentum, ``(3,)`` or ``(3, n)``.

    ``i = (px, py, p_s)/(1 + delta)`` with ``p_s = sqrt((1+delta)^2 - px^2 - py^2)``, so
    ``|i| = 1`` identically. That identity is the whole content of the function, and it
    is worth stating because the reference gets it wrong: xtrack 0.106.4's own
    ``direction_of_motion`` (``track_magnet_radiation.h:22``) computes
    ``sqrt(1 - ix*ix + iy*iy)`` — a ``+`` where a ``-`` belongs — so the vector it
    returns is longer than one by ``py^2/(1+delta)^2`` and is used, unnormalised, for
    both the spin precession and the perpendicular field the radiation kick needs. The
    reference suite asserts that disagreement *and its order in* ``py`` rather than
    dodging it by tracking with ``py = 0``.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    one_plus = 1.0 + np.asarray(delta, dtype=float)
    ix = px / one_plus
    iy = py / one_plus
    iz = np.sqrt(np.maximum(1.0 - ix * ix - iy * iy, 0.0))
    return np.stack(np.broadcast_arrays(ix, iy, iz))


def precession_vector(
    bx: np.ndarray | float,
    by: np.ndarray | float,
    px: np.ndarray | float,
    py: np.ndarray | float,
    delta: np.ndarray | float,
    ref: ReferenceParticle,
) -> np.ndarray:
    r"""``Omega`` [rad/m] — the Thomas-BMT precession rate per unit path length.

    ``(bx, by)`` is the element's field at the particle, normalised to ``(B rho)_0``
    (:meth:`~accsim.elements.element.Element.normalized_field`); ``(px, py, delta)`` is
    the particle's momentum. Returns ``(3,)`` or ``(3, n)`` in the element's own frame.

    Every element in this package has a purely **transverse** field, so ``b_s = 0``.
    That does *not* make the ``(1 + G)`` parallel term dead code: ``b_par`` is the
    component of ``b`` **along the direction of motion**, which is non-zero as soon as
    the particle has a transverse angle. The term therefore enters at ``O(px b_x)``, and
    the analytic suite gates it at that order rather than waiting for a solenoid.
    """
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    delta = np.asarray(delta, dtype=float)

    i_hat = direction_of_motion(px, py, delta)
    b = np.stack(np.broadcast_arrays(bx, by, np.zeros_like(bx + by)))
    # b_par is the projection onto the direction of motion; b_perp is what is left. The
    # longitudinal field is zero, so the dot product has only two terms -- but i_hat's
    # third component still matters, through b_par's own s-component below.
    b_dot_i = np.einsum("i...,i...->...", b, i_hat)
    b_par = b_dot_i * i_hat
    b_perp = b - b_par

    g = anomalous_moment(ref)
    gamma = _particle_gamma(delta, ref)
    return -((1.0 + g * gamma) * b_perp + (1.0 + g) * b_par) / (1.0 + delta)


def rotate(spin: np.ndarray, omega: np.ndarray, path_length: np.ndarray | float) -> np.ndarray:
    r"""Rotate ``spin`` about ``omega`` by ``|omega| * path_length``, right-handed.

    Rodrigues' formula, which is the exact flow of ``dS/ds = Omega x S`` for a constant
    ``Omega``. Both arrays are ``(3,)`` or ``(3, n)``.

    ``|omega| = 0`` is handled without a branch or a threshold: the axis is replaced by
    an arbitrary unit vector and the angle is exactly zero, so ``cos = 1``, ``sin = 0``
    and the formula returns ``spin`` **bit for bit**. xtrack skips the rotation below
    ``|Omega| = 1e-10`` instead, which is a discontinuity this does not reproduce.
    """
    spin = np.asarray(spin, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mod = np.sqrt(np.einsum("i...,i...->...", omega, omega))
    phi = mod * np.asarray(path_length, dtype=float)
    axis = omega / np.where(mod == 0.0, 1.0, mod)  # arbitrary unit axis where |Omega| = 0

    cos, sin = np.cos(phi), np.sin(phi)
    cross = np.cross(axis, spin, axis=0)
    dot = np.einsum("i...,i...->...", axis, spin)
    return spin * cos + cross * sin + axis * (dot * (1.0 - cos))


def rotate_about_s(spin: np.ndarray, phi: float) -> np.ndarray:
    """Carry a spin through a **roll**: the passive rotation of the frame about ``s``.

    The 3-vector twin of :func:`accsim.elements.alignment.s_rotation`, and the same
    passive sense: ``S_x -> +cos S_x + sin S_y``, ``S_y -> -sin S_x + cos S_y``, with
    ``S_s`` untouched. Exactly the identity at ``phi = 0`` -- an aligned element must
    not pay for the machinery, nor lose a bit to it.
    """
    if phi == 0.0:
        return spin
    c, s = math.cos(phi), math.sin(phi)
    out = np.empty_like(spin)
    out[0] = spin[0] * c + spin[1] * s
    out[1] = -spin[0] * s + spin[1] * c
    out[2] = spin[2]
    return out


def along_direction_of_motion(state: np.ndarray) -> np.ndarray:
    """The spin of a particle whose spin points along its own momentum, ``(3,)``/``(3, n)``.

    A convenience with a purpose: it is the initial condition of the ``G = 0`` identity
    (see the module docstring), which is the sharpest gate on this axis and the only one
    that catches a sign error, a mis-scaled coefficient and a missing bend frame
    rotation with a single number.
    """
    state = np.asarray(state, dtype=float)
    return direction_of_motion(state[PX], state[PY], state[DELTA])


def _particle_gamma(delta: np.ndarray | float, ref: ReferenceParticle) -> np.ndarray | float:
    """The **particle's** Lorentz factor at momentum deviation ``delta``.

    ``gamma = E/(m c^2)`` with ``E = sqrt(p^2 + m^2)`` and ``p = p_0 (1 + delta)`` — not
    the reference ``gamma0``, because the whole precession scales with it and a
    chromatic spin effect that used ``gamma0`` would be silently absent.
    """
    p = ref.momentum_eV * (1.0 + np.asarray(delta, dtype=float))
    return np.sqrt(p * p + ref.mass_eV * ref.mass_eV) / ref.mass_eV


def _rotate_about_y(spin: np.ndarray, angle: float) -> np.ndarray:
    """The curvilinear frame's own turn inside a bend: ``x -> x c + s_z s``, ``y`` fixed.

    Right-handed about ``y``, and applied in two halves either side of the BMT rotation
    (which is the midpoint rule, and is what xtrack does). Its *sign* is not a
    convention to be chosen: with ``G = 0`` the spin must come out of a sector bend
    still pointing along the design orbit, and only ``+theta`` does that.
    """
    if angle == 0.0:
        return spin
    c, s = math.cos(angle), math.sin(angle)
    out = np.empty_like(spin)
    out[0] = spin[0] * c + spin[2] * s
    out[1] = spin[1]
    out[2] = -spin[0] * s + spin[2] * c
    return out


def spin_precession(
    element: Element,
    spin: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    ref: ReferenceParticle,
) -> np.ndarray:
    r"""Rotate ``spin`` through ``element``, given the states it has just mapped.

    ``before`` / ``after`` are the 6D states entering and leaving the element **in its
    own body frame** — the same pair :func:`accsim.radiation_kick.radiation_kick`
    receives, and for the same reason: a misaligned magnet must precess a spin according
    to where it really is. ``spin`` is ``(3,)`` or ``(3, n)``, matching the state's
    shape. Returns a new array; the input is not modified.

    Three things happen, in order: half the bend's frame rotation, the BMT rotation
    about the field at the **mid-point** of the traversal, and the other half of the
    frame rotation. A straight element has no frame rotation; a field-free one has no
    BMT rotation; a thin one has neither (see the module docstring's *Scope*).
    """
    spin = np.array(spin, dtype=float, copy=True)
    if spin.shape[0] != SPIN_DIM:
        raise ValueError(f"spin must be a ({SPIN_DIM},) vector or ({SPIN_DIM}, n) array")

    half_frame = 0.5 * element.frame_rotation_angle
    if half_frame != 0.0 and element.roll != 0.0:
        raise NotImplementedError(
            f"spin precession through a rolled *bending* magnet ({element.name!r}) is not "
            "implemented: a roll moves a bend's exit face (K2's rigid-body geometry), so "
            "the frame rotation is no longer a rotation about y in the lattice frame. A "
            "rolled straight element is fine -- there the roll is a plain conjugation."
        )

    spin = _rotate_about_y(spin, half_frame)

    length = element.length
    if length > 0.0:
        mid_x = 0.5 * (before[X] + after[X])
        mid_y = 0.5 * (before[Y] + after[Y])
        bx, by = element.normalized_field(mid_x, mid_y)
        if not (np.all(bx == 0.0) and np.all(by == 0.0)):
            mid_px = 0.5 * (before[PX] + after[PX])
            mid_py = 0.5 * (before[PY] + after[PY])
            delta = after[DELTA]
            omega = precession_vector(bx, by, mid_px, mid_py, delta, ref)
            # The same path length the radiation kick integrates over: a longer
            # trajectory precesses further, and the two routes must not disagree about
            # how long the trajectory is.
            p = ref.momentum_eV * (1.0 + delta)
            energy = np.sqrt(p * p + ref.mass_eV * ref.mass_eV)
            rvv = (p / energy) / ref.beta0
            l_path = rvv * (length - (after[ZETA] - before[ZETA]))
            spin = rotate(spin, omega, l_path)

    return _rotate_about_y(spin, half_frame)


# ---------------------------------------------------------------------------
# N2: the closed spin solution and the spin tune
# ---------------------------------------------------------------------------


class SpinSolutionError(ValueError):
    """Raised when a lattice has no unique periodic spin direction.

    The spin twin of :class:`accsim.orbit.ClosedOrbitError`, and the same
    degeneracy: there the one-turn map has an orbital eigenvalue ``1`` and no orbit
    closes; here the one-turn **rotation** is the identity, every direction closes,
    and ``n_0`` is not a property of the ring.
    """


@dataclass(frozen=True)
class ClosedSpinSolution:
    """The periodic spin direction of a ring, and the rate it precesses about it.

    :attr:`n0` is the unit vector a spin must lie along at the lattice entrance to
    come back to itself after one turn; :attr:`spin_tune` is how many times per turn
    a spin *not* along it winds around it. :attr:`one_turn_matrix` is the rotation
    both are read from, and :attr:`orbit` is the closed orbit they were built on --
    kept because on a flat, unsteered ring the answer is ``y`` no matter what the
    map does (see :func:`closed_spin_solution`), so *which orbit* is half the claim.
    """

    n0: np.ndarray
    spin_tune: float
    one_turn_matrix: np.ndarray
    orbit: np.ndarray


#: Below this, the one-turn rotation is the identity to working precision and
#: ``n_0`` is undefined. The quantity compared against it is the second-smallest
#: singular value of ``R - I``, which is ``2 |sin(pi nu_0)|`` -- so the test is
#: literally "is the spin tune an integer?", the same question
#: :func:`accsim.orbit.closed_orbit` asks of the orbital tune.
_SPIN_DEGENERACY_LIMIT = 1e-12


def spin_axis_and_tune(one_turn: np.ndarray) -> tuple[np.ndarray, float]:
    r"""Read ``(n_0, nu_0)`` off a 3x3 one-turn spin rotation.

    ``n_0`` is the eigenvector of ``one_turn`` with eigenvalue ``1`` -- the fixed
    point of the rotation, exactly as the closed orbit is the fixed point of the
    affine map -- taken as the null space of ``R - I`` rather than by an eigenvalue
    solve, because that stays accurate at a half-integer spin tune, where ``R``'s
    antisymmetric part vanishes and the axis cannot be read off it at all.

    **Two sign conventions, both forced rather than chosen.**

    - ``n_0`` is oriented so that ``n_0 . y > 0`` -- the vertical component, which on
      any flat ring is the whole of it. That is xtrack's convention too (its fixed
      point search sets ``s_y = +sqrt(1 - s_x^2 - s_z^2)``, so it can only ever
      return an upward solution), which is what makes the two comparable at all.
      Where ``n_0`` is exactly horizontal the rule falls back to ``n_0 . x > 0`` and
      then to ``n_0 . z > 0``.
    - ``nu_0`` is returned as a **fraction in** ``[0, 1)``, defined by
      ``R = R(n_0, -2 pi nu_0)``: the spin turns by ``2 pi nu_0`` about ``n_0`` in
      the *negative* sense. That minus sign is not decoration. A flat ring's net spin
      rotation is ``-(1 + G gamma) theta`` from Thomas-BMT plus ``+theta`` from the
      frame, i.e. ``-G gamma theta`` about ``+y``; writing it as ``+2 pi nu_0`` would
      make ``nu_0 = -G gamma``, and every textbook -- and the whole of N1 -- quotes
      ``nu_0 = +G gamma``. The convention is picked to agree with that.

    Only the fraction is knowable: a rotation matrix has no memory of how many whole
    turns produced it, exactly as a one-turn transfer matrix has none of the integer
    part of the betatron tune.

    Raises :class:`SpinSolutionError` when ``R`` is the identity to working precision
    -- an **integer** spin tune, where every direction is periodic and none is *the*
    periodic one.
    """
    R = np.asarray(one_turn, dtype=float)
    if R.shape != (SPIN_DIM, SPIN_DIM):
        raise ValueError(f"one_turn must be a ({SPIN_DIM}, {SPIN_DIM}) rotation, got {R.shape}")

    _, singular, right = np.linalg.svd(R - np.eye(SPIN_DIM))
    if singular[-2] < _SPIN_DEGENERACY_LIMIT:
        raise SpinSolutionError(
            f"no unique closed spin solution: the one-turn spin rotation is the identity "
            f"to working precision (2|sin(pi nu_0)| = {singular[-2]:.3g}), i.e. the spin "
            "tune is an **integer**. Every direction is then periodic and none is the "
            "periodic one -- the spin twin of an integer betatron tune, where no closed "
            "orbit exists because a kick repeats in phase every turn"
        )
    n0 = right[-1]

    # Orient: up if it has a vertical component at all, then +x, then +z.
    for component in (1, 0, 2):
        if abs(n0[component]) > _SPIN_DEGENERACY_LIMIT:
            if n0[component] < 0.0:
                n0 = -n0
            break

    # The antisymmetric part is ``n sin(theta)``, which fixes the *sign* of the turn
    # about the n_0 just oriented; the trace fixes its size.
    axis_sin = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    theta = math.acos(min(1.0, max(-1.0, 0.5 * (float(np.trace(R)) - 1.0))))
    signed = theta if float(axis_sin @ n0) >= 0.0 else -theta
    return n0, (-signed / (2.0 * math.pi)) % 1.0


def _closed_state(lattice: Lattice, orbit0: np.ndarray | None) -> np.ndarray:
    """The 6D state a spin is carried around the ring on.

    ``orbit0`` defaults to :func:`accsim.orbit.closed_orbit_nonlinear` -- the fixed
    point of :meth:`~accsim.elements.element.Element.track`, **not** of
    :meth:`~accsim.elements.element.Element.matrix`. That is the whole reason the
    default is the expensive one: the spin is rotated by what ``track`` does, so a
    spin carried around the *linear* closed orbit is carried around a trajectory that
    does not quite close, and its "one-turn" rotation is really a rotation between two
    different points. With the exact maps of axis L the gap is ``O(x_co^3)``; on N2's
    own gate ring it shows up as a one-turn orbit residual of ``1e-8`` where the
    tracked orbit gives ``1e-18``.
    """
    from .orbit import closed_orbit_nonlinear

    orbit = closed_orbit_nonlinear(lattice) if orbit0 is None else np.asarray(orbit0, dtype=float)
    if orbit.shape != (4,):
        raise ValueError(f"orbit0 must be a length-4 (x, px, y, py) vector, got {orbit.shape}")
    state = np.zeros(DIM)
    state[[X, PX, Y, PY]] = orbit
    return state


def spin_one_turn_matrix(lattice: Lattice, orbit0: np.ndarray | None = None) -> np.ndarray:
    r"""The 3x3 rotation one turn applies to a spin riding the closed orbit.

    Obtained by carrying the three Cartesian basis vectors around as a ``(3, 3)`` spin
    on a ``(6, 3)`` bunch of identical closed-orbit particles: the columns of the
    result are the images of ``x``, ``y``, ``z``, which *is* the matrix. It is
    **exact**, with no differencing step, because the spin map is linear in the spin
    -- every element rotates it about an axis fixed by the *orbit*, never by the spin
    itself. xtrack builds the same matrix by finite differences with ``ds = 1e-5``
    (``twiss.py``, ``_get_spin_polarization``), which is why the reference comparison
    lands near ``1e-10`` rather than at round-off, and which way round the two codes'
    accuracies go.

    That the result comes out orthogonal is *not* a check on the physics: a product of
    rotations is orthogonal whatever fields it was built from.
    """
    from .tracking import Tracker

    state = np.tile(_closed_state(lattice, orbit0)[:, None], (1, SPIN_DIM))
    _, matrix = Tracker(lattice).track_once_with_spin(state, np.eye(SPIN_DIM))
    return matrix


def closed_spin_solution(lattice: Lattice, orbit0: np.ndarray | None = None) -> ClosedSpinSolution:
    r"""``n_0`` and ``nu_0`` for a ring: the spin analogue of the closed orbit and the tune.

    **The degeneracy is the first thing to know about this function.** On a flat,
    unsteered ring every field a spin meets is vertical, every rotation is about ``y``,
    and ``n_0 = y`` exactly -- not to a tolerance but bit for bit, for any lattice, any
    energy, and any implementation of the map that gets the *plane* right. Nothing
    about the transverse coefficient, the quadrupoles, or even the ``(1 + G gamma)``
    factor can be read off such a ring. ``n_0`` becomes informative only once the
    closed orbit has a **vertical** excursion through a field, which is why the
    analytic gates for this milestone are built on a steered ring and assert the
    *order in the steering*. M3 found the same shape one axis earlier; I1's correctors
    are what break it here.

    **What tilts it, and by how much.** Take a ring whose entire spin perturbation is a
    single localized rotation by ``chi`` about ``x`` -- one thick quadrupole inside a
    closed vertical bump, with every bend on the design orbit. Then, to first order in
    ``chi``, observed at the entrance with the kick at the top of the turn,

        ``n_0 = ( -(chi/2) cot(pi nu_0),  1,  -chi/2 )``.

    The two transverse components say different things, and only together do they pin
    the map. The ``z`` component is ``-chi/2`` with **no** resonance denominator, so it
    measures the kick itself -- and hence the ``(1 + G gamma) k1 int y ds`` behind it.
    The ``x`` component carries ``cot(pi nu_0)``, which diverges at every **integer**
    spin tune: that is the first-order (imperfection) spin resonance, and since
    ``nu_0 = G gamma`` on a flat ring it is crossed by *scanning the beam energy*,
    which is why a polarized ring measures its own energy to a part in ``10^4``.

    Their ratio ``n_0 . x / n_0 . z = cot(pi nu_0)`` drops ``chi`` altogether, so the
    **direction** the solution tilts in measures the spin tune on its own, with nothing
    about the strength of the perturbation in it.

    ``nu_0`` itself is unmoved at first order in ``chi`` -- it shifts by
    ``chi^2 cot(pi nu_0) / (8 pi)`` -- which is this axis's version of N1's "the spin
    tune is a control": the number everybody quotes is the one that cannot see the
    perturbation. Use the tilt, not the tune.

    The vertical resonance ``nu_0 = k +- Q_y`` is a different statement about a
    different object and is **not** here: it is a property of the invariant spin field
    of a particle with vertical betatron *amplitude*, not of ``n_0``, which lives on
    the closed orbit and therefore sees only one-turn-periodic perturbations -- integer
    harmonics, integer resonances. See ``docs/ROADMAP.md`` under N3.
    """
    orbit = _closed_state(lattice, orbit0)[[X, PX, Y, PY]]
    matrix = spin_one_turn_matrix(lattice, orbit)
    n0, nu0 = spin_axis_and_tune(matrix)
    return ClosedSpinSolution(n0=n0, spin_tune=nu0, one_turn_matrix=matrix, orbit=orbit)


def spin_tune(lattice: Lattice, orbit0: np.ndarray | None = None) -> float:
    """The fractional spin tune ``nu_0`` of ``lattice`` -- see :func:`closed_spin_solution`."""
    return closed_spin_solution(lattice, orbit0).spin_tune


def propagate_spin_solution(
    lattice: Lattice, n0: np.ndarray | None = None, orbit0: np.ndarray | None = None
) -> list[np.ndarray]:
    """``n_0`` at every element boundary -- ``len(lattice) + 1`` unit vectors.

    The counterpart of :func:`accsim.orbit.propagate_orbit`, and read the same way:
    the entrance, then the exit of each element in order. With ``n0`` defaulted to
    :func:`closed_spin_solution`'s the last vector equals the first -- that is what
    "closed" means for a spin as much as for an orbit -- and a lattice whose elements
    are all thin returns the same vector ``len(lattice) + 1`` times, because thin
    elements do not precess (see the module docstring).

    Pass an explicit ``n0`` to follow an arbitrary spin down the ring instead; it is
    normalised on the way in, since only its direction means anything.
    """
    state = _closed_state(lattice, orbit0)
    if n0 is None:
        spin = closed_spin_solution(lattice, state[[X, PX, Y, PY]]).n0
    else:
        spin = np.asarray(n0, dtype=float)
        if spin.shape != (SPIN_DIM,):
            raise ValueError(f"n0 must be a ({SPIN_DIM},) vector, got {spin.shape}")
        spin = spin / np.linalg.norm(spin)

    points = [spin]
    for elem in lattice.elements:
        state, spin = elem.track_with_spin(state, spin, lattice.ref)
        points.append(spin)
    return points
