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
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .coords import DELTA, PX, PY, ZETA, X, Y
from .reference import ReferenceParticle

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from .elements.element import Element

__all__ = [
    "SPIN_DIM",
    "anomalous_moment",
    "along_direction_of_motion",
    "direction_of_motion",
    "precession_vector",
    "rotate",
    "rotate_about_s",
    "spin_precession",
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
