r"""The classical ("mean") synchrotron-radiation kick a tracked particle takes (B2).

Axis B's :mod:`accsim.radiation` is entirely a **design-route** module: the radiation
integrals ride the Twiss functions, and the damping times and equilibrium emittance are
closed forms evaluated on them. Nothing there touches a tracked particle. This module is
the other half — the map that makes a tracked particle actually lose the energy it
radiates, so that damping is something the simulation *exhibits* rather than asserts.

**The physics, in one paragraph.** In an element the particle follows a curved path of
local radius ``rho_p`` and radiates

    ``U = (C_gamma / 2 pi) * E^4 * kappa^2 * l_path``      [eV],  ``kappa = 1/rho_p``,

the same ``C_gamma E^4 I2 / 2 pi`` the radiation integrals integrate, evaluated on the
particle's own trajectory instead of the design orbit. The photons leave along the
direction of motion, so the momentum *vector* shrinks in magnitude with its direction
unchanged: every Cartesian component — ``px``, ``py`` **and** ``1 + delta`` — is scaled
by one common factor. That single fact is what produces transverse damping, and it is
the part a plausible implementation drops (see *The wrong map* below).

**The factor.** The particle stays on shell, so losing energy ``U`` fixes the momentum:

    ``f = P_new/P = sqrt((E-U)^2 - m^2) / sqrt(E^2 - m^2)
        = sqrt(1 - U(2E - U)/(E^2 - m^2))``

written in the second (rationalised) form because the first cancels two numbers of size
``E`` — the numerical trap L1 recorded for the drift and L3 for the bend. To first order
``f = 1 - U/(beta^2 E)``, and exactly ``1 - U/E`` in the massless limit.

**The wrong map.** Reducing ``delta`` alone and leaving ``px, py`` is the natural-looking
mistake. It gets the *longitudinal* damping exactly right, so half the gates cannot see
it; inside the element it **anti**-damps the angle ``x' = px/pz`` at first order; and per
turn it produces **exactly zero** transverse damping, because ``py`` is never touched and
the RF restores ``delta``. It is available as ``model="mean_delta_only"`` for the gate
that asserts precisely this, and is not a physical model.

**Scope and costs.**

- *Not symplectic.* Radiation is dissipative — this is the first map in the package that
  must **fail** :func:`accsim.symplectic.is_symplectic_map`, and the analytic suite
  asserts that rejection rather than working around it.
- *A tracking mode, never the design route.* ``matrix()`` and ``kick()`` are untouched,
  so every optics quantity in the package is bit-for-bit unchanged and the invariant
  that bounds axis L — ``matrix()`` is the exact origin Jacobian of ``track()`` — still
  holds for the map as such. With radiation on it does not, because the reference
  particle radiates too; that is why radiation is opt-in per tracking call.
- *One kick per element.* The loss is evaluated at the element's **entry** energy, so it
  over-counts by ``U_elem/E`` relative to the continuous answer; slicing the lattice
  converges it as ``(N-1)/N``, which the analytic suite asserts as that law rather than
  as a tolerance. xtrack does the same thing with its own sub-stepping — its default
  ``integrator='adaptive'`` resolves to eight uniform steps for a plain bend, and
  ``integrator='uniform', num_multipole_kicks=1`` reproduces the single lumped kick.
- *Thin elements do not radiate.* A zero-length element has no path to radiate over, so
  correctors, thin quadrupoles, thin multipoles and the RF cavity contribute nothing.
  That is a scope statement, not an approximation: a real short magnet radiates, and
  modelling it means giving it a length.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .coords import DELTA, PX, PY, ZETA, X, Y
from .reference import ReferenceParticle

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from .elements.element import Element

__all__ = ["RADIATION_MODELS", "mean_radiation_kick", "radiation_constant_cgamma"]

#: The radiation models ``track`` accepts. ``"mean_delta_only"`` is the deliberately
#: wrong map the discriminating gate needs; it is not a physical choice.
RADIATION_MODELS: tuple[str, ...] = ("off", "mean", "mean_delta_only")


def radiation_constant_cgamma(ref: ReferenceParticle) -> float:
    r"""``C_gamma = 4 pi r0 / (3 (m c^2)^3)`` [m/eV^3] for the reference species.

    Computed from the particle's own classical radius and rest energy, so it is correct
    for any species (``r0 ∝ 1/m`` ⇒ ``C_gamma ∝ 1/m^3``). For the electron this is the
    familiar ``8.846e-5 m/GeV^3``. :func:`accsim.radiation.radiation_constant_cgamma`
    is this function — the design route and the tracked route must not carry two copies
    of the constant that sets the size of the whole effect.
    """
    return 4.0 * math.pi * ref.classical_radius_m / (3.0 * ref.mass_eV**3)


def _perpendicular_field(
    bx: np.ndarray | float,
    by: np.ndarray | float,
    px: np.ndarray | float,
    py: np.ndarray | float,
    delta: np.ndarray | float,
) -> np.ndarray:
    r"""``|B_perp| / (B rho)_0`` — the part of the field the particle actually feels.

    Only the field component **perpendicular** to the velocity bends the trajectory and
    so radiates. With the direction of motion ``i = (px, py, pz)/(1+delta)`` and a purely
    transverse field ``(bx, by, 0)``, this is ``|b - (b.i) i|``. On the design orbit it
    is just ``|b|``; the projection matters at the ``(x')^2`` level.
    """
    ix = np.asarray(px) / (1.0 + np.asarray(delta))
    iy = np.asarray(py) / (1.0 + np.asarray(delta))
    iz = np.sqrt(np.maximum(1.0 - ix * ix - iy * iy, 0.0))
    b_par = bx * ix + by * iy  # the longitudinal field component is zero
    ex = bx - b_par * ix
    ey = by - b_par * iy
    ez = -b_par * iz
    return np.sqrt(ex * ex + ey * ey + ez * ez)


def mean_radiation_kick(
    element: Element,
    before: np.ndarray,
    after: np.ndarray,
    ref: ReferenceParticle,
    model: str = "mean",
) -> np.ndarray:
    r"""Apply the classical radiation loss of ``element`` to a state it has just mapped.

    ``before`` / ``after`` are the states entering and leaving the element **in its own
    body frame** (so a misaligned magnet radiates according to where it really is), each
    a ``(6,)`` vector or a ``(6, n)`` bunch. Returns a new array; the input is not
    modified. ``model="off"`` returns ``after`` unchanged.

    The field is sampled at the **mid-point** of the traversal — the mean of the entry
    and exit positions, and the mean of the entry and exit angles for the perpendicular
    projection — which is the one-step midpoint rule for
    ``U = (C_gamma/2 pi) E^4 ∮ kappa^2 ds`` and the convention xtrack uses, so the two
    are directly comparable per element. The path length is the element's own
    ``l_path = rvv (L - Delta zeta)``, so a longer trajectory radiates more without any
    of that being put in by hand.
    """
    if model == "off":
        return after
    if model not in RADIATION_MODELS:
        raise ValueError(f"radiation model must be one of {RADIATION_MODELS}, got {model!r}")
    length = element.length
    if length == 0.0:
        return after  # a thin element has no path to radiate over

    out = np.array(after, dtype=float, copy=True)
    mid_x = 0.5 * (before[X] + after[X])
    mid_y = 0.5 * (before[Y] + after[Y])
    mid_px = 0.5 * (before[PX] + after[PX])
    mid_py = 0.5 * (before[PY] + after[PY])
    bx, by = element.normalized_field(mid_x, mid_y)
    if np.all(bx == 0.0) and np.all(by == 0.0):
        return out  # no field, no radiation (a drift, or a bend switched off)

    delta = after[DELTA]
    kappa = _perpendicular_field(bx, by, mid_px, mid_py, delta) / (1.0 + delta)

    m = ref.mass_eV
    p = ref.momentum_eV * (1.0 + delta)
    energy = np.sqrt(p * p + m * m)
    rvv = (p / energy) / ref.beta0  # beta/beta0
    l_path = rvv * (length - (after[ZETA] - before[ZETA]))

    u = radiation_constant_cgamma(ref) / (2.0 * math.pi) * energy**4 * kappa * kappa * l_path
    # On shell: f = P_new/P with E_new = E - U, rationalised so no two numbers of size E
    # are subtracted (the trap L1 recorded for the drift, L3 for the bend).
    f = np.sqrt(np.maximum(1.0 - u * (2.0 * energy - u) / (energy * energy - m * m), 0.0))

    out[DELTA] = f * (1.0 + delta) - 1.0
    if model == "mean":
        # The photons leave along the direction of motion, so the transverse momenta
        # scale by the SAME factor -- this line, and only this line, is what damps the
        # betatron amplitude. See the module docstring's *The wrong map*.
        out[PX] = after[PX] * f
        out[PY] = after[PY] * f
    return out
