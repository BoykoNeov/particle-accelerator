r"""Relativistic kinematics for the toy event generator (Phase 2 learning module).

**Natural units.** Unlike the beam-dynamics core (SI metres, eV; see
``docs/CONVENTIONS.md`` -> *Units*), the event-physics module works in **natural
units** ``hbar = c = 1`` with energies/momenta in **GeV**. This is the universal
convention of particle-physics cross-section calculations, and keeping it local to
``accsim.events`` avoids threading ``c`` factors through every Mandelstam. The
one boundary crossing back to laboratory units is the cross-section, converted from
``GeV^-2`` to barns via ``(hbar c)^2`` in :mod:`accsim.events.generator`.

**Metric.** Mostly-minus ``(+, -, -, -)``, so a four-vector ``p = (E, px, py, pz)``
has invariant ``p.p = E^2 - |p_vec|^2 = m^2``. Four-vectors are plain
``numpy`` arrays of shape ``(4,)`` (single) or ``(..., 4)`` (batched, energy in the
last axis's index 0).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "METRIC",
    "minkowski_dot",
    "invariant_mass_squared",
    "mandelstam_s",
    "mandelstam_t",
    "mandelstam_u",
]

# Mostly-minus metric diag(+1, -1, -1, -1). Contracting with it turns the naive
# Euclidean dot into the Minkowski one.
METRIC: npt.NDArray[np.float64] = np.array([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(
    p: npt.NDArray[np.float64], q: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""Minkowski inner product ``p.q = E_p E_q - vec p . vec q``.

    Accepts single four-vectors (shape ``(4,)``) or batches (shape ``(..., 4)``);
    contraction is over the last axis, so batched inputs return the array of dots.
    """
    return np.sum(p * METRIC * q, axis=-1)


def invariant_mass_squared(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | float:
    """``p.p = m^2`` — the squared invariant mass of a (possibly summed) four-vector."""
    return minkowski_dot(p, p)


def mandelstam_s(
    p1: npt.NDArray[np.float64], p2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``s = (p1 + p2)^2`` — the squared CM energy of the two incoming particles."""
    return invariant_mass_squared(p1 + p2)


def mandelstam_t(
    p1: npt.NDArray[np.float64], p3: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``t = (p1 - p3)^2`` — the momentum transfer from incoming ``p1`` to outgoing ``p3``."""
    return invariant_mass_squared(p1 - p3)


def mandelstam_u(
    p1: npt.NDArray[np.float64], p4: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | float:
    r"""``u = (p1 - p4)^2`` — the crossed momentum transfer (incoming ``p1`` to outgoing ``p4``)."""
    return invariant_mass_squared(p1 - p4)
