"""Base class for lattice elements."""

from __future__ import annotations

import abc

import numpy as np

from ..reference import ReferenceParticle


class Element(abc.ABC):
    """A lattice element.

    Each element exposes its action on the 6D phase-space state as a 6x6 linear
    transfer matrix via :meth:`matrix`. Later stages may add a (possibly
    nonlinear) exact map alongside the linear matrix; the linear matrix is the
    minimum every element must provide, since Twiss propagation is built on it.
    """

    def __init__(self, length: float, name: str | None = None) -> None:
        if length < 0:
            raise ValueError(f"element length must be >= 0, got {length}")
        self.length = float(length)
        self.name = name

    @abc.abstractmethod
    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """Return the 6x6 linear transfer matrix for reference particle ``ref``.

        The map acts as ``state_out = matrix @ state_in`` on the column vector
        ``(x, px, y, py, zeta, delta)``.
        """
        raise NotImplementedError

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """Map a single 6D ``state`` through the element (returns a new array).

        The default is the linear map ``matrix(ref) @ state`` — exact for every
        linear element, so element-by-element tracking of a purely linear lattice
        equals a single ``one_turn_matrix`` product. Nonlinear elements (e.g.
        :class:`~accsim.elements.rfcavity.RFCavity`, whose ``sin`` kick gives the RF
        bucket its separatrix) override this. This is the seam the long-term
        (Stage 3+) tracker plugs into.
        """
        return self.matrix(ref) @ state

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"{type(self).__name__}(length={self.length}{name})"
