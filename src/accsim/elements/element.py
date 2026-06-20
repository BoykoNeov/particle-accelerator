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

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"{type(self).__name__}(length={self.length}{name})"
