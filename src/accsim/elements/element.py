"""Base class for lattice elements."""

from __future__ import annotations

import abc

import numpy as np

from ..coords import DIM
from ..reference import ReferenceParticle


class Element(abc.ABC):
    """A lattice element.

    Each element exposes its action on the 6D phase-space state as a 6x6 linear
    transfer matrix via :meth:`matrix`. Later stages may add a (possibly
    nonlinear) exact map alongside the linear matrix; the linear matrix is the
    minimum every element must provide, since Twiss propagation is built on it.

    The general linear action is **affine**, ``state_out = matrix @ state_in +
    kick``: a dipole corrector deflects every particle by the same angle whatever
    its coordinates, which no 6x6 acting on ``(x, px, ...)`` can express. The
    constant part lives in :meth:`kick` and is **zero for every element except a
    corrector**, so the homogeneous ``matrix`` remains the whole story for optics
    (beta, tune, chromaticity, dispersion) — a constant kick moves the closed
    orbit, not the map about it.
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

    def kick(self, ref: ReferenceParticle) -> np.ndarray:
        """Constant (inhomogeneous) part of the affine map, ``(6,)``.

        Zero for every element whose action is a pure linear map — which is all of
        them but :class:`~accsim.elements.corrector.Corrector`. Overriding this is
        how an element adds a coordinate-independent offset, and the *only*
        supported way: it is what :meth:`track` and
        :meth:`~accsim.lattice.Lattice.transfer_map` accumulate.
        """
        return np.zeros(DIM)

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """Map a 6D ``state`` through the element (returns a new array).

        The default is the affine map ``matrix(ref) @ state + kick(ref)`` — exact
        for every linear element, so element-by-element tracking of a purely linear
        lattice equals a single :meth:`~accsim.lattice.Lattice.transfer_map` product.
        Nonlinear elements (e.g. :class:`~accsim.elements.rfcavity.RFCavity`, whose
        ``sin`` kick gives the RF bucket its separatrix) override this. This is the
        seam the long-term (Stage 3+) tracker plugs into.

        ``state`` may be a single ``(6,)`` vector or a ``(6, n)`` bunch; the kick
        broadcasts over the particle axis (it is the same for every particle).
        """
        out = self.matrix(ref) @ state
        k = self.kick(ref)
        return out + (k if out.ndim == 1 else k[:, None])

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"{type(self).__name__}(length={self.length}{name})"
