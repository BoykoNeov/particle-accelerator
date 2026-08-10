"""Lattice: an ordered sequence of elements and its accumulated transfer map."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .coords import DIM
from .elements.element import Element
from .reference import ReferenceParticle


class Lattice:
    """An ordered sequence of :class:`Element` sharing one reference particle.

    Stage 0 provides the accumulated linear transfer map. Twiss propagation,
    closed-orbit finding, tunes and chromaticity are layered on in later stages.
    """

    def __init__(self, elements: Iterable[Element], ref: ReferenceParticle) -> None:
        self.elements: list[Element] = list(elements)
        self.ref = ref

    @property
    def length(self) -> float:
        """Total geometric length [m]."""
        return float(sum(e.length for e in self.elements))

    def element_matrices(self) -> list[np.ndarray]:
        """Per-element transfer matrices, in lattice order."""
        return [e.matrix(self.ref) for e in self.elements]

    def transfer_matrix(self) -> np.ndarray:
        """Accumulated 6x6 map from entrance to exit.

        Elements act left-to-right on the beam, so the matrices compose
        right-to-left: ``M_total = M_last @ ... @ M_first``.
        """
        M = np.eye(DIM)
        for elem in self.elements:
            M = elem.matrix(self.ref) @ M
        return M

    def transfer_map(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Accumulated **affine** map ``(M, k)``: ``state_out = M @ state_in + k``.

        :meth:`transfer_matrix` is the homogeneous half; ``k`` accumulates the
        constant kicks of any
        :class:`~accsim.elements.corrector.Corrector` in the sequence and is
        exactly zero otherwise, so ``M`` alone remains the whole map for every
        optics calculation in the package.

        Composition follows the same right-to-left rule as the matrices. Applying
        element 1 then element 2,

            x -> M2 (M1 x + k1) + k2 = (M2 M1) x + (M2 k1 + k2),

        so a kick is transported by *everything downstream of it* — which is why
        the same kick at two different places closes into two different orbits.
        """
        M = np.eye(DIM)
        k = np.zeros(DIM)
        for elem in self.elements:
            Me = elem.matrix(self.ref)
            M = Me @ M
            k = Me @ k + elem.kick(self.ref)
        return M, k

    def one_turn_matrix(self) -> np.ndarray:
        """One-turn map for a closed sequence (alias of :meth:`transfer_matrix`)."""
        return self.transfer_matrix()

    def one_turn_map(self) -> tuple[np.ndarray, np.ndarray]:
        """One-turn affine map for a closed sequence (alias of :meth:`transfer_map`)."""
        return self.transfer_map()

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, index: int) -> Element:
        return self.elements[index]

    def __repr__(self) -> str:
        return f"Lattice({len(self.elements)} elements, length={self.length} m)"


def matrix_of(elements: Sequence[Element], ref: ReferenceParticle) -> np.ndarray:
    """Convenience: accumulated transfer matrix of a bare element sequence."""
    return Lattice(elements, ref).transfer_matrix()
