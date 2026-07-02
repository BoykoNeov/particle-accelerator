"""Aperture / collimator: a geometric transverse acceptance boundary."""

from __future__ import annotations

import numpy as np

from ..coords import DIM, X, Y
from ..reference import ReferenceParticle
from .element import Element

_SHAPES = ("circular", "elliptical", "rectangular")


class Aperture(Element):
    r"""A geometric transverse acceptance boundary — particles outside are lost.

    The aperture is **optics-transparent**: its linear transfer matrix is the
    identity (:meth:`matrix` returns ``I``), so inserting one never perturbs
    Twiss, tunes, dispersion, or the one-turn map. Its physics is a *predicate*,
    :meth:`survives`, tested against the transverse coordinates ``(x, y)``. Loss
    *accounting* (which particle dies, and at what longitudinal ``s``) is not done
    here — it lives in the loss-aware tracking pass, which walks the lattice and
    consults each aperture's :meth:`survives`. Keeping the aperture in the element
    sequence is what makes its ``s`` well-defined.

    Shapes (half-apertures ``half_x``, ``half_y`` in metres, aperture centred on
    the reference orbit):

    - ``"circular"``    — radius ``R = half_x``; survives if ``x² + y² ≤ R²``.
    - ``"elliptical"``  — survives if ``(x/half_x)² + (y/half_y)² ≤ 1``.
    - ``"rectangular"`` — survives if ``|x| ≤ half_x`` **and** ``|y| ≤ half_y``.

    Boundary convention: a particle exactly *on* the boundary **survives**
    (inclusive ``≤``), matching xtrack's ``LimitRect``/``LimitEllipse``. Tests
    stay off the knife-edge, so the convention only matters at the measure-zero
    edge.

    A :class:`Collimator` (a thin jaw of finite ``length``) is the same geometric
    test with ``length > 0`` and a label. **Approximation (Stage 4):** survival is
    checked at the element only, not continuously along the jaw, so a particle
    whose transverse excursion *peaks inside* a finite-length collimator and
    returns within the aperture at the exit is not caught. For the pencil-thin
    collimators of a simple loss map this is negligible; it costs accuracy only
    for long jaws with large local betatron slope.
    """

    def __init__(
        self,
        shape: str,
        half_x: float,
        half_y: float | None = None,
        length: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(length, name)
        if shape not in _SHAPES:
            raise ValueError(f"aperture shape must be one of {_SHAPES}, got {shape!r}")
        if half_x <= 0:
            raise ValueError(f"half_x must be > 0, got {half_x}")
        self.shape = shape
        if shape == "circular":
            if half_y is not None and half_y != half_x:
                raise ValueError(
                    "circular aperture takes a single radius (half_x); leave half_y unset"
                )
            half_y = half_x
        elif half_y is None:
            raise ValueError(f"{shape} aperture needs both half_x and half_y")
        elif half_y <= 0:
            raise ValueError(f"half_y must be > 0, got {half_y}")
        self.half_x = float(half_x)
        self.half_y = float(half_y)

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """Identity: an aperture does not bend, focus, or slip the beam."""
        return np.eye(DIM)

    def survives(self, states: np.ndarray) -> np.ndarray | np.bool_:
        """Which particles are *inside* the aperture (``True`` = survives).

        ``states`` is a coordinate array whose first axis is the 6D state: shape
        ``(6,)`` returns a scalar bool; shape ``(6, N)`` returns a ``(N,)`` bool
        array (one per particle). Only ``x`` and ``y`` are consulted.
        """
        states = np.asarray(states, dtype=float)
        x = states[X]
        y = states[Y]
        if self.shape == "rectangular":
            return (np.abs(x) <= self.half_x) & (np.abs(y) <= self.half_y)
        # circular is the elliptical test with half_x == half_y.
        return (x / self.half_x) ** 2 + (y / self.half_y) ** 2 <= 1.0

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return f"Aperture({self.shape!r}, half_x={self.half_x}, half_y={self.half_y}{name})"


class Collimator(Aperture):
    """A finite-length geometric aperture (a jaw). See :class:`Aperture`.

    Identical geometric test to :class:`Aperture`, but with a non-zero
    ``length`` (default 1 mm) so it occupies real longitudinal space in the loss
    map. The entry/exit-only survival check (see the :class:`Aperture`
    approximation note) is the only fidelity cost.
    """

    def __init__(
        self,
        shape: str,
        half_x: float,
        half_y: float | None = None,
        length: float = 1.0e-3,
        name: str | None = None,
    ) -> None:
        super().__init__(shape, half_x, half_y=half_y, length=length, name=name)
