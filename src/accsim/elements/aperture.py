"""Acceptance boundaries: geometric (transverse) and momentum (longitudinal)."""

from __future__ import annotations

import abc

import numpy as np

from ..coords import DELTA, DIM, X, Y
from ..reference import ReferenceParticle
from .element import Element

_SHAPES = ("circular", "elliptical", "rectangular")


class AcceptanceElement(Element):
    """Base for an **optics-transparent acceptance boundary** — a predicate, not a map.

    Every acceptance element shares two properties. Its linear transfer matrix is
    the identity, so inserting one never perturbs Twiss, tunes, dispersion or the
    one-turn map; and its physics is a *predicate*, :meth:`survives`, rather than a
    transformation. Loss **accounting** — which particle dies, and at what
    longitudinal ``s`` — is not done here: it lives in the loss-aware tracking pass
    (:meth:`accsim.tracking.Tracker.track_bunch_losses`), which walks the lattice
    and consults each acceptance element's predicate. Keeping the boundary in the
    element sequence is what makes its ``s`` well-defined.

    Subclasses differ only in *which* coordinates they test:
    :class:`Aperture` tests ``(x, y)``, :class:`MomentumAperture` tests ``delta``.
    """

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        """Identity: an acceptance boundary does not bend, focus, or slip the beam."""
        return np.eye(DIM)

    @abc.abstractmethod
    def survives(self, states: np.ndarray) -> np.ndarray | np.bool_:
        """Which particles are *inside* the boundary (``True`` = survives).

        ``states`` is a coordinate array whose first axis is the 6D state: shape
        ``(6,)`` returns a scalar bool; shape ``(6, N)`` returns a ``(N,)`` bool
        array, one per particle.
        """


class Aperture(AcceptanceElement):
    r"""A geometric transverse acceptance boundary — particles outside are lost.

    The transverse :class:`AcceptanceElement`: optics-transparent, with its physics
    in the predicate :meth:`survives`, tested against ``(x, y)``. See that base class
    for how loss accounting is divided between the element and the tracking pass.
    :class:`MomentumAperture` is the longitudinal counterpart.

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
        return (
            f"Aperture({self.shape!r}, half_x={self.half_x}, half_y={self.half_y}"
            f"{self._repr_tail()})"
        )


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


class MomentumAperture(AcceptanceElement):
    r"""A **momentum acceptance**: particles outside ``|delta − center| ≤ half_delta`` are lost.

    The longitudinal counterpart of :class:`Aperture`, and the boundary a quantum
    lifetime is measured at (``docs/ROADMAP.md`` → B4). Only ``delta`` is consulted;
    ``zeta`` is not tested, so this is a momentum acceptance rather than a full
    longitudinal one — a ``zeta`` boundary is a different physical object (the RF
    bucket's separatrix is not a rectangle in ``(zeta, delta)``) and is not this
    element.

    ``center`` is why this class exists rather than a bare ``|delta| ≤ A``
    -----------------------------------------------------------------------
    **Set ``center`` to the local closed-orbit ``delta``, not to zero, on any ring
    with radiation.** Synchrotron radiation drains ``delta`` steadily through the
    arcs and the RF cavity restores it in one lump, so the periodic fixed point is
    *not* ``delta = 0`` at most elements — it sags below zero after the cavity and
    climbs back through it. The excursion is of order ``U0/E``, which on a 6.5 GeV
    electron ring is ``3.8e-3`` against an equilibrium spread of ``2.0e-3``: nearly
    two sigma peak to peak.

    That matters far more than it looks, because the quantum lifetime goes as
    ``e^ξ`` with ``ξ = A²/2σ²``. On the B4 ring a symmetric cut placed at the worst
    element is ``ξ = 1.73`` on one side and ``7.20`` on the other where both should
    read ``4.00`` — an order of magnitude in the lifetime, from a boundary that
    looks perfectly reasonable. Centring on the closed orbit is also the correct
    physics: the amplitude that
    :func:`~accsim.lifetime.quantum_lifetime_exact` speaks about is measured from
    the fixed point, not from the design momentum.

    Getting the number: propagate the radiation fixed point through the lattice
    (Newton on ``track_once(s) == s`` with ``radiation="mean"``; tracking to it
    instead would still be drifting after thousands of turns) and read ``delta`` at
    the element's own position. A ring whose losses shift when the boundary is moved
    to a different ``s`` at the same physical acceptance has this wrong — which is
    the gate B4 uses to prove it right.

    Boundary convention: a particle exactly *on* the boundary **survives**
    (inclusive ``≤``), matching :class:`Aperture` and xtrack's limit elements.

    Parameters
    ----------
    half_delta
        Half-width ``A`` of the acceptance in ``delta = Δp/p₀`` (dimensionless).
    center
        The ``delta`` the acceptance is centred on — the local closed orbit. Default
        ``0.0``, which is correct only where the closed orbit is on-momentum.
    length
        Longitudinal extent [m]; ``0.0`` (thin) by default, as for :class:`Aperture`.
    """

    def __init__(
        self,
        half_delta: float,
        center: float = 0.0,
        length: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(length, name)
        if half_delta <= 0:
            raise ValueError(f"half_delta must be > 0, got {half_delta}")
        self.half_delta = float(half_delta)
        self.center = float(center)

    def survives(self, states: np.ndarray) -> np.ndarray | np.bool_:
        """``|delta − center| ≤ half_delta``. Only ``delta`` is consulted."""
        states = np.asarray(states, dtype=float)
        return np.abs(states[DELTA] - self.center) <= self.half_delta

    def __repr__(self) -> str:
        return (
            f"MomentumAperture(half_delta={self.half_delta}, center={self.center}"
            f"{self._repr_tail()})"
        )
