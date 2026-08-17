"""Base class for lattice elements."""

from __future__ import annotations

import abc

import numpy as np

from ..coords import DIM, X, Y
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
    constant part lives in :meth:`kick`, so the homogeneous ``matrix`` remains the
    whole story for optics (beta, tune, chromaticity, dispersion) — a constant kick
    moves the closed orbit, not the map about it. Two things put something there: a
    :class:`~accsim.elements.corrector.Corrector`, and a **transverse
    misalignment** (below).

    Misalignment: ``(dx, dy)`` [m]
    ------------------------------
    ``dx``/``dy`` say where the element actually **is**, relative to where the
    lattice puts it — ``dx > 0`` means the magnet has moved towards positive ``x``,
    so a particle on the design orbit passes through it at body coordinate
    ``-dx``. The map is the element's own map **conjugated by that translation**,

        track(state) = d + body(state - d),      d = (dx, 0, dy, 0, 0, 0),

    which is exactly what a shift means: step into the magnet's frame, apply the
    magnet, step back out. It is also, element for element, what xtrack's
    ``shift_x`` / ``shift_y`` do — the sign is pinned against them in the reference
    suite rather than argued from this docstring.

    **A translation does not touch the homogeneous matrix.** Expanding the
    conjugation for a linear body,

        M (state - d) + k = M state + (k + (I - M) d),

    so the whole linear effect of a misalignment is the *constant* term
    ``(I - M) d``, added in :meth:`kick`, and :meth:`matrix` is returned unchanged.
    A displaced element therefore moves the closed orbit and leaves ``beta``, the
    tunes, dispersion and the coupling **exactly** alone — which is what lets K1's
    ensemble average over displacements be taken at fixed optics. For a thin
    quadrupole ``(I - M) d`` works out to ``theta_x = +k1l dx``,
    ``theta_y = -k1l dy``: one displacement sign, opposite kick signs, because the
    thin quad is ``px -> px - k1l x`` but ``py -> py + k1l y``. A :class:`Drift`
    and a :class:`~accsim.elements.corrector.Corrector` get exactly zero — both
    are translation-invariant.

    **Nonlinear elements are the exception, and knowingly so.** A thin sextupole's
    ``matrix`` is the identity, so ``(I - M) d`` is exactly zero and the *linear*
    theory sees a displaced sextupole as nothing at all — while its real map has an
    ``O(d)`` gradient and an ``O(d^2)`` dipole kick. That is not new blindness: it
    is the same statement as "``matrix`` is the Jacobian **at the origin**", which
    is why :func:`accsim.orbit.closed_orbit_nonlinear` and
    :func:`accsim.orbit.linearised_element_maps` exist. It is asserted rather than
    left to be discovered — see ``docs/CONVENTIONS.md`` -> *Misalignments*.

    Subclasses override :meth:`_kick_body` and :meth:`_track_body` — the element's
    own map in its **own** frame. The public :meth:`kick` and :meth:`track` add the
    misalignment on top; overriding those directly would apply the shift twice or
    not at all.
    """

    def __init__(
        self, length: float, name: str | None = None, *, dx: float = 0.0, dy: float = 0.0
    ) -> None:
        if length < 0:
            raise ValueError(f"element length must be >= 0, got {length}")
        self.length = float(length)
        self.name = name
        self.dx = float(dx)
        self.dy = float(dy)

    @property
    def is_misaligned(self) -> bool:
        """Whether this element carries a nonzero transverse offset."""
        return self.dx != 0.0 or self.dy != 0.0

    def offset(self) -> np.ndarray:
        """The misalignment as a 6D vector ``d = (dx, 0, dy, 0, 0, 0)``."""
        d = np.zeros(DIM)
        d[X] = self.dx
        d[Y] = self.dy
        return d

    @abc.abstractmethod
    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """Return the 6x6 linear transfer matrix for reference particle ``ref``.

        The map acts as ``state_out = matrix @ state_in`` on the column vector
        ``(x, px, y, py, zeta, delta)``.
        """
        raise NotImplementedError

    def _kick_body(self, ref: ReferenceParticle) -> np.ndarray:
        """Constant part of the element's **own** map, in its own frame, ``(6,)``.

        Zero for every element whose action is a pure linear map — which is all of
        them but :class:`~accsim.elements.corrector.Corrector`. Overriding this is
        how an element adds a coordinate-independent offset, and the *only*
        supported way: it is what :meth:`kick` builds on and what
        :meth:`~accsim.lattice.Lattice.transfer_map` ultimately accumulates.
        """
        return np.zeros(DIM)

    def kick(self, ref: ReferenceParticle) -> np.ndarray:
        """Constant (inhomogeneous) part of the affine map, ``(6,)``.

        The element's own constant part (:meth:`_kick_body`) plus the misalignment
        term ``(I - matrix) d`` derived in the class docstring. Exactly zero for a
        perfectly aligned linear element, which is what keeps
        :meth:`~accsim.lattice.Lattice.transfer_map` equal to
        :meth:`~accsim.lattice.Lattice.transfer_matrix` on a design lattice.

        For a **linear** element this is exact: ``matrix @ state + kick`` *is* the
        misaligned map, with no remainder at all. For a nonlinear element it is the
        constant part of the map linearised about the element's own frame origin —
        which for a thin multipole is zero, the deliberate blindness recorded in the
        class docstring.
        """
        k = self._kick_body(ref)
        if not self.is_misaligned:
            return k
        return k + (np.eye(DIM) - self.matrix(ref)) @ self.offset()

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """Map a 6D ``state`` through the element **in its own frame**.

        The default is the affine map ``matrix(ref) @ state + _kick_body(ref)`` —
        exact for every linear element, so element-by-element tracking of a purely
        linear lattice equals a single
        :meth:`~accsim.lattice.Lattice.transfer_map` product. Nonlinear elements
        override this: the :class:`~accsim.elements.rfcavity.RFCavity`, whose ``sin``
        kick gives the RF bucket its separatrix, and the
        :class:`~accsim.elements.sextupole.ThinSextupole` /
        :class:`~accsim.elements.sextupole.Sextupole`, whose ``x^2 - y^2`` kick is
        invisible to ``matrix`` (it has no linear part at the origin) and acts only
        here. This is the seam the long-term tracker plugs into.

        **Override this, not** :meth:`track` — the misalignment conjugation lives
        there, and an override of ``track`` would either apply the shift twice or
        drop it. For the same reason an override that short-circuits to the base
        affine map must call ``super()._track_body(...)``, never ``super().track``.

        An overriding element's map must still be **symplectic** — check it with
        :func:`accsim.symplectic.is_symplectic_map`, which linearises ``track`` by
        finite differences at a given amplitude.

        ``state`` may be a single ``(6,)`` vector or a ``(6, n)`` bunch; the kick
        broadcasts over the particle axis (it is the same for every particle).
        """
        out = self.matrix(ref) @ state
        k = self._kick_body(ref)
        return out + (k if out.ndim == 1 else k[:, None])

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """Map a 6D ``state`` (or a ``(6, n)`` bunch) through the element, as placed.

        :meth:`_track_body` conjugated by the element's misalignment,
        ``d + body(state - d)`` — the exact map of the element where it actually is.
        Identical to ``_track_body`` when ``dx = dy = 0``, so a design lattice is
        untouched by this wrapper.
        """
        if not self.is_misaligned:
            return self._track_body(state, ref)
        state = np.asarray(state, dtype=float)
        d = self.offset()
        d = d if state.ndim == 1 else d[:, None]
        return self._track_body(state - d, ref) + d

    def _repr_tail(self) -> str:
        """The trailing ``, name=..., dx=..., dy=...`` every element's repr shares.

        Each part appears only when it is set, so an aligned element's repr is
        unchanged by K1 — and a misaligned one never hides the offset, which would
        make a printed lattice look perfect while its orbit says otherwise.
        """
        parts = ""
        if self.name is not None:
            parts += f", name={self.name!r}"
        if self.dx != 0.0:
            parts += f", dx={self.dx}"
        if self.dy != 0.0:
            parts += f", dy={self.dy}"
        return parts

    def __repr__(self) -> str:
        return f"{type(self).__name__}(length={self.length}{self._repr_tail()})"
