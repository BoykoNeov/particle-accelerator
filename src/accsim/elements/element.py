"""Base class for lattice elements."""

from __future__ import annotations

import abc

import numpy as np

from ..coords import DIM, X, Y
from ..reference import ReferenceParticle
from .alignment import s_rotation


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

    Misalignment: ``roll`` [rad]
    ----------------------------
    ``roll`` turns the element about the beam axis ``s``, keeping the machine where
    it is — MAD-X ``EALIGN``'s ``DPSI``, xtrack's ``rot_s_rad_no_frame``. It is
    **not** the same thing as a *design* tilt (MAD-X ``TILT``, xtrack's plain
    ``rot_s_rad``), which rolls the reference frame along with the magnet and is a
    lattice-design choice rather than an error; accsim does not offer that one.

    For every **straight** element the two coincide and the map is the conjugation

        track(state) = R(-roll) . body( R(+roll) state ),

    with ``R`` the passive frame rotation :func:`~accsim.elements.alignment.s_rotation`
    — one rotation in, its inverse back out. A rolled quadrupole is then *exactly*
    G1's skew quadrupole at ``roll = 45 deg``, and a rolled sextupole exactly J3's
    skew sextupole at ``30 deg``; both are asserted rather than assumed.

    **A bending dipole is the exception, and it is the whole of K2.** A bend carries
    the reference frame around with it, so its rolled exit face is displaced, pitched
    and yawed relative to where the lattice expects it, and only ``roll * cos(angle)``
    of the entrance rotation is left to undo. :class:`~accsim.elements.dipole.Dipole`
    therefore replaces the exit half with a rigid motion
    (:func:`~accsim.elements.alignment.frame_change`). Composing entry, body and exit
    the general way — an affine map either side of the body — is what makes the two
    cases one code path.

    **Unlike an offset, a roll changes** :meth:`matrix`: it mixes the transverse
    blocks and the ``delta`` column, so beta, the tunes, the dispersion *and* the
    coupling all move. K1's "displacements leave the optics bit-for-bit alone" is a
    statement about translations only.

    Subclasses override :meth:`_matrix_body`, :meth:`_kick_body` and
    :meth:`_track_body` — the element's own map in its **own** frame. The public
    :meth:`matrix`, :meth:`kick` and :meth:`track` add the misalignment on top;
    overriding those directly would apply it twice or not at all.
    """

    def __init__(
        self,
        length: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        if length < 0:
            raise ValueError(f"element length must be >= 0, got {length}")
        self.length = float(length)
        self.name = name
        self.dx = float(dx)
        self.dy = float(dy)
        self.roll = float(roll)

    @property
    def is_misaligned(self) -> bool:
        """Whether this element carries a nonzero transverse offset or roll."""
        return self.dx != 0.0 or self.dy != 0.0 or self.roll != 0.0

    @property
    def is_displaced(self) -> bool:
        """Whether this element carries a nonzero transverse offset (K1), roll aside."""
        return self.dx != 0.0 or self.dy != 0.0

    def offset(self) -> np.ndarray:
        """The misalignment as a 6D vector ``d = (dx, 0, dy, 0, 0, 0)``."""
        d = np.zeros(DIM)
        d[X] = self.dx
        d[Y] = self.dy
        return d

    def _alignment_entry(self, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
        """Affine map ``(M, k)`` from the lattice frame into the element's own frame.

        Roll first as a frame rotation, then step across the offset: a displaced,
        rolled magnet is entered at body coordinate ``R(roll) (state - d)``.
        Overriding this is how an element whose *geometry* differs (a bend) changes
        what "its own frame" means.
        """
        M = s_rotation(self.roll)
        return M, -(M @ self.offset())

    def _alignment_exit(self, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
        """Affine map ``(M, k)`` from the element's own frame back to the lattice's.

        The exact inverse of :meth:`_alignment_entry` for a straight element, which
        is what makes a misalignment a *conjugation* there.
        :class:`~accsim.elements.dipole.Dipole` overrides it, because for a bend it
        is not.
        """
        return s_rotation(-self.roll), self.offset()

    @abc.abstractmethod
    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        """The 6x6 linear transfer matrix of the element in its **own** frame.

        The map acts as ``state_out = matrix @ state_in`` on the column vector
        ``(x, px, y, py, zeta, delta)``. Every element must provide this; the public
        :meth:`matrix` wraps it in the element's alignment.
        """
        raise NotImplementedError

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """Return the 6x6 linear transfer matrix for reference particle ``ref``.

        :meth:`_matrix_body` conjugated by the element's alignment. Returned
        **unchanged** (the same array the subclass built) when the element is aligned
        *or* only displaced — a translation leaves the homogeneous matrix alone, which
        is K1's central fact and is asserted bit-for-bit.
        """
        body = self._matrix_body(ref)
        if self.roll == 0.0:
            return body
        M_in, _ = self._alignment_entry(ref)
        M_out, _ = self._alignment_exit(ref)
        return M_out @ body @ M_in

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

        The element's own constant part (:meth:`_kick_body`) carried through the
        alignment, ``M_out (body k_in + k_body) + k_out``. Exactly zero for a
        perfectly aligned linear element, which is what keeps
        :meth:`~accsim.lattice.Lattice.transfer_map` equal to
        :meth:`~accsim.lattice.Lattice.transfer_matrix` on a design lattice.

        For a pure displacement this collapses to K1's ``(I - matrix) d``, and the
        offset-only path still evaluates that form so a displaced element's kick is
        unchanged to the last bit. For a rolled **bend** the ``k_out`` term is where
        the vertical kick and offset of K2 come from: they are the exit face being
        somewhere else, not the body doing anything new.

        For a **linear** element this is exact: ``matrix @ state + kick`` *is* the
        misaligned map, with no remainder at all. For a nonlinear element it is the
        constant part of the map linearised about the element's own frame origin —
        which for a thin multipole is zero, the deliberate blindness recorded in the
        class docstring.
        """
        k = self._kick_body(ref)
        if not self.is_misaligned:
            return k
        if self.roll == 0.0:  # K1: a translation, and the derived closed form
            return k + (np.eye(DIM) - self.matrix(ref)) @ self.offset()
        M_in, k_in = self._alignment_entry(ref)
        M_out, k_out = self._alignment_exit(ref)
        return M_out @ (self._matrix_body(ref) @ k_in + k) + k_out

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """Map a 6D ``state`` through the element **in its own frame**.

        The default is the affine map ``_matrix_body(ref) @ state + _kick_body(ref)``.
        Elements whose real map is nonlinear override this: the
        :class:`~accsim.elements.rfcavity.RFCavity`, whose ``sin`` kick gives the RF
        bucket its separatrix; the
        :class:`~accsim.elements.sextupole.ThinSextupole` /
        :class:`~accsim.elements.sextupole.Sextupole`, whose ``x^2 - y^2`` kick is
        invisible to ``matrix`` (it has no linear part at the origin) and acts only
        here; :class:`~accsim.elements.drift.Drift`, whose exact geometric map
        carries the ``1/pz`` a matrix cannot; and
        :class:`~accsim.elements.quadrupole.Quadrupole` (with
        :class:`~accsim.elements.skew_quadrupole.SkewQuadrupole`, the same magnet
        rolled), whose focusing is ``k1/(1 + delta)`` and so is a different matrix for
        every momentum. This is the seam the long-term tracker plugs into.

        **Element-by-element tracking of a lattice is therefore not the same thing as
        one** :meth:`~accsim.lattice.Lattice.transfer_map` **product, even when every
        element is "linear".** It was, until the drift's exact map landed; the two now
        agree only for a particle on the design orbit at ``px = py = delta = 0``,
        where the exact maps' Jacobians are the linear matrices entry for entry. Off
        that orbit the difference is physical — the dispersion a transverse angle
        produces, and the chromaticity a momentum spread does — and any code that
        swaps one for the other as a fast path is choosing the linear answer, not an
        equivalent one.

        **Override this, not** :meth:`track` — the misalignment lives there, and an
        override of ``track`` would either apply the shift and roll twice or drop
        them. For the same reason an override that short-circuits to the base affine
        map must call ``super()._track_body(...)``, never ``super().track``.

        An overriding element's map must still be **symplectic** — check it with
        :func:`accsim.symplectic.is_symplectic_map`, which linearises ``track`` by
        finite differences at a given amplitude.

        ``state`` may be a single ``(6,)`` vector or a ``(6, n)`` bunch; the kick
        broadcasts over the particle axis (it is the same for every particle).
        """
        out = self._matrix_body(ref) @ state
        k = self._kick_body(ref)
        return out + (k if out.ndim == 1 else k[:, None])

    def track(
        self,
        state: np.ndarray,
        ref: ReferenceParticle,
        radiation: str = "off",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Map a 6D ``state`` (or a ``(6, n)`` bunch) through the element, as placed.

        :meth:`_track_body` wrapped in the element's alignment,
        ``exit(body(entry(state)))`` — the exact map of the element where it actually
        is. For a pure displacement that is K1's ``d + body(state - d)``, evaluated
        by the same path so a displaced element tracks to the last bit as before.
        Identical to ``_track_body`` for an aligned element, so a design lattice is
        untouched by this wrapper.

        ``radiation`` (default ``"off"``) turns on the classical synchrotron-radiation
        energy loss of :mod:`accsim.radiation_kick`, applied once per element in the
        element's own **body** frame — so a misaligned magnet radiates according to
        where it really is. It is dissipative, so with it on the map is deliberately
        **not** symplectic and ``matrix()`` is no longer its origin Jacobian; that is
        why it is a per-call mode and never the default.

        ``rng`` is required by (and only by) the stochastic radiation models — see
        :data:`accsim.radiation_kick.STOCHASTIC_MODELS`.
        """
        out, _ = self._track_impl(state, ref, radiation, rng, None)
        return out

    def track_with_spin(
        self,
        state: np.ndarray,
        spin: np.ndarray,
        ref: ReferenceParticle,
        radiation: str = "off",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`track`, carrying a spin alongside — returns ``(state, spin)``.

        ``spin`` is a ``(3,)`` unit vector, or a ``(3, n)`` array matching a ``(6, n)``
        bunch, in the same ``(x, y, s)`` axes the state uses. The 6D half is
        **bit-for-bit** what :meth:`track` returns: a spin does not act back on the
        orbit, which is why this axis re-baselines nothing. See :mod:`accsim.spin` for
        the map, and for the two things it deliberately does not do (thin elements do
        not precess; a rolled *bend* raises).

        A roll rotates the spin into and out of the element's body frame, exactly as it
        does the transverse coordinates — a rolled quadrupole precesses a spin like the
        skew quadrupole it is.
        """
        return self._track_impl(state, ref, radiation, rng, spin)

    def _track_impl(
        self,
        state: np.ndarray,
        ref: ReferenceParticle,
        radiation: str,
        rng: np.random.Generator | None,
        spin: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The shared body of :meth:`track` and :meth:`track_with_spin`.

        One code path, so the two can never disagree about which body-frame states the
        radiation and the precession are evaluated on. With ``spin=None`` it is
        :meth:`track` unchanged, down to which arrays get allocated.
        """
        from ..spin import rotate_about_s

        if not self.is_misaligned:
            out = self._track_body(state, ref)
            return self._radiate(state, out, ref, radiation, rng), self._precess(
                state, out, ref, spin
            )
        state = np.asarray(state, dtype=float)
        if self.roll == 0.0:  # K1: a translation, and nothing to rotate
            d = self.offset()
            d = d if state.ndim == 1 else d[:, None]
            body_in = state - d
            body_out = self._track_body(body_in, ref)
            return (
                self._radiate(body_in, body_out, ref, radiation, rng) + d,
                self._precess(body_in, body_out, ref, spin),
            )
        M_in, k_in = self._alignment_entry(ref)
        M_out, k_out = self._alignment_exit(ref)
        if state.ndim != 1:
            k_in, k_out = k_in[:, None], k_out[:, None]
        body_in = M_in @ state + k_in
        body_out = self._track_body(body_in, ref)
        spin_out = self._precess(
            body_in, body_out, ref, rotate_about_s(spin, self.roll) if spin is not None else None
        )
        if spin_out is not None:
            spin_out = rotate_about_s(spin_out, -self.roll)
        return M_out @ self._radiate(body_in, body_out, ref, radiation, rng) + k_out, spin_out

    @property
    def frame_rotation_angle(self) -> float:
        """The angle [rad] by which this element turns the **reference frame** about ``y``.

        Zero for every straight element, and :attr:`.dipole.Dipole.angle` for a bend.
        Nothing in the 6D map needs it — the curvilinear coordinates already have the
        turn built in — but a **spin** is a vector in that frame, so the frame's own
        rotation is half of what a dipole does to it (:mod:`accsim.spin`). Its sign is
        pinned by the ``G = 0`` identity, not chosen.
        """
        return 0.0

    def normalized_field(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        r"""The element's transverse field at ``(x, y)``, normalised to ``(B rho)_0``.

        Returns ``(bx, by)`` in ``1/m``: ``by = h + k1 x + ...``, ``bx = k1 y + ...``,
        the same normalisation the ``k`` strengths already use. It is what the
        synchrotron-radiation kick needs and the *only* thing it needs from an element,
        so a magnet that bends light is a magnet that defines this.

        The default is no field at all, which is right for a drift, an aperture, a
        beam-beam kick and the RF cavity, and harmless for the thin elements (they have
        no length to radiate over — see :mod:`accsim.radiation_kick`).
        """
        zero = np.zeros_like(np.asarray(x, dtype=float))
        return zero, zero

    def _radiate(
        self,
        before: np.ndarray,
        after: np.ndarray,
        ref: ReferenceParticle,
        model: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """The radiation seam: a no-op unless a tracking call asked for it."""
        if model == "off":
            return after
        from ..radiation_kick import radiation_kick

        return radiation_kick(self, before, after, ref, model, rng)

    def _precess(
        self,
        before: np.ndarray,
        after: np.ndarray,
        ref: ReferenceParticle,
        spin: np.ndarray | None,
    ) -> np.ndarray | None:
        """The spin seam: a no-op unless a tracking call is carrying a spin.

        Deliberately shaped like :meth:`_radiate` and fed the same body-frame
        ``before``/``after`` pair, because it needs exactly the same thing — the
        element's own field along the trajectory it just mapped.
        """
        if spin is None:
            return None
        from ..spin import spin_precession

        return spin_precession(self, spin, before, after, ref)

    def _repr_tail(self) -> str:
        """The trailing ``, name=..., dx=..., dy=..., roll=...`` every element shares.

        Each part appears only when it is set, so an aligned element's repr is
        unchanged by K1 or K2 — and a misaligned one never hides the offset or the
        roll, which would make a printed lattice look perfect while its orbit says
        otherwise.
        """
        parts = ""
        if self.name is not None:
            parts += f", name={self.name!r}"
        if self.dx != 0.0:
            parts += f", dx={self.dx}"
        if self.dy != 0.0:
            parts += f", dy={self.dy}"
        if self.roll != 0.0:
            parts += f", roll={self.roll}"
        return parts

    def __repr__(self) -> str:
        return f"{type(self).__name__}(length={self.length}{self._repr_tail()})"
