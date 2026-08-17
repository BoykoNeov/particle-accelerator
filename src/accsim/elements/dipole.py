"""Dipole: a bending magnet, optionally combined-function and with edge angles."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .alignment import arc_motion, frame_change, roll_motion
from .element import Element
from .quadrupole import _focusing_block


def _dispersion_integrals(K: float, L: float) -> tuple[float, float, float]:
    r"""Branch-smooth path integrals for the horizontal Hill equation ``u'' + K u = drive``.

    Returns ``(c1, s1, c2)`` where, with ``w = sqrt(|K|)``:

    - ``s1 = sin(wL)/w``          (focusing) / ``sinh(wL)/w`` (defocusing) -> ``L`` as ``K -> 0``
    - ``c1 = (1 - cos wL)/K``     / ``(1 - cosh wL)/K``                     -> ``L^2/2``
    - ``c2 = (s1 - L)/K``                                                   -> ``-L^3/6``

    The combined-function dipole's dispersion is ``R16 = h*c1``, ``R26 = h*s1``,
    and its longitudinal slip carries ``h^2*c2`` (see :meth:`Dipole._arc_matrix`).
    All three have removable singularities at ``K = 0`` (the ``h^2 = -k1`` tune),
    handled by the leading Taylor terms so a combined-function magnet tuned exactly
    there is still exact to machine precision.
    """
    if abs(K) < 1e-9:
        s1 = L - K * L**3 / 6.0
        c1 = L**2 / 2.0 - K * L**4 / 24.0
        c2 = -(L**3) / 6.0 + K * L**5 / 120.0
        return c1, s1, c2
    if K > 0.0:
        w = math.sqrt(K)
        cos_wl, s1 = math.cos(w * L), math.sin(w * L) / w
    else:
        w = math.sqrt(-K)
        cos_wl, s1 = math.cosh(w * L), math.sinh(w * L) / w
    c1 = (1.0 - cos_wl) / K
    c2 = (s1 - L) / K
    return c1, s1, c2


def _edge_matrix(h: float, e: float) -> np.ndarray:
    r"""Thin hard-edge pole-face focusing kick for edge angle ``e`` [rad].

    A pole face rotated by ``e`` (``e = 0`` is the sector face; ``e = theta/2``
    the symmetric rectangular face) acts as a thin quadrupole-like kick at the
    entrance/exit of the body. In the **hard-edge** limit (zero fringe extent,
    ``FINT = 0``) the linear map is the identity except:

        px -> px + h*tan(e) * x      (R21 = +h tan e)
        py -> py - h*tan(e) * y      (R43 = -h tan e)

    So a positive edge angle **defocuses horizontally and focuses vertically** --
    the sign is fixed by the geometry of the rotated face (the field the particle
    sees lengthens on the outside of the bend), not remembered. Each 2x2 block
    ``[[1, 0], [+-h tan e, 1]]`` has unit determinant, so the kick is symplectic.

    The fringe-field correction (``e -> e - psi`` in the *vertical* plane only,
    ``psi = h*g*fint*(1 + sin^2 e)/cos e``) is deliberately **not** applied here:
    this is the hard-edge map, the apples-to-apples match to MAD-X ``sbend`` with
    its default ``FINT = HGAP = 0``. Fringe is a separate, opt-in refinement.
    """
    E = np.eye(DIM)
    if h == 0.0 or e == 0.0:
        return E  # no bending or no rotation -> no edge focusing
    t = h * math.tan(e)
    E[PX, X] = t  # horizontal defocus for e > 0
    E[PY, Y] = -t  # vertical focus for e > 0
    return E


class Dipole(Element):
    r"""A dipole of arc length ``L`` and bend angle ``theta`` [rad].

    The reference orbit curves with radius ``rho = L/theta`` (curvature
    ``h = 1/rho = theta/L``); bending is horizontal (the ``x`` plane). Two optional
    refinements, both **off by default** (so the default is a pure sector bend,
    byte-identical to the original):

    - ``k1`` -- a **combined-function** quadrupole gradient in the body [m^-2];
    - ``e1`` / ``e2`` -- entrance / exit **pole-face** rotation angles [rad].

    The body's linear 6x6 map is ``exp(L*A)`` of the (combined-function) bend
    Hamiltonian generator (pinned symbolically and cross-checked entrywise against
    xtrack and MAD-X). With ``k1 = 0`` the non-trivial body entries are, with
    ``C = cos theta``, ``S = sin theta``:

    - **Horizontal** (weak geometric focusing): ``R11 = R22 = C``,
      ``R12 = S/h = rho*S``, ``R21 = -h*S``.
    - **Dispersion** (coupling to ``delta``): ``R16 = (1-C)/h = rho*(1-C)``,
      ``R26 = S``. A higher-momentum particle bends less, so it is displaced
      outward (``R16 > 0``).
    - **Vertical**: a plain drift (``R34 = L``) -- a pure sector bend has no
      vertical focusing.
    - **Longitudinal** (path-length / time-of-flight): ``R51 = -S``,
      ``R52 = (C-1)/h = -rho*(1-C) = -R16``, and
      ``R56 = rho*S - L + L/gamma0^2``. The ``R51``/``R52`` terms are exactly the
      symplectic partners of the dispersion (``R51 = R21*R16 - R11*R26``); ``R56``
      is the drift slip ``L/gamma0^2`` minus the extra arc the design orbit
      travels, ``rho*(theta - S)``.

    **Combined function** (``k1 != 0``). The horizontal focusing becomes
    ``K_x = h^2 + k1`` (geometric weak focusing *plus* the gradient) and the
    vertical ``K_y = -k1``, so ``k1 > 0`` focuses ``x`` and defocuses ``y`` just
    like a :class:`Quadrupole`. Dispersion, ``R51``/``R52`` and the ``R56`` slip
    all pick up the gradient through ``K_x``; the map reduces to the pure sector at
    ``k1 = 0`` and to a pure :class:`Quadrupole` at ``h = 0``. See
    :meth:`_combined_function_body`.

    **Edge angles.** The full map is ``Edge(e2) @ Body @ Edge(e1)`` -- the
    entrance edge acts first. Each edge is the hard-edge kick of
    :func:`_edge_matrix`. Two consequences worth naming, both exact in this linear
    hard-edge model:

    - **Rectangular bend** (``e1 = e2 = theta/2``): the two edges *exactly* cancel
      the body's horizontal weak focusing, leaving the horizontal block equal to a
      drift ``[[1, rho*sin theta], [0, 1]]`` (``R21 = 0`` to machine precision),
      while the vertical plane gets all its focusing from the edges
      (``R43 ~ -2 h tan(theta/2)``).
    - Edges are optics-active (they change beta, tune, chromaticity and dispersion
      through composition) but add no length and no direct longitudinal coupling.

    As ``theta -> 0`` every curvature term vanishes (and the edges too, since
    ``h -> 0``) and the map reduces exactly to a :class:`Drift` of length ``L``
    (``R56 -> L/gamma0^2``).

    **A bending dipole refuses to be displaced** (K1), and the refusal is measured
    rather than cautious. K1's misalignment is the translation ``d + body(state - d)``
    — one translation in, the same one back out — which is right for a *straight*
    element. A bend rotates the reference frame through itself, so the exit
    translation lives in a frame turned by ``theta`` and is not the entry one:
    displacing a bend is a **rigid-body** displacement of a curved body, with an
    angular and a path-length consequence the straight formula does not have. xtrack
    implements exactly that distinction (its misalignment header falls back to the
    straight formula only when ``angle == 0``), and the two models differ by ``3.6e-5``
    on a 0.3 mm shift where the *aligned* maps differ by ``5.8e-9``
    (``tests/reference/test_misalignment_xtrack.py``). Rather than quietly ship the
    wrong one, :meth:`kick` and :meth:`_track_body` raise
    :class:`NotImplementedError` when ``angle != 0`` and an offset is set. A
    straight dipole (``angle = 0``, i.e. a gradient magnet) is displaced normally.

    **A bending dipole may be rolled, and that curved geometry is now implemented**
    (K2) — it is the piece the refusal above said was missing, done for the roll
    rather than for the offset. ``roll`` turns the magnet about the beam axis while
    the machine stays where it is (MAD-X ``EALIGN``'s ``DPSI``, xtrack's
    ``rot_s_rad_no_frame``), which is *not* the same as rolling the reference frame
    with it (MAD-X ``TILT``): the frame-following version has **exactly zero kick**,
    because the design orbit was rolled too. What a real roll error produces, to
    first order in ``phi`` and exactly in the bend angle, is

        Delta p_y = -phi sin(angle),     Delta y = -phi rho (1 - cos angle),

    an angle **and** an offset — the arc's sagitta tipped out of the plane. The
    horizontal loss is only second order (``1 - cos phi``), and there is a residual
    frame roll ``phi (1 - cos angle)`` that makes a rolled bend a genuine **coupling**
    source as well. Because the kick is momentum-dependent through the body, a rolled
    bend is the package's first **source** of vertical dispersion: everything else
    that produces ``D_y`` (G1's skew quadrupole) only rotates dispersion the
    horizontal bends already made. See :meth:`_alignment_exit`.
    """

    def __init__(
        self,
        length: float,
        angle: float,
        k1: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        if length == 0.0 and angle != 0.0:
            raise ValueError("a finite bend angle requires a positive length")
        self.angle = float(angle)
        self.k1 = float(k1)
        self.e1 = float(e1)
        self.e2 = float(e2)

    @property
    def curvature(self) -> float:
        """Curvature ``h = 1/rho = theta/L`` [m^-1] (0 for a straight dipole)."""
        return self.angle / self.length if self.length > 0.0 else 0.0

    @property
    def rho(self) -> float:
        """Bending radius ``rho = L/theta`` [m] (``inf`` for a straight dipole)."""
        return self.length / self.angle if self.angle != 0.0 else math.inf

    def _arc_matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """The bare bend body (no edges)."""
        L = self.length
        theta = self.angle
        M = np.eye(DIM)

        # Straight limit with no gradient: a zero-angle "bend" is just a drift.
        if theta == 0.0 and self.k1 == 0.0:
            M[X, PX] = L
            M[Y, PY] = L
            M[ZETA, DELTA] = L / ref.gamma0**2
            return M

        if self.k1 != 0.0:
            return self._combined_function_body(ref)

        # Pure sector bend (no gradient): the original closed form, byte-identical.
        h = theta / L  # = 1/rho
        c, s = math.cos(theta), math.sin(theta)

        # Horizontal plane + dispersion.
        M[X, X] = c
        M[X, PX] = s / h
        M[X, DELTA] = (1.0 - c) / h
        M[PX, X] = -h * s
        M[PX, PX] = c
        M[PX, DELTA] = s
        # Vertical plane: drift.
        M[Y, PY] = L
        # Longitudinal: path-length coupling (symplectic partners of dispersion)
        # plus the drift-like slip reduced by the extra design-orbit arc length.
        M[ZETA, X] = -s
        M[ZETA, PX] = (c - 1.0) / h
        M[ZETA, DELTA] = s / h - L + L / ref.gamma0**2
        return M

    def _combined_function_body(self, ref: ReferenceParticle) -> np.ndarray:
        r"""Body map with a quadrupole gradient ``k1`` (``exp(L*A)``, closed form).

        Equations of motion ``x'' + (h^2 + k1) x = h*delta``, ``y'' - k1 y = 0``:
        horizontal focusing is the *sum* of geometric weak focusing ``h^2`` and
        the gradient ``k1`` (``K_x = h^2 + k1``), while the vertical plane sees
        ``K_y = -k1`` -- so ``k1 > 0`` focuses ``x`` and defocuses ``y``, exactly
        as in :class:`Quadrupole`. Reduces to the pure sector at ``k1 = 0`` and to
        a pure :class:`Quadrupole` at ``h = 0`` (dispersion vanishes with ``h``).
        """
        L = self.length
        h = self.curvature
        Kx = h * h + self.k1
        M = np.eye(DIM)
        # Transverse blocks: Hill equation with K_x (x) and K_y = -k1 (y).
        M[np.ix_([X, PX], [X, PX])] = _focusing_block(Kx, L)
        M[np.ix_([Y, PY], [Y, PY])] = _focusing_block(-self.k1, L)
        # Dispersion (driven by the h*delta term) and its symplectic partners.
        c1, s1, c2 = _dispersion_integrals(Kx, L)
        r16, r26 = h * c1, h * s1
        M[X, DELTA] = r16
        M[PX, DELTA] = r26
        M[ZETA, X] = -r26  # R51 = -R26
        M[ZETA, PX] = -r16  # R52 = -R16
        M[ZETA, DELTA] = L / ref.gamma0**2 + h * h * c2
        return M

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        body = self._arc_matrix(ref)
        if self.e1 == 0.0 and self.e2 == 0.0:
            return body  # pure sector: byte-identical to the original map
        h = self.curvature
        # Entrance edge acts first: M = Edge(e2) @ Body @ Edge(e1).
        return _edge_matrix(h, self.e2) @ body @ _edge_matrix(h, self.e1)

    def _alignment_exit(self, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
        """The rigid motion that puts a **rolled bend's** exit face back (K2).

        For a straight element (or an unrolled one) this is the base class's plain
        inverse rotation. For a *bending* magnet it is not, and that is the whole of
        K2: rolling the magnet by ``phi`` about the entrance ``s`` axis leaves its
        exit frame at

            T = A^-1 . R_s(phi) . A,

        where ``A`` (:func:`~accsim.elements.alignment.arc_motion`) is the design
        arc's own rigid motion. Conjugating by ``A`` is what makes ``T`` *not* a
        rotation about ``s``: it comes out as a displacement, a pitch, a yaw and only
        ``phi cos(angle)`` of roll, so undoing it needs the full frame change
        (:func:`~accsim.elements.alignment.frame_change`) rather than
        ``s_rotation(-roll)``.

        Two consequences, both first order in ``phi`` and both measured against
        xtrack rather than argued:

        - a **vertical angle** ``-phi sin(angle)`` — the roll acts on the bend's
          *chord*, not on its angle, so this is ``phi theta`` only for a weak bend;
        - a **vertical offset** ``-phi rho (1 - cos angle)`` — the sagitta of the arc,
          tipped out of the plane. In accsim's dispersion solve this term dominates
          the vertical dispersion, so dropping it is not a small error.
        """
        if self.angle == 0.0 or self.roll == 0.0:
            return super()._alignment_exit(ref)
        arc = arc_motion(self.angle, self.rho)
        motion = np.linalg.solve(arc, roll_motion(self.roll) @ arc)
        return frame_change(motion, ref)

    def _refuse_misalignment(self) -> None:
        """A *bending* dipole may not be **displaced** — class docstring (K1).

        Rolled it may be: K2 implements the curved-body geometry for the roll, which
        is exactly what this refusal said was missing. The offset is still refused,
        because a *translated* curved body is a different rigid motion again and
        nothing in the package needs it — see the class docstring.
        """
        if self.is_displaced and self.angle != 0.0:
            raise NotImplementedError(
                f"cannot displace the bending Dipole {self.name!r} (angle={self.angle}): "
                "K1's misalignment is a translation of a straight element, and a bend's "
                "reference frame rotates through it, so entry and exit translations are "
                "not the same transformation. xtrack models this as a rigid-body "
                "displacement of the curved body (measured disagreement 3.6e-5 against "
                "an aligned-model difference of 5.8e-9 — tests/reference/"
                "test_misalignment_xtrack.py), which accsim does not implement. Represent "
                "a bend's steering error with an explicit Corrector, or displace the "
                "quadrupoles, which is where a real machine's orbit comes from anyway"
            )

    def kick(self, ref: ReferenceParticle) -> np.ndarray:
        self._refuse_misalignment()
        return super().kick(ref)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        self._refuse_misalignment()
        return super()._track_body(state, ref)

    def __repr__(self) -> str:
        grad = f", k1={self.k1}" if self.k1 else ""
        edges = f", e1={self.e1}, e2={self.e2}" if (self.e1 or self.e2) else ""
        return f"Dipole(length={self.length}, angle={self.angle}{grad}{edges}{self._repr_tail()})"
