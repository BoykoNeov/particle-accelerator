"""Skew quadrupole: a normal quadrupole rolled 45 deg -- the x-y coupling source."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .alignment import s_rotation
from .element import Element
from .quadrupole import _focusing_block, thick_quadrupole_map

_SKEW_ROLL = math.pi / 4.0


class SkewQuadrupole(Element):
    r"""A thick skew quadrupole of length ``L`` and skew gradient ``k1s`` [m^-2].

    A skew quadrupole is an ordinary :class:`Quadrupole` **rolled by 45 deg** about
    the longitudinal axis. Where a normal quad focuses one plane and defocuses the
    other independently, a skew quad rotates that field pattern so its force on a
    particle mixes the planes: it is the canonical source of **betatron (x-y)
    coupling**. ``k1s`` is the skew gradient of the equivalent normal quad
    (``k1s = k1`` of the unrolled magnet).

    The body map is the roll conjugation ``R(pi/4) @ Q_body(k1s) @ R(-pi/4)`` of the
    normal quad body (``R`` rotates ``(x, y)`` and ``(px, py)`` together). Working
    that out in closed form gives the block structure, on the pairs ``(x, px)`` and
    ``(y, py)``,

        M_4x4 = [[A, B], [B, A]],   A = (F + D) / 2,   B = (D - F) / 2,

    where ``F = _focusing_block(k1s, L)`` (cos/sin) and ``D = _focusing_block(-k1s, L)``
    (cosh/sinh) are the same two blocks a normal quad uses. The **diagonal** blocks
    ``A`` are the plane's own focusing (the average of the two normal-quad blocks);
    the **off-diagonal** blocks ``B`` are the coupling the roll introduces. This
    form is exact, symplectic (verified symbolically), and:

    - ``k1s = 0`` -> ``F = D`` -> ``B = 0`` and ``A`` is a drift block, so the map is
      a plain :class:`Drift`;
    - ``k1s -> -k1s`` swaps ``F`` and ``D``, flipping ``B`` (the coupling reverses),
      so a negative skew gradient is handled with no special case.

    Longitudinal: like a quad (and a drift), the roll leaves ``(zeta, delta)``
    untouched, so ``R56 = L / gamma0^2``. The whole 6x6 is symplectic by
    construction. The sign of ``k1s`` (which sense of coupling a positive gradient
    produces) is pinned empirically against xtrack in the reference suite; the
    ``R(pi/4)``-roll-of-a-normal-quad identity is the internal analytic gate.
    """

    def __init__(
        self,
        length: float,
        k1s: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        self.k1s = float(k1s)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        F = _focusing_block(self.k1s, L)  # cos/sin  block (focusing plane)
        D = _focusing_block(-self.k1s, L)  # cosh/sinh block (defocusing plane)
        A = 0.5 * (F + D)  # each plane's own focusing (diagonal blocks)
        B = 0.5 * (D - F)  # the coupling the 45 deg roll introduces
        M[np.ix_([X, PX], [X, PX])] = A
        M[np.ix_([Y, PY], [Y, PY])] = A
        M[np.ix_([X, PX], [Y, PY])] = B
        M[np.ix_([Y, PY], [X, PX])] = B
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        r"""The normal quadrupole's L2 map, conjugated by the 45 degree roll.

        ``R(+45) . Q(k1s) . R(-45)`` — the identity the class docstring and
        :meth:`_matrix_body` are already built on, now applied to the *exact* map
        rather than to the matrix. A skew quadrupole **is** a normal quadrupole that
        has been turned, so it must be exactly as momentum-dependent as one: leaving
        it linear while ``Quadrupole(..., roll=-pi/4)`` became chromatic would have
        made the same magnet behave two different ways depending on how it was
        spelled. (``Element.track`` already builds the rolled quadrupole's map by
        this same conjugation, so the two agree by construction, and the analytic
        suite asserts it rather than trusting the arithmetic.)

        :class:`ThinSkewQuadrupole` needs no such treatment, for the reason
        :class:`~accsim.elements.quadrupole.ThinQuadrupole` does not: a thin kick is
        already exact in ``delta``.
        """
        st = np.asarray(state, dtype=float)
        into = s_rotation(-_SKEW_ROLL) @ st
        return s_rotation(+_SKEW_ROLL) @ thick_quadrupole_map(into, self.length, self.k1s, ref)

    def normalized_field(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        r"""The normal quadrupole's field, carried through the same 45 deg roll.

        The position is rotated *into* the body frame and the resulting field vector
        rotated back out, exactly as :meth:`_track_body` does with the map — the same
        magnet must not have two different fields depending on how it is spelled.
        ``|b|^2 = k1s^2 (x^2 + y^2)`` is roll-invariant, so a skew quadrupole radiates
        exactly as much as the normal one it is.
        """
        c, s_ = math.cos(_SKEW_ROLL), math.sin(_SKEW_ROLL)
        xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        xb, yb = c * xa + s_ * ya, -s_ * xa + c * ya  # into the body frame
        bx_b, by_b = self.k1s * yb, self.k1s * xb
        return c * bx_b - s_ * by_b, s_ * bx_b + c * by_b

    def __repr__(self) -> str:
        return f"SkewQuadrupole(length={self.length}, k1s={self.k1s}{self._repr_tail()})"


class ThinSkewQuadrupole(Element):
    r"""A thin skew quadrupole: a zero-length coupling kick of strength ``k1sl`` [m^-1].

    ``k1sl = k1s * L`` is the integrated skew gradient. The map is a pure momentum
    kick that couples the planes:

        px -> px + k1sl * y,
        py -> py + k1sl * x.

    This is the ``L -> 0`` limit of :class:`SkewQuadrupole` at fixed ``k1s * L``
    (verified symbolically) -- equivalently the 45 deg roll of a
    :class:`~accsim.elements.quadrupole.ThinQuadrupole`. Each plane's momentum gains
    a kick proportional to the *other* plane's position (symmetric, ``R[px, y] =
    R[py, x] = k1sl``), which is what makes the 4x4 symplectic while coupling ``x``
    and ``y``. It is the building block for the single-source closest-tune-approach
    (``DeltaQ_min``) analytic gate.
    """

    def __init__(
        self,
        k1sl: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(0.0, name=name, dx=dx, dy=dy, roll=roll)
        self.k1sl = float(k1sl)

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        M = np.eye(DIM)
        M[PX, Y] = self.k1sl  # px kicked by the vertical position
        M[PY, X] = self.k1sl  # py kicked by the horizontal position
        return M

    def __repr__(self) -> str:
        return f"ThinSkewQuadrupole(k1sl={self.k1sl}{self._repr_tail()})"
