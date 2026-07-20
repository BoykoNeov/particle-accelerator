r"""Weak-strong beam-beam: the head-on kick from a Gaussian strong bunch.

In the *weak-strong* approximation one beam (the "strong" bunch) is a fixed
Gaussian charge distribution; each test particle of the "weak" beam receives a
transverse kick from its electromagnetic field as the two beams cross head-on.
The strong bunch is rigid (it is not itself perturbed), which is what makes this a
single lattice element rather than a self-consistent two-beam solve
(strong-strong, crab cavities, and dynamic aperture are out of scope — see
``docs/ROADMAP.md``).

Both the **round** (``sigma_x = sigma_y``, Stage 6) and the **elliptical**
(``sigma_x != sigma_y``, Bassetti-Erskine, C1) cases are handled by the same
element. They share one prefactor and differ only in the field *shape*:

    Delta p_perp = (q2/q1) * (2 N r0 / gamma) * S(x, y),

with ``r0 = q1^2 e^2/(4 pi eps0 m c^2)`` the **test** particle's classical radius
(``ref.classical_radius_m``), ``gamma``/``q1`` its Lorentz factor / charge, ``N``/``q2``
the strong bunch's population / charge, and ``S = 2 pi eps0 E`` the normalised field
shape [1/m].

**The shape is derived from Coulomb's law, not remembered.** Writing
``1/r^2 = int_0^inf e^{-r^2 t} dt`` makes the convolution of the 2D point field with
the Gaussian charge an elementary Gaussian integral; sympy then returns
(``tests/analytic/test_beam_beam_elliptical.py``, exact)

    S_x = (1/2) int_0^inf dq  x exp(-x^2/(2A) - y^2/(2B)) / (A^{3/2} B^{1/2}),
    S_y = (1/2) int_0^inf dq  y exp(-x^2/(2A) - y^2/(2B)) / (A^{1/2} B^{3/2}),
    A = q + sigma_x^2,   B = q + sigma_y^2.

**Round bunch** (``sigma_x = sigma_y = sigma``). The substitution ``w = 1/(q+sigma^2)``
collapses the integral to the radial closed form
``S = (r_vec/r^2)(1 - exp(-r^2/2 sigma^2))``, i.e. per plane, regularised at the axis
(``g(u) = (1-e^{-u})/u -> 1``, ``u = r^2/2 sigma^2``):

    Delta px = K x g(u),   Delta py = K y g(u),   K = (q2/q1) N r0 / (gamma sigma^2).

**Elliptical bunch** (Bassetti-Erskine). For ``sigma_x > sigma_y`` the same integral has
the closed form, with ``w`` the Faddeeva function (``scipy.special.wofz``) and
``d = 2(sigma_x^2 - sigma_y^2)``:

    S_y + i S_x = sqrt(pi/d) * [ w((x + i y)/sqrt(d))
                    - exp(-x^2/2 sigma_x^2 - y^2/2 sigma_y^2)
                      * w((x sigma_y/sigma_x + i y sigma_x/sigma_y)/sqrt(d)) ].

Note the assignment ``S_y + i S_x`` — **not** ``S_x + i S_y``. That transposition is the
classic Bassetti-Erskine error: it survives both the round limit and the on-axis values,
and only shows up in the off-axis angular structure. It is pinned here against a
brute-force 2D Coulomb integral that shares no code with ``wofz``.

**Sign (derived from the Lorentz force, not remembered).** ``E`` and ``B`` add for
counter-propagating beams (the factor ``2N``). Like charges (``q1 q2 > 0``, e.g.
proton-proton) **repel -> defocus**; opposite charges (``q1 q2 < 0``, e.g. ``e+ e-``)
**attract -> focus**. The historical ``-(2 N r0/gamma)`` textbook form is the
*opposite-charge* case; the signed ``q2/q1`` here reproduces both.

**Invariants.** The kick derives from a potential, so the force is curl-free
(``d Delta px/dy = d Delta py/dx``) — the property that keeps long-term tracking
symplectic — for **both** shapes. The round beam is additionally *radial*, so it exerts
no torque and conserves the transverse angular momentum ``L_z = x py - y px`` exactly;
**the elliptical beam does not**, and that is physical, not a defect.

**Linear limit.** At the axis the kick is a thin lens with *per-plane* gradients

    K_x = (q2/q1)(2 N r0/gamma) / (sigma_x (sigma_x + sigma_y)),
    K_y = (q2/q1)(2 N r0/gamma) / (sigma_y (sigma_x + sigma_y)),

equal to a single ``K`` only for a round beam. They satisfy Gauss's law exactly
(``K_x + K_y`` is the central charge density), which fixes the normalisation
independently of the round limit.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from ..coords import DIM, PX, PY, X, Y
from ..reference import ReferenceParticle
from .element import Element

#: Relative ellipticity ``|sx - sy|/(sx + sy)`` below which the round closed form is
#: used instead of Bassetti-Erskine. At this separation the round formula differs from
#: the true elliptical field by ~1.1e-8 relative (measured: the error is cleanly linear
#: in the ellipticity), which is at or below the accuracy the ``wofz`` difference itself
#: achieves near the axis — so the fallback costs nothing measurable while removing the
#: ``1/sqrt(sigma_x^2 - sigma_y^2)`` division by zero at exact equality.
_ROUND_TOL = 1.0e-8


def _shape_round(x: np.ndarray, y: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial shape ``(r_vec/r^2)(1 - e^{-r^2/2 sigma^2})``, regular on the axis."""
    u = (x * x + y * y) / (2.0 * sigma * sigma)
    # g(u) = (1 - e^{-u})/u as -expm1(-u)/u; the u -> 0 limit is 1.
    safe = np.where(u > 0.0, u, 1.0)
    g = np.where(u > 0.0, -np.expm1(-u) / safe, 1.0)
    return x * g / (2.0 * sigma * sigma), y * g / (2.0 * sigma * sigma)


def _shape_elliptical(
    x: np.ndarray, y: np.ndarray, sigma_x: float, sigma_y: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bassetti-Erskine shape for ``sigma_x != sigma_y`` (either ordering)."""
    # The closed form assumes sigma_x > sigma_y; for a tall bunch, solve the
    # transposed problem and swap the components back.
    if sigma_y > sigma_x:
        s_y, s_x = _shape_elliptical(y, x, sigma_y, sigma_x)
        return s_x, s_y

    # The charge is symmetric in both x and y, so evaluate in the first quadrant and
    # restore the signs. This keeps w(z) off the lower half plane, where it grows like
    # 2 exp(-z^2) and would overflow for a large negative argument.
    ax, ay = np.abs(x), np.abs(y)

    d = 2.0 * (sigma_x * sigma_x - sigma_y * sigma_y)
    root = math.sqrt(d)
    z1 = (ax + 1j * ay) / root
    z2 = (ax * sigma_y / sigma_x + 1j * ay * sigma_x / sigma_y) / root
    damp = np.exp(-(ax**2) / (2.0 * sigma_x**2) - ay**2 / (2.0 * sigma_y**2))

    val = math.sqrt(math.pi / d) * (special.wofz(z1) - damp * special.wofz(z2))
    return np.sign(x) * val.imag, np.sign(y) * val.real


class BeamBeam(Element):
    r"""A thin head-on weak-strong beam-beam kick from a Gaussian strong bunch.

    Parameters
    ----------
    n_particles
        Population ``N`` of the strong bunch (``> 0``).
    sigma
        RMS **horizontal** size ``sigma_x`` [m] of the strong bunch (``> 0``). If
        ``sigma_y`` is omitted this is the (round) size in both planes.
    sigma_y
        RMS **vertical** size [m] (``> 0``). ``None`` (default) means a round bunch.
        Either ordering is allowed — ``sigma_y > sigma`` describes a tall bunch.
    strong_charge
        Charge of the strong-bunch particles in units of ``e`` (signed). Defaults
        to ``+1`` (proton-like). The kick strength scales with ``strong_charge /
        ref.charge``: equal to ``+1`` for a same-species collider (defocusing),
        ``-1`` for particle-antiparticle collisions (focusing).
    """

    def __init__(
        self,
        n_particles: float,
        sigma: float,
        sigma_y: float | None = None,
        strong_charge: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(0.0, name=name)
        if n_particles <= 0:
            raise ValueError(f"n_particles must be > 0, got {n_particles}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if sigma_y is not None and sigma_y <= 0:
            raise ValueError(f"sigma_y must be > 0, got {sigma_y}")
        self.n_particles = float(n_particles)
        self.sigma = float(sigma)
        self.sigma_y = None if sigma_y is None else float(sigma_y)
        self.strong_charge = float(strong_charge)

    @property
    def sigma_x(self) -> float:
        """RMS horizontal size [m] (an alias for ``sigma``)."""
        return self.sigma

    @property
    def is_round(self) -> bool:
        """Whether the bunch is round to within :data:`_ROUND_TOL`."""
        if self.sigma_y is None:
            return True
        total = self.sigma + self.sigma_y
        return abs(self.sigma - self.sigma_y) / total < _ROUND_TOL

    @property
    def _round_sigma(self) -> float:
        """The size used by the round branch: the **geometric** mean.

        ``sqrt(sigma_x sigma_y)`` keeps Gauss's law
        (``K_x + K_y = amplitude/(sigma_x sigma_y)``) exact by construction, for any
        ellipticity, which is why it is the principled choice. Be honest about what
        that buys here, though: the branch only runs below :data:`_ROUND_TOL`, where
        the geometric and arithmetic means differ by ``O(eps^2) ~ 1e-16`` — below
        double precision. The choice is a matter of principle, not accuracy, and no
        test can distinguish the two (verified by mutation testing).
        """
        if self.sigma_y is None:
            return self.sigma
        return math.sqrt(self.sigma * self.sigma_y)

    def amplitude(self, ref: ReferenceParticle) -> float:
        """Prefactor ``(q2/q1)(2 N r0/gamma)`` [m], shared by both shapes."""
        return (
            (self.strong_charge / ref.charge)
            * 2.0
            * self.n_particles
            * ref.classical_radius_m
            / ref.gamma0
        )

    def strengths(self, ref: ReferenceParticle) -> tuple[float, float]:
        r"""Per-plane linear kick strengths ``(K_x, K_y)`` [1/m].

        ``K_u = (q2/q1)(2 N r0/gamma) / (sigma_u (sigma_x + sigma_y))`` — the
        small-amplitude focusing gradients, i.e. the thin-lens map
        ``px -> px + K_x x``, ``py -> py + K_y y``. Positive defocuses (like charges),
        negative focuses (opposite charges). For a round bunch both are equal to
        :meth:`strength`; for a flat one the *narrow* plane is focused harder.
        """
        amp = self.amplitude(ref)
        if self.is_round:
            s = self._round_sigma
            k = amp / (2.0 * s * s)
            return k, k
        sx, sy = self.sigma, self.sigma_y
        assert sy is not None  # guaranteed by is_round
        total = sx + sy
        return amp / (sx * total), amp / (sy * total)

    def strength(self, ref: ReferenceParticle) -> float:
        r"""Linear kick strength ``K = (q2/q1) N r0 / (gamma sigma^2)`` [1/m].

        Round bunches only — an elliptical bunch has no single ``K`` (the two planes
        are focused differently), so this raises; use :meth:`strengths` instead.
        """
        if not self.is_round:
            raise ValueError(
                "an elliptical beam-beam bunch has no single strength K "
                f"(sigma_x={self.sigma}, sigma_y={self.sigma_y}); use strengths(ref) "
                "for the per-plane (K_x, K_y)"
            )
        return self.strengths(ref)[0]

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear (axis) limit: a thin lens focusing the planes by K_x, K_y.
        M = np.eye(DIM)
        kx, ky = self.strengths(ref)
        M[PX, X] = kx
        M[PY, Y] = ky
        return M

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        r"""Full nonlinear Gaussian kick; ``state`` is ``(6,)`` or ``(6, N)``.

        Only ``(px, py)`` change — a thin kick leaves the positions untouched. Round
        bunches take the regularised radial form, elliptical ones Bassetti-Erskine.
        """
        out = np.array(state, dtype=float, copy=True)
        x, y = out[X], out[Y]
        if self.is_round:
            s_x, s_y = _shape_round(x, y, self._round_sigma)
        else:
            assert self.sigma_y is not None  # guaranteed by is_round
            s_x, s_y = _shape_elliptical(x, y, self.sigma, self.sigma_y)
        amp = self.amplitude(ref)
        out[PX] = out[PX] + amp * s_x
        out[PY] = out[PY] + amp * s_y
        return out

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        sigma_y = "" if self.sigma_y is None else f", sigma_y={self.sigma_y}"
        return (
            f"BeamBeam(n_particles={self.n_particles}, sigma={self.sigma}{sigma_y}, "
            f"strong_charge={self.strong_charge}{name})"
        )
