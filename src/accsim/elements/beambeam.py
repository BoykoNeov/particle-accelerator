r"""Weak-strong beam-beam: the head-on kick from a round Gaussian strong bunch.

In the *weak-strong* approximation one beam (the "strong" bunch) is a fixed
Gaussian charge distribution; each test particle of the "weak" beam receives a
transverse kick from its electromagnetic field as the two beams cross head-on.
The strong bunch is rigid (it is not itself perturbed), which is what makes this a
single lattice element rather than a self-consistent two-beam solve
(strong-strong, crab cavities, and dynamic aperture are out of scope — see
``docs/ROADMAP.md``).

For a **round** Gaussian strong bunch (``sigma_x = sigma_y = sigma``, ``N``
particles) the field is radial, so the integrated kick is purely radial:

    Delta p_perp = (q2/q1) * (2 N r0 / gamma) * (r_vec / r^2) * (1 - exp(-r^2/2 sigma^2)),

with ``r^2 = x^2 + y^2``, ``r0 = q1^2 e^2/(4 pi eps0 m c^2)`` the **test**
particle's classical radius (``ref.classical_radius_m``), ``gamma``/``q1`` its
Lorentz factor / charge, and ``q2`` the strong-bunch charge. Written per plane and
regularised at ``r -> 0`` (``g(u) = (1-e^{-u})/u -> 1``, ``u = r^2/2 sigma^2``):

    Delta px = K x g(u),   Delta py = K y g(u),   K = (q2/q1) N r0 / (gamma sigma^2).

**Sign (derived from the Lorentz force, not remembered).** ``E`` and ``B`` add for
counter-propagating beams (the factor ``2N``). Like charges (``q1 q2 > 0``, e.g.
proton-proton) **repel -> defocus** (``K > 0``, ``Delta px`` has the sign of ``x``);
opposite charges (``q1 q2 < 0``, e.g. ``e+ e-``) **attract -> focus** (``K < 0``).
The historical ``-(2 N r0/gamma)`` textbook form is the *opposite-charge* case; the
signed ``q2/q1`` here reproduces both.

**Invariants (what "conserves the expected invariants" means, Stage 6 gate 3).**
The kick derives from a potential, so the force is curl-free
(``d Delta px/dy = d Delta py/dx``) — the property that keeps long-term tracking
symplectic. Being radial, it exerts no torque, so the transverse angular momentum
``L_z = x py - y px`` is exactly conserved (the positions are unchanged by the thin
kick). Both hold **only for the round beam**; the elliptical Bassetti-Erskine kick
(``scipy.special.wofz``) is optional generality, not needed for the gate, and would
break the ``L_z`` test.

The **linear** map (:meth:`matrix`) is the ``r -> 0`` limit ``Delta px = K x``,
``Delta py = K y`` — a thin lens focusing **both** planes equally (round symmetry),
unlike a quadrupole. Its strength ``K`` is what the Stage-6 beam-beam tune shift is
built on.
"""

from __future__ import annotations

import numpy as np

from ..coords import DIM, PX, PY, X, Y
from ..reference import ReferenceParticle
from .element import Element


class BeamBeam(Element):
    r"""A thin head-on weak-strong beam-beam kick from a round Gaussian strong bunch.

    Parameters
    ----------
    n_particles
        Population ``N`` of the strong bunch (``> 0``).
    sigma
        RMS transverse size ``sigma`` [m] of the (round) strong bunch (``> 0``).
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
        strong_charge: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(0.0, name=name)
        if n_particles <= 0:
            raise ValueError(f"n_particles must be > 0, got {n_particles}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.n_particles = float(n_particles)
        self.sigma = float(sigma)
        self.strong_charge = float(strong_charge)

    def strength(self, ref: ReferenceParticle) -> float:
        r"""Linear kick strength ``K = (q2/q1) N r0 / (gamma sigma^2)`` [1/m].

        This is the small-amplitude focusing gradient: the thin-lens map is
        ``px -> px + K x`` (both planes). ``K > 0`` defocuses (like charges),
        ``K < 0`` focuses (opposite charges). The effective thin-quad strength is
        ``k1l = -K`` (same magnitude in both planes, unlike a real quadrupole).
        """
        return (
            (self.strong_charge / ref.charge)
            * self.n_particles
            * ref.classical_radius_m
            / (ref.gamma0 * self.sigma**2)
        )

    def matrix(self, ref: ReferenceParticle) -> np.ndarray:
        # Linear (r -> 0) limit: a thin lens focusing both planes by K.
        M = np.eye(DIM)
        K = self.strength(ref)
        M[PX, X] = K
        M[PY, Y] = K
        return M

    def track(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        r"""Full nonlinear round-Gaussian kick; ``state`` is ``(6,)`` or ``(6, N)``.

        Only ``(px, py)`` change: ``Delta px = K x g(u)``, ``Delta py = K y g(u)``
        with ``u = (x^2 + y^2)/(2 sigma^2)`` and ``g(u) = (1 - e^{-u})/u`` (using
        ``-expm1(-u)/u`` for accuracy, ``-> 1`` at ``u = 0`` so the axis is
        singularity-free). A thin kick: positions are untouched.
        """
        out = np.array(state, dtype=float, copy=True)
        x, y = out[X], out[Y]
        u = (x * x + y * y) / (2.0 * self.sigma**2)
        # g(u) = (1 - e^{-u})/u, evaluated as -expm1(-u)/u; the u->0 limit is 1.
        g = np.where(u > 0.0, -np.expm1(-u) / np.where(u > 0.0, u, 1.0), 1.0)
        K = self.strength(ref)
        out[PX] = out[PX] + K * x * g
        out[PY] = out[PY] + K * y * g
        return out

    def __repr__(self) -> str:
        name = f", name={self.name!r}" if self.name is not None else ""
        return (
            f"BeamBeam(n_particles={self.n_particles}, sigma={self.sigma}, "
            f"strong_charge={self.strong_charge}{name})"
        )
