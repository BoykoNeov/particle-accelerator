"""Collider figures of merit: luminosity (Stage 6).

The luminosity ``L`` relates a process cross-section ``sigma`` to its event rate,
``rate = L * sigma``. For two bunched beams collided head-on it is fixed by the
transverse overlap of the two densities and the collision frequency; a crossing
angle reduces it geometrically (the Piwinski factor).

Conventions (see ``docs/CONVENTIONS.md`` -> *Luminosity*):

- **Equal Gaussian beams.** ``L = f_rev * n_b * N1 * N2 / (4 pi sigma_x sigma_y)``.
  The ``4 pi`` (rather than ``2 pi`` with per-beam sizes) *bakes in* ``sigma_1 =
  sigma_2`` in each plane; the general two-size form replaces ``sigma_u`` by
  ``sqrt((sigma_{1u}^2 + sigma_{2u}^2)/2)`` and reduces to this when they match.
- **SI metres internally.** ``L`` comes out in ``m^-2 s^-1``; textbooks quote
  ``cm^-2 s^-1`` (multiply by ``1e-4``). The ``sigma`` inputs are metres.
- ``sigma_u = sqrt(epsilon_u * beta_u^*)`` uses the **geometric** emittance, not
  the normalized ``epsilon_n = beta0 gamma0 epsilon`` (the stray-``gamma`` trap).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special

from .elements.beambeam import BeamBeam
from .reference import ReferenceParticle

__all__ = ["luminosity", "piwinski_reduction", "hourglass_reduction", "beam_beam_tune_shift"]


def piwinski_reduction(
    crossing_angle: float,
    sigma_z: float,
    sigma_cross: float,
) -> float:
    """Geometric (Piwinski) luminosity reduction ``S`` for a crossing angle.

    ``S = 1 / sqrt(1 + (sigma_z * tan(phi/2) / sigma_cross)^2)`` for a *full*
    crossing angle ``phi = crossing_angle`` in the plane whose transverse beam
    size is ``sigma_cross`` (the "crossing plane"); ``sigma_z`` is the RMS bunch
    length. Each beam tilts by half the full angle, hence ``tan(phi/2)``.

    ``S -> 1`` as ``phi -> 0`` (head-on) or ``sigma_z -> 0`` (point-like bunches).
    The **hourglass** effect (``beta`` varying across ``sigma_z`` when
    ``sigma_z >~ beta^*``) is a *separate* reduction and is out of scope here.
    """
    if sigma_cross <= 0:
        raise ValueError(f"sigma_cross must be positive, got {sigma_cross}")
    piwinski = sigma_z * math.tan(0.5 * crossing_angle) / sigma_cross
    return 1.0 / math.sqrt(1.0 + piwinski * piwinski)


def hourglass_reduction(
    sigma_z: float,
    beta_x_star: float,
    beta_y_star: float | None = None,
) -> float:
    """Hourglass luminosity reduction ``H`` for a finite bunch length [-].

    Collisions happen over the whole length of the crossing, not at a point, and
    ``beta`` grows away from the waist as ``beta(s) = beta* (1 + s^2/beta*^2)`` —
    so the beams are *fatter* than ``sigma*`` almost everywhere and the luminosity
    is lower than the point-collision formula says. Weighting the transverse
    overlap ``1/(4 pi sigma_x(s) sigma_y(s))`` by where the collisions actually
    happen gives the multiplicative factor

    .. code-block:: text

        H = (1/(sqrt(pi) sigma_z)) int ds exp(-s^2/sigma_z^2)
                / sqrt((1 + s^2/beta_x*^2) (1 + s^2/beta_y*^2))

    Note the ``exp(-s^2/sigma_z^2)``: the *collision points* have rms
    ``sigma_z/sqrt(2)``, not ``sigma_z``, because both bunches have to be present.
    ``sigma_z`` here is the **per-bunch** rms bunch length [m], the same meaning it
    carries in :func:`piwinski_reduction`.

    For a **round waist** (``beta_y_star is None`` or equal) the integral is exact::

        H = sqrt(pi) * a * exp(a^2) * erfc(a),    a = beta* / sigma_z

    evaluated via the scaled ``erfcx`` so that large ``a`` (short bunches) does not
    overflow. Unequal ``beta*`` has no such closed form and is quadratured.

    Limits: ``H -> 1`` as ``sigma_z/beta* -> 0`` (leading correction
    ``1 - sigma_z^2/(2 beta*^2)``) and ``H -> sqrt(pi) beta*/sigma_z -> 0`` for a
    long bunch.

    This is a **head-on** factor. It does *not* factorise with the crossing-angle
    :func:`piwinski_reduction`: a crossing angle couples the transverse and
    longitudinal integrals through the same growing ``sigma_x(s)``, so the product
    ``S * H`` is an approximation valid for a short bunch or a small angle. See
    ``docs/CONVENTIONS.md`` -> *Hourglass effect*.
    """
    if sigma_z < 0:
        raise ValueError(f"sigma_z must be non-negative, got {sigma_z}")
    if beta_x_star <= 0:
        raise ValueError(f"beta_x_star must be positive, got {beta_x_star}")
    if beta_y_star is not None and beta_y_star <= 0:
        raise ValueError(f"beta_y_star must be positive, got {beta_y_star}")

    if sigma_z == 0.0:
        return 1.0
    if beta_y_star is None or beta_y_star == beta_x_star:
        a = beta_x_star / sigma_z
        # sqrt(pi) a e^{a^2} erfc(a), written with erfcx = e^{a^2} erfc(a) so that
        # the short-bunch limit (a -> inf) stays finite instead of inf * 0.
        return float(math.sqrt(math.pi) * a * special.erfcx(a))

    ax = sigma_z / beta_x_star
    ay = sigma_z / beta_y_star

    def integrand(u: float) -> float:
        return math.exp(-(u**2)) / math.sqrt((1 + (ax * u) ** 2) * (1 + (ay * u) ** 2))

    value, _ = integrate.quad(integrand, -np.inf, np.inf)
    return float(value / math.sqrt(math.pi))


def luminosity(
    n_particles_1: float,
    n_particles_2: float,
    sigma_x: float,
    sigma_y: float,
    f_rev: float,
    n_bunches: int = 1,
    *,
    crossing_angle: float = 0.0,
    sigma_z: float = 0.0,
    crossing_plane: str = "x",
) -> float:
    """Peak luminosity of two colliding bunched beams [``m^-2 s^-1``].

    Head-on, equal Gaussian beams::

        L = f_rev * n_bunches * N1 * N2 / (4 pi sigma_x sigma_y)

    ``f_rev`` is the revolution frequency [Hz] and ``n_bunches`` the number of
    equally-spaced colliding bunches, so ``f_rev * n_bunches`` is the bunch
    collision rate. ``N1``/``N2`` are the bunch populations; ``sigma_x``/
    ``sigma_y`` are the RMS transverse beam sizes **at the interaction point**
    [m] (``sqrt(epsilon * beta^*)`` with *geometric* emittance).

    A non-zero ``crossing_angle`` (full angle [rad], with the bunch length
    ``sigma_z`` [m]) applies the multiplicative :func:`piwinski_reduction` in the
    ``crossing_plane`` (``"x"`` or ``"y"``). The returned value is in
    ``m^-2 s^-1``; multiply by ``1e-4`` for the customary ``cm^-2 s^-1``.

    Assumes equal beam sizes (the ``4 pi`` form) and a Gaussian transverse
    profile; the hourglass effect is neglected.
    """
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError(f"beam sizes must be positive, got sigma_x={sigma_x}, sigma_y={sigma_y}")
    if n_bunches < 1:
        raise ValueError(f"n_bunches must be >= 1, got {n_bunches}")
    if crossing_plane not in ("x", "y"):
        raise ValueError(f"crossing_plane must be 'x' or 'y', got {crossing_plane!r}")

    lum = f_rev * n_bunches * n_particles_1 * n_particles_2 / (4.0 * math.pi * sigma_x * sigma_y)

    if crossing_angle != 0.0:
        sigma_cross = sigma_x if crossing_plane == "x" else sigma_y
        lum *= piwinski_reduction(crossing_angle, sigma_z, sigma_cross)

    return lum


def beam_beam_tune_shift(
    beambeam: BeamBeam,
    ref: ReferenceParticle,
    beta_x: float,
    beta_y: float | None = None,
) -> tuple[float, float]:
    r"""Linear (small-amplitude) head-on beam-beam tune shift ``(dQx, dQy)``.

    The :class:`~accsim.elements.beambeam.BeamBeam` kick linearises at the axis to
    a thin lens ``px -> px + K_x x``, ``py -> py + K_y y``, i.e. an effective
    thin-quad strength ``k1l = -K_u`` with ``(K_x, K_y) = beambeam.strengths(ref)``.
    A thin lens of strength ``k1l`` at a point with beta ``beta_u`` shifts the tune by
    ``dQ_u = beta_u * k1l / (4 pi)`` (derived symbolically from the one-turn
    trace), so

        dQ_u = - beta_u * K_u / (4 pi),
        K_u  = (q2/q1)(2 N r0/gamma) / (sigma_u (sigma_x + sigma_y)).

    For a **round** bunch both reduce to ``K = (q2/q1) N r0/(gamma sigma^2)`` and the
    two planes shift equally; for a **flat** one (C1) the narrow plane takes the
    larger shift.

    ``beta_x``/``beta_y`` are the *unperturbed* beta functions at the interaction
    point [m] (``beta_y`` defaults to ``beta_x`` for a round IP). The shift is
    **signed**: negative for like charges (defocusing, e.g. proton-proton),
    positive for opposite charges (focusing). Its magnitude is the standard
    **beam-beam parameter** ``xi_u = N r0 beta_u* / (4 pi gamma sigma^2)`` (round
    beam) — e.g. ``xi ~ 0.0037`` per IP at the LHC.

    This is the first-order tune shift; the amplitude-dependent detuning carried by
    the full nonlinear kick is out of scope here.
    """
    if beta_y is None:
        beta_y = beta_x
    kx, ky = beambeam.strengths(ref)
    return -beta_x * kx / (4.0 * math.pi), -beta_y * ky / (4.0 * math.pi)
