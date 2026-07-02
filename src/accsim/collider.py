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

from .elements.beambeam import BeamBeam
from .reference import ReferenceParticle

__all__ = ["luminosity", "piwinski_reduction", "beam_beam_tune_shift"]


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
    a thin lens ``px -> px + K x`` (both planes), i.e. an effective thin-quad
    strength ``k1l = -K`` with ``K = beambeam.strength(ref)``. A thin lens of
    strength ``k1l`` at a point with beta ``beta_u`` shifts the tune by
    ``dQ_u = beta_u * k1l / (4 pi)`` (derived symbolically from the one-turn
    trace), so

        dQ_u = - beta_u * K / (4 pi),   K = (q2/q1) N r0 / (gamma sigma^2).

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
    k = beambeam.strength(ref)
    coeff = k / (4.0 * math.pi)
    return -beta_x * coeff, -beta_y * coeff
