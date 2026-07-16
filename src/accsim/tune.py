"""Tracking-based tune measurement (NAFF) — independent of ``twiss.tunes``.

The matrix tune in :func:`accsim.twiss.tunes` comes from accumulating the
Courant-Snyder phase advance element-by-element. This module measures the tune a
second, unrelated way: **track a particle for many turns and find the frequency
of its betatron oscillation**. That is how a real machine does it (turn-by-turn
BPM data), and two independent routes to the same number catch a convention error
that either route alone would hide.

Scope of the check (do not oversell it): with ``nonlinear=False`` the tracking
applies the *same* one-turn matrix that ``tunes()`` is built from, so agreement
validates the **extraction method**, not the one-turn map itself. The map is
pinned separately by the element tests and the xtrack cross-check.

Conventions (see ``docs/CONVENTIONS.md`` → *Tracking-based tune (NAFF)*):

- **Only the fractional tune is observable.** Turn-by-turn data samples the
  betatron phase once per turn, so an integer number of full rotations is
  invisible. ``tracked_tunes`` returns ``Q mod 1``; compare it against
  ``tunes()`` (which returns the *full* integer+fractional tune) modulo 1.
- **The ellipse comes from the tracked data, not from Twiss.** Normalising with
  ``closed_twiss``'s ``beta``/``alpha`` would import the very module this is meant
  to cross-check — a bug in ``match_periodic`` would corrupt both sides and cancel.
  Instead ``ellipse_from_trajectory`` recovers ``beta``/``alpha`` from the
  trajectory's own covariance, so nothing here reads ``twiss.py``.
- **The signal is complex, so the tune is unambiguous.** A real signal (position
  only) cannot distinguish ``Q`` from ``1-Q`` — its spectrum is symmetric. Using
  the phase-space pair as ``z = X - i*PX`` gives a *signed* rotation direction and
  resolves ``Q`` over the full ``[0, 1)``. The sign of that combination is fixed by
  this codebase's phase convention and is pinned empirically by
  ``tests/analytic/test_tracked_tune.py``.

Long-term symplecticity (that the tracked motion neither damps nor blows up) is a
separate, already-established check — ``tests/analytic/test_tracking_stability.py``.

Baseline module: numpy/scipy only, so no feature switch (see the working agreement).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .coords import PX, PY, X, Y
from .lattice import Lattice
from .tracking import Particle, Tracker

__all__ = ["ellipse_from_trajectory", "naff", "tracked_tunes"]


def _hann(n: int) -> np.ndarray:
    """Periodic Hann window ``1 - cos(2*pi*k/n)``, symmetric about the midpoint.

    Laskar's NAFF weights the inner product with this window: it kills the
    spectral leakage of a plain (rectangular) transform, which is what lets the
    refined frequency converge far past the ``1/N`` FFT bin resolution. The
    normalisation is irrelevant here — the frequency is found by *maximising* a
    modulus, which is scale-invariant.
    """
    return 1.0 - np.cos(2.0 * np.pi * np.arange(n) / n)


def _projection(windowed: np.ndarray, freq: float) -> float:
    """``|<windowed, exp(2i*pi*freq*k)>|`` — the NAFF inner product at ``freq``."""
    k = np.arange(windowed.size)
    return float(abs(np.sum(windowed * np.exp(-2.0j * np.pi * freq * k))))


def _dprojection2(windowed: np.ndarray, freq: float) -> float:
    """``d|<windowed, exp(2i*pi*f*k)>|^2 / df`` — zero exactly at the spectral peak."""
    k = np.arange(windowed.size)
    e = np.exp(-2.0j * np.pi * freq * k)
    p = np.sum(windowed * e)
    dp = np.sum(windowed * (-2.0j * np.pi * k) * e)
    return float((np.conj(p) * dp).real)


def naff(signal: np.ndarray, *, window: bool = True) -> float:
    """Frequency of the dominant spectral line of a complex ``signal``, in ``[0, 1)``.

    Units are cycles per sample — for turn-by-turn data that is exactly the
    fractional tune.

    Two steps, following Laskar:

    1. a windowed FFT locates the peak bin (resolution ``1/N``);
    2. the frequency is refined by maximising the continuous inner product
       ``|<z, exp(2i*pi*f*k)>|`` over the bracketing bins with Brent's method —
       this is the step that beats the FFT's ``1/N`` grid;
    3. that maximum is then *polished* by root-finding its derivative.

    Step 3 is not redundant. Locating a maximum by comparing function values is
    limited to ~``sqrt(eps)`` in the argument, because the modulus is quadratic at
    its peak — float64 noise of ``eps`` in the value maps to ``sqrt(eps)`` in ``f``
    (scipy's ``fminbound`` makes this explicit, flooring its tolerance at
    ``sqrt(eps)*|f|``, which caps the answer near 1e-9). The derivative instead
    crosses *zero linearly* there, so root-finding it recovers the lost half of the
    digits and lands at ~1e-13.

    A *complex* signal is required: the returned frequency is signed (a real
    signal's spectrum is symmetric and cannot separate ``f`` from ``1-f``).
    """
    z = np.asarray(signal, dtype=complex)
    if z.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {z.shape}")
    n = z.size
    if n < 8:
        raise ValueError(f"need at least 8 samples to resolve a frequency, got {n}")

    zw = z * _hann(n) if window else z
    df = 1.0 / n
    f0 = int(np.argmax(np.abs(np.fft.fft(zw)))) * df

    # The Hann main lobe is two bins wide, so the true peak lies within +/-1 bin of
    # the coarse maximum -- a bracket Brent can search without catching a side lobe.
    res = minimize_scalar(
        lambda f: -_projection(zw, f),
        bounds=(f0 - df, f0 + df),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    f = float(res.x)

    # Polish (see the docstring): the derivative changes sign across the peak, so a
    # root-find on it beats the value-comparison floor. The bracket only has to
    # straddle the true peak -- a few hundred times fminbound's ~sqrt(eps)*|f| error
    # is ample and stays far inside the main lobe. If the sign does not change the
    # peak is not bracketed (a pathological/degenerate spectrum): keep the coarse
    # answer rather than trusting a root outside the bracket.
    pad = max(1.0e-6, 1.0e-6 * df)
    lo, hi = f - pad, f + pad
    if _dprojection2(zw, lo) > 0.0 > _dprojection2(zw, hi):
        f = float(brentq(lambda x: _dprojection2(zw, x), lo, hi, xtol=1.0e-15))
    return f % 1.0


def ellipse_from_trajectory(u: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    r"""Courant-Snyder ``(beta, alpha)`` recovered from a tracked trajectory alone.

    Over many turns a non-resonant betatron phase samples the invariant ellipse
    uniformly, so the trajectory's covariance is the CS ellipse scaled by the
    action ``J``::

        Sigma = <[[u*u, u*up], [u*up, up*up]]> = J * [[beta, -alpha], [-alpha, gamma]]

    Since ``beta*gamma - alpha^2 = 1`` exactly, ``det Sigma = J^2``, which fixes the
    scale without knowing ``J``::

        J = sqrt(det Sigma),   beta = Sigma_11 / J,   alpha = -Sigma_12 / J

    This is what makes the tune check independent: ``beta``/``alpha`` come from the
    data, never from ``twiss.py``. The mean is removed first so a non-zero closed
    orbit does not leak into the ellipse.

    Raises ``ValueError`` if the motion is degenerate (zero amplitude, or confined
    to a line — e.g. an exactly-resonant tune that samples only a few phases).
    """
    a = np.asarray(u, dtype=float)
    b = np.asarray(up, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"u and up must be matching 1-D arrays, got {a.shape} and {b.shape}")

    sigma = np.cov(np.vstack((a, b)), bias=True)
    det = float(np.linalg.det(sigma))
    if det <= 0.0:
        raise ValueError(
            "degenerate trajectory: covariance has non-positive determinant "
            f"({det:.3e}) — zero amplitude or motion confined to a line"
        )
    action = np.sqrt(det)
    return float(sigma[0, 0] / action), float(-sigma[0, 1] / action)


def _plane_tune(u: np.ndarray, up: np.ndarray) -> float:
    """Fractional tune of one plane from its turn-by-turn ``(u, up)`` samples."""
    beta, alpha = ellipse_from_trajectory(u, up)
    root_beta = np.sqrt(beta)

    # Normalised coordinates turn the CS ellipse into a circle, so the motion is a
    # pure rotation and its spectrum is a single line:
    #     U = u / sqrt(beta),  PU = (alpha*u + beta*up) / sqrt(beta)
    # In this codebase's convention the phase advances so that z = U - i*PU rotates
    # with +Q (pinned by test_tracked_tune.py::test_signal_sign_gives_forward_tune).
    u_n = (u - u.mean()) / root_beta
    pu_n = (alpha * (u - u.mean()) + beta * (up - up.mean())) / root_beta
    return naff(u_n - 1.0j * pu_n)


def tracked_tunes(
    lattice: Lattice,
    n_turns: int = 1024,
    *,
    x0: float = 1.0e-6,
    y0: float = 1.0e-6,
    nonlinear: bool = False,
) -> tuple[float, float]:
    """Fractional tunes ``(Qx, Qy)`` measured by tracking, each in ``[0, 1)``.

    Launches one particle at a small transverse offset in both planes (on-momentum,
    ``delta = 0``, so dispersion does not mix into the betatron signal), tracks it
    for ``n_turns``, and reads the betatron frequency of each plane with :func:`naff`.

    Compare against :func:`accsim.twiss.tunes` **modulo 1** — turn-by-turn sampling
    cannot see the integer part. ``n_turns`` need not be a power of two, but the FFT
    is fastest when it is. Amplitudes only set the signal scale in the linear map;
    they matter once ``nonlinear=True`` brings amplitude-dependent detuning.
    """
    if x0 == 0.0 or y0 == 0.0:
        raise ValueError("x0 and y0 must be non-zero — a plane at rest has no tune")

    traj = Tracker(lattice).track_turns(Particle(x=x0, y=y0), n_turns, nonlinear=nonlinear)
    # Drop the duplicate final sample: track_turns returns n_turns + 1 states, and a
    # power-of-two-length record keeps the FFT on its fast path.
    traj = traj[:n_turns]

    qx = _plane_tune(traj[:, X], traj[:, PX])
    qy = _plane_tune(traj[:, Y], traj[:, PY])
    return qx, qy
