r"""The ``pp`` dilution of ``A_FB`` and its unfolding (milestone A3).

In ``p pbar`` the quark almost always comes from the proton, so the
Collins-Soper axis can be oriented by the beam. In ``pp`` both beams are
identical: the quark direction is *not* known event by event, and the standard
experimental proxy is ``sign(Q_z)`` -- the di-lepton boost direction -- on the
grounds that the valence quark statistically carries more momentum than the sea
antiquark it annihilates. That proxy is right only most of the time, and every
event it gets wrong enters ``A_FB`` with the opposite sign. The measured
asymmetry is therefore **diluted** toward zero.

This module inverts that dilution: given the parton luminosities that feed the
two orientations, recover the parton-level ``A_FB`` that
:func:`accsim.events.afb_hadronic` predicts. It is baseline (numpy only) and
reuses A2's validated angular strengths ``S``/``D`` rather than re-deriving
them; see :mod:`accsim.events.electroweak`.

**Natural units** (``hbar = c = 1``, GeV), as everywhere in :mod:`accsim.events`.

The master formula
------------------
At a fixed parton configuration ``(x_1, x_2)`` the pair's longitudinal momentum
is ``Q_z = (x_1 - x_2) sqrt(s) / 2``, so the proxy ``sign(Q_z)`` is
``sign(x_1 - x_2)``: it is a *deterministic* function of the configuration, not
a random draw. Split the luminosity of flavour ``q`` accordingly:

    L_q^+  = luminosity of configurations where the quark travels along
             ``sign(Q_z)``  -- the proxy is **right**
    L_q^-  = luminosity of configurations where it travels the other way
             -- the proxy is **wrong**

(For ``y > 0``, ``L_q^+ = q(x_1) qbar(x_2)`` and ``L_q^- = qbar(x_1) q(x_2)``;
for ``y < 0`` the two swap, which is why the split is stated by *orientation*
rather than by beam.)

A wrong orientation sends ``cos(theta) -> -cos(theta)``, which flips the sign of
the antisymmetric term and leaves the symmetric one alone. Writing ``c`` for the
``cos(theta)`` actually measured against the proxy axis, the observed
distribution of one flavour is

    L^+ [S (1 + c^2) + 2 D c] + L^- [S (1 + c^2) - 2 D c]
      = (L^+ + L^-) S (1 + c^2) + 2 (L^+ - L^-) D c

so, summing over flavours,

    A_FB^obs(m) = (3/4) * sum_q (L_q^+ - L_q^-) D_q / sum_q (L_q^+ + L_q^-) S_q
    A_FB^true(m) = (3/4) * sum_q (L_q^+ + L_q^-) D_q / sum_q (L_q^+ + L_q^-) S_q

**Dilution reweights the numerator only** -- the denominator (the rate) is
untouched, since a mis-oriented event is still an event. That single difference
is all of A3.

Why the scalar dilution factor is not a PDF quantity
----------------------------------------------------
The ratio of those two lines,

    D_eff(m) = sum_q (L_q^+ - L_q^-) D_q / sum_q (L_q^+ + L_q^-) D_q ,

is the number one would divide by to undo the dilution. It is often quoted as
though it were a property of the PDFs alone -- and for a **single** flavour it
is, collapsing to ``(L^+ - L^-)/(L^+ + L^-)``. With more than one flavour it
carries the per-flavour ``D_q``, and therefore **depends on**
``sin^2(theta_W)``: the very parameter A2 extracts from the unfolded curve. Up-
and down-type quarks have both different asymmetries *and* different valence
content, so the weighting does not cancel.

:func:`dilution_factor` takes ``sin2_theta_w`` for exactly this reason.
:func:`pdf_dilution` provides the flavour-blind PDF-only ratio that the naive
treatment uses; it is **not** the correct factor, and the difference between
the two is asserted to be resolvable in
``tests/analytic/test_dilution.py``. Prefer :func:`unfold_afb` fed by
:func:`dilution_factor`.

Where this degrades: ``D_eff -> 0``
-----------------------------------
At central rapidity ``x_1 -> x_2``, so ``L^+ -> L^-`` and ``D_eff -> 0``: the
proxy is a coin flip and the asymmetry is completely diluted away. Unfolding
divides by ``D_eff``, so bins there are not merely noisy but *undefined* -- no
amount of statistics recovers a signal that was multiplied by zero. Both
:func:`unfold_afb` and :func:`dilution_factor` mask such bins to ``nan`` via
``min_dilution`` rather than returning a large number that looks like a
measurement. A real analysis avoids the region by binning in rapidity and
dropping the central bins.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .electroweak import CHARGED_LEPTON, GAMMA_Z, M_Z, Fermion, _s_and_d

__all__ = [
    "MIN_DILUTION",
    "parton_x",
    "afb_diluted",
    "dilution_factor",
    "pdf_dilution",
    "unfold_afb",
]

#: Default floor on ``|D_eff|`` below which unfolding is refused (masked to
#: ``nan``). Not a physical constant -- a guard against dividing by a dilution
#: factor consistent with zero. See the module docstring.
MIN_DILUTION: float = 1e-3


def parton_x(
    mass: npt.NDArray[np.float64] | float,
    rapidity: npt.NDArray[np.float64] | float,
    sqrt_s: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""LO momentum fractions ``(x_1, x_2)`` for a pair of mass ``m``, rapidity ``y``.

    At leading order the pair takes all of both partons' momentum, so
    ``x_1 x_2 = m^2 / s`` and ``y = (1/2) ln(x_1 / x_2)``, giving

        x_1 = (m / sqrt(s)) e^{+y},    x_2 = (m / sqrt(s)) e^{-y}.

    ``x_1`` belongs to the beam travelling along ``+z``. Note ``x_1 > x_2``
    exactly when ``y > 0``, i.e. when ``Q_z > 0`` -- which is why the
    ``sign(Q_z)`` proxy is equivalent to "the quark came from the higher-``x``
    beam".
    """
    m = np.asarray(mass, dtype=float)
    y = np.asarray(rapidity, dtype=float)
    tau = m / sqrt_s
    return tau * np.exp(y), tau * np.exp(-y)


def _accumulate(
    mass: npt.NDArray[np.float64],
    sin2_theta_w: float,
    lum_aligned: dict[Fermion, npt.NDArray[np.float64] | float],
    lum_reversed: dict[Fermion, npt.NDArray[np.float64] | float],
    *,
    lepton: Fermion,
    m_z: float,
    width_z: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Return ``(sum (L+ - L-) D, sum (L+ + L-) D, sum (L+ + L-) S)`` per mass bin.

    The three sums every quantity in this module is built from. ``S``/``D`` come
    from A2's :func:`~accsim.events.electroweak._s_and_d`, so a sign error there
    would be caught by A2's own gate rather than silently absorbed here.
    """
    if set(lum_aligned) != set(lum_reversed):
        raise ValueError("lum_aligned and lum_reversed must cover the same flavours")
    if not lum_aligned:
        raise ValueError("need at least one quark flavour")

    d_diluted = np.zeros_like(mass)
    d_true = np.zeros_like(mass)
    s_total = np.zeros_like(mass)
    for quark in lum_aligned:
        l_plus = np.broadcast_to(np.asarray(lum_aligned[quark], dtype=float), mass.shape)
        l_minus = np.broadcast_to(np.asarray(lum_reversed[quark], dtype=float), mass.shape)
        s_q, d_q = _s_and_d(mass, quark, sin2_theta_w, lepton=lepton, m_z=m_z, width_z=width_z)
        d_diluted += (l_plus - l_minus) * d_q
        d_true += (l_plus + l_minus) * d_q
        s_total += (l_plus + l_minus) * s_q
    return d_diluted, d_true, s_total


def afb_diluted(
    mass: npt.NDArray[np.float64] | float,
    sin2_theta_w: float,
    lum_aligned: dict[Fermion, npt.NDArray[np.float64] | float],
    lum_reversed: dict[Fermion, npt.NDArray[np.float64] | float],
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
) -> npt.NDArray[np.float64]:
    r"""The ``A_FB(m)`` an experiment measures with the ``sign(Q_z)`` proxy.

    ``(3/4) sum_q (L_q^+ - L_q^-) D_q / sum_q (L_q^+ + L_q^-) S_q`` -- the master
    formula from the module docstring. Setting ``lum_reversed = 0`` (a perfectly
    oriented axis) reproduces :func:`accsim.events.afb_hadronic` exactly; setting
    ``lum_reversed = lum_aligned`` (no orientation information at all) gives zero.

    Parameters
    ----------
    lum_aligned, lum_reversed:
        Per-flavour parton luminosities for the proxy-correct and proxy-wrong
        orientations, as scalars or arrays broadcastable against ``mass``. Only
        the *relative* sizes matter; a common scale cancels. Supply these from
        LHAPDF (see :func:`parton_x`) for a real hadronic prediction.
    """
    m = np.atleast_1d(np.asarray(mass, dtype=float))
    d_diluted, _, s_total = _accumulate(
        m, sin2_theta_w, lum_aligned, lum_reversed, lepton=lepton, m_z=m_z, width_z=width_z
    )
    result = 0.75 * d_diluted / s_total
    return result if np.ndim(mass) else result[0]


def dilution_factor(
    mass: npt.NDArray[np.float64] | float,
    sin2_theta_w: float,
    lum_aligned: dict[Fermion, npt.NDArray[np.float64] | float],
    lum_reversed: dict[Fermion, npt.NDArray[np.float64] | float],
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
    min_dilution: float = MIN_DILUTION,
) -> npt.NDArray[np.float64]:
    r"""The factor ``D_eff(m)`` with ``A_FB^obs = D_eff * A_FB^true``.

    ``sum_q (L_q^+ - L_q^-) D_q / sum_q (L_q^+ + L_q^-) D_q``. **This depends on**
    ``sin2_theta_w`` whenever more than one flavour contributes -- see the module
    docstring; that is not an implementation wart but the physics, and it means
    an unfolded ``A_FB`` carries a mild dependence on the parameter A2 then fits
    from it. The dependence is weak (the ``D_q`` enter only as relative flavour
    weights) but it is not zero, so it should be propagated as a systematic or
    removed by iterating the fit.

    Bins with ``|D_eff| < min_dilution`` are returned as ``nan``: there the
    proxy carries no orientation information and the asymmetry is not
    recoverable at any statistics.
    """
    m = np.atleast_1d(np.asarray(mass, dtype=float))
    d_diluted, d_true, _ = _accumulate(
        m, sin2_theta_w, lum_aligned, lum_reversed, lepton=lepton, m_z=m_z, width_z=width_z
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = d_diluted / d_true
    result = np.where(np.abs(result) < min_dilution, np.nan, result)
    return result if np.ndim(mass) else result[0]


def pdf_dilution(
    lum_aligned: dict[Fermion, npt.NDArray[np.float64] | float],
    lum_reversed: dict[Fermion, npt.NDArray[np.float64] | float],
) -> npt.NDArray[np.float64]:
    r"""The flavour-blind PDF-only dilution ``sum_q (L^+ - L^-) / sum_q (L^+ + L^-)``.

    The quantity the naive treatment divides by. It is **exact for a single
    flavour** and wrong otherwise, because it weights the flavours by their
    luminosity instead of by their contribution ``L D_q`` to the asymmetry --
    and up- and down-type quarks differ in both. Provided for comparison and
    because it is what the literature usually plots; use
    :func:`dilution_factor` to actually unfold.
    """
    if set(lum_aligned) != set(lum_reversed):
        raise ValueError("lum_aligned and lum_reversed must cover the same flavours")
    if not lum_aligned:
        raise ValueError("need at least one quark flavour")
    num = sum(
        np.asarray(lum_aligned[q], dtype=float) - np.asarray(lum_reversed[q], dtype=float)
        for q in lum_aligned
    )
    den = sum(
        np.asarray(lum_aligned[q], dtype=float) + np.asarray(lum_reversed[q], dtype=float)
        for q in lum_aligned
    )
    return np.asarray(num / den, dtype=float)


def unfold_afb(
    afb_observed: npt.NDArray[np.float64],
    afb_error: npt.NDArray[np.float64],
    dilution: npt.NDArray[np.float64],
    *,
    min_dilution: float = MIN_DILUTION,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Undo the dilution: ``A_FB^true = A_FB^obs / D_eff``, error scaled likewise.

    The error is divided by ``|D_eff|`` -- unfolding inflates the uncertainty by
    exactly the factor it inflates the central value, which is the honest
    statement that dilution destroys information rather than merely rescaling
    it. ``D_eff`` is treated as an exact model input; its own uncertainty (PDFs,
    and the ``sin^2(theta_W)`` dependence noted in :func:`dilution_factor`) is a
    systematic to be propagated separately, not folded in here.

    Bins where ``|D_eff| < min_dilution`` or any input is non-finite come back as
    ``(nan, nan)``, so a fit downstream drops them (they fail
    :func:`accsim.events.fit_sin2_theta_w`'s ``sigma > 0`` filter).
    """
    obs = np.asarray(afb_observed, dtype=float)
    err = np.asarray(afb_error, dtype=float)
    d = np.asarray(dilution, dtype=float)
    if not (obs.shape == err.shape == d.shape):
        raise ValueError("afb_observed, afb_error and dilution must have the same shape")

    usable = np.isfinite(obs) & np.isfinite(err) & np.isfinite(d) & (np.abs(d) >= min_dilution)
    safe = np.where(usable, d, 1.0)
    true = np.where(usable, obs / safe, np.nan)
    true_err = np.where(usable, err / np.abs(safe), np.nan)
    return true, true_err
