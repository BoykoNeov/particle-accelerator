r"""Neutral-current Drell-Yan: the LO ``gamma*/Z`` angular structure and the
extraction of ``sin^2(theta_W)`` from ``A_FB(m)`` (milestone A2).

This is how LEP and the LHC actually measure the weak mixing angle: the
forward-backward asymmetry of ``q qbar -> gamma*/Z -> l- l+`` is driven by the
*leptonic vector coupling* ``g_V^l = T3_l - 2 Q_l sin^2(theta_W) = -1/2 + 2
sin^2(theta_W)``, which sits close to zero for the physical value. A small change
in ``sin^2(theta_W)`` therefore moves ``g_V^l`` by a large *relative* amount, and
``A_FB`` inherits that amplified sensitivity.

**Natural units** (``hbar = c = 1``, GeV), as everywhere in :mod:`accsim.events`.

Derivation, not recollection
----------------------------
The angular structure below is *derived* symbolically (explicit Dirac gamma
matrices, mostly-minus metric, massless fermions) in
``tests/analytic/test_electroweak_afb.py`` -- not transcribed from a textbook.
For a mediator pair ``(V, V')`` the spin-summed squared amplitude is

    |M|^2_{VV'} ~ 4 s^2 [ (1 + cos^2 t) (v_l v_l' + a_l a_l')(v_q v_q' + a_q a_q')
                          + 2 cos t     (a_l v_l' + a_l' v_l)(a_q v_q' + a_q' v_q) ]

so that, summing over mediator pairs with complex propagator factors ``P_V``,

    S(s) = sum_{V,V'} Re[P_V P_V'^*] (v_l v_l' + a_l a_l') (v_q v_q' + a_q a_q')
    D(s) = sum_{V,V'} Re[P_V P_V'^*] (a_l v_l' + a_l' v_l) (a_q v_q' + a_q' v_q)

    dsigma/dcos(t) ~ S (1 + cos^2 t) + 2 D cos(t)
    A_FB = (3/4) D / S          and        A_4 = 2 D / S

The second identity reproduces the ``A_FB = (3/8) A_4`` anchor that
:func:`accsim.events.angular_coefficients` is pinned against, by construction.

Mediators and the role of ``sin^2(theta_W)``
--------------------------------------------
Two mediators, with the common ``e^2`` stripped (it cancels in the ratio
``D/S``):

    photon:  v = Q_f, a = 0,        P_gamma = 1 / s
    Z:       v = g_V^f, a = g_A^f,  P_Z = kappa / (s - M_Z^2 + i M_Z Gamma_Z)

with ``kappa = 1 / (4 sin^2(theta_W) cos^2(theta_W))`` from ``g^2/(4 cos^2) =
e^2/(4 sin^2 cos^2)``, and ``cos^2 = 1 - sin^2``.

**Which angle this recovers.** The fitted parameter enters the couplings, so what
``A_FB`` is sensitive to is the *effective* leptonic mixing angle (Pythia's
``StandardModel:sin2thetaWbar``), not the on-shell one. This model lets the single
parameter float in ``kappa`` as well, which is a tree-level simplification: the
sensitivity is overwhelmingly through ``g_V^l``, and ``kappa`` only reweights
``gamma`` vs ``Z``. See ``docs/CONVENTIONS.md`` -> *sin^2(theta_W) from A_FB(m)*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

__all__ = [
    "M_Z",
    "GAMMA_Z",
    "UP_TYPE",
    "DOWN_TYPE",
    "CHARGED_LEPTON",
    "Fermion",
    "neutral_current_couplings",
    "afb_parton",
    "afb_hadronic",
    "Sin2ThetaWFit",
    "fit_sin2_theta_w",
]

# PDG Z-boson pole parameters (GeV). Defaults only -- every entry point takes
# them as arguments so a generator's own configured values can be used instead.
M_Z: float = 91.1876
GAMMA_Z: float = 2.4952


@dataclass(frozen=True)
class Fermion:
    """A fermion species, identified by what the neutral current cares about.

    Attributes
    ----------
    charge:
        Electric charge ``Q_f`` in units of ``e``.
    t3:
        Third component of weak isospin ``T3_f`` for the left-handed field.
    """

    charge: float
    t3: float


UP_TYPE = Fermion(charge=2.0 / 3.0, t3=+0.5)  # u, c, t
DOWN_TYPE = Fermion(charge=-1.0 / 3.0, t3=-0.5)  # d, s, b
CHARGED_LEPTON = Fermion(charge=-1.0, t3=-0.5)  # e, mu, tau


def neutral_current_couplings(
    fermion: Fermion, sin2_theta_w: float | npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Return the Z vector/axial couplings ``(g_V, g_A)`` of ``fermion``.

    ``g_V = T3 - 2 Q sin^2(theta_W)`` and ``g_A = T3``. Note ``g_A`` carries no
    ``sin^2(theta_W)`` dependence at all -- the entire sensitivity of ``A_FB`` to
    the mixing angle flows through ``g_V``, and for a charged lepton
    ``g_V = -1/2 + 2 sin^2(theta_W)`` is near zero, which is precisely why the
    measurement is sharp.
    """
    s2w = np.asarray(sin2_theta_w, dtype=float)
    g_v = fermion.t3 - 2.0 * fermion.charge * s2w
    g_a = np.full_like(g_v, fermion.t3)
    return g_v, g_a


def _s_and_d(
    mass: npt.NDArray[np.float64],
    quark: Fermion,
    sin2_theta_w: float,
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""The symmetric (``S``) and antisymmetric (``D``) angular strengths.

    Implemented as the literal double sum over mediator pairs from the module
    docstring -- deliberately *not* hand-expanded into
    ``gamma-gamma + 2 Re(gamma-Z) + Z-Z``, so there is no opportunity to drop or
    mis-sign an interference term.
    """
    s = np.asarray(mass, dtype=float) ** 2
    cos2_theta_w = 1.0 - sin2_theta_w

    gv_q, ga_q = neutral_current_couplings(quark, sin2_theta_w)
    gv_l, ga_l = neutral_current_couplings(lepton, sin2_theta_w)

    kappa = 1.0 / (4.0 * sin2_theta_w * cos2_theta_w)
    prop_gamma = np.ones_like(s, dtype=complex) / s
    prop_z = kappa / (s - m_z**2 + 1j * m_z * width_z)

    # (propagator, v_q, a_q, v_l, a_l) per mediator.
    mediators = [
        (
            prop_gamma,
            np.full_like(s, quark.charge),
            np.zeros_like(s),
            np.full_like(s, lepton.charge),
            np.zeros_like(s),
        ),
        (
            prop_z,
            np.broadcast_to(gv_q, s.shape),
            np.broadcast_to(ga_q, s.shape),
            np.broadcast_to(gv_l, s.shape),
            np.broadcast_to(ga_l, s.shape),
        ),
    ]

    s_tot = np.zeros_like(s)
    d_tot = np.zeros_like(s)
    for p1, vq1, aq1, vl1, al1 in mediators:
        for p2, vq2, aq2, vl2, al2 in mediators:
            weight = np.real(p1 * np.conj(p2))
            s_tot += weight * (vl1 * vl2 + al1 * al2) * (vq1 * vq2 + aq1 * aq2)
            d_tot += weight * (al1 * vl2 + al2 * vl1) * (aq1 * vq2 + aq2 * vq1)
    return s_tot, d_tot


def afb_parton(
    mass: npt.NDArray[np.float64] | float,
    quark: Fermion,
    sin2_theta_w: float,
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
) -> npt.NDArray[np.float64]:
    r"""Parton-level ``A_FB(m)`` for a single ``q qbar -> gamma*/Z -> l- l+`` flavour.

    This is the *undiluted* asymmetry: it assumes the quark direction is known, so
    ``cos(theta)`` is measured against the true quark axis. In ``pp`` the direction
    is only inferred, which dilutes the observable -- that is milestone A3's
    problem, deliberately kept out of this model.

    Returns ``(3/4) D / S``, with ``S``/``D`` as in the module docstring.
    """
    m = np.atleast_1d(np.asarray(mass, dtype=float))
    s_tot, d_tot = _s_and_d(m, quark, sin2_theta_w, lepton=lepton, m_z=m_z, width_z=width_z)
    result = 0.75 * d_tot / s_tot
    return result if np.ndim(mass) else result[0]


def afb_hadronic(
    mass: npt.NDArray[np.float64] | float,
    sin2_theta_w: float,
    flavour_weights: dict[Fermion, npt.NDArray[np.float64] | float],
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
) -> npt.NDArray[np.float64]:
    r"""Luminosity-weighted ``A_FB(m)`` summed over quark flavours.

    In a hadron collider the observable is a sum over the contributing ``q qbar``
    initial states, each weighted by its **parton luminosity** ``L_q(m)``. Up- and
    down-type quarks have different asymmetries and their mix shifts with ``m``
    through the PDFs, so the flavour sum is not a detail -- it moves the curve.

    The weights are combined at the level of ``S`` and ``D`` (*not* by averaging
    per-flavour ``A_FB`` values, which would be wrong -- ``A_FB`` is a ratio):

        A_FB(m) = (3/4) * sum_q L_q(m) D_q(m) / sum_q L_q(m) S_q(m)

    Parameters
    ----------
    flavour_weights:
        Maps a quark :class:`Fermion` (typically :data:`UP_TYPE` /
        :data:`DOWN_TYPE`) to its parton luminosity -- a scalar, or an array
        broadcastable against ``mass``. Only relative sizes matter; an overall
        scale cancels. Supply these from LHAPDF for a real hadronic prediction.
    """
    m = np.atleast_1d(np.asarray(mass, dtype=float))
    if not flavour_weights:
        raise ValueError("flavour_weights must contain at least one quark flavour")

    s_sum = np.zeros_like(m)
    d_sum = np.zeros_like(m)
    for quark, weight in flavour_weights.items():
        w = np.broadcast_to(np.asarray(weight, dtype=float), m.shape)
        s_tot, d_tot = _s_and_d(m, quark, sin2_theta_w, lepton=lepton, m_z=m_z, width_z=width_z)
        s_sum += w * s_tot
        d_sum += w * d_tot

    result = 0.75 * d_sum / s_sum
    return result if np.ndim(mass) else result[0]


@dataclass(frozen=True)
class Sin2ThetaWFit:
    """Outcome of a ``sin^2(theta_W)`` fit to a binned ``A_FB(m)`` measurement.

    Attributes
    ----------
    sin2_theta_w:
        Best-fit effective leptonic mixing angle.
    error:
        One-sigma uncertainty, from the ``chi^2 = chi^2_min + 1`` curvature
        (the Jacobian-based covariance of the least-squares solution).
    chi2:
        Minimum ``chi^2``.
    ndof:
        Degrees of freedom (bins minus one fitted parameter).
    n_bins:
        Number of mass bins that entered the fit.
    """

    sin2_theta_w: float
    error: float
    chi2: float
    ndof: int
    n_bins: int

    @property
    def chi2_per_dof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else float("nan")


def fit_sin2_theta_w(
    mass: npt.NDArray[np.float64],
    afb: npt.NDArray[np.float64],
    afb_error: npt.NDArray[np.float64],
    flavour_weights: dict[Fermion, npt.NDArray[np.float64] | float],
    *,
    lepton: Fermion = CHARGED_LEPTON,
    m_z: float = M_Z,
    width_z: float = GAMMA_Z,
    initial: float = 0.23,
    bounds: tuple[float, float] = (0.05, 0.45),
) -> Sin2ThetaWFit:
    r"""Fit binned ``A_FB(m)`` for the effective leptonic mixing angle.

    Minimises ``chi^2 = sum_i [(A_FB^obs_i - A_FB^model(m_i; s2w)) / sigma_i]^2``
    over the single parameter ``sin^2(theta_W)``.

    Bins with a non-finite or non-positive error are dropped (an empty or
    single-entry mass bin yields ``nan`` from
    :func:`accsim.events.forward_backward_asymmetry`).

    Parameters
    ----------
    mass:
        Bin centres in GeV. Use the *undiluted* ``A_FB``; see :func:`afb_parton`.
    afb, afb_error:
        Measured asymmetry per bin and its one-sigma error.
    flavour_weights:
        As for :func:`afb_hadronic`; arrays are indexed per mass bin.
    """
    m = np.asarray(mass, dtype=float)
    y = np.asarray(afb, dtype=float)
    sigma = np.asarray(afb_error, dtype=float)
    if not (m.shape == y.shape == sigma.shape):
        raise ValueError("mass, afb and afb_error must have the same shape")

    good = np.isfinite(m) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    weights = {
        q: np.broadcast_to(np.asarray(w, dtype=float), m.shape)[good]
        for q, w in flavour_weights.items()
    }
    m, y, sigma = m[good], y[good], sigma[good]
    if m.size < 2:
        raise ValueError(f"need at least 2 usable bins to fit, got {m.size}")

    def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        model = afb_hadronic(m, float(params[0]), weights, lepton=lepton, m_z=m_z, width_z=width_z)
        return (y - model) / sigma

    result = least_squares(residuals, x0=[initial], bounds=([bounds[0]], [bounds[1]]), method="trf")
    if not result.success:
        raise RuntimeError(f"sin^2(theta_W) fit did not converge: {result.message}")

    # `least_squares` reports success when it converges *onto a bound*, where the
    # returned value is the edge of the search window rather than a minimum. Left
    # unchecked that hands back a bound as if it were a measurement (observed for
    # a far-off starting guess). A pinned solution is a failed fit, not a result.
    best = float(result.x[0])
    edge = 1e-6 * (bounds[1] - bounds[0])
    if best <= bounds[0] + edge or best >= bounds[1] - edge:
        raise RuntimeError(
            f"sin^2(theta_W) fit converged onto the bound {best:.6g} "
            f"(search window {bounds}); the minimum is outside the window or the "
            f"starting guess {initial:.6g} is in a different basin"
        )

    chi2 = float(2.0 * result.cost)  # least_squares cost is 0.5 * sum(res^2)
    ndof = m.size - 1
    # Covariance from the Gauss-Newton approximation: (J^T J)^-1, with residuals
    # already scaled by sigma so no extra chi2/ndof inflation is applied.
    jtj = float(result.jac[:, 0] @ result.jac[:, 0])
    error = float(np.sqrt(1.0 / jtj)) if jtj > 0.0 else float("nan")

    return Sin2ThetaWFit(
        sin2_theta_w=best,
        error=error,
        chi2=chi2,
        ndof=ndof,
        n_bins=int(m.size),
    )
