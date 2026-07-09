r"""Matrix element for ``e+ e- -> mu+ mu-`` (QED, s-channel photon).

At lowest order (tree level) the process ``e+(p2) e-(p1) -> mu-(p3) mu+(p4)``
proceeds through a single virtual photon. In the **massless** limit
(``sqrt(s) >> m_e, m_mu``) the spin-**averaged**, spin-**summed** squared amplitude
is (Peskin & Schroeder, *An Introduction to QFT*, eq. 5.10, massless limit):

    <|M|^2> = 2 e^4 (t^2 + u^2) / s^2 = 32 pi^2 alpha^2 (t^2 + u^2) / s^2,

using ``e^2 = 4 pi alpha`` so ``e^4 = 16 pi^2 alpha^2``. The 1/4 spin average over
the four initial e+/e- helicity combinations is already included.

Expressed through the scattering angle (``t = -(s/2)(1 - cos th)``,
``u = -(s/2)(1 + cos th)`` massless, so ``t^2 + u^2 = (s^2/2)(1 + cos^2 th)``):

    <|M|^2> = 16 pi^2 alpha^2 (1 + cos^2 th),

which drives the classic ``1 + cos^2 th`` angular distribution. The differential
and total cross-sections that this integrates to are provided as **analytic**
closed forms (used as acceptance gates, not as the MC result):

    dsigma/dOmega = alpha^2 (1 + cos^2 th) / (4 s),   sigma = 4 pi alpha^2 / (3 s).

See ``docs/CONVENTIONS.md`` -> *Toy event generator*.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .kinematics import mandelstam_s, mandelstam_t, mandelstam_u

__all__ = ["ALPHA_EM", "EEtoMuMu"]

# Fine-structure constant (CODATA 2018, Thomson limit). The running of alpha to
# the collision scale is a real effect but out of scope for the tree-level toy.
ALPHA_EM: float = 1.0 / 137.035999084


class EEtoMuMu:
    r"""Tree-level QED matrix element for ``e+ e- -> mu+ mu-`` (massless limit).

    ``alpha`` defaults to the Thomson-limit :data:`ALPHA_EM`; pass an effective
    value to model (crudely) the running coupling at the collision scale.
    """

    def __init__(self, alpha: float = ALPHA_EM) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def squared_amplitude(
        self,
        p_in1: npt.NDArray[np.float64],
        p_in2: npt.NDArray[np.float64],
        p_out1: npt.NDArray[np.float64],
        p_out2: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64] | float:
        r"""``<|M|^2> = 32 pi^2 alpha^2 (t^2 + u^2)/s^2`` from the four momenta.

        Computed from Lorentz invariants (Mandelstam ``s, t, u``) so it is frame
        independent and works batched (leading axis = events). ``p_in1`` is the
        electron, ``p_out1`` the ``mu-``; the ``t``/``u`` assignment follows.
        """
        s = mandelstam_s(p_in1, p_in2)
        t = mandelstam_t(p_in1, p_out1)
        u = mandelstam_u(p_in1, p_out2)
        return 32.0 * math.pi**2 * self.alpha**2 * (t * t + u * u) / (s * s)

    def dsigma_domega(self, s: float, cos_theta: float) -> float:
        r"""Analytic ``dsigma/dOmega = alpha^2 (1 + cos^2 th)/(4 s)`` [GeV^-2 sr^-1]."""
        if s <= 0:
            raise ValueError(f"s must be positive, got {s}")
        return self.alpha**2 * (1.0 + cos_theta * cos_theta) / (4.0 * s)

    def total_cross_section(self, s: float) -> float:
        r"""Analytic total ``sigma = 4 pi alpha^2 / (3 s)`` [GeV^-2]."""
        if s <= 0:
            raise ValueError(f"s must be positive, got {s}")
        return 4.0 * math.pi * self.alpha**2 / (3.0 * s)
