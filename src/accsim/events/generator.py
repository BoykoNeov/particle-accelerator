r"""Toy 2->2 Monte-Carlo event generator (Phase 2 learning module).

Ties the three pieces together for ``e+ e- -> mu+ mu-``:

  matrix element (:mod:`.matrix_element`) x flat phase space (:mod:`.phase_space`,
  RAMBO) -> Monte-Carlo estimate of the total cross-section, with events and a
  labelled angular distribution.

**This is a clearly-labelled learning module**, not a physics-grade generator: tree
level, massless, fixed coupling, single process. The roadmap's *orchestrate, don't
rebuild* rule stands — for real physics use Pythia/MadGraph. What this earns is the
Phase 2 acceptance gate: *the toy total cross-section matches the analytic value
within Monte-Carlo error.*

**Cross-section master formula.** For a 2->2 process with incoming momenta ``p1,
p2`` and flux factor ``F = 4 sqrt((p1.p2)^2 - m1^2 m2^2) = 2 s`` (massless),

    sigma = (1 / 2s) integral <|M|^2> dPhi_2  ~=  (weight / 2s) * <|M|^2>,

since RAMBO samples ``dPhi_2`` flat with constant ``weight`` (= ``1/8pi``). The
result is in ``GeV^-2``; :func:`gev2_to_barn` converts to barns via ``(hbar c)^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .kinematics import mandelstam_s
from .matrix_element import ALPHA_EM, EEtoMuMu
from .phase_space import rambo

__all__ = [
    "GEV2_TO_MBARN",
    "gev2_to_barn",
    "CrossSectionEstimate",
    "AngularDistribution",
    "ee_to_mumu_cross_section",
    "ee_to_mumu_events",
]

# (hbar c)^2 = 0.3893793721 GeV^2 . mbarn (PDG). This is *the* natural-units ->
# laboratory-units bridge for cross-sections; kept as a single tested constant so
# the 0.389 factor is never sprinkled inline. 1 barn = 1e-28 m^2 = 100 fm^2.
GEV2_TO_MBARN: float = 0.3893793721


def gev2_to_barn(sigma_gev2: float) -> float:
    """Convert a cross-section from ``GeV^-2`` to **barns** via ``(hbar c)^2``.

    ``sigma[mbarn] = sigma[GeV^-2] * 0.3893793721``; barns = mbarn * 1e-3.
    """
    return sigma_gev2 * GEV2_TO_MBARN * 1.0e-3


@dataclass(frozen=True)
class CrossSectionEstimate:
    """Monte-Carlo total cross-section with its statistical error (both ``GeV^-2``)."""

    value: float
    error: float
    n_events: int

    @property
    def value_nb(self) -> float:
        """Central value in nanobarns (``1 nb = 1e-9 barn``)."""
        return gev2_to_barn(self.value) / 1.0e-9

    @property
    def error_nb(self) -> float:
        """Statistical error in nanobarns."""
        return gev2_to_barn(self.error) / 1.0e-9


@dataclass(frozen=True)
class AngularDistribution:
    """A labelled ``cos(theta)`` histogram of the outgoing ``mu-`` — the Phase 2
    end-to-end deliverable (a labelled distribution).

    ``bin_edges`` has length ``len(counts) + 1``; ``counts`` are raw event counts.
    """

    bin_edges: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    label: str


def _incoming_momenta(sqrt_s: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Massless ``e-``/``e+`` beams collinear on ``+z``/``-z`` in the CM frame."""
    half = 0.5 * sqrt_s
    p1 = np.array([half, 0.0, 0.0, half])  # e- along +z
    p2 = np.array([half, 0.0, 0.0, -half])  # e+ along -z
    return p1, p2


def ee_to_mumu_cross_section(
    sqrt_s: float,
    n_events: int,
    rng: np.random.Generator,
    *,
    alpha: float = ALPHA_EM,
) -> CrossSectionEstimate:
    r"""Monte-Carlo estimate of ``sigma(e+ e- -> mu+ mu-)`` at ``sqrt_s`` [GeV].

    Samples ``n_events`` flat 2-body phase-space points (RAMBO), evaluates the
    tree-level ``<|M|^2>``, and forms ``sigma = (weight/2s) <|M|^2>`` with the MC
    error ``= (weight/2s) * std(|M|^2)/sqrt(N)``.
    """
    if sqrt_s <= 0:
        raise ValueError(f"sqrt_s must be positive, got {sqrt_s}")
    me = EEtoMuMu(alpha=alpha)
    p1, p2 = _incoming_momenta(sqrt_s)
    s = float(mandelstam_s(p1, p2))

    batch = rambo(2, sqrt_s, n_events, rng)
    p3 = batch.momenta[:, 0, :]  # mu-
    p4 = batch.momenta[:, 1, :]  # mu+
    m2 = np.asarray(me.squared_amplitude(p1, p2, p3, p4))

    prefactor = batch.weight / (2.0 * s)
    value = prefactor * float(m2.mean())
    error = prefactor * float(m2.std(ddof=1)) / math.sqrt(n_events)
    return CrossSectionEstimate(value=value, error=error, n_events=n_events)


def ee_to_mumu_events(
    sqrt_s: float,
    n_events: int,
    rng: np.random.Generator,
    *,
    n_bins: int = 20,
) -> tuple[npt.NDArray[np.float64], AngularDistribution]:
    r"""Generate unweighted-*ish* events and the labelled ``cos(theta)`` distribution.

    Returns ``(cos_theta, AngularDistribution)`` where ``cos_theta`` is the scattering
    angle of the outgoing ``mu-`` relative to the ``e-`` beam (``+z``) for every RAMBO
    event. Because RAMBO is *flat* in phase space, the physical ``1 + cos^2 th`` shape
    is recovered by accept-reject against ``<|M|^2>`` — the histogram is the visible
    end-to-end deliverable.
    """
    me = EEtoMuMu()
    p1, p2 = _incoming_momenta(sqrt_s)
    batch = rambo(2, sqrt_s, n_events, rng)
    p3 = batch.momenta[:, 0, :]  # mu-
    p4 = batch.momenta[:, 1, :]  # mu+

    # cos(theta) of mu- vs the +z (e-) beam.
    cos_theta = p3[:, 3] / np.sqrt(p3[:, 1] ** 2 + p3[:, 2] ** 2 + p3[:, 3] ** 2)

    # Accept-reject to turn the flat sample into physically-distributed events.
    m2 = np.asarray(me.squared_amplitude(p1, p2, p3, p4))
    accept = rng.random(n_events) < (m2 / m2.max())
    cos_accepted = cos_theta[accept]

    counts, edges = np.histogram(cos_accepted, bins=n_bins, range=(-1.0, 1.0))
    dist = AngularDistribution(
        bin_edges=edges,
        counts=counts.astype(np.int64),
        label=r"$e^+e^- \to \mu^+\mu^-$: $\cos\theta_{\mu^-}$ (expect $1+\cos^2\theta$)",
    )
    return cos_accepted, dist
