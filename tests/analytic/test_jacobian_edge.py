r"""Analytic gate for the Jacobian-edge locator (E1, the measuring device).

``jacobian_edge`` is what the E1 pipeline gates on, so it needs its own gate:
an estimator that merely *looked* stable would let a wrong pipeline result pass.

The samples here are drawn from the **exact CDF** ``1 - sqrt(1 - m_T^2/M^2)``
(inverted analytically), never from the module under test, so a bug in
``jacobian_peak_pdf`` cannot cancel a bug in the locator.

What is asserted, in order of strength:

1. **It tracks the mass.** The recovered edge minus the true mass is a *constant*
   across ``M = 60..100 GeV`` — the estimator measures the mass, not a fixed
   artifact of the shape. This is the property the pipeline actually relies on.
2. **It beats ``argmax``**, the naive alternative, on both accuracy and binning
   stability — asserted head-to-head rather than claimed in a docstring.
3. **The bias is the documented one** (``~ +1 GeV + 0.73 sigma``), pinned so a
   future change to the estimator cannot silently move it.
4. **``falloff_width`` is monotone in the smearing** — the truth-vs-reco
   "rounder edge" contrast in the pipeline depends on this and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim.events import jacobian_edge

M_W = 80.379


def _sample_ideal(mass: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``m_T`` from the exact CDF ``F = 1 - sqrt(1 - m_T^2/M^2)``.

    Inverting: ``m_T = M sqrt(1 - (1 - u)^2)`` for ``u ~ U(0,1)``. Written out
    here rather than imported, so the locator is tested against an independent
    construction of the distribution.
    """
    u = rng.uniform(0.0, 1.0, n)
    return mass * np.sqrt(1.0 - (1.0 - u) ** 2)


# --------------------------------------------------------------------------
# 1. it tracks the mass — the property the pipeline relies on
# --------------------------------------------------------------------------
def test_edge_tracks_the_mass_with_a_constant_offset() -> None:
    """Across a 40 GeV range of true masses the offset must stay constant.

    This is what separates a mass *measurement* from a shape artifact: if the
    locator returned, say, a fixed fraction of the window, the offsets would
    drift with ``M``.
    """
    rng = np.random.default_rng(20260720)
    offsets = []
    for mass in (60.0, 70.0, 80.379, 90.0, 100.0):
        m_t = _sample_ideal(mass, 600_000, rng) + rng.normal(0.0, 2.0, 600_000)
        edge, _ = jacobian_edge(m_t, window=(30.0, 160.0))
        offsets.append(edge - mass)
    offsets_arr = np.array(offsets)
    assert np.ptp(offsets_arr) < 0.25, f"offset drifts with mass: {offsets_arr}"
    assert np.all(np.abs(offsets_arr - 1.55) < 0.25), offsets_arr


def test_recovers_the_mass_within_the_gate_tolerance_at_w_width_smearing() -> None:
    """At smearing comparable to ``Gamma_W`` (~2.1 GeV) the edge lands within the
    few-GeV tolerance the pipeline gate uses. The tolerance is set by the measured
    bias above, **not** tuned until this passed."""
    rng = np.random.default_rng(4)
    m_t = _sample_ideal(M_W, 600_000, rng) + rng.normal(0.0, 2.1, 600_000)
    edge, _ = jacobian_edge(m_t)
    assert abs(edge - M_W) < 3.0, edge


# --------------------------------------------------------------------------
# 2. head-to-head against the naive alternative
# --------------------------------------------------------------------------
def test_beats_argmax_on_accuracy_and_binning_stability() -> None:
    """``argmax`` is the obvious thing to reach for and it is worse on both counts.

    Asserted rather than asserted-in-prose: the peak bin sits *below* the true
    mass (the divergence is binned away) and moves with the binning.
    """
    rng = np.random.default_rng(9)
    m_t = _sample_ideal(M_W, 600_000, rng) + rng.normal(0.0, 2.0, 600_000)

    edges_hm, edges_am = [], []
    for nbins in (30, 40, 60, 80, 120):
        edge, _ = jacobian_edge(m_t, bins=nbins)
        edges_hm.append(edge)
        counts, bin_edges = np.histogram(m_t, bins=nbins, range=(40.0, 140.0))
        centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        edges_am.append(centres[int(np.argmax(counts))])

    hm, am = np.array(edges_hm), np.array(edges_am)
    # binning stability: half-max spread is far tighter
    assert np.ptp(hm) < 0.5, hm
    assert np.ptp(hm) < np.ptp(am), (hm, am)
    # accuracy: argmax undershoots the mass, half-max brackets it much closer
    assert np.all(am < M_W - 0.5), am
    assert np.max(np.abs(hm - M_W)) < np.max(np.abs(am - M_W)), (hm, am)


# --------------------------------------------------------------------------
# 3. the documented bias, pinned
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("sigma", "expected_bias"),
    [(0.0, 1.03), (1.0, 0.93), (2.0, 1.49), (3.0, 2.18), (5.0, 3.61), (8.0, 5.90)],
)
def test_bias_matches_the_documented_table(sigma: float, expected_bias: float) -> None:
    """The bias table in the docstring is a measurement, so it is pinned as one.

    Not a physics constant — a property of this estimator. If the estimator
    changes, this table must be re-measured and the docstring updated with it,
    which is exactly the intent.
    """
    rng = np.random.default_rng(1)
    m_t = _sample_ideal(M_W, 600_000, rng)
    if sigma > 0:
        m_t = m_t + rng.normal(0.0, sigma, m_t.size)
    edge, _ = jacobian_edge(m_t)
    assert edge - M_W == pytest.approx(expected_bias, abs=0.35)


# --------------------------------------------------------------------------
# 4. falloff_width — the truth-vs-reco contrast in the pipeline
# --------------------------------------------------------------------------
def test_falloff_width_increases_monotonically_with_smearing() -> None:
    """The pipeline's "reco edge is rounder than truth" gate rests entirely on
    this being monotone in the smearing width."""
    rng = np.random.default_rng(2)
    base = _sample_ideal(M_W, 600_000, rng)
    widths = []
    for sigma in (0.0, 2.0, 5.0, 10.0, 15.0):
        m_t = base if sigma == 0 else base + rng.normal(0.0, sigma, base.size)
        _, width = jacobian_edge(m_t, window=(30.0, 200.0))
        widths.append(width)
    w = np.array(widths)
    assert np.all(np.diff(w) > 0), w
    assert w[-1] > 3.0 * w[0], w


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def test_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError):
        jacobian_edge(np.array([1.0, 2.0, 3.0]), window=(40.0, 140.0))


def test_non_finite_samples_are_dropped_not_propagated() -> None:
    rng = np.random.default_rng(6)
    m_t = _sample_ideal(M_W, 100_000, rng)
    polluted = np.concatenate([m_t, np.full(500, np.nan), np.full(500, np.inf)])
    clean_edge, _ = jacobian_edge(m_t)
    dirty_edge, _ = jacobian_edge(polluted)
    assert dirty_edge == pytest.approx(clean_edge, abs=1e-12)


def test_edge_out_of_window_degrades_gracefully() -> None:
    """A window that ends before the edge must return the window bound, not crash
    and not silently report a spurious interior edge."""
    rng = np.random.default_rng(8)
    m_t = _sample_ideal(M_W, 100_000, rng)
    edge, _ = jacobian_edge(m_t, window=(40.0, 60.0))
    assert 55.0 < edge <= 60.0
