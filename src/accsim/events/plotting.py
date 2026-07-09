r"""Rendering for the toy generator's labelled distributions (Phase 2).

Kept inside :mod:`accsim.events` (rather than the core :mod:`accsim.plotting`) so
the event-physics learning module stays self-contained — the core plotter has no
reason to import event kinematics. Same pattern as ``plotting.plot_beam_envelope``:
lazy ``matplotlib`` import, optional ``ax``, returns the ``Axes``.

This closes the *literal* Phase 2 deliverable — "the pipeline produces a labelled
distribution" — by actually rendering the ``cos theta`` histogram, not just holding
its counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .generator import AngularDistribution

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

__all__ = ["plot_angular_distribution"]


def plot_angular_distribution(dist: AngularDistribution, ax: Axes | None = None) -> Axes:
    r"""Render an :class:`AngularDistribution` as a labelled ``cos theta`` histogram.

    Overlays the physical ``1 + cos^2 theta`` shape (area-matched to the counts) so
    the ``e+ e- -> mu+ mu-`` angular law is visible against the Monte-Carlo events.
    Returns the ``Axes`` for further composition.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots()

    edges = dist.bin_edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, dist.counts, width=widths, alpha=0.7, label="MC events", align="center")

    # Area-matched 1 + cos^2(theta) overlay: integral over [-1, 1] is 8/3.
    total = dist.counts.sum()
    bin_width = float(widths.mean())
    shape = (1.0 + centers**2) * total * bin_width / (8.0 / 3.0)
    ax.plot(centers, shape, "r-", lw=2, label=r"$1 + \cos^2\theta$")

    ax.set_xlabel(r"$\cos\theta_{\mu^-}$")
    ax.set_ylabel("events / bin")
    ax.set_title(dist.label)
    ax.legend()
    return ax
