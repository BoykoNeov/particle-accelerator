"""Smoke tests for the plotting helpers — they should run headless without error.

Physics is checked elsewhere; here we only guard against the plotting code
rotting (wrong attribute names, signature drift). Skipped if matplotlib is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Drift, Lattice, Quadrupole, ReferenceParticle, closed_twiss, propagate_twiss


@pytest.fixture(autouse=True)
def _agg_backend() -> None:
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")  # headless, no display needed


def test_plot_beta_functions_runs() -> None:
    from accsim.plotting import plot_beta_functions

    ref = ReferenceParticle.from_gamma(938.27208816e6, 20.0)
    lat = Lattice([Quadrupole(0.3, 1.2), Drift(1.0), Quadrupole(0.3, -1.2), Drift(1.0)], ref)
    pts = propagate_twiss(lat, closed_twiss(lat))

    ax = plot_beta_functions(pts)
    assert len(ax.lines) == 2  # beta_x and beta_y
    ax_env = plot_beta_functions(pts, emittance=1e-9)
    assert ax_env.get_ylabel().startswith("$\\sigma$")


def test_plot_phase_space_runs() -> None:
    from accsim import Bunch
    from accsim.plotting import plot_phase_space

    bunch = Bunch(np.random.default_rng(0).normal(scale=1e-3, size=(6, 50)))
    ax = plot_phase_space(bunch, plane="x")
    assert ax.get_xlabel() == "x [m]"
