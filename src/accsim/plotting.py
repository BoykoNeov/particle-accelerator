"""Static plotting helpers (matplotlib).

Kept deliberately small for Stage 0 — only a phase-space scatter, which is the
visualisation the test harness needs now. Twiss / beam-envelope plotting is
added in Stage 1+ alongside the optics it depicts. ``matplotlib`` is imported
lazily so importing :mod:`accsim` stays cheap and headless-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .coords import COORD_NAMES
from .tracking import Bunch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


def plot_phase_space(bunch: Bunch, plane: str = "x", ax: Axes | None = None) -> Axes:
    """Scatter a bunch in one transverse phase plane.

    ``plane`` is ``"x"`` (plots ``x`` vs ``px``) or ``"y"`` (``y`` vs ``py``).
    """
    import matplotlib.pyplot as plt

    pos, mom = {"x": ("x", "px"), "y": ("y", "py")}.get(plane, (None, None))
    if pos is None:
        raise ValueError(f"plane must be 'x' or 'y', got {plane!r}")
    i_pos = COORD_NAMES.index(pos)
    i_mom = COORD_NAMES.index(mom)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(bunch.states[i_pos], bunch.states[i_mom], s=4)
    ax.set_xlabel(f"{pos} [m]")
    ax.set_ylabel(f"{mom} [rad]")
    ax.set_title(f"{plane}-plane phase space")
    return ax
