"""Static plotting helpers (matplotlib).

Kept deliberately small: a phase-space scatter (Stage 0) and a beta-function /
beam-envelope plot (Stage 1). ``matplotlib`` is imported lazily so importing
:mod:`accsim` stays cheap and headless-safe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .coords import COORD_NAMES
from .tracking import Bunch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from .twiss import Twiss


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


def plot_beta_functions(
    twiss: Sequence[Twiss], ax: Axes | None = None, emittance: float | None = None
) -> Axes:
    r"""Plot ``beta_x(s)`` and ``beta_y(s)`` along the lattice from a Twiss table.

    ``twiss`` is the output of :func:`accsim.twiss.propagate_twiss`. If
    ``emittance`` (geometric, [m·rad]) is given, the right axis instead shows the
    1-sigma beam envelope ``sigma = sqrt(emittance * beta)`` [m] — the beam-size
    view of the same optics.
    """
    import matplotlib.pyplot as plt

    s = [t.s for t in twiss]
    bx = [t.beta_x for t in twiss]
    by = [t.beta_y for t in twiss]

    if ax is None:
        _, ax = plt.subplots()

    if emittance is None:
        ax.plot(s, bx, label=r"$\beta_x$")
        ax.plot(s, by, label=r"$\beta_y$")
        ax.set_ylabel(r"$\beta$ [m]")
    else:
        ax.plot(s, [math.sqrt(emittance * b) for b in bx], label=r"$\sigma_x$")
        ax.plot(s, [math.sqrt(emittance * b) for b in by], label=r"$\sigma_y$")
        ax.set_ylabel(r"$\sigma$ [m]")

    ax.set_xlabel("s [m]")
    ax.set_title("beta functions" if emittance is None else "beam envelope")
    ax.legend()
    return ax
