"""Static plotting helpers (matplotlib).

Kept deliberately small: a phase-space scatter (Stage 0) and a beta-function /
beam-envelope plot (Stage 1). ``matplotlib`` is imported lazily so importing
:mod:`accsim` stays cheap and headless-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .coords import COORD_NAMES
from .tracking import Bunch
from .twiss import beam_sigma

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
    ``emittance`` (geometric, [m·rad]) is given, plot the 1-sigma **betatron**
    envelope ``sigma = sqrt(emittance * beta)`` [m] instead — the beam-size view
    of the same optics, with no momentum-spread term. For the full envelope
    (including dispersion) use :func:`plot_beam_envelope`.
    """
    import matplotlib.pyplot as plt

    s = [t.s for t in twiss]

    if ax is None:
        _, ax = plt.subplots()

    if emittance is None:
        ax.plot(s, [t.beta_x for t in twiss], label=r"$\beta_x$")
        ax.plot(s, [t.beta_y for t in twiss], label=r"$\beta_y$")
        ax.set_ylabel(r"$\beta$ [m]")
        ax.set_title("beta functions")
    else:
        # Delegate to the single envelope formula (sigma_delta = 0 => betatron only)
        # so there is never a second, divergent sigma expression in the codebase.
        sx, sy = beam_sigma(twiss, emittance)
        ax.plot(s, sx, label=r"$\sigma_x$")
        ax.plot(s, sy, label=r"$\sigma_y$")
        ax.set_ylabel(r"$\sigma$ [m]")
        ax.set_title("beam envelope (betatron)")

    ax.set_xlabel("s [m]")
    ax.legend()
    return ax


def plot_beam_envelope(
    twiss: Sequence[Twiss],
    emit_x: float,
    emit_y: float | None = None,
    sigma_delta: float = 0.0,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the full 1-sigma beam envelope ``sigma_x(s)``, ``sigma_y(s)``.

    Thin wrapper over :func:`accsim.twiss.beam_sigma` (where the physics lives and
    is tested): each plane adds the betatron width and the momentum-spread offset
    in quadrature,

        sigma_u(s) = sqrt(emit_u * beta_u(s) + (D_u(s) * sigma_delta)^2).

    ``emit_x`` / ``emit_y`` are geometric emittances [m·rad] (``emit_y`` defaults to
    ``emit_x``); ``sigma_delta`` is the RMS relative momentum spread (dimensionless).
    With ``sigma_delta = 0`` this reduces to the betatron envelope of
    :func:`plot_beta_functions`.
    """
    import matplotlib.pyplot as plt

    s = [t.s for t in twiss]
    sx, sy = beam_sigma(twiss, emit_x, emit_y, sigma_delta)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(s, sx, label=r"$\sigma_x$")
    ax.plot(s, sy, label=r"$\sigma_y$")
    ax.set_xlabel("s [m]")
    ax.set_ylabel(r"$\sigma$ [m]")
    ax.set_title("beam envelope")
    ax.legend()
    return ax
