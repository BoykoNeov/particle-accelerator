"""Drift: a field-free straight section."""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


class Drift(Element):
    r"""A field-free drift of length ``L``.

    **This element has two maps, and the difference between them is physics.**
    :meth:`_matrix_body` is the linear matrix every optics function is built on;
    :meth:`_track_body` is the **exact** map a tracked particle actually follows.
    The first is the Jacobian of the second at the origin, and only there.

    Linear transfer matrix (derived symbolically in ``docs/CONVENTIONS.md`` and
    pinned by ``tests/analytic/test_drift.py``):

    - transverse:    ``x -> x + L*px``,  ``y -> y + L*py``  (R12 = R34 = L)
    - longitudinal:  ``zeta -> zeta + (L / gamma0^2) * delta``           (R56 = L/gamma0^2)

    The longitudinal coupling ``R56 = L/gamma0^2`` is the time-of-flight effect:
    a higher-momentum particle (``delta > 0``) is faster and arrives earlier, so
    ``zeta`` increases. It uses ``delta`` (momentum deviation), giving the
    ``1/gamma0^2`` coefficient; the energy-deviation convention would instead give
    ``L/(beta0^2 gamma0^2)``. As ``gamma0 -> inf`` the coupling vanishes — at
    ultrarelativistic energy all particles travel at ~c regardless of ``delta``.

    Exact map
    ---------
    A drift is a straight line, so the exact map is geometry, not an expansion. With
    ``pz`` the longitudinal momentum in units of ``P0``,

        pz    = sqrt((1 + delta)^2 - px^2 - py^2)
        x    -> x + L px / pz            y -> y + L py / pz
        zeta -> zeta + L (delta (2 + delta) / gamma0^2 - px^2 - py^2)
                       / (pz (pz + E / E0))

    and ``px``, ``py``, ``delta`` are constants of the motion (no field). The
    longitudinal form is the time of flight written out: the geometric path is
    ``L (1 + delta) / pz``, travelled at ``beta``, so
    ``dzeta = L - beta0 * path / beta``, and ``beta0 / beta = E / (E0 (1 + delta))``
    collapses that to ``L (1 - E / (E0 pz))``. It matches ``xt.Drift(model="exact")``
    to ``4.4e-16`` on every coordinate.

    ``1 - E / (E0 pz)`` is **not** how it is evaluated, though, and the difference
    matters more than it looks. Both terms are ~1 and they nearly cancel, so the
    subtraction throws away most of the significant digits of a small answer. That is
    invisible in the value itself but not in its *derivative*: the finite-difference
    Jacobian :func:`~accsim.orbit.linearised_element_maps` takes would carry
    ``~eps/step ~ 2e-9`` per drift, and on a 16-drift ring it showed up as ``3.6e-8``
    where the design-optics gate asks for ``1e-10``. Multiplying through by
    ``(pz + E/E0)`` removes the cancellation exactly — with
    ``(E/E0)^2 = 1 + beta0^2 delta (2 + delta)`` the numerator collapses to
    ``delta (2 + delta) / gamma0^2 - px^2 - py^2`` — and the form above is the result.
    It is also the more legible physics: a momentum term that speeds the particle up
    and an angle term that lengthens its path, competing, with the ``1/gamma0^2``
    showing directly why the effect dies ultrarelativistically.

    **What the linear matrix drops, and why it is not a rounding error.** Expanding
    the transverse map, ``L px / pz = L px (1 - delta + ...)``: the linear matrix keeps
    ``L px`` and drops ``-L px delta``. That missing term is **bilinear** — it is a
    product of two small quantities — so no 6x6 matrix can carry it, and the loss is
    structural rather than a truncation to be tightened. Its consequence is real: a
    particle with a transverse *angle* on the closed orbit acquires dispersion, so a
    ring with a vertical steerer and no bend at all has ``D_y != 0``. accsim reported
    exactly ``0`` there before this map existed; it now reports ``0.2590571``, which is
    xtrack's own answer to seven figures. See ``docs/CONVENTIONS.md`` ->
    *Orbit-driven vertical dispersion*.

    **The dropped term has a canonical partner, and taking only one is wrong.**
    ``x``'s dependence on ``delta`` and ``zeta``'s dependence on ``px`` are conjugate:
    per element both are ``-L px``, and a map carrying one without the other is **not
    symplectic** — wrong at first order in the amplitude, where the correct map is
    exactly right. That is why the transverse and longitudinal halves land together
    and cannot be split across two changes. Gated by
    :func:`~accsim.symplectic.is_symplectic_map_canonical`; note that plain
    :func:`~accsim.symplectic.is_symplectic_map` *rejects this correct map*, because
    ``(zeta, delta)`` is not a canonical pair — see that module's docstring.

    **On the design orbit nothing changes.** At ``px = py = 0`` the exact map's
    Jacobian *is* the linear matrix, entry for entry: ``d(L px / pz)/d(delta) = 0``
    when ``px = 0``, and so does the conjugate ``d(zeta)/d(px)``. So every quantity
    computed from :meth:`matrix` — beta, the tunes, the chromaticity, the dispersion of
    an aligned on-axis lattice — is **bit-for-bit** what it was, and every
    design-optics cross-check is untouched. The new terms switch on with the orbit,
    which is the only place they are physical.

    The one route that does move on an unsteered ring is
    :func:`~accsim.orbit.linearised_element_maps`, which reaches the same matrices by
    *finite differences* and so has a floor rather than an identity: measured
    ``2.7e-13`` on the one-turn map of a 16-drift ring, against ``~1e-16`` when this map
    was linear. That is the differencing, not the physics, and it sits three orders
    below the ``1e-10`` those gates ask for. It would have been ``3.6e-8`` — and would
    have *broken* them — had the longitudinal term been left in its cancelling form.

    A particle with no forward momentum (``px^2 + py^2 >= (1 + delta)^2``) cannot
    traverse a drift at all; ``pz`` is imaginary and the map returns ``NaN``. That is a
    loss condition, and losses belong to
    :class:`~accsim.elements.aperture.Aperture`, not here — a drift declining to
    invent a trajectory is the honest answer.

    **Displacing a drift does exactly nothing** (K1): there is no field to be off the
    centre of, and the misalignment kick ``(I - M) d`` of
    :meth:`~accsim.elements.element.Element.kick` comes out identically zero because a
    drift moves ``x`` only through ``px``. It accepts ``dx``/``dy`` all the same, so
    that a lattice can be misaligned wholesale and the invariance asserted rather
    than assumed. That statement is about the *linear* map; the exact map is
    translation-invariant too, and for the same reason — it depends on ``px``/``py``
    and never on ``x``/``y``.
    """

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        M[X, PX] = L
        M[Y, PY] = L
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """The exact field-free map — see the class docstring for the derivation.

        Vectorised over a trailing particle axis, so a ``(6,)`` state and a ``(6, n)``
        bunch take the same path. A zero-length drift returns the state untouched
        rather than evaluating ``pz``, so a marker-length element is exactly the
        identity even for a particle whose ``pz`` would be ``NaN``.
        """
        st = np.asarray(state, dtype=float)
        L = self.length
        if L == 0.0:
            return st.copy()

        px, py, delta = st[PX], st[PY], st[DELTA]
        one_plus = 1.0 + delta
        angle_sq = px * px + py * py
        # NaN for a particle with no forward momentum is a *documented* return value
        # (class docstring), and callers such as
        # :func:`accsim.orbit.closed_orbit_nonlinear` turn it into their own error. So
        # the sqrt's warning is noise rather than a signal, and is silenced here only —
        # never the value itself, which still propagates as NaN.
        with np.errstate(invalid="ignore"):
            pz = np.sqrt(one_plus * one_plus - angle_sq)

        # 1 - E/(E0 pz), rationalised so nothing cancels — see the class docstring.
        # E/E0 via hypot to keep the large-momentum limit clean.
        E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
        slip = delta * (2.0 + delta) / ref.gamma0**2 - angle_sq

        out = st.copy()
        out[X] = st[X] + L * px / pz
        out[Y] = st[Y] + L * py / pz
        out[ZETA] = st[ZETA] + L * slip / (pz * (pz + E_over_E0))
        return out
