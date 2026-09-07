"""Wiggler: the magnet built to radiate, and the plane its focusing lands in (T1)."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import CLIGHT, ReferenceParticle
from .element import Element
from .quadrupole import _focusing_block, _focusing_functions


class Wiggler(Element):
    r"""A planar wiggler: ``periods`` full periods of ``period`` [m] at peak curvature ``h0``.

    Every other magnet in this package either bends the reference trajectory or focuses
    about it, and every one has a field that is constant along ``s`` inside its body. A
    wiggler is neither. Its vertical field alternates,

        b_y = h0 cos(k s) cosh(k y),    b_s = -h0 sin(k s) sinh(k y),    k = 2 pi / period,

    so over a whole period the beam is deflected one way and then the other and leaves
    travelling exactly as it arrived: **no net bend**. Its purpose is the light emitted on
    the way through — it is how a storage ring buys damping it was not designed with, and
    how a light source makes photons at all.

    ``h0`` is the **peak** normalised curvature ``B_peak/(B rho)_0`` [1/m], in the same
    normalisation ``k1`` and ``ks`` already use; :meth:`from_peak_field` converts from
    tesla. The length is **derived**, ``periods * period``, and ``periods`` must be a
    positive integer — an independent length is exactly how the closing of the reference
    trajectory gets broken without anything noticing.

    The map, and the plane its focusing lands in
    --------------------------------------------
    The field above is what Maxwell allows: a planar pole face forces the longitudinal
    component ``b_s``, since ``div b = 0`` and ``curl b = 0`` in the gap (both checked
    symbolically in the analytic suite, not asserted here). The paraxial equations of
    motion are then

        x'' = -b_y + y' b_s,      y'' = -x' b_s

    and they separate. Horizontally ``b_y`` does not depend on ``x`` or ``x'`` at all, so
    the reference trajectory is the sine ``x'(s) = -(h0/k) sin(k s)`` — peak deflection
    ``theta = h0/k`` — and the *map about it* is a **drift**. Vertically the sine orbit
    crosses ``b_s``, giving ``y'' = -h0^2 y sin^2(k s)``, whose period average is

        y'' = -(h0^2 / 2) y.

    **So a wiggler focuses vertically, at ``k_y = h0^2/2 = 1/(2 rho_peak^2)``, in the plane
    a flat sector bend leaves exactly free — and it does not focus horizontally, in the
    plane every bend-shaped intuition puts it.** That inversion is the whole of T1. The
    focusing is *even* in ``h0``, so it does not care about the field polarity or the sign
    of the charge, and it does not vanish when the net bend does.

    The shipped map is that period average:
    :func:`~accsim.elements.quadrupole._focusing_block` at ``h0^2/2`` vertically and a drift
    horizontally (the ``R56`` is *not* a straight element's — see below). It reproduces the
    exact field's own transfer matrix — obtained by integrating the Lorentz force through
    the real ``cos``/``sinh`` field with ``solve_ivp`` — to ``1.26e-05``, where a **drift**
    (what a bend-based model gives vertically) is off by ``9.94e-02``: a discrimination of
    ~7900x, order-unity rather than a tolerance. The residual is not numerical. It is the
    averaging remainder, and it is exactly ``L theta^2 / 4`` — the same quantity as the
    path lengthening below.

    ⚠️ **The structural gates cannot see the coefficient.** Maxwell holds for the field
    above whatever ``k_y`` this class ships (``k_y`` does not appear in it), and *any*
    focusing block is symplectic. ``k_y = h0^2``, ``h0^2/2`` and ``h0^2/4`` all pass both.
    Only the comparison against the integrated field separates them, and the analytic suite
    asserts the blindness as well as the gate — J1's lesson, in the form this element takes
    it.

    The path lengthening, which is a constant
    -----------------------------------------
    The wiggle is a longer road. Over an integer number of periods the cross term with the
    particle's own angle averages away, so the extra path separates cleanly into the usual
    coordinate-dependent piece and a **constant**

        Delta s = L <x'^2> / 2 = L theta^2 / 4,     theta = h0 / k / (1 + delta),

    which is closed-form exact (it reproduces the integrated trajectory to ``3.3e-12``
    relative) and makes this the only magnet in the package whose *aligned, on-design* map
    has a nonzero :meth:`~accsim.elements.element.Element.kick` — the constant term
    :func:`~accsim.orbit.closed_orbit_6d` consumes and every other element leaves at zero.
    ``1.3e-05 m`` for the 1 m probe magnet. The roadmap entry did not name it; it is gated
    against the same integration as everything else.

    **What it does to a ring, measured:** with the RF frequency fixed to the design
    circumference, the longer closed orbit makes the beam run **off momentum** —
    ``delta_co ~ -k_zeta/R56`` — and dispersion carries that into a transverse orbit
    distortion. The synchronous phase does *not* move: ``zeta`` stays at the cavity's
    zero-crossing, which is where a non-radiating ring's synchronous particle sits whatever
    the path length. (An earlier draft of this docstring said the opposite; solving the
    orbit is what corrected it.) Without an RF cavity there is no restoring force for a
    constant ``zeta`` kick at all, and the 6D solve raises.

    **And that path depends on momentum, so ``R56`` is not a drift's.** The deflection is
    ``theta/(1+delta)``, so a stiffer particle wiggles less and travels a shorter road —
    a purely *geometric* momentum dependence that a straight element is not supposed to
    have. Differentiating the ``zeta`` row at the origin,

        R56 = L / gamma0^2  +  (L theta^2 / 4) (2 + 1 / gamma0^2),

    and **the ratio of those two terms is the wiggler parameter, squared and halved**:

        wiggle / drift = (theta^2 / 4) gamma0^2 (2 + 1/gamma0^2) = K^2 / 2 + theta^2 / 4,

    with ``K = gamma0 theta`` — the conventional parameter this class otherwise avoids,
    arriving on its own. ``K`` is a property of the *magnet* (field and period) and carries
    no energy, so that ratio is the **same at every energy**: ``98.08`` on the probe magnet
    at 1 GeV, and still ``98.08`` at 200 MeV and at 5 GeV. Both terms fall as
    ``1/gamma0^2`` together.

    So the longitudinal map of a wiggler is dominated, by two orders for any ``K`` worth
    building, by a term a straight element is not supposed to have at all. This term was
    found by the package's own contract that :meth:`matrix` be the origin Jacobian of
    :meth:`track` — it was missing from the first draft of this class, and nothing else in
    the milestone would have caught it.

    **This looks like a contribution to momentum compaction that lives outside every lattice
    integral, and T2 measured that it is not.** An earlier draft of this docstring said a
    wiggler "generates no dispersion", so that its compaction contribution was purely
    geometric and appeared in no integral. Both halves are wrong: a wiggler generates its
    **own** dispersion, ``eta = (h0/k^2)(1 - cos ks)``, which closes in both coordinates at
    the exit, and ``int eta h ds = -L theta^2 / 2`` is **exactly** the ``gamma``-free part of
    the ``R56`` term above. The geometric picture and the dispersion picture are one
    mechanism seen from two sides, and :func:`~accsim.radiation.radiation_integrals` now
    carries it, so ``I1 == alpha_c * C`` holds on a ring with a wiggler in it. What is left
    over, ``(L theta^2/4)/gamma0^2``, is not compaction at all but velocity slip along the
    longer path.

    Momentum dependence: the **second** power
    -----------------------------------------
    ``h0`` is normalised to the *reference* rigidity, so a particle at ``(1 + delta)`` is
    deflected by ``theta/(1 + delta)`` — and the focusing is quadratic in that deflection,
    not linear in the field. Hence

        k_y(delta) = h0^2 / (2 (1 + delta)^2),

    a **squared** factor where a :class:`~accsim.elements.quadrupole.Quadrupole` carries
    ``k1/(1 + delta)`` and a :class:`~accsim.elements.solenoid.Solenoid` carries
    ``ks/(1 + delta)``. Every gate the roadmap pre-committed is at ``delta = 0`` and so is
    blind to this; it is gated here against the integrated field at ``delta = 0.05``, where
    the first power misses by ``4.4e-03`` and the second by ``1.1e-05`` — the averaging
    remainder, unchanged. Getting the power wrong is a *chromaticity* error, and nothing
    structural would have caught it.

    What this element is not exact in
    ---------------------------------
    Paraxial in the angles, like every thick element here. Integrating the **non**-paraxial
    Lorentz force (the ``sqrt(1 + x'^2 + y'^2)`` factors) moves the horizontal ``R12`` to
    ``L (1 + 3 theta^2 / 4)`` — the wiggle orbit's own ``1/p_s``, averaged along the sine —
    while ``R21`` stays zero. That is this element's kinematic remainder, the analogue of
    the one :class:`~accsim.elements.quadrupole.Quadrupole` records; it is measured, gated
    by its ``3/4`` coefficient and its ``theta^2`` scaling, and **not** shipped, because
    building it in would make the horizontal block something other than a drift and there
    is no reference code to arbitrate it.

    Radiation, and where it comes from
    ----------------------------------
    T1 refused radiation outright. T2 gave the period-averaged ``<h^2>`` and ``<|h|^3>`` to
    :func:`~accsim.radiation.radiation_integrals`, so a wiggler damped a ring on the design
    route while still radiating **nothing** in tracking. T3 closes that: :meth:`field_at`
    is the field with the ``s`` :meth:`normalized_field` has no room for, and
    :meth:`radiation_sample_positions` says where along the body to look. The map is
    untouched — it is still T1's period average — because a sub-period piece of it has no
    wiggle in it at all (``y'' = -h0^2 y sin^2(ks)`` is Mathieu-like and has no closed form
    on a partial period). What T3 resolves is the **sampling**, not the map.

    ``radiation_slices`` is how many field samples the traversal takes **per period** — not
    per element, and the distinction is a finding rather than a convenience: the integrand
    repeats every period, so a count fixed per element silently under-resolves a long magnet
    (T2's ``slices = 64`` on this ten-period probe wiggler is 6.4 per period, where
    ``int |kappa|^3 ds`` is already ~1.5% wrong). The three moments radiation needs converge
    at three *different* rates under mid-point sampling, measured rather than assumed:

    * ``int kappa^2 ds`` — the **mean loss** — is **exact at every n >= 3**, because
      ``cos^2 = (1 + cos 2ks)/2`` and the uniform mid-point sum of ``cos 2ks`` vanishes
      identically. So the headline gate, "the tracked loss equals ``C_gamma E^4 I2 / 2 pi``",
      cannot see this number at all.
    * ``int |kappa| ds`` — the **photon count** — converges as ``n^-2``: ``|cos|`` has a jump
      in its first derivative.
    * ``int kappa^3 ds`` — the **excitation variance**, T2's ``I3`` — converges as ``n^-4``:
      near its zero ``|cos|^3 ~ |u|^3`` is ``C^2``, so the jump is in the *third* derivative,
      two orders later than the kink suggests.

    The default 32 leaves the variance at ``1.6e-05`` and the count at ``1.6e-03``, with the
    mean exact; the floor of 3 is where the mean stops being exact. It is deliberately **not**
    part of :mod:`accsim.scenario`: that format describes the *machine* — geometry and
    strengths — and this is a property of the integration, so a saved scenario reloads at the
    default however it was tracked. Nothing the format does describe depends on it, since the
    map is untouched.

    Still refused, and loudly: **spin** (T4). See :meth:`normalized_field`.
    """

    def __init__(
        self,
        period: float,
        h0: float,
        periods: int,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
        radiation_slices: int = 32,
    ) -> None:
        if period <= 0.0:
            raise ValueError(f"wiggler period must be > 0, got {period}")
        if periods != int(periods) or periods < 1:
            raise ValueError(
                f"wiggler periods must be a positive integer, got {periods!r}: a partial "
                "period leaves the beam deflected on exit, which is a bend, not a wiggler"
            )
        if radiation_slices != int(radiation_slices) or radiation_slices < 3:
            raise ValueError(
                f"radiation_slices must be an integer >= 3, got {radiation_slices!r}: "
                "below three samples per period the mid-point rule no longer integrates "
                "cos^2 exactly and the mean energy loss itself becomes wrong"
            )
        super().__init__(float(period) * int(periods), name=name, dx=dx, dy=dy, roll=roll)
        self.period = float(period)
        self.h0 = float(h0)
        self.periods = int(periods)
        # Per *period*, not per element -- see the class docstring's *Radiation* section.
        self.radiation_slices = int(radiation_slices)

    @classmethod
    def from_peak_field(
        cls,
        period: float,
        peak_field_T: float,
        periods: int,
        ref: ReferenceParticle,
        name: str | None = None,
    ) -> Wiggler:
        """Build from a peak field in **tesla**, the way a wiggler is actually specified.

        ``h0 = B_peak / (B rho)_0`` with ``(B rho)_0 = p0c / c / q0``, which is the same
        rigidity :mod:`accsim.radiation` states for the normalisation of
        :meth:`~accsim.elements.element.Element.normalized_field`. The charge enters, so an
        electron and a proton in the *same* magnet get opposite ``h0`` — and the map is even
        in ``h0``, so they get the same map anyway. That is what makes this conversion a
        convenience rather than a physics decision.
        """
        brho = ref.momentum_eV / CLIGHT / ref.charge
        return cls(period, peak_field_T / brho, periods, name)

    @property
    def wavenumber(self) -> float:
        """``k = 2 pi / period`` [1/m] — the rate at which the field reverses."""
        return 2.0 * math.pi / self.period

    @property
    def deflection(self) -> float:
        """Peak deflection angle ``theta = h0 / k`` [rad] of the reference trajectory.

        **Not** the conventional wiggler parameter ``K = gamma theta``; the two differ by
        ``gamma``, and it is this one the orbit expansions are actually in.
        """
        return self.h0 / self.wavenumber

    @property
    def focusing(self) -> float:
        """``k_y = h0^2 / 2`` [1/m^2] — the vertical focusing, and the milestone."""
        return 0.5 * self.h0 * self.h0

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        M = np.eye(DIM)
        M[np.ix_([X, PX], [X, PX])] = _focusing_block(0.0, L)  # a drift: no weak focusing
        M[np.ix_([Y, PY], [Y, PY])] = _focusing_block(self.focusing, L)
        M[ZETA, DELTA] = L / ref.gamma0**2 + 0.25 * L * self.deflection**2 * (
            2.0 + 1.0 / ref.gamma0**2
        )
        return M

    def _kick_body(self, ref: ReferenceParticle) -> np.ndarray:
        """The path lengthening of the wiggle itself, ``-L theta^2 / 4`` on ``zeta``.

        The one constant term an aligned, on-design element in this package has. It is the
        origin value of the map, so it belongs here and not in the matrix: the wiggle is a
        longer road even for a particle that enters exactly on the axis with exactly the
        reference momentum, and ``zeta`` is what records that.
        """
        k = np.zeros(DIM)
        k[ZETA] = -0.25 * self.length * self.deflection**2
        return k

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        r"""The period-averaged map at the particle's own momentum.

        The same closed form as :meth:`_matrix_body` at ``k_y = h0^2/(2(1+delta)^2)`` — the
        **squared** rigidity factor of the class docstring — plus the path lengthening.

        ``zeta`` is where this element stops resembling
        :func:`~accsim.elements.quadrupole.thick_quadrupole_map`, and the reason is worth
        stating because it is the only structural difference between them. Averaging the
        wiggle gives the Hamiltonian

            H = -(1+delta) + (px^2 + py^2)/(2(1+delta))
                + (h0^2 / (4 (1+delta))) (y^2 + 1/k^2),

        whose **potential carries its own** ``1/(1+delta)``: a quadrupole's ``(k1/2)(x^2 -
        y^2)`` does not, which is exactly why a quadrupole's focusing is
        ``k1/(1+delta)`` and a wiggler's is ``h0^2/(2(1+delta)^2)``. Hamilton's equation
        ``dzeta/ds = dH/dp_zeta`` therefore picks up the potential's own momentum
        derivative, and the bracket to integrate is

            (x'^2 + y'^2)/2  +  (k_y/2) y^2  +  theta^2/(4(1+delta)^2),

        where a quadrupole has only the first term. The middle one integrates in closed form
        with no work at all: ``y'^2 + k_y y^2`` is the vertical oscillator's own conserved
        energy, so its integral over the body is just ``L`` times its entrance value.

        **Dropping that middle term makes the map non-symplectic** — it is not a small
        correction to a longitudinal coordinate, it is the term that keeps ``(zeta, p_zeta)``
        conjugate to the transverse pair. The first draft of this class omitted it and
        :func:`~accsim.symplectic.is_symplectic_map_canonical` is what refused it.
        """
        st = np.asarray(state, dtype=float)
        L = self.length

        delta = st[DELTA]
        one_plus = 1.0 + delta
        Ky = self.focusing / (one_plus * one_plus)
        Cy, Sy = _focusing_functions(Ky, L)

        xp = st[PX] / one_plus  # the geometric angle dx/ds, paraxially
        yp = st[PY] / one_plus

        out = st.copy()
        out[X] = st[X] + L * xp  # horizontally a drift: the reference trajectory is straight
        out[Y] = st[Y] * Cy + yp * Sy
        out[PY] = (-Ky * st[Y] * Sy + yp * Cy) * one_plus
        # px is untouched: the wiggle it acquires inside the magnet is given back by the
        # second half of every period, which is what an integer `periods` guarantees.

        E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
        slip = L * delta * (2.0 + delta) / ref.gamma0**2 / (one_plus * (one_plus + E_over_E0))
        # Three terms, and the middle one is unique to this element in the package --
        # see the docstring above for why a quadrupole does not have it.
        path = (
            0.5 * L * (xp * xp + yp * yp + Ky * st[Y] * st[Y])
            + 0.25 * L * (self.deflection / one_plus) ** 2
        )
        out[ZETA] = st[ZETA] + slip - path * E_over_E0 / one_plus
        return out

    def field_at(
        self, s: np.ndarray | float, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
        np.ndarray | float,
    ]:
        r"""The real field, at ``s`` metres into the body — and the wiggle in ``a_x`` (T3).

        The three components are T1's, now with the ``s`` they were always about::

            b_x = 0,   b_y = h0 cos(ks) cosh(ky),   b_s = -h0 sin(ks) sinh(ky)

        and the **vector potential is where the milestone actually lives**. All of the
        above comes from a single component,

            ``a_x = (h0/k) sin(ks) cosh(ky) = theta sin(ks) cosh(ky)``,   ``a_y = 0``,

        since ``curl a`` gives ``b_y = da_x/ds`` and ``b_s = -da_x/dy``. The consequence
        is not bookkeeping: the state vector carries **canonical** momentum, the physics
        wants the **kinetic** one, and

            ``px - a_x = px - theta sin(ks) cosh(ky)``

        is *exactly* the sub-period orbit ``x'(s) = -theta sin(ks)`` that
        :meth:`_track_body` deliberately does not carry. That is *why* the shipped map
        holds ``px`` constant through the magnet — the wiggle lives in ``a``, not in
        ``p`` — and it means
        :func:`~accsim.radiation_kick.radiation_kick` reconstructs the wiggle orbit
        through the ``p - a`` subtraction S2 already put there, with no orbit code of its
        own.

        **The transverse *position* excursion is not here, and does not need to be.**
        ``x(s) = x0 + (theta/k)(cos ks - 1)`` is the other half of the sub-period orbit
        and equally absent from the state — but ``b_y`` carries no ``x`` dependence at
        all and the wiggle is horizontal, so it moves the curvature by nothing. Only the
        angle enters, through the perpendicular projection, at ``O(theta^2)``. That is
        what makes "resolve the sampling, not the map" sufficient rather than merely
        convenient.

        ``s`` broadcasts against ``x`` and ``y``: a column of sample positions against a
        row of particles returns the whole grid.
        """
        del x  # a planar wiggler's field does not depend on x -- see the docstring
        s_arr = np.asarray(s, dtype=float)
        ky = self.wavenumber * np.asarray(y, dtype=float)
        ks = self.wavenumber * s_arr
        by = self.h0 * np.cos(ks) * np.cosh(ky)
        bs = -self.h0 * np.sin(ks) * np.sinh(ky)
        ax = self.deflection * np.sin(ks) * np.cosh(ky)
        zero = np.zeros_like(by)
        return zero, by, bs, ax, zero

    def radiation_sample_positions(self) -> np.ndarray:
        """The mid-points of :attr:`radiation_slices` uniform steps **per period**.

        ``radiation_slices * periods`` samples across the body, each standing for an equal
        share of the path. Uniform in ``s`` and mid-point rather than end-point because
        that is what makes ``int cos^2 ds`` exact — see :attr:`radiation_slices` for the
        three different convergence rates this choice buys.
        """
        n = self.radiation_slices * self.periods
        return (np.arange(n) + 0.5) * (self.length / n)

    def normalized_field(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """**Raises.** The accessor has no ``s``, and a wiggler's field is nothing without it.

        ``b_y = h0 cos(k s) cosh(k y)`` reverses sign ``2 * periods`` times inside this
        element and averages to **exactly zero** over a period, while
        :func:`~accsim.radiation_kick.radiation_kick` samples the field **once**, at the
        mid-point of the traversal. So the inherited ``(0, 0)`` would report *no radiation
        from the magnet whose only purpose is to radiate* — and radiation goes as the field
        **squared**, so no choice of sample point repairs it. The accessor's *shape* is
        wrong for this element, not merely its argument list; this is the third interface
        finding on this line of work, after S1 (no field along the beam) and S2 (no vector
        potential).

        Raising rather than returning zero is S1's precedent, and it is deliberately louder
        than the roadmap asked for: a silent zero here is the exact failure the axis-T entry
        identifies as the hazard.

        **It still raises after T3, and that is the point.** Radiation no longer comes
        through here — it goes through :meth:`field_at`, which has the ``s`` this one
        lacks — so ``radiation="mean"``, ``"quantum"`` and ``"photons"`` all work, and so
        does :func:`~accsim.tapering.taper`. What still arrives here is
        :mod:`accsim.spin`, which samples the field once at the mid-point exactly as
        radiation used to, and would precess a spin through a field that averages to zero.
        Spin through a wiggler is T4; until it lands, this refusal is what stands between
        a caller and a silently wrong polarisation.
        """
        raise NotImplementedError(
            f"Wiggler({self.name!r}) has no s-independent field: b_y = h0 cos(k s) cosh(k y) "
            f"reverses {2 * self.periods} times inside it and averages to zero, so a single "
            "mid-point sample would report no field at all from the magnet built to "
            "radiate. Radiation tracking no longer comes through here (it uses field_at(), "
            "which has the s); what does is spin precession, and spin through a wiggler is "
            "T4. Until then a wiggler can be tracked and tapered but not spin-tracked."
        )

    def __repr__(self) -> str:
        return (
            f"Wiggler(period={self.period}, h0={self.h0}, "
            f"periods={self.periods}{self._repr_tail()})"
        )
