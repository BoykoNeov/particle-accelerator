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

    **This is a contribution to the ring's momentum compaction that
    :func:`~accsim.radiation.radiation_integrals` does not see, and the distinction matters
    for T2.** The ``alpha_c = I1/C`` that ``I1`` "links to" is the *dispersion-driven* part,
    ``int D_x h ds`` — a wiggler contributes nothing to it, because it has no net curvature
    and generates no dispersion. What it contributes here is **geometric**: the wiggle path
    itself shortening with momentum, which lives in ``R56`` and in no lattice integral. Both
    statements are true at once, and a session that opens ``radiation_integrals`` to the
    wiggler in T2 will still not have made ``alpha_c = I1/C`` complete for a ring with one
    in it.

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

    Also refused, and loudly: **radiation** (T2). See :meth:`normalized_field`.
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
    ) -> None:
        if period <= 0.0:
            raise ValueError(f"wiggler period must be > 0, got {period}")
        if periods != int(periods) or periods < 1:
            raise ValueError(
                f"wiggler periods must be a positive integer, got {periods!r}: a partial "
                "period leaves the beam deflected on exit, which is a bend, not a wiggler"
            )
        super().__init__(float(period) * int(periods), name=name, dx=dx, dy=dy, roll=roll)
        self.period = float(period)
        self.h0 = float(h0)
        self.periods = int(periods)

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
        identifies as the hazard. The consequence is that a ``Wiggler`` cannot be tracked
        with any radiation model other than ``"off"``, cannot be spin-tracked, and cannot be
        :func:`~accsim.tapering.taper`-ed — each of which is a refusal the analytic suite
        asserts, and each of which T2 lifts by giving the period average a place to go.
        """
        raise NotImplementedError(
            f"Wiggler({self.name!r}) has no s-independent field: b_y = h0 cos(k s) cosh(k y) "
            f"reverses {2 * self.periods} times inside it and averages to zero, so the "
            "single mid-point sample radiation_kick() and spin_precession() take would "
            "report no radiation from the magnet built to radiate. The period-averaged "
            "<h^2> and <|h|^3> reach the radiation integrals in T2; until then a wiggler "
            "tracks only with radiation='off'."
        )

    def __repr__(self) -> str:
        return (
            f"Wiggler(period={self.period}, h0={self.h0}, "
            f"periods={self.periods}{self._repr_tail()})"
        )
