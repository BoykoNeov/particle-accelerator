"""Solenoid: the magnet whose field points along the beam (S1)."""

from __future__ import annotations

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .element import Element


def _larmor_functions(K: np.ndarray | float, L: float) -> tuple[np.ndarray, np.ndarray]:
    r"""``(cos(KL), L sinc(KL))`` — the two functions every entry of the map is built from.

    The second one is ``sin(KL) / K`` written so that it **never divides by** ``K``:
    ``sin(u)/K = L sin(u)/u``. Every ``1/K`` in the solenoid's closed form appears as
    ``sin(KL) cos(KL)/K`` or ``sin^2(KL)/K``, i.e. as this quantity times a ``cos`` or a
    ``sin``, so the whole matrix is *entire* in ``K``: ``ks = 0`` collapses to a
    :class:`~accsim.elements.drift.Drift` with no branch to get wrong, and a weak solenoid
    has no cancellation. It is the device
    :func:`~accsim.elements.dipole.exact_sector_bend_map` and :mod:`accsim.geometry` already
    use for a weak bend, for the same reason.

    ``K`` may be an array — the *exact* map's Larmor wavenumber is ``ks/(2(1+delta))``, a
    per-particle number, which is the whole reason :meth:`Solenoid._track_body` is not a
    matrix multiply.
    """
    u = np.asarray(K, dtype=float) * L
    u_safe = np.where(u == 0.0, 1.0, u)
    sinc = np.where(u == 0.0, 1.0, np.sin(u_safe) / u_safe)
    return np.cos(u), L * sinc


class Solenoid(Element):
    r"""A thick solenoid of length ``L`` and normalised strength ``ks`` [rad/m].

    The one core magnet whose field points **along** the beam, ``B = (0, 0, B_s)``, and the
    only one in this package that couples ``x`` and ``y`` without being a rolled version of
    something else. ``ks = B_s / (B rho)_0`` — MAD-X's ``KS``, xtrack's ``ks`` — and it is
    **charge-free**: an electron and a proton at the same ``ks`` get the same map. That is
    not a choice made here; both reference codes were probed for it before this class was
    written (bit-identical maps for ``q0 = +1`` and ``q0 = -1``), because a solenoid's
    coupling sense really is set by the charge and the sign could plausibly have lived in
    either place. It lives in ``ks``, so :meth:`_matrix_body` never reads ``ref.charge``.

    The map
    -------
    From the paraxial Hamiltonian with the uniform solenoid's normalised vector potential
    ``a = (-ks y / 2, +ks x / 2)`` (whose curl is ``ks`` along ``s``), the transverse map
    over a length ``L`` is exactly

        M4 = Rot(K L) . blockdiag(F(K^2, L), F(K^2, L)),      K = ks / 2,

    where ``Rot(theta)`` turns ``(x, y)`` and ``(px, py)`` together through ``theta`` and
    ``F`` is :func:`~accsim.elements.quadrupole._focusing_block` — the **quadrupole's own**
    2x2 — at strength ``K^2``, *equal in both planes*. Written out, with ``C = cos KL`` and
    ``S = sin KL``, on ``(x, px, y, py)``:

        [[  C^2,   SC/K,  SC,    S^2/K],
         [-K SC,    C^2, -K S^2,  SC  ],
         [  -SC, -S^2/K,  C^2,   SC/K ],
         [K S^2,    -SC, -K SC,   C^2 ]]

    **The ``1/2`` is the element.** A charge in a longitudinal field precesses at the
    cyclotron rate ``ks``; its *orbit* rotates at the **Larmor** rate ``ks/2``. The factor is
    derived from the Hamiltonian in ``tests/analytic/test_solenoid.py`` with sympy rather
    than recalled, and both reference codes agree with the result (MAD-X entrywise to
    ``2.2e-16``).

    What it actually does to a particle
    -----------------------------------
    Conjugating the matrix by the shear that turns a canonical momentum into a geometric
    angle *inside* the magnet, ``x' = px/(1+delta) + K y``, leaves a map whose **angle rows
    contain no position at all**: the transverse velocity is simply **rotated**, rigidly,
    through ``2 K L``, with its magnitude untouched. A solenoid does not focus the velocity —
    it turns it — and every focusing entry above is the projection of that turning velocity
    onto position. Two things follow, and both are gated:

    - ``x'^2 + y'^2`` is *exactly* constant through the body, so the path lengthening is a
      drift's, ``L (x'^2 + y'^2) / 2``, evaluated with the **inside** angles;
    - that shear **is** the hard edge. The fringe field at each face is what makes the
      canonical momentum continuous while the geometric angle jumps, so the boundary-to-
      boundary map above already has both edges in it — which is what makes it a drift at
      ``ks = 0``. Code that treated ``px`` as the angle would have silently dropped both
      faces and be wrong at first order in ``ks``.

    Longitudinal: the reference orbit is straight, so ``R56 = L / gamma0^2`` exactly as for a
    drift and a quadrupole.

    Exact in ``delta``, paraxial in the angles
    ------------------------------------------
    :meth:`matrix` is evaluated at ``delta = 0``; :meth:`track` uses the per-particle
    ``K = ks / (2 (1 + delta))``, because ``ks`` is normalised to the *reference* rigidity
    and a higher-momentum particle is rotated less. That momentum dependence is the
    solenoid's chromaticity and its contribution to the coupling's momentum dependence; it is
    **three orders** above what is left over (measured: at ``delta = 1e-3`` a frozen-``K``
    map misses xtrack by ``6.2e-7`` against a ``1.1e-10`` floor).

    What is left over is P2 (iv)'s **kinematic** remainder — the paraxial expansion of
    ``sqrt((1+delta)^2 - px^2 - py^2)`` — and it is measured, not assumed: the residual
    against xtrack falls by exactly ``8`` for every halving of the transverse amplitude, i.e.
    it is cubic. ``kinematic_slices`` is deliberately **not** offered here, unlike on
    :class:`~accsim.elements.quadrupole.Quadrupole`: in a solenoid that remainder is a
    function of the *mechanical* momenta, not of the canonical ones, so it is not the
    momentum-only drift :func:`~accsim.elements.quadrupole.kinematic_drift` integrates. The
    gap is recorded and gated by its scaling rather than papered over.

    Axial symmetry
    --------------
    A solenoid is invariant about its own axis, so a ``roll`` does **nothing** to it — its
    matrix is bit-identical at any roll angle. No other element in this package can say that,
    and the analytic suite asserts it. A transverse *displacement* is another matter and
    behaves as K1 says: the matrix is untouched and the whole effect is the constant kick
    ``(I - M) d``.

    Refusals
    --------
    :meth:`normalized_field` **raises**. The accessor returns ``(bx, by)`` and has no ``s``
    component to put a solenoid's field in, so the honest answer is not ``(0, 0)`` — that
    would make :mod:`accsim.spin` report no precession in the one magnet that is a spin
    rotator, and :mod:`accsim.radiation_kick` report no radiation for an off-axis particle.
    Spin (and radiation) in a solenoid is a separate milestone that changes the accessor.

    A solenoid also trips, by design, the *measured* coupling guards in :mod:`accsim.twiss`:
    :func:`~accsim.twiss.closest_tune_approach` and
    :func:`~accsim.twiss.resonance_driving_terms` sum over skew quadrupoles by type and
    refuse a coupling source they do not know, rather than reporting ``0`` for a ring that is
    demonstrably coupled. The eigen path — :func:`~accsim.twiss.normal_mode_tunes` and the
    Edwards-Teng :func:`~accsim.twiss.coupled_twiss` — sees a solenoid exactly, with no
    change needed.
    """

    def __init__(
        self,
        length: float,
        ks: float,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        self.ks = float(ks)

    def _transverse(
        self, K: np.ndarray | float, L: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(C^2, SC/K, SC, S^2/K, K S^2)`` — every distinct entry of the 4x4, once.

        Shared by :meth:`_matrix_body` (scalar ``K = ks/2``) and :meth:`_track_body`
        (per-particle ``K = ks/(2(1+delta))``), so the matrix cannot drift away from the map
        it is supposed to be the origin Jacobian of.
        """
        C, L_sinc = _larmor_functions(K, L)
        S = np.sin(np.asarray(K, dtype=float) * L)
        return C * C, L_sinc * C, S * C, L_sinc * S, np.asarray(K, dtype=float) * S * S

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        L = self.length
        K = 0.5 * self.ks  # the Larmor rate: half the cyclotron rate. Charge-free.
        CC, SC_K, SC, SS_K, K_SS = self._transverse(K, L)
        M = np.eye(DIM)
        M[np.ix_([X, PX, Y, PY], [X, PX, Y, PY])] = np.array(
            [
                [CC, SC_K, SC, SS_K],
                [-K * SC, CC, -K_SS, SC],
                [-SC, -SS_K, CC, SC_K],
                [K_SS, -SC, -K * SC, CC],
            ]
        )
        M[ZETA, DELTA] = L / ref.gamma0**2
        return M

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        r"""The momentum-dependent map: the same closed form at ``K = ks/(2(1+delta))``.

        ``state`` is a ``(6,)`` vector or a ``(6, n)`` bunch; ``K``, and therefore every
        trigonometric function below, is per-particle.

        The transverse half is the matrix of :meth:`_matrix_body` evaluated at the particle's
        own Larmor rate, acting on the *angles* ``px/(1+delta)`` and returning momenta —
        exactly the structure :func:`~accsim.elements.quadrupole.thick_quadrupole_map` has,
        for exactly the same reason (``ks`` is normalised to the reference rigidity).

        The longitudinal half keeps its two terms apart, as the quadrupole's does, so that
        neither is formed by subtracting two numbers of size ``L``. The **path lengthening**
        is a drift's, ``L (x'^2 + y'^2) / 2``, and that is not an approximation: the solenoid
        rotates the transverse velocity without changing its magnitude, so ``x'^2 + y'^2`` is
        constant along the body (derived symbolically in the analytic suite). The angles it
        is evaluated at are the ones **inside** the magnet, ``x' = px/(1+delta) + K y`` —
        the hard-edge shear — which is where a dropped fringe would show up.
        """
        st = np.asarray(state, dtype=float)
        L = self.length
        if L == 0.0:
            return st.copy()

        delta = st[DELTA]
        one_plus = 1.0 + delta
        K = 0.5 * self.ks / one_plus
        CC, SC_K, SC, SS_K, K_SS = self._transverse(K, L)

        xp = st[PX] / one_plus  # the geometric angle dx/ds, paraxially
        yp = st[PY] / one_plus

        out = st.copy()
        out[X] = CC * st[X] + SC_K * xp + SC * st[Y] + SS_K * yp
        out[PX] = (-K * SC * st[X] + CC * xp - K_SS * st[Y] + SC * yp) * one_plus
        out[Y] = -SC * st[X] - SS_K * xp + CC * st[Y] + SC_K * yp
        out[PY] = (K_SS * st[X] - SC * xp - K * SC * st[Y] + CC * yp) * one_plus

        # The angles just *inside* the entrance face: the canonical momentum is continuous
        # across a hard edge, the geometric angle is not.
        x_in = xp + K * st[Y]
        y_in = yp - K * st[X]
        E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
        slip = L * delta * (2.0 + delta) / ref.gamma0**2 / (one_plus * (one_plus + E_over_E0))
        path = 0.5 * L * (x_in * x_in + y_in * y_in)
        out[ZETA] = st[ZETA] + slip - path * E_over_E0 / one_plus
        return out

    def normalized_field(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Refused: a solenoid's field has no transverse component to return.

        The accessor's contract is ``(bx, by)``, the field *across* the beam, because that is
        what bends light and what a radiation kick needs. An ideal uniform solenoid's field
        is ``(0, 0, ks)`` — entirely along ``s`` — so the truthful transverse answer is zero
        and the truthful *physical* answer is not representable here at all.

        Returning ``(0, 0)`` would be the worse of the two errors, because it is silent:
        :mod:`accsim.spin` would report **no precession** through the one magnet that is
        built to rotate spins, and :mod:`accsim.radiation_kick` would report no radiation for
        an off-axis particle, which does radiate (its transverse velocity crosses ``B_s``).
        Both consumers reach an element only through this method, so one refusal covers both.
        """
        raise NotImplementedError(
            f"Solenoid {self.name!r} has a purely longitudinal field, and "
            "normalized_field() returns only the transverse (bx, by) components. Spin "
            "precession and radiation in a solenoid need a field accessor with an s "
            "component, which is a separate milestone (S2); returning (0, 0) here would "
            "silently report no precession in a spin rotator."
        )

    def __repr__(self) -> str:
        return f"Solenoid(length={self.length}, ks={self.ks}{self._repr_tail()})"
