"""Dipole: a bending magnet, optionally combined-function and with edge angles."""

from __future__ import annotations

import math

import numpy as np

from ..coords import DELTA, DIM, PX, PY, ZETA, X, Y
from ..reference import ReferenceParticle
from .alignment import arc_motion, frame_change, roll_motion
from .element import Element
from .quadrupole import _focusing_block, _focusing_functions


def _dispersion_integrals(K: float, L: float) -> tuple[float, float, float]:
    r"""Branch-smooth path integrals for the horizontal Hill equation ``u'' + K u = drive``.

    Returns ``(c1, s1, c2)`` where, with ``w = sqrt(|K|)``:

    - ``s1 = sin(wL)/w``          (focusing) / ``sinh(wL)/w`` (defocusing) -> ``L`` as ``K -> 0``
    - ``c1 = (1 - cos wL)/K``     / ``(1 - cosh wL)/K``                     -> ``L^2/2``
    - ``c2 = (s1 - L)/K``                                                   -> ``-L^3/6``

    The combined-function dipole's dispersion is ``R16 = h*c1``, ``R26 = h*s1``,
    and its longitudinal slip carries ``h^2*c2`` (see :meth:`Dipole._arc_matrix`).
    All three have removable singularities at ``K = 0`` (the ``h^2 = -k1`` tune),
    handled by the leading Taylor terms so a combined-function magnet tuned exactly
    there is still exact to machine precision.
    """
    if abs(K) < 1e-9:
        s1 = L - K * L**3 / 6.0
        c1 = L**2 / 2.0 - K * L**4 / 24.0
        c2 = -(L**3) / 6.0 + K * L**5 / 120.0
        return c1, s1, c2
    if K > 0.0:
        w = math.sqrt(K)
        cos_wl, s1 = math.cos(w * L), math.sin(w * L) / w
    else:
        w = math.sqrt(-K)
        cos_wl, s1 = math.cosh(w * L), math.sinh(w * L) / w
    c1 = (1.0 - cos_wl) / K
    c2 = (s1 - L) / K
    return c1, s1, c2


def _sinc(z: np.ndarray) -> np.ndarray:
    """``sin(z)/z``, continued to ``1`` at ``z = 0``.

    The exact bend's map is written entirely in terms of this and its half-angle
    square, which is what makes it **free of any division by the curvature** and so
    continuous into the straight limit — see :func:`exact_sector_bend_map`.
    """
    z = np.asarray(z, dtype=float)
    safe = np.where(z == 0.0, 1.0, z)
    return np.where(z == 0.0, 1.0, np.sin(safe) / safe)


def exact_sector_bend_map(
    state: np.ndarray, length: float, h: float, ref: ReferenceParticle
) -> np.ndarray:
    r"""The **exact** map of a pure sector bend (L3), as a free function.

    A particle of momentum ``1 + delta`` in the uniform field ``B0 = P0 h / q`` moves,
    in projection onto the bend plane, on a *circle* of radius

        r = p_perp / h,      p_perp = sqrt((1 + delta)^2 - py^2),

    and the whole map is that circle meeting the exit face. So unlike the quadrupole's
    (L2), this map is exact in the angles as well as in ``delta`` — there is a closed
    form here because a uniform field has one, and it is the same closed form xtrack
    tracks with (``model="bend-kick-bend"``). See :class:`Dipole` for what that buys
    and where it stops.

    Writing it in terms of the swept angle ``phi``, with ``pz = sqrt((1+delta)^2 -
    px^2 - py^2)``, ``u = pz - 1``, ``C = u - h x`` and ``theta = h L``:

        px  -> px cos(theta) + C sin(theta)
        x   -> x cos(theta) + px L sinc(theta) + [(v - u) + u (1 - cos theta)] / h
        y   -> y + py (L + D/h)                          [ phi = theta + D ]
        zeta-> zeta + L(1 - 1/rvv) - (delta L/(1+delta) + D/h) E/E0
        D   = asin(px/p_perp) - asin(px_out/p_perp)      [ the *extra* angle swept ]

    ``state`` is a ``(6,)`` vector or a ``(6, n)`` bunch.

    **Nothing of size one is subtracted from anything else**, which is the whole of the
    numerical work and is what L1 warned would be needed here. Transcribed as xtrack
    writes it — ``x = (pz_out h - dpx/ds - k)/(h k)``, a numerator of size ``h`` giving
    an answer of size ``x`` — the origin Jacobian comes out at ``3.2e-9`` against
    ``matrix()`` and *degrades* as the finite-difference step shrinks, which would have
    broken every design-optics gate in the package. Rearranged as above it is
    ``4.9e-15`` and improves with the step, as a truncation error should.

    **There is no division by ``h`` left**, so the straight limit needs no branch: at
    ``h = 0`` every curvature term vanishes analytically and what remains *is*
    :class:`~accsim.elements.drift.Drift`'s exact map, agreeing with it to ``6.5e-19``
    (a few ulp — the same map by two arithmetic routes, not a special case). A weak
    bend degrades gracefully rather than falling off a cliff: at ``h = 1e-4`` the
    Jacobian is still ``2.6e-13`` where the transcribed form is ``1.4e-5``.

    A particle with no forward momentum gives ``NaN``, exactly as the drift does, and
    for the same reason: losses belong to
    :class:`~accsim.elements.aperture.Aperture`, and a bend declining to invent a
    trajectory is the honest answer.
    """
    st = np.asarray(state, dtype=float)
    L = length
    if L == 0.0:
        return st.copy()

    x, px, py, delta = st[X], st[PX], st[PY], st[DELTA]
    theta = h * L
    cos_t = math.cos(theta)
    sinc_t = float(_sinc(theta))
    # (1 - cos theta) / h == h * half_chord, with no cancellation and no 1/h.
    half_chord = 0.5 * L * L * float(_sinc(0.5 * theta)) ** 2

    one_plus = 1.0 + delta
    angle_sq = px * px + py * py
    # NaN for a particle with no forward momentum is a *documented* return value; the
    # sqrt's warning is noise, the value itself still propagates. See Drift.
    with np.errstate(invalid="ignore"):
        pz = np.sqrt(one_plus * one_plus - angle_sq)
        # u = pz - 1, rationalised: two numbers of size 1 would otherwise be differenced.
        u = (delta * (2.0 + delta) - angle_sq) / (pz + 1.0)
        C = u - h * x
        px_out = px * cos_t + C * h * L * sinc_t  # C sin(theta)
        pz_out = np.sqrt(one_plus * one_plus - px_out * px_out - py * py)

        # Q = (px - px_out)/h, regular at h = 0 and the only place the geometry enters.
        Q = px * h * half_chord - C * L * sinc_t
        x_out = (
            x * cos_t
            + px * L * sinc_t
            + Q * (px + px_out) / (pz_out + pz)  # (pz_out - pz)/h
            + u * h * half_chord
        )

        # D/h, with D = asin(a) - asin(b) the *extra* angle swept beyond the design
        # bend. Written as asin of a product so that neither the difference of two
        # arcsines nor the difference inside the arcsine identity is ever formed:
        #   a sqrt(1-b^2) - b sqrt(1-a^2) = (a - b) [ Sigma/2 + (a+b)^2 / (2 Sigma) ].
        inv_p_perp = 1.0 / np.sqrt(one_plus * one_plus - py * py)
        a, b = px * inv_p_perp, px_out * inv_p_perp
        sigma = np.sqrt(1.0 - a * a) + np.sqrt(1.0 - b * b)
        scale = 0.5 * sigma + 0.5 * (a + b) ** 2 / sigma
        w = (a - b) * scale
        w_safe = np.where(w == 0.0, 1.0, w)
        arcsinc = np.where(w == 0.0, 1.0, np.arcsin(w_safe) / w_safe)
        d_over_h = arcsinc * scale * inv_p_perp * Q

    # zeta: the speed term and the path term, kept apart so neither is formed by
    # subtracting two numbers of size L (the trap L1 recorded, and L2 met again).
    E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
    slip = L * delta * (2.0 + delta) / ref.gamma0**2 / (one_plus * (one_plus + E_over_E0))
    path = delta * L / one_plus + d_over_h

    out = st.copy()
    out[X] = x_out
    out[PX] = px_out
    out[Y] = st[Y] + py * (L + d_over_h)
    out[ZETA] = st[ZETA] + slip - path * E_over_E0
    return out


def _cfd_path_integrals(
    K: np.ndarray, L: float, C: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(c2, t1)``: the two path integrals of the Hill equation that divide by ``K``.

    With ``C = cos(sqrt(K) L)`` and ``S = sin(sqrt(K) L)/sqrt(K)`` (continued to
    ``cosh``/``sinh`` for ``K < 0``, as :func:`_focusing_functions` does),

        c2 = (S - L) / K                    -> ``-L^3/6``  as ``K -> 0``
        t1 = (L - C S) / (2 K)              -> ``+L^3/3``  as ``K -> 0``

    ``c2`` is what the dispersion drive contributes to ``int x ds`` and ``t1`` what it
    contributes to ``int x'^2/2 ds``; see :func:`expanded_cfd_map`. Both are **entire**
    in ``K`` -- the pole is removable -- but both are written as a quotient whose
    numerator cancels to nothing at small ``K``, so the closed form is used only where it
    is accurate and the (equally exact) Taylor series elsewhere:

        c2 = -L^3 * sum_m (-K L^2)^m / (2m+3)!
        t1 = 2 L^3 * sum_m (-4 K L^2)^m / (2m+3)!

    The switch is at ``|K L^2| = 1e-2``, where five series terms truncate at ``1e-19``
    relative and the closed form has lost only ``~1e-14`` to cancellation -- so there is
    no window in which either side is the inaccurate one. ``K`` is a **per-particle**
    array here (it carries the ``1/(1+delta)`` rigidity), which is why this is vectorised
    rather than the scalar branch :func:`_dispersion_integrals` uses for ``matrix()``.
    """
    z = np.asarray(K, dtype=float) * L * L
    small = np.abs(z) < 1.0e-2
    # Series argument, clamped so the polynomial never sees a large z (it is discarded
    # there anyway, but np.where evaluates both sides).
    v = np.where(small, -z, 0.0)
    c2_series = -(L**3) * (
        1.0 / 6.0 + v * (1.0 / 120.0 + v * (1.0 / 5040.0 + v * (1.0 / 362880.0 + v / 39916800.0)))
    )
    w = 4.0 * v
    t1_series = (2.0 * L**3) * (
        1.0 / 6.0 + w * (1.0 / 120.0 + w * (1.0 / 5040.0 + w * (1.0 / 362880.0 + w / 39916800.0)))
    )
    K_safe = np.where(small, 1.0, K)
    c2 = np.where(small, c2_series, (S - L) / K_safe)
    t1 = np.where(small, t1_series, (L - C * S) / (2.0 * K_safe))
    return c2, t1


def expanded_cfd_map(
    state: np.ndarray, length: float, h: float, k1: float, ref: ReferenceParticle
) -> np.ndarray:
    r"""The **curved quadrupole's** momentum-dependent map (L4), as a free function.

    The linear (``mat``) half of :class:`Dipole`'s combined-function tracking: MAD-X's
    ``track_thick_cfd``, which is xtrack's ``mat-kick-mat``. It is the exact flow of the
    **paraxial** combined-function Hamiltonian, so -- exactly like L2's quadrupole and
    unlike L3's pure bend -- it is exact in ``delta`` to all orders and drops
    ``O(angle^3)`` relative. See :class:`Dipole` for why that trade is forced here.

    Every strength is normalised to the *reference* rigidity, so a particle of momentum
    ``1 + delta`` feels ``k0 = h/(1+delta)`` and ``k1/(1+delta)``, while the **curvature**
    ``h`` is geometry and does not scale. Writing ``q = 1 + delta``, ``x' = px/q``,
    ``y' = py/q``:

        K_x = (h^2 + k1)/q      K_y = -k1/q      G = h - k0 = h delta/q

    ``G`` is the whole of the dispersion: the design particle (``delta = 0``) sits on the
    reference circle and feels no net drive, a stiffer one is under-bent and drifts
    outward. The equations of motion are then ``x'' + K_x x = G`` and ``y'' + K_y y = 0``,
    and with ``C``, ``S`` the Hill pair of :func:`_focusing_functions`,
    ``c1 = (1-C_x)/K_x``, and ``A = -K_x x + G``, ``B = x'``:

        x  -> x C_x + x' S_x + G c1           px -> (A S_x + B C_x) q
        y  -> y C_y + y' S_y                  py -> (-K_y y S_y + y' C_y) q
        zeta -> zeta + L(1 - 1/rvv) - (Lambda - L)/rvv

    with the extra path length ``Lambda - L = h int x ds + int (x'^2 + y'^2)/2 ds``. The
    first term is the bend's own geometry -- a particle on the outside of the arc travels
    further -- and is the term a straight magnet does not have; the second is L2's
    :func:`~accsim.elements.quadrupole._path_lengthening`, generalised to an
    inhomogeneous ``A``:

        int x ds        = x S_x + x' c1 - G c2
        int u'^2/2 ds   = (A^2 t1 + A B S^2 + B^2 (L - K t1)) / 2

    ``c1`` is evaluated as ``2 S(K, L/2)^2`` -- the half-angle identity
    ``(1 - cos u) = 2 sin^2(u/2)`` -- so the one term of the *transverse* map that would
    otherwise divide by ``K`` never does, at any ``K``, with no branch. ``c2`` and ``t1``
    are :func:`_cfd_path_integrals`. And ``zeta`` is split the way L1, L2 and L3 all had
    to split it: ``L(1 - 1/rvv)`` rationalised through ``(1+delta) + E/E0``, never
    xtrack's cancelling ``length - Lambda/rvv``.

    ``state`` is a ``(6,)`` vector or a ``(6, n)`` bunch; ``K_x``, ``K_y`` and ``G`` are
    all per-particle, which is why this is not a matrix multiply.
    """
    st = np.asarray(state, dtype=float)
    L = length
    if L == 0.0:
        return st.copy()

    x, px, y, py, delta = st[X], st[PX], st[Y], st[PY], st[DELTA]
    one_plus = 1.0 + delta

    # The strengths a particle of momentum (1 + delta) actually feels. The curvature h
    # is the *geometry* of the reference orbit and is not divided: that asymmetry is
    # precisely what makes G nonzero, i.e. what dispersion is.
    Kx = (h * h + k1) / one_plus
    Ky = -k1 / one_plus
    G = h * delta / one_plus  # = h - k0, the drive of x'' + Kx x = G

    Cx, Sx = _focusing_functions(Kx, L)
    Cy, Sy = _focusing_functions(Ky, L)
    c1 = 2.0 * _focusing_functions(Kx, 0.5 * L)[1] ** 2  # (1 - Cx)/Kx, half-angle form
    c2, t1x = _cfd_path_integrals(Kx, L, Cx, Sx)
    t1y = _cfd_path_integrals(Ky, L, Cy, Sy)[1]

    xp = px / one_plus  # the geometric angle dx/ds, paraxially
    yp = py / one_plus
    A, B = -Kx * x + G, xp
    Cv, D = -Ky * y, yp

    out = st.copy()
    out[X] = x * Cx + xp * Sx + G * c1
    out[PX] = (A * Sx + B * Cx) * one_plus
    out[Y] = y * Cy + yp * Sy
    out[PY] = (Cv * Sy + D * Cy) * one_plus

    # zeta: the speed term and the path term, kept apart so neither is formed by
    # subtracting two numbers of size L (L1's trap, met again by L2 and L3).
    path = h * (x * Sx + xp * c1 - G * c2)  # h * int x ds: the curvature's own share
    path += 0.5 * (A * A * t1x + A * B * Sx * Sx + B * B * (L - Kx * t1x))
    path += 0.5 * (Cv * Cv * t1y + Cv * D * Sy * Sy + D * D * (L - Ky * t1y))

    E_over_E0 = np.hypot(ref.momentum_eV * one_plus, ref.mass_eV) / ref.total_energy_eV
    slip = L * delta * (2.0 + delta) / ref.gamma0**2 / (one_plus * (one_plus + E_over_E0))
    out[ZETA] = st[ZETA] + slip - path * E_over_E0 / one_plus
    return out


def curvature_sextupole_kick(state: np.ndarray, hk1l: float) -> np.ndarray:
    r"""The thin kick of F2's Maxwell curvature-sextupole term ``psi_3`` (L4).

    A combined-function *sector* magnet cannot have exactly ``B_y = B0(h + k1 x)``,
    ``B_x = B0 k1 y``: that field has ``div B = h k1 y != 0`` in the curved frame.
    Maxwell forces a third-order correction to the potential, and F2 pinned its split
    against xtrack and MAD-X (``docs/CONVENTIONS.md`` -> *Dipole chromaticity*):

        psi_3 = -(h k1 / 3) x^3 + (h k1 / 2) x y^2

    Since the Hamiltonian carries ``-psi``, ``H_3 = (h k1/3) x^3 - (h k1/2) x y^2`` and
    the thin kick over an integrated ``h k1 L`` is ``Delta p = -L grad H_3``:

        px -> px + h k1 L (-x^2 + y^2/2)
        py -> py + h k1 L (x y)

    the same expression xtrack applies as its ``k1_h_correction`` in ``mat-kick-mat``.
    Three properties are worth naming, because they are what make it safe to insert
    between two halves of :func:`expanded_cfd_map`:

    - it is the gradient of a potential in ``(x, y)`` alone, so it is **exactly
      symplectic** and leaves ``delta`` (and ``zeta``, since ``H_3`` has no ``delta``)
      untouched;
    - it carries **no** ``1/(1+delta)``, for the same reason a
      :class:`~accsim.elements.quadrupole.ThinQuadrupole` does not: a field changes every
      particle's *momentum* by the same amount, and it is the angle that responds to
      rigidity;
    - its Jacobian at the origin is **zero** (the kick is quadratic in the coordinates),
      so a centred kick cannot disturb the identity that :meth:`Dipole._matrix_body` is
      the origin Jacobian of :meth:`Dipole._track_body`.

    The ``2:-1`` ratio is **not** an ordinary sextupole's symmetric one, because
    ``psi_3`` is not a pure sextupole. That is the coefficient the feed-down gates in
    ``tests/analytic/test_curved_quadrupole.py`` exist to pin.
    """
    st = np.asarray(state, dtype=float)
    x, y = st[X], st[Y]
    out = st.copy()
    out[PX] = st[PX] + hk1l * (y * y * 0.5 - x * x)
    out[PY] = st[PY] + hk1l * x * y
    return out


def _edge_matrix(h: float, e: float) -> np.ndarray:
    r"""Thin hard-edge pole-face focusing kick for edge angle ``e`` [rad].

    A pole face rotated by ``e`` (``e = 0`` is the sector face; ``e = theta/2``
    the symmetric rectangular face) acts as a thin quadrupole-like kick at the
    entrance/exit of the body. In the **hard-edge** limit (zero fringe extent,
    ``FINT = 0``) the linear map is the identity except:

        px -> px + h*tan(e) * x      (R21 = +h tan e)
        py -> py - h*tan(e) * y      (R43 = -h tan e)

    So a positive edge angle **defocuses horizontally and focuses vertically** --
    the sign is fixed by the geometry of the rotated face (the field the particle
    sees lengthens on the outside of the bend), not remembered. Each 2x2 block
    ``[[1, 0], [+-h tan e, 1]]`` has unit determinant, so the kick is symplectic.

    The fringe-field correction (``e -> e - psi`` in the *vertical* plane only,
    ``psi = h*g*fint*(1 + sin^2 e)/cos e``) is deliberately **not** applied here:
    this is the hard-edge map, the apples-to-apples match to MAD-X ``sbend`` with
    its default ``FINT = HGAP = 0``. Fringe is a separate, opt-in refinement.
    """
    E = np.eye(DIM)
    if h == 0.0 or e == 0.0:
        return E  # no bending or no rotation -> no edge focusing
    t = h * math.tan(e)
    E[PX, X] = t  # horizontal defocus for e > 0
    E[PY, Y] = -t  # vertical focus for e > 0
    return E


class Dipole(Element):
    r"""A dipole of arc length ``L`` and bend angle ``theta`` [rad].

    The reference orbit curves with radius ``rho = L/theta`` (curvature
    ``h = 1/rho = theta/L``); bending is horizontal (the ``x`` plane). Two optional
    refinements, both **off by default** (so the default is a pure sector bend,
    byte-identical to the original):

    - ``k1`` -- a **combined-function** quadrupole gradient in the body [m^-2];
    - ``e1`` / ``e2`` -- entrance / exit **pole-face** rotation angles [rad].

    The body's linear 6x6 map is ``exp(L*A)`` of the (combined-function) bend
    Hamiltonian generator (pinned symbolically and cross-checked entrywise against
    xtrack and MAD-X). With ``k1 = 0`` the non-trivial body entries are, with
    ``C = cos theta``, ``S = sin theta``:

    - **Horizontal** (weak geometric focusing): ``R11 = R22 = C``,
      ``R12 = S/h = rho*S``, ``R21 = -h*S``.
    - **Dispersion** (coupling to ``delta``): ``R16 = (1-C)/h = rho*(1-C)``,
      ``R26 = S``. A higher-momentum particle bends less, so it is displaced
      outward (``R16 > 0``).
    - **Vertical**: a plain drift (``R34 = L``) -- a pure sector bend has no
      vertical focusing.
    - **Longitudinal** (path-length / time-of-flight): ``R51 = -S``,
      ``R52 = (C-1)/h = -rho*(1-C) = -R16``, and
      ``R56 = rho*S - L + L/gamma0^2``. The ``R51``/``R52`` terms are exactly the
      symplectic partners of the dispersion (``R51 = R21*R16 - R11*R26``); ``R56``
      is the drift slip ``L/gamma0^2`` minus the extra arc the design orbit
      travels, ``rho*(theta - S)``.

    **Combined function** (``k1 != 0``). The horizontal focusing becomes
    ``K_x = h^2 + k1`` (geometric weak focusing *plus* the gradient) and the
    vertical ``K_y = -k1``, so ``k1 > 0`` focuses ``x`` and defocuses ``y`` just
    like a :class:`Quadrupole`. Dispersion, ``R51``/``R52`` and the ``R56`` slip
    all pick up the gradient through ``K_x``; the map reduces to the pure sector at
    ``k1 = 0`` and to a pure :class:`Quadrupole` at ``h = 0``. See
    :meth:`_combined_function_body`.

    **Edge angles.** The full map is ``Edge(e2) @ Body @ Edge(e1)`` -- the
    entrance edge acts first. Each edge is the hard-edge kick of
    :func:`_edge_matrix`. Two consequences worth naming, both exact in this linear
    hard-edge model:

    - **Rectangular bend** (``e1 = e2 = theta/2``): the two edges *exactly* cancel
      the body's horizontal weak focusing, leaving the horizontal block equal to a
      drift ``[[1, rho*sin theta], [0, 1]]`` (``R21 = 0`` to machine precision),
      while the vertical plane gets all its focusing from the edges
      (``R43 ~ -2 h tan(theta/2)``).
    - Edges are optics-active (they change beta, tune, chromaticity and dispersion
      through composition) but add no length and no direct longitudinal coupling.

    As ``theta -> 0`` every curvature term vanishes (and the edges too, since
    ``h -> 0``) and the map reduces exactly to a :class:`Drift` of length ``L``
    (``R56 -> L/gamma0^2``).

    The exact map (L3) and the expanded one (L4)
    -------------------------------------------
    Like the :class:`~accsim.elements.drift.Drift` and the
    :class:`~accsim.elements.quadrupole.Quadrupole`, a bend has two maps:
    :meth:`_matrix_body` above, which every optics function is built on, and
    :meth:`_track_body`, which is what a tracked particle actually follows. The first is
    the Jacobian of the second at the origin, and only there.

    **Which second map applies is decided by ``k1``, and the split is forced rather than
    chosen.** With ``k1 = 0`` the vertical equation ``y' = py (1 + h x)/pz`` is a
    *quadrature* over a known ``x(s)``, because ``py`` is conserved — which is exactly
    why a closed form exists. With ``k1 != 0`` the same equation becomes a second-order
    ODE with an ``s``-dependent coefficient, and the geometric term and vertical focusing
    become mutually exclusive in closed form.

    **``k1 = 0`` — the exact circle (L3).** A uniform field has a closed-form flow and it
    is a **circle**: a particle of momentum ``1 + delta`` moves, in projection onto the
    bend plane, on a circle of radius ``r = p_perp/h``,
    ``p_perp = sqrt((1+delta)^2 - py^2)``, and the map is that circle meeting the exit
    face (:func:`exact_sector_bend_map`). So — unlike the quadrupole's — this map is
    exact in the **angles as well as in** ``delta``. It reproduces
    ``xt.Bend(model="bend-kick-bend")`` to ``1.9e-16``, and an independent
    plane-geometry construction to ``1e-15`` at bend angles up to ``1.5 rad``.

    **``k1 != 0`` — the expanded map (L4).** Two halves of :func:`expanded_cfd_map`
    around one centred :func:`curvature_sextupole_kick`: MAD-X's ``track_thick_cfd`` with
    F2's Maxwell term, which is exactly ``xt.Bend(model="mat-kick-mat")`` with one
    uniform kick, reproduced to ``1.0e-16``. Exact in ``delta`` to all orders, paraxial
    in the angles. The composition keeps :meth:`_matrix_body` the *exact* origin Jacobian
    of :meth:`_track_body` — two half-length Hill solutions compose to the full one
    identically, and a cubic potential's kick has zero Jacobian at the origin — which is
    the invariant that bounds this whole axis and is what rules out slicing families.

    ⚠️ **What the expanded family drops, and it is not only the third-order angle
    terms.** It solves ``x' = px/(1+delta)`` where the exact curvilinear equation is
    ``x' = px(1 + h x)/pz``, keeping the ``(1 + h x)`` metric factor only in the path
    length. Evaluated on the dispersed orbit that factor **is** F2's
    ``h(gamma_x D_x - 2 alpha_x D_px)`` / ``gamma_y h D_x`` chromaticity group, which is
    the term that largely cancels the geometric ``-beta_x h^2`` focusing. So a *bending*
    combined-function magnet's **tracked** chromaticity converges to F2 minus that group,
    not to F2 — measured in closed form in ``tests/analytic/test_curved_quadrupole.py``
    and confirmed from the other side by xtrack, whose own converged ``mat-kick-mat``
    lands on the same value while its exact families land on F2. A *straight* gradient
    magnet has ``h = 0``, the group vanishes identically, and tracking is complete.
    :func:`~accsim.natural_chromaticity` is unaffected and remains the deliverable.

    ⚠️ **A bending magnet is therefore discontinuous in ``k1`` at zero** — by ``1.8e-5``
    at millimetre amplitudes, and *not* shrinking as ``k1 -> 0``. The jump is
    **quadratic** in the coordinates, not L2's ``O(angle^3)``, because it is two things
    at once: the expanded square root, which for a bend enters ``px' = h p_z - h``
    already at ``O(p^2)``, and the dropped metric factor, whose signature is a bilinear
    ``h x px``. Both are measured, and the discontinuity is the price of keeping the
    strictly better map on the sub-case that provably admits one.

    **What the exact map buys, per element and to first order in the orbit.** Writing
    ``t`` for the bend angle, the Jacobian at an orbit with angles ``px``, ``py`` gains

        M[y, delta] = M[zeta, py] = -py rho sin t          (the source K2 specified)
        M[y, x]  = +py sin t,      M[y, px] = +py rho (1 - cos t)     (plane coupling)
        M[x, delta] = -px rho sin t cos t

    The middle line is the surprise and it is **not** in K2's formula: those two entries
    are ``py`` times the bend's *own* dispersion entries, so an upright sector bend on a
    vertical orbit **couples the planes** and transports horizontal dispersion into the
    vertical. On a real arc that path is the larger one. The last line is the other
    trap: the horizontal response is not the plane swap of the vertical one — ``px`` is
    not conserved, so the response feeds back through the bend's own focusing and picks
    up an extra ``cos t``. Both are derived in ``tests/analytic/test_exact_dipole.py``
    from the equations of motion, not recalled.

    **Edges stay linear.** :meth:`_track_body` composes ``Edge(e2) . body . Edge(e1)``
    with the same hard-edge kicks :meth:`_matrix_body` uses. Each edge is *exactly*
    linear, so it is not an approximation inside the composition and the Jacobian
    identity survives it. The real pole-face map is nonlinear (xtrack's wedge and
    fringe); that is out of scope here, as :func:`_edge_matrix` already records.

    **On the design orbit nothing changes.** Every new entry is proportional to an orbit
    angle, and at the origin the exact map's Jacobian *is* the linear matrix — measured
    at ``4.9e-15`` with a ``1e-7`` finite difference, improving as the step shrinks.

    **A bending dipole refuses to be displaced** (K1), and the refusal is measured
    rather than cautious. K1's misalignment is the translation ``d + body(state - d)``
    — one translation in, the same one back out — which is right for a *straight*
    element. A bend rotates the reference frame through itself, so the exit
    translation lives in a frame turned by ``theta`` and is not the entry one:
    displacing a bend is a **rigid-body** displacement of a curved body, with an
    angular and a path-length consequence the straight formula does not have. xtrack
    implements exactly that distinction (its misalignment header falls back to the
    straight formula only when ``angle == 0``), and the two models differ by ``3.6e-5``
    on a 0.3 mm shift where the *aligned* maps differ by ``5.8e-9``
    (``tests/reference/test_misalignment_xtrack.py``). Rather than quietly ship the
    wrong one, :meth:`kick` and :meth:`_track_body` raise
    :class:`NotImplementedError` when ``angle != 0`` and an offset is set. A
    straight dipole (``angle = 0``, i.e. a gradient magnet) is displaced normally.

    **A bending dipole may be rolled, and that curved geometry is now implemented**
    (K2) — it is the piece the refusal above said was missing, done for the roll
    rather than for the offset. ``roll`` turns the magnet about the beam axis while
    the machine stays where it is (MAD-X ``EALIGN``'s ``DPSI``, xtrack's
    ``rot_s_rad_no_frame``), which is *not* the same as rolling the reference frame
    with it (MAD-X ``TILT``): the frame-following version has **exactly zero kick**,
    because the design orbit was rolled too. What a real roll error produces, to
    first order in ``phi`` and exactly in the bend angle, is

        Delta p_y = -phi sin(angle),     Delta y = -phi rho (1 - cos angle),

    an angle **and** an offset — the arc's sagitta tipped out of the plane. The
    horizontal loss is only second order (``1 - cos phi``), and there is a residual
    frame roll ``phi (1 - cos angle)`` that makes a rolled bend a genuine **coupling**
    source as well.

    A rolled bend is the first element in this package whose ``matrix`` carries a
    vertical ``delta`` column — G1's skew quadrupole only *rotates* dispersion the
    horizontal bends already made, and a
    :class:`~accsim.elements.corrector.Corrector`'s matrix is the identity — so it
    produces ``D_y`` in a ring with no coupling element at all. ⚠️ That is **not** the
    same as being the only way a machine gets vertical dispersion: in the exact maps a
    vertical orbit *angle* makes ``D_y`` too, and accsim's linear elements are blind
    to that route (``docs/CONVENTIONS.md`` -> *Orbit-driven vertical dispersion*). On
    a realistic arc that route is the **larger** of the two. See
    :meth:`_alignment_exit`.
    """

    def __init__(
        self,
        length: float,
        angle: float,
        k1: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
        name: str | None = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        super().__init__(length, name=name, dx=dx, dy=dy, roll=roll)
        if length == 0.0 and angle != 0.0:
            raise ValueError("a finite bend angle requires a positive length")
        self.angle = float(angle)
        self.k1 = float(k1)
        self.e1 = float(e1)
        self.e2 = float(e2)

    @property
    def curvature(self) -> float:
        """Curvature ``h = 1/rho = theta/L`` [m^-1] (0 for a straight dipole)."""
        return self.angle / self.length if self.length > 0.0 else 0.0

    @property
    def rho(self) -> float:
        """Bending radius ``rho = L/theta`` [m] (``inf`` for a straight dipole)."""
        return self.length / self.angle if self.angle != 0.0 else math.inf

    def _arc_matrix(self, ref: ReferenceParticle) -> np.ndarray:
        """The bare bend body (no edges)."""
        L = self.length
        theta = self.angle
        M = np.eye(DIM)

        # Straight limit with no gradient: a zero-angle "bend" is just a drift.
        if theta == 0.0 and self.k1 == 0.0:
            M[X, PX] = L
            M[Y, PY] = L
            M[ZETA, DELTA] = L / ref.gamma0**2
            return M

        if self.k1 != 0.0:
            return self._combined_function_body(ref)

        # Pure sector bend (no gradient): the original closed form, byte-identical.
        h = theta / L  # = 1/rho
        c, s = math.cos(theta), math.sin(theta)

        # Horizontal plane + dispersion.
        M[X, X] = c
        M[X, PX] = s / h
        M[X, DELTA] = (1.0 - c) / h
        M[PX, X] = -h * s
        M[PX, PX] = c
        M[PX, DELTA] = s
        # Vertical plane: drift.
        M[Y, PY] = L
        # Longitudinal: path-length coupling (symplectic partners of dispersion)
        # plus the drift-like slip reduced by the extra design-orbit arc length.
        M[ZETA, X] = -s
        M[ZETA, PX] = (c - 1.0) / h
        M[ZETA, DELTA] = s / h - L + L / ref.gamma0**2
        return M

    def _combined_function_body(self, ref: ReferenceParticle) -> np.ndarray:
        r"""Body map with a quadrupole gradient ``k1`` (``exp(L*A)``, closed form).

        Equations of motion ``x'' + (h^2 + k1) x = h*delta``, ``y'' - k1 y = 0``:
        horizontal focusing is the *sum* of geometric weak focusing ``h^2`` and
        the gradient ``k1`` (``K_x = h^2 + k1``), while the vertical plane sees
        ``K_y = -k1`` -- so ``k1 > 0`` focuses ``x`` and defocuses ``y``, exactly
        as in :class:`Quadrupole`. Reduces to the pure sector at ``k1 = 0`` and to
        a pure :class:`Quadrupole` at ``h = 0`` (dispersion vanishes with ``h``).
        """
        L = self.length
        h = self.curvature
        Kx = h * h + self.k1
        M = np.eye(DIM)
        # Transverse blocks: Hill equation with K_x (x) and K_y = -k1 (y).
        M[np.ix_([X, PX], [X, PX])] = _focusing_block(Kx, L)
        M[np.ix_([Y, PY], [Y, PY])] = _focusing_block(-self.k1, L)
        # Dispersion (driven by the h*delta term) and its symplectic partners.
        c1, s1, c2 = _dispersion_integrals(Kx, L)
        r16, r26 = h * c1, h * s1
        M[X, DELTA] = r16
        M[PX, DELTA] = r26
        M[ZETA, X] = -r26  # R51 = -R26
        M[ZETA, PX] = -r16  # R52 = -R16
        M[ZETA, DELTA] = L / ref.gamma0**2 + h * h * c2
        return M

    def _matrix_body(self, ref: ReferenceParticle) -> np.ndarray:
        body = self._arc_matrix(ref)
        if self.e1 == 0.0 and self.e2 == 0.0:
            return body  # pure sector: byte-identical to the original map
        h = self.curvature
        # Entrance edge acts first: M = Edge(e2) @ Body @ Edge(e1).
        return _edge_matrix(h, self.e2) @ body @ _edge_matrix(h, self.e1)

    def _alignment_exit(self, ref: ReferenceParticle) -> tuple[np.ndarray, np.ndarray]:
        """The rigid motion that puts a **rolled bend's** exit face back (K2).

        For a straight element (or an unrolled one) this is the base class's plain
        inverse rotation. For a *bending* magnet it is not, and that is the whole of
        K2: rolling the magnet by ``phi`` about the entrance ``s`` axis leaves its
        exit frame at

            T = A^-1 . R_s(phi) . A,

        where ``A`` (:func:`~accsim.elements.alignment.arc_motion`) is the design
        arc's own rigid motion. Conjugating by ``A`` is what makes ``T`` *not* a
        rotation about ``s``: it comes out as a displacement, a pitch, a yaw and only
        ``phi cos(angle)`` of roll, so undoing it needs the full frame change
        (:func:`~accsim.elements.alignment.frame_change`) rather than
        ``s_rotation(-roll)``.

        Two consequences, both first order in ``phi`` and both measured against
        xtrack rather than argued:

        - a **vertical angle** ``-phi sin(angle)`` — the roll acts on the bend's
          *chord*, not on its angle, so this is ``phi theta`` only for a weak bend;
        - a **vertical offset** ``-phi rho (1 - cos angle)`` — the sagitta of the arc,
          tipped out of the plane. In accsim's dispersion solve this term dominates
          the vertical dispersion, so dropping it is not a small error.
        """
        if self.angle == 0.0 or self.roll == 0.0:
            return super()._alignment_exit(ref)
        arc = arc_motion(self.angle, self.rho)
        motion = np.linalg.solve(arc, roll_motion(self.roll) @ arc)
        return frame_change(motion, ref)

    def _refuse_misalignment(self) -> None:
        """A *bending* dipole may not be **displaced** — class docstring (K1).

        Rolled it may be: K2 implements the curved-body geometry for the roll, which
        is exactly what this refusal said was missing. The offset is still refused,
        because a *translated* curved body is a different rigid motion again and
        nothing in the package needs it — see the class docstring.
        """
        if self.is_displaced and self.angle != 0.0:
            raise NotImplementedError(
                f"cannot displace the bending Dipole {self.name!r} (angle={self.angle}): "
                "K1's misalignment is a translation of a straight element, and a bend's "
                "reference frame rotates through it, so entry and exit translations are "
                "not the same transformation. xtrack models this as a rigid-body "
                "displacement of the curved body (measured disagreement 3.6e-5 against "
                "an aligned-model difference of 5.8e-9 — tests/reference/"
                "test_misalignment_xtrack.py), which accsim does not implement. Represent "
                "a bend's steering error with an explicit Corrector, or displace the "
                "quadrupoles, which is where a real machine's orbit comes from anyway"
            )

    def kick(self, ref: ReferenceParticle) -> np.ndarray:
        self._refuse_misalignment()
        return super().kick(ref)

    def _track_body(self, state: np.ndarray, ref: ReferenceParticle) -> np.ndarray:
        """The bend's real map, with the edges as the thin linear kicks they are.

        Two bodies, and which one applies is decided by ``k1`` alone:

        - ``k1 == 0`` — :func:`exact_sector_bend_map`, exact in the angles *and* in
          ``delta`` (L3), because a uniform field's flow is a circle;
        - ``k1 != 0`` — ``mat . kick . mat``: two halves of :func:`expanded_cfd_map`
          around one centred :func:`curvature_sextupole_kick` (L4), exact in ``delta``
          and paraxial in the angles, because a *curved* quadrupole has no closed form.

        The two halves compose to the full linear map identically (a Hill solution over
        ``L/2`` twice is the one over ``L``, and the path integrals add), and the kick's
        Jacobian vanishes at the origin, so :meth:`_matrix_body` remains the **exact**
        origin Jacobian of this method in both branches — the invariant that bounds the
        whole exact-map axis.

        The edges are applied as ``Edge(e2) . body . Edge(e1)``, the same composition
        :meth:`_matrix_body` uses and in the same order, so the Jacobian identity
        survives them too: each edge is *exactly* linear, so a linear factor in the
        composition is not an approximation to anything.

        Vectorised over a trailing particle axis, so a ``(6,)`` state and a ``(6, n)``
        bunch take the same path.
        """
        self._refuse_misalignment()
        st = np.asarray(state, dtype=float)
        h = self.curvature
        if self.e1 != 0.0:
            st = _edge_matrix(h, self.e1) @ st
        if self.k1 == 0.0:
            st = exact_sector_bend_map(st, self.length, h, ref)
        else:
            half = 0.5 * self.length
            st = expanded_cfd_map(st, half, h, self.k1, ref)
            st = curvature_sextupole_kick(st, h * self.k1 * self.length)
            st = expanded_cfd_map(st, half, h, self.k1, ref)
        if self.e2 != 0.0:
            st = _edge_matrix(h, self.e2) @ st
        return st

    def __repr__(self) -> str:
        grad = f", k1={self.k1}" if self.k1 else ""
        edges = f", e1={self.e1}, e2={self.e2}" if (self.e1 or self.e2) else ""
        return f"Dipole(length={self.length}, angle={self.angle}{grad}{edges}{self._repr_tail()})"
