"""M2's arbiter: the minimal ring, and its ``Q''`` derived without accsim.

M1 shipped a second-order chromaticity that three codes disagreed about on a ring
with bends, and pinned it as unarbitrated. M2 arbitrates it, and this module is the
independent side of the gate: a five-element ring small enough that its ``Q''`` can
be produced from first principles, at a precision no finite-difference comparison
between two codes can reach.

**The ring** — ``ThinQuadrupole(+k) Drift(L) Dipole(Lb, theta) ThinQuadrupole(-k)
Drift(L)`` — is the smallest one that exhibits the effect. The roadmap's
pre-committed "one thin quadrupole plus one sector bend" was one element short:
the whole disagreement lives in how a **drift** propagates a particle that has a
transverse angle, so a ring without a drift cannot show it. Two thin quadrupoles of
opposite sign are needed because a sector bend focuses only horizontally, so a
single quadrupole leaves one plane unstable.

**The maps are derived here, not imported.** The bend is built from lab-frame
geometry — a circle of radius ``p_perp/h`` meeting the exit face — rather than
transcribed from :func:`accsim.elements.dipole.exact_sector_bend_map`, which is
heavily rearranged for numerical stability; porting that rearrangement would test it
against itself. ``test_chromatic_arbiter.py`` asserts the two agree.

**Why mpmath rather than sympy.** A second difference at ``delta = 1e-12`` carried at
60 decimal digits has ``O(delta^2)`` truncation error near ``1e-24`` and round-off
near ``1e-36``, so the number it returns is the closed-form answer to twenty digits.
The sympy route (a Taylor series in ``delta`` about the closed orbit) reaches the
same place, and was run once to confirm it, but it is far too slow to sit in the
analytic suite — see ``docs/CONVENTIONS.md`` -> *Test-suite cost*.

Every routine here is built **twice**, over the exact drift ``x += L px/pz`` and the
paraxial drift ``x += L px/(1+delta)``, because the difference between those two is
the milestone's entire finding.
"""

from __future__ import annotations

import functools

from mpmath import acos, cos, findroot, matrix, mp, mpf, pi, sin, sqrt

from accsim import Dipole, Drift, Lattice, ReferenceParticle, ThinQuadrupole

# The ring. Chosen so both planes are stable (|Tr| < 2) and neither tune sits near a
# half integer, where a second difference amplifies everything — asserted, not assumed.
KF = 0.9  # thin-quad strength [1/m]
LD = 0.5  # drift length [m]
LB = 1.0  # bend length [m]
ANG = 0.12  # bending angle [rad]

DPS = 60  # working precision of the arbiter
STEP = mpf(10) ** -12  # momentum step of its second difference

# Any reference particle gives the same answer: this ring's transverse map has no
# reference-particle term at all (only ``zeta`` does, and ``zeta`` cannot feed back
# without an RF cavity). Asserted in the analytic suite.
MASS0 = 938.27208816e6
GAMMA0 = 20.0


def lattice(gamma: float = GAMMA0) -> Lattice:
    """The minimal ring as accsim builds it."""
    ref = ReferenceParticle.from_gamma(MASS0, gamma)
    return Lattice(
        [
            ThinQuadrupole(KF),
            Drift(LD),
            Dipole(LB, ANG),
            ThinQuadrupole(-KF),
            Drift(LD),
        ],
        ref,
    )


# ---------------------------------------------------------------------------
# the maps, derived here
# ---------------------------------------------------------------------------


def thin_quad_map(state: list, k1l) -> list:
    """``px -> px - k1l x``, ``py -> py + k1l y``: no ``1/(1+delta)``.

    ``k1l`` is normalised to ``P0``, so the kick a fixed gradient gives a particle in
    *normalised* momentum is the same at every momentum. That is why a thin-lens ring
    has no chromaticity at all, and why the bend-free control M1 could solve in closed
    form was a control rather than a warm-up.
    """
    x, px, y, py, delta = state
    return [x, px - k1l * x, y, py + k1l * y, delta]


def drift_map(state: list, length, *, exact: bool) -> list:
    r"""The field-free map, in the two models the milestone is about.

    ``exact``   : ``x += L px / pz``, ``pz = sqrt((1+delta)^2 - px^2 - py^2)``
    ``paraxial``: ``x += L px / (1 + delta)``

    They differ at ``O(angle^2)`` relatively — the exact form is the paraxial one
    times ``1 + (px^2 + py^2)/(2 (1+delta)^2) + ...`` — so on a ring whose closed
    orbit is straight they are the same map, and on one with bends they are not.
    """
    x, px, y, py, delta = state
    one = 1 + delta
    inv = 1 / sqrt(one**2 - px**2 - py**2) if exact else 1 / one
    return [x + length * px * inv, px, y + length * py * inv, py, delta]


def bend_map(state: list, length, angle) -> list:
    r"""The exact sector bend, from lab-frame geometry.

    Take the entrance face as the plane ``Z = 0``, with the design particle entering
    the lab origin along ``+Z`` and curving toward ``-X`` on a circle of radius
    ``rho = 1/h``, ``h = angle/length``, centred at ``(-rho, 0)``. A particle whose
    momentum projected into the bend plane is ``p_perp = sqrt((1+delta)^2 - py^2)``
    rides a circle of radius ``r = p_perp/h`` — a uniform field bends normalised
    momentum, so a stiffer particle turns on a wider circle — whose centre lies a
    quarter turn to its left:

        C = P_in + r R90(v_hat),   R90(px, pz) = (-pz, px)

    Writing the point at swept angle ``phi`` as ``C + r (cos a, sin a)`` with
    ``a = phi0 + phi`` and ``(cos phi0, sin phi0) = (pz, -px)/p_perp``, the map is
    the ``phi`` at which that point meets the **exit face** — the line through the
    design exit point normal to the design direction there — read out in the exit
    local frame. ``py`` is untouched (the field is vertical) and ``y`` advances by
    ``py phi / h``, the projected arc divided by the projected momentum.

    This is a derivation, not a transcription: nothing here comes from
    :func:`accsim.elements.dipole.exact_sector_bend_map`, whose form is rearranged so
    that no two numbers of size one are ever subtracted.
    """
    x, px, y, py, delta = state
    h = angle / length
    rho = 1 / h
    one = 1 + delta
    p_perp = sqrt(one**2 - py**2)
    pz = sqrt(one**2 - px**2 - py**2)
    r = p_perp / h

    centre_x, centre_z = x - pz / h, px / h
    cos0, sin0 = pz / p_perp, -px / p_perp

    cos_t, sin_t = cos(angle), sin(angle)
    exit_x, exit_z = -rho * (1 - cos_t), rho * sin_t
    dir_x, dir_z = -sin_t, cos_t  # design direction at the exit
    loc_x, loc_z = cos_t, sin_t  # local transverse unit vector at the exit

    def _angles(phi):
        return (
            cos0 * cos(phi) - sin0 * sin(phi),
            sin0 * cos(phi) + cos0 * sin(phi),
        )

    def _face(phi):
        ca, sa = _angles(phi)
        return (centre_x + r * ca - exit_x) * dir_x + (centre_z + r * sa - exit_z) * dir_z

    # mpmath compares |f|^2 against tol, so the exponent is halved in effect: this asks
    # for a residual near 10^-dps, i.e. full working precision, not the impossible
    # 10^-2dps the literal reads as. Written explicitly rather than left as an accident.
    phi = findroot(_face, angle, tol=mpf(10) ** (-2 * (mp.dps - 5)))
    ca, sa = _angles(phi)
    pos_x, pos_z = centre_x + r * ca, centre_z + r * sa

    return [
        (pos_x - exit_x) * loc_x + (pos_z - exit_z) * loc_z,
        p_perp * (-sa * loc_x + ca * loc_z),
        y + py * phi / h,
        py,
        delta,
    ]


def turn_map(state: list, *, exact_drift: bool, angle: float = ANG, kick: float = 0.0) -> list:
    """One turn of the minimal ring. ``angle`` is swept by the scaling-law gate.

    ``kick`` adds a thin steerer at the entrance — a constant, momentum-independent
    ``px`` deflection. It is zero for everything M2 does, and non-zero for M3's gate on
    *when* the drift model reaches the closed orbit: with it the on-momentum orbit no
    longer runs down the axis, and the two drift models stop agreeing at second order.
    """
    if kick:
        state = list(state)
        state[1] = state[1] + mpf(kick)
    state = thin_quad_map(state, mpf(KF))
    state = drift_map(state, mpf(LD), exact=exact_drift)
    # angle = 0 replaces the bend by a drift of the same length, in the *same* drift
    # model, so the sweep's zero-angle end is a ring with no bend at all rather than a
    # ring with one exact drift buried in it.
    if angle:
        state = bend_map(state, mpf(LB), mpf(angle))
    else:
        state = drift_map(state, mpf(LB), exact=exact_drift)
    state = thin_quad_map(state, -mpf(KF))
    return drift_map(state, mpf(LD), exact=exact_drift)


# ---------------------------------------------------------------------------
# closed orbit, tunes, and the second difference
# ---------------------------------------------------------------------------


def closed_orbit(delta, *, exact_drift: bool, angle: float = ANG, kick: float = 0.0) -> tuple:
    """``(x, px)`` of the 4D fixed point at fixed ``delta`` — ``y = py = 0`` by symmetry."""

    def residual(x, px):
        out = turn_map(
            [x, px, mpf(0), mpf(0), delta], exact_drift=exact_drift, angle=angle, kick=kick
        )
        return out[0] - x, out[1] - px

    root = findroot(residual, (mpf(0), mpf(0)), tol=mpf(10) ** (-2 * (mp.dps - 5)))
    return root[0], root[1]


def one_turn_traces(delta, *, exact_drift: bool, angle: float = ANG) -> dict[str, object]:
    """Trace of each plane's ``2x2`` one-turn Jacobian about the closed orbit."""
    xco, pco = closed_orbit(delta, exact_drift=exact_drift, angle=angle)
    base = [xco, pco, mpf(0), mpf(0), delta]
    step = mpf(10) ** (-mp.dps // 3)
    out = {}
    for plane, idx in (("x", (0, 1)), ("y", (2, 3))):
        block = matrix(2, 2)
        for j, col in enumerate(idx):
            plus, minus = list(base), list(base)
            plus[col] += step
            minus[col] -= step
            fwd = turn_map(plus, exact_drift=exact_drift, angle=angle)
            bwd = turn_map(minus, exact_drift=exact_drift, angle=angle)
            for i, row in enumerate(idx):
                block[i, j] = (fwd[row] - bwd[row]) / (2 * step)
        out[plane] = block[0, 0] + block[1, 1]
    return out


def tunes(delta, *, exact_drift: bool, angle: float = ANG) -> dict[str, object]:
    """Fractional tunes from the trace. Both stay well inside ``(0, 1/2)`` on this ring."""
    traces = one_turn_traces(delta, exact_drift=exact_drift, angle=angle)
    return {p: acos(t / 2) / (2 * pi) for p, t in traces.items()}


@functools.lru_cache(maxsize=32)
def second_order_chromaticity(*, exact_drift: bool, angle: float = ANG) -> dict[str, float]:
    """``d^2Q/ddelta^2`` of the minimal ring, to twenty digits, without accsim."""
    with mp.workdps(DPS):
        plus = tunes(+STEP, exact_drift=exact_drift, angle=angle)
        zero = tunes(mpf(0), exact_drift=exact_drift, angle=angle)
        minus = tunes(-STEP, exact_drift=exact_drift, angle=angle)
        return {p: float((plus[p] - 2 * zero[p] + minus[p]) / STEP**2) for p in ("x", "y")}


@functools.lru_cache(maxsize=4)
def design_tunes(*, exact_drift: bool = True) -> dict[str, float]:
    """On-momentum fractional tunes, for the stability/half-integer guard."""
    with mp.workdps(DPS):
        return {p: float(v) for p, v in tunes(mpf(0), exact_drift=exact_drift).items()}


@functools.lru_cache(maxsize=4)
def design_traces(*, exact_drift: bool = True) -> dict[str, float]:
    """On-momentum ``Tr M`` per plane, for the stability guard."""
    with mp.workdps(DPS):
        return {p: float(v) for p, v in one_turn_traces(mpf(0), exact_drift=exact_drift).items()}


# ---------------------------------------------------------------------------
# M3: the same ring's dispersion orders, derived the same way
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def dispersion_orders(
    *, exact_drift: bool, angle: float = ANG, kick: float = 0.0
) -> dict[str, float]:
    """First and second ``delta``-derivatives of the closed orbit, to twenty digits.

    ``D_x = dx/ddelta``, ``dd_x = d^2x/ddelta^2`` and the same for ``px``. Built from
    the same sixty-digit fixed point :func:`second_order_chromaticity` differentiates,
    which is what makes M3's gate an extension of M2's rather than a new claim.

    Both drift models are offered for symmetry with everything else here. At
    ``kick = 0`` they return the **same** numbers to ``1e-15``: with the on-momentum
    orbit on the axis, the exact and paraxial drifts place the closed orbit differently
    only at ``O(delta^3)`` (:func:`drift_model_orbit_split`), and a symmetric second
    difference cannot see an odd function. Turn the steerer on and that stops being
    true — see :func:`dispersion_drift_model_split`, and M3 in ``docs/ROADMAP.md``.

    ``px_on_momentum`` is the on-momentum closed orbit's ``px``, which is the quantity
    the split is proportional to, reported so a gate can assert the *reason* rather than
    the coincidence.
    """
    with mp.workdps(DPS):
        plus = closed_orbit(+STEP, exact_drift=exact_drift, angle=angle, kick=kick)
        zero = closed_orbit(mpf(0), exact_drift=exact_drift, angle=angle, kick=kick)
        minus = closed_orbit(-STEP, exact_drift=exact_drift, angle=angle, kick=kick)
        out = {"px_on_momentum": float(zero[1])}
        for name, i in (("x", 0), ("px", 1)):
            out[f"D_{name}"] = float((plus[i] - minus[i]) / (2 * STEP))
            out[f"dd_{name}"] = float((plus[i] - 2 * zero[i] + minus[i]) / STEP**2)
        return out


@functools.lru_cache(maxsize=32)
def drift_model_orbit_split(delta_exp: int, *, angle: float = ANG) -> dict[str, float]:
    """``(exact - paraxial)`` closed orbit at ``delta = 10^delta_exp``, and over ``delta^3``.

    The ratio is what carries the finding: it is the *same* number over three decades
    of ``delta``, so the split is a pure cubic and contributes nothing to a second
    derivative.
    """
    with mp.workdps(DPS):
        d = mpf(10) ** delta_exp
        xe, pe = closed_orbit(d, exact_drift=True, angle=angle)
        xp, pp = closed_orbit(d, exact_drift=False, angle=angle)
        return {
            "dx": float(xe - xp),
            "dpx": float(pe - pp),
            "dx_over_delta3": float((xe - xp) / d**3),
            "dpx_over_delta3": float((pe - pp) / d**3),
        }


@functools.lru_cache(maxsize=64)
def dispersion_drift_model_split(*, kick: float = 0.0, angle: float = ANG) -> dict[str, float]:
    r"""``(exact - paraxial)`` second-order dispersion, as a function of the steering.

    The bound on M3's headline. The exact drift exceeds the paraxial one by
    ``L px (px^2 + py^2)/(2 (1+delta)^3)``; writing ``px = a + b delta`` on the closed
    orbit, its ``delta^2`` coefficient is ``3 a b^2`` (in the flat, ``py = 0`` case).
    So the split into ``d^2x/ddelta^2`` is **first order in the on-momentum orbit angle
    ``a``** and **second order in the dispersion angle ``b``** — and it vanishes
    identically when *either* is zero. ``a = 0`` is what every ring M2 and M3 measured
    the agreement on: they close on the axis. ``b = 0`` is a ring with no bend.

    Returns the split in both orders plus the on-momentum ``px`` it is proportional to.
    """
    exact = dispersion_orders(exact_drift=True, angle=angle, kick=kick)
    paraxial = dispersion_orders(exact_drift=False, angle=angle, kick=kick)
    return {
        "dd_x": exact["dd_x"] - paraxial["dd_x"],
        "dd_x_relative": abs((exact["dd_x"] - paraxial["dd_x"]) / exact["dd_x"]),
        "D_x": exact["D_x"] - paraxial["D_x"],
        "D_x_relative": abs((exact["D_x"] - paraxial["D_x"]) / exact["D_x"]),
        "px_on_momentum": exact["px_on_momentum"],
    }
