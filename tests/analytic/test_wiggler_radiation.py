r"""Analytic gates for the wiggler's radiation integrals (T2) — the milestone it exists for.

T1 shipped the magnet's *map*. This is what the magnet is **for**: a wiggler is built to
radiate, and until now :func:`~accsim.radiation.radiation_integrals` keyed on ``Dipole``
alone, so a ring with one in it reported the damping of a ring without one.

Ordered by how much each can catch:

  * **The staircase separation, with its direction.** Against a hard-edge stack matched to
    the *same* ``I2`` and with no net bend, the wiggler's ``I3`` is higher by exactly
    ``8 sqrt(2)/(3 pi) = 1.2004217548761416`` — a **ratio**, so a uniformly mis-scaled
    field cannot pass it, and the bound is real (the two closed forms land one ulp apart).
    A sinusoid spends more of its length near peak curvature than a square wave of the same
    mean square does, and ``I3`` is the odd power that notices.
  * **The two closed forms**, ``I2 = h0^2 L/2`` and ``I3 = 4 h0^3 L/(3 pi)``, from
    ``<cos^2> = 1/2`` and ``<|cos|^3> = 4/(3 pi)`` — both derived in sympy below rather
    than recalled.
  * **A wiggler's own dispersion, which is the finding.** The roadmap entry said the
    wiggler's contribution to momentum compaction is *geometric* — its wiggle path
    shortening with momentum, living in ``R56`` — and "appears in no lattice integral,
    because a wiggler generates no dispersion". Both halves of that are wrong, and in a way
    that is measurable to eleven digits: the wiggler generates its **own** dispersion
    ``eta = (h0/k^2)(1 - cos k s)``, and ``int eta h ds = -L theta^2/2`` is *exactly* the
    ``R56`` term T1 shipped. The two mechanisms are one mechanism seen from two sides, and
    the gate is that ``I1 == alpha_c * C`` — a contract
    :func:`~accsim.radiation.radiation_integrals` has always claimed — holds on a ring with
    a wiggler in it to the same residual as on one without.
  * **The ring's dispersion contributes exactly zero to ``I1`` and ``I4``**, gated by
    *invariance*: the same wiggler in rings with different ``D_x`` at it contributes the
    same number. That is the entry's "``I4`` is the control" in the only form that can
    fail, and it is what licenses the ``I3`` ratio above to be attributed to ``I3`` alone.
  * **``I5`` factorises exactly**, for two reasons that must hold together: curly-H is a
    *drift* invariant and a wiggler is a drift horizontally, and the constant and linear
    moments of ``|cos|^3`` both vanish. Neither alone is enough.
  * **The end-to-end consequence**, ``sigma_delta``, gated twice: as an exact algebraic
    identity at any ring strength, and as the ``9.56%`` the entry pre-committed, in the
    limit where the wiggler is the ring's only meaningful radiator.
  * **The refusals that lifted**: a wiggler is now a powered magnet ``taper()`` can scale,
    and its focusing scales as the **square** of the factor. What did not lift is
    ``radiation_kick``, and ``taper()`` on a whole ring still refuses through it — measured,
    not predicted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Wiggler,
    radiation_integrals,
    taper,
)
from accsim.radiation import (
    _curly_h,
    _wiggler_own_curly_h,
    damping_partition_numbers,
    damping_times,
    equilibrium_energy_spread,
)
from accsim.twiss import (
    _blocks,
    _dispersive_kick,
    _propagate_block,
    _transverse_4d,
    closed_twiss,
    momentum_compaction,
)

# The probe machine of the axis-T entry: B0 = 1.5 T, lambda_w = 0.1 m, N = 10, 1 GeV e-.
PERIOD = 0.1
PERIODS = 10
H0 = 0.449689
STAIRCASE_RATIO = 8.0 * math.sqrt(2.0) / (3.0 * math.pi)


@pytest.fixture
def ref() -> ReferenceParticle:
    """1 GeV electrons — the probe beam of the roadmap's axis-T entry."""
    return ReferenceParticle.from_total_energy(0.51099895069e6, 1.0e9, charge=-1.0)


@pytest.fixture
def wig() -> Wiggler:
    """The probe wiggler itself."""
    return Wiggler(PERIOD, H0, PERIODS, "w")


def _staircase() -> list[Dipole]:
    """The comparator: ``2*PERIODS`` hard-edge poles, same ``I2``, **no net bend**.

    Matched on ``I2`` means ``|h| = h0/sqrt(2)``, since ``<cos^2> = 1/2``. The poles
    alternate, so the stack bends the ring by exactly nothing and can be swapped for the
    wiggler without moving the rest of the optics — which a *single* dipole of the same
    ``I2`` emphatically cannot (it is a 0.32 rad bend), and that is why this is a stack.
    """
    hs, half = H0 / math.sqrt(2.0), PERIOD / 2.0
    return [Dipole(half, (hs if j % 2 == 0 else -hs) * half) for j in range(2 * PERIODS)]


def _ring(ref: ReferenceParticle, middle: object, bend: float = 0.3) -> Lattice:
    """A stable FODO cell with a bend in it and ``middle`` in the straight."""
    mid = middle if isinstance(middle, list) else [middle]
    return Lattice(
        [
            Quadrupole(0.5, 1.2, "qf"),
            Dipole(2.0, bend),
            Quadrupole(0.5, -1.2, "qd"),
            Drift(0.5),
            *mid,  # type: ignore[list-item]
            Drift(0.5),
        ],
        ref,
    )


def _contribution(ref: ReferenceParticle, middle: object, bend: float = 0.3) -> dict[str, float]:
    """What ``middle`` adds to each integral, against the same ring with a drift there."""
    items = middle if isinstance(middle, list) else [middle]
    length = sum(e.length for e in items)  # type: ignore[attr-defined]
    with_it = radiation_integrals(_ring(ref, middle, bend))
    without = radiation_integrals(_ring(ref, Drift(length), bend))
    return {n: getattr(with_it, n) - getattr(without, n) for n in ("i1", "i2", "i3", "i4", "i5")}


def _optics_at_the_wiggler(
    ref: ReferenceParticle, ring: Lattice
) -> tuple[float, float, np.ndarray]:
    """``(beta_x, alpha_x, dispersion)`` at the entrance face of the ring's wiggler.

    The same running state :func:`~accsim.radiation.radiation_integrals` has when it reaches
    the element, rebuilt here so the brute-force checks below start from the same place the
    shipped code does rather than from an assumed one.
    """
    tw = closed_twiss(ring)
    bx, ax = tw.beta_x, tw.alpha_x
    disp = np.array([tw.disp_x, tw.disp_px, tw.disp_y, tw.disp_py])
    for elem in ring.elements:
        if isinstance(elem, Wiggler):
            return bx, ax, disp
        m = elem.matrix(ref)
        bx, ax, _ = _propagate_block(_blocks(m)[0], bx, ax)
        disp = _transverse_4d(m) @ disp + _dispersive_kick(m)
    raise AssertionError("no wiggler in this ring")


# --- the coefficients, derived rather than recalled --------------------------------------
def test_the_period_averages_are_derived_in_sympy_not_recalled() -> None:
    r"""``<cos^2> = 1/2``, ``<|cos|^3> = 4/(3 pi)``, and the staircase ratio.

    The project rule, in the place it matters most: ``4/(3 pi)`` is exactly the kind of
    constant that is remembered wrong (``2/pi``, ``4/(3 pi)`` and ``8/(3 pi)`` are all
    plausible-looking), and every number this milestone ships rests on it.

    The ratio is the *whole* discriminating gate, so it is derived end to end — from the two
    averages, through the staircase matched on ``I2``, to the number — rather than written
    down. The last two lines check the bound is not vacuous, which P2 (i) requires: the two
    closed forms evaluate one ulp apart, so ``1e-14`` sits two orders above round-off.
    """
    s, k, h0, ell = sp.symbols("s k h0 L", positive=True)
    period = 2 * sp.pi / k

    avg_c2 = sp.integrate(sp.cos(k * s) ** 2, (s, 0, period)) / period
    avg_c3 = sp.integrate(sp.Abs(sp.cos(k * s)) ** 3, (s, 0, period)) / period
    assert sp.simplify(avg_c2 - sp.Rational(1, 2)) == 0
    assert sp.simplify(avg_c3 - 4 / (3 * sp.pi)) == 0

    i2 = sp.simplify(h0**2 * ell * avg_c2)
    i3 = sp.simplify(h0**3 * ell * avg_c3)
    assert sp.simplify(i2 - h0**2 * ell / 2) == 0
    assert sp.simplify(i3 - 4 * h0**3 * ell / (3 * sp.pi)) == 0

    # The staircase: constant |h| over the same length, matched on I2 => |h| = h0/sqrt2.
    hs = sp.Symbol("hs", positive=True)
    h_stair = sp.solve(sp.Eq(hs**2 * ell, i2), hs)[0]
    assert sp.simplify(h_stair - h0 / sp.sqrt(2)) == 0
    ratio = sp.simplify(i3 / (h_stair**3 * ell))
    assert sp.simplify(ratio - 8 * sp.sqrt(2) / (3 * sp.pi)) == 0

    # The bound is real rather than vacuous, which P2 (i) requires be checked: evaluated to
    # 40 digits and rounded, the ratio is within one ulp of the float the gates compare
    # against, so ``1e-14`` sits two orders above round-off and can actually fail. (An
    # assertion that the constant equals its own literal would check nothing; this one
    # checks the literal against the symbolic value it claims to be.)
    assert abs(float(sp.N(ratio, 40)) - STAIRCASE_RATIO) <= 2.3e-16


# --- gates 1 and 8: the dispatch opened, and nothing else moved --------------------------
def test_the_integrals_stop_keying_on_dipole_alone(ref: ReferenceParticle, wig: Wiggler) -> None:
    """The milestone in one line: a wiggler is no longer invisible here.

    This is T1's ``test_the_wiggler_contributes_exactly_nothing_to_the_radiation_integrals``
    turned around. That test asserted the silent zero and said in its own docstring that it
    was written to fail the day T2 landed; this is that day, and it has been rewritten
    rather than debugged.
    """
    got = _contribution(ref, wig)
    assert got["i2"] > 0.0
    assert got["i3"] > 0.0
    assert got["i5"] > 0.0
    # ...and it is not small: the wiggler's own i2 is more than twice the dipole's, so the
    # silent zero was a factor-of-three error in the energy loss of this ring.
    assert got["i2"] > 2.0 * radiation_integrals(_ring(ref, Drift(1.0))).i2


def test_a_wiggler_at_zero_field_is_bit_identical_to_a_drift(ref: ReferenceParticle) -> None:
    """Gate 8, in its strongest local form: the new branch collapses onto the oldest one.

    Not a claim about the rest of the suite (the suite itself is that claim), but the one
    place a new ``isinstance`` branch could plausibly have changed an existing answer: an
    element that is a drift in every entry must give drift answers *bit for bit*, including
    through the ``i5`` path, which is the only one that reads the running optics.
    """
    off = _contribution(ref, Wiggler(PERIOD, 0.0, PERIODS, "off"))
    assert off == {"i1": 0.0, "i2": 0.0, "i3": 0.0, "i4": 0.0, "i5": 0.0}


# --- gate 2: the closed forms ------------------------------------------------------------
def test_i2_and_i3_are_the_closed_forms_at_machine_precision(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    """``I2 = h0^2 L/2`` and ``I3 = 4 h0^3 L/(3 pi)``.

    Both are pure geometry — no dispersion, no optics, nothing the rest of the ring can
    perturb — so the only tolerance they deserve is the one float arithmetic imposes, and
    they must not move when the ring around them is refocused.
    """
    got = _contribution(ref, wig)
    assert got["i2"] == pytest.approx(0.5 * H0**2 * wig.length, rel=1e-15)
    assert got["i3"] == pytest.approx(4.0 * H0**3 * wig.length / (3.0 * math.pi), rel=1e-15)

    for kq in (0.9, 1.5):
        alt = Lattice(
            [Quadrupole(0.5, kq), Dipole(2.0, 0.3), wig, Quadrupole(0.5, -kq), Drift(1.0)],
            ref,
        )
        base = Lattice(
            [
                Quadrupole(0.5, kq),
                Dipole(2.0, 0.3),
                Drift(wig.length),
                Quadrupole(0.5, -kq),
                Drift(1.0),
            ],
            ref,
        )
        delta = radiation_integrals(alt).i2 - radiation_integrals(base).i2
        assert delta == pytest.approx(0.5 * H0**2 * wig.length, rel=1e-14)


# --- gate 3: the discriminating gate -----------------------------------------------------
def test_the_wiggler_beats_a_matched_staircase_in_i3_by_exactly_8_sqrt2_over_3pi(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""The one gate a uniformly mis-scaled field cannot pass, because it is a **ratio**.

    Matched on ``I2`` — checked here, not assumed — the two differ only in *how* the
    curvature is distributed along the magnet, and ``I3 = int |h|^3`` is the odd power that
    notices: a sinusoid dwells near its peak, a square wave never exceeds its mean square.
    Scale ``h0`` by any factor and both ``I3`` move together; the ratio does not, which is
    the second half of this test.

    Asserted at ``1e-14``, two orders above the one-ulp separation of the closed forms.
    """
    w, s = _contribution(ref, wig), _contribution(ref, _staircase())
    assert w["i2"] == pytest.approx(s["i2"], rel=1e-14)  # the premise
    assert w["i3"] / s["i3"] == pytest.approx(STAIRCASE_RATIO, rel=1e-14)
    assert w["i3"] > s["i3"]  # the direction, stated separately from the size

    for factor in (0.5, 2.0):
        ws = _contribution(ref, Wiggler(PERIOD, factor * H0, PERIODS))
        ss = _contribution(ref, [Dipole(d.length, factor * d.angle) for d in _staircase()])
        assert ws["i3"] != pytest.approx(w["i3"], rel=1e-3)  # both really moved
        assert ws["i3"] / ss["i3"] == pytest.approx(STAIRCASE_RATIO, rel=1e-14)


# --- gate 4, rewritten: the ring's dispersion is the control ------------------------------
def test_the_rings_dispersion_contributes_exactly_zero_to_i1_and_i4(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""Gated by **invariance**, the only form of "exactly zero" that can fail here.

    The roadmap gated ``I4`` as an exact zero. That is right about the *ring's* dispersion
    and wrong about the total, so the assertion moves to the statement that survives: ``h``
    and ``h^3`` are odd about each half period and the incoming ``D_x`` is linear across the
    body (horizontally a wiggler is a drift), so the ring's dispersion cancels **whatever it
    is**. Put the same wiggler in rings focused three different ways — genuinely different
    ``D_x`` and ``D_x'`` at the magnet, checked — and the contribution does not move.

    An omitted term would satisfy that too, so the *value* it does not move from is gated in
    the next test. The two together are gate 4.
    """
    seen, optics = [], []
    for kq in (0.9, 1.2, 1.5):
        ring = Lattice(
            [
                Quadrupole(0.5, kq, "qf"),
                Dipole(2.0, 0.3),
                Quadrupole(0.5, -kq, "qd"),
                Drift(0.5),
                wig,
                Drift(0.5),
            ],
            ref,
        )
        base = Lattice(
            [
                Quadrupole(0.5, kq, "qf"),
                Dipole(2.0, 0.3),
                Quadrupole(0.5, -kq, "qd"),
                Drift(0.5),
                Drift(wig.length),
                Drift(0.5),
            ],
            ref,
        )
        a, b = radiation_integrals(ring), radiation_integrals(base)
        seen.append((a.i1 - b.i1, a.i4 - b.i4))
        bx, ax, disp = _optics_at_the_wiggler(ref, ring)
        optics.append((bx, disp[0], disp[1]))

    # the premise: the three rings really do present different optics to the same magnet
    assert optics[0][1] != pytest.approx(optics[1][1], rel=1e-3)
    assert optics[1][1] != pytest.approx(optics[2][1], rel=1e-3)

    for i1, i4 in seen[1:]:
        assert i1 == pytest.approx(seen[0][0], rel=1e-11)
        assert i4 == pytest.approx(seen[0][1], rel=1e-13)


def test_the_wigglers_own_dispersion_is_what_is_left_and_it_has_a_closed_form(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""``int eta h ds = -L theta^2/2`` and ``int eta h^3 ds = -(3/8) L h0^2 theta^2``.

    The wiggler's own dispersion solves ``eta'' = h`` with ``eta(0) = eta'(0) = 0`` — the
    element starts at a field *maximum*, which is exactly what makes the reference orbit
    close over an integer number of periods, and which also makes the excursion **one
    sided**: a plain wiggler has no half-strength end poles to centre it, so ``eta >= 0``
    throughout and its mean is ``theta/k``, not zero. That one-sidedness is why the ``I5``
    cross term below exists at all.

    Both integrals are even in ``h0``, so neither cares about the field polarity or the sign
    of the charge — S1's charge-free gate in the form this element takes it.
    """
    ell, th = wig.length, wig.deflection
    got = _contribution(ref, wig)
    assert got["i1"] == pytest.approx(-0.5 * ell * th**2, rel=1e-11)
    assert got["i4"] == pytest.approx(-0.375 * ell * H0**2 * th**2, rel=1e-12)

    # derived, not recalled -- and the derivation is what fixes the sign
    s, k, h0 = sp.symbols("s k h0", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    eta = h0 / k**2 * (1 - sp.cos(k * s))
    assert sp.simplify(sp.diff(eta, s, 2) - h0 * sp.cos(k * s)) == 0  # eta'' = h
    span = 2 * sp.pi * n / k
    assert sp.simplify(eta.subs(s, span)) == 0  # and it closes...
    assert sp.simplify(sp.diff(eta, s).subs(s, span)) == 0  # ...in both coordinates
    theta = h0 / k
    i1 = sp.integrate(eta * h0 * sp.cos(k * s), (s, 0, span))
    i4 = sp.integrate(eta * (h0 * sp.cos(k * s)) ** 3, (s, 0, span))
    assert sp.simplify(i1 + span * theta**2 / 2) == 0
    assert sp.simplify(i4 + sp.Rational(3, 8) * span * h0**2 * theta**2) == 0

    flipped = _contribution(ref, Wiggler(PERIOD, -H0, PERIODS))
    assert flipped["i1"] == got["i1"]
    assert flipped["i4"] == got["i4"]


def test_the_geometric_r56_term_and_the_i1_term_are_one_mechanism(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""The finding, gated by a contract the module has always claimed: ``I1 == alpha_c * C``.

    T1 shipped ``R56 = L/gamma0^2 + (L theta^2/4)(2 + 1/gamma0^2)`` and called the second
    term *geometric* — the wiggle path shortening with momentum — noting explicitly that it
    "appears in no lattice integral". It does. Its ``gamma``-free part, ``L theta^2/2``, is
    exactly ``-int eta h ds``, and the ``(L theta^2/4)/gamma0^2`` left over is not compaction
    at all but the *velocity* slip along the extra path length.

    The two routes are as disjoint as they get: ``alpha_c`` by the identity route is read off
    the one-turn **longitudinal row** (T1's ``R56``), while ``I1`` is a dispersion integral
    over the transverse plane. Before this milestone they disagreed by ``2.56e-05`` on this
    ring — 100% of the wiggler's own term — and the disagreement was invisible because
    nothing had ever compared them on a lattice with a wiggler in it.
    """
    ring, base = _ring(ref, wig), _ring(ref, Drift(wig.length))

    def identity_break(lat: Lattice) -> float:
        return (
            radiation_integrals(lat).i1 - momentum_compaction(lat, method="identity") * lat.length
        )

    # the ring's own residual is the dipole quadrature's; the wiggler now adds nothing to it
    assert identity_break(ring) == pytest.approx(identity_break(base), rel=1e-5)
    assert abs(identity_break(ring) - identity_break(base)) < 1e-11

    # the size of what was repaired, stated rather than implied
    ell, th, g2 = wig.length, wig.deflection, ref.gamma0**2
    r56_wiggle = 0.25 * ell * th**2 * (2.0 + 1.0 / g2)
    assert r56_wiggle == pytest.approx(0.5 * ell * th**2, rel=1e-6)  # gamma-free to 1.3e-7
    assert radiation_integrals(base).i1 - radiation_integrals(ring).i1 == pytest.approx(
        0.5 * ell * th**2, rel=1e-11
    )

    # and the two *routes* to alpha_c agree again, which they did not before T2
    def route_gap(lat: Lattice) -> float:
        return (
            momentum_compaction(lat, method="quadrature")
            / momentum_compaction(lat, method="identity")
            - 1.0
        )

    assert route_gap(ring) == pytest.approx(route_gap(base), rel=1e-3)


# --- I5: the gap the roadmap entry never named -------------------------------------------
def test_i5_factorises_exactly_and_needs_two_facts_to_do_so(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""``int |h|^3 curlyH ds = <|cos|^3> |h0|^3 curlyH L`` for the **ring's** dispersion.

    Two independent facts, and neither alone is enough. (a) curly-H is a *drift* invariant
    and a wiggler is a drift horizontally, so curlyH is constant across the body — checked
    here to round-off, entrance against exit. (b) The constant *and linear* moments of
    ``|cos|^3`` about a period both vanish, so even a linear curlyH would factorise; it is
    the **quadratic** moment that does not, and curlyH has no quadratic content under a
    drift.

    Had (a) failed, ``I5`` would need the ``slices`` sub-stepping the dipole branch uses. It
    does not, which is why ``slices`` never enters the wiggler branch at all.
    """
    ring = _ring(ref, wig)
    bx, ax, disp = _optics_at_the_wiggler(ref, ring)
    entry = _curly_h(bx, ax, disp[0], disp[1])

    dm = Drift(wig.length).matrix(ref)
    b1, a1, _ = _propagate_block(_blocks(dm)[0], bx, ax)
    d1 = _transverse_4d(dm) @ disp
    assert b1 != pytest.approx(bx, rel=1e-3)  # the optics really did move across the body
    assert _curly_h(b1, a1, d1[0], d1[1]) == pytest.approx(entry, rel=1e-14)  # (a)

    s, k = sp.symbols("s k", positive=True)
    period, weight = 2 * sp.pi / k, sp.Abs(sp.cos(k * s)) ** 3
    avg = 4 / (3 * sp.pi)

    def moment_gap(power: int) -> sp.Expr:
        return sp.simplify(
            sp.integrate((weight * s**power).rewrite(sp.Piecewise), (s, 0, period))
            - avg * sp.integrate(s**power, (s, 0, period))
        )

    assert moment_gap(0) == 0  # (b)
    assert moment_gap(1) == 0
    assert moment_gap(2) != 0  # and the reason (a) is load-bearing


def test_the_own_dispersion_part_of_i5_matches_a_resolved_quadrature(
    ref: ReferenceParticle, wig: Wiggler
) -> None:
    r"""The one piece of ``I5`` that does **not** factorise, against a brute-force integral.

    ``eta`` correlates with ``|h|^3`` by construction — that is what "own dispersion" means
    here — so the shipped value is a per-period closed form summed with the optics
    drift-transported between periods. Its arbiter is the same integral by trapezoid with the
    ``cos`` fully resolved, and the statement is that the brute force **converges onto** the
    shipped value rather than sitting at a fixed distance from it. That is what separates a
    correct closed form from a plausible one; a tolerance alone could not.
    """
    ring = _ring(ref, wig)
    bx, ax, disp = _optics_at_the_wiggler(ref, ring)
    k, h0, ell = wig.wavenumber, wig.h0, wig.length

    def brute(n: int, *, own: bool) -> float:
        ds = ell / n
        dm = Drift(ds).matrix(ref)
        xb, d4 = _blocks(dm)[0], _transverse_4d(dm)
        b, a, d = bx, ax, disp.copy()
        total = 0.0
        for i in range(n + 1):
            s = i * ds
            eta = (h0 / k**2) * (1.0 - math.cos(k * s)) if own else 0.0
            etap = (h0 / k) * math.sin(k * s) if own else 0.0
            total += (
                (0.5 if i in (0, n) else 1.0)
                * abs(math.cos(k * s)) ** 3
                * _curly_h(b, a, d[0] + eta, d[1] + etap)
            )
            if i < n:
                d = d4 @ d
                b, a, _ = _propagate_block(xb, b, a)
        return abs(h0) ** 3 * total * ds

    shipped = abs(h0) ** 3 * _wiggler_own_curly_h(wig, ref, bx, ax, disp)
    errs = [
        abs(brute(n, own=True) - brute(n, own=False) - shipped) / abs(shipped)
        for n in (10_000, 40_000)
    ]
    assert errs[0] < 1e-5
    assert errs[1] < errs[0] / 5.0  # converging onto it, not merely near it

    # it is a real term rather than a rounding artefact of the ring part, and it is small
    # *here* only because this ring has dispersion at the wiggler -- in the dispersion-free
    # straight a damping wiggler actually lives in, it is the whole of I5.
    ring_part = 4.0 / (3.0 * math.pi) * abs(h0) ** 3 * _curly_h(bx, ax, disp[0], disp[1]) * ell
    assert shipped / ring_part == pytest.approx(9.39e-4, rel=2e-2)
    assert _contribution(ref, wig)["i5"] == pytest.approx(ring_part + shipped, rel=1e-11)


# --- gate 5: the end-to-end statement, which is what the milestone exists for -------------
def test_a_damping_wiggler_damps_all_three_planes(ref: ReferenceParticle, wig: Wiggler) -> None:
    r"""The rates ``J_x I2 = I2 - I4``, ``J_y I2 = I2``, ``J_z I2 = 2 I2 + I4`` all rise.

    The gate is on the **rates**, not on the partition numbers, and the roadmap entry is
    right about why: ``J_x = 1 - I4/I2`` is *not* unchanged — it moves toward 1 as ``I2``
    rises, measurably (``0.906 -> 0.971`` here) — while the product that actually sets the
    damping moves the right way regardless. All three damping times shorten, by more than a
    factor of three on this ring.

    ``J_y`` stays exactly 1. A wiggler adds a vertical **gradient**, which is what
    :func:`~accsim.radiation.damping_partition_numbers` warns about, but not vertical
    *bending*: the partitions are set by ``I4``, which is weighted by horizontal dispersion,
    and a focusing block contributes to neither.
    """
    with_w, without = _ring(ref, wig), _ring(ref, Drift(wig.length))
    tw, td = damping_times(with_w), damping_times(without)
    for damped, undamped in zip(tw, td, strict=True):
        assert damped < undamped
    assert min(b / a for a, b in zip(tw, td, strict=True)) > 3.0

    jw, jd = damping_partition_numbers(with_w), damping_partition_numbers(without)
    assert jd[0] < jw[0] < 1.0  # J_x moved toward 1 -- the correction, stated
    assert jw[1] == 1.0 == jd[1]
    assert sum(jw) == pytest.approx(4.0, rel=1e-14)  # Robinson still holds

    iw, iw0 = radiation_integrals(with_w), radiation_integrals(without)
    for rate_w, rate_d in (
        (iw.i2 - iw.i4, iw0.i2 - iw0.i4),
        (iw.i2, iw0.i2),
        (2.0 * iw.i2 + iw.i4, 2.0 * iw0.i2 + iw0.i4),
    ):
        assert rate_w > rate_d


def test_the_staircase_separation_reaches_sigma_delta(ref: ReferenceParticle, wig: Wiggler) -> None:
    r"""Gate 3's consequence, twice: as an exact identity, and as the entry's ``9.56%``.

    ``sigma_delta^2 = C_q gamma^2 I3 / (J_z I2)``, so the ratio of spreads between the two
    rings carries the ``I3`` ratio exactly once ``J_z`` and ``I2`` are divided out — an
    identity that holds at **any** ring strength, to round-off, and needs no limit.

    The bare ``9.56%`` the entry pre-committed is that identity in the limit where the
    wiggler is the ring's only meaningful radiator, and it is *approached* rather than hit,
    because the staircase's own dispersion gives it an ``I4`` the wiggler does not have —
    precisely the control the ``I4`` tests above establish. Weakening the ring's own bend
    from ``0.3`` to ``0.01`` rad walks the measured ratio ``1.0776 -> 1.0956``.
    """
    for bend in (0.3, 0.05, 0.01):
        rw, rs = _ring(ref, wig, bend), _ring(ref, _staircase(), bend)
        gw, gs = radiation_integrals(rw), radiation_integrals(rs)
        jz_w, jz_s = damping_partition_numbers(rw)[2], damping_partition_numbers(rs)[2]
        spread = equilibrium_energy_spread(rw) / equilibrium_energy_spread(rs)
        assert spread**2 * (jz_w / jz_s) * (gw.i2 / gs.i2) == pytest.approx(
            gw.i3 / gs.i3, rel=1e-14
        )

    weak = equilibrium_energy_spread(_ring(ref, wig, 0.01)) / equilibrium_energy_spread(
        _ring(ref, _staircase(), 0.01)
    )
    strong = equilibrium_energy_spread(_ring(ref, wig)) / equilibrium_energy_spread(
        _ring(ref, _staircase())
    )
    assert weak == pytest.approx(math.sqrt(STAIRCASE_RATIO), rel=5e-5)
    assert strong < weak < math.sqrt(STAIRCASE_RATIO)  # approached from below


# --- gates 6 and 7: what lifted, and what did not ----------------------------------------
def test_a_wiggler_is_a_powered_magnet_and_its_focusing_tapers_as_the_square(
    wig: Wiggler,
) -> None:
    """Gate 7, decided rather than discovered: ``h0`` is the strength a taper scales.

    Q2's statement is that a tapered magnet *is* the design magnet seen at a rescaled
    momentum, and for a wiggler that lands somewhere no other entry in
    ``tapering._STRENGTHS`` does: the focusing is ``h0^2/2``, so it scales as the **square**
    of the factor while every other magnet's optics is linear in it. T1 left this refused
    because nothing had decided it; ``_scaled`` no longer raises.
    """
    from accsim.tapering import _scaled

    for factor in (0.5, 1.001, 1.5):
        scaled = _scaled(wig, factor)
        assert isinstance(scaled, Wiggler)
        assert scaled.h0 == pytest.approx(factor * wig.h0, rel=1e-15)
        assert scaled.focusing / wig.focusing == pytest.approx(factor**2, rel=1e-14)
        # the geometry is untouched -- a taper changes the field, never the magnet
        assert (scaled.period, scaled.periods, scaled.length) == (
            wig.period,
            wig.periods,
            wig.length,
        )
        assert wig.h0 == H0  # and the original is not mutated


def test_tapering_a_whole_ring_still_refuses_and_now_for_one_reason(
    ref: ReferenceParticle,
) -> None:
    """What did **not** lift, and the measurement that says which refusal is left.

    T1's version of this test asserted *two* refusals: the field accessor (reached first,
    because ``taper`` builds the radiating closed orbit before it scales anything) and
    ``_scaled``'s own guard. The second is gone — a wiggler is a powered magnet now — so the
    refusal collapses to one, and it is the honest one: a taper needs the ring's radiating
    orbit, ``radiation_kick`` samples the field once per traversal, and a wiggler's field
    reverses twenty times inside itself. That is the gap T2 prices rather than papers over,
    and closing it is a milestone of its own (per-period tracking through the real field).

    Written to fail the day that lands.
    """
    from accsim.tapering import _scaled

    lattice = Lattice([Dipole(2.0, 0.3), Wiggler(PERIOD, H0, PERIODS, "w")], ref)
    with pytest.raises(NotImplementedError, match="no s-independent field"):
        taper(lattice)

    # the refusal that lifted, asserted as lifted rather than left implied
    assert _scaled(Wiggler(PERIOD, H0, PERIODS, "w"), 1.001).h0 == pytest.approx(
        1.001 * H0, rel=1e-15
    )
