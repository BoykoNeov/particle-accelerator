r"""P2 (iv) against xtrack: the quadrupole's kinematic term, by **tracking**.

Marked ``reference``: skips when xtrack or its JIT compiler is unavailable.

**What is being arbitrated.** There is no closed form for the exact quadrupole
Hamiltonian, so every code picks a splitting. accsim ships the *paraxial* Hamiltonian's
exact flow (L2) and, since P2 (iv), can interleave the piece that was dropped —

    ``H_kin = (1+delta) - sqrt((1+delta)^2 - px^2 - py^2) - (px^2+py^2)/(2(1+delta))``

— which depends on the momenta alone, so it has its own explicit flow. xtrack's
``model="drift-kick-drift-exact"`` splits the *same* exact Hamiltonian a different way:
exact drifts alternating with thin quadrupole kicks. Two different splittings of one
Hamiltonian converge to one map, and that map is what this file measures accsim against.

**The measured result.** With the reference refined until it stops moving, accsim's
default misses it by ``2.6e-10`` on a modest trajectory and ``2.9e-8`` on a steeper one;
``kinematic_slices=256`` closes both to ``5e-15`` and ``8e-14`` — the arithmetic floor for
coordinates of this size. In between the residual falls as ``1/n^2``, cleanly, over five
decades. That the ladder is accsim's own convergence and not a shared structure is what
makes this a cross-check: nothing about it collapses when ``n`` happens to match the
reference's kick count.

**Refining the reference is not optional, and the failure mode is silence.** With
``num_multipole_kicks`` left at its 7-kick minimum the reference is ``2.3e-6`` from its
own limit, and *every* accsim setting — flag off, flag on at any ``n`` — lands the same
``2.3e-6`` away. A coarse reference does not disagree loudly; it agrees with everything
equally, and a gate written against it would have passed before P2 (iv) existed and after.
``test_a_reference_too_coarse_to_see_the_term_agrees_with_everything`` pins that, and the
sweep behind the number chosen here is recorded in ``docs/CONVENTIONS.md``. P2 (ii) and
P2 (iii) both had to match the reference's integration before the comparison meant
anything; this is the third time.

**Two integrator details that bite.** ``integrator="yoshida4"`` rounds the kick count *up
to a multiple of seven* (a Yoshida step is seven kicks), so ``num_multipole_kicks`` of 1,
2, 4 and 7 are all the same map — the sweep below starts where it does for that reason.
And the reference's own error falls as ``1/N^4``, so it overtakes accsim's ``1/n^2``
quickly: at ``N=112`` accsim's residual *plateaus* at ``1e-13`` and stops improving, which
looks exactly like accsim hitting a floor and is in fact the reference hitting one.
``N=224`` moves that plateau by four decades. See
``test_the_two_splittings_meet_in_the_limit_and_the_reference_is_converged``.

The analytic side — that the term is the right Hamiltonian's exact flow, that it is
quartic in the momenta, that the composition is symplectic, that the design orbit does not
move — is ``tests/analytic/test_kinematic_quadrupole.py``. The PTC leg, which reaches the
same map through a third splitting and through second-order *coefficients* rather than
tracking, is ``tests/reference/test_second_order_map_madx.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import Drift, Quadrupole, ReferenceParticle

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0, GAMMA0 = 938.27208816e6, 20.0
L_Q, K1 = 0.7, 1.2

#: Kick count at which the reference has converged — chosen by the sweep in the module
#: docstring, not by taste. ``REF_KICKS_FINE`` is the witness that it *has*: doubling it
#: moves the residual accsim is gated on by a few percent, not by decades.
REF_KICKS, REF_KICKS_FINE = 224, 448

#: The 7-kick minimum of ``yoshida4`` — the coarse reference kept as a live counter-example.
REF_KICKS_COARSE = 7

#: Two off-axis probes and one on the axis. The first is the analytic file's probe; the
#: second has angles ~3x larger, which raises the dropped term by ~100x (it is cubic in
#: the angle) and so separates "the flag does something" from "the flag does the right
#: thing". The third carries momentum but no angle: ``H_kin`` is a function of the momenta,
#: so on the axis it is the identity however large ``delta`` is, and both codes must
#: already agree there — with the flag on and with it off alike.
STATES = [
    np.array([2.0e-3, 1.5e-3, -1.0e-3, 8.0e-4, 5.0e-4, 1.0e-3]),
    np.array([1.0e-3, 5.0e-3, 2.0e-3, -3.0e-3, 1.0e-3, 2.0e-2]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0e-2]),
]
ON_AXIS = 2

#: Every ``xt.Line`` build JIT-compiles a fresh C kernel — xobjects names each module
#: ``uuid4().hex``, so nothing is ever cached between builds (``docs/CONVENTIONS.md`` ->
#: *Test-suite cost*), at ~12 s apiece. Four are needed here; they are cached by their
#: construction arguments so the parametrised gates do not rebuild them.
_TRACKED: dict[tuple[int, float], np.ndarray] = {}


def _xtracked(kicks: int, k1: float = K1) -> np.ndarray:
    """Track every state in ``STATES`` through one exactly-split ``xt.Quadrupole``.

    Returns a ``(len(STATES), 6)`` array in accsim's coordinate order. All the states go
    through as one bunch so a single build serves them all; ``particle_id`` is used to
    undo any reordering xtrack applies.
    """
    key = (kicks, k1)
    if key not in _TRACKED:
        line = xt.Line(
            elements=[
                xt.Quadrupole(
                    length=L_Q,
                    k1=k1,
                    model="drift-kick-drift-exact",
                    integrator="yoshida4",
                    num_multipole_kicks=kicks,
                )
            ]
        )
        line.particle_ref = xt.Particles(mass0=MASS0, q0=1, gamma0=GAMMA0)
        try:
            line.build_tracker()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
        s = np.array(STATES)
        p = xt.Particles(
            mass0=MASS0,
            q0=1,
            gamma0=GAMMA0,
            x=s[:, 0].copy(),
            px=s[:, 1].copy(),
            y=s[:, 2].copy(),
            py=s[:, 3].copy(),
            zeta=s[:, 4].copy(),
            delta=s[:, 5].copy(),
        )
        line.track(p)
        order = np.argsort(p.particle_id)
        _TRACKED[key] = np.stack(
            [p.x[order], p.px[order], p.y[order], p.py[order], p.zeta[order], p.delta[order]],
            axis=1,
        )
    return _TRACKED[key]


def _gaps(kicks: int, n: int, *, k1: float = K1) -> list[float]:
    """Largest coordinate disagreement with the reference, per state in ``STATES``."""
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    quad = Quadrupole(L_Q, k1) if n == 0 else Quadrupole(L_Q, k1, kinematic_slices=n)
    theirs = _xtracked(kicks, k1)
    return [float(np.max(np.abs(quad.track(st, ref) - theirs[i]))) for i, st in enumerate(STATES)]


def test_the_kinematic_flag_lands_on_xtracks_exact_quadrupole() -> None:
    r"""**The gate.** ``kinematic_slices`` turns a ``1e-8`` disagreement into a ``1e-14`` one.

    Against the converged reference, accsim's shipped default misses by ``2.58e-10`` on
    the modest probe and ``2.90e-8`` on the steeper one. Those are the term L2 dropped,
    measured by tracking rather than by second-order coefficients — the same physics P1
    saw as ``5.6e-5`` in ``T[x, px, px]``.

    With the flag at ``256`` slices the same two states come back within ``5e-15`` and
    ``8e-14``, which for coordinates of size ``1e-3`` is the last bit or two. That is a
    factor of ``5e4`` and ``3.6e5``: not a tolerance being met, a disagreement being
    removed.

    The steeper state is what makes this a check of the *coefficient* and not merely of
    the presence of a correction. Its angles are about three times the first's and its gap
    is about 113 times larger — the cube the term is supposed to scale as, times the extra
    ``delta`` dependence (the analytic file gates the cubic scaling directly, at fixed
    ``delta``). A correction of the right shape but the wrong size could close one of these
    two; it could not close both.
    """
    off = _gaps(REF_KICKS, 0)
    on = _gaps(REF_KICKS, 256)
    assert off[0] == pytest.approx(2.58e-10, rel=0.05)
    assert off[1] == pytest.approx(2.90e-8, rel=0.05)
    assert on[0] < 1e-14
    assert on[1] < 2e-13
    assert off[0] / on[0] > 1e4
    assert off[1] / on[1] > 1e4


def test_the_two_splittings_meet_in_the_limit_and_the_reference_is_converged() -> None:
    r"""The residual falls as ``1/n^2`` over five decades — and the reference is not the floor.

    accsim's symmetric interleave is second order in the slice length, so its distance
    from the exact map should quarter per doubling. Measured against the converged
    reference, on the modest probe: ``9.8e-11, 2.5e-11, 6.2e-12, 1.6e-12, 3.9e-13,
    9.5e-14`` for ``n = 2 ... 64``. Every ratio is ``4.00``.

    **Why this is the whole cross-check and not decoration.** The two codes split
    different pieces of the Hamiltonian in different orders; there is no shared structure
    that could make a wrong coefficient converge to xtrack's answer. A clean ``1/n^2``
    approach to *someone else's map* is the strongest statement available here — stronger
    than any single-``n`` tolerance, which a mis-scaled term could still meet by accident
    at one amplitude.

    **The trap this test exists to keep shut.** Against ``REF_KICKS = 112`` the same
    ladder runs ``9.8e-11 ... 8.6e-14`` and then *stops*, sitting at ``1.0e-13`` for
    ``n = 128`` and ``1.2e-13`` for ``n = 256``. Read naively that is accsim bottoming out
    at ``1e-13``. It is not: it is the reference's own ``1/N^4`` error, which at ``N=112``
    is about ``1e-13``. Doubling ``N`` to 224 drops the ``n=256`` residual to ``4.9e-15``,
    and doubling again to 448 moves it by less than a factor of two — so the number the
    gates above rest on belongs to accsim, not to xtrack. The assertion below is that
    doubling the reference does *not* move the residual, which is the only honest way to
    claim a reference has converged.
    """
    gaps = {n: _gaps(REF_KICKS, n)[0] for n in (2, 4, 8, 16, 32, 64)}
    ns = sorted(gaps)
    for coarse, fine in zip(ns, ns[1:], strict=False):
        assert gaps[coarse] / gaps[fine] == pytest.approx(4.0, rel=0.05), (coarse, fine)

    # Doubling the reference leaves the residual where it was: it is accsim's, not xtrack's.
    for n in (16, 32, 64):
        finer = _gaps(REF_KICKS_FINE, n)[0]
        assert finer == pytest.approx(gaps[n], rel=0.05), n


def test_a_reference_too_coarse_to_see_the_term_agrees_with_everything() -> None:
    r"""**The counter-example, kept live.** At 7 kicks the reference cannot tell the flag apart.

    ``yoshida4`` rounds ``num_multipole_kicks`` up to a multiple of seven, so seven is its
    floor — and there the reference sits ``2.26e-6`` from its own limit, four decades above
    the term P2 (iv) restores. Flag off, flag on at one slice, flag on at 256: all three
    land within ``0.1%`` of the same ``2.26e-6``. Every gate in this file would pass
    against it, before the milestone and after, because the reference's own integration
    error swamps the physics being measured.

    This is why ``REF_KICKS`` is 224 and why the sweep that chose it is written down. It
    is the same lesson P2 (ii) learned from the sliced multipole and P2 (iii) from the
    cavity: *match the reference's integration first, then compare*. Recorded as a test
    rather than a comment because a comment cannot fail.
    """
    coarse = [_gaps(REF_KICKS_COARSE, n)[0] for n in (0, 1, 256)]
    assert all(g == pytest.approx(2.26e-6, rel=0.05) for g in coarse), coarse
    assert max(coarse) / min(coarse) < 1.001  # indistinguishable, not merely close

    # ... and the term it is blind to is four decades below its own error.
    assert _gaps(REF_KICKS, 0)[0] < 1e-3 * min(coarse)


def test_on_the_axis_the_flag_is_the_identity_and_both_codes_already_agreed() -> None:
    r"""``H_kin`` depends on the momenta, so a particle with none of them is untouched.

    The third probe is on the axis with ``delta = 3e-2`` — a long way off momentum, which
    is the axis L2's map is *exact* along. Both codes return the same state to ``6.7e-17``
    with the flag off, and turning it on at any number of slices leaves that number
    unmoved to three digits.

    Two things are pinned at once. That accsim's default was already right where the
    dropped term vanishes (so P2 (iv) is not papering over a different error), and that the
    new machinery does not perturb what was already correct — the property the analytic
    file gates as bit-identity on the design orbit, confirmed here against another code.
    """
    off = _gaps(REF_KICKS, 0)[ON_AXIS]
    assert off < 1e-16
    for n in (1, 4, 64, 256):
        assert _gaps(REF_KICKS, n)[ON_AXIS] == pytest.approx(off, rel=1e-3), n


def test_a_zero_strength_quadrupole_with_the_flag_is_xtracks_exact_drift() -> None:
    r"""L2's documented inconsistency, closed and checked against another code.

    L2 shipped a thick quadrupole whose ``k1 -> 0`` limit is the *paraxial* drift, not the
    exact one accsim's :class:`~accsim.Drift` is — an acknowledged wart, left alone because
    the obvious fix (branch on ``k1 == 0``) makes the map discontinuous in the strength,
    which is the trap P2 (ii) later found in the sextupole.

    The kinematic split closes it structurally instead: at ``k1 = 0`` the paraxial factor
    and the kinematic factor commute, the symmetric interleave telescopes, and what is left
    is the exact drift **identically**, at any ``n`` and with no branch anywhere. Measured
    against xtrack's own zero-gradient exact map: ``2.2e-16``, the same as accsim's
    :class:`~accsim.Drift`, against the ``1.5e-9`` and ``5.6e-8`` the default misses by.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    theirs = _xtracked(REF_KICKS, 0.0)
    for i, st in enumerate(STATES):
        assert np.max(np.abs(Drift(L_Q).track(st, ref) - theirs[i])) < 1e-15, i
        for n in (1, 2, 4):
            got = Quadrupole(L_Q, 0.0, kinematic_slices=n).track(st, ref)
            assert np.max(np.abs(got - theirs[i])) < 1e-15, (i, n)
    # Non-vacuous off the axis: the default really does miss the exact drift.
    default = [
        np.max(np.abs(Quadrupole(L_Q, 0.0).track(st, ref) - theirs[i]))
        for i, st in enumerate(STATES)
    ]
    assert default[0] > 1e-9
    assert default[1] > 1e-8
