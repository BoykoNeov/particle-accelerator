r"""P1 against MAD-X: ``sectormap`` element by element, and PTC's ``maptable`` for the turn.

Two of the three arbiters, ranked by method as the roadmap did. ``TWISS, sectormap`` is an
independent *analytic* second-order map, one row per element, at exactly accsim's
granularity. ``ptc_twiss, maptable`` is differential algebra composing exact maps about
the closed orbit, labelled by monomial — the leg that needs no storage convention and
the only one that reaches the ``t`` column with a cavity in the ring (``icase=6``).

**The frame change is the milestone's own composition rule, applied to a coordinate
map.** ``PT`` is a nonlinear function of ``delta`` and ``M R M^-1`` does not carry to
second order; ``_madx.to_accsim_frame_second_order`` builds the conversion as
``Phi^-1 . g . Phi`` with :func:`accsim.taylor.compose`. The drift is the gate on that:
its ``T`` is derived symbolically in the analytic suite, and MAD-X's ``t566 =
-3L/(2 beta0^3 gamma0^2)`` becomes accsim's ``-L (2 + beta0^2)/(2 gamma0^2)`` only when
``PT``'s quadratic term is carried — with the first-order transform reused it misses by
``1.25e-3`` relative, a control asserted below rather than remembered.

**What each leg covers, stated.** The ``sectormap`` leg gates every entry of every
element on a design-orbit ring — drift, thick quadrupole, sector bend, thin sextupole,
thin octupole — including the thick sextupole, whose gap P1 found here and P2 (ii) then
closed: its sliced body drifts through the *exact* map now, so what is left is the
integrator alone, gated by its ``1/n_slices^2`` scaling rather than by a number.
The PTC leg gates the composed one-turn map on the same ring, then on a bunched ring
(the ``t`` column), then about a **steered** closed orbit. The sign of PTC's sixth
variable is pinned on ``R56`` before anything at second order is read.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _madx import (
    beam_beta0,
    frame_maps,
    madx_session,
    ptc_maptable,
    sectormap_rows,
    to_accsim_frame_second_order,
)

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    Sextupole,
    TaylorMap,
    ThinOctupole,
    ThinQuadrupole,
    ThinSextupole,
    closed_orbit_6d,
    closed_orbit_nonlinear,
    second_order_element_maps,
    second_order_one_turn_map,
    taylor_expand,
)
from accsim.coords import DELTA, DIM, PX, PY, ZETA, X, Y

pytestmark = pytest.mark.reference

MASS0, GAMMA0 = 938.27208816e6, 20.0

#: The design-orbit ring: every element kind with a second-order map, explicit drifts so
#: that MAD-X's sectormap has one row per accsim element.
KF, KD = 1.2, -1.2
LQ, LB, ANGLE = 0.3, 1.0, 0.1
K2L, K3L = 2.0, 300.0
L_SEXT, K2_THICK = 0.2, 40.0

#: Measured floors (2026-09-02). Per-element sectormap agreement after the frame change is
#: ``2e-12`` on the drift, ``4e-11`` on the quadrupole and bend, i.e. accsim's own
#: differencing floor; the composed one-turn map against PTC lands at ``1.0e-10`` on
#: entries up to ``~600`` at ``step = 2.5e-4`` (``1.2e-9`` at ``1e-3``: the fourth-order
#: truncation of 36 thick elements, which is why the PTC gates pass the step explicitly).
#: Gates carry 10-100x headroom and no more.
ELEMENT_ATOL = 1e-9
TURN_ATOL = 1e-8

#: MAD-X and PTC apply the **hard-edge dipole fringe field** at second order by default.
#: accsim's :class:`Dipole` models it too since P2 — but *opt in*, ``fringe=True``, so the
#: package default is still the bare body. Killed on the default ring fixtures so those
#: compare body against body; ``fringe=True`` on both sides is the gate in
#: :func:`test_the_fringe_on_bend_matches_madx_entry_for_entry` and
#: :func:`test_one_turn_map_with_the_fringe_agrees_with_ptc`.
NO_FRINGE = "kill_ent_fringe=true, kill_exi_fringe=true"

#: The rows PTC returns at ``icase=5``: everything but the arrival time.
FIVE = [X, PX, Y, PY, DELTA]

#: Second harmonic of the 4-cell ring (``C = 14.4 m``): ``2 beta0 c / C``.
RING_LENGTH = 4 * (LQ + 0.5 + 0.5 + LB + LQ + 0.4 + 0.6)
RF_FREQ_HZ = 2 * ReferenceParticle.from_gamma(MASS0, GAMMA0).beta0 * 299792458.0 / RING_LENGTH


def _elements(ref: ReferenceParticle, *, fringe: bool = False, faces: float = 0.0) -> list:
    return [
        Quadrupole(LQ, KF, name="qf"),
        Drift(0.5, name="d1"),
        ThinSextupole(K2L, name="ms"),
        Drift(0.5, name="d2"),
        Dipole(LB, ANGLE, e1=faces, e2=faces, name="mb", fringe=fringe),
        Quadrupole(LQ, KD, name="qd"),
        Drift(0.4, name="d3"),
        ThinOctupole(K3L, name="mo"),
        Drift(0.6, name="d4"),
    ]


def _madx_line(
    *,
    cells: int = 4,
    kick: float = 0.0,
    cavity: bool = False,
    thin_quads: bool = False,
    fringe: bool = False,
    faces: float = 0.0,
) -> str:
    """A ``line``, not a ``sequence``: elements abut exactly and nothing is implicit.

    ``thin_quads`` replaces each thick quadrupole by a thin one of the same integrated
    strength followed by a drift of its length — the fixture for the steered-orbit gate,
    where accsim's thick quadrupole is paraxial in the angles (L2) and PTC's is not.
    """
    if thin_quads:
        quads = [
            f"qf: multipole, knl={{0, {KF * LQ!r}}};",
            f"qd: multipole, knl={{0, {KD * LQ!r}}};",
            f"lq: drift, l={LQ};",
        ]
        cell = "cell: line = (qf, lq, d1, ms, d2, mb, qd, lq, d3, mo, d4);"
    else:
        quads = [
            f"qf: quadrupole, l={LQ}, k1={KF};",
            f"qd: quadrupole, l={LQ}, k1={KD};",
        ]
        cell = "cell: line = (qf, d1, ms, d2, mb, qd, d3, mo, d4);"
    defs = quads + [
        "d1: drift, l=0.5;",
        "d2: drift, l=0.5;",
        "d3: drift, l=0.4;",
        "d4: drift, l=0.6;",
        f"ms: multipole, knl={{0, 0, {K2L}}};",
        f"mo: multipole, knl={{0, 0, 0, {K3L}}};",
        f"mb: sbend, l={LB}, angle={ANGLE}, e1={faces!r}, e2={faces!r}"
        + ("" if fringe else f", {NO_FRINGE}")
        + ";",
        cell,
    ]
    head = []
    if kick:
        defs.append(f"kck: hkicker, kick={kick!r};")
        head.append("kck")
    if cavity:
        # MAD-X: volt in MV, freq in MHz, lag in turns of 2 pi. The frequency must be a
        # harmonic of the revolution frequency or PTC's fixed point walks away from
        # ``t = 0`` (measured: ``k[t] = 4.9965`` at 30 MHz on a 14.4 m ring), and the
        # phase must be the *stable* zero crossing above transition — ``lag = 0.5`` here,
        # accsim's ``phi_s = pi``; at ``lag = 0`` the slope sign is the unstable one and
        # both codes' longitudinal traces exceed 2.
        defs.append(f"cav: rfcavity, volt=1.0, freq={RF_FREQ_HZ / 1e6!r}, lag=0.5;")
        head.append("cav")
    ring = ", ".join(head + [f"{cells}*cell"])
    return f"""
beam, particle=proton, gamma={GAMMA0!r};
{chr(10).join(defs)}
ring: line = ({ring});
use, sequence=ring;
"""


def _accsim_ring(
    *,
    cells: int = 4,
    kick: float = 0.0,
    cavity: bool = False,
    thin_quads: bool = False,
    fringe: bool = False,
    faces: float = 0.0,
) -> Lattice:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    head: list = []
    if kick:
        head.append(Corrector(kick_x=kick))
    if cavity:
        head.append(RFCavity(1.0e6, RF_FREQ_HZ, np.pi))
    elements = _elements(ref, fringe=fringe, faces=faces)
    if thin_quads:
        thinned: list = []
        for e in elements:
            if isinstance(e, Quadrupole):
                thinned += [ThinQuadrupole(e.k1 * e.length), Drift(e.length)]
            else:
                thinned.append(e)
        elements = thinned
    return Lattice(head + elements * cells, ref)


@pytest.fixture(scope="module")
def sectormap():
    """``(rows, beta0)`` of one cell's sectormap on the design orbit."""
    with madx_session() as madx:
        madx.input(_madx_line(cells=1))
        madx.input(
            f'twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";'
        )
        return sectormap_rows(madx), beam_beta0(madx, "ring")


@pytest.fixture(scope="module")
def ring():
    return _accsim_ring(cells=1)


def _rows_in_beam_order(rows: dict, names: list[str]) -> list:
    return [rows[n] for n in names]


def test_ptc_sixth_variable_is_minus_t_and_fifth_is_pt() -> None:
    """Pinned at first order on a bare drift before any second-order row is read.

    ``c6_000010 = -L/(beta0^2 gamma0^2)`` where MAD-X's own ``R56`` is ``+``: the sixth
    variable is ``-T``. ``c1_010010 = -L/beta0`` is the *sum* of the symmetric pair
    ``2 x (-L/(2 beta0))`` of ``t126``: the fifth is ``PT`` and the monomial decode halves
    it.
    """
    with madx_session() as madx:
        madx.input(f"""
        beam, particle=proton, gamma={GAMMA0!r};
        el: drift, l=1.0;
        seq: sequence, l=1.0; el, at=0.5; endsequence;
        use, sequence=seq;
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst=5, exact=true;
        ptc_twiss, icase=6, no=2, betx=1, bety=1, betz=1, maptable;
        ptc_end;
        """)
        k, R, T = ptc_maptable(madx)
        beta0 = beam_beta0(madx, "seq")
    gamma0 = GAMMA0
    assert abs(R[ZETA, DELTA] + 1.0 / (beta0**2 * gamma0**2)) < 1e-12  # minus: -T
    assert abs(T[X, PX, DELTA] + 0.5 / beta0) < 1e-12  # PT, and the pair halved
    assert abs(T[ZETA, PX, PX] - 0.5 / beta0) < 1e-12  # -T flips the zeta row
    assert np.max(np.abs(k)) < 1e-15


def test_drift_second_order_entries_and_the_transform_that_reaches_them(sectormap, ring) -> None:
    """MAD-X's drift entries are ``-L/(2 beta0)`` and ``-3L/(2 beta0^3 gamma0^2)``, and only
    the nonlinear ``PT`` transform turns them into accsim's."""
    rows, beta0 = sectormap
    k, R, T = rows["d1"]
    L = 0.5
    g = GAMMA0
    # MAD-X's own numbers, in its own frame: scale 1/beta0 (not 1/beta0^2 as an earlier
    # roadmap note had it — corrected on measurement, 2026-09-02).
    assert abs(T[X, PX, DELTA] + L / (2 * beta0)) < 1e-12
    assert abs(T[ZETA, PX, PX] + L / (2 * beta0)) < 1e-12
    assert abs(T[ZETA, DELTA, DELTA] + 1.5 * L / (beta0**3 * g**2)) < 1e-12
    assert np.max(np.abs(k)) == 0.0

    ours = second_order_element_maps(ring)[1]
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)
    assert np.max(np.abs(theirs.T - ours.T)) < 1e-11
    assert np.max(np.abs(theirs.R - ours.R)) < 1e-10
    assert abs(ours.T[ZETA, DELTA, DELTA] + L * (2 + beta0**2) / (2 * g**2)) < 1e-11

    # Control: the first-order transform reused at second order misses by 1/(2 gamma0^2)
    # of the linear term on the momentum-indexed zeta entry — the roadmap's 1.25e-3.
    m = np.diag([1.0, 1.0, 1.0, 1.0, beta0, 1.0 / beta0])
    minv = np.linalg.inv(m)
    naive = np.einsum("ia,abc,bj,ck->ijk", m, T, minv, minv)
    miss = naive[ZETA, DELTA, DELTA] - ours.T[ZETA, DELTA, DELTA]
    # The naive rule drops beta0 R56 (beta0/(2 gamma0^2)) = L/(2 gamma0^4): derived, and
    # 8e-4 of the entry here — a thousand times the gate below.
    assert abs(miss + L / (2 * g**4)) < 1e-12
    assert abs(miss / ours.T[ZETA, DELTA, DELTA]) > 5e-4


@pytest.mark.parametrize("name", ["qf", "d1", "ms", "d2", "mb", "qd", "d3", "mo", "d4"])
def test_every_element_map_agrees_with_sectormap(sectormap, ring, name: str) -> None:
    """All 216 entries of ``T`` (and 36 of ``R``) per element, after the frame change."""
    rows, beta0 = sectormap
    names = ["qf", "d1", "ms", "d2", "mb", "qd", "d3", "mo", "d4"]
    ours = second_order_element_maps(ring)[names.index(name)]
    k, R, T = rows[name]
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)
    assert np.max(np.abs(theirs.R - ours.R)) < 1e-10, name
    assert np.max(np.abs(theirs.T - ours.T)) < ELEMENT_ATOL, (
        name,
        np.max(np.abs(theirs.T - ours.T)),
    )
    if name in ("ms", "mo", "d1", "qf", "mb"):
        assert np.max(np.abs(ours.T)) > 1e-3 or name == "mo"  # not a zero matched to a zero
    if name == "mo":
        assert np.max(np.abs(theirs.T)) < 1e-12  # MAD-X agrees: no octupole in T


def test_the_thick_sextupole_carries_the_drifts_own_t_now() -> None:
    """P1 found this gap here; P2 (ii) closed it. Both halves are asserted, in that order.

    MAD-X's thick sextupole carries the drift's chromatic and path-length second-order
    terms. accsim's sliced body did not, because the gaps between its slices were the
    linear drift *matrix*: the residual was the standalone drift's whole ``T``, and its
    largest entry was ``T[x, px, delta] = -L/2`` — not small, and not a tolerance.

    The gaps are ``Drift.track`` now, so what is left is the **slicing** error alone, and
    it is gated on its mechanism rather than on a measured number: drift-kick-drift is a
    second-order integrator, so halving the slice size must quarter the residual. It does
    — ``x4.00`` per doubling across 50..400 slices (``5.3e-6`` down to ``8.3e-8``) — which
    a map that were merely small-and-wrong could not reproduce, and which no fixed
    tolerance would have distinguished from one.

    **The witness is kept.** The term did not vanish from either code, it moved into the
    element: MAD-X still reports ``t126 = -L/(2 beta0)`` and accsim's bare
    :class:`~accsim.elements.drift.Drift` still expands to ``T[x, px, delta] = -L/2``.
    Asserting the agreement without them would lose the proof that anything was there.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    with madx_session() as madx:
        madx.input(f"""
        beam, particle=proton, gamma={GAMMA0!r};
        el: sextupole, l={L_SEXT}, k2={K2_THICK};
        ring: line = (el);
        use, sequence=ring;
        twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";
        """)
        (k, R, T), beta0 = sectormap_rows(madx)["el"], beam_beta0(madx, "ring")
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)

    # The witness: the term is still in MAD-X's map and still in accsim's bare drift.
    assert abs(T[X, PX, DELTA] + L_SEXT / (2 * beta0)) < 1e-12
    drift = taylor_expand(lambda s: Drift(L_SEXT).track(s, ref), np.zeros(DIM))
    assert abs(drift.T[X, PX, DELTA] + L_SEXT / 2) < 1e-12

    gaps = {}
    for n in (50, 100, 200, 400):
        sliced = Sextupole(L_SEXT, K2_THICK, n_slices=n)
        ours = taylor_expand(lambda s, e=sliced: e.track(s, ref), np.zeros(DIM))
        assert np.max(np.abs(theirs.R - ours.R)) < 1e-10, n  # first order was never in it
        gaps[n] = float(np.max(np.abs(theirs.T - ours.T)))

    # The mechanism: a second-order integrator, so 4x per doubling of the slice count.
    counts = sorted(gaps)
    for coarse, fine in zip(counts[:-1], counts[1:], strict=True):
        assert gaps[coarse] / gaps[fine] == pytest.approx(4.0, rel=0.05), gaps

    # And it really has come down: the drift's own T is no longer anywhere in the gap.
    assert gaps[400] < 2e-7
    assert np.max(np.abs(drift.T)) > 1e-3  # non-vacuous: that T is not itself small


def test_the_bend_gap_is_the_hard_edge_fringe_and_nothing_else() -> None:
    r"""With MAD-X's default fringe on, the sector bend misses by ``hL/2``; killed, by ``6e-12``.

    **P1's finding, kept as the record of the default.** Both MAD-X's TWISS and PTC
    (``exact=true``, two integrator models) apply a hard-edge dipole fringe at each face —
    the second-order map of the field's termination, whose entrance form is
    ``x += (h/2) y^2``, ``py -= h y px`` and whose exit form is the reverse. Composed with
    the body that leaves ``T[x, y, py] = T[y, px, y] = -hL/2``,
    ``T[py, px, py] = +hL cos(theta)/2``, ``T[x, y, y] = -h (1 - cos theta)/2`` and
    ``T[px, y, y] = -h^2 sin(theta)/2`` — the entries measured here, all of them
    ``y``-dependent in a magnet whose *body* field is ``y``-independent.

    P2 shipped that map (``Dipole(..., fringe=True)``, gated in the next test), and this
    one is deliberately unchanged: accsim's **default** bend still carries F1's linear
    pole-face matrices and no fringe, exactly as ``xt.Bend`` does with its default
    ``linear`` edge model, and the gap it leaves against a default MAD-X ``sbend`` is a
    fact about the package that a caller has to know. With the fringe killed on MAD-X's
    side the three agree to the differencing floor.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    ours = taylor_expand(lambda s: Dipole(LB, ANGLE).track(s, ref), np.zeros(DIM))
    h = ANGLE / LB
    gaps = {}
    for label, flags in (("fringe", ""), ("killed", ", " + NO_FRINGE)):
        with madx_session() as madx:
            madx.input(f"""
            beam, particle=proton, gamma={GAMMA0!r};
            mb: sbend, l={LB}, angle={ANGLE}{flags};
            ring: line = (mb);
            use, sequence=ring;
            twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";
            """)
            (k, R, T), beta0 = sectormap_rows(madx)["mb"], beam_beta0(madx, "ring")
        gaps[label] = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0).T - ours.T
    assert np.max(np.abs(gaps["killed"])) < 1e-10
    gap = gaps["fringe"]
    assert abs(gap[X, Y, PY] + h * LB / 2) < 1e-10
    assert abs(gap[Y, PX, Y] + h * LB / 2) < 1e-10
    assert abs(gap[PY, PX, PY] - h * LB * np.cos(ANGLE) / 2) < 1e-10
    assert abs(gap[X, Y, Y] + h * (1 - np.cos(ANGLE)) / 2) < 1e-10
    assert abs(gap[PX, Y, Y] + h * h * np.sin(ANGLE) / 2) < 1e-10
    # And nothing in the horizontal-only block: the fringe is a y-effect at this order.
    hx = np.ix_([X, PX], [X, PX], [X, PX])
    assert np.max(np.abs(gap[hx])) < 1e-10


def test_the_fringe_on_bend_matches_madx_entry_for_entry() -> None:
    r"""P2 (i): ``Dipole(fringe=True)`` against a default MAD-X ``sbend``, all 216 entries.

    The gate the milestone is judged on, and it is deliberately *not* the five closed
    forms P1 named: those were the entries that happened to be large, and the composed
    map has **twelve** distinct nonzero ones — the two ``x`` shifts nearly cancelling, the
    entrance kick carried through the body's vertical drift, a ``T[zeta, y, y]`` from the
    path length and a ``T[py, y, delta]`` from the rigidity. Every one of them is compared
    against MAD-X's analytic ``sectormap`` at the same ``1e-10`` the fringe-off bend meets.

    The five are asserted by name as well, because they are the numbers the roadmap
    carries, and the *previous* map is asserted to fail this same comparison by ``hL/2`` —
    otherwise a fringe that did nothing would pass.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    with madx_session() as madx:
        madx.input(f"""
        beam, particle=proton, gamma={GAMMA0!r};
        mb: sbend, l={LB}, angle={ANGLE};
        ring: line = (mb);
        use, sequence=ring;
        twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";
        """)
        (k, R, T), beta0 = sectormap_rows(madx)["mb"], beam_beta0(madx, "ring")
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)

    ours = taylor_expand(lambda s: Dipole(LB, ANGLE, fringe=True).track(s, ref), np.zeros(DIM))
    assert np.max(np.abs(theirs.R - ours.R)) < 1e-10
    assert np.max(np.abs(theirs.T - ours.T)) < 1e-10, np.max(np.abs(theirs.T - ours.T))

    # The twelve entries are real: without the fringe the same comparison misses by hL/2.
    bare = taylor_expand(lambda s: Dipole(LB, ANGLE).track(s, ref), np.zeros(DIM))
    h = ANGLE / LB
    assert np.max(np.abs(theirs.T - bare.T)) == pytest.approx(h * LB / 2, rel=1e-3)
    upper = [
        theirs.T[i, j, l] - bare.T[i, j, l]
        for i in range(DIM)
        for j in range(DIM)
        for l in range(j, DIM)
    ]
    assert np.count_nonzero(np.abs(np.array(upper)) > 1e-9) == 12

    # ...and the five the roadmap names, on MAD-X's side of the comparison.
    c, sn = np.cos(ANGLE), np.sin(ANGLE)
    for idx, want in (
        ((X, Y, PY), -h * LB / 2),
        ((Y, PX, Y), -h * LB / 2),
        ((PY, PX, PY), +h * LB * c / 2),
        ((X, Y, Y), -h * (1 - c) / 2),
        ((PX, Y, Y), -h * h * sn / 2),
    ):
        assert theirs.T[idx] - bare.T[idx] == pytest.approx(want, abs=1e-10), idx
        assert abs(theirs.T[idx] - ours.T[idx]) < 1e-10, idx


#: The rotated faces P3 gates. Deliberately unequal and of opposite sign, so a map that
#: applied one face twice, or that mixed the entrance and exit compositions, cannot pass.
E1_FACE, E2_FACE = 0.12, -0.07


@pytest.mark.parametrize(
    ("e1", "e2", "label"),
    [(E1_FACE, E2_FACE, "asymmetric"), (ANGLE / 2, ANGLE / 2, "rectangular")],
)
def test_the_rotated_face_matches_madx_entry_for_entry(e1: float, e2: float, label: str) -> None:
    r"""P3 (a): ``Dipole(e1, e2, fringe=True)`` against a default MAD-X ``sbend``.

    The second-order arbiter for the wedge, and the one leg that is *analytic* rather
    than tracked. MAD-X applies the pole-face fringe by default (``kill_ent_fringe`` is
    off), so a plain ``sbend`` with ``e1``/``e2`` carries the whole rotated face; accsim
    now does too, and all 216 entries agree at the same ``1e-10`` the unrotated bend meets.

    Three controls, because "the entries agree" is worth very little on its own:

    - the **fringe-off** bend, which is F2's linear ``h tan(e)`` edge and nothing else,
      misses this same comparison — that is the size of what P3 adds at second order;
    - the **unrotated** ``fringe=True`` bend misses it too, by more, which is what says
      the wedge is being applied and not merely the P2 (i) fringe with an edge matrix
      bolted on;
    - and the ``rectangular`` case is run as well as the asymmetric one, because
      ``e1 = e2 = angle/2`` is the face a real lattice actually has.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    with madx_session() as madx:
        madx.input(f"""
        beam, particle=proton, gamma={GAMMA0!r};
        mb: sbend, l={LB}, angle={ANGLE}, e1={e1!r}, e2={e2!r};
        ring: line = (mb);
        use, sequence=ring;
        twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";
        """)
        (k, R, T), beta0 = sectormap_rows(madx)["mb"], beam_beta0(madx, "ring")
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)

    faced = Dipole(LB, ANGLE, e1=e1, e2=e2, fringe=True)
    ours = taylor_expand(lambda s: faced.track(s, ref), np.zeros(DIM))
    assert np.max(np.abs(theirs.R - ours.R)) < 1e-10, label
    residual = np.max(np.abs(theirs.T - ours.T))
    assert residual < 1e-10, residual

    linear_edges = taylor_expand(
        lambda s: Dipole(LB, ANGLE, e1=e1, e2=e2).track(s, ref), np.zeros(DIM)
    )
    unrotated = taylor_expand(lambda s: Dipole(LB, ANGLE, fringe=True).track(s, ref), np.zeros(DIM))
    # Both controls miss by more than a millionth-of-a-percent of nothing: measured
    # 2.5e-3 (fringe off) and 2.5e-3 (unrotated), against a 1e-10 agreement.
    assert np.max(np.abs(theirs.T - linear_edges.T)) > 1e6 * residual
    assert np.max(np.abs(theirs.T - unrotated.T)) > 1e6 * residual
    assert (
        min(np.max(np.abs(theirs.T - linear_edges.T)), np.max(np.abs(theirs.T - unrotated.T)))
        > 1e-4
    )
    # The linear map, on the other hand, is the *same* with and without the face: F2's
    # edge matrix is already the whole first-order story, which is P3's headline.
    assert np.max(np.abs(ours.R - linear_edges.R)) < 1e-10


def test_what_the_wedge_adds_at_second_order_is_first_order_in_the_face_angle() -> None:
    r"""The size of the new content, and its **order** — which is the gate, not the size.

    P2 (i) refused the rotated face because "the wedge is first order in the face angle
    where the fringe is second". That is true of the second-order map, and it is measured
    here rather than asserted: the gap between the fringe-only bend's ``T`` and MAD-X's
    rotated-face ``T`` falls by ten per decade of ``e``, where the *linear* map's gap is
    zero at every ``e``. Both halves matter — the first is why P3 exists, the second is
    why it could be a quiet opt-in.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    unrotated = taylor_expand(lambda s: Dipole(LB, ANGLE, fringe=True).track(s, ref), np.zeros(DIM))
    previous = None
    for e in (1e-1, 1e-2, 1e-3):
        with madx_session() as madx:
            madx.input(f"""
            beam, particle=proton, gamma={GAMMA0!r};
            mb: sbend, l={LB}, angle={ANGLE}, e1={e!r}, e2=0.0;
            ring: line = (mb);
            use, sequence=ring;
            twiss, betx=1.0, bety=1.0, sectormap, sectortable=smap, sectorfile="{os.devnull}";
            """)
            (k, R, T), beta0 = sectormap_rows(madx)["mb"], beam_beta0(madx, "ring")
        theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0)
        faced = Dipole(LB, ANGLE, e1=e, fringe=True)
        ours = taylor_expand(lambda s, b=faced: b.track(s, ref), np.zeros(DIM))
        assert np.max(np.abs(theirs.T - ours.T)) < 1e-10, e

        gap = np.max(np.abs(theirs.T - unrotated.T))
        if previous is not None:
            assert previous / gap == pytest.approx(10.0, rel=0.1), e
        previous = gap


def _ptc_turn(sequence: str, *, icase: int, closed_orbit: bool, order: int = 2, start: str = ""):
    """PTC's one-turn ``(k, R, T)`` in its own frame, plus ``beta0``.

    ``start`` is an explicit expansion point (``x=..., px=...`` in MAD-X syntax) for a map
    about a trajectory that need not close; ``closed_orbit`` asks PTC to find the fixed
    point itself.
    """
    with madx_session() as madx:
        madx.input(sequence)
        co = "closed_orbit," if closed_orbit else ""
        madx.input(f"""
        ptc_create_universe;
        ptc_create_layout, model=2, method=6, nst=5, exact=true;
        ptc_twiss, icase={icase}, no={order}, {co} {start} maptable;
        ptc_end;
        """)
        return ptc_maptable(madx), beam_beta0(madx, "ring")


def test_one_turn_map_agrees_with_ptc_on_the_design_orbit() -> None:
    """The composed ring against differential algebra, ``icase=5`` (no cavity: five
    variables, so the ``t`` column is absent and only the ``t`` row's zero is read)."""
    lat = _accsim_ring()
    (k, R, T), beta0 = _ptc_turn(_madx_line(), icase=5, closed_orbit=True)
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0, time_sign=-1.0)
    ours = second_order_one_turn_map(lat, step=2.5e-4)
    # icase=5: five variables. Nothing depends on t, PTC exposes no t column — and no t
    # *row* either (measured: no ``c6`` rows at all), so the arrival-time row is the
    # sectormap leg's and the icase=6 test's to gate, not this one's.
    assert np.max(np.abs(ours.T[:, :, ZETA])) < 1e-10 and np.max(np.abs(ours.T[:, ZETA, :])) < 1e-10
    assert np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE])) < 1e-9
    diff = np.abs(theirs.T[FIVE] - ours.T[FIVE])
    assert np.max(diff) < TURN_ATOL, np.max(diff)
    assert np.max(np.abs(ours.T)) > 100.0


def test_one_turn_map_with_the_fringe_agrees_with_ptc() -> None:
    r"""The whole ring, fringe on, against differential algebra — eight faces composed.

    The ``sectormap`` gate above is one magnet in isolation, which cannot see whether the
    two faces are applied in the right order or with the right relative sign: swapping
    them changes the composed ring, not the single element's ``T`` at leading order. Here
    the four bends' eight faces are composed with everything else in the cell and read
    against PTC's exact maps, on a ring whose ``T`` reaches ``600``. It is the same
    comparison :func:`test_one_turn_map_agrees_with_ptc_on_the_design_orbit` makes with the
    fringe killed on both sides, run with it live on both.
    """
    lat = _accsim_ring(fringe=True)
    (k, R, T), beta0 = _ptc_turn(_madx_line(fringe=True), icase=5, closed_orbit=True)
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0, time_sign=-1.0)
    ours = second_order_one_turn_map(lat, step=2.5e-4)
    assert np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE])) < 1e-9
    diff = np.abs(theirs.T[FIVE] - ours.T[FIVE])
    assert np.max(diff) < TURN_ATOL, np.max(diff)

    # And the fringe is not a rounding correction on this ring: the fringe-off map misses
    # PTC's fringe-on one by ~0.1, seven orders above the gate.
    bare = second_order_one_turn_map(_accsim_ring(), step=2.5e-4)
    assert np.max(np.abs(theirs.T[FIVE] - bare.T[FIVE])) > 1e-2


def test_one_turn_map_with_rotated_faces_agrees_with_ptc() -> None:
    r"""P3 (a) on the whole ring: four bends, eight *rotated* faces, against PTC.

    The third arbiter, and the only one that can see the two faces' **order** and their
    relative sign. A single element's ``T`` is nearly blind to swapping the entrance and
    exit compositions; a ring of four bends composed with quadrupoles, sextupoles and an
    octupole is not. Same comparison as the unrotated fringe leg above, with
    ``e1 = e2 = angle/2`` — the rectangular bend — on both sides.

    ``sectormap`` and PTC are not independent of each other the way accsim and xtrack are
    (both are MAD-X), so what this leg adds is the eightfold composition and the
    closed-orbit search, not a fresh derivation of the face.

    **The step is 1e-3 here and 2.5e-4 elsewhere in this file, and that is measured.**
    :func:`~accsim.second_order_one_turn_map` differences twice, so its error is a U:
    third-order truncation falling as ``step^4`` on the left, round-off rising as
    ``1/step^2`` on the right. Rotating the faces raises the round-off branch by ``600x``
    (the state passes through ``px ~ sin(e)`` and back at every face, which costs
    precision on the small coordinates) without touching the truncation branch at all —
    so the minimum moves from ``2.5e-4`` to ``1e-3``. Both rings land on the *same*
    ``1.18e-9`` there, which is the ring's own third-order leftover and not a
    disagreement about the faces. That is asserted as a sweep below rather than written
    in a comment, because a comment cannot fail.
    """
    half = ANGLE / 2.0
    lat = _accsim_ring(fringe=True, faces=half)
    (k, R, T), beta0 = _ptc_turn(_madx_line(fringe=True, faces=half), icase=5, closed_orbit=True)
    theirs = to_accsim_frame_second_order(k, R, T, np.zeros(DIM), beta0, time_sign=-1.0)
    ours = second_order_one_turn_map(lat, step=1e-3)
    assert np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE])) < 1e-9
    diff = np.max(np.abs(theirs.T[FIVE] - ours.T[FIVE]))
    assert diff < TURN_ATOL, diff
    assert np.max(np.abs(ours.T)) > 600.0

    # Controls. The linear edge alone misses PTC's rotated face by ~0.1, and the ring's
    # *first-order* map is the same with the faces on and off — F2's edge matrix is the
    # whole first-order story, which is what let P3 stay a quiet opt-in.
    linear_only = second_order_one_turn_map(_accsim_ring(faces=half), step=1e-3)
    assert np.max(np.abs(theirs.T[FIVE] - linear_only.T[FIVE])) > 1e6 * diff
    assert np.max(np.abs(ours.R - linear_only.R)) < 1e-9

    # The left arm of the U: pure truncation, so it falls as step^4 and it is the *same*
    # for the sector ring, which is what says it belongs to the ring and not to the face.
    sector = _accsim_ring(fringe=True)
    (k0, R0, T0), b0 = _ptc_turn(_madx_line(fringe=True), icase=5, closed_orbit=True)
    theirs0 = to_accsim_frame_second_order(k0, R0, T0, np.zeros(DIM), b0, time_sign=-1.0)
    previous = None
    for step in (4e-3, 2e-3, 1e-3):
        gap = np.max(np.abs(theirs.T[FIVE] - second_order_one_turn_map(lat, step=step).T[FIVE]))
        if previous is not None:
            assert previous / gap == pytest.approx(16.0, rel=0.15), step
        previous = gap
        flat = np.max(
            np.abs(theirs0.T[FIVE] - second_order_one_turn_map(sector, step=step).T[FIVE])
        )
        assert flat == pytest.approx(gap, rel=0.15), step

    # ...and the right arm, which is the one rotating the faces moves.
    tight = np.max(np.abs(theirs.T[FIVE] - second_order_one_turn_map(lat, step=2.5e-4).T[FIVE]))
    tight_sector = np.max(
        np.abs(theirs0.T[FIVE] - second_order_one_turn_map(sector, step=2.5e-4).T[FIVE])
    )
    assert tight > 10.0 * previous
    assert tight > 100.0 * tight_sector


def test_one_turn_map_agrees_with_ptc_on_a_bunched_ring() -> None:
    r"""With a cavity, ``icase=6`` and the ``t`` column live — and P1's named gap, closed.

    **P1 found here that accsim's cavity applied a momentum kick where PTC's applies an
    energy kick**, the conversion from energy to ``delta`` being frozen at ``delta = 0``.
    P2 (iii) made it the energy kick it always should have been —
    ``delta' = psi(PT(delta) + Delta PT(zeta))`` — whose ``zeta delta`` cross term is

        T[delta, zeta, delta] = -R65 / (2 gamma0^2),   R65 = d(Delta delta)/d zeta,

    ``-5.83e-8`` here, reaching ``T[x, delta, zeta]`` through the dispersion at ``1.1e-7``.
    So this test now gates the whole ``T`` directly, at the same ``TURN_ATOL`` as the
    other two PTC legs, with no correction term applied on either side.

    **The control is inverted rather than deleted**, because a rewrite that only checks
    agreement can pass for the wrong reason — nothing here would notice if PTC's cavity
    and accsim's had *both* been momentum kicks. So the pre-P2 (iii) cavity is
    reconstructed by zeroing exactly those two entries, recomposed into the ring, and
    asserted to miss PTC by exactly ``-R65/(2 gamma0^2)`` and by ``> 1e-7`` overall. That
    is the same measurement P1 made, read from the other end.
    """
    lat = _accsim_ring(cavity=True)
    (k, R, T), beta0 = _ptc_turn(_madx_line(cavity=True), icase=6, closed_orbit=True)
    co = closed_orbit_6d(lat)
    assert np.max(np.abs(co)) < 1e-10  # no radiation, phi_s at a zero crossing: the axis
    z0 = np.zeros(DIM)
    theirs = to_accsim_frame_second_order(k, R, T, z0, beta0, time_sign=-1.0)
    maps = second_order_element_maps(lat, z0, step=2.5e-4)
    ours = TaylorMap.identity()
    for m in maps:
        ours = ours.then(m)
    lon = ours.R[np.ix_([ZETA, DELTA], [ZETA, DELTA])]
    assert abs(np.trace(lon)) < 2.0  # longitudinally stable: the right zero crossing
    assert np.max(np.abs(theirs.R - ours.R)) < 1e-9
    assert np.max(np.abs(ours.T[:, ZETA, :])) > 1e-9  # the t column is live

    # The gate: the whole second-order map, no correction term on either side.
    diff = np.abs(theirs.T - ours.T)
    assert np.max(diff) < TURN_ATOL, np.max(diff)

    # The inverted control: put the momentum-kick cavity back and it misses, decisively.
    cav = maps[0]
    assert isinstance(lat.elements[0], RFCavity)
    missing = -cav.R[DELTA, ZETA] / (2 * lat.ref.gamma0**2)
    assert abs(missing) > 1e-8  # not a round-off statement
    assert abs(cav.T[DELTA, ZETA, DELTA] - missing) < 1e-11
    assert abs(cav.T[DELTA, DELTA, ZETA] - missing) < 1e-11

    T_old = cav.T.copy()
    T_old[DELTA, ZETA, DELTA] = T_old[DELTA, DELTA, ZETA] = 0.0
    was = TaylorMap(cav.origin, cav.k, cav.R, T_old)
    for m in maps[1:]:
        was = was.then(m)
    gap = theirs.T - was.T
    assert abs(gap[DELTA, ZETA, DELTA] - missing) < 1e-11
    assert abs(gap[DELTA, DELTA, ZETA] - missing) < 1e-11
    assert np.max(np.abs(gap)) > 1e-7


def test_one_turn_map_agrees_with_ptc_about_a_steered_orbit() -> None:
    """A horizontal steerer bumps the orbit through the sextupoles, and the two maps about
    that orbit must agree — ``T`` about a displaced point, where the sextupoles' feed-down
    moves ``R`` and the drifts' ``T`` depends on the orbit angle.

    Two things were measured before this could be written as a gate. **PTC's own fixed
    point is not sharp enough**: ``closed_orbit`` lands within ``1e-9`` of accsim's orbit
    (accsim's Newton is at ``1e-15``), and about a map whose ``T`` reaches ``600`` a
    ``1e-10`` orbit difference moves ``R`` by ``2 T dz ~ 1e-7``; so the expansion point is
    handed to PTC explicitly and both codes expand about the same point. And **the thick
    quadrupole is where the two codes part off-axis** — accsim's is paraxial in the angles
    (L2), PTC's is exact, and about a point with an orbit angle that is a second-order
    difference (:func:`test_the_thick_quadrupole_departs_from_ptc_off_axis_as_l2_says`).
    The ring here therefore carries **thin** quadrupoles, so that what is compared is the
    map and not a known approximation.
    """
    kick = 3.0e-4
    lat = _accsim_ring(kick=kick, thin_quads=True)
    orbit = closed_orbit_nonlinear(lat)
    assert abs(orbit[0]) > 1e-4  # a real bump
    (k_co, _, _), _ = _ptc_turn(_madx_line(kick=kick, thin_quads=True), icase=5, closed_orbit=True)
    assert np.max(np.abs(k_co[[X, PX, Y, PY]] - orbit)) < 1e-9  # the fixed points agree
    z0 = np.zeros(DIM)
    z0[[X, PX, Y, PY]] = orbit
    start = f"betx=1, bety=1, x={float(orbit[0])!r}, px={float(orbit[1])!r},"
    (k, R, T), beta0 = _ptc_turn(
        _madx_line(kick=kick, thin_quads=True), icase=5, closed_orbit=False, start=start
    )
    theirs = to_accsim_frame_second_order(k, R, T, z0, beta0, time_sign=-1.0)
    ours = second_order_one_turn_map(lat, orbit, step=2.5e-4)
    assert np.max(np.abs(theirs.k[[X, PX]] - ours.k[[X, PX]])) < 1e-10  # same image
    assert np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE])) < 1e-9
    assert np.max(np.abs(theirs.T[FIVE] - ours.T[FIVE])) < TURN_ATOL
    assert np.max(np.abs(ours.T)) > 100.0


def _ptc_single(element: str, z: np.ndarray, *, model: int = 2, nst: int = 5):
    """One element's PTC map about ``z``.

    ``model`` selects PTC's splitting family (1 = drift-kick-drift, 2 = matrix-kick-matrix)
    and ``nst`` the number of integration steps per element. The defaults are this file's
    long-standing ring settings; P2 (iv) passes them explicitly because the quantity it
    measures is *below* the error a default ``nst`` leaves behind — see
    :func:`test_the_kinematic_flag_closes_the_gap_to_ptc`.
    """
    with madx_session() as madx:
        madx.input(f"""
        beam, particle=proton, gamma={GAMMA0!r};
        el: {element};
        ring: line = (el);
        use, sequence=ring;
        ptc_create_universe;
        ptc_create_layout, model={model}, method=6, nst={nst}, exact=true;
        ptc_twiss, icase=5, no=2, betx=1, bety=1, x={float(z[X])!r}, px={float(z[PX])!r},
                   maptable;
        ptc_end;
        """)
        return ptc_maptable(madx), beam_beta0(madx, "ring")


def test_the_thick_quadrupole_departs_from_ptc_off_axis_as_l2_says() -> None:
    r"""About a point with an orbit angle, accsim's thick quadrupole misses PTC's exact one
    at second order by an amount that grows with the angle; the bend and the drift do not.

    L2 shipped the quadrupole "exact in ``delta`` and paraxial in the angles" — the
    kinematic ``(px^2 + py^2)^2 / 8`` of the exact Hamiltonian is dropped. On the axis
    that is invisible to ``T`` (it is quartic), which is why the design-orbit gates pass
    at ``1e-11``. About ``px_co != 0`` its third derivative ``3 px_co`` enters ``T``:
    measured ``4.2e-5, 5.6e-5, 8.4e-5`` on ``T[x, px, px]`` for ``px_co = 3.3e-5, 6.6e-5,
    1.3e-4`` at ``x_co = 3.8e-4``, and ``8e-9`` on ``R``. The exact sector bend (L3) and
    the exact drift (L1) agree about the same point to ``2e-12``. Recorded as the price of
    L2's approximation at this order, and named in ``docs/ROADMAP.md``.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    z0 = np.array([3.8e-4, -6.6e-5, 0.0, 0.0, 0.0, 0.0])
    F = np.ix_(FIVE, FIVE, FIVE)
    for element, ours_el in (
        (f"sbend, l={LB}, angle={ANGLE}, {NO_FRINGE}", Dipole(LB, ANGLE)),
        ("drift, l=0.5", Drift(0.5)),
    ):
        (k, R, T), beta0 = _ptc_single(element, z0)
        theirs = to_accsim_frame_second_order(k, R, T, z0, beta0, time_sign=-1.0)
        ours = taylor_expand(lambda s, e=ours_el: e.track(s, ref), z0, step=2.5e-4)
        assert np.max(np.abs(theirs.T[F] - ours.T[F])) < 1e-10, element
    misses = []
    for scale in (0.5, 1.0, 2.0):
        z = z0.copy()
        z[PX] *= scale
        (k, R, T), beta0 = _ptc_single(f"quadrupole, l={LQ}, k1={KF}", z)
        theirs = to_accsim_frame_second_order(k, R, T, z, beta0, time_sign=-1.0)
        ours = taylor_expand(lambda s: Quadrupole(LQ, KF).track(s, ref), z, step=2.5e-4)
        misses.append(np.max(np.abs(theirs.T[F] - ours.T[F])))
        assert np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE])) < 1e-7
    assert misses[0] < misses[1] < misses[2]  # grows with the orbit angle
    assert 1e-5 < misses[1] < 1e-4  # and it is not the floor


# --------------------------------------------------------------------------
# P2 (iv) — the gap above, closed
# --------------------------------------------------------------------------

#: PTC settings at which the quadrupole's map has stopped moving. ``nst`` is swept, not
#: guessed: with the drift-kick-drift family the map moves ``4.0e-6`` from ``nst=1`` to the
#: limit, ``4.1e-12`` from ``nst=10``, and ``6.4e-14`` from ``nst=20`` — so 40 steps is
#: converged to well under this file's ``1e-10`` differencing floor. The default ``nst=5``
#: used by the ring fixtures is **not**: it sits ``6.4e-8`` out, which would swamp the
#: ``8e-10`` residual the gate below rests on.
PTC_CONVERGED = {"model": 1, "nst": 40}


def _ptc_quad_second_order(z: np.ndarray, **ptc):
    """PTC's second-order map of the thick quadrupole, in accsim's frame."""
    (k, R, T), beta0 = _ptc_single(f"quadrupole, l={LQ}, k1={KF}", z, **ptc)
    return to_accsim_frame_second_order(k, R, T, z, beta0, time_sign=-1.0)


def test_the_kinematic_flag_closes_the_gap_to_ptc() -> None:
    r"""P2 (iv): ``kinematic_slices`` takes P1's ``5.6e-5`` gap to the differencing floor.

    The test above records the price of L2's paraxial quadrupole: about a point with an
    orbit angle, its second-order coefficients miss PTC's exact map by ``5.6e-5``, almost
    all of it in ``T[x, px, px]``. That is the term ``H_kin = (1+delta) - sqrt((1+delta)^2
    - p^2) - p^2/(2(1+delta))`` — a function of the momenta alone, hence with an explicit
    flow of its own, hence interleavable with the paraxial map it was split from.

    Interleaved 64 times the residual is ``8.2e-10`` on ``T`` and ``2.7e-13`` on ``R``,
    from ``5.6e-5`` and ``8.2e-9``. Four and a half decades on the second-order
    coefficients, four on the first-order ones — and the first-order improvement matters
    because a correction that fixed ``T`` while spoiling ``R`` would be a different bug
    wearing this one's clothes.

    **The ladder, not the number, is the physics.** The symmetric interleave is second
    order in the slice length, so the residual should quarter per doubling of ``n``:
    measured ``3.4e-6, 8.4e-7, 2.1e-7, 5.2e-8, 1.3e-8, 3.3e-9, 8.2e-10`` for
    ``n = 1 ... 64``. From ``n = 2`` on, every ratio is ``4.00`` to half a percent; the
    first step is ``4.08``, because one slice is not yet in the asymptotic regime — the
    ``1/n^4`` term of the expansion is still worth two percent there. That is asserted as
    what it is rather than absorbed into a wider band on the whole ladder, since a wider
    band is exactly what would let a wrong exponent through. A term with the wrong
    coefficient would
    converge just as prettily — to the wrong place — so the ladder is asserted *together
    with* the endpoint, and the endpoint is asserted against a reference that has been
    swept to convergence.

    Below ``8e-10`` there is nothing left to measure here: ``taylor_expand`` differences a
    tracked map twice at ``step = 2.5e-4``, and the second derivative's rounding noise
    bottoms out near ``1e-10``. Pushing ``n`` to 256 does not improve the residual, it
    starts to wander (``1.2e-10``, then ``2.5e-10`` at 512). The *tracking* leg,
    ``tests/reference/test_kinematic_quadrupole_xtrack.py``, has no such floor and follows
    the same convergence four decades further down.
    """
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    z0 = np.array([3.8e-4, -6.6e-5, 0.0, 0.0, 0.0, 0.0])
    F = np.ix_(FIVE, FIVE, FIVE)
    theirs = _ptc_quad_second_order(z0, **PTC_CONVERGED)

    def gap(n: int) -> tuple[float, float]:
        quad = Quadrupole(LQ, KF) if n == 0 else Quadrupole(LQ, KF, kinematic_slices=n)
        ours = taylor_expand(lambda s: quad.track(s, ref), z0, step=2.5e-4)
        return (
            float(np.max(np.abs(theirs.T[F] - ours.T[F]))),
            float(np.max(np.abs(theirs.R[FIVE] - ours.R[FIVE]))),
        )

    off_t, off_r = gap(0)
    assert off_t == pytest.approx(5.63e-5, rel=0.05)  # P1's measured gap, reproduced
    assert off_r == pytest.approx(8.2e-9, rel=0.05)

    on_t, on_r = gap(64)
    assert on_t < 1e-9
    assert on_r < 1e-12
    assert off_t / on_t > 1e4
    assert off_r / on_r > 1e4

    ladder = {n: gap(n)[0] for n in (1, 2, 4, 8, 16, 32, 64)}
    ns = sorted(ladder)
    for coarse, fine in zip(ns[1:], ns[2:], strict=False):
        assert ladder[coarse] / ladder[fine] == pytest.approx(4.0, rel=0.02), (coarse, fine)
    # One slice is still outside the asymptotic regime, by a measured 2%.
    assert ladder[1] / ladder[2] == pytest.approx(4.08, rel=0.01)


def test_ptcs_two_splitting_families_land_on_the_same_map() -> None:
    r"""The arbiter is the Hamiltonian's flow, not any one code's way of splitting it.

    There is no closed form for the exact quadrupole, so PTC integrates it — and it offers
    two different integrators: ``model=1`` alternates *exact drifts* with thin kicks, and
    ``model=2`` alternates the *paraxial matrix* with thin kicks. Run to convergence they
    agree to ``2.1e-11``, below this file's ``1e-10`` differencing floor, i.e. they are the
    same map. accsim's interleave is a third splitting, and xtrack's ``yoshida4``
    drift-kick-drift a fourth; all four meet.

    That is what licenses the gate above. accsim's own split reuses the paraxial flow, so
    it is at least a cousin of PTC's ``model=2`` — if the agreement only held against that
    family, the cross-check would be partly circular. It holds against ``model=1`` too,
    which shares nothing with accsim's construction but the Hamiltonian.

    **How far from converged the defaults are.** At ``nst=1`` the two families sit
    ``4.1e-6`` and ``3.4e-8`` from the limit and ``3.4e-6`` from *each other* — larger than
    the ``8.2e-10`` residual P2 (iv) is measured at, and in ``model=1``'s case comparable
    to L2's whole ``5.6e-5`` gap. Whichever one had been picked without sweeping ``nst``
    would have produced a number, and the number would have been the integrator's.
    """
    z0 = np.array([3.8e-4, -6.6e-5, 0.0, 0.0, 0.0, 0.0])
    F = np.ix_(FIVE, FIVE, FIVE)
    gold = _ptc_quad_second_order(z0, **PTC_CONVERGED)
    other = _ptc_quad_second_order(z0, model=2, nst=40)
    assert np.max(np.abs(gold.T[F] - other.T[F])) < 1e-10  # the same map, at the floor
    assert np.max(np.abs(gold.R[FIVE] - other.R[FIVE])) < 1e-13

    # Neither family is anywhere near that map at one step per element.
    coarse = [_ptc_quad_second_order(z0, model=m, nst=1) for m in (1, 2)]
    misses = [float(np.max(np.abs(gold.T[F] - c.T[F]))) for c in coarse]
    assert min(misses) > 1e-8  # both are above the residual the gate above rests on
    assert np.max(np.abs(coarse[0].T[F] - coarse[1].T[F])) > 1e-6  # and they disagree


def test_frame_maps_round_trip_and_reduce_to_the_first_order_transform() -> None:
    ref = ReferenceParticle.from_gamma(MASS0, GAMMA0)
    z0 = np.array([1e-3, -2e-4, 5e-4, 1e-4, 2e-3, 3e-3])
    phi, phi_inv = frame_maps(z0, ref.beta0)
    assert np.max(np.abs(phi_inv.k - z0)) < 1e-15
    m = np.diag([1.0, 1.0, 1.0, 1.0, 1.0 / ref.beta0, ref.beta0])
    assert np.max(np.abs(frame_maps(np.zeros(DIM), ref.beta0)[0].R - m)) < 1e-15
    assert (
        abs(
            frame_maps(np.zeros(DIM), ref.beta0)[0].T[DELTA, DELTA, DELTA]
            - ref.beta0 / (2 * ref.gamma0**2)
        )
        < 1e-15
    )
    both = phi.then(phi_inv)
    assert np.max(np.abs(both.R - np.eye(DIM))) < 1e-13
    assert np.max(np.abs(both.T)) < 1e-13
