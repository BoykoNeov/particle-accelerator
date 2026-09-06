r"""Cross-check the taper profile (Q1) against MAD-X — and refuse MAD-X's own tapering.

Marked ``reference``: skips when cpymad is unavailable.

**Which half of MAD-X can see this milestone.** Its ``TWISS`` with ``radiate=true`` reports a
``pt`` column: the local energy deviation of the radiating ring, element by element, without
being asked to compensate anything. That is the same sawtooth ``accsim.taper_profile``
computes and xtrack stores as ``delta_taper``, arrived at by a third implementation, and it
is a real leg. Its ``TWISS, TAPERING``, on the same ring, is **not**: ``ktap`` reads
identically zero with tapering on and off, and the radiation orbit moves by under 1% where
xtrack collapses it by a factor 14,300. Naming the half that works is worth more than
counting codes — P3 (b)'s refusal of PTC in the same shape — so both facts are asserted
here rather than one being quietly omitted.

**The centring is where the two references differ interestingly.** xtrack *imposes* the zero
mean (``delta0='zero_mean'``). MAD-X imposes nothing: its ``pt`` is just where the ring runs,
and it comes out centred to within a percent of the span anyway. That is I4's finding — a
radiating ring's fixed point is where the sag is centred — seen from outside accsim, so the
two get different tolerances for a stated reason rather than the same one for convenience.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from _madx import import_madx, madx_session  # noqa: E402
from test_closed_orbit_6d import RING, ring  # noqa: E402

from accsim.tapering import taper_profile  # noqa: E402

#: Element-by-element agreement, as a fraction of the sawtooth's height. Larger than
#: xtrack's ``0.1446 span^2`` because MAD-X's own span differs by ``1.7e-4`` relative.
MADX_GAP = 6.7e-4


def _sequence() -> str:
    """I4's ring in MAD-X: thin quadrupoles, 40 sector bends, one cavity.

    The bends carry an explicit ``k0`` as well as ``angle`` so the field and the geometry
    are both stated — Q2's distinction, written here because MAD-X allows it and
    :class:`~accsim.elements.dipole.Dipole` does not yet.
    """
    cells, focal = RING["cells"], RING["focal"]
    angle = 2.0 * math.pi / (2 * cells)
    return f"""
    b: sbend, l=1.0, angle={angle}, k0={angle};
    qf: multipole, knl:={{0, {0.5 / focal}}};
    qd: multipole, knl:={{0, {-1.0 / focal}}};
    rf: rfcavity, l=0.0, volt={RING["voltage"] / 1e6}, harmon={RING["harmonic"]}, lag=0.5;
    ring: line=({cells}*(qf, b, qd, b, qf), rf);
    beam, particle=electron, energy={RING["energy"] / 1e9}, radiate=true;
    use, sequence=ring;
    """


def _twiss(tapering: bool):
    """``(names, pt, x, ktap)`` from a ``TWISS``, with or without MAD-X's own tapering."""
    import_madx()
    with madx_session() as madx:
        madx.input(_sequence())
        madx.input("select, flag=twiss, column=name,s,x,pt,ktap;")
        madx.input("twiss, tapering;" if tapering else "twiss;")
        table = madx.table.twiss
        return (
            list(table.name),
            np.array(table.pt),
            np.array(table.x),
            np.array(table.ktap) if "ktap" in list(table) else None,
        )


@pytest.fixture(scope="module")
def plain():
    return _twiss(tapering=False)


def test_the_untapered_pt_column_is_the_same_sawtooth(plain) -> None:
    """Height and shape, against a third implementation.

    MAD-X reports ``pt`` at each element's **exit**, so the comparison is against accsim's
    boundary ramp rather than its per-element midpoints — matching the quantity, not the
    array shape, which is the step an index-aligned comparison would silently skip.
    """
    names, pt, _, _ = plain
    profile = taper_profile(ring()[0])

    keep = [i for i, n in enumerate(names) if n.split(":")[0] in ("b", "qf", "qd")]
    assert len(keep) == 100

    theirs = pt[keep]
    ours = profile.boundary[1 : len(keep) + 1]

    assert float(theirs.max() - theirs.min()) == pytest.approx(profile.span, rel=3e-4)
    assert np.max(np.abs(ours - theirs)) / profile.span == pytest.approx(MADX_GAP, rel=0.1)


def test_madx_centres_the_sawtooth_without_being_asked(plain) -> None:
    """No convention, no argument — and it still lands within a percent of centre.

    Gated loosely *on purpose*: this is a physics claim about where a radiating ring runs,
    not a check that two codes implement one convention. accsim's own mean, which **is** a
    convention, is gated at ``1e-12`` of the span in the analytic suite.
    """
    _, pt, _, _ = plain
    span = float(pt.max() - pt.min())
    assert abs(float(pt.mean())) < 2e-2 * span


def test_madx_and_xtrack_produce_the_same_untapered_orbit(plain) -> None:
    """``7.088e-3`` against xtrack's ``7.0898e-3``: the sag orbit is not one code's opinion."""
    _, _, x, _ = plain
    assert float(np.max(np.abs(x))) == pytest.approx(7.0898e-3, rel=1e-3)


def test_madxs_own_tapering_is_not_an_arbiter_for_this_milestone(plain) -> None:
    """The refusal, asserted rather than asserted-about.

    ``TWISS, TAPERING`` leaves ``ktap`` at zero and barely moves the orbit it is supposed to
    remove. If a future MAD-X makes this work, this test fails and the entry gets rewritten
    — which is the point of writing a refusal as a test instead of a sentence.
    """
    _, _, x_plain, ktap_plain = plain
    _, _, x_taper, ktap_taper = _twiss(tapering=True)

    if ktap_plain is not None:
        assert np.max(np.abs(ktap_plain)) == 0.0
        assert np.max(np.abs(ktap_taper)) == 0.0

    plain_max = float(np.max(np.abs(x_plain)))
    taper_max = float(np.max(np.abs(x_taper)))
    assert taper_max / plain_max > 0.99  # xtrack's ratio on the same ring is 7e-5
