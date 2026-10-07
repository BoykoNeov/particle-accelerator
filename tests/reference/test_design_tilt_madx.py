r"""V1 — the design tilt against MAD-X's ``TILT``: the survey out of the plane, and the optics.

Marked ``reference``: skips when cpymad is unavailable.

This is where the **sense** of the tilt is pinned. Everything in
``tests/analytic/test_design_tilt.py`` that is sign-blind (the ring rolled as a whole, the
tunes, the conjugation) would pass with the tilt running the other way; the gates here
would not:

- the 3D survey, all six columns, on F3 (a horizontal, a ``0.7``-tilted, a vertical and a
  horizontal bend — the order matters, rotations about different axes do not commute) and on
  the dogleg ring. A flipped tilt sense misses F3's end point by metres;
- the **signed** ``D_y`` and ``D_py`` of the dogleg ring, element by element. Compared signed,
  never as a magnitude: this is the one number that ties the *map's* sense of the tilt to the
  *survey's*;
- ``alfa``, which the dogleg moves through its own ``h D_y`` — and which caught accsim's
  identity route leaving out the vertical half of the slip (3e-6) before this file existed.

MAD-X's twiss ``dx``/``dy`` are derivatives with respect to ``PT``, an energy deviation;
accsim's are with respect to ``delta``, a momentum one, so ``D_delta = beta0 D_PT`` (see
``_madx.py``). At 3 GeV the factor is ``1 - 1.5e-8``, and it is applied rather than dropped.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analytic"))

from _madx import beam_beta0, import_madx, madx_session  # noqa: E402
from test_design_tilt import ANG, F3, VA, dogleg_ring, f3_line  # noqa: E402

from accsim.geometry import survey  # noqa: E402
from accsim.twiss import closed_twiss, momentum_compaction, propagate_twiss, tunes  # noqa: E402

PI2 = math.pi / 2
COLUMNS = ("x", "y", "z", "theta", "phi", "psi")


def _f3_sequence() -> str:
    defs, members = [], []
    for i, (length, angle, tilt) in enumerate(F3):
        defs.append(f"b{i}: sbend, l={length!r}, angle={angle!r}, tilt={tilt!r};")
        defs.append(f"d{i}: drift, l=0.8;")
        members += [f"b{i}", f"d{i}"]
    return "\n".join(defs) + f"\nprobe: line=({', '.join(members)});\n"


def _dogleg_sequence() -> str:
    return f"""
    qf: quadrupole, l=0.5, k1=0.30;
    qd: quadrupole, l=0.5, k1=-0.30;
    b:  sbend, l=1.5, angle={ANG!r};
    d:  drift, l=0.6;
    vb1: sbend, l=0.3, angle={VA!r}, tilt={PI2!r};
    vb2: sbend, l=0.3, angle={-VA!r}, tilt={PI2!r};
    dd: drift, l=1.0;
    cell: line=(qf, d, b, d, qd, d, b, d);
    probe: line=(cell, vb1, dd, vb2, 3*cell);
    """


def _run(sequence: str, *, twiss: bool):
    import_madx()
    with madx_session() as madx:
        madx.input(sequence + "\nbeam, particle=electron, energy=3.0;\n")
        madx.use(sequence="probe")
        madx.survey()
        sv = madx.table.survey
        # The last row repeats the one before it ($end); the N + 1 boundaries are the rest.
        surv = {c: np.array(getattr(sv, c), dtype=float)[:-1] for c in COLUMNS}
        if not twiss:
            return surv, None
        madx.twiss()
        tw = madx.table.twiss
        cols = {
            c: np.array(getattr(tw, c), dtype=float)[:-1]
            for c in ("betx", "bety", "dx", "dy", "dpx", "dpy")
        }
        cols["q1"], cols["q2"] = float(madx.table.summ.q1[0]), float(madx.table.summ.q2[0])
        cols["alfa"] = float(madx.table.summ.alfa[0])
        cols["beta0"] = beam_beta0(madx, "probe")
        return surv, cols


@pytest.fixture(scope="module")
def f3():
    return _run(_f3_sequence(), twiss=False)[0]


@pytest.fixture(scope="module")
def dogleg():
    return _run(_dogleg_sequence(), twiss=True)


def _accsim_columns(table) -> dict[str, np.ndarray]:
    return {
        "x": table.X, "y": table.Y, "z": table.Z,
        "theta": table.theta, "phi": table.phi, "psi": table.psi,
    }  # fmt: skip


@pytest.mark.parametrize("column", COLUMNS)
def test_the_f3_survey_matches_madx_in_every_column(f3, column: str) -> None:
    """Measured ``<= 1.8e-15`` between MAD-X and xtrack on 2026-10-06; accsim's walk lands
    on both."""
    ours = _accsim_columns(survey(f3_line()))[column]
    assert ours.shape == f3[column].shape
    assert np.allclose(ours, f3[column], rtol=0.0, atol=1e-13)


def test_f3_really_leaves_the_plane_in_madx(f3) -> None:
    """Control on the fixture: the columns above are not all zero, so they gate something."""
    assert np.max(np.abs(f3["y"])) > 0.1
    assert np.max(np.abs(f3["phi"])) > 0.1
    assert np.max(np.abs(f3["psi"])) > 0.05  # measured 0.0659


@pytest.mark.parametrize("column", COLUMNS)
def test_the_dogleg_survey_matches_madx_in_every_column(dogleg, column: str) -> None:
    surv, _ = dogleg
    ours = _accsim_columns(survey(dogleg_ring()))[column]
    assert np.allclose(ours, surv[column], rtol=0.0, atol=1e-12)


def test_the_dogleg_tunes_match_madx(dogleg) -> None:
    _, tw = dogleg
    qx, qy = tunes(dogleg_ring())
    assert qx == pytest.approx(tw["q1"], abs=1e-9)
    assert qy == pytest.approx(tw["q2"], abs=1e-9)


def test_the_dogleg_optics_match_madx_element_by_element(dogleg) -> None:
    """Betas, ``D_x``, and the **signed** ``D_y``/``D_py`` at every boundary.

    The probe measured MAD-X against xtrack at ``3.4e-10`` on ``D_y`` (peak ``2.4e-2`` m);
    accsim's linear maps are the same linear maps, so the bar is set from that, with the
    ``beta0`` conversion applied.
    """
    _, tw = dogleg
    lat = dogleg_ring()
    pts = propagate_twiss(lat, closed_twiss(lat))
    b0 = tw["beta0"]
    ours = {
        "betx": np.array([p.beta_x for p in pts]),
        "bety": np.array([p.beta_y for p in pts]),
        "dx": np.array([p.disp_x for p in pts]),
        "dpx": np.array([p.disp_px for p in pts]),
        "dy": np.array([p.disp_y for p in pts]),
        "dpy": np.array([p.disp_py for p in pts]),
    }
    for col in ("betx", "bety"):
        assert np.allclose(ours[col], tw[col], rtol=1e-8, atol=0.0), col
    for col in ("dx", "dpx", "dy", "dpy"):
        assert np.allclose(ours[col], b0 * tw[col], rtol=0.0, atol=1e-9), col
    assert np.max(np.abs(tw["dy"])) > 1e-2  # the signed comparison above has something to see


def test_the_dogleg_momentum_compaction_matches_madx(dogleg) -> None:
    """``alfa`` includes the dogleg's vertical ``h D_y`` — and so must both accsim routes.
    The flat-ring agreement on record is ``1e-10`` (``test_fodo_twiss_madx.py``)."""
    _, tw = dogleg
    lat = dogleg_ring()
    assert momentum_compaction(lat) == pytest.approx(tw["alfa"], rel=1e-10)
    assert momentum_compaction(lat, method="quadrature", slices=1024) == pytest.approx(
        tw["alfa"], rel=1e-8
    )
