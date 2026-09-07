r"""What xtrack can and cannot say about a wiggler (T1, gate 7).

**xtrack has no wiggler.** Its element inventory — 110 public classes, down to
``ElectronCooler``, ``NonLinearLens`` and ``Wire`` — contains no wiggler and no undulator,
and MAD-X does not merely reject the keyword but takes the *process* down on it
(``+=+=+= fatal: unknown class type: wiggler``), which is why no MAD-X leg exists for this
milestone at all. So this file does not cross-check accsim's wiggler against a reference
implementation. There is none. What it does instead is pin, **as a mechanism**, the two
ways a wiggler can be spelled out of the elements xtrack *does* have, and the opposite
planes in which each of them fails.

That distinction matters enough to state plainly: **no tolerance gate is pre-committed here
against either construction.** A tolerance would be a claim that one of them approximates
the magnet, and neither does. The assertions below are about *structure* — a matrix entry
that is identically zero, and stays identically zero as the model is refined — which is the
shape S2's solenoid leg took for the same reason.

  * **A straight-reference sliced line** (``angle = 0``, alternating ``k0``) puts the
    focusing in **no plane at all**. Its ``R43`` is ``+0.000000e+00`` at 8, 64 and 256
    slices alike — it does not converge to the right answer, it is structurally incapable of
    producing one, because the vertical focusing comes from ``dby/dy = -dbs/ds``, which any
    piecewise-constant-in-``s`` model sets to exactly zero inside every slice.
  * **An alternating-curved-bend line** (``h = k0``) puts a focusing term of the same size
    in the **horizontal** plane, which is the plane a real wiggler leaves free, and *still*
    leaves ``R43`` at zero.

So the two natural spellings are near mirror images of each other and neither is the
magnet: one focuses in no plane, the other in the wrong one. The arbiter that decides the
coefficient is the direct field integration in ``tests/analytic/test_wiggler.py``.

**T2 looked for a radiation leg here and there is none either.** The obvious one — read
xtrack's own ``I1..I5`` off the cos-sampled stack above, which *does* converge to the
wiggler's curvature distribution — does not exist: this xtrack exposes no radiation-integral
accessor on the twiss table at all (no ``get_radiation_integrals``, no ``rad_int_*``). What it
does have is ``radiation_analysis=True``, which needs a 6D twiss with a cavity and returns the
**damped-map eigenanalysis**, a route ``accsim.radiation``'s own docstring already records as
differing from the integral method at the ~1% level — a second approximation to argue with
rather than an independent leg. And it reports ``eneloss_turn = 0.0`` for the straight
construction anyway: with ``angle = 0`` there is no curvature for it to radiate on, whatever
``k0`` says. That is the same structural blindness as ``R43`` above, in the one quantity T2
cared about, so T2's arbiters are its closed forms, a resolved quadrature, and the
``I1 == alpha_c * C`` identity — all in ``tests/analytic/test_wiggler_radiation.py``.

**Cost.** Each ``xt.Line`` build JIT-compiles a fresh C kernel (~12 s, one leaked ``.pyd``),
so the lines here are module-scoped and built exactly once. Run this file with ``-n 8``.
"""

from __future__ import annotations

import numpy as np
import pytest

from accsim import ReferenceParticle, Wiggler

pytestmark = pytest.mark.reference

xt = pytest.importorskip("xtrack")

MASS0 = 0.51099895069e6  # electron, eV
ENERGY0 = 1.0e9
PERIOD = 0.1
PERIODS = 10
H0 = 0.4496887457113502  # B0 = 1.5 T at 1 GeV: the roadmap entry's probe magnet
LENGTH = PERIOD * PERIODS


def _ref() -> ReferenceParticle:
    """``q0 = +1``, matching ``xt.Particles`` below — as the other reference files do.

    The charge is irrelevant to every assertion here (the map is even in ``h0`` and
    ``gamma0`` is charge-blind), but pairing accsim's ``-1`` with xtrack's ``+1`` would be a
    difference a later reader has to rule out. `H0` is quoted directly rather than converted
    from tesla for the same reason.
    """
    return ReferenceParticle.from_total_energy(MASS0, ENERGY0, charge=1.0)


def _rmatrix(elements: list[object]) -> np.ndarray:
    """6x6 R-matrix of an xtrack line, or skip if the JIT cannot build."""
    ref = xt.Particles(mass0=MASS0, q0=1, energy0=ENERGY0)
    line = xt.Line(elements=elements)
    line.particle_ref = ref
    try:
        line.build_tracker()
        res = line.get_R_matrix(particle_on_co=ref.copy())
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"xtrack JIT compilation unavailable: {type(exc).__name__}: {exc}")
    return np.asarray(res["R_matrix"])


def _straight_slices(n_per_period: int) -> list[object]:
    """The first spelling: straight bends whose ``k0`` alternates as ``cos(ks)``.

    ``angle = 0`` everywhere, so the reference trajectory is straight and the ``k0`` is a
    pure transverse kick — which is what a wiggler's field *is*, sampled. The sampling is
    the midpoint of each slice, so the ``I2`` of the staircase converges to the wiggler's.
    """
    n = n_per_period * PERIODS
    ds = LENGTH / n
    k = 2.0 * np.pi / PERIOD
    return [xt.Bend(length=ds, angle=0.0, k0=H0 * np.cos(k * (i + 0.5) * ds)) for i in range(n)]


def _curved_slices(n_per_period: int) -> list[object]:
    """The second spelling: the same field, but as *curved* bends (``h = k0``).

    The reference frame now follows the wiggle, which is the other thing "a list of bends"
    can reasonably mean — and it is the construction that puts a weak-focusing term in the
    horizontal plane, where the magnet has none.
    """
    n = n_per_period * PERIODS
    ds = LENGTH / n
    k = 2.0 * np.pi / PERIOD
    return [xt.Bend(length=ds, angle=H0 * np.cos(k * (i + 0.5) * ds) * ds) for i in range(n)]


@pytest.mark.parametrize("n_per_period", [8, 64])
def test_a_straight_sliced_xtrack_line_focuses_in_no_plane(n_per_period: int) -> None:
    """``R43`` is **identically zero** at every slice count — a mechanism, not a tolerance.

    The vertical focusing a wiggler has comes from the longitudinal field component, and
    that component exists only because ``b_y`` varies along ``s``. A model that holds the
    field constant inside each slice has ``dbs/ds = 0`` in every slice by construction, so
    no amount of slicing produces any vertical focusing at all. Refining the model does not
    approach the answer; it stays at exactly zero.

    Asserted as an exact ``0.0``, deliberately, rather than as a small number: a bound like
    ``< 1e-3`` would be satisfied by a model that was merely converging slowly, and the
    point of this test is that nothing is converging.
    """
    R = _rmatrix(_straight_slices(n_per_period))
    assert R[3, 2] == 0.0  # R43: no vertical focusing, at any n

    # ...and its distance from the shipped map's vertical block does not shrink with n: it
    # is pinned at the whole of the wiggler's focusing.
    wig = Wiggler(PERIOD, H0, PERIODS)
    shipped = wig.matrix(_ref())
    assert abs(R[3, 2] - shipped[3, 2]) > 9e-2


def test_xtrack_confirms_the_kinematic_coefficients_it_can_see() -> None:
    r"""The one thing this arbiter *can* see, it confirms — and it is not the focusing.

    The sliced line's focusing entries are dead: ``R21``, ``R43`` exactly ``0.0`` and
    ``R33``, ``R44`` exactly ``1.0``, bit for bit. But its *drift lengths* are not ``L``, and
    what they are is exactly what accsim's own non-paraxial integration found independently:

        R12 = L (1 + 3 theta^2 / 4),      R34 = L (1 + theta^2 / 4).

    Both to within ``1e-3`` of those coefficients, the residual being the 640-slice
    discretisation. This is a real cross-check and worth having: the kinematic remainder is
    the piece of the wiggler's map that a code with no wiggler in it *can* still express —
    xtrack's drift is exact where accsim's element is paraxial — so the two codes agree on
    the term accsim deliberately does not ship and disagree, structurally and totally, on
    the term it does.

    That split is the honest summary of what a reference code buys this milestone. It
    validates the approximation accsim made; it cannot validate the physics accsim added.
    """
    R = _rmatrix(_straight_slices(64))
    theta = Wiggler(PERIOD, H0, PERIODS).deflection

    # Nothing focuses, in either plane, to the last bit.
    assert R[1, 0] == 0.0 and R[3, 2] == 0.0
    assert R[2, 2] == 1.0 and R[3, 3] == 1.0

    # ...and both drift lengths carry the wiggle's own path, with the two coefficients
    # accsim measured by integrating the Lorentz force non-paraxially.
    assert (R[0, 1] - LENGTH) / (LENGTH * theta**2) == pytest.approx(0.75, rel=2e-3)
    assert (R[2, 3] - LENGTH) / (LENGTH * theta**2) == pytest.approx(0.25, rel=2e-3)

    # accsim's shipped map, being paraxial, has neither -- both blocks are exact drifts of
    # length L. The gap is the recorded approximation, not a disagreement about the magnet.
    shipped = Wiggler(PERIOD, H0, PERIODS).matrix(_ref())
    assert shipped[0, 1] == pytest.approx(LENGTH, rel=1e-15)


def test_the_curved_construction_puts_the_focusing_in_the_wrong_plane() -> None:
    """Alternating **curved** bends focus horizontally — the plane a wiggler leaves free.

    Same field, same slicing, one modelling choice different (the reference frame follows
    the wiggle), and the focusing term of the right magnitude appears in the *other* plane.
    ``R21`` lands near ``-9.9e-02`` where the real magnet's is zero to ``1e-9``, and ``R43``
    is still exactly zero.

    This is the finding worth carrying forward: the two obvious ways to build a wiggler out
    of bends are near mirror images of each other, and a session that checked only one of
    them could conclude either that a wiggler does not focus or that it focuses
    horizontally. Both conclusions are wrong, and neither reference code can say so.
    """
    R = _rmatrix(_curved_slices(64))

    assert R[1, 0] < -9e-2  # a real horizontal focusing term...
    assert R[3, 2] == 0.0  # ...and still nothing vertical

    # The two constructions' errors are in opposite planes and of the same size: this is
    # the mirror-image statement, asserted rather than described.
    straight = _rmatrix(_straight_slices(64))
    shipped = Wiggler(PERIOD, H0, PERIODS).matrix(_ref())
    curved_wrong = abs(R[1, 0] - shipped[1, 0])  # horizontal, should be 0
    straight_wrong = abs(straight[3, 2] - shipped[3, 2])  # vertical, should be -9.94e-2
    assert curved_wrong == pytest.approx(straight_wrong, rel=0.05)
