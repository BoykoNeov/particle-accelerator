"""Closed orbit: where the beam actually goes when the machine is imperfect.

Every optics quantity in the package so far — beta, tune, chromaticity,
dispersion — describes motion *about* the design orbit and assumes the beam is
on it. A real machine never is: a steering error, a misaligned quadrupole or a
deliberate :class:`~accsim.elements.corrector.Corrector` deflects the beam, and
the trajectory that closes on itself turn after turn is displaced. That
trajectory is the **closed orbit**, and steering it back is the single most
common operational task in an accelerator.

**The one equation.** A corrector makes the one-turn map affine,
``x -> M x + k`` (see :meth:`accsim.Lattice.transfer_map`). An orbit closes when
it is a fixed point of that map,

    x_co = M4 x_co + k4    =>    (I - M4) x_co = k4,

solved on the 4D transverse subspace ``(x, px, y, py)`` at ``delta = 0``. This is
*literally* the same solve as the matched dispersion
(:func:`~accsim.twiss._matched_dispersion`, ``D = (I - M4)^-1 d``); only the
inhomogeneity differs — there it is the map's ``delta`` column, here the
accumulated corrector kicks. The dispersion is nothing but the closed orbit of an
off-momentum particle, so sharing the algebra is the physics, not a shortcut. By
linearity a particle at momentum ``delta`` rides ``x_co + D * delta``.

``I - M4`` is singular exactly when the map has an eigenvalue 1 — an **integer
tune**, where a kick's effect adds up in phase turn after turn and nothing
closes. The resulting ``1/sin(pi Q)`` in the single-kick closed form

    x_co(s) = theta sqrt(beta_k beta(s)) / (2 sin(pi Q)) * cos(dpsi - pi Q)

is the same statement said with Twiss parameters. That form is a *consequence*
here, used in the tests as an independent reference; the module always solves the
fixed point.

**Feed-down: when the orbit stops being a linear solve.** Everything above is the
linear theory, and it is exact only while every element's map is its matrix. A
sextupole's is not. Its linear map is a drift *only because* the Jacobian is
taken at ``(x, y) = 0``; expanding the kick about an orbit offset
``(x_co, y_co)`` (see :mod:`accsim.elements.sextupole`) splits it into

    dipole      theta_x = -1/2 k2l (x_co^2 - y_co^2),  theta_y = +k2l x_co y_co
    normal quad k1l_eff  = +k2l x_co
    skew quad   k1sl_eff = +k2l y_co
    sextupole   unchanged

so an off-axis sextupole *is* a corrector, a gradient error and a coupling source
at once. An off-axis **octupole** (J3) is all of that *and* a sextupole: its cubic
kick reaches two orders below itself, adding
``k2l_eff = k3l x_co`` and ``k2sl_eff = k3l y_co`` to the same four terms, with the
gradients now quadratic in the orbit (``k1l_eff = 1/2 k3l (x_co^2 - y_co^2)``) and
the dipole cubic. Two consequences run through this module:

- The dipole term depends on the orbit it displaces, so the closed orbit becomes
  the fixed point of a **nonlinear** map rather than the solve ``(I - M4) x = k4``
  — that is :func:`closed_orbit_nonlinear`, which Newtons on the tracked map.
- The quadrupole terms mean **correctors do move the optics** once a sextupole is
  off-axis: beta, the tunes and (through the skew term) the x-y coupling all
  respond to steering. The linear-order claim above is a statement about
  on-axis sextupoles, and :func:`linearised_element_maps` is how the optics
  *about* a distorted orbit are read instead — packaged as
  :func:`~accsim.twiss.propagate_twiss_on_orbit` and
  :func:`~accsim.twiss.chromaticity_on_orbit`.

**Scope, stated plainly.** The linear entry points (:func:`closed_orbit`,
:func:`propagate_orbit`, :func:`orbit_response_matrix`) remain linear,
``delta = 0`` orbit theory and are unchanged; the feed-down entry points are
opt-in, and :func:`correct_orbit` takes a ``nonlinear`` flag rather than changing
under existing callers. Beyond that:

- ``delta = 0`` throughout, linear and nonlinear alike: the longitudinal
  coordinates are not solved for (see :func:`closed_orbit_nonlinear`).
- Misalignments are not modelled as such. A quadrupole displaced by ``dx``
  produces a kick ``-k1 L dx``; represent it by placing an explicit
  :class:`~accsim.elements.corrector.Corrector` of that angle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coords import DELTA, DIM, PX, PY, X, Y
from .elements.corrector import Corrector
from .lattice import Lattice
from .symplectic import jacobian

_TRANSVERSE = [X, PX, Y, PY]  # the 4D transverse subspace (x, px, y, py)

# Above this condition number the fixed-point solve has lost every significant
# digit: the lattice is on (or numerically indistinguishable from) an integer
# resonance. Chosen well below 1/eps ~ 4.5e15 so the failure is reported rather
# than returned as a huge but meaningless orbit.
_COND_LIMIT = 1e12


class ClosedOrbitError(ValueError):
    """Raised when no closed orbit exists (or the solve for it is meaningless).

    The fixed-point condition ``(I - M4) x = k4`` fails exactly when ``M4`` has an
    eigenvalue 1 — an **integer tune** in one plane. Physically a kick then
    repeats in phase every turn and the excursion grows without bound instead of
    closing; the machine has no closed orbit to find.
    """


def _affine_4d(M: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The transverse ``(4x4, 4)`` part of a 6D affine map ``(M, k)``."""
    return M[np.ix_(_TRANSVERSE, _TRANSVERSE)], k[_TRANSVERSE]


def closed_orbit(lattice: Lattice, *, delta: float = 0.0) -> np.ndarray:
    """Closed orbit ``(x, px, y, py)`` at the entrance of a periodic ``lattice``.

    Solves the fixed point ``(I - M4) x_co = k4 + d delta`` of the one-turn affine
    map. A lattice with no :class:`~accsim.elements.corrector.Corrector` at
    ``delta = 0`` has a zero right-hand side and returns exactly zero — the design
    orbit.

    ``delta`` selects the momentum the orbit is closed at. The extra term is the
    map's own dispersive column ``d = [R16, R26, R36, R46]``, so the answer at
    ``delta`` is the corrector orbit **plus** ``D delta`` with ``D`` the matched
    dispersion — the two are the same solve, which is why
    :func:`~accsim.twiss._matched_dispersion` and this function share a formula.
    Both terms must be present: seeding
    :func:`closed_orbit_nonlinear` from the corrector part alone would start a
    whole dispersion orbit away from the answer.

    Raises :class:`ClosedOrbitError` on an integer tune, where ``I - M4`` is
    singular and no orbit closes. **That check comes first, before the zero-kick
    shortcut**, so the contract does not depend on whether the machine happens to
    be perfect: on an integer tune a kick-free lattice does have zero as *a* fixed
    point, but not as the *only* one, and returning it would quietly claim a
    uniqueness the map does not have. It also keeps
    :func:`orbit_response_matrix`'s zeroed baseline on the same code path as its
    unit-kick columns, so the two cannot disagree about whether an orbit exists.
    """
    one_turn, kick = lattice.one_turn_map()
    m4, k4 = _affine_4d(one_turn, kick)
    rhs = k4 + one_turn[_TRANSVERSE, DELTA] * delta
    A = np.eye(4) - m4
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > _COND_LIMIT:
        raise ClosedOrbitError(
            f"no closed orbit: (I - M4) is singular to working precision "
            f"(condition number {cond:.3g}), i.e. the lattice is on an **integer** "
            "tune in at least one plane, where a kick repeats in phase every turn "
            "and the excursion never closes"
        )
    if not rhs.any():
        return np.zeros(4)  # a perfect machine sits exactly on the design orbit
    return np.linalg.solve(A, rhs)


def propagate_orbit(lattice: Lattice, orbit0: np.ndarray | None = None) -> list[np.ndarray]:
    """Orbit ``(x, px, y, py)`` at every element boundary.

    Returns ``len(lattice) + 1`` points — the entrance, then the exit of each
    element in order — mirroring
    :func:`~accsim.twiss.propagate_twiss`. ``orbit0`` defaults to
    :func:`closed_orbit`, in which case the last point equals the first (that is
    what "closed" means); pass an explicit start to follow a **trajectory** down a
    transfer line instead.

    Each step is the element's own affine map ``x -> M4 x + k4``, so a corrector
    shows up as a kink in ``px`` and the orbit only starts moving in ``x``
    downstream of it.
    """
    o = closed_orbit(lattice) if orbit0 is None else np.asarray(orbit0, dtype=float)
    if o.shape != (4,):
        raise ValueError(f"orbit0 must be a length-4 (x, px, y, py) vector, got {o.shape}")
    points = [o]
    for elem in lattice.elements:
        m4, k4 = _affine_4d(elem.matrix(lattice.ref), elem.kick(lattice.ref))
        o = m4 @ o + k4
        points.append(o)
    return points


# ---------------------------------------------------------------------------
# Feed-down: the orbit that has to solve itself
# ---------------------------------------------------------------------------


class OrbitConvergenceError(ClosedOrbitError):
    """Newton failed to reach the nonlinear closed orbit.

    Distinct from a plain :class:`ClosedOrbitError`, which says the map has an
    eigenvalue 1 and no orbit exists at all. This says only that the iteration ran
    out of budget; the orbit may exist perfectly well.

    **Instability is not a cause.** A closed orbit needs ``(I - M4)`` invertible,
    not *stable*, so Newton converges happily through a machine feed-down has
    wrecked — and feed-down is in any case self-limiting, since the gradient that
    destabilises the lattice also stiffens the ``(I - M4)`` being inverted (raising
    ``k2l`` by five orders of magnitude *shrinks* the orbit; measured in the
    analytic suite). What actually raises this is a starting guess far enough out
    that the quadratic kick dominates, or a ``max_iter`` too small for the
    excursion.

    It subclasses :class:`ClosedOrbitError` so callers that already roll back on
    "no orbit" — :func:`correct_orbit` does — keep working unchanged.
    """


def _embed(x4: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """A transverse ``(x, px, y, py)`` as a full 6D state at ``zeta = 0``, ``delta``."""
    state = np.zeros(DIM)
    state[_TRANSVERSE] = x4
    state[DELTA] = delta
    return state


def _tracked_turn(lattice: Lattice, x4: np.ndarray, delta: float = 0.0) -> np.ndarray:
    """One turn of the **real** (element-by-element, nonlinear) map, transverse part.

    ``delta`` is re-imposed at the entrance of every call rather than carried, which
    is what makes the 4D fixed point well posed: see :func:`closed_orbit_nonlinear`.
    """
    state = _embed(x4, delta)
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref)
    return state[_TRANSVERSE]


def _turn_jacobian(lattice: Lattice, x4: np.ndarray, step: float, delta: float = 0.0) -> np.ndarray:
    """Central-difference ``4x4`` Jacobian of :func:`_tracked_turn` at ``x4``."""
    out = np.empty((4, 4))
    for j in range(4):
        plus, minus = x4.copy(), x4.copy()
        plus[j] += step
        minus[j] -= step
        out[:, j] = (_tracked_turn(lattice, plus, delta) - _tracked_turn(lattice, minus, delta)) / (
            2.0 * step
        )
    return out


def closed_orbit_nonlinear(
    lattice: Lattice,
    guess: np.ndarray | None = None,
    *,
    delta: float = 0.0,
    tol: float = 1e-14,
    max_iter: int = 50,
    step: float = 1e-8,
) -> np.ndarray:
    r"""Closed orbit ``(x, px, y, py)`` of the **nonlinear** one-turn map.

    The fixed point of the map particles actually follow —
    :meth:`~accsim.elements.element.Element.track` element by element, so a
    sextupole's ``x^2 - y^2`` kick acts — found by Newton:

        x <- x - (J - I)^-1 (T(x) - x),

    where ``T`` is the tracked turn and ``J`` its Jacobian. This is
    :func:`closed_orbit` when nothing in the lattice is nonlinear, and differs
    from it by the sextupole dipole feed-down ``-1/2 k2l (x_co^2 - y_co^2)``
    otherwise: that kick is a corrector whose strength is set by the very orbit it
    displaces, which is exactly what makes the problem a fixed point rather than a
    solve. The departure is ``O(k2l x_co^2)`` and vanishes as the orbit is
    steered flat, so a well-corrected machine barely notices — and a badly steered
    one does.

    ``guess`` defaults to :func:`closed_orbit`, the correct first-order answer, so
    the iteration starts one order in and typically converges in three or four
    steps. Pass an explicit start to chase a different fixed point (the map has
    more than one; the far ones are the unstable orbits outside the dynamic
    aperture, and this function makes **no claim** about which one you land on if
    you start far from the linear orbit).

    ``delta`` closes the orbit at a fixed momentum instead of on the reference one,
    which is what makes an *off-momentum* linearisation possible: a sextupole then
    sits at ``x_co + D_x delta`` and feeds down a gradient that varies with
    ``delta``. Since accsim's linear element maps carry no ``delta`` dependence of
    their own, that feed-down is the **entire** momentum dependence of the
    linearised optics — see :func:`~accsim.twiss.chromaticity_on_orbit`. The seed
    is :func:`closed_orbit` at the same ``delta``, i.e. the dispersion orbit, so
    the iteration still starts one order in.

    **The 4D subspace is the whole solve.** Newton runs on ``(x, px, y, py)`` with
    ``zeta = 0`` and ``delta`` held fixed at the entrance, matching
    :func:`closed_orbit`'s contract.
    It is not a restriction that could be lifted by iterating on all six: without
    an RF cavity there *is* no longitudinal fixed point — the drift's ``R56``
    leaves ``zeta -> zeta + const``, so ``J - I`` is exactly singular in the
    longitudinal block and 6D Newton has nothing to converge to. With an RF cavity
    present ``zeta`` slips through the turn and feeds back into ``delta``, so the
    4D fixed point found here is not the 6D one; the full 6D orbit is out of scope
    here exactly as it is for the linear solve.

    Raises :class:`OrbitConvergenceError` if Newton does not reach ``tol``
    (max-abs residual, in the mixed units of the state vector), and
    :class:`ClosedOrbitError` from the default ``guess`` if the *linear* lattice
    has no closed orbit to start from.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}")

    if guess is None:
        x = closed_orbit(lattice, delta=delta)
    else:
        x = np.asarray(guess, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"guess must be a length-4 (x, px, y, py) vector, got {x.shape}")
        x = x.copy()

    residual = _tracked_turn(lattice, x, delta) - x
    for _ in range(max_iter):
        if np.max(np.abs(residual)) < tol:
            return x
        A = _turn_jacobian(lattice, x, step, delta) - np.eye(4)
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > _COND_LIMIT:
            raise ClosedOrbitError(
                f"no closed orbit: the tracked map's (J - I) is singular to working "
                f"precision (condition number {cond:.3g}) at the current iterate, i.e. "
                "the machine — feed-down gradient included — sits on an **integer** tune"
            )
        x = x - np.linalg.solve(A, residual)
        residual = _tracked_turn(lattice, x, delta) - x
    if np.max(np.abs(residual)) < tol:
        return x
    raise OrbitConvergenceError(
        f"the nonlinear closed orbit did not converge in {max_iter} Newton steps: "
        f"max residual {np.max(np.abs(residual)):.3g} > tol {tol:.3g}. Either the "
        "orbit excursion is large enough that the sextupole kick dominates, or the "
        "feed-down gradient has pushed the machine outside its stable tune range"
    )


def propagate_orbit_nonlinear(
    lattice: Lattice, orbit0: np.ndarray | None = None, *, delta: float = 0.0
) -> list[np.ndarray]:
    """Nonlinear orbit ``(x, px, y, py)`` at every element boundary.

    The counterpart of :func:`propagate_orbit`: ``len(lattice) + 1`` points, but
    each step is the element's **tracked** map rather than its affine one, so the
    orbit through a sextupole is bent by that sextupole's own kick. ``orbit0``
    defaults to :func:`closed_orbit_nonlinear`, in which case the last point equals
    the first.

    ``zeta`` starts at zero and ``delta`` at the given value; both are carried along
    but not solved for (see :func:`closed_orbit_nonlinear`), and only the transverse
    part is returned.
    """
    o = (
        closed_orbit_nonlinear(lattice, delta=delta)
        if orbit0 is None
        else np.asarray(orbit0, dtype=float)
    )
    if o.shape != (4,):
        raise ValueError(f"orbit0 must be a length-4 (x, px, y, py) vector, got {o.shape}")
    state = _embed(o, delta)
    points = [state[_TRANSVERSE].copy()]
    for elem in lattice.elements:
        state = elem.track(state, lattice.ref)
        points.append(state[_TRANSVERSE].copy())
    return points


def linearised_element_maps(
    lattice: Lattice, orbit0: np.ndarray | None = None, *, delta: float = 0.0, step: float = 1e-7
) -> list[np.ndarray]:
    r"""Each element's ``6x6`` map **linearised about the orbit at its entrance**.

    The optics a particle near the closed orbit actually sees. For every linear
    element this returns :meth:`~accsim.elements.element.Element.matrix` to
    round-off; for a sextupole at orbit offset ``(x_co, y_co)`` it returns the
    matrix of a drift *plus* the feed-down gradients ``k1l_eff = k2l x_co``
    (normal) and ``k1sl_eff = k2l y_co`` (skew) — which is why steering a machine
    with sextupoles moves beta, the tunes and the coupling.

    This is the primitive the feed-down optics are read from, rather than a single
    one-turn Jacobian, because Twiss propagation needs a matrix *per element*:
    :func:`~accsim.twiss.propagate_twiss` calls each element's on-axis
    ``matrix()`` and would miss the feed-down entirely. Their product in beam
    order is the one-turn Jacobian by the chain rule, exactly — see
    :func:`linearised_one_turn_map`.

    The constant (dipole) part of the feed-down does **not** appear: a Jacobian is
    the linear part only. That term has already done its work in placing the orbit
    these maps are taken about.

    ``step`` is the central-difference increment. The default is looser than
    :func:`closed_orbit_nonlinear`'s because here the answer *is* the Jacobian
    rather than a Newton direction, so its round-off (``~eps/step``) is the
    accuracy of the result. A sextupole kick has an exactly constant second
    derivative, so the ``O(step^2)`` truncation error vanishes identically and
    only round-off remains. Measured on a steered sextupole-free FODO ring
    (2026-08-10): ``1.9e-13`` per element map, ``2.4e-12`` on their product.

    ``delta`` linearises about the orbit at that momentum instead of the reference
    one. It has to be given explicitly even when ``orbit0`` is supplied, because a
    4D transverse vector does not carry the momentum it belongs to.
    """
    o = (
        closed_orbit_nonlinear(lattice, delta=delta)
        if orbit0 is None
        else np.asarray(orbit0, dtype=float)
    )
    if o.shape != (4,):
        raise ValueError(f"orbit0 must be a length-4 (x, px, y, py) vector, got {o.shape}")
    ref = lattice.ref
    state = _embed(o, delta)
    maps = []
    for elem in lattice.elements:
        maps.append(jacobian(lambda s, e=elem: e.track(s, ref), state, step=step))
        state = elem.track(state, ref)
    return maps


def linearised_one_turn_map(
    lattice: Lattice, orbit0: np.ndarray | None = None, *, delta: float = 0.0, step: float = 1e-7
) -> np.ndarray:
    """One-turn ``6x6`` map linearised about the (nonlinear) closed orbit.

    The product of :func:`linearised_element_maps` in beam order — ``M_n ... M_1``,
    last element leftmost, the same ordering
    :meth:`accsim.Lattice.one_turn_map` uses. Feed it to
    :func:`~accsim.twiss.match_periodic` for the beta functions an off-axis
    sextupole produces, or to :func:`~accsim.twiss.match_periodic_coupled` for the
    x-y coupling a *vertically* off-axis one produces.
    :func:`~accsim.twiss.closed_twiss_on_orbit` is the packaged form of the first.
    """
    M = np.eye(DIM)
    for m in linearised_element_maps(lattice, orbit0, delta=delta, step=step):
        M = m @ M
    return M


def linearised_lattice(
    lattice: Lattice, orbit0: np.ndarray | None = None, *, delta: float = 0.0
) -> Lattice:
    r"""The equivalent **linear** machine the beam on the real orbit actually sees.

    Every element is passed through unchanged except a thin sextupole, which is
    joined by I2's derived feed-down split evaluated at its own orbit offset:

        ThinQuadrupole(k1l_eff  = +k2l x_co)
        ThinSkewQuadrupole(k1sl_eff = +k2l y_co)
        ThinSextupole(k2l)              — kept, unchanged

    The sextupole is **kept** because the split above is the *static* feed-down at
    the orbit offset, while the sextupole still feeds down a further
    ``delta``-dependent gradient ``k2 D_x delta`` at dispersion — different terms,
    both physical. The dipole part of the split does not appear: it is what placed
    the orbit these gradients are read at, and it is invisible to every
    matrix-based optics function anyway (a
    :class:`~accsim.elements.corrector.Corrector`'s ``matrix()`` is the identity).

    This is the same machine :func:`linearised_element_maps` describes, reached
    from I2's *derived* coefficients instead of by differentiating ``track()``; the
    analytic suite gates the two against each other. It exists because the
    chromaticity integrals (:func:`~accsim.twiss.natural_chromaticity` and the
    sextupole feed-down term) walk element *types*, not maps, so they need a
    lattice rather than a list of matrices.

    Raises :class:`NotImplementedError` for a **thick** sextupole of non-zero
    strength: its offset varies across the body, so collapsing it onto a single
    gradient at the entrance orbit would carry an ``O(L^2)`` error — exactly the
    error I2 avoided by using thin sextupoles throughout its own gates.
    :func:`~accsim.twiss.propagate_twiss_on_orbit` has no such restriction, because
    it differentiates the thick element's real ``track()``.

    A **thin octupole** is joined by J3's split, the same expansion carried one
    order further — its cubic kick reaches two orders below itself, so it produces a
    sextupole pair as well as a gradient pair:

        ThinQuadrupole(k1l_eff  = +1/2 k3l (x_co^2 - y_co^2))
        ThinSkewQuadrupole(k1sl_eff = +k3l x_co y_co)
        ThinSextupole(k2l_eff  = +k3l x_co)
        ThinSkewSextupole(k2sl_eff = +k3l y_co)
        ThinOctupole(k3l)               — kept, unchanged

    The sextupole pair is what makes an octupole a **first-order** chromatic element
    on a distorted orbit: ``k2l_eff`` at dispersion feeds down the usual
    ``beta k2l D_x / (4 pi)``, where an on-axis octupole contributes to ``Q'``
    exactly nothing (its own ``delta`` term is a sextupole, not a gradient — J2).
    The gradient pair moves beta, the tunes and the coupling as for the sextupole,
    but one order later: it is quadratic in the orbit, not linear. Neither the
    octupole nor the skew sextupole is read by any chromaticity integral, so keeping
    them costs nothing and dropping them would be a claim rather than an omission.

    Raises :class:`NotImplementedError` for a **thick** octupole of non-zero
    strength, for exactly the thick sextupole's reason: its offset varies across the
    body, so a single entrance-orbit split would carry an ``O(L^2)`` error.
    :func:`linearised_element_maps` handles both, because it differentiates
    ``track()`` rather than walking element types.
    """
    from .elements.octupole import Octupole, ThinOctupole
    from .elements.quadrupole import ThinQuadrupole
    from .elements.sextupole import Sextupole, ThinSextupole, ThinSkewSextupole
    from .elements.skew_quadrupole import ThinSkewQuadrupole

    orbit = propagate_orbit_nonlinear(lattice, orbit0, delta=delta)
    elements: list = []
    for i, elem in enumerate(lattice.elements):
        if isinstance(elem, ThinSextupole):
            x_co, y_co = float(orbit[i][0]), float(orbit[i][2])
            tag = elem.name
            elements.append(ThinQuadrupole(elem.k2l * x_co, name=tag and f"{tag}_fd_quad"))
            elements.append(ThinSkewQuadrupole(elem.k2l * y_co, name=tag and f"{tag}_fd_skew"))
            elements.append(elem)
        elif isinstance(elem, Sextupole) and elem.k2 != 0.0 and elem.length > 0.0:
            raise NotImplementedError(
                f"cannot linearise the thick Sextupole {elem.name!r} about an orbit: its "
                "offset varies across the body, so a single entrance-orbit gradient would "
                "carry an O(L^2) error. Slice it into ThinSextupole kicks, or use "
                "propagate_twiss_on_orbit(), which differentiates track() directly"
            )
        elif isinstance(elem, ThinOctupole):
            x_co, y_co = float(orbit[i][0]), float(orbit[i][2])
            tag = elem.name
            elements.append(
                ThinQuadrupole(
                    0.5 * elem.k3l * (x_co * x_co - y_co * y_co), name=tag and f"{tag}_fd_quad"
                )
            )
            elements.append(
                ThinSkewQuadrupole(elem.k3l * x_co * y_co, name=tag and f"{tag}_fd_skew")
            )
            elements.append(ThinSextupole(elem.k3l * x_co, name=tag and f"{tag}_fd_sext"))
            elements.append(ThinSkewSextupole(elem.k3l * y_co, name=tag and f"{tag}_fd_skewsext"))
            elements.append(elem)
        elif isinstance(elem, Octupole) and elem.k3 != 0.0 and elem.length > 0.0:
            raise NotImplementedError(
                f"cannot linearise the thick Octupole {elem.name!r} about an orbit: its "
                "offset varies across the body, so a single entrance-orbit split would "
                "carry an O(L^2) error. Slice it into ThinOctupole kicks, or use "
                "propagate_twiss_on_orbit(), which differentiates track() directly"
            )
        else:
            elements.append(elem)
    return Lattice(elements, lattice.ref)


# ---------------------------------------------------------------------------
# Correction: response matrix and the SVD steering solve
# ---------------------------------------------------------------------------

_PLANES = {"x": (X, "kick_x"), "y": (Y, "kick_y")}


class OrbitCorrectionError(ValueError):
    """An orbit-correction problem is ill-posed (bad plane, monitor or corrector)."""


@dataclass(frozen=True)
class OrbitCorrection:
    """Outcome of :func:`correct_orbit`. The kicks are already applied.

    - ``kicks`` / ``initial_kicks`` — corrector angles [rad] after / before.
    - ``rms_before`` / ``rms_after`` — RMS orbit at the monitors [m]. **Both are
      measured**, by solving for the closed orbit of the actual lattice; neither
      is the linear prediction ``x0 + R dtheta``. That distinction is the whole
      point when the response matrix is one the machine handed you rather than
      one the model computed — see :func:`correct_orbit`.
    - ``singular_values`` — the full spectrum of the response matrix, largest
      first, so a caller can see the conditioning that was truncated.
    - ``n_used`` — how many singular values the solve kept.
    """

    kicks: np.ndarray
    initial_kicks: np.ndarray
    rms_before: float
    rms_after: float
    singular_values: np.ndarray
    n_used: int

    @property
    def n_correctors(self) -> int:
        return int(self.kicks.size)


def _check_plane(plane: str) -> tuple[int, str]:
    try:
        return _PLANES[plane]
    except KeyError:
        raise OrbitCorrectionError(
            f"plane must be 'x' or 'y', got {plane!r}; horizontal and vertical "
            "correction are separate problems in an uncoupled lattice"
        ) from None


def _check_correctors(lattice: Lattice, correctors: Sequence[Corrector]) -> list[Corrector]:
    corr = list(correctors)
    if not corr:
        raise OrbitCorrectionError("no correctors given; there is nothing to steer with")
    in_lattice = {id(e) for e in lattice.elements}
    seen: set[int] = set()
    for c in corr:
        if not isinstance(c, Corrector):
            raise OrbitCorrectionError(
                f"{type(c).__name__} cannot steer the orbit; correctors must be "
                "Corrector elements (a quadrupole moves the optics, not the orbit)"
            )
        if id(c) not in in_lattice:
            raise OrbitCorrectionError(
                f"{c!r} is not in this lattice, so it has no response; pass the "
                "element objects the lattice was built from"
            )
        if id(c) in seen:
            raise OrbitCorrectionError(
                f"{c!r} is listed twice, so its column would be ambiguous; list it "
                "once (repeated *placements* in the lattice are fine and are summed)"
            )
        seen.add(id(c))
    return corr


def _check_monitors(lattice: Lattice, monitors: Sequence[int] | None) -> list[int]:
    if monitors is None:
        return list(range(len(lattice) + 1))
    mon = [int(m) for m in monitors]
    if not mon:
        raise OrbitCorrectionError("no monitors given; there is nothing to measure")
    n = len(lattice)
    for m in mon:
        if not 0 <= m <= n:
            raise OrbitCorrectionError(
                f"monitor index {m} is outside [0, {n}]; monitors are element "
                "*boundaries*, the same points propagate_orbit returns"
            )
    return mon


def _orbit_at(
    lattice: Lattice, monitors: Sequence[int], coord: int, nonlinear: bool = False
) -> np.ndarray:
    """One plane's closed orbit sampled at the monitor boundaries."""
    table = propagate_orbit_nonlinear(lattice) if nonlinear else propagate_orbit(lattice)
    return np.array([table[m][coord] for m in monitors])


def orbit_response_matrix(
    lattice: Lattice,
    correctors: Sequence[Corrector],
    monitors: Sequence[int] | None = None,
    plane: str = "x",
) -> np.ndarray:
    r"""Orbit response matrix ``R[i, j] = d(orbit at monitor i) / d(kick of corrector j)``.

    Shape ``(n_monitors, n_correctors)``, units m/rad. ``monitors`` are element
    **boundary** indices (``None`` = every boundary); ``plane`` selects ``'x'``
    (driving ``kick_x``) or ``'y'`` (driving ``kick_y``).

    **This is the model response, and within the linear model it is exact rather
    than a finite difference.** The *linear* closed orbit is strictly affine in the
    corrector kicks — the fixed point ``(I - M4)^-1 k4`` is linear in ``k4``, and
    ``k4`` is linear in the kicks — so

        orbit(theta) = orbit(0) + R theta

    holds for any kick, however large, with no truncation error. Column ``j`` is
    therefore taken by setting corrector ``j`` to one radian, all other *listed*
    correctors to zero, and subtracting the baseline.

    **A live sextupole breaks that affineness at second order.** Once the orbit is
    off-axis at a sextupole, feed-down adds a dipole ``-1/2 k2l x_co^2`` and a
    gradient ``k2l x_co`` — the first is a corrector the steering itself creates,
    the second changes ``M4``, so neither the inhomogeneity nor the matrix stays
    fixed and the true orbit is only ``orbit(0) + R theta + O(k2l theta^2)``. This
    function keeps returning the affine model response, which is the right thing:
    it is the response an operational machine's *model* predicts, the correction it
    drives converges quadratically, and the correction loop is closed by
    **measuring** the resulting orbit (``nonlinear=True`` in :func:`correct_orbit`)
    rather than trusting the prediction. Feed-down is the reason real orbit
    correction iterates.

    Kicks not owned by a listed corrector (a steering *error* you are correcting
    against) stay put and cancel in the baseline subtraction, so the matrix is the
    machine's response and not a property of its current orbit — to round-off,
    since the cancellation is a floating-point ``(col + base) - base``, not an
    algebraic one.

    Like :func:`~accsim.matching.insertion_response_matrix` this **mutates the
    lattice while it works** and restores every corrector from a snapshot in a
    ``finally``, so an exception cannot leave the machine mis-set.
    """
    coord, attr = _check_plane(plane)
    corr = _check_correctors(lattice, correctors)
    mon = _check_monitors(lattice, monitors)

    snapshot = [getattr(c, attr) for c in corr]
    try:
        for c in corr:
            setattr(c, attr, 0.0)
        base = _orbit_at(lattice, mon, coord)
        columns = []
        for c in corr:
            setattr(c, attr, 1.0)
            columns.append(_orbit_at(lattice, mon, coord) - base)
            setattr(c, attr, 0.0)
    finally:
        for c, v in zip(corr, snapshot, strict=True):
            setattr(c, attr, v)
    return np.column_stack(columns)


def correct_orbit(
    lattice: Lattice,
    correctors: Sequence[Corrector],
    monitors: Sequence[int] | None = None,
    plane: str = "x",
    *,
    n_singular: int | None = None,
    response: np.ndarray | None = None,
    nonlinear: bool = False,
) -> OrbitCorrection:
    r"""Steer the closed orbit to zero at the monitors, and apply the kicks.

    Solves ``R dtheta = -x0`` for the corrector changes, where ``x0`` is the
    present orbit at the monitors. In a linear lattice ``R`` is exact (see
    :func:`orbit_response_matrix`) and this is **one linear solve**, not an
    iteration: with enough independent correctors the orbit lands on zero to
    machine precision in a single application.

    ``nonlinear=True`` measures ``x0`` and :attr:`~OrbitCorrection.rms_after` from
    the **nonlinear** closed orbit (:func:`closed_orbit_nonlinear`) instead, which
    is what a machine with live sextupoles actually has. One application then no
    longer lands on zero: sextupole feed-down leaves a residual ``O(k2l x_co^2)``,
    because the model response ``R`` knows nothing about the dipole kick the
    steering itself created at each off-axis sextupole. Applying it again corrects
    that residual, and again. **This is why real orbit correction is a loop rather
    than a solve**, and it is the operational content of feed-down.

    That loop converges **linearly, not quadratically**, and the distinction is
    physics rather than pedantry. ``R`` is recomputed from the *linear* model every
    pass, so it never learns the feed-down gradient ``k2l x_co``; the iteration is
    a stale-Jacobian fixed-point map, whose contraction factor is the relative size
    of that gradient error and stays *constant* pass after pass (measured at
    ``4.95e-4`` in the analytic suite, three passes running). It is fast — four
    passes take a 0.3 mm orbit to machine precision — but a true Newton, which
    would relinearise about the current orbit each pass, is what would be
    quadratic. The flag defaults to ``False`` so the linear contract is unchanged
    for lattices where it holds.

    ``n_singular`` keeps only the largest ``n`` singular values of ``R``, the
    standard defence against an ill-conditioned corrector set. Two correctors
    close together in betatron phase have nearly parallel response columns; the
    untruncated least-squares answer then buys a small orbit improvement with a
    huge, nearly cancelling pair of kicks that no real magnet could deliver.
    Truncation drops that direction and returns a sane, if slightly less perfect,
    correction. Default ``None`` keeps every singular value above the usual
    round-off cutoff.

    ``response`` lets the caller supply a response matrix instead of computing
    one — an operational machine *measures* its response matrix, and that
    measurement disagrees with the model. The consequence is deliberate and is
    why :attr:`OrbitCorrection.rms_after` is obtained by **re-solving the closed
    orbit of the corrected lattice**, never by evaluating ``x0 + R dtheta``: with
    a wrong ``R`` the prediction is perfect and the machine is not.

    N vs M is handled the way :func:`~accsim.matching.match_insertion` handles it:
    more correctors than monitors gives the minimum-norm solution, fewer gives the
    least-squares one. Neither is reported as exact — the returned RMS says what
    was actually achieved.

    Raises :class:`ClosedOrbitError` (and rolls the kicks back) if the corrected
    lattice has no closed orbit.
    """
    coord, attr = _check_plane(plane)
    corr = _check_correctors(lattice, correctors)
    mon = _check_monitors(lattice, monitors)

    R = (
        orbit_response_matrix(lattice, corr, mon, plane)
        if response is None
        else np.asarray(response, dtype=float)
    )
    if R.shape != (len(mon), len(corr)):
        raise OrbitCorrectionError(
            f"response has shape {R.shape}, expected ({len(mon)}, {len(corr)}) for "
            f"{len(mon)} monitors and {len(corr)} correctors"
        )
    if n_singular is not None and not 1 <= n_singular <= min(R.shape):
        raise OrbitCorrectionError(
            f"n_singular must be in [1, {min(R.shape)}] for a {R.shape} response "
            f"matrix, got {n_singular}"
        )

    x0 = _orbit_at(lattice, mon, coord, nonlinear)
    rms_before = float(np.sqrt(np.mean(x0**2)))

    u, s, vt = np.linalg.svd(R, full_matrices=False)
    keep = len(s) if n_singular is None else n_singular
    cutoff = s[0] * max(R.shape) * np.finfo(float).eps if s.size else 0.0
    keep = min(keep, int(np.count_nonzero(s > cutoff)))
    if keep == 0:
        raise OrbitCorrectionError(
            "the response matrix is numerically zero: none of these correctors "
            "moves the orbit at any of these monitors"
        )
    dtheta = -(vt[:keep].T @ ((u[:, :keep].T @ x0) / s[:keep]))

    initial = np.array([getattr(c, attr) for c in corr])
    kicks = initial + dtheta
    for c, v in zip(corr, kicks, strict=True):
        setattr(c, attr, float(v))
    try:
        # Measured, not predicted: the orbit of the machine as it now stands.
        x1 = _orbit_at(lattice, mon, coord, nonlinear)
    except ClosedOrbitError:
        for c, v in zip(corr, initial, strict=True):
            setattr(c, attr, float(v))
        raise
    return OrbitCorrection(
        kicks=kicks,
        initial_kicks=initial,
        rms_before=rms_before,
        rms_after=float(np.sqrt(np.mean(x1**2))),
        singular_values=s,
        n_used=keep,
    )
