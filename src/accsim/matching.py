r"""Matching: solve for element strengths that hit target optics (milestones H1, H2).

H1 matches two **global** scalars — the tunes, then the chromaticities — with two
knobs each. H2 (:func:`match_insertion`, at the bottom of this file) matches
**local** optics at a chosen point: ``beta*``, a waist (``alpha* = 0``), or the
dispersion, with N knobs against M targets. Same skeleton in all three — knobs,
approximate Jacobian, exact residual, backtracking, rollback — but the H1 pair
below have closed-form response matrices and H2 does not; see its own docstring
for why.

Two problems, and they are **not** the same shape:

- Two quadrupole families -> a target ``(Q_x, Q_y)``: an *iteration*, because
  moving a quadrupole moves the ``beta`` the response is computed from.
- Two sextupole families -> a target ``(Q'_x, Q'_y)``: an **exact linear solve**,
  because a sextupole's linear map is a drift, so it changes neither ``beta``,
  nor the dispersion, nor the tunes. The chromaticity is strictly affine in the
  sextupole strengths, and one shot lands on the target from anywhere.

Matching the tunes first and the chromaticity second is therefore not a
convention but a consequence: the second step cannot disturb the first.

The tune half rests on one formula, the **tune response matrix**: perturbing a
quadrupole gradient by ``dk1`` shifts the tunes by the beta-weighted integral

    dQ_x = +(1/4pi) integral(beta_x dk1 ds),
    dQ_y = -(1/4pi) integral(beta_y dk1 ds),

i.e. more focusing in x raises ``Q_x`` and lowers ``Q_y``. This is the same
first-order perturbation integral that gives the natural chromaticity in
:func:`accsim.twiss.natural_chromaticity` (there the gradient perturbation is the
off-momentum weakening ``dk1 = -k1 delta``, which is why that expression carries
the opposite sign and an extra factor ``k1``).

**Approximate Jacobian, exact residual.** The response matrix is first-order:
``beta`` itself changes as the strengths move, so a single Newton step does not
land on the target. But the residual is evaluated with the exact
:func:`accsim.twiss.tunes`, so the *fixed point* is exact — the iteration
converges to strengths that hit the target to machine precision, not to
first-order strengths. The Jacobian is recomputed each iteration (with the
current ``beta``), which is what makes this Newton rather than a chord method on
a stale matrix.

**Knobs are MAD-X expression semantics.** A :class:`Knob` sets each family
member's strength to ``w_i * v`` for one shared variable ``v``. That is the only
form that handles both of the cases a matcher meets: a family split into
half-quads at the ends of a cell (weights ``0.5``, which a purely additive knob
would desynchronise) and a family starting from zero strength (which a purely
multiplicative knob could never move).

**Mutation and rollback.** :class:`accsim.Lattice` shares its element objects, so
matching necessarily mutates them in place — copying the ``Lattice`` would not
protect the strengths. Every entry point snapshots the raw per-element strengths
and restores them if it raises, so a failed match leaves the lattice exactly as
it found it. On success the lattice carries the matched strengths.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .elements.element import Element
from .elements.quadrupole import Quadrupole, ThinQuadrupole, _focusing_block
from .elements.sextupole import Sextupole, ThinSextupole
from .lattice import Lattice
from .twiss import (
    Twiss,
    UnstableLatticeError,
    _blocks,
    _dispersive_kick,
    _propagate_block,
    _transverse_4d,
    chromaticity,
    closed_twiss,
    propagate_twiss,
    tunes,
)

__all__ = [
    "InsertionMatchResult",
    "Knob",
    "MatchResult",
    "MatchingError",
    "Target",
    "chromaticity_response_matrix",
    "insertion_response_matrix",
    "match_chromaticity",
    "match_insertion",
    "match_tunes",
    "tune_response_matrix",
]

_INV_4PI = 1.0 / (4.0 * math.pi)

# Which attribute a knob drives, per element type. ``k1`` is [m^-2] and ``k1l``
# is [m^-1]: different units, so one knob may never span both (see ``Knob``).
_STRENGTH_ATTRS: tuple[tuple[type[Element], str], ...] = (
    (ThinQuadrupole, "k1l"),
    (Quadrupole, "k1"),
    (ThinSextupole, "k2l"),
    (Sextupole, "k2"),
)


class MatchingError(ValueError):
    """A matching problem is ill-posed, degenerate, or failed to converge."""


def _strength_attr(elem: Element) -> str:
    """Name of the strength attribute a knob drives on ``elem``."""
    for cls, attr in _STRENGTH_ATTRS:
        if isinstance(elem, cls):
            return attr
    raise MatchingError(
        f"{type(elem).__name__} has no matchable strength; knobs accept "
        "Quadrupole (k1), ThinQuadrupole (k1l), Sextupole (k2) and ThinSextupole (k2l)"
    )


class Knob:
    r"""One matching variable ``v``, shared by a family of elements.

    Setting the knob to ``v`` sets member ``i``'s strength to ``w_i * v`` — the
    semantics of a MAD-X expression (``kqf``, ``0.5 * kqf``, ``-kqf``). Weights
    default to ``1.0``.

    The current :attr:`value` is read back from the members, so the family must
    already *be* ganged: every ``strength_i / w_i`` must agree, or the knob has no
    well-defined value and the constructor refuses. Mixing element types whose
    strengths carry different units (thick ``k1`` [m^-2] with thin ``k1l``
    [m^-1]) is refused for the same reason.

    Aliasing is allowed in the lattice but not in the knob: if the *same* element
    object appears at two places in a :class:`Lattice`, both occurrences respond
    to the knob and :func:`tune_response_matrix` sums both. Listing that object
    twice in one knob's ``elements`` would make its weight ambiguous, so it is
    refused.
    """

    def __init__(
        self,
        elements: Iterable[Element],
        weights: Sequence[float] | None = None,
        name: str | None = None,
    ) -> None:
        elems = list(elements)
        if not elems:
            raise MatchingError("a knob needs at least one element")
        if weights is None:
            weights = [1.0] * len(elems)
        w = [float(x) for x in weights]
        if len(w) != len(elems):
            raise MatchingError(f"got {len(elems)} elements but {len(w)} weights")
        if any(x == 0.0 for x in w):
            raise MatchingError(
                "a member with zero weight can never move; drop it from the knob "
                "instead of giving it zero weight"
            )
        seen: set[int] = set()
        for e in elems:
            if id(e) in seen:
                raise MatchingError(
                    f"element {e!r} is listed twice in one knob, so its weight is "
                    "ambiguous; list it once (repeated *placements* in the lattice "
                    "are fine and are summed)"
                )
            seen.add(id(e))

        attrs = {_strength_attr(e) for e in elems}
        if len(attrs) != 1:
            raise MatchingError(
                f"all members of a knob must drive the same strength attribute, got "
                f"{sorted(attrs)} — k1 [m^-2] and k1l [m^-1] are different units, so "
                "thick and thin elements cannot share one knob"
            )

        self.elements = elems
        self.weights = w
        self.name = name
        self.attr = attrs.pop()

        self.check_ganged()

    def check_ganged(self) -> None:
        """Verify the members still share one value, i.e. that :attr:`value` means something.

        Called at construction *and* again by every matcher before it reads
        :attr:`value`: nothing stops a caller from setting one member's strength
        directly in between, after which the family is no longer ganged and
        :attr:`value` — read from the first member — would silently misreport it.
        """
        values = [
            getattr(e, self.attr) / wi for e, wi in zip(self.elements, self.weights, strict=True)
        ]
        v0 = values[0]
        if not all(math.isclose(v, v0, rel_tol=1e-9, abs_tol=1e-12) for v in values):
            raise MatchingError(
                f"knob members are not consistent with one shared value: "
                f"{self.attr}/weight = {values} — either pass weights matching the "
                "current strengths, or set them so the family is ganged first"
            )

    @property
    def value(self) -> float:
        """Current knob value ``v``, read back from the first member.

        Meaningful only while the family is ganged; see :meth:`check_ganged`.
        """
        return getattr(self.elements[0], self.attr) / self.weights[0]

    def apply(self, v: float) -> None:
        """Set every member's strength to ``w_i * v``."""
        for e, wi in zip(self.elements, self.weights, strict=True):
            setattr(e, self.attr, wi * float(v))

    def snapshot(self) -> list[float]:
        """Raw per-member strengths, for rollback."""
        return [getattr(e, self.attr) for e in self.elements]

    def restore(self, snap: Sequence[float]) -> None:
        """Undo :meth:`apply` from a :meth:`snapshot`."""
        for e, s in zip(self.elements, snap, strict=True):
            setattr(e, self.attr, s)

    def __repr__(self) -> str:
        name = f"{self.name!r}" if self.name is not None else "<unnamed>"
        return f"Knob({name}, {len(self.elements)} x {self.attr}, value={self.value:g})"


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a successful match.

    ``values`` are the converged knob values (already applied to the lattice),
    ``initial`` the values on entry, ``achieved``/``targets`` the optics
    quantities, ``residual`` the 2-norm ``|achieved - targets|``, and
    ``iterations`` the number of Newton steps taken (``0`` if the lattice was
    already on target).
    """

    values: tuple[float, ...]
    initial: tuple[float, ...]
    achieved: tuple[float, float]
    targets: tuple[float, float]
    residual: float
    iterations: int


def _knob_index(knobs: Sequence[Knob]) -> dict[int, tuple[int, float]]:
    """Map ``id(element) -> (knob index, weight)``, refusing overlap between knobs."""
    index: dict[int, tuple[int, float]] = {}
    for j, knob in enumerate(knobs):
        for e, w in zip(knob.elements, knob.weights, strict=True):
            if id(e) in index:
                raise MatchingError(
                    f"element {e!r} belongs to two knobs; the knobs would not be "
                    "independent variables"
                )
            index[id(e)] = (j, w)
    return index


def tune_response_matrix(lattice: Lattice, knobs: Sequence[Knob], slices: int = 64) -> np.ndarray:
    r"""First-order tune response ``J[u, j] = dQ_u / dv_j`` of ``lattice`` to ``knobs``.

    Rows are ``(Q_x, Q_y)``, columns are the knobs, so a Newton step solves
    ``J dv = -(Q - Q_target)``. Each entry is the beta-weighted perturbation
    integral, summed over the family with its weights:

        dQ_x/dv_j = +(1/4pi) sum_i w_i integral(beta_x ds over member i),
        dQ_y/dv_j = -(1/4pi) sum_i w_i integral(beta_y ds over member i).

    A thin quadrupole is an exact single-point contribution ``w_i beta_u`` (beta
    is continuous across a thin kick, so the entrance value is the value "at" the
    element). A thick quadrupole's body is integrated by ``slices``-fold
    trapezoidal sub-stepping of ``beta`` through the element's *current* map — so
    unlike :func:`accsim.twiss.natural_chromaticity`, a thick quad at ``k1 = 0``
    is still integrated (it has zero chromaticity but a perfectly good
    ``dQ/dk1``).

    The integer part of the tune is irrelevant to a derivative, so no phase
    unwrapping is needed here.

    Raises :class:`accsim.UnstableLatticeError` if the lattice has no matched
    optics to perturb.
    """
    for knob in knobs:
        if knob.attr not in ("k1", "k1l"):
            raise MatchingError(
                f"{knob!r} drives {knob.attr}, which cannot change the tunes: a "
                "sextupole's *linear* map is a drift, so it leaves beta, dispersion "
                "and the tunes untouched. Tune knobs must be quadrupoles."
            )
    index = _knob_index(knobs)
    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    integrals = np.zeros((len(knobs), 2))

    for elem in lattice.elements:
        hit = index.get(id(elem))
        if hit is not None and isinstance(elem, Quadrupole) and elem.length > 0.0:
            # Thick body: trapezoidal integral of beta through the element.
            j, w = hit
            ds = elem.length / slices
            xb = _focusing_block(elem.k1, ds)  # x'' + k1 x = 0
            yb = _focusing_block(-elem.k1, ds)  # y'' - k1 y = 0
            int_bx, int_by = 0.5 * bx, 0.5 * by  # half-weight the entrance sample
            for i in range(slices):
                bx, ax, _ = _propagate_block(xb, bx, ax)
                by, ay, _ = _propagate_block(yb, by, ay)
                wt = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                int_bx += wt * bx
                int_by += wt * by
            integrals[j, 0] += w * int_bx * ds
            integrals[j, 1] += w * int_by * ds
            continue  # beta has already been advanced across the whole element
        if hit is not None:
            # Thin kick: beta is continuous, so the entrance value is "at" it.
            j, w = hit
            integrals[j, 0] += w * bx
            integrals[j, 1] += w * by
        cx, cy = _blocks(elem.matrix(ref))
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)

    jac = np.empty((2, len(knobs)))
    jac[0, :] = +_INV_4PI * integrals[:, 0]
    jac[1, :] = -_INV_4PI * integrals[:, 1]
    return jac


def _check_conditioning(jac: np.ndarray, what: str, hint: str) -> None:
    """Refuse a response matrix that cannot be inverted meaningfully.

    Two knobs at equivalent optics (or one with no response at all) give
    proportional or zero columns; a bare solve would return a huge, meaningless
    step instead of reporting that the problem is under-determined.
    """
    if not np.all(np.isfinite(jac)):
        raise MatchingError(f"{what} is not finite:\n{jac}")
    cond = float(np.linalg.cond(jac))
    if not math.isfinite(cond) or cond > 1e10:
        raise MatchingError(
            f"{what} is singular or ill-conditioned (cond = {cond:.3g}): the two knobs "
            f"do not act independently — {hint}"
        )


def match_tunes(
    lattice: Lattice,
    targets: tuple[float, float],
    knobs: Sequence[Knob],
    *,
    tol: float = 1e-12,
    max_iter: int = 60,
    slices: int = 64,
) -> MatchResult:
    r"""Match ``(Q_x, Q_y)`` to ``targets`` with exactly two quadrupole knobs.

    Newton iteration on the beta-weighted :func:`tune_response_matrix`, with the
    residual taken from the exact :func:`accsim.twiss.tunes`, so the converged
    strengths hit the target to ``tol`` (default ``1e-12``, i.e. machine
    precision in tune units) rather than to first order.

    ``targets`` are **full** tunes, integer part included — :func:`accsim.tunes`
    accumulates phase advance rather than reading the one-turn trace, so a target
    of ``6.28`` is meaningful and distinct from ``0.28``.

    Each step is backtracked (halved) until it both stays stable and reduces the
    residual: a first-order step from far away routinely overshoots the stability
    boundary, where :func:`tunes` raises rather than returning a wrong number.

    **Mutates ``lattice`` in place** on success. If it raises, the original
    strengths are restored.

    Raises :class:`MatchingError` if the knobs are degenerate, if no step
    reduces the residual (usually an unreachable target), or if ``max_iter`` is
    exhausted.
    """
    knobs = tuple(knobs)
    if len(knobs) != 2:
        raise MatchingError(
            f"match_tunes needs exactly two knobs for two targets (Q_x, Q_y), got {len(knobs)}"
        )
    target = np.asarray(targets, dtype=float)
    if target.shape != (2,):
        raise MatchingError(f"targets must be (Q_x, Q_y), got {targets!r}")
    _knob_index(knobs)  # fail fast on overlapping knobs, before touching anything

    for knob in knobs:
        knob.check_ganged()  # ``value`` is only meaningful while the family is ganged
    snapshots = [k.snapshot() for k in knobs]
    initial = tuple(k.value for k in knobs)

    def rollback() -> None:
        for knob, snap in zip(knobs, snapshots, strict=True):
            knob.restore(snap)

    try:
        try:
            resid = np.asarray(tunes(lattice), dtype=float) - target
        except UnstableLatticeError as exc:
            raise MatchingError(f"the starting lattice has no matched optics: {exc}") from exc

        v = np.array(initial, dtype=float)
        for iteration in range(max_iter + 1):
            norm = float(np.linalg.norm(resid))
            if norm <= tol:
                achieved = resid + target
                return MatchResult(
                    values=tuple(float(x) for x in v),
                    initial=initial,
                    achieved=(float(achieved[0]), float(achieved[1])),
                    targets=(float(target[0]), float(target[1])),
                    residual=norm,
                    iterations=iteration,
                )
            if iteration == max_iter:
                break

            jac = tune_response_matrix(lattice, knobs, slices)
            _check_conditioning(
                jac,
                "tune response matrix",
                "check that they are different quadrupole families at different optics",
            )
            step = np.linalg.solve(jac, -resid)

            # Backtrack until the step is both stable and an improvement.
            lam = 1.0
            while True:
                trial = v + lam * step
                for knob, value in zip(knobs, trial, strict=True):
                    knob.apply(float(value))
                try:
                    new_resid = np.asarray(tunes(lattice), dtype=float) - target
                except UnstableLatticeError:
                    new_resid = None  # stepped past the stability boundary
                if new_resid is not None and float(np.linalg.norm(new_resid)) < norm:
                    v, resid = trial, new_resid
                    break
                lam *= 0.5
                if lam < 1e-8:
                    raise MatchingError(
                        f"no step reduced the tune residual (|dQ| = {norm:.3g} at "
                        f"iteration {iteration}); the target {tuple(target)} is "
                        "probably unreachable with these knobs"
                    )
        raise MatchingError(
            f"tune matching did not converge in {max_iter} iterations "
            f"(|dQ| = {float(np.linalg.norm(resid)):.3g}, tol = {tol:g})"
        )
    except Exception:
        rollback()
        raise


def chromaticity_response_matrix(
    lattice: Lattice, knobs: Sequence[Knob], slices: int = 64
) -> np.ndarray:
    r"""Chromaticity response ``S[u, j] = dQ'_u / dv_j`` of ``lattice`` to sextupole ``knobs``.

    Rows are ``(Q'_x, Q'_y)``, columns the knobs. A sextupole at dispersion sees
    ``x = x_beta + D_x delta``, so its quadratic kick carries a ``delta``-dependent
    linear gradient ``k1_eff = k2 D_x delta``, giving

        dQ'_x/dv_j = +(1/4pi) sum_i w_i integral(beta_x D_x ds over member i),
        dQ'_y/dv_j = -(1/4pi) sum_i w_i integral(beta_y D_x ds over member i).

    The opposite signs are the ``x^2 - y^2`` structure of the kick — which is
    exactly what lets sextupoles at ``D_x > 0`` pull both (negative) natural
    chromaticities back toward zero with a well-conditioned 2x2.

    **This matrix is exact, not first-order.** A sextupole's linear map is a drift
    with no ``k2`` dependence, so changing the knobs leaves ``beta``, ``D_x`` and
    the tunes untouched: the chromaticity is a strictly *affine* function of the
    knob values, and this matrix is its constant gradient. It is therefore
    identical at every baseline — a property :func:`match_chromaticity` exploits
    to solve in one shot instead of iterating, and which the gate asserts directly.

    The integrand mirrors :func:`accsim.twiss.chromaticity`'s own quadrature term
    for term (same trapezoidal weights, same drift transport of ``D_x`` through a
    thick body), so the residual after the solve is at machine precision rather
    than at the discretisation error of ``slices``.
    """
    for knob in knobs:
        if knob.attr not in ("k2", "k2l"):
            raise MatchingError(
                f"{knob!r} drives {knob.attr}, so the chromaticity response would not "
                "be linear: moving a quadrupole moves beta, the dispersion and the "
                "tunes as well. Chromaticity knobs must be sextupoles (match the "
                "tunes first with match_tunes, then the chromaticity)."
            )
    index = _knob_index(knobs)
    ref = lattice.ref
    tw0 = closed_twiss(lattice)
    bx, ax = tw0.beta_x, tw0.alpha_x
    by, ay = tw0.beta_y, tw0.alpha_y
    disp = np.array([tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py])
    integrals = np.zeros((len(knobs), 2))

    for elem in lattice.elements:
        M = elem.matrix(ref)
        hit = index.get(id(elem))
        if hit is not None and isinstance(elem, Sextupole) and elem.length > 0.0:
            # Thick body: trapezoidal integral of beta_u * D_x. The linear map is a
            # drift, so beta and D transport as through a drift.
            j, w = hit
            ds = elem.length / slices
            db = _focusing_block(0.0, ds)  # [[1, ds], [0, 1]]
            int_x = 0.5 * bx * float(disp[0])  # trapezoid: half-weight entrance
            int_y = 0.5 * by * float(disp[0])
            for i in range(slices):
                bx, ax, _ = _propagate_block(db, bx, ax)
                by, ay, _ = _propagate_block(db, by, ay)
                disp[0] += disp[1] * ds  # drift transport of the dispersion
                disp[2] += disp[3] * ds
                wt = 0.5 if i == slices - 1 else 1.0  # half-weight the exit sample
                int_x += wt * bx * float(disp[0])
                int_y += wt * by * float(disp[0])
            integrals[j, 0] += w * int_x * ds
            integrals[j, 1] += w * int_y * ds
            continue  # beta / dispersion already advanced across the body
        if hit is not None:
            # Thin kick: beta and D_x are continuous across the zero-length kick.
            j, w = hit
            integrals[j, 0] += w * bx * float(disp[0])
            integrals[j, 1] += w * by * float(disp[0])
        cx, cy = _blocks(M)
        bx, ax, _ = _propagate_block(cx, bx, ax)
        by, ay, _ = _propagate_block(cy, by, ay)
        disp = _transverse_4d(M) @ disp + _dispersive_kick(M)

    resp = np.empty((2, len(knobs)))
    resp[0, :] = +_INV_4PI * integrals[:, 0]
    resp[1, :] = -_INV_4PI * integrals[:, 1]
    return resp


def match_chromaticity(
    lattice: Lattice,
    targets: tuple[float, float],
    knobs: Sequence[Knob],
    *,
    tol: float = 1e-9,
    slices: int = 64,
) -> MatchResult:
    r"""Match ``(Q'_x, Q'_y)`` to ``targets`` with exactly two sextupole knobs.

    **One exact linear solve, not an iteration.** A sextupole's linear map is a
    drift, so it changes neither ``beta``, nor the dispersion, nor the tunes:
    :func:`accsim.twiss.chromaticity` is a strictly *affine* function of the knob
    values, ``Q'(v) = Q'(v0) + S (v - v0)`` with the constant
    :func:`chromaticity_response_matrix` ``S``. The answer is therefore

        v = v0 + S^-1 (targets - Q'(v0)),

    reached in one step from any starting strengths, including zero. Newton
    iteration here would be machinery pretending the problem is harder than it is.

    ``targets`` are the **total** chromaticity (natural + feed-down), which is what
    :func:`accsim.chromaticity` returns and what a real machine measures; the
    natural part is the ``k2``-independent constant in the affine relation. The
    usual target is ``(0, 0)``, or a small positive pair for head-tail stability.

    Because the relation is exact, the residual is checked against ``tol`` after
    the solve and a miss is an error rather than a cue to iterate: it would mean
    the affineness assumption is broken, not that the step was too long.

    **Mutates ``lattice`` in place** on success; restores the original strengths if
    it raises.
    """
    knobs = tuple(knobs)
    if len(knobs) != 2:
        raise MatchingError(
            f"match_chromaticity needs exactly two knobs for two targets "
            f"(Q'_x, Q'_y), got {len(knobs)}"
        )
    target = np.asarray(targets, dtype=float)
    if target.shape != (2,):
        raise MatchingError(f"targets must be (Q'_x, Q'_y), got {targets!r}")
    _knob_index(knobs)  # fail fast on overlapping knobs, before touching anything

    for knob in knobs:
        knob.check_ganged()  # ``value`` is only meaningful while the family is ganged
    snapshots = [k.snapshot() for k in knobs]
    initial = tuple(k.value for k in knobs)

    try:
        try:
            xi0 = np.asarray(chromaticity(lattice, slices), dtype=float)
        except UnstableLatticeError as exc:
            raise MatchingError(f"the starting lattice has no matched optics: {exc}") from exc

        resp = chromaticity_response_matrix(lattice, knobs, slices)
        _check_conditioning(
            resp,
            "chromaticity response matrix",
            "a sextupole at D_x = 0 has no chromaticity response at all, and two "
            "sextupoles at the same optics are one knob wearing two hats",
        )

        v = np.array(initial, dtype=float) + np.linalg.solve(resp, target - xi0)
        for knob, value in zip(knobs, v, strict=True):
            knob.apply(float(value))

        achieved = np.asarray(chromaticity(lattice, slices), dtype=float)
        residual = float(np.linalg.norm(achieved - target))
        if residual > tol:
            raise MatchingError(
                f"the exact linear solve missed by |dQ'| = {residual:.3g} (tol = {tol:g}); "
                "the chromaticity is not affine in these knobs, which it must be for "
                "sextupoles — check that no knob element is also acting as something else"
            )
        return MatchResult(
            values=tuple(float(x) for x in v),
            initial=initial,
            achieved=(float(achieved[0]), float(achieved[1])),
            targets=(float(target[0]), float(target[1])),
            residual=residual,
            iterations=1,
        )
    except Exception:
        for knob, snap in zip(knobs, snapshots, strict=True):
            knob.restore(snap)
        raise


# --------------------------------------------------------------------------
# H2: insertion matching — local optics at a point, N knobs -> M targets
# --------------------------------------------------------------------------

#: Twiss attributes a :class:`Target` may name. Phases are deliberately absent:
#: ``mu`` accumulates from the lattice start, so it is a property of everything
#: upstream rather than a local optics function at the point.
_TARGET_QUANTITIES: tuple[str, ...] = (
    "beta_x",
    "alpha_x",
    "beta_y",
    "alpha_y",
    "disp_x",
    "disp_px",
    "disp_y",
    "disp_py",
)


@dataclass(frozen=True)
class Target:
    """One optics constraint: ``quantity`` at boundary point ``at`` should equal ``value``.

    ``at`` indexes the :func:`accsim.twiss.propagate_twiss` boundary points, so it
    runs ``0 .. len(lattice)``: ``0`` is the lattice entrance, ``k`` the exit of
    element ``k-1``. Elements carry no names in this codebase, so the index is the
    identifier; put a zero-length marker where you want to observe if the natural
    boundary is not one.

    **Weights make the residual dimensionally meaningful.** ``beta`` is metres and
    can be ~100, ``alpha`` is dimensionless and ~1, dispersion is metres and ~1 —
    an unweighted 2-norm over a mixed set is dominated by whichever target happens
    to carry the largest number, and the matcher would quietly satisfy that one
    first. The default ``weight = 1 / max(|value|, 1)`` is relative for large
    targets and absolute for small ones, so it stays finite for the common
    ``alpha* = 0``. Pass ``weight`` explicitly to prioritise a target.
    """

    quantity: str
    at: int
    value: float
    weight: float | None = None

    def __post_init__(self) -> None:
        if self.quantity not in _TARGET_QUANTITIES:
            raise MatchingError(
                f"unknown target quantity {self.quantity!r}; expected one of "
                f"{list(_TARGET_QUANTITIES)}"
            )
        if not isinstance(self.at, int) or isinstance(self.at, bool) or self.at < 0:
            raise MatchingError(
                f"target 'at' must be a non-negative boundary index, got {self.at!r}"
            )
        if self.weight is not None and not (self.weight > 0.0 and math.isfinite(self.weight)):
            raise MatchingError(f"target weight must be finite and positive, got {self.weight!r}")

    @property
    def scale(self) -> float:
        """Effective weight applied to this target's residual."""
        if self.weight is not None:
            return float(self.weight)
        return 1.0 / max(abs(float(self.value)), 1.0)

    def __str__(self) -> str:
        return f"{self.quantity}@{self.at} = {self.value:g}"


@dataclass(frozen=True)
class InsertionMatchResult:
    """Outcome of a successful :func:`match_insertion`.

    Unlike :class:`MatchResult` this carries the ``targets`` themselves, not just
    their values: with M targets and N knobs the interesting question when a match
    is tight is *which* constraint is limiting, and ``residuals`` (per target,
    unweighted, ``achieved - value``) is the only honest way to say so.

    ``residual`` is the **weighted** 2-norm actually driven to ``tol``.
    """

    values: tuple[float, ...]
    initial: tuple[float, ...]
    targets: tuple[Target, ...]
    achieved: tuple[float, ...]
    residuals: tuple[float, ...]
    residual: float
    iterations: int


def _check_targets(lattice: Lattice, targets: Sequence[Target]) -> tuple[Target, ...]:
    """Validate a target list against ``lattice`` and return it as a tuple."""
    out = tuple(targets)
    if not out:
        raise MatchingError("match_insertion needs at least one target")
    n_points = len(lattice) + 1
    for t in out:
        if not isinstance(t, Target):
            raise MatchingError(f"targets must be Target instances, got {t!r}")
        if t.at >= n_points:
            raise MatchingError(
                f"target {t} observes boundary {t.at}, but this lattice has "
                f"{n_points} boundary points (0 .. {len(lattice)})"
            )
    return out


def _check_optics_knobs(knobs: Sequence[Knob]) -> None:
    """Refuse knobs that provably cannot move the linear optics at a point."""
    for knob in knobs:
        if knob.attr in ("k2", "k2l"):
            raise MatchingError(
                f"{knob!r} drives {knob.attr}: a sextupole's *linear* map is a drift, so "
                "it cannot move beta, alpha or the dispersion at any point — no sextupole "
                "setting can satisfy an insertion target. Use quadrupole knobs (and "
                "match_chromaticity for what sextupoles can do)."
            )


def _observe(lattice: Lattice, targets: Sequence[Target], twiss0: Twiss | None) -> np.ndarray:
    """Current values of ``targets``, from the periodic solution or from ``twiss0``.

    With ``twiss0=None`` the closed solution is **re-solved** here, which is what
    makes the periodic branch honest: moving a quadrupole moves the matched optics
    everywhere in the ring, not only downstream of the quadrupole.
    """
    start = closed_twiss(lattice) if twiss0 is None else twiss0
    points = propagate_twiss(lattice, start)
    return np.array([float(getattr(points[t.at], t.quantity)) for t in targets])


def insertion_response_matrix(
    lattice: Lattice,
    targets: Sequence[Target],
    knobs: Sequence[Knob],
    *,
    twiss0: Twiss | None = None,
    fd_step: float = 1e-6,
) -> np.ndarray:
    r"""Response ``J[i, j] = d(target i) / dv_j``, by central finite differences.

    Rows are the targets in order, columns the knobs, **unweighted** (physical
    units: ``dbeta/dv`` is m per knob unit). A Newton step solves ``J dv = -r``.

    **Why finite differences here and a closed form in H1.** The tune response is
    one universal integral, ``dQ/dv = +-(1/4pi) integral(beta dk1 ds)``, valid for
    every lattice; the response of a *local* ``beta`` or dispersion is not — it
    depends on the target quantity, on where the knob sits relative to the
    observation point, and, in the periodic branch, on the re-solved closed
    solution. Differencing the exact :func:`accsim.twiss.propagate_twiss` covers
    all of that uniformly and works identically for a ring and for a transfer
    line. The gate pins this matrix against a symbolic ``dbeta/dv`` differentiated
    from the closed solution of a thin lattice, so "approximate" here means
    *truncation*, not *unvalidated*.

    As in :func:`match_tunes` the Jacobian is only used for the *step*: the
    residual comes from the exact optics, so the converged fixed point is exact.

    The step is ``h_j = fd_step * max(|v_j|, 1)``. The floor matters — a knob may
    legitimately start at ``v = 0`` (an off sextupole family's quadrupole
    equivalent, a corrector at rest), and a purely relative step would give it a
    zero column and be reported as a degenerate knob. If one side of the central
    difference falls outside the stability boundary the column falls back to a
    one-sided difference against the baseline, over ``h`` and **not** ``2h`` —
    the gate pins that denominator bit-exactly, because the surrounding
    convergence tests cannot see it: with an exact residual the fixed point is
    right however wrong the Jacobian is, so a halved column would only have cost
    iterations. Near the boundary ``beta`` diverges and the one-sided truncation
    error runs to tens of percent, which is why the gate compares against the
    exact quotient rather than against a finer central difference.

    **This is the only response matrix in the package that mutates the lattice.**
    :func:`tune_response_matrix` and :func:`chromaticity_response_matrix` are pure
    integrals over the current optics; differencing has to *move* the knobs. Each
    knob is restored from a raw per-element snapshot in a ``finally``, so the
    lattice is byte-identical on return **and** on any exception — which matters
    because this function is public and a standalone caller has no outer rollback
    to fall back on (``UnstableLatticeError`` is handled here, but a
    :class:`~accsim.twiss.CoupledLatticeError` from the periodic branch is not).
    Restoring by re-applying ``v`` would instead round-trip through
    ``w_i * (strength_i / w_0)`` and could land an ULP away for awkward weights.
    """
    _check_optics_knobs(knobs)
    _knob_index(knobs)
    targets = _check_targets(lattice, targets)
    if not (fd_step > 0.0 and math.isfinite(fd_step)):
        raise MatchingError(f"fd_step must be finite and positive, got {fd_step!r}")

    base = _observe(lattice, targets, twiss0)
    jac = np.empty((len(targets), len(knobs)))
    for j, knob in enumerate(knobs):
        v = knob.value
        h = fd_step * max(abs(v), 1.0)
        snapshot = knob.snapshot()
        sides: list[np.ndarray | None] = []
        try:
            for sign in (+1.0, -1.0):
                knob.apply(v + sign * h)
                try:
                    sides.append(_observe(lattice, targets, twiss0))
                except UnstableLatticeError:
                    sides.append(None)
        finally:
            knob.restore(snapshot)
        plus, minus = sides
        if plus is not None and minus is not None:
            jac[:, j] = (plus - minus) / (2.0 * h)
        elif plus is not None:
            jac[:, j] = (plus - base) / h
        elif minus is not None:
            jac[:, j] = (base - minus) / h
        else:
            raise MatchingError(
                f"{knob!r} cannot be differenced: the lattice is unstable on both sides "
                f"of v = {v!r} at step {h:g}, so the stable window around v is narrower "
                "than the step. Either v sits on the stability boundary, or fd_step is "
                "too large for this knob"
            )
    return jac


def match_insertion(
    lattice: Lattice,
    targets: Sequence[Target],
    knobs: Sequence[Knob],
    *,
    twiss0: Twiss | None = None,
    tol: float = 1e-12,
    max_iter: int = 60,
    fd_step: float = 1e-6,
) -> InsertionMatchResult:
    r"""Match local optics (``beta*``, ``alpha*``, dispersion) at one or more points.

    N quadrupole knobs against M targets, by Gauss-Newton on the finite-difference
    :func:`insertion_response_matrix` with the residual taken from the exact
    optics. This is the H1 pattern one dimension wider: H1 was 2 knobs -> 2 global
    targets twice, this is N -> M on *local* quantities.

    **Two branches, chosen by ``twiss0``.**

    - ``twiss0=None`` (default) — **periodic**: the closed solution is re-solved
      at every evaluation, so a quadrupole legitimately moves the optics
      everywhere, including upstream of itself. This is the ring case.
    - ``twiss0=<Twiss>`` — **transfer line**: the optics are propagated from a
      fixed entrance, which is how a real insertion is matched (from the exit
      Twiss of the periodic arc cell, into the interaction point). No periodicity
      is imposed and the lattice need not be stable.

    **N and M need not be equal, and the step is honest about which case it is.**
    The step comes from :func:`numpy.linalg.lstsq` on the weighted Jacobian: for
    ``N > M`` (more knobs than targets) that is the *minimum-norm* solution, which
    is the right default — it moves the strengths as little as the target allows
    instead of picking an arbitrary point of the solution family. For ``N < M`` it
    is the least-squares step, which converges to a floor rather than to zero.
    **A least-squares floor is not success**: the match is only reported as
    converged when the weighted residual reaches ``tol``, and otherwise raises
    with the per-target misses, so an over-constrained problem cannot be mistaken
    for a solved one.

    ``alpha = 0`` is a waist, and a waist target is generally **not unique** — for
    a single thin lens the condition is a quadratic in ``1/f``, with two roots and
    two different emergent ``beta*``. Newton converges to whichever root the
    starting strengths are nearest; there is no "the" solution to select.

    Each step is backtracked (halved) until it both keeps the optics computable
    and reduces the weighted residual, exactly as in :func:`match_tunes` — in the
    periodic branch a long step routinely crosses the stability boundary, where
    :func:`accsim.twiss.closed_twiss` raises rather than returning a wrong number.

    **Mutates ``lattice`` in place** on success; restores every original strength
    if it raises.

    Raises :class:`MatchingError` if a target names an out-of-range boundary, if a
    knob is a sextupole (whose linear map is a drift and so cannot move local
    optics at all), if the knobs are degenerate, if no step reduces the residual,
    or if ``max_iter`` is exhausted.
    """
    knobs = tuple(knobs)
    if not knobs:
        raise MatchingError("match_insertion needs at least one knob")
    targets = _check_targets(lattice, targets)
    _check_optics_knobs(knobs)
    _knob_index(knobs)  # fail fast on overlapping knobs, before touching anything

    for knob in knobs:
        knob.check_ganged()  # ``value`` is only meaningful while the family is ganged
    snapshots = [k.snapshot() for k in knobs]
    initial = tuple(k.value for k in knobs)

    wanted = np.array([float(t.value) for t in targets])
    weight = np.array([t.scale for t in targets])

    def weighted(achieved: np.ndarray) -> np.ndarray:
        return weight * (achieved - wanted)

    def rollback() -> None:
        for knob, snap in zip(knobs, snapshots, strict=True):
            knob.restore(snap)

    def report(misses: np.ndarray) -> str:
        return ", ".join(f"{t} (off by {m:+.3g})" for t, m in zip(targets, misses, strict=True))

    try:
        try:
            achieved = _observe(lattice, targets, twiss0)
        except UnstableLatticeError as exc:
            raise MatchingError(f"the starting lattice has no matched optics: {exc}") from exc
        resid = weighted(achieved)

        v = np.array(initial, dtype=float)
        for iteration in range(max_iter + 1):
            norm = float(np.linalg.norm(resid))
            if norm <= tol:
                misses = achieved - wanted
                return InsertionMatchResult(
                    values=tuple(float(x) for x in v),
                    initial=initial,
                    targets=targets,
                    achieved=tuple(float(x) for x in achieved),
                    residuals=tuple(float(x) for x in misses),
                    residual=norm,
                    iterations=iteration,
                )
            if iteration == max_iter:
                break

            jac = insertion_response_matrix(lattice, targets, knobs, twiss0=twiss0, fd_step=fd_step)
            _check_conditioning(
                weight[:, None] * jac,
                "insertion response matrix",
                "two knobs at equivalent optics move the observation point the same way, "
                "and a knob downstream of it cannot move it at all in a transfer line",
            )
            step = np.linalg.lstsq(weight[:, None] * jac, -resid, rcond=None)[0]

            # Backtrack until the step is both computable and an improvement.
            lam = 1.0
            while True:
                trial = v + lam * step
                for knob, value in zip(knobs, trial, strict=True):
                    knob.apply(float(value))
                try:
                    new_achieved = _observe(lattice, targets, twiss0)
                except UnstableLatticeError:
                    new_achieved = None  # stepped past the stability boundary
                if new_achieved is not None:
                    new_resid = weighted(new_achieved)
                    if float(np.linalg.norm(new_resid)) < norm:
                        v, resid, achieved = trial, new_resid, new_achieved
                        break
                lam *= 0.5
                if lam < 1e-8:
                    raise MatchingError(
                        f"no step reduced the insertion residual (weighted |r| = {norm:.3g} "
                        f"at iteration {iteration}, tol = {tol:g}) with {len(knobs)} knob(s) "
                        f"for {len(targets)} target(s): {report(achieved - wanted)}. With "
                        "fewer knobs than targets this is a least-squares floor, not a "
                        "solution — the targets are mutually unreachable."
                    )
        raise MatchingError(
            f"insertion matching did not converge in {max_iter} iterations "
            f"(weighted |r| = {float(np.linalg.norm(resid)):.3g}, tol = {tol:g}): "
            f"{report(achieved - wanted)}"
        )
    except Exception:
        rollback()
        raise
