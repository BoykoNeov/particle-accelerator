r"""The transfer map as an object: the Taylor expansion of ``track()`` to second order (P1).

Every map this package returns as a *matrix* is first order. :meth:`Element.matrix` is
the linear map read off the element type, and :func:`~accsim.orbit.linearised_element_maps`
differentiates ``track()`` once. Everything beyond first order — the sextupole kick, the
octupole kick, the amplitude detuning, the driving terms — is expressed *per effect*.
Each is a projection of one object nobody had built: the expansion of the map itself
one order further,

    z_i(out) = k_i + sum_j R_ij u_j + sum_{j,k} T_ijk u_j u_k + O(u^3),   u = z(in) - z_0,

with ``R`` the ``6x6`` the package already has and ``T`` a ``6x6x6`` it did not. This
module builds ``T`` for one element, for one turn, and — the structural content — the
rule that composes them.

**The storage convention is symmetric and it is pinned, because it is a factor of two.**
Two conventions are live for ``sum_{j,k} T_ijk u_j u_k``: summing over *all* ``j, k``
with ``T`` symmetric in its last two indices, or over ``j <= k`` only with the cross terms
carrying a factor ``2``. This module stores the **symmetric** one, so ``T_ijk = T_ikj`` and
the diagonal ``T_ijj`` is the *half* second derivative ``d^2 z_i / d u_j^2 / 2``. On a thin
sextupole of integrated strength ``k2l`` that reads

    T[px, x, x] = -k2l/2,   T[px, y, y] = +k2l/2,   T[py, x, y] = T[py, y, x] = +k2l/2,

and the ratio ``|T[py,x,y]| / |T[px,x,x]|`` is ``1``, not ``2``. MAD-X's ``sectormap``
and xtrack's ``get_T_matrix`` store the same convention entry for entry; MAD-X PTC's
``maptable`` labels *monomials*, so its mixed coefficient is the **sum** of the symmetric
pair — the decode is in ``tests/reference/`` and ``docs/CONVENTIONS.md``.

**The composition rule is exact and blind to no factor.** For ``B`` after ``A``,

    R^BA = R^B R^A,
    T^BA_ijk = sum_a R^B_ia T^A_ajk + sum_{a,b} T^B_iab R^A_aj R^A_bk,

valid when ``B`` is expanded about ``A``'s image of its own expansion point — which is
how :func:`second_order_element_maps` builds them. A wrong ``1/2`` anywhere in ``T``
breaks a ring's composed map against its directly expanded one, and the analytic suite
gates exactly that.

**Symplecticity at second order is a set of exact identities on ``T``, not a
tolerance.** The Jacobian of the truncated map is ``M(u) = R + 2 sum_k T_k u_k`` with
``(T_k)_ij = T_ijk``, and ``M^T J M = J`` at every ``u`` requires, at first order,

    R^T J T_k + T_k^T J R = 0   for every k.

:func:`second_order_symplectic_residual` returns the left-hand side. It holds
identically for every thin kick and for every exact map **in canonical coordinates**;
in accsim's ``(zeta, delta)`` it holds only for a map that couples nothing transverse
into ``zeta`` — the same caveat :mod:`accsim.symplectic` carries at first order, met one
order up. A sector bend fails it in ``(zeta, delta)`` by a closed-form amount and passes
it in ``(zeta, p_zeta)``; see :func:`canonical_map`.

**How ``T`` is obtained, and what that costs.** ``track()`` is differentiated twice by
finite differences on fourth-order stencils — the first-order primitive already
differentiates it once, and no element exposes a Taylor map of its own. The floor is
measured rather than assumed (``docs/CONVENTIONS.md`` → *The second-order transfer
map*): ``~1e-12`` per entry at the default step, the truncation error vanishing
identically for every thin kick (a polynomial of degree at most three under a stencil
exact to degree four) and the round-off scaling as ``eps / step``. It is the same
method xtrack's ``get_T_matrix`` uses, which is why that reference leg gates conventions
and bookkeeping rather than physics; MAD-X's ``sectormap`` and PTC's differential
algebra are analytic and are the legs that gate the numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .coords import DELTA, DIM, PX, PY, X, Y
from .lattice import Lattice
from .reference import ReferenceParticle
from .symplectic import J6, from_canonical, to_canonical

_TRANSVERSE = [X, PX, Y, PY]

#: Fourth-order central first-derivative stencil, ``f' = sum_a c_a f(z + a h) / h``.
_D1 = {-2: 1.0 / 12.0, -1: -2.0 / 3.0, 1: 2.0 / 3.0, 2: -1.0 / 12.0}
#: Fourth-order central second-derivative stencil, ``f'' = sum_a c_a f(z + a h) / h^2``.
_D2 = {-2: -1.0 / 12.0, -1: 4.0 / 3.0, 0: -5.0 / 2.0, 1: 4.0 / 3.0, 2: -1.0 / 12.0}


@dataclass(frozen=True)
class TaylorMap:
    r"""A map expanded to second order about a point.

    ``z_out = k + R (z_in - origin) + T (z_in - origin)(z_in - origin)``, with ``T``
    **symmetric** in its last two indices (module docstring). ``k`` is the image of
    ``origin``, so a one-turn map about a closed orbit has ``k == origin`` to the orbit
    solver's tolerance, and an element map has ``k`` equal to the orbit at its exit.
    """

    origin: np.ndarray
    """The expansion point ``z_0``, shape ``(6,)``."""
    k: np.ndarray
    """The image of ``origin``, shape ``(6,)``."""
    R: np.ndarray
    """The first-order map, shape ``(6, 6)``."""
    T: np.ndarray
    """The second-order map, shape ``(6, 6, 6)``, symmetric in its last two indices."""

    def __post_init__(self) -> None:
        shapes = (("origin", (DIM,)), ("k", (DIM,)), ("R", (DIM, DIM)), ("T", (DIM,) * 3))
        for name, shape in shapes:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            object.__setattr__(self, name, value)

    @classmethod
    def identity(cls, origin: np.ndarray | None = None) -> TaylorMap:
        """The identity map about ``origin`` (the design orbit by default)."""
        o = np.zeros(DIM) if origin is None else np.asarray(origin, dtype=float)
        return cls(o, o.copy(), np.eye(DIM), np.zeros((DIM,) * 3))

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Evaluate the truncated map on a ``(6,)`` state or a ``(6, n)`` bunch."""
        z = np.asarray(state, dtype=float)
        if z.shape[:1] != (DIM,) or z.ndim not in (1, 2):
            raise ValueError(f"expected a (6,) state or a (6, n) bunch, got shape {z.shape}")
        u = z - (self.origin if z.ndim == 1 else self.origin[:, None])
        if z.ndim == 1:
            return self.k + self.R @ u + np.einsum("ijk,j,k->i", self.T, u, u)
        return self.k[:, None] + self.R @ u + np.einsum("ijk,jn,kn->in", self.T, u, u)

    def symmetry_defect(self) -> float:
        """``max |T_ijk - T_ikj|`` — zero for a map stored in the symmetric convention."""
        return float(np.max(np.abs(self.T - np.swapaxes(self.T, 1, 2))))

    def then(self, other: TaylorMap, *, atol: float = 1e-9) -> TaylorMap:
        """``other`` after ``self`` — beam order. See :func:`compose`."""
        return compose(other, self, atol=atol)

    def __matmul__(self, other: TaylorMap) -> TaylorMap:
        """``self @ other`` is ``self`` after ``other`` — matrix order, like ``R @ R``."""
        return compose(self, other)


def compose(outer: TaylorMap, inner: TaylorMap, *, atol: float = 1e-9) -> TaylorMap:
    r"""The map ``outer . inner`` to second order — the rule in the module docstring.

    ``outer`` must be expanded about ``inner``'s image of its own expansion point,
    ``outer.origin == inner.k``; otherwise the two expansions are about different
    points and the rule does not apply. That is checked to ``atol`` rather than
    assumed, because it is exactly the mistake a caller composing maps taken about the
    design orbit with maps taken about a steered one would make silently.
    """
    if not np.allclose(outer.origin, inner.k, rtol=0.0, atol=atol):
        raise ValueError(
            "cannot compose: the outer map is expanded about "
            f"{outer.origin} but the inner map sends its origin to {inner.k} "
            f"(max difference {np.max(np.abs(outer.origin - inner.k)):.3g} > atol {atol:.3g})"
        )
    R = outer.R @ inner.R
    T = np.einsum("ia,ajk->ijk", outer.R, inner.T) + np.einsum(
        "iab,aj,bk->ijk", outer.T, inner.R, inner.R
    )
    return TaylorMap(inner.origin.copy(), outer.k.copy(), R, T)


def _steps(step: float | np.ndarray) -> np.ndarray:
    h = np.broadcast_to(np.asarray(step, dtype=float), (DIM,)).copy()
    if np.any(h <= 0.0):
        raise ValueError(f"step must be > 0 in every coordinate, got {h}")
    return h


def taylor_expand(
    map_fn: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    step: float | np.ndarray = 5e-4,
    vectorised: bool = True,
) -> TaylorMap:
    r"""``(k, R, T)`` of ``map_fn`` about ``state``, by fourth-order finite differences.

    ``R`` from the five-point first-derivative stencil and ``T`` from the five-point
    second-derivative one on the diagonal and the ``4x4``-point product stencil off it —
    ``265`` evaluations in all, gathered into one ``(6, 265)`` bunch and pushed through
    ``map_fn`` in a single call when ``vectorised`` (every element's ``track`` accepts a
    bunch). With ``vectorised=False`` the samples are pushed one at a time, for a map
    that only takes a ``(6,)`` state — :func:`canonical_map`'s conjugation, for one.

    ``step`` is absolute, one number or one per coordinate. The default ``5e-4`` sits
    where the two error terms cross for the maps in this package: the stencils are exact
    on polynomials up to degree four, so every thin kick is differentiated *exactly* and
    only round-off remains, scaling as ``eps / step``; a thick map's sixth derivative
    enters as ``step^4``. Measured on a 1 m drift (analytic suite): ``4e-13`` at ``5e-4``,
    ``6e-12`` at ``1e-3``, ``3e-14`` at ``2.5e-4``; a 36-element ring composed against
    PTC: ``1.7e-10`` at ``5e-4``, ``1.2e-9`` at ``1e-3``. The one map that prefers a
    *larger* step is the RF cavity, whose own ``sin(phi_s - k zeta) - sin(phi_s)``
    cancellation is the floor (``4e-7`` relative on its curvature at ``5e-4``, ``4e-10``
    at ``1e-2``); the per-coordinate form exists for it.
    """
    z0 = np.asarray(state, dtype=float)
    if z0.shape != (DIM,):
        raise ValueError(f"expected a length-{DIM} state vector, got shape {z0.shape}")
    h = _steps(step)

    # Assemble every sample point once: index 0 is the origin; then the 24 axial points
    # (j, a) for a in (-2, -1, 1, 2); then the 240 off-diagonal points (j < k, a, b).
    samples: list[np.ndarray] = [z0]
    axial: dict[tuple[int, int], int] = {}
    for j in range(DIM):
        for a in (-2, -1, 1, 2):
            axial[j, a] = len(samples)
            p = z0.copy()
            p[j] += a * h[j]
            samples.append(p)
    mixed: dict[tuple[int, int, int, int], int] = {}
    for j in range(DIM):
        for k in range(j + 1, DIM):
            for a in (-2, -1, 1, 2):
                for b in (-2, -1, 1, 2):
                    mixed[j, k, a, b] = len(samples)
                    p = z0.copy()
                    p[j] += a * h[j]
                    p[k] += b * h[k]
                    samples.append(p)
    points = np.array(samples).T  # (6, 265)

    if vectorised:
        values = np.asarray(map_fn(points), dtype=float)
        if values.shape != points.shape:
            raise ValueError(
                f"map_fn returned shape {values.shape} for a {points.shape} bunch; pass "
                "vectorised=False for a map that only takes a (6,) state"
            )
    else:
        values = np.empty_like(points)
        for n in range(points.shape[1]):
            values[:, n] = np.asarray(map_fn(points[:, n]), dtype=float)

    k = values[:, 0].copy()
    R = np.empty((DIM, DIM))
    T = np.zeros((DIM, DIM, DIM))
    for j in range(DIM):
        R[:, j] = sum(c * values[:, axial[j, a]] for a, c in _D1.items()) / h[j]
        d2 = _D2[0] * values[:, 0] + sum(
            c * values[:, axial[j, a]] for a, c in _D2.items() if a != 0
        )
        T[:, j, j] = 0.5 * d2 / (h[j] * h[j])
    for j in range(DIM):
        for k_ in range(j + 1, DIM):
            d2 = sum(
                ca * cb * values[:, mixed[j, k_, a, b]]
                for a, ca in _D1.items()
                for b, cb in _D1.items()
            )
            T[:, j, k_] = T[:, k_, j] = 0.5 * d2 / (h[j] * h[k_])
    return TaylorMap(z0.copy(), k, R, T)


def second_order_symplectic_residual(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    r"""``S[:, :, k] = R^T J T_k + T_k^T J R`` with ``(T_k)_ij = T_ijk`` — zero iff symplectic.

    The first-order-in-amplitude condition for the truncated map's Jacobian to satisfy
    ``M^T J M = J`` at every point (module docstring; derived symbolically in the
    analytic suite). Exact — no step, no tolerance — for a ``T`` obtained exactly; for a
    finite-differenced ``T`` it reports the differencing floor, which is the point.
    """
    R = np.asarray(R, dtype=float)
    T = np.asarray(T, dtype=float)
    if R.shape != (DIM, DIM) or T.shape != (DIM,) * 3:
        raise ValueError(f"expected R (6, 6) and T (6, 6, 6), got {R.shape} and {T.shape}")
    out = np.empty_like(T)
    for k in range(DIM):
        Tk = T[:, :, k]
        out[:, :, k] = R.T @ J6 @ Tk + Tk.T @ J6 @ R
    return out


def canonical_map(
    map_fn: Callable[[np.ndarray], np.ndarray], ref: ReferenceParticle
) -> Callable[[np.ndarray], np.ndarray]:
    r"""``map_fn`` conjugated into ``(x, px, y, py, zeta, p_zeta)`` — the canonical pair.

    ``g = to_canonical . map_fn . from_canonical``: takes and returns states whose sixth
    coordinate is ``p_zeta`` rather than ``delta``. Expand *this* with
    :func:`taylor_expand` when the symplectic identity is the question, because
    ``(zeta, delta)`` is not a conjugate pair and an exact map's ``T`` fails the identity
    there for a reason that is not a bug (module docstring). Takes a ``(6,)`` state or a
    ``(6, n)`` bunch, as the two conversions do.
    """

    def conjugated(canonical: np.ndarray) -> np.ndarray:
        return to_canonical(map_fn(from_canonical(canonical, ref)), ref)

    return conjugated


def _expansion_point(lattice: Lattice, orbit0: np.ndarray | None, delta: float) -> np.ndarray:
    from .orbit import closed_orbit_nonlinear

    if orbit0 is None:
        o = closed_orbit_nonlinear(lattice, delta=delta)
    else:
        o = np.asarray(orbit0, dtype=float)
    if o.shape == (DIM,):
        return o.copy()
    if o.shape != (4,):
        raise ValueError(
            f"orbit0 must be a length-4 (x, px, y, py) or length-6 vector, got {o.shape}"
        )
    state = np.zeros(DIM)
    state[_TRANSVERSE] = o
    state[DELTA] = delta
    return state


def second_order_element_maps(
    lattice: Lattice,
    orbit0: np.ndarray | None = None,
    *,
    delta: float = 0.0,
    step: float | np.ndarray = 5e-4,
) -> list[TaylorMap]:
    r"""Each element's ``(k, R, T)`` expanded about the orbit at its entrance.

    The second-order sibling of :func:`~accsim.orbit.linearised_element_maps`, and
    built the same way: the expansion point is the (nonlinear) closed orbit at momentum
    ``delta`` — or ``orbit0``, a ``(x, px, y, py)`` vector taken at that ``delta``, or a
    full 6D state, say the one :func:`~accsim.orbit.closed_orbit_6d` returns for a ring
    with RF — and each element is expanded about the tracked orbit at its own entrance,
    so that map ``n+1`` is expanded exactly about map ``n``'s ``k`` and
    :func:`compose` applies without approximation.

    On the design orbit every element's ``R`` is its ``matrix()`` to the differencing
    floor, and its ``T`` is the second-order content that ``matrix()`` cannot carry: a
    thin sextupole's kick, a drift's ``-L px delta`` chromatic term and ``-L px^2 / 2``
    path lengthening, a thick quadrupole's ``k1/(1 + delta)`` focusing, a bend's
    geometric aberrations. On a steered orbit ``R`` picks up the feed-down gradients
    exactly as the first-order sibling does, while a thin sextupole's ``T`` is
    **unchanged** — its second derivative is constant — which is the sharpest single
    statement of what feed-down is.
    """
    ref = lattice.ref
    state = _expansion_point(lattice, orbit0, delta)
    maps = []
    for elem in lattice.elements:
        maps.append(taylor_expand(lambda s, e=elem: e.track(s, ref), state, step=step))
        state = elem.track(state, ref)
    return maps


def second_order_one_turn_map(
    lattice: Lattice,
    orbit0: np.ndarray | None = None,
    *,
    delta: float = 0.0,
    step: float | np.ndarray = 5e-4,
) -> TaylorMap:
    r"""The one-turn ``(k, R, T)`` about the closed orbit, composed element by element.

    :func:`second_order_element_maps` joined by :func:`compose` in beam order. Its ``R``
    is :func:`~accsim.orbit.linearised_one_turn_map` to the differencing floor, and its
    ``T`` is the object the chromaticity (``dR/ddelta = 2 T[:, :, delta]`` on a
    dispersion-free ring), the second-order dispersion (the ``T[:, delta, delta]``
    column, through the fixed point) and the first-order driving terms are projections
    of — each of which the analytic suite reproduces from it.

    Composing is preferred over expanding the whole turn in one go because the
    element-wise floors add while a turn's higher derivatives multiply; the two agree to
    the measured floor and the suite gates that too.
    """
    maps = second_order_element_maps(lattice, orbit0, delta=delta, step=step)
    turn = TaylorMap.identity(maps[0].origin if maps else _expansion_point(lattice, orbit0, delta))
    for m in maps:
        turn = turn.then(m)
    return turn
