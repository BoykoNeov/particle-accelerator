r"""I3 acceptance: linear optics evaluated on the real (steered) closed orbit.

I2 established that a sextupole at an orbit offset ``(x_co, y_co)`` splits into a
dipole, a normal quadrupole ``k1l_eff = k2l x_co``, a skew quadrupole
``k1sl_eff = k2l y_co``, and the sextupole itself, and it shipped
:func:`~accsim.orbit.linearised_element_maps` — each element's map *about that
orbit*. It then stopped, and asserted the gap it left: every optics function in
the package still walks each element's **on-axis** ``matrix()``, so a steered
machine is reported with unperturbed ``beta``, dispersion and chromaticity
however far off-axis the beam actually is.

This file is that gap closed. Two things are genuinely new here, and only two:

1. **beta as a function of ``s``.** The *value at the ring start* is not new —
   I2 already gates ``match_periodic(linearised_one_turn_map(...))``, and
   "the propagated table multiplies back to the one-turn map" is vacuous, since
   that map **is** the product of the per-element maps. What is new is the
   ``s``-dependence, and the gate for it is the closed form for a single
   gradient error,

       dbeta(s)/beta(s) = -Delta_k1l beta(s_src) cos(2 |dpsi| - 2 pi Q) / (2 sin 2 pi Q)

   (``+`` in ``y``, where the same element defocuses), **derived** in
   :func:`test_beta_beat_closed_form_is_derived` rather than recalled, against
   accsim's own :class:`~accsim.elements.quadrupole.ThinQuadrupole` and its own
   Twiss propagation rule. It pins the magnitude *and* the phase dependence.

2. **Chromaticity.** Here the package's own structure decides the route.
   accsim's linear element maps carry **no ``delta`` dependence of their own**
   (J1), so linearising the tracked map about the off-momentum orbit measures the
   sextupole feed-down term and *nothing else* — it is exactly blind to the
   natural chromaticity, which accsim supplies analytically. So the implementation
   evaluates the existing (F2-validated) chromaticity integrals on the equivalent
   linear lattice, and the **tracked** route is kept as the independent gate on
   the half it can see: :func:`test_tracked_feeddown_matches_the_on_orbit_split`.

The lattices are deliberately different for the two halves. The beta-beat ring is
a **dispersion-free** thin FODO (drifts and thin quads only) so that the beat has
exactly one source — the feed-down gradient — with no dispersion term muddying it;
the sextupole sits **mid-ring** so that sample points exist on both sides of it
and the ``|dpsi|`` branch is actually exercised. The chromaticity ring has dipoles,
because a sextupole only feeds down to chromaticity at dispersion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from accsim import (
    Corrector,
    Dipole,
    Drift,
    Lattice,
    Quadrupole,
    ReferenceParticle,
    Sextupole,
    ThinQuadrupole,
    ThinSextupole,
    chromaticity,
    chromaticity_on_orbit,
    closed_orbit,
    closed_orbit_nonlinear,
    closed_twiss,
    closed_twiss_on_orbit,
    coupled_twiss,
    coupled_twiss_on_orbit,
    linearised_element_maps,
    linearised_lattice,
    linearised_one_turn_map,
    match_periodic,
    natural_chromaticity,
    natural_chromaticity_on_orbit,
    propagate_orbit_nonlinear,
    propagate_twiss,
    propagate_twiss_on_orbit,
    tunes,
    tunes_on_orbit,
)
from accsim.coords import DELTA, DIM, PX, PY, X, Y
from accsim.twiss import CoupledLatticeError

# --------------------------------------------------------------------------
# The dispersion-free ring for the beta-beat gates
# --------------------------------------------------------------------------

VF = 1.0 / 1.5  # full-quad inverse focal length, F family [m^-1]
VD = 1.0 / 1.6  # ditto, D family [m^-1]
L_HALF = 1.0  # half-cell drift [m]
N_CELLS = 6
SX_AFTER = 3  # sextupole inserted after this many cells: mid-ring, not at s = 0
SX_INDEX = 1 + SX_AFTER * 5  # element index of the sextupole (1 = past the steerer)

K2L = 20.0  # integrated sextupole strength [m^-2]
KICK = 2e-4  # steerer angle [rad] -> a sub-mm orbit


@pytest.fixture
def ref() -> ReferenceParticle:
    # Thin quads + drifts are energy-independent; any reference works.
    return ReferenceParticle.from_gamma(938.27208816e6, 20.0)


def _cell(tag: str) -> list:
    return [
        ThinQuadrupole(0.5 * VF, name=f"qf_a{tag}"),
        Drift(L_HALF, name=f"d1{tag}"),
        ThinQuadrupole(-VD, name=f"qd{tag}"),
        Drift(L_HALF, name=f"d2{tag}"),
        ThinQuadrupole(0.5 * VF, name=f"qf_b{tag}"),
    ]


def _flat(
    ref: ReferenceParticle,
    k2l: float = 0.0,
    kick_x: float = 0.0,
    kick_y: float = 0.0,
    n_cells: int = N_CELLS,
):
    """Steerer at ``s = 0``, one thin sextupole **mid-ring**, dispersion-free FODO."""
    els: list = [Corrector(kick_x=kick_x, kick_y=kick_y, name="steerer")]
    for i in range(n_cells):
        if i == SX_AFTER:
            els.append(ThinSextupole(k2l, name="sx"))
        els.extend(_cell(f"_{i}"))
    return Lattice(els, ref)


# --------------------------------------------------------------------------
# 1. The closed form, derived
# --------------------------------------------------------------------------


def test_beta_beat_closed_form_is_derived() -> None:
    r"""The single-gradient-error beat, from accsim's own quadrupole and Twiss rule.

    Nothing here is recalled from a textbook and then trusted. The Courant-Snyder
    transfer parameterisation is first *checked against accsim's own propagation
    rule* — :func:`accsim.twiss._propagate_block` transports ``(beta, alpha)`` as
    ``B1 = C B C^T`` with ``B = [[beta, -alpha], [-alpha, gamma]]`` and reads the
    phase off ``atan2(C12, beta C11 - alpha C12)`` — and the two arcs are shown to
    multiply back to the Courant-Snyder one-turn form. Only then is the thin
    quadrupole inserted, and the quadrupole used is accsim's own sign convention
    ``px -> px - k1l x``.

    The square-root branch of ``sin mu`` never appears: ``beta = M12 / sin mu`` is
    differentiated using ``d(sin)/dk = -cos d(cos)/dk / sin``, which is rational in
    ``sin mu`` and ``cos mu``. That matters because this ring runs at
    ``Q_x = 0.690``, where ``sin 2 pi Q < 0``.
    """
    b0, bs = sp.symbols("beta_0 beta_s", positive=True)
    a0, a_s = sp.symbols("alpha_0 alpha_s", real=True)
    psi, mu, dk = sp.symbols("psi mu Deltak", real=True)

    def cs(b1, a1, b2, a2, phi):
        """Transfer matrix from ``(b1, a1)`` to ``(b2, a2)`` over phase advance ``phi``."""
        return sp.Matrix(
            [
                [
                    sp.sqrt(b2 / b1) * (sp.cos(phi) + a1 * sp.sin(phi)),
                    sp.sqrt(b1 * b2) * sp.sin(phi),
                ],
                [
                    ((a1 - a2) * sp.cos(phi) - (1 + a1 * a2) * sp.sin(phi)) / sp.sqrt(b1 * b2),
                    sp.sqrt(b1 / b2) * (sp.cos(phi) - a2 * sp.sin(phi)),
                ],
            ]
        )

    # (a) the parameterisation reproduces accsim's own transport of (beta, alpha)...
    T = cs(b0, a0, bs, a_s, psi)
    B0 = sp.Matrix([[b0, -a0], [-a0, (1 + a0**2) / b0]])
    B1 = sp.simplify(T * B0 * T.T)
    assert sp.simplify(T.det() - 1) == 0
    assert sp.simplify(B1[0, 0] - bs) == 0
    assert sp.simplify(-B1[0, 1] - a_s) == 0
    # ...and accsim's phase rule dmu = atan2(C12, beta C11 - alpha C12) gives psi.
    assert sp.simplify(T[0, 1] - sp.sqrt(b0 * bs) * sp.sin(psi)) == 0
    assert sp.simplify(b0 * T[0, 0] - a0 * T[0, 1] - sp.sqrt(b0 * bs) * sp.cos(psi)) == 0

    # (b) observer -> (phase psi) -> source -> (phase mu - psi) -> observer is one turn.
    T1 = cs(bs, a_s, b0, a0, psi)  # observer to the error
    T2 = cs(b0, a0, bs, a_s, mu - psi)  # the error back around to the observer
    turn = sp.Matrix(
        [
            [sp.cos(mu) + a_s * sp.sin(mu), bs * sp.sin(mu)],
            [-(1 + a_s**2) / bs * sp.sin(mu), sp.cos(mu) - a_s * sp.sin(mu)],
        ]
    )
    assert sp.simplify(sp.expand_trig(T2 * T1 - turn)) == sp.zeros(2, 2)

    # (c) insert accsim's thin quadrupole and expand beta to first order in k1l.
    M = T2 * sp.Matrix([[1, 0], [-dk, 1]]) * T1  # ThinQuadrupole: px -> px - k1l x
    cos_new = sp.trigsimp(sp.expand_trig(sp.simplify(M.trace() / 2)))
    m12 = sp.trigsimp(sp.expand_trig(sp.simplify(M[0, 1])))
    # beta = M12 / sin(mu), with d(sin)/dk = -cos d(cos)/dk / sin: no sqrt, no branch.
    rel = sp.simplify(
        sp.diff(m12, dk).subs(dk, 0) / (bs * sp.sin(mu))
        + sp.cos(mu) * sp.diff(cos_new, dk).subs(dk, 0) / sp.sin(mu) ** 2
    )
    target = -b0 * sp.cos(2 * psi - mu) / (2 * sp.sin(mu))
    assert sp.simplify(sp.expand_trig(rel - target)) == 0


def _beat_prediction(design, src_index: int, dk: float, plane: str) -> list[float]:
    """``dbeta/beta`` at every boundary from the derived single-gradient closed form.

    ``dk`` is the feed-down gradient in accsim's ``ThinQuadrupole.k1l`` convention.
    The vertical sign is opposite because the same element defocuses in ``y``.
    """
    mu = design[-1].mu_x if plane == "x" else design[-1].mu_y
    psi_src = design[src_index].mu_x if plane == "x" else design[src_index].mu_y
    beta_src = design[src_index].beta_x if plane == "x" else design[src_index].beta_y
    sign = -1.0 if plane == "x" else +1.0
    out = []
    for t in design:
        psi = t.mu_x if plane == "x" else t.mu_y
        out.append(
            sign * dk * beta_src * math.cos(2.0 * abs(psi - psi_src) - mu) / (2.0 * math.sin(mu))
        )
    return out


def test_beta_beat_matches_the_closed_form_and_converges_at_the_right_order(
    ref: ReferenceParticle,
) -> None:
    r"""**The headline gate.** The steered machine's ``beta(s)`` is the predicted beat.

    Four steerer sizes, both planes, *every* element boundary. The content is the
    **order**, in the shape I2 used: the beat itself is first order in the orbit
    offset and so halves per halving of the steerer, while the residual against the
    first-order closed form is second order and so falls by four. A feed-down
    gradient off by a constant factor would break the beat ratio; a wrongly
    propagated phase would break individual points without touching either ratio,
    which is why every boundary is compared and not just the extremes.

    The sextupole sits mid-ring on purpose, so roughly half the sample points lie
    upstream of it: the ``cos(2 |dpsi| - 2 pi Q)`` branch is only exercised when
    both signs of ``psi - psi_src`` occur.

    ``Q_y = 0.559`` sits 0.059 from the half integer, where the beat formula's
    ``1 / sin 2 pi Q`` amplifies by about 2.6x relative to the horizontal. That is
    **deliberate** — it makes the vertical the more demanding plane — so retuning
    this ring away from the half integer would silently weaken the gate.
    """
    beats, residuals, offsets = [], [], []
    for kick in (4e-4, 2e-4, 1e-4, 5e-5):
        design = propagate_twiss(_flat(ref, kick_x=kick), closed_twiss(_flat(ref, kick_x=kick)))
        lat = _flat(ref, k2l=K2L, kick_x=kick)
        x_co = propagate_orbit_nonlinear(lat)[SX_INDEX][X]
        on_orbit = propagate_twiss_on_orbit(lat)
        assert len(on_orbit) == len(design)

        worst_beat = worst_resid = 0.0
        for plane in ("x", "y"):
            predicted = _beat_prediction(design, SX_INDEX, K2L * x_co, plane)
            for point, des, pred in zip(on_orbit, design, predicted, strict=True):
                bd = des.beta_x if plane == "x" else des.beta_y
                bo = point.beta_x if plane == "x" else point.beta_y
                measured = bo / bd - 1.0
                worst_beat = max(worst_beat, abs(measured))
                worst_resid = max(worst_resid, abs(measured - pred))
        beats.append(worst_beat)
        residuals.append(worst_resid)
        offsets.append(abs(x_co))

    # The orbit offset itself halves, so the comparison is against the steerer.
    for a, b in zip(offsets, offsets[1:], strict=False):
        assert a / b == pytest.approx(2.0, rel=0.02)
    # First order: the beat halves. Measured 2.12 / 2.05 / 2.01 (2026-08-10).
    for a, b in zip(beats, beats[1:], strict=False):
        assert 1.9 < a / b < 2.3
    # Second order: the residual falls by four. Measured 4.30 / 4.14 / 4.07.
    for a, b in zip(residuals, residuals[1:], strict=False):
        assert 3.7 < a / b < 4.6
    # ...and it is a residual, not the whole signal: ~10 % of the beat at the
    # largest steerer, ~1 % at the smallest.
    assert residuals[0] / beats[0] < 0.15
    assert residuals[-1] / beats[-1] < 0.02


def test_a_mis_scaled_feeddown_gradient_breaks_the_beat_ratio(ref: ReferenceParticle) -> None:
    """The beat gate has teeth: doubling the predicted gradient is caught outright.

    I2's warning was that structural checks pass for a consistently mis-scaled
    sextupole. This asserts the beat comparison is not one of them — the closed
    form is compared against a *predicted* gradient, so a factor-two error in that
    gradient shows up immediately and at every steerer size, rather than hiding in
    the convergence order.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK)
    design = propagate_twiss(_flat(ref, kick_x=KICK), closed_twiss(_flat(ref, kick_x=KICK)))
    x_co = propagate_orbit_nonlinear(lat)[SX_INDEX][X]
    on_orbit = propagate_twiss_on_orbit(lat)

    good = _beat_prediction(design, SX_INDEX, K2L * x_co, "x")
    bad = _beat_prediction(design, SX_INDEX, 2.0 * K2L * x_co, "x")
    err_good = max(
        abs(p.beta_x / d.beta_x - 1.0 - g) for p, d, g in zip(on_orbit, design, good, strict=True)
    )
    err_bad = max(
        abs(p.beta_x / d.beta_x - 1.0 - b) for p, d, b in zip(on_orbit, design, bad, strict=True)
    )
    assert err_bad > 20.0 * err_good


# --------------------------------------------------------------------------
# 2. The default path is untouched, and the finite-difference floor is measured
# --------------------------------------------------------------------------


def test_the_design_orbit_path_is_bit_identical(ref: ReferenceParticle) -> None:
    """``maps=None`` reproduces today's answer exactly, element by element.

    The new ``maps`` argument must not perturb the existing design-orbit result by
    so much as a round-off, so this compares the full table entry by entry with
    ``==``, not a tolerance. The companion is that passing the *on-axis* matrices
    explicitly is the same thing again — which is what makes ``maps`` a
    substitution of the transport and nothing else.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK)
    tw0 = closed_twiss(lat)
    default = propagate_twiss(lat, tw0)
    explicit = propagate_twiss(lat, tw0, maps=[e.matrix(ref) for e in lat.elements])
    for a, b in zip(default, explicit, strict=True):
        assert a == b


def test_the_finite_difference_floor_is_measured_not_assumed(ref: ReferenceParticle) -> None:
    """With no sextupole the on-orbit optics *is* the design optics, to a floor.

    ``linearised_element_maps`` central-differences ``track()``, so this can never
    be exact; the honest statement is the size of the floor. Measured 2026-08-10:
    ``1.9e-13`` on individual element maps, ``2.4e-12`` on the one-turn map, and
    ``<1e-12`` relative on ``beta`` — far below the ``~1e-2`` beat the gates above
    measure, which is the point.

    **"Nothing nonlinear to feed down" is no longer the same as "nothing nonlinear".**
    A :class:`~accsim.elements.drift.Drift`'s exact map departs from its matrix at a
    closed-orbit angle by ``L p_co`` — ``1.9e-4`` on this ring, nine orders above the
    floor. So the floor is measured here on the elements that still have one, and the
    drift is checked against its *derived* departure instead. The unsteered case, where
    every element does return its matrix to the floor, is the test below.
    """
    lat = _flat(ref, k2l=0.0, kick_x=KICK)  # steered, but nothing nonlinear to feed down
    orbit = propagate_orbit_nonlinear(lat)[:-1]
    drifts = 0
    for m, elem, o in zip(linearised_element_maps(lat), lat.elements, orbit, strict=True):
        gap = float(np.abs(m - elem.matrix(ref)).max())
        if isinstance(elem, Drift) and max(abs(o[PX]), abs(o[PY])) > 1e-9:
            assert gap == pytest.approx(elem.length * max(abs(o[PX]), abs(o[PY])), rel=1e-3)
            drifts += 1
        else:
            assert gap < 1e-11
    assert drifts > 0  # the derived branch was actually exercised

    # beta, the phases and the tunes still land on the design optics to a floor, because
    # the drift's new entries are the `delta` column and the `zeta` row — neither of
    # which a 2x2 Courant-Snyder reduction of the transverse blocks reads. The
    # transverse block itself moves only at O(angle^2), hence the 1e-7 here against the
    # 1e-11 of an unsteered ring.
    design = propagate_twiss(lat, closed_twiss(lat))
    for a, b in zip(propagate_twiss_on_orbit(lat), design, strict=True):
        assert a.beta_x == pytest.approx(b.beta_x, rel=1e-6)
        assert a.beta_y == pytest.approx(b.beta_y, rel=1e-6)
        assert a.mu_x == pytest.approx(b.mu_x, abs=1e-6)
        assert a.mu_y == pytest.approx(b.mu_y, abs=1e-6)
    assert tunes_on_orbit(lat) == pytest.approx(tunes(lat), abs=1e-6)


def test_an_unsteered_machine_reports_the_design_optics(ref: ReferenceParticle) -> None:
    """A live sextupole on-axis moves nothing — the I1 claim, now for the optics too.

    With ``kick = 0`` the closed orbit is exactly zero, so every feed-down gradient
    is zero and the on-orbit optics collapses onto the design optics. This is the
    statement that the new functions add no spurious signal of their own.
    """
    lat = _flat(ref, k2l=K2L)
    assert np.abs(closed_orbit_nonlinear(lat)).max() == 0.0
    assert closed_twiss_on_orbit(lat).beta_x == pytest.approx(closed_twiss(lat).beta_x, rel=1e-12)
    assert tunes_on_orbit(lat) == pytest.approx(tunes(lat), abs=1e-11)


# --------------------------------------------------------------------------
# 3. The two routes to the on-orbit optics agree
# --------------------------------------------------------------------------


def test_the_derived_split_and_the_jacobian_agree(ref: ReferenceParticle) -> None:
    r"""I2's *derived* four-element split == the *finite-differenced* Jacobian.

    :func:`linearised_lattice` builds the equivalent machine from I2's symbolically
    derived coefficients (``k1l_eff = k2l x_co``, ``k1sl_eff = k2l y_co``);
    :func:`linearised_element_maps` measures the same thing by central-differencing
    ``track()``. Neither is derived from the other, so their agreement is a real
    check on the split — and it is what lets the chromaticity functions be built on
    the first while the beta functions are built on the second.

    The dipole part of the feed-down is deliberately absent from both: a Jacobian
    carries the linear part only, and that kick has already done its work in
    placing the orbit these maps are taken about.

    Each drift's differentiated map is replaced by its own ``matrix`` before the product
    is formed. The exact drift map departs from that matrix wherever the orbit has an
    angle, and :func:`linearised_lattice` — built from accsim elements — has nothing to
    represent the departure with, so leaving it in would compare the sextupole split
    against the sum of two unrelated effects. The drift's own terms are gated in
    ``test_exact_drift_dispersion.py``; this test is about the split.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK, kick_y=KICK)
    equivalent = linearised_lattice(lat)
    co = closed_orbit_nonlinear(lat)

    product = np.eye(DIM)
    for elem, m in zip(lat.elements, linearised_element_maps(lat, co), strict=True):
        product = (elem.matrix(ref) if isinstance(elem, Drift) else m) @ product
    assert np.abs(equivalent.one_turn_matrix() - product).max() < 1e-10


def test_a_thick_sextupole_is_refused_rather_than_approximated(ref: ReferenceParticle) -> None:
    """The scope line, enforced: a thick body's orbit varies across it.

    Collapsing a thick sextupole onto one gradient at its entrance orbit would
    carry an ``O(L^2)`` error — precisely the error I2 avoided by using thin
    sextupoles throughout its gates. Refusing is the honest answer;
    :func:`propagate_twiss_on_orbit` has no such limit, because it differentiates
    the thick element's real ``track()``.
    """
    els = [Corrector(kick_x=KICK, name="steerer"), Sextupole(0.4, K2L / 0.4, name="sx")]
    els += [e for i in range(N_CELLS) for e in _cell(f"_{i}")]
    lat = Lattice(els, ref)
    with pytest.raises(NotImplementedError, match="thick"):
        linearised_lattice(lat)
    with pytest.raises(NotImplementedError, match="thick"):
        chromaticity_on_orbit(lat)
    # ...but the beta functions do handle it, because they differentiate track().
    assert propagate_twiss_on_orbit(lat)[0].beta_x > 0.0

    # A thick sextupole at k2 = 0 is a drift and feeds nothing down, so it passes.
    zero = Lattice(
        [Sextupole(0.4, 0.0, name="sx"), *[e for i in range(N_CELLS) for e in _cell(f"_{i}")]], ref
    )
    assert linearised_lattice(zero).one_turn_matrix().shape == (DIM, DIM)


# --------------------------------------------------------------------------
# 4. Chromaticity on the real orbit
# --------------------------------------------------------------------------

CHROMA_K2L = 0.8


def _dispersive(ref: ReferenceParticle, k2l: float = 0.0, kick_x: float = 0.0) -> Lattice:
    """FODO-with-dipoles (nonzero ``D_x``) carrying one thin sextupole per cell.

    A sextupole feeds down to *chromaticity* only at dispersion, so the flat ring
    above cannot gate this half at all.
    """
    els: list = [Corrector(kick_x=kick_x, name="steerer")]
    for i in range(3):
        els += [
            Quadrupole(0.3, 1.2, name=f"qf_{i}"),
            Drift(0.5),
            ThinSextupole(k2l, name=f"sx_{i}"),
            Drift(0.5),
            Dipole(1.0, 0.12, name=f"b1_{i}"),
            Quadrupole(0.3, -1.2, name=f"qd_{i}"),
            Dipole(1.0, 0.12, name=f"b2_{i}"),
            Drift(0.5),
        ]
    return Lattice(els, ref)


def _tracked_feeddown_chromaticity(lattice: Lattice, h: float = 1e-5) -> tuple[float, float]:
    """``dQ/ddelta`` of the tracked map, linearised about the orbit at each ``delta``.

    The J1 route, now on a *steered* machine: Newton for the nonlinear closed orbit
    at fixed ``delta``, the finite-difference Jacobian there, and the accumulated
    phase advance of the resulting table. No Twiss integral is involved anywhere, so
    this is independent of everything :func:`chromaticity_on_orbit` does.

    Because accsim's linear element maps carry no ``delta`` dependence, this sees
    the sextupole feed-down term **only** — it is exactly blind to the natural
    chromaticity, which is why it gates a difference and not a total.
    """
    qx_p, qy_p = tunes_on_orbit(lattice, delta=+h)
    qx_m, qy_m = tunes_on_orbit(lattice, delta=-h)
    return (qx_p - qx_m) / (2.0 * h), (qy_p - qy_m) / (2.0 * h)


def test_tracked_feeddown_matches_the_on_orbit_split(ref: ReferenceParticle) -> None:
    r"""**The chromaticity gate.** Tracking and the integrals agree on the half both see.

    ``chromaticity_on_orbit - natural_chromaticity_on_orbit`` is the sextupole
    feed-down term ``+/-(1/4pi) oint beta k2 D_x ds`` evaluated on the *beaten*
    ``beta`` and ``D_x`` of the steered machine — an analytic integral over
    ``matrix()``. The tracked route reaches the same number through Newton and a
    finite-difference Jacobian, with no integral at all. J1 gated these two against
    each other on the design orbit; this extends it to a machine that is steered,
    where both the beat and the dispersion change.

    Measured 2026-08-10 across four steerer sizes: agreement to ``2.2e-8``
    absolute on values of order 2, i.e. relative ``1e-8`` — the ``delta``-step and
    Jacobian floor, flat in the orbit offset rather than growing with it.

    The tracked side has a sextupole-free baseline subtracted at each steerer size. The
    exact :class:`~accsim.elements.drift.Drift` map makes a drift a first-order chromatic
    element — see the blind-spot test below — and that contribution is in the tracked
    number but not in ``chromaticity_on_orbit - natural_chromaticity_on_orbit``, which is
    the sextupole term alone. It is the same ring in both arms, so differencing removes it
    exactly.
    """
    for kick in (8e-4, 4e-4, 2e-4, 1e-4):
        lat = _dispersive(ref, CHROMA_K2L, kick)
        baseline = _dispersive(ref, 0.0, kick)
        total = chromaticity_on_orbit(lat)
        natural = natural_chromaticity_on_orbit(lat)
        tracked = _tracked_feeddown_chromaticity(lat)
        drift_only = _tracked_feeddown_chromaticity(baseline)
        # The 1e-3 bound, against values of order 2, is the cross term the subtraction
        # cannot remove: the sextupole's fed-down dipole moves the orbit, which changes
        # the drift's own chromatic contribution slightly, so the two arms' drift terms
        # are not identical. It is 3e-4 relative and does not grow with the steerer.
        assert total[0] - natural[0] == pytest.approx(tracked[0] - drift_only[0], abs=1e-3)
        assert total[1] - natural[1] == pytest.approx(tracked[1] - drift_only[1], abs=1e-3)
        # Non-vacuous: the feed-down term itself is three orders above that bound.
        assert abs(total[0] - natural[0]) > 0.1


def test_the_tracked_route_is_now_blind_only_to_the_bends_share_of_the_chromaticity(
    ref: ReferenceParticle,
) -> None:
    r"""Why the gate above is a *difference* — and how much of that blind spot is left.

    It used to be total. Every accsim element map carried no ``delta`` dependence at
    all, so the tracked tunes of a sextupole-free machine did not move with momentum —
    exactly zero — while the machine plainly had a natural chromaticity that accsim
    supplied analytically. Two milestones have eaten into it:

    - **L1**, the exact :class:`~accsim.elements.drift.Drift`: ``x`` moves by
      ``L px / pz`` with ``pz ~ 1 + delta``, so a drift's effective length is
      ``L (1 - delta)`` — a first-order chromatic element with no magnet involved.
      That took this ring from ``3.7e-8`` to ``-0.1289``, 45% of its ``-0.2893``.
    - **L2**, the momentum-dependent :class:`~accsim.elements.quadrupole.Quadrupole`:
      a stiffer particle is focused by ``k1/(1+delta)``. That takes it to ``-0.1665``,
      **58%**.

    What is left is the **dipole**, whose map is still linear until L3 — its ``h^2``,
    dispersion and edge terms have no tracked counterpart at all. On a ring with *no*
    bend the tracked route now recovers the natural chromaticity in full, which is
    asserted in ``tests/analytic/test_exact_quadrupole.py`` against this same integral
    and is what makes the shortfall here attributable rather than merely observed.

    So ``chromaticity_on_orbit`` must still be built as a difference on a bendy
    machine. The fraction is asserted to be neither 0 nor 1, because both would be
    wrong for different reasons and a bound on the size alone would not say so.
    """
    lat = _dispersive(ref, k2l=0.0, kick_x=4e-4)
    tracked = _tracked_feeddown_chromaticity(lat)
    natural = natural_chromaticity(lat)
    assert natural[0] < -0.2 and natural[1] < -0.3  # the machine plainly has one...

    # ...and the tracked route now recovers most of it, but not the bends' part.
    assert tracked[0] == pytest.approx(-0.166549, rel=1e-4)
    assert tracked[1] == pytest.approx(-0.168901, rel=1e-4)
    for plane in (0, 1):
        share = tracked[plane] / natural[plane]
        assert 0.4 < share < 0.8, "neither blind nor complete — the bends' share is missing"

    # And the same sign as the natural chromaticity, so it adds to it rather than
    # fighting it: a drift is focusing-neutral but not momentum-neutral.
    assert tracked[0] * natural[0] > 0.0
    assert tracked[1] * natural[1] > 0.0


def test_the_natural_half_is_the_beta_weighted_sum_over_the_real_optics(
    ref: ReferenceParticle,
) -> None:
    r"""The half the difference gate above **cancels**, pinned on its own.

    :func:`test_tracked_feeddown_matches_the_on_orbit_split` gates
    ``chromaticity_on_orbit - natural_chromaticity_on_orbit``, so the natural term
    drops out of it entirely. It needs its own closed form, and on a
    **dispersion-free** ring it has a simple exact one: there is no sextupole
    feed-down chromaticity at all (``D_x = 0``), so the whole answer is the
    thin-lens sum

        xi_x = -(1/4pi) sum_e beta_x(e) k1l(e),   xi_y = +(1/4pi) sum_e beta_y(e) k1l(e)

    taken over the real quadrupoles **and** each sextupole's feed-down gradient
    ``k1l_eff = k2l x_co``, with ``beta`` read off :func:`propagate_twiss_on_orbit`
    — which is gated independently, against the derived beat closed form, and comes
    from finite-differencing ``track()`` rather than from the equivalent lattice
    ``natural_chromaticity_on_orbit`` is built on. Agreement 5e-13.

    **The gate has teeth because the two contributions fight each other.** Of the
    total shift ``-1.51e-3``, the sextupole's own direct term is ``-3.03e-3`` and
    the beta-beat acting on the pre-existing quadrupoles supplies ``+1.52e-3``.
    Dropping the direct term does not shrink the answer, it **flips its sign** —
    asserted below, so an implementation that beat ``beta`` correctly but forgot
    that an off-axis sextupole is itself a quadrupole cannot pass.
    """
    inv_4pi = 1.0 / (4.0 * math.pi)
    lat = _flat(ref, k2l=K2L, kick_x=KICK)
    table = propagate_twiss_on_orbit(lat)
    orbit = propagate_orbit_nonlinear(lat)

    sum_x = sum_y = 0.0
    direct_x = 0.0
    for i, elem in enumerate(lat.elements):
        if isinstance(elem, ThinQuadrupole):
            k1l = elem.k1l
        elif isinstance(elem, ThinSextupole):
            k1l = elem.k2l * float(orbit[i][X])  # I2's derived feed-down gradient
            direct_x = -inv_4pi * table[i].beta_x * k1l
        else:
            continue
        sum_x += -inv_4pi * table[i].beta_x * k1l
        sum_y += +inv_4pi * table[i].beta_y * k1l

    # The two sides now differ by 2.6e-8 on a sum of 0.66, where before L1 they agreed
    # to 1e-11. The cause is the derived-vs-differentiated gap:
    # `natural_chromaticity_on_orbit` reaches its optics through `linearised_lattice`,
    # which is built from accsim elements and so cannot carry the exact Drift map's
    # content, while `propagate_twiss_on_orbit` (the `table` above) differentiates
    # `track()` and does. It is therefore **second order in the orbit angle**, and that
    # is asserted rather than merely bounded — halving the steerer quarters it, where a
    # genuine error in the beta weighting would not move at all.
    natural = natural_chromaticity_on_orbit(lat)
    assert natural[0] == pytest.approx(sum_x, abs=1e-7)
    assert natural[1] == pytest.approx(sum_y, abs=1e-7)

    def route_gap(kick: float) -> float:
        lt = _flat(ref, k2l=K2L, kick_x=kick)
        tbl = propagate_twiss_on_orbit(lt)
        orb = propagate_orbit_nonlinear(lt)
        total = 0.0
        for j, e in enumerate(lt.elements):
            if isinstance(e, ThinQuadrupole):
                g = e.k1l
            elif isinstance(e, ThinSextupole):
                g = e.k2l * float(orb[j][X])
            else:
                continue
            total += -inv_4pi * tbl[j].beta_x * g
        return abs(natural_chromaticity_on_orbit(lt)[0] - total)

    assert route_gap(KICK) / route_gap(KICK / 2) == pytest.approx(4.0, rel=0.15)

    # On a dispersion-free ring the sextupole term is the whole of the difference,
    # so the two entry points coincide exactly rather than nearly.
    assert chromaticity_on_orbit(lat) == natural

    # ...and the direct term is not a correction to the beat, it dominates and
    # opposes it: without it the reported shift would have the wrong sign.
    shift = natural[0] - natural_chromaticity(lat)[0]
    assert direct_x < 0.0 < shift - direct_x  # opposite signs
    assert abs(direct_x) > 1.5 * abs(shift)  # ...and the larger of the two


def test_chromaticity_on_orbit_reduces_to_chromaticity_when_flat(ref: ReferenceParticle) -> None:
    """Zero steering, bit-for-bit — no tolerance, because no orbit enters.

    At ``kick = 0`` every feed-down gradient is identically zero, so the equivalent
    lattice differs from the original only by inserted zero-strength quadrupoles,
    whose matrices are the identity. The answer must therefore be the *same float*,
    not a close one.
    """
    lat = _dispersive(ref, CHROMA_K2L, kick_x=0.0)
    assert chromaticity_on_orbit(lat) == chromaticity(lat)
    assert natural_chromaticity_on_orbit(lat) == natural_chromaticity(lat)


def test_steering_moves_the_chromaticity_linearly(ref: ReferenceParticle) -> None:
    """The whole point of the milestone: the reported chromaticity now responds.

    ``chromaticity`` is a design-orbit quantity and returns the same number however
    the machine is steered (I2 pinned that, and it stays true). ``chromaticity_on_orbit``
    moves, and moves **linearly** in the orbit offset, because the feed-down gradient
    it is picking up is itself linear in the offset. The linearity is the check with
    teeth — a wrong-but-plausible implementation that responded quadratically, or
    saturated, would fail it while a bare "it changed" assertion would not.
    """
    design = chromaticity(_dispersive(ref, CHROMA_K2L, 0.0))
    shifts = []
    for kick in (8e-4, 4e-4, 2e-4, 1e-4):
        lat = _dispersive(ref, CHROMA_K2L, kick)
        assert chromaticity(lat) == design  # unmoved, by construction
        shifts.append(abs(chromaticity_on_orbit(lat)[0] - design[0]))
    # Measured 2026-08-10: 0.041 at the largest steerer, on a design value of 1.75
    # -- a 2.4 % error in the reported chromaticity of a machine steered by 1.4 mm.
    assert shifts[0] > 0.03
    for a, b in zip(shifts, shifts[1:], strict=False):
        assert 1.9 < a / b < 2.1  # ...linearly in the steerer


def test_the_off_momentum_orbit_is_the_dispersion_orbit(ref: ReferenceParticle) -> None:
    r"""``delta`` on the orbit solvers is seeded and solved consistently.

    The linear closed orbit at ``delta`` is ``(I - M4)^-1 (k4 + d delta)``: the
    corrector kicks *plus* the map's dispersive column. Without the second term the
    Newton seed would start a full dispersion orbit away from the answer. On a
    sextupole-free lattice the nonlinear solve must return exactly that, and the
    displacement per unit ``delta`` must be the matched dispersion
    :func:`~accsim.twiss.closed_twiss` reports.
    """
    lat = _dispersive(ref, k2l=0.0, kick_x=2e-4)
    disp = closed_twiss(lat).disp_x
    h = 1e-4
    base = closed_orbit(lat)
    plus = closed_orbit(lat, delta=+h)
    assert (plus[X] - base[X]) / h == pytest.approx(disp, rel=1e-12)

    # The nonlinear solve no longer lands on the linear orbit exactly, because the
    # drift's exact map is itself nonlinear: at delta != 0 the dispersion orbit carries a
    # real angle, and `x += L px / pz` differs from `x += L px` there. The gap is 3.1e-9
    # on an orbit of 3e-4 — a relative 1e-5, second order in the dispersion angle — and
    # the point of the test, that the seed already includes the dispersive column, is
    # untouched: without it the two would differ by a whole dispersion orbit, 1e-4.
    for d in (0.0, +h, -h):
        gap = np.abs(closed_orbit_nonlinear(lat, delta=d) - closed_orbit(lat, delta=d)).max()
        assert gap < 1e-7
    assert (
        np.abs(closed_orbit_nonlinear(lat, delta=+h) - closed_orbit(lat, delta=0.0)).max() > 1e-5
    )  # ...and the delta really does move the orbit, so the bound above is not vacuous

    # With the sextupole live the two part company, by the feed-down scale.
    live = _dispersive(ref, CHROMA_K2L, kick_x=2e-4)
    assert abs(closed_orbit_nonlinear(live, delta=h)[X] - closed_orbit(live, delta=h)[X]) > 1e-9


# --------------------------------------------------------------------------
# 5. The vertical fork: a normal sextupole off-axis in y is a skew quadrupole
# --------------------------------------------------------------------------


def test_vertical_steering_couples_the_on_orbit_optics(ref: ReferenceParticle) -> None:
    r"""The uncoupled path refuses, and the coupled path shows real coupling.

    I2 established that a *normal* sextupole at ``y_co != 0`` acts as a **skew**
    quadrupole. The design optics of this lattice is exactly uncoupled — a
    sextupole's ``matrix()`` is a drift — so ``coupled_twiss`` reports zero
    coupling, while the optics the beam actually sees does not. That contrast is
    impossible in the linear theory at any kick, which is what makes it a gate
    rather than a consistency check.

    ``closed_twiss_on_orbit`` must *raise* here rather than return a plausible
    2x2 answer: the Courant-Snyder reduction is only valid on a block-diagonal map.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK, kick_y=KICK)
    with pytest.raises(CoupledLatticeError):
        closed_twiss_on_orbit(lat)
    with pytest.raises(CoupledLatticeError):
        propagate_twiss_on_orbit(lat)

    assert coupled_twiss(lat).coupling_angle == pytest.approx(0.0, abs=1e-14)
    assert abs(coupled_twiss_on_orbit(lat).coupling_angle) > 1e-4

    # Horizontal steering alone leaves the map exactly block-diagonal, so the
    # uncoupled path is not merely tolerated there but exact.
    flat_x = _flat(ref, k2l=K2L, kick_x=KICK)
    M = linearised_one_turn_map(flat_x)
    assert np.abs(M[np.ix_([X, PX], [Y, PY])]).max() == 0.0
    assert np.abs(M[np.ix_([Y, PY], [X, PX])]).max() == 0.0


def test_the_vertical_feeddown_reaches_the_skew_gradient(ref: ReferenceParticle) -> None:
    """The coupling is I2's ``k1sl_eff = k2l y_co``, not merely nonzero.

    Compares the on-orbit coupled optics against the equivalent machine built from
    the derived skew gradient, so the gate pins the coefficient rather than the
    existence of an effect.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK, kick_y=KICK)
    y_co = propagate_orbit_nonlinear(lat)[SX_INDEX][Y]
    assert abs(y_co) > 1e-5  # the vertical orbit is genuinely there
    # 1e-5 rather than 1e-6: the exact Drift map's transverse block picks up an
    # O(orbit angle^2) correction the derived equivalent machine cannot carry, worth a
    # relative 1e-6 in the coupling angle here. The coefficient being pinned is the skew
    # gradient, which is first order in the vertical orbit and four orders larger.
    equivalent = linearised_lattice(lat)
    assert coupled_twiss_on_orbit(lat).coupling_angle == pytest.approx(
        coupled_twiss(equivalent).coupling_angle, rel=1e-5
    )

    # ...and that residual is the drift's, not a wrong skew gradient: it is second order
    # in the orbit while the coupling angle itself is first, so halving both steerers
    # halves the angle and quarters the gap — a net factor of two in the *relative*
    # discrepancy. A mis-derived k1sl_eff would hold its relative size instead.
    def relative_gap(kick: float) -> float:
        lt = _flat(ref, k2l=K2L, kick_x=kick, kick_y=kick)
        a = coupled_twiss_on_orbit(lt).coupling_angle
        b = coupled_twiss(linearised_lattice(lt)).coupling_angle
        return abs(a - b) / abs(b)

    assert relative_gap(KICK) / relative_gap(KICK / 2) == pytest.approx(2.0, rel=0.2)


def test_chromaticity_on_orbit_refuses_a_coupled_machine(ref: ReferenceParticle) -> None:
    """Chromaticity stays a two-plane quantity, and says so.

    With ``y_co != 0`` the equivalent machine carries a skew quadrupole, and the
    2x2 Courant-Snyder chromaticity integrals are not valid there. G2's
    Edwards-Teng optics is the machinery a coupled chromaticity would be built on;
    it is not built here, and the failure is explicit rather than a wrong number.
    """
    els: list = [Corrector(kick_x=2e-4, kick_y=2e-4, name="steerer")]
    for i in range(3):
        els += [
            Quadrupole(0.3, 1.2),
            Drift(0.5),
            ThinSextupole(CHROMA_K2L, name=f"sx_{i}"),
            Drift(0.5),
            Dipole(1.0, 0.12),
            Quadrupole(0.3, -1.2),
            Dipole(1.0, 0.12),
            Drift(0.5),
        ]
    with pytest.raises(CoupledLatticeError):
        chromaticity_on_orbit(Lattice(els, ref))


# --------------------------------------------------------------------------
# 6. Argument handling
# --------------------------------------------------------------------------


def test_maps_length_is_checked(ref: ReferenceParticle) -> None:
    """A short ``maps`` list is an error, not a silently truncated table."""
    lat = _flat(ref)
    tw0 = closed_twiss(lat)
    with pytest.raises(ValueError, match="maps"):
        propagate_twiss(lat, tw0, maps=[e.matrix(ref) for e in lat.elements][:-1])


def test_tunes_on_orbit_are_full_tunes_not_fractional(ref: ReferenceParticle) -> None:
    r"""The accumulated phase, so the ``delta`` difference has no wrap hazard.

    ``chromaticity_on_orbit``'s independent gate central-differences
    :func:`tunes_on_orbit`. Reading the tune off the one-turn map with ``acos``
    would give only the fractional part, and the difference would be wrong by an
    integer whenever the two ``delta`` points straddled a half integer. Taking the
    accumulated phase advance instead removes the hazard entirely rather than
    guarding it, and this asserts the integer part really is carried.

    The ring is stretched to 24 cells here on purpose: the 6-cell ring the beat
    gates use runs at ``Q = 0.69``, where the integer part is zero and the two
    conventions are indistinguishable, so the assertion would pass vacuously.
    """
    lat = _flat(ref, k2l=K2L, kick_x=KICK, n_cells=24)
    qx, qy = tunes_on_orbit(lat)
    assert qx > 2.0 and qy > 2.0  # 24 FODO cells: several integers of phase
    M = linearised_one_turn_map(lat)
    fractional = match_periodic(M)
    assert fractional.beta_x == pytest.approx(closed_twiss_on_orbit(lat).beta_x, rel=1e-12)
    # The fractional parts still agree with the one-turn map, integer aside.
    half_trace = 0.5 * (M[X, X] + M[PX, PX])
    frac = math.acos(half_trace) / (2.0 * math.pi)
    frac = frac if M[X, PX] >= 0.0 else 1.0 - frac
    assert qx - math.floor(qx) == pytest.approx(frac, abs=1e-9)


def test_delta_is_carried_into_the_linearised_maps(ref: ReferenceParticle) -> None:
    """``delta`` reaches every layer, and moves the answer where it should.

    A sextupole at dispersion sees ``x_co + D_x delta``, so its feed-down gradient
    — and hence the on-orbit beta — depends on ``delta``. If the argument were
    dropped anywhere between :func:`closed_orbit_nonlinear` and
    :func:`propagate_twiss_on_orbit`, the two tables below would be identical.
    """
    lat = _dispersive(ref, CHROMA_K2L, kick_x=2e-4)
    flat = propagate_twiss_on_orbit(lat, delta=0.0)
    off = propagate_twiss_on_orbit(lat, delta=1e-3)
    assert any(abs(a.beta_x / b.beta_x - 1.0) > 1e-6 for a, b in zip(flat, off, strict=True))
    # ...and a multipole-free lattice is *almost* unmoved by delta. "Unmoved" was the
    # claim while accsim's maps carried no delta dependence at all; both L1 and L2 have
    # since put one in — a drift's effective length is L/pz ~ L(1 - delta), and a thick
    # quadrupole focuses an off-momentum particle by k1/(1 + delta) — so beta really
    # moves. That is chromatic beta-beat, which a real machine has and this package
    # previously could not produce at all.
    #
    # Measured 2.4e-4 relative (1.8e-4 with the drift's share alone, before the quadrupole
    # gained its own), still two orders below the 1e-2 sextupole signal above, so the
    # discrimination this test rests on survives. The bound is stated as that ratio
    # rather than as a tolerance, and the sextupole-free beat is asserted to be nonzero
    # so the new effect is documented rather than hidden inside a widened number.
    linear = _dispersive(ref, k2l=0.0, kick_x=2e-4)
    beat = [
        abs(a.beta_x / b.beta_x - 1.0)
        for a, b in zip(
            propagate_twiss_on_orbit(linear, delta=0.0),
            propagate_twiss_on_orbit(linear, delta=1e-3),
            strict=True,
        )
    ]
    assert max(beat) == pytest.approx(2.4135e-4, rel=1e-2)
    sextupole_beat = max(abs(a.beta_x / b.beta_x - 1.0) for a, b in zip(flat, off, strict=True))
    assert max(beat) < 0.05 * sextupole_beat


def test_orbit0_and_delta_are_not_both_free(ref: ReferenceParticle) -> None:
    """An explicit ``orbit0`` is honoured; ``delta`` still sets the momentum it sits at.

    ``linearised_element_maps`` accepts a supplied orbit so a caller can linearise
    about a trajectory it already has (``correct_orbit`` does this). ``delta`` is
    then still needed, because it is not recoverable from a 4D transverse vector.
    """
    lat = _dispersive(ref, CHROMA_K2L, kick_x=2e-4)
    h = 1e-3
    orbit = closed_orbit_nonlinear(lat, delta=h)
    supplied = linearised_element_maps(lat, orbit, delta=h)
    solved = linearised_element_maps(lat, delta=h)
    for a, b in zip(supplied, solved, strict=True):
        assert np.abs(a - b).max() < 1e-12

    state = np.zeros(DIM)
    state[[X, PX, Y, PY]] = orbit
    state[DELTA] = h
    for elem in lat.elements:
        state = elem.track(state, ref)
    assert state[[X, PX, Y, PY]] == pytest.approx(orbit, abs=1e-13)  # it is closed
