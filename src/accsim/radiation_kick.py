r"""The classical ("mean") synchrotron-radiation kick a tracked particle takes (B2).

Axis B's :mod:`accsim.radiation` is entirely a **design-route** module: the radiation
integrals ride the Twiss functions, and the damping times and equilibrium emittance are
closed forms evaluated on them. Nothing there touches a tracked particle. This module is
the other half — the map that makes a tracked particle actually lose the energy it
radiates, so that damping is something the simulation *exhibits* rather than asserts.

**The physics, in one paragraph.** In an element the particle follows a curved path of
local radius ``rho_p`` and radiates

    ``U = (C_gamma / 2 pi) * E^4 * kappa^2 * l_path``      [eV],  ``kappa = 1/rho_p``,

the same ``C_gamma E^4 I2 / 2 pi`` the radiation integrals integrate, evaluated on the
particle's own trajectory instead of the design orbit. The photons leave along the
direction of motion, so the momentum *vector* shrinks in magnitude with its direction
unchanged: every Cartesian component — ``px``, ``py`` **and** ``1 + delta`` — is scaled
by one common factor. That single fact is what produces transverse damping, and it is
the part a plausible implementation drops (see *The wrong map* below).

**The factor.** The particle stays on shell, so losing energy ``U`` fixes the momentum:

    ``f = P_new/P = sqrt((E-U)^2 - m^2) / sqrt(E^2 - m^2)
        = sqrt(1 - U(2E - U)/(E^2 - m^2))``

written in the second (rationalised) form because the first cancels two numbers of size
``E`` — the numerical trap L1 recorded for the drift and L3 for the bend. To first order
``f = 1 - U/(beta^2 E)``, and exactly ``1 - U/E`` in the massless limit.

**The wrong map.** Reducing ``delta`` alone and leaving ``px, py`` is the natural-looking
mistake. It gets the *longitudinal* damping exactly right, so half the gates cannot see
it; inside the element it **anti**-damps the angle ``x' = px/pz`` at first order; and per
turn it produces **exactly zero** transverse damping, because ``py`` is never touched and
the RF restores ``delta``. It is available as ``model="mean_delta_only"`` for the gate
that asserts precisely this, and is not a physical model.

**The graininess (B3).** The paragraphs above are the *mean* — the energy an electron
would lose if radiation were a continuous drag. It is not: light comes in photons, and a
particle crossing one magnet emits a countable number of them at random. In a path
``l_path`` of curvature ``kappa`` the emission is a Poisson process of

    ``n_gamma = (5 / (2 sqrt3)) alpha gamma |kappa l_path|``    photons,

each drawn from the synchrotron-radiation spectrum, whose moments in units of the
critical energy ``u_c = (3/2) hbar c gamma^3 kappa`` are ``<u> = 8/(15 sqrt3) u_c`` and
``<u^2> = 11/27 u_c^2`` (both derived from ``int_x^inf K_{5/3}`` in the analytic suite,
not quoted). The mean of that sum is exactly the ``U`` above — that is the consistency
check between the two constant systems — and its **variance** is

    ``sigma_U^2 = n_gamma <u^2> = 2 C_q E gamma^2 kappa U``,

written in terms of the package's own :func:`accsim.radiation.quantum_constant_cq` so
the design route and the tracked route cannot carry two copies of the constant that
sets the size of the whole effect. ``model="quantum"`` draws the loss from a Gaussian
of that mean and that variance; everything downstream — the on-shell factor, the one
common scaling of ``(px, py, 1 + delta)`` — is identical to ``"mean"``.

*Why a Gaussian is enough, and where it is not.* The equilibrium the beam settles into
depends on the emission process **only through its first two moments**, because it is
the fixed point of "diffusion in, damping out" and the diffusion coefficient *is* the
variance. So a Gaussian with the right mean and variance reproduces
:func:`accsim.radiation.equilibrium_emittance` and
:func:`accsim.radiation.equilibrium_energy_spread` exactly, and the reference arm
checks that against xtrack, which emits genuine photons off the true spectrum. What a
Gaussian gets wrong is the **tail** — the single hard photon that throws a particle out
of the bucket — which is what Stage 4's ``quantum_lifetime`` is about.

**The photons themselves (B5).** ``model="photons"`` is the thing the Gaussian stands
in for: a Poisson count of photons from :func:`photon_rate`, each drawn from the real
synchrotron spectrum (:mod:`accsim.photon_spectrum`) in units of
:func:`critical_photon_energy`. Note that it **replaces** the classical loss rather
than perturbing it — the Gaussian adds a zero-mean draw to ``u``, and the photon sum
*is* ``u``; adding it by analogy would double the mean.

*What changes, and what must not.* Every aggregate stays put, exactly: the sum's mean
is ``n_gamma <u> = U`` and its variance is ``n_gamma <u^2> =``
:func:`photon_energy_variance`, both identities rather than approximations, so B3's
whole equilibrium battery re-runs under the new model and lands on the same closed
forms. What changes is the **shape**. The Gaussian hands a particle *energy* in about
1% of draws (see below); the photon sum never can. The Gaussian is symmetric; the
photon sum is skewed, by ``<u^3> / (sqrt(n_gamma) <u^2>^(3/2))``, which is the
compound-Poisson identity B3's reference arm already used to count xtrack's photons
from the outside — and, run on ``delta``, it lands at ``-0.92`` against xtrack's
measured ``-0.91``.

*What it costs.* Roughly ``n_gamma`` uniforms per particle per magnet — of order 20 —
so a long tracking run is a few times dearer than ``"quantum"``. That is the price of
the tail, and the tail is the only thing being bought: at any energy and bending radius
this package can build, the hard photon that would empty the bucket is suppressed by
``e^-640`` and the two models give the *same lifetime*.

*The model can draw an energy gain.* The Gaussian is unclamped, deliberately. With
``n_gamma`` of order 20 per magnet the relative fluctuation is ``sqrt(4.30 / n_gamma)``
~ 0.42, so ``u < 0`` sits at 2.4 sigma — about 1% of draws, not a tail event. Clamping
at zero would bias the mean *and* the variance by a percent, which is the size of the
very quantities this model exists to get right; an unclamped Gaussian keeps both exact.
The on-shell factor handles it without special-casing (``f > 1``, no branch), and it is
the price of not resolving photons. It is asserted, not hidden.

**Scope and costs.**

- *Not symplectic.* Radiation is dissipative — this is the first map in the package that
  must **fail** :func:`accsim.symplectic.is_symplectic_map`, and the analytic suite
  asserts that rejection rather than working around it.
- *A tracking mode, never the design route.* ``matrix()`` and ``kick()`` are untouched,
  so every optics quantity in the package is bit-for-bit unchanged and the invariant
  that bounds axis L — ``matrix()`` is the exact origin Jacobian of ``track()`` — still
  holds for the map as such. With radiation on it does not, because the reference
  particle radiates too; that is why radiation is opt-in per tracking call.
- *One kick per element.* The loss is evaluated at the element's **entry** energy, so it
  over-counts by ``U_elem/E`` relative to the continuous answer; slicing the lattice
  converges it as ``(N-1)/N``, which the analytic suite asserts as that law rather than
  as a tolerance. xtrack does the same thing with its own sub-stepping — its default
  ``integrator='adaptive'`` resolves to eight uniform steps for a plain bend, and
  ``integrator='uniform', num_multipole_kicks=1`` reproduces the single lumped kick.
- *No vertical excitation floor.* The photons leave along the direction of motion, so
  ``"quantum"`` adds no transverse recoil spread — the real one has an opening angle
  ``~1/gamma``. On a flat lattice the vertical emittance therefore damps to **exactly**
  zero rather than to the opening-angle limit ``(13/55) C_q / J_y * <beta_y/|rho|^3> /
  I2``. That is the same flat-lattice boundary :func:`accsim.radiation.equilibrium_emittance`
  already records from the design side, now visible from inside the tracking.
- *Thin elements do not radiate.* A zero-length element has no path to radiate over, so
  correctors, thin quadrupoles, thin multipoles and the RF cavity contribute nothing.
  That is a scope statement, not an approximation: a real short magnet radiates, and
  modelling it means giving it a length.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .coords import DELTA, PX, PY, ZETA, X, Y
from .reference import ReferenceParticle

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from .elements.element import Element

__all__ = [
    "HBAR_C_EV_M",
    "RADIATION_MODELS",
    "STOCHASTIC_MODELS",
    "critical_photon_energy",
    "fine_structure_constant",
    "photon_energy_variance",
    "photon_rate",
    "quantum_constant_cq",
    "radiation_constant_cgamma",
    "radiation_kick",
]

#: The radiation models ``track`` accepts. ``"mean_delta_only"`` is the deliberately
#: wrong map the discriminating gate needs; it is not a physical choice.
RADIATION_MODELS: tuple[str, ...] = ("off", "mean", "mean_delta_only", "quantum", "photons")

#: The subset that draws random numbers, and so requires an explicit
#: :class:`numpy.random.Generator`. The package never seeds a global one.
STOCHASTIC_MODELS: frozenset[str] = frozenset({"quantum", "photons"})


#: CODATA ``hbar*c = 197.3269804 MeV*fm``. The one physical constant radiation adds
#: beyond the reference particle's own. (xtrack hardcodes ``1.973269804593025e-7``,
#: which agrees to 4.7e-11 — so unlike its pre-2019 electron charge it is *not* a
#: named owner of any cross-check residual on this axis.)
HBAR_C_EV_M: float = 1.9732698045e-7


def quantum_constant_cq(ref: ReferenceParticle) -> float:
    r"""``C_q = 55/(32 sqrt3) * hbar c / (m c^2)`` [m] for the reference species.

    The quantum-excitation constant; ``55/(32 sqrt3)`` is the ratio of moments of the
    synchrotron-radiation spectrum (Sands). For the electron, ``3.832e-13 m``.
    :func:`accsim.radiation.quantum_constant_cq` is this function — the design-route
    equilibrium and the tracked route's :func:`photon_energy_variance` must not carry
    two copies of the constant that sets the size of the whole effect.
    """
    return 55.0 / (32.0 * math.sqrt(3.0)) * HBAR_C_EV_M / ref.mass_eV


def radiation_constant_cgamma(ref: ReferenceParticle) -> float:
    r"""``C_gamma = 4 pi r0 / (3 (m c^2)^3)`` [m/eV^3] for the reference species.

    Computed from the particle's own classical radius and rest energy, so it is correct
    for any species (``r0 ∝ 1/m`` ⇒ ``C_gamma ∝ 1/m^3``). For the electron this is the
    familiar ``8.846e-5 m/GeV^3``. :func:`accsim.radiation.radiation_constant_cgamma`
    is this function — the design route and the tracked route must not carry two copies
    of the constant that sets the size of the whole effect.
    """
    return 4.0 * math.pi * ref.classical_radius_m / (3.0 * ref.mass_eV**3)


def _perpendicular_field(
    bx: np.ndarray | float,
    by: np.ndarray | float,
    px: np.ndarray | float,
    py: np.ndarray | float,
    delta: np.ndarray | float,
) -> np.ndarray:
    r"""``|B_perp| / (B rho)_0`` — the part of the field the particle actually feels.

    Only the field component **perpendicular** to the velocity bends the trajectory and
    so radiates. With the direction of motion ``i = (px, py, pz)/(1+delta)`` and a purely
    transverse field ``(bx, by, 0)``, this is ``|b - (b.i) i|``. On the design orbit it
    is just ``|b|``; the projection matters at the ``(x')^2`` level.
    """
    ix = np.asarray(px) / (1.0 + np.asarray(delta))
    iy = np.asarray(py) / (1.0 + np.asarray(delta))
    iz = np.sqrt(np.maximum(1.0 - ix * ix - iy * iy, 0.0))
    b_par = bx * ix + by * iy  # the longitudinal field component is zero
    ex = bx - b_par * ix
    ey = by - b_par * iy
    ez = -b_par * iz
    return np.sqrt(ex * ex + ey * ey + ez * ez)


def photon_energy_variance(
    u: np.ndarray | float,
    energy: np.ndarray | float,
    kappa: np.ndarray | float,
    ref: ReferenceParticle,
) -> np.ndarray | float:
    r"""``sigma_U^2 = 2 C_q E gamma^2 kappa U`` [eV^2] — the graininess of a mean loss ``U``.

    The variance of the energy actually radiated in one traversal, given the mean ``u``
    it would lose classically, the particle's own total ``energy`` [eV] and the
    magnitude ``kappa`` [1/m] of its trajectory's curvature. It is
    ``n_gamma * <u^2>`` for the Poisson emission of photons off the synchrotron
    spectrum — see the module docstring — collapsed onto
    :func:`accsim.radiation.quantum_constant_cq`, which is the *same* ``C_q`` the
    design-route :func:`accsim.radiation.equilibrium_energy_spread` divides by its
    damping. Two routes, one constant.

    Note the scaling: ``U ∝ kappa^2 l`` but ``sigma_U^2 ∝ kappa^3 l``. Slicing an
    element in ``N`` therefore converges the *mean* as ``(N-1)/N`` (each slice radiates
    at a slightly lower energy) while leaving the *variance* invariant at leading order,
    because a sum of ``N`` independent variances of ``1/N`` the size is the same total.
    """
    gamma = np.asarray(energy) / ref.mass_eV
    return 2.0 * quantum_constant_cq(ref) * energy * gamma * gamma * kappa * u


def fine_structure_constant(ref: ReferenceParticle) -> float:
    r"""``alpha = r_0 m c^2 / (hbar c)`` — the bridge between the two constant systems.

    The photon picture speaks ``alpha`` and ``hbar c``; the radiation integrals speak
    ``C_gamma`` and ``r_0``. They meet here and nowhere else, which is why this is
    computed from the reference particle's *own* classical radius and rest energy rather
    than typed in as ``1/137``: any species gives the same number, and if it did not,
    the two routes would be describing different electrodynamics.
    """
    return ref.classical_radius_m * ref.mass_eV / HBAR_C_EV_M


def critical_photon_energy(
    energy: np.ndarray | float, kappa: np.ndarray | float, ref: ReferenceParticle
) -> np.ndarray | float:
    r"""``u_c = (3/2) hbar c gamma^3 kappa`` [eV] — the scale of the photon spectrum.

    The energy that divides the synchrotron spectrum in half *by power*: half the
    radiated energy comes out above it and half below. Photon energies are drawn in
    units of it (:mod:`accsim.photon_spectrum` works entirely in ``x = u/u_c``), so it,
    together with :func:`photon_rate`, is the whole of what couples a dimensionless
    spectrum to a particular ring.

    On this package's rings it is small: 5 GeV in a 10 m bend gives ``u_c/E = 5.5e-6``,
    which is the number that makes the single-hard-photon loss channel unreachable.
    """
    gamma = np.asarray(energy) / ref.mass_eV
    return 1.5 * HBAR_C_EV_M * gamma**3 * kappa


def photon_rate(
    energy: np.ndarray | float,
    kappa: np.ndarray | float,
    path_length: np.ndarray | float,
    ref: ReferenceParticle,
) -> np.ndarray | float:
    r"""``n_gamma = (5 / 2 sqrt3) alpha gamma |kappa| l`` — photons emitted in one traversal.

    The Poisson mean of the number of photons radiated over a path ``path_length`` [m]
    of curvature ``kappa`` [1/m]. ``(5 / 2 sqrt3) alpha gamma`` per radian of bend is the
    textbook rate; B3's suite already recovers it from the *outside*, by inverting the
    relative fluctuation of a Gaussian that never counted anything.

    The two dimensional numbers meet the classical loss exactly:
    ``n_gamma <u> = (C_gamma / 2 pi) E^4 kappa^2 l``, both sides being
    ``(2/3) alpha hbar c gamma^4 kappa^2 l`` once ``alpha hbar c = r_0 m c^2``. That is
    the bridge B3 gated symbolically; B5's suite gates it again through these two
    functions, where a ``kappa`` or an ``l_path`` computed differently from the mean
    route would show up and a symbolic identity could not.
    """
    gamma = np.asarray(energy) / ref.mass_eV
    return 2.5 / math.sqrt(3.0) * fine_structure_constant(ref) * gamma * np.abs(kappa) * path_length


def radiation_kick(
    element: Element,
    before: np.ndarray,
    after: np.ndarray,
    ref: ReferenceParticle,
    model: str = "mean",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Apply the synchrotron-radiation loss of ``element`` to a state it has just mapped.

    ``before`` / ``after`` are the states entering and leaving the element **in its own
    body frame** (so a misaligned magnet radiates according to where it really is), each
    a ``(6,)`` vector or a ``(6, n)`` bunch. Returns a new array; the input is not
    modified. ``model="off"`` returns ``after`` unchanged.

    The field is sampled at the **mid-point** of the traversal — the mean of the entry
    and exit positions, and the mean of the entry and exit angles for the perpendicular
    projection — which is the one-step midpoint rule for
    ``U = (C_gamma/2 pi) E^4 ∮ kappa^2 ds`` and the convention xtrack uses, so the two
    are directly comparable per element. The path length is the element's own
    ``l_path = rvv (L - Delta zeta)``, so a longer trajectory radiates more without any
    of that being put in by hand.

    ``model="quantum"`` adds the graininess: the loss is drawn from a Gaussian of that
    same mean and of variance :func:`photon_energy_variance`. ``model="photons"`` draws
    the photons themselves — a Poisson count off :func:`photon_rate`, each energy off
    the synchrotron spectrum — which has that same mean and that same variance and, in
    addition, the right tail. Both need an explicit ``rng``: the package never seeds a
    global generator, because an unseeded stochastic track is not reproducible, so
    asking for one **raises**.
    """
    if model == "off":
        return after
    if model not in RADIATION_MODELS:
        raise ValueError(f"radiation model must be one of {RADIATION_MODELS}, got {model!r}")
    if model in STOCHASTIC_MODELS and rng is None:
        raise ValueError(
            f"radiation={model!r} draws random numbers and needs an explicit rng "
            "(numpy.random.Generator): an unseeded stochastic track is not reproducible."
        )
    length = element.length
    if length == 0.0:
        return after  # a thin element has no path to radiate over

    out = np.array(after, dtype=float, copy=True)
    mid_x = 0.5 * (before[X] + after[X])
    mid_y = 0.5 * (before[Y] + after[Y])
    mid_px = 0.5 * (before[PX] + after[PX])
    mid_py = 0.5 * (before[PY] + after[PY])
    bx, by = element.normalized_field(mid_x, mid_y)
    if np.all(bx == 0.0) and np.all(by == 0.0):
        return out  # no field, no radiation (a drift, or a bend switched off)

    delta = after[DELTA]
    kappa = _perpendicular_field(bx, by, mid_px, mid_py, delta) / (1.0 + delta)

    m = ref.mass_eV
    p = ref.momentum_eV * (1.0 + delta)
    energy = np.sqrt(p * p + m * m)
    rvv = (p / energy) / ref.beta0  # beta/beta0
    l_path = rvv * (length - (after[ZETA] - before[ZETA]))

    u = radiation_constant_cgamma(ref) / (2.0 * math.pi) * energy**4 * kappa * kappa * l_path
    if model == "quantum":
        # Light comes in photons: the loss is a compound-Poisson sum, and this is the
        # Gaussian with its mean and its variance. Deliberately NOT clamped at zero --
        # see the module docstring's *The model can draw an energy gain*.
        assert rng is not None  # guaranteed above; narrows the type
        u = u + rng.normal(0.0, np.sqrt(photon_energy_variance(u, energy, kappa, ref)))
    elif model == "photons":
        # ...and this is the sum itself: a Poisson count of photons, each drawn from the
        # synchrotron spectrum. It REPLACES the classical loss rather than perturbing it
        # -- adding it, by analogy with the Gaussian above, would double the mean.
        from .photon_spectrum import sample_photon_sum

        assert rng is not None  # guaranteed above; narrows the type
        u_c = critical_photon_energy(energy, kappa, ref)
        rate = photon_rate(energy, kappa, l_path, ref)
        shape = np.shape(u)
        drawn = sample_photon_sum(np.broadcast_to(rate, shape) if shape else rate, rng)
        u = u_c * (drawn.reshape(shape) if shape else drawn[0])
    # On shell: f = P_new/P with E_new = E - U, rationalised so no two numbers of size E
    # are subtracted (the trap L1 recorded for the drift, L3 for the bend).
    shrink = u * (2.0 * energy - u) / (energy * energy - m * m)
    f = np.sqrt(np.maximum(1.0 - shrink, 0.0))

    # ...and the SAME trap one level down. f is 1 - 3e-7 here, so `f*(1+delta) - 1`
    # subtracts two numbers of size 1 to produce one of size 1e-7 and keeps six digits
    # of it. Written as `delta + (f-1)(1+delta)` with `f - 1 = -shrink/(1+f)` -- the
    # rationalised form again -- the increment carries full relative precision, which is
    # what lets B5 gate the photon route against the mean route at 1e-13 instead of
    # 1e-10. The maximum reproduces `f - 1 = -1` when a loss takes the whole momentum.
    out[DELTA] = delta + np.maximum(-shrink / (1.0 + f), -1.0) * (1.0 + delta)
    if model != "mean_delta_only":
        # The photons leave along the direction of motion, so the transverse momenta
        # scale by the SAME factor -- this line, and only this line, is what damps the
        # betatron amplitude. See the module docstring's *The wrong map*.
        out[PX] = after[PX] * f
        out[PY] = after[PY] * f
    return out


#: B2's name for the same seam, kept so the gates written before the graininess existed
#: still read as what they mean. The two are the same function.
mean_radiation_kick = radiation_kick
