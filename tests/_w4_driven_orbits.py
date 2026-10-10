"""Shared fixture and exact truth for W4 (the amplitude-dependent driven response).

The split-tune FODO of ``tests/analytic/test_ac_dipole_detuned.py`` and the period-``q``
orbits of its exact driven map, found by Newton on the ``q``-turn map. Shared so that the
reference cross-checks (xtrack, MAD-X) are handed exactly the orbits the analytic gates
test against.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from accsim import (
    PROTON_MASS_EV,
    PX,
    PY,
    ACDipole,
    Drift,
    Lattice,
    ReferenceParticle,
    ThinOctupole,
    ThinQuadrupole,
    X,
    Y,
)

REF = ReferenceParticle.from_gamma(PROTON_MASS_EV, 10.0)
LD, NCELL = 5.0, 6
KF, KD = 0.25, -0.22  # split tunes: Qx = 1.3441, Qy = 1.0409
IDX = 2  # the dipole sits after the first drift, where alpha != 0
LAG = 0.13
NU = 0.325  # = 13/40: every steady state is a period-40 orbit
HIGH_BETA = 5  # an octupole here sees beta_x = 15.7 against the dipole's 3.80


def _ring(
    acd: ACDipole | None = None,
    k3l: float = 0.0,
    oct_at: int | None = None,
    kf: float = KF,
    kd: float = KD,
    first: bool = False,
) -> Lattice:
    """The split-tune FODO; the octupole at ``oct_at`` (default: right after the dipole).
    ``first`` rotates the same ring to start at the dipole."""
    els: list = []
    for _ in range(NCELL):
        els += [ThinQuadrupole(kf), Drift(LD), ThinQuadrupole(kd), Drift(LD)]
    if k3l:
        els.insert(IDX if oct_at is None else oct_at, ThinOctupole(k3l))
    if acd is not None:
        els.insert(IDX, acd)
    if first:
        els = els[IDX:] + els[:IDX]
    return Lattice(els, REF)


def _acd(kick: float, nu: float = NU, plane: str = "x", ramp=None) -> ACDipole:
    return ACDipole(kick=kick, tune=nu, lag=LAG, plane=plane, ramp=ramp)


# --- the exact truth: period-q orbits of the driven map -----------------------------------
def _q_turns(lat: Lattice, states: np.ndarray, q: int) -> np.ndarray:
    """``(6, n)`` states at the lattice start, carried ``q`` turns by ``track()`` + drive."""
    s = states.copy()
    for turn in range(q):
        for e in lat.elements:
            s = e.track(s, lat.ref)
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(turn, lat.ref)[:, None]
    return s


def _plane_ix(plane: str) -> tuple[int, int]:
    return (X, PX) if plane == "x" else (Y, PY)


def _periodic_orbit(lat: Lattice, z: np.ndarray, q: int, plane: str = "x", it: int = 60):
    """Newton on the ``q``-turn map in ``(u, p_u)``: ``(z, |F|, Jacobian)``, or
    ``(None, inf, None)`` if it does not converge to a finite orbit."""
    iu, ip = _plane_ix(plane)
    z = np.array(z, dtype=float)

    def residual(zz: np.ndarray) -> np.ndarray:
        s = np.zeros((6, 1))
        s[iu, 0], s[ip, 0] = zz
        out = _q_turns(lat, s, q)
        return np.array([out[iu, 0], out[ip, 0]]) - zz

    J = None
    for _ in range(it):
        h = 1e-9 * max(1e-3, float(np.max(np.abs(z))))
        cols = np.zeros((6, 5))
        for k, (a, b) in enumerate(((0, 0), (h, 0), (-h, 0), (0, h), (0, -h))):
            cols[iu, k], cols[ip, k] = z[0] + a, z[1] + b
        out = _q_turns(lat, cols, q)[[iu, ip]]
        if not np.all(np.isfinite(out)):
            return None, math.inf, None
        F = out[:, 0] - z
        J = np.column_stack(((out[:, 1] - out[:, 2]) / (2 * h), (out[:, 3] - out[:, 4]) / (2 * h)))
        t, f0 = 1.0, float(np.max(np.abs(F)))
        if f0 < 1e-15 * max(1e-6, float(np.max(np.abs(z)))):
            break  # at round-off: J is the Jacobian at the orbit
        step = np.linalg.solve(J - np.eye(2), -F)
        while t > 1e-4:
            trial = residual(z + t * step)
            if np.all(np.isfinite(trial)) and np.max(np.abs(trial)) <= f0:
                break
            t /= 2
        z = z + t * step
        if np.max(np.abs(t * step)) < 1e-15 * max(1e-6, float(np.max(np.abs(z)))):
            break
    F = residual(z)
    if not np.all(np.isfinite(F)) or np.max(np.abs(F)) > 1e-14:
        return None, math.inf, None
    return z, float(np.max(np.abs(F))), J


def _dipole_amplitude(lat: Lattice, z: np.ndarray, q: int, nu: float, plane: str = "x"):
    """``(in-phase, quadrature)`` amplitude of ``u`` at the dipole over one period."""
    iu, ip = _plane_ix(plane)
    s = np.zeros((6, 1))
    s[iu, 0], s[ip, 0] = z
    us = []
    for turn in range(q):
        for e in lat.elements:
            if isinstance(e, ACDipole):
                us.append(s[iu, 0])
            s = e.track(s, lat.ref)
            if isinstance(e, ACDipole):
                s = s + e.drive_kick(turn, lat.ref)[:, None]
    ph = 2 * np.pi * (nu * np.arange(q) + LAG)
    us = np.array(us)
    return 2 / q * float(np.sum(us * np.sin(ph))), 2 / q * float(np.sum(us * np.cos(ph)))


def _seed(lat: Lattice, u: float, kick: float, nu: float, plane: str = "x") -> np.ndarray:
    """The steady state at amplitude ``u`` of the ring with W2's gradient ``kick / u``: the
    free oscillation of that substituted ring, phased onto the drive."""
    iu, ip = _plane_ix(plane)
    ms = [e.matrix(lat.ref) for e in lat.elements]
    idx = next(i for i, e in enumerate(lat.elements) if isinstance(e, ACDipole))
    A = np.eye(6)
    for m in ms[:idx]:
        A = m @ A
    B = np.eye(6)
    for m in ms[idx + 1 :]:
        B = m @ B
    G = np.eye(6)
    G[ip, iu] = kick / u
    blk = np.ix_([iu, ip], [iu, ip])
    w, v = np.linalg.eig((B @ G @ A)[blk])
    k = int(np.argmin(np.abs(w - np.exp(2j * np.pi * nu))))
    at_dipole = (A[blk] @ v[:, k])[0]
    # u_n = Re(c lam^n) = u sin(2 pi (nu n + lag))  =>  c = -i u e^{2 pi i lag}
    c = -1j * u * np.exp(2j * np.pi * LAG) / at_dipole
    return np.real(c * v[:, k])


def _exact_state(lat: Lattice, u_pred: float, nu: float = NU, plane: str = "x"):
    """The exact period-q orbit seeded at a predicted state: ``(u, quadrature, Jacobian)``."""
    acd = next(e for e in lat.elements if isinstance(e, ACDipole))
    q = Fraction(nu).limit_denominator(1000).denominator
    z, _, J = _periodic_orbit(lat, _seed(lat, u_pred, acd.amplitude, nu, plane), q, plane)
    if z is None:
        return None, None, None
    u, quad = _dipole_amplitude(lat, z, q, nu, plane)
    return u, quad, J
