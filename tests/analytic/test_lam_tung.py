r"""Analytic gate — the Lam-Tung relation ``A_0 = A_2`` at O(alpha_s) (closed form).

This is the *physics* gate of the A1 milestone. The angular-coefficient extraction
machinery is pinned separately (``test_angular_coefficients.py``); a round-trip that
injects ``A_0 = A_2`` and recovers it would be **circular** here. The Lam-Tung
relation is *dynamical* — the Drell-Yan analog of the Callan-Gross relation
``2xF_1 = F_2``: it follows from the spin-1/2 nature of the quark coupling, not from
kinematics or current conservation. So the honest check is a genuine O(alpha_s) QCD
computation, done here from first principles with **explicit Dirac gamma matrices**
(no remembered helicity-amplitude constants):

  1. Build the production hadronic tensor ``W^{mu nu}`` for single-parton emission
     (both ``q qbar -> V* g`` and ``q g -> V* q``) via the two Feynman diagrams,
     with quark spin sums and the gluon-polarization sum as traces of explicit
     gamma matrices, in the Collins-Soper (V rest) frame.
  2. Contract with the leptonic tensor ``L^{mu nu}`` of ``V -> l- l+`` at CS angles
     ``(theta, phi)`` to get the lepton angular distribution ``dsigma/dOmega``.
  3. Project ``A_0`` and ``A_2`` with the (independently validated) moment weights.
  4. Show ``A_0 - A_2 == 0`` on the parton on-shell surface. For ``q qbar -> V g``
     this is a **closed-form symbolic** proof: the massless-parton constraint
     ``k^2`` divides the ``A_0 - A_2`` numerator (polynomial remainder zero). Both
     channels are also confirmed to machine precision by exact Gauss quadrature
     (``I`` is a bounded-degree trig polynomial, so the moments integrate exactly).

Correctness anchors so a buggy tensor can't sneak through: the computed ``W`` is
real, symmetric, and V-current-conserved (``q_mu W^{mu nu} = 0``); and the extracted
``A_0`` is a nonzero, physical (>= 0) coefficient, so ``A_0 = A_2`` is not vacuous.
"""

from __future__ import annotations

import functools

import mpmath as mp
import numpy as np
import sympy as sp

# ===========================================================================
# Numeric Dirac algebra (Dirac basis), metric diag(+,-,-,-). Used by the fast
# exact-quadrature checks for both partonic channels.
# ===========================================================================
_I2, _Z2 = np.eye(2), np.zeros((2, 2))
_sx = np.array([[0, 1], [1, 0]], complex)
_sy = np.array([[0, -1j], [1j, 0]], complex)
_sz = np.array([[1, 0], [0, -1]], complex)


def _nblk(a, b, c, d):
    return np.block([[a, b], [c, d]])


GN = [
    _nblk(_I2, _Z2, _Z2, -_I2).astype(complex),
    _nblk(_Z2, _sx, -_sx, _Z2),
    _nblk(_Z2, _sy, -_sy, _Z2),
    _nblk(_Z2, _sz, -_sz, _Z2),
]
MET = np.array([1.0, -1, -1, -1])


def _nsl(p):  # Feynman slash p-mu gamma^mu
    return GN[0] * p[0] - GN[1] * p[1] - GN[2] * p[2] - GN[3] * p[3]


def _ndot(a, b):  # Minkowski dot with metric diag(+,-,-,-)
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def _cs_momenta(E1: float, E2: float, d: float) -> tuple[np.ndarray, ...]:
    """CS-frame (V rest) momenta: incoming partons symmetric about z in the x-z
    plane (the CS-axis definition); V at rest; recoil parton by conservation."""
    Q = (E1 + E2) - np.sqrt((E1 + E2) ** 2 - 4 * E1 * E2 * np.cos(d) ** 2)
    p1 = np.array([E1, E1 * np.sin(d), 0, E1 * np.cos(d)])
    p2 = np.array([E2, E2 * np.sin(d), 0, -E2 * np.cos(d)])
    qV = np.array([Q, 0, 0, 0.0])
    recoil = p1 + p2 - qV  # gluon (qqbar) or outgoing quark (qg); massless
    return p1, p2, qV, recoil, Q


def _W_qqbar(E1: float, E2: float, d: float) -> tuple[np.ndarray, float]:
    """Hadronic tensor for q(p1) qbar(p2) -> V(qV) g(k)."""
    p1, p2, qV, k, Q = _cs_momenta(E1, E2, d)
    DA, DB = -2 * _ndot(p2, k), -2 * _ndot(p1, k)  # (p1-qV)^2=(k-p2)^2 ; (p1-k)^2
    kmp2s, p1mks, p1s, p2s = _nsl(k - p2), _nsl(p1 - k), _nsl(p1), _nsl(p2)

    def Omu(m, a):
        return GN[a] @ kmp2s @ GN[m] / DA + GN[m] @ p1mks @ GN[a] / DB

    def Ob(n, a):
        return GN[n] @ kmp2s @ GN[a] / DA + GN[a] @ p1mks @ GN[n] / DB

    W = np.array(
        [
            [
                sum((-MET[a]) * np.trace(p2s @ Omu(m, a) @ p1s @ Ob(n, a)) for a in range(4))
                for n in range(4)
            ]
            for m in range(4)
        ]
    )
    return W.real, Q


def _W_qg(E1: float, E2: float, d: float) -> tuple[np.ndarray, float]:
    """Hadronic tensor for q(p1) g(p2) -> V(qV) q(p3) (the crossed channel)."""
    p1, p2, qV, p3, Q = _cs_momenta(E1, E2, d)
    SH, D2 = 2 * _ndot(p1, p2), _ndot(p1 - qV, p1 - qV)  # (p1+p2)^2 ; (p1-qV)^2
    p1p2s, p1mqVs, p1s, p3s = _nsl(p1 + p2), _nsl(p1 - qV), _nsl(p1), _nsl(p3)

    def Omu(m, a):
        return GN[m] @ p1p2s @ GN[a] / SH + GN[a] @ p1mqVs @ GN[m] / D2

    def Ob(n, a):
        return GN[a] @ p1p2s @ GN[n] / SH + GN[n] @ p1mqVs @ GN[a] / D2

    W = np.array(
        [
            [
                sum((-MET[a]) * np.trace(p3s @ Omu(m, a) @ p1s @ Ob(n, a)) for a in range(4))
                for n in range(4)
            ]
            for m in range(4)
        ]
    )
    return W.real, Q


# Exact quadrature grid: Gauss-Legendre in cos(theta) (the sin(theta) Jacobian is
# absorbed into dcos), uniform in phi. I(theta,phi) is a quadratic form in the unit
# lepton direction, so P_i*I is a bounded-degree trig polynomial -> integrated
# exactly by a modest grid (no Monte-Carlo ratio bias).
_XG, _WG = np.polynomial.legendre.leggauss(24)
_NPHI = 48
_PHG = (np.arange(_NPHI) + 0.5) * 2 * np.pi / _NPHI
_CT, _PHm = np.meshgrid(_XG, _PHG, indexing="ij")
_WT = np.outer(_WG, np.full(_NPHI, 2 * np.pi / _NPHI))
_STm = np.sqrt(1 - _CT**2)


def _moments_from_W(W: np.ndarray, Q: float) -> tuple[float, float]:
    """Return (A_0, A_2) by exact-quadrature moment projection over the sphere."""
    nx, ny, nz = _STm * np.cos(_PHm), _STm * np.sin(_PHm), _CT
    intensity = np.zeros_like(_CT)
    for i in range(_CT.shape[0]):
        for j in range(_CT.shape[1]):
            l1 = np.array([Q / 2, Q / 2 * nx[i, j], Q / 2 * ny[i, j], Q / 2 * nz[i, j]])
            l2 = np.array([Q / 2, -l1[1], -l1[2], -l1[3]])
            l1s, l2s = _nsl(l1), _nsl(l2)
            L = np.array(
                [[np.trace(l1s @ GN[m] @ l2s @ GN[n]).real for n in range(4)] for m in range(4)]
            )
            intensity[i, j] = np.sum(MET[:, None] * MET[None, :] * L * W)
    norm = np.sum(_WT * intensity)
    A0 = np.sum(_WT * (4 - 10 * _CT**2) * intensity) / norm
    A2 = np.sum(_WT * (10 * _STm**2 * np.cos(2 * _PHm)) * intensity) / norm
    return A0, A2


# ---------------------------------------------------------------------------
# Correctness anchor — W is a valid hadronic tensor (real, symmetric, conserved).
# ---------------------------------------------------------------------------
def test_hadronic_tensor_is_physical() -> None:
    W, Q = _W_qqbar(2.0, 3.0, 0.8)
    assert np.abs(W - W.T).max() < 1e-10  # symmetric
    qlow = MET * np.array([Q, 0, 0, 0.0])
    assert np.abs(qlow @ W).max() < 1e-9  # V-current conservation q_mu W^{mu nu} = 0


# ---------------------------------------------------------------------------
# Lam-Tung, both channels, by exact quadrature (machine precision, no MC bias).
# ---------------------------------------------------------------------------
def test_lam_tung_both_channels_exact_quadrature() -> None:
    rng = np.random.default_rng(2)
    for name, Wf in (("qqbar->Vg", _W_qqbar), ("qg->Vq", _W_qg)):
        max_diff, sample_a0 = 0.0, None
        for _ in range(8):
            e1, e2, d = rng.uniform(1, 5), rng.uniform(1, 5), rng.uniform(0.2, 1.35)
            W, Q = Wf(e1, e2, d)
            a0, a2 = _moments_from_W(W, Q)
            max_diff = max(max_diff, abs(a0 - a2))
            sample_a0 = a0
        assert max_diff < 1e-10, f"{name}: max|A0-A2| = {max_diff:.2e}"
        # Not vacuous: A_0 is a genuine, nonzero, physical (>= 0) coefficient.
        assert sample_a0 is not None and 0.0 <= sample_a0 <= 2.0 and abs(sample_a0) > 0.05


# ===========================================================================
# Closed-form symbolic proof for q qbar -> V g: k^2 divides the A_0 - A_2
# numerator, so A_0 = A_2 identically on the gluon on-shell surface k^2 = 0.
# ===========================================================================
_SI2, _SZ2 = sp.eye(2), sp.zeros(2, 2)
_ssx = sp.Matrix([[0, 1], [1, 0]])
_ssy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
_ssz = sp.Matrix([[1, 0], [0, -1]])


def _sblk(a, b, c, d):
    return sp.Matrix(sp.BlockMatrix([[a, b], [c, d]]))


GS = [
    _sblk(_SI2, _SZ2, _SZ2, -_SI2),
    _sblk(_SZ2, _ssx, -_ssx, _SZ2),
    _sblk(_SZ2, _ssy, -_ssy, _SZ2),
    _sblk(_SZ2, _ssz, -_ssz, _SZ2),
]
_METS = [1, -1, -1, -1]


def _ssl(p):  # symbolic Feynman slash
    return GS[0] * p[0] - GS[1] * p[1] - GS[2] * p[2] - GS[3] * p[3]


def _sdot(a, b):  # symbolic Minkowski dot
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


@functools.lru_cache(maxsize=1)
def _qqbar_symbolic():
    """Build (W, Lw, {DA:val,DB:val}, k, symbols) for q qbar -> V g symbolically.

    W[m][n] carries no angular dependence (kept with DA, DB as bare symbols for
    speed); Lw[m][n] = MET_m MET_n Tr[l1slash G^m l2slash G^n] holds all the
    theta/phi dependence. By linearity of the solid-angle integral the moment of
    the contracted intensity factorizes,

        int P_i (sum_mn Lw[m][n] W[m][n]) dOmega
            = sum_mn (int P_i Lw[m][n] dOmega) W[m][n],

    so the caller integrates the ~10 low-degree leptonic-basis polynomials once and
    contracts with the symbolic W afterward. This is algebraically identical to
    integrating the fully contracted intensity, but collapses hours (sympy pushing
    every E1,E2,d,Q,DA,DB factor through the integrator) to seconds."""
    E1, E2, d, Q = sp.symbols("E1 E2 delta Q", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    sd, cd = sp.sin(d), sp.cos(d)
    p1 = [E1, E1 * sd, 0, E1 * cd]
    p2 = [E2, E2 * sd, 0, -E2 * cd]
    k = [p1[i] + p2[i] - [Q, 0, 0, 0][i] for i in range(4)]
    DAs, DBs = sp.symbols("DA DB")
    DA = sp.expand(-2 * _sdot(p2, k))
    DB = sp.expand(-2 * _sdot(p1, k))
    kmp2s, p1mks, p1s, p2s = (
        _ssl([k[i] - p2[i] for i in range(4)]),
        _ssl([p1[i] - k[i] for i in range(4)]),
        _ssl(p1),
        _ssl(p2),
    )

    def Omu(m, a):
        return GS[a] * kmp2s * GS[m] / DAs + GS[m] * p1mks * GS[a] / DBs

    def Ob(n, a):
        return GS[n] * kmp2s * GS[a] / DAs + GS[a] * p1mks * GS[n] / DBs

    W = [
        [
            sum((-_METS[a]) * (p2s * Omu(m, a) * p1s * Ob(n, a)).trace() for a in range(4))
            for n in range(4)
        ]
        for m in range(4)
    ]
    st, ct = sp.sin(th), sp.cos(th)
    l1 = [Q / 2, Q / 2 * st * sp.cos(ph), Q / 2 * st * sp.sin(ph), Q / 2 * ct]
    l2 = [Q / 2, -l1[1], -l1[2], -l1[3]]
    l1s, l2s = _ssl(l1), _ssl(l2)
    Lw = [
        [_METS[m] * _METS[n] * (l1s * GS[m] * l2s * GS[n]).trace() for n in range(4)]
        for m in range(4)
    ]
    return W, Lw, {DAs: DA, DBs: DB}, k, (E1, E2, d, Q), (th, ph)


def test_lam_tung_qqbar_symbolic_closed_form() -> None:
    W, Lw, dsub, k, (E1, E2, d, Q), (th, ph) = _qqbar_symbolic()
    DAs, DBs = sorted(dsub, key=lambda s: s.name)  # the bare DA, DB denom symbols
    st, ct = sp.sin(th), sp.cos(th)

    # Solid-angle moments of the small leptonic basis Lw[m][n] (theta/phi only, low
    # degree; sin theta Jacobian REQUIRED). Fast -- W carries no angular dependence.
    def solid(expr):
        e = sp.integrate(sp.expand_trig(sp.expand(expr * st)), (ph, 0, 2 * sp.pi))
        return sp.integrate(sp.expand_trig(sp.expand(e)), (th, 0, sp.pi))

    b0 = [[solid((4 - 10 * ct**2) * Lw[m][n]) for n in range(4)] for m in range(4)]
    b2 = [[solid((10 * st**2 * sp.cos(2 * ph)) * Lw[m][n]) for n in range(4)] for m in range(4)]
    bnorm = [[solid(Lw[m][n]) for n in range(4)] for m in range(4)]

    # Each W[m][n] is a sum of terms over DA^2, DA*DB, DB^2 -> common denominator
    # DA^2 DB^2. Clearing it turns every moment into a pure polynomial numerator, so
    # A_0 - A_2 = (P0 - P2)/Pn needs NO multivariate GCD. (sp.cancel on the fully
    # contracted rational intensity is the real bottleneck -- ~2 h; clearing the
    # KNOWN denominator instead is polynomial-only and runs in seconds.)
    numW = [[sp.expand(W[m][n] * DAs**2 * DBs**2) for n in range(4)] for m in range(4)]

    def numerator(bmat):
        p = sum(bmat[m][n] * numW[m][n] for m in range(4) for n in range(4))
        return sp.expand(p.subs(dsub))  # substitute DA, DB kinematic polynomials

    Ndiff = sp.expand(numerator(b0) - numerator(b2))

    # A_0 - A_2 = Ndiff / Pn vanishes on the gluon on-shell surface k^2 = 0 iff k^2
    # DIVIDES Ndiff. Prove it by polynomial remainder in Q (exact; avoids sqrt(Q)).
    k2 = sp.expand(_sdot(k, k))
    rem = sp.rem(sp.Poly(Ndiff, Q), sp.Poly(k2, Q)).as_expr()
    assert sp.simplify(rem) == 0, "k^2 does not divide the (A0 - A2) numerator"

    # Not vacuous: A_0 = P0/Pn is a nonzero physical (>= 0) number on-shell.
    a0 = numerator(b0) / numerator(bnorm)
    mp.mp.dps = 30
    A0f = sp.lambdify((E1, E2, d, Q), a0, "mpmath")
    e1, e2, dv = mp.mpf("2"), mp.mpf("3"), mp.mpf("0.8")
    qv = (e1 + e2) - mp.sqrt((e1 + e2) ** 2 - 4 * e1 * e2 * mp.cos(dv) ** 2)
    assert abs(A0f(e1, e2, dv, qv)) > mp.mpf("0.1")
