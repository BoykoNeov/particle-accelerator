"""Injection and filamentation in the editor (``editor/inject.js`` on ``accsim-track.js``).

A beam injected off-centre or with the wrong shape does not stay that way: every particle
keeps its own Courant-Snyder action ``J``, but their betatron phases spread — here through
the energy spread and the chromaticity — until the beam is a filled ellipse matched to the
ring, with emittance equal to its mean action. These gates hold the editor's version of
that against closed forms, the package, and the number the editor displays:

  * **the injected beam** — moment-matched, so ``<J>`` (ring Twiss) is the closed form
    ``eps_inj Bmag + J(offset)`` to round-off; at a point where ``alpha != 0``, so a flipped
    ``alpha`` in the sampler cannot hide. Gates the sampler, not the tracker.
  * **the decoherence rate** — a zero-emittance offset beam with frozen energies: its
    centroid follows ``|(1/N) sum exp(i 2pi Q' delta_p n)|`` with the sample's own deltas and
    the editor's displayed ``Q'``. The residual is owned (first order in ``sigma_delta``,
    asserted by its scaling). Blind to the SIGN of ``Q'`` — so:
  * **the sign** — one particle at ``+delta`` and one at ``-delta``, tracked in the editor,
    their tunes measured: the slope is the displayed ``Q'`` on every ring preset. This is
    the gate that ties the moving beam to the number on the screen.
  * **the filamented emittance** — turn-averaged second moments about the closed orbit
    land on ``<J>``, and with **no** energy spread a mismatched beam never filaments at all.
  * **the statistics** — ``beamStats`` against numpy on the same states.

All run on the cavity-free ``kicked-ring`` (rolled one element on, so the injection point
has ``alpha != 0`` and ``D' != 0``). On the cavity rings the energy oscillates every ~14
turns and the chromatic phase spread never exceeds ~0.06 rad, so an injected beam there
barely filaments — that is physics, and radiation damping is OFF in the editor.

Skipped, not failed, when Node is not on the PATH (CI runs without it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

import accsim as ac
from accsim.scenario import load_scenario
from accsim.tune import _plane_tune

ROOT = Path(__file__).resolve().parents[2]
EDITOR = ROOT / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")


def _presets() -> list[dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return json.loads(text[start : text.rindex("];") + 1])


PRESETS = {p["id"]: p for p in _presets()}


def _ring() -> dict:
    """``kicked-ring`` with its first element moved to the end: no RF (energies frozen), a
    displaced closed orbit, and at the new start ``alpha_x = 2.18``, ``alpha_y = -0.45``,
    ``D' = -0.24`` — every term of the injection formulas is non-zero."""
    p = PRESETS["kicked-ring"]
    q = dict(p)
    q["elements"] = p["elements"][1:] + p["elements"][:1]
    return q


RING = _ring()


def _node(cases: list[dict]) -> list[dict]:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, str(EDITOR / "inject-selftest.js")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    for r in out:
        assert "error" not in r, r.get("error")
    return out


def _action(u: np.ndarray, up: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    gamma = (1.0 + alpha * alpha) / beta
    return 0.5 * (gamma * u * u + 2.0 * alpha * u * up + beta * up * up)


def _bmag(beta: float, alpha: float, beta_i: float, alpha_i: float) -> float:
    return 0.5 * (
        beta * (1 + alpha_i**2) / beta_i - 2 * alpha * alpha_i + (1 + alpha**2) / beta * beta_i
    )


# --- 1. the closed forms, derived --------------------------------------------------------
def test_the_injected_mean_action_is_eps_bmag_plus_the_offsets_own() -> None:
    r"""``<J> = eps_i Bmag + J(dx, dpx)``, from the sampler's own construction.

    The sampler draws ``x = sqrt(eps beta_i) a + dx``, ``px = sqrt(eps/beta_i)(b - alpha_i a)
    + dpx`` with ``a, b`` independent unit normals. Taking the expectation of the ring's
    ``J = (gamma x^2 + 2 alpha x px + beta px^2)/2`` symbolically gives the mismatch factor
    ``Bmag = (beta gamma_i - 2 alpha alpha_i + gamma beta_i)/2`` times ``eps``, plus the
    offset's own action and **no cross term** (the offset and the spread are independent).
    ``Bmag >= 1`` with equality only when matched, so injection can only grow a beam.
    """
    eps, b, bi = sp.symbols("eps beta beta_i", positive=True)
    a, ai, dx, dpx = sp.symbols("alpha alpha_i dx dpx", real=True)
    A, B = sp.symbols("A B", real=True)  # the unit normals
    x = sp.sqrt(eps * bi) * A + dx
    px = sp.sqrt(eps / bi) * (B - ai * A) + dpx
    g = (1 + a**2) / b
    J = sp.expand((g * x**2 + 2 * a * x * px + b * px**2) / 2)
    # E[A] = E[B] = E[AB] = 0, E[A^2] = E[B^2] = 1.
    poly = sp.Poly(J, A, B)
    moments = {(0, 0): 1, (1, 0): 0, (0, 1): 0, (1, 1): 0, (2, 0): 1, (0, 2): 1}
    mean_J = sum(c * moments[m] for m, c in zip(poly.monoms(), poly.coeffs(), strict=True))
    gi = (1 + ai**2) / bi
    bmag = (b * gi - 2 * a * ai + g * bi) / 2
    j_off = (g * dx**2 + 2 * a * dx * dpx + b * dpx**2) / 2
    assert sp.simplify(mean_J - (eps * bmag + j_off)) == 0
    # Bmag - 1 is a perfect square over (beta beta_i): matched iff Bmag = 1.
    assert sp.simplify((bmag - 1) * 2 * b * bi - ((b - bi) ** 2 + (a * bi - ai * b) ** 2)) == 0


def test_the_injected_beam_has_exactly_the_requested_moments() -> None:
    """The editor's sampler against the closed form, with the package's own Twiss and orbit.

    Moment-matched (whitened) sampling makes this a round-off gate, not a ``1/sqrt(N)`` one —
    measured ``<= 7e-16`` relative. The Twiss, closed orbit and dispersion used to strip and
    weigh the particles here are the **package's** (``closed_twiss``, ``closed_orbit``), not
    the editor's, so a wrong optics port would fail it too.
    """
    inj = {
        "n": 500,
        "seed": 4,
        "emit_x": 5e-8,
        "emit_y": 2e-9,
        "sigma_delta": 7e-4,
        "delta0": 2e-4,
        "beta_x": 9.0,
        "alpha_x": 1.1,
        "beta_y": 4.0,
        "alpha_y": 0.2,
        "dx": 4e-4,
        "dpx": -3e-5,
        "dy": -1e-4,
        "dpy": 2e-5,
    }
    (r,) = _node([{"scenario": RING, "mode": "inject", "inj": inj}])
    S = np.array(r["initial"], dtype=float)
    lat = load_scenario(RING).lattice
    tw = ac.closed_twiss(lat)
    co = np.asarray(ac.closed_orbit(lat))
    assert abs(tw.alpha_x) > 1.0 and abs(tw.disp_px) > 0.1 and abs(co[0]) > 1e-4  # not blind

    d = S[:, 5]
    assert d.mean() == pytest.approx(inj["delta0"], rel=1e-12)
    assert d.std() == pytest.approx(inj["sigma_delta"], rel=1e-12)
    for pl, (iu, ip), D, Dp in (
        ("x", (0, 1), tw.disp_x, tw.disp_px),
        ("y", (2, 3), tw.disp_y, tw.disp_py),
    ):
        u = S[:, iu] - co[iu] - D * d
        up = S[:, ip] - co[ip] - Dp * d
        beta, alpha = (tw.beta_x, tw.alpha_x) if pl == "x" else (tw.beta_y, tw.alpha_y)
        want = inj[f"emit_{pl}"] * _bmag(beta, alpha, inj[f"beta_{pl}"], inj[f"alpha_{pl}"])
        want += float(_action(np.array(inj[f"d{pl}"]), np.array(inj[f"dp{pl}"]), beta, alpha))
        assert _action(u, up, beta, alpha).mean() == pytest.approx(want, rel=1e-13), pl


# --- 2. the decoherence rate ---------------------------------------------------------------
def _decoherence_residual(sigma_delta: float) -> tuple[float, float]:
    """Max |centroid_n / centroid_0 - envelope_n| over the run, both planes, for a zero-
    emittance beam offset in x and y. The run is long enough for the same total phase
    spread at every ``sigma_delta`` (``n sigma_delta`` fixed), so the envelope reaches the
    same depth and only the model's error changes."""
    turns = int(round(150 * 7e-4 / sigma_delta))
    inj = {
        "n": 400,
        "seed": 5,
        "emit_x": 0.0,
        "emit_y": 0.0,
        "sigma_delta": sigma_delta,
        "dx": 1e-3,
        "dy": 3e-4,
    }
    (r,) = _node([{"scenario": RING, "mode": "stats", "inj": inj, "turns": turns}])
    deltas = np.array(r["deltas"])
    out = []
    for i, pl in enumerate(("x", "y")):
        c = np.array([s[pl]["centroid"] for s in r["stats"]])
        qp = r["optics"]["chromaticity"][i]
        n = np.arange(turns + 1)
        env = np.abs(np.exp(2j * np.pi * qp * np.outer(n, deltas)).mean(axis=1))
        assert env[-1] < 0.05  # the run really decoheres (else the gate is idle)
        out.append(float(np.max(np.abs(c / c[0] - env))))
    return out[0], out[1]


def test_the_centroid_decoheres_at_the_displayed_chromaticity() -> None:
    """The tracked centroid follows ``|(1/N) sum exp(i 2pi Q' delta_p n)|`` to ``3.4e-3``
    while it falls from 1 to below ``0.05`` — and that residual is **owned**: it halves
    when ``sigma_delta`` halves (at fixed total phase spread), the first-order signature of
    what the linear model leaves out — the chromatic beta-beat (each particle's invariant
    belongs to its own off-momentum optics) and ``Q''``. A wrong ``Q'``, or a tracker with
    no chromaticity, misses by order one.
    """
    r1 = _decoherence_residual(7e-4)
    r2 = _decoherence_residual(3.5e-4)
    for a, b in zip(r1, r2, strict=True):
        assert a < 5e-3
        assert a / b == pytest.approx(2.0, rel=0.1)


# --- 3. the sign ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preset_id", [k for k, p in PRESETS.items() if p.get("periodic", True)], ids=str
)
def test_the_tracked_tune_slope_is_the_displayed_chromaticity(preset_id: str) -> None:
    """``+delta`` and ``-delta`` tracked in the editor, tunes by NAFF: slope = displayed Q'.

    The RF cavity is removed so the energy stays where it was put. The particle starts on
    the off-momentum closed orbit (the package's, so the betatron amplitude is the small
    one put in). This includes the ``wiggler-ring``, whose ``Q'_y`` was 0.4% wrong in the
    display until the wiggler's own term was added (2026-10-07) — this gate is what would
    have caught it from the editor's side.
    """
    p = dict(PRESETS[preset_id])
    p["elements"] = [e for e in p["elements"] if e.get("type") != "RFCavity"]
    h, turns = 1e-4, 2048
    lat = load_scenario(p).lattice
    states = []
    for d in (+h, -h):
        co = np.asarray(ac.closed_orbit(lat, delta=d)) if lat.elements else np.zeros(4)
        states.append([co[0] + 1e-6, co[1], co[2] + 1e-6, co[3], 0.0, d])
    (r,) = _node([{"scenario": p, "mode": "tbt", "states": states, "turns": turns}])
    q = []
    for traj in r["tbt"]:
        t = np.array(traj)
        q.append(
            (
                _plane_tune(t[:, 0] - t[:, 0].mean(), t[:, 1] - t[:, 1].mean()),
                _plane_tune(t[:, 2] - t[:, 2].mean(), t[:, 3] - t[:, 3].mean()),
            )
        )
    (shown,) = _node([{"scenario": p, "mode": "inject", "inj": {"n": 8}}])
    xi = shown["optics"]["chromaticity"]
    for i in range(2):
        slope = (q[0][i] - q[1][i]) / (2 * h)
        assert slope == pytest.approx(xi[i], rel=2e-4, abs=2e-4), (preset_id, i)


# --- 4. the filamented emittance -------------------------------------------------------------
def _filament(sigma_delta: float, turns: int) -> dict:
    inj = {
        "n": 400,
        "seed": 6,
        "emit_x": 5e-8,
        "emit_y": 5e-10,
        "sigma_delta": sigma_delta,
        "beta_x": 6.0,
        "alpha_x": 0.4,
        "dx": 3e-4,
        "dy": 5e-5,
    }
    (r,) = _node([{"scenario": RING, "mode": "stats", "inj": inj, "turns": turns}])
    return r


def test_a_mismatched_offset_beam_filaments_to_its_mean_action() -> None:
    """Injected at 47% of its eventual emittance, the beam ends at ``<J>``.

    Two readings of "ends at": the per-turn rms emittance late in the run sits within
    ``3/sqrt(N)`` of ``<J>`` (it fluctuates turn to turn by ``~1/sqrt(N)`` — a finite bunch),
    and the second moments **averaged over turns, about the closed orbit**, land on ``<J>``
    to ``5.6e-4``. About the orbit and not about the centroid: a finite bunch's centroid never
    settles (``|<z>|^2 ~ <2J>/N``), so central moments sit low by ``1/N`` on average however
    long you average — the first version of this gate measured exactly that, ``-2e-3`` at
    ``N = 400``, and it did not move with ``sigma_delta``.

    The remaining ``5.6e-4`` is the energy spread's: with the window's own floor (``2e-5``,
    the ``sigma_delta = 0`` value below) removed it falls ~4x per halving of
    ``sigma_delta`` — second order, the chromatic beta-beat entering a determinant.
    """
    turns = 1500
    st = _filament(7e-4, turns)["stats"]
    J0 = st[0]["x"]["meanJ"]
    assert st[0]["x"]["emit"] / J0 == pytest.approx(0.4747, abs=1e-3)  # injected well inside

    late = st[turns // 3 :]
    emit_late = np.array([s["x"]["emit"] for s in late])
    assert np.all(np.abs(emit_late / J0 - 1.0) < 3.0 / np.sqrt(400))
    m2 = np.array([s["x"]["m2ref"] for s in late]).mean(axis=0)
    assert np.sqrt(m2[0] * m2[2] - m2[1] ** 2) / J0 - 1.0 == pytest.approx(0.0, abs=1e-3)


def test_with_no_energy_spread_the_beam_never_filaments() -> None:
    """The control that makes the gate above mean something: ``sigma_delta = 0``.

    Every particle then has the same tune, the mismatched ellipse turns rigidly, and the rms
    emittance stays at its injected 47% of ``<J>`` for good (to a ``5e-5`` wobble, measured:
    the exact drift is not quite linear, which bends the ellipse slightly without spreading it) — in this editor the energy
    spread is the ONLY thing that spreads the phases (no octupoles, no radiation), so
    without it there is no filamentation. The turn-averaged moments still reach ``<J>``,
    to the window's floor (``2e-5`` measured) — that average is blind to filamentation, which
    is why the per-turn reading above is the one that tests it.
    """
    turns = 1500
    st = _filament(0.0, turns)["stats"]
    J0 = st[0]["x"]["meanJ"]
    emit_0 = st[0]["x"]["emit"]
    emit = np.array([s["x"]["emit"] for s in st]) / emit_0
    # It wobbles by ~5e-5 (the exact drift's kinematic nonlinearity distorts the ellipse a
    # little) and does not grow: the last third of the run is no wider than the first.
    assert np.max(np.abs(emit - 1.0)) < 1e-4
    third = turns // 3
    assert np.max(emit[-third:]) <= np.max(emit[:third]) + 1e-5
    assert emit_0 / J0 == pytest.approx(0.4747, abs=1e-3)
    late = st[turns // 3 :]
    m2 = np.array([s["x"]["m2ref"] for s in late]).mean(axis=0)
    assert abs(np.sqrt(m2[0] * m2[2] - m2[1] ** 2) / J0 - 1.0) < 1e-4


# --- 5. the statistics themselves ---------------------------------------------------------------
def test_beam_stats_match_numpy_on_the_same_states() -> None:
    """``beamStats`` on the final states, recomputed in numpy with the same optics record:
    centroid amplitude, rms emittance, ``<J>`` and both moment sets, to round-off."""
    r = _filament(7e-4, 40)
    S = np.array(r["final"], dtype=float)
    o = r["optics"]
    t, co = o["twiss"], o["orbit0"]
    last = r["stats"][-1]
    for pl, (iu, ip) in (("x", (0, 1)), ("y", (2, 3))):
        D, Dp = t[f"disp_{pl}"], t[f"disp_p{pl}"]
        beta, alpha = t[f"beta_{pl}"], t[f"alpha_{pl}"]
        u = S[:, iu] - co[iu] - D * S[:, 5]
        up = S[:, ip] - co[ip] - Dp * S[:, 5]
        mu, mup = u.mean(), up.mean()
        cov = np.cov(np.vstack([u, up]), bias=True)
        want = {
            "centroid": np.hypot(mu, alpha * mu + beta * mup) / np.sqrt(beta),
            "emit": np.sqrt(np.linalg.det(cov)),
            "meanJ": _action(u, up, beta, alpha).mean(),
        }
        for key, val in want.items():
            assert last[pl][key] == pytest.approx(val, rel=1e-10), (pl, key)
        assert last[pl]["m2"] == pytest.approx([cov[0, 0], cov[0, 1], cov[1, 1]], rel=1e-10)
        raw = [np.mean(u * u), np.mean(u * up), np.mean(up * up)]
        assert last[pl]["m2ref"] == pytest.approx(raw, rel=1e-10)
