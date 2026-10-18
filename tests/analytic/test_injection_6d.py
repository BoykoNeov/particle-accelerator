"""The 6D-matched injected beam (``editor/inject.js`` on a ring with an RF cavity).

A cavity where the dispersion is not zero couples the energy oscillation into the sideways
motion, so the beam the one-turn map leaves unchanged is not the uncoupled-Twiss-plus-
dispersion beam the Injection tab used to draw: that one's measured size pulsed 1.0x-1.7x
with no aiming error at all. The matched beam is ``Sigma = W diag(eps) W^T`` with ``W`` the
6D normal form, and the editor builds ``W`` without an eigen-solver (``normalForm6`` in
``editor/accsim-optics.js``: a cubic for the three ``cos mu`` and a real projector per mode).
These gates hold it against the package and against tracking:

  * **the port** — ``normalForm6``'s ``W``, tunes, mode Twiss and dynamic dispersion against
    ``accsim.twiss.normal_form(M, method="6d")``, on rings where the injection point has
    ``alpha != 0``, ``D' != 0`` and a displaced orbit; ``W`` symplectic, ``W^-1 M W`` a rotation;
  * **the sampler** — the drawn beam's covariance is ``W diag(eps) W^T`` with the PACKAGE's
    ``W``; a mismatch and an aiming error give ``eps Bmag + J(offset)`` with the package's
    mode Twiss; an energy offset rides the dynamic dispersion;
  * **tracking** — the matched beam keeps its mode emittance over 1000 turns off resonance,
    and the old beam sits at the mismatch its own covariance predicts (~1.2x);
  * **the owner of the default ring's growth** — ``electron-ring`` sits 0.0009 from the
    synchrotron sideband ``Q_1 + 3 Q_s = 6``; at fixed voltage the growth follows the tune.

Node-driven; skipped (not failed) when Node is not on the PATH, as ``test_injection.py``.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from accsim.orbit import closed_orbit
from accsim.scenario import load_scenario
from accsim.twiss import normal_form

ROOT = Path(__file__).resolve().parents[2]
EDITOR = ROOT / "editor"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")


def _presets() -> dict[str, dict]:
    text = (EDITOR / "presets.js").read_text(encoding="utf-8")
    marker = "\nvar ACCSIM_PRESETS = "
    start = text.index(marker) + len(marker)
    return {p["id"]: p for p in json.loads(text[start : text.rindex("];") + 1])}


PRESETS = _presets()


def _rolled(p: dict) -> dict:
    q = copy.deepcopy(p)
    q["elements"] = q["elements"][1:] + q["elements"][:1]
    return q


def _kicked_with_cavity() -> dict:
    """``kicked-ring`` (a displaced quadrupole and a corrector: the orbit is off-centre) with
    ``electron-ring``'s cavity put back at the end, then rolled one element on."""
    q = copy.deepcopy(PRESETS["kicked-ring"])
    cavity = next(e for e in PRESETS["electron-ring"]["elements"] if e["type"] == "RFCavity")
    q["elements"].append(copy.deepcopy(cavity))
    return _rolled(q)


def _with_voltage(p: dict, volts: float) -> dict:
    q = copy.deepcopy(p)
    for e in q["elements"]:
        if e["type"] == "RFCavity":
            e["voltage"] = volts
    return q


def _scaled_qf(p: dict, f: float) -> dict:
    q = copy.deepcopy(p)
    for e in q["elements"]:
        if e.get("name") == "QF.h":
            e["k1l"] *= 1.0 + f
    return q


FIXTURES = {
    "electron-ring": PRESETS["electron-ring"],
    "wiggler-ring": PRESETS["wiggler-ring"],
    "electron-ring-rolled": _rolled(PRESETS["electron-ring"]),
    "kicked-ring-cavity-rolled": _kicked_with_cavity(),
}
ROLLED = ("electron-ring-rolled", "kicked-ring-cavity-rolled")


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


def _package_form(scenario: dict):
    lat = load_scenario(scenario).lattice
    M, _ = lat.one_turn_map()
    return lat, M, normal_form(M, method="6d")


def _S() -> np.ndarray:
    S = np.zeros((6, 6))
    for p in range(3):
        S[2 * p, 2 * p + 1], S[2 * p + 1, 2 * p] = 1.0, -1.0
    return S


def _bmag(beta: float, alpha: float, beta_i: float, alpha_i: float) -> float:
    return 0.5 * (
        beta * (1 + alpha_i**2) / beta_i - 2 * alpha * alpha_i + (1 + alpha**2) / beta * beta_i
    )


def _cs_action(u: float, up: float, beta: float, alpha: float) -> float:
    return 0.5 * ((1 + alpha**2) / beta * u * u + 2 * alpha * u * up + beta * up * up)


def _eps(W: np.ndarray, emit_x: float, emit_y: float, sigma_delta: float) -> np.ndarray:
    eps_z = sigma_delta**2 / (W[5, 4] ** 2 + W[5, 5] ** 2)
    return np.repeat([emit_x, emit_y, eps_z], 2)


# --- 1. the port ------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(FIXTURES), ids=str)
def test_the_editor_normal_form_is_the_packages(name: str) -> None:
    """``normalForm6`` (cubic + projectors) against ``normal_form(M, "6d")`` (numpy ``eig``).

    Two routes to one ``W``: measured agreement ``<= 1.5e-15`` of ``max |W|``. Not blind: on
    every fixture the dynamic dispersion differs from the static one by more than 10%, so
    the longitudinal mode really is mixed into ``x`` (and ``W`` is not block-diagonal).
    """
    sc = FIXTURES[name]
    (r,) = _node([{"scenario": sc, "mode": "inject", "inj": {"n": 8}}])
    nf_js = r["optics"]["normal"]
    assert nf_js is not None, r["optics"].get("normalError")
    lat, M, nf = _package_form(sc)
    W = np.array(nf_js["W"])
    scale = np.max(np.abs(nf.w))
    assert np.max(np.abs(W - nf.w)) < 1e-12 * scale
    for a, b in zip(nf_js["tunes"], nf.tunes, strict=True):
        assert abs((a - b + 0.5) % 1.0 - 0.5) < 1e-12
    assert nf_js["modeBeta"] == pytest.approx(nf.mode_beta, rel=1e-12)
    assert nf_js["modeAlpha"] == pytest.approx(nf.mode_alpha, rel=1e-12, abs=1e-12)
    assert np.allclose(nf_js["dispersion"], nf.dispersion, rtol=1e-12, atol=1e-12)
    # The editor's W^-1 is -S W^T S: that is the inverse only if W is symplectic.
    S = _S()
    assert np.max(np.abs(W.T @ S @ W - S)) < 1e-13
    Rm = np.array(nf_js["R"])
    assert np.max(np.abs(Rm - np.linalg.inv(W) @ M @ W)) < 1e-12
    for k in range(3):
        blk = Rm[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
        assert blk[0, 0] == pytest.approx(blk[1, 1], abs=1e-12)
        assert blk[0, 1] == pytest.approx(-blk[1, 0], abs=1e-12)
        off = np.delete(Rm[2 * k : 2 * k + 2], [2 * k, 2 * k + 1], axis=1)
        assert np.max(np.abs(off)) < 1e-12
    static = r["optics"]["twiss4"]["disp_x"]
    assert abs(nf.dispersion[0] / static - 1.0) > 0.1  # not blind: the cavity mixes delta into x


# --- 2. the sampler ---------------------------------------------------------------------------
def _sample(name: str, inj: dict) -> tuple[np.ndarray, dict, np.ndarray, object]:
    sc = FIXTURES[name]
    (r,) = _node([{"scenario": sc, "mode": "inject", "inj": inj}])
    lat, _M, nf = _package_form(sc)
    co = np.zeros(6)
    co[:4] = np.asarray(closed_orbit(lat))[:4]
    return np.array(r["initial"], dtype=float) - co, r, co, nf


@pytest.mark.parametrize("name", ROLLED, ids=str)
def test_the_matched_beam_is_w_diag_eps_w_transpose(name: str) -> None:
    """The drawn beam's covariance about the package's closed orbit is ``W diag(eps) W^T``
    with the package's ``W``, entry by entry, each relative to ``sqrt(Sigma_ii Sigma_jj)``.
    Moment-matched sampling makes this a round-off gate. The control: the 4D sampler's
    beam on the same ring misses the same comparison at the percent level."""
    inj = {"n": 600, "seed": 7}
    X, r, _co, nf = _sample(name, inj)
    p = r["inj"]
    want = nf.w @ np.diag(_eps(nf.w, p["emit_x"], p["emit_y"], p["sigma_delta"])) @ nf.w.T
    got = X.T @ X / len(X)
    norm = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    rel = np.abs(got - want)[norm > 0] / norm[norm > 0]
    assert np.max(rel) < 1e-12
    assert abs(np.trace(got) / np.trace(want) - 1.0) < 1e-12


@pytest.mark.parametrize("name", ROLLED, ids=str)
def test_mismatch_and_aiming_error_give_eps_bmag_plus_the_offsets_own(name: str) -> None:
    """``<J_k> = eps Bmag(beta_k, alpha_k; beta_i, alpha_i) + J(offset)`` per transverse mode,
    with the package's ``mode_beta`` / ``mode_alpha`` and actions from the package's ``W^-1``.
    The aiming error is a pure betatron excursion, so the longitudinal mode keeps exactly
    the matched ``eps_3``."""
    base = {"n": 500, "seed": 8}
    _X0, r0, _co, nf = _sample(name, base)
    bx, ax = nf.mode_beta[0], nf.mode_alpha[0]
    by, ay = nf.mode_beta[1], nf.mode_alpha[1]
    assert abs(ax) > 0.3  # not blind: alpha enters Bmag and J(offset)
    inj = dict(
        base,
        beta_x=1.4 * bx,
        alpha_x=ax - 0.6,
        beta_y=0.7 * by,
        alpha_y=ay + 0.3,
        dx=4e-4,
        dpx=-3e-5,
        dy=-1e-4,
        dpy=2e-5,
    )
    X, r, _co, nf = _sample(name, inj)
    p = r["inj"]
    u = X @ np.linalg.inv(nf.w).T
    J = 0.5 * (u[:, 0::2] ** 2 + u[:, 1::2] ** 2)
    want_x = p["emit_x"] * _bmag(bx, ax, p["beta_x"], p["alpha_x"]) + _cs_action(
        p["dx"], p["dpx"], bx, ax
    )
    want_y = p["emit_y"] * _bmag(by, ay, p["beta_y"], p["alpha_y"]) + _cs_action(
        p["dy"], p["dpy"], by, ay
    )
    eps = _eps(nf.w, p["emit_x"], p["emit_y"], p["sigma_delta"])
    assert J[:, 0].mean() == pytest.approx(want_x, rel=1e-12)
    assert J[:, 1].mean() == pytest.approx(want_y, rel=1e-12)
    assert J[:, 2].mean() == pytest.approx(eps[4], rel=1e-12)
    assert _bmag(bx, ax, p["beta_x"], p["alpha_x"]) > 1.2  # the mismatch is real


@pytest.mark.parametrize("name", ROLLED, ids=str)
def test_an_energy_offset_rides_the_dynamic_dispersion(name: str) -> None:
    """``delta0`` alone moves the centroid by ``NormalForm.dispersion * delta0`` (the
    response in phase with ``delta``, not the static ``D``), keeps ``<zeta> = 0`` and leaves
    the transverse modes' ``<J>`` at ``eps`` — it puts no betatron oscillation in."""
    d0 = 1e-3
    X, r, _co, nf = _sample(name, {"n": 500, "seed": 9, "delta0": d0})
    p = r["inj"]
    mean = X.mean(axis=0)
    D = np.asarray(nf.dispersion)
    assert np.max(np.abs(mean[:4] - D * d0)) < 1e-12 * abs(D[0] * d0)
    assert abs(mean[4]) < 1e-15 and mean[5] == pytest.approx(d0, rel=1e-12)
    u = X @ np.linalg.inv(nf.w).T
    assert (0.5 * (u[:, 0] ** 2 + u[:, 1] ** 2)).mean() == pytest.approx(p["emit_x"], rel=1e-12)
    assert abs(D[0] / r["optics"]["twiss4"]["disp_x"] - 1.0) > 0.1  # not the static D


# --- 3. tracking ------------------------------------------------------------------------------
TRACK_RINGS = {
    "wiggler-ring": PRESETS["wiggler-ring"],
    "electron-ring-6.5MV": _with_voltage(PRESETS["electron-ring"], 6.5e6),
}


def _tracked(sc: dict, sigma_delta: float, turns: int, sampler: str | None = None, n: int = 500):
    case = {
        "scenario": sc,
        "mode": "stats",
        "inj": {"n": n, "seed": 3, "sigma_delta": sigma_delta},
        "turns": turns,
    }
    if sampler:
        case["sampler"] = sampler
    (r,) = _node([case])
    return r


@pytest.mark.parametrize("name", list(TRACK_RINGS), ids=str)
def test_the_matched_beam_holds_and_the_old_one_was_mismatched(name: str) -> None:
    """1000 turns off resonance. The 6D-matched beam's per-turn mode emittance stays within
    ``1.5e-2`` of the injected value at ``sigma_delta = 7e-4`` and ``1e-3`` at ``3.5e-4``
    (measured ``7.1e-3`` / ``4.7e-4`` on ``wiggler-ring``, ``4.9e-3`` / ``3.3e-4`` at 6.5 MV),
    and the residual falls by at least 2 per halving: it is the nonlinearity the linear match
    leaves out, and it vanishes with the energy spread. Its owner is NOT localised (the RF
    sine, the chromatic beta-beat and resonance tails are candidates).

    The control is the 4D sampler's beam on the same ring, measured the same way: its
    ``<J_1>`` is the mismatch its own covariance predicts (``~1.21 eps``), and tracking keeps it
    there — the old beam was not matched, by an amount the closed form names.
    """
    sc = TRACK_RINGS[name]
    eps = sc["beam"]["emit_x"]
    resid = []
    for sd in (7e-4, 3.5e-4):
        st = _tracked(sc, sd, 1000)["stats"]
        emit = np.array([s["x"]["emit"] for s in st])
        assert emit[0] == pytest.approx(eps, rel=1e-12)
        resid.append(float(np.max(np.abs(emit / eps - 1.0))))
    assert resid[0] < 1.5e-2
    assert resid[1] < 1e-3
    assert resid[0] / resid[1] >= 2.0

    ctl = _tracked(sc, 7e-4, 1000, sampler="4d")
    # The closed form from the old sampler's construction: uncoupled Twiss, static D, the
    # matched bunch length, as a covariance; its mode-1 action through the package's W^-1.
    lat, _M, nf = _package_form(sc)
    t4 = ctl["optics"]["twiss4"]
    sd = 7e-4
    sz = (
        sd * abs(ctl["optics"]["eta"]) * ctl["optics"]["length"] / (2 * np.pi * ctl["optics"]["Qs"])
    )
    Sig = np.zeros((6, 6))
    for k, pl in ((0, "x"), (2, "y")):
        b, a = t4[f"beta_{pl}"], t4[f"alpha_{pl}"]
        e = sc["beam"][f"emit_{pl}"]
        Sig[k : k + 2, k : k + 2] = e * np.array([[b, -a], [-a, (1 + a * a) / b]])
    d = np.array([t4["disp_x"], t4["disp_px"], t4["disp_y"], t4["disp_py"], 0.0, 1.0])
    Sig += sd * sd * np.outer(d, d)
    Sig[4, 4] = sz * sz
    Wi = np.linalg.inv(nf.w)
    Su = Wi @ Sig @ Wi.T
    J_cf = 0.5 * (Su[0, 0] + Su[1, 1])
    J = np.array([s["x"]["meanJ"] for s in ctl["stats"]])
    assert J[0] == pytest.approx(J_cf, rel=1e-10)
    assert J_cf / eps > 1.15
    assert np.max(np.abs(J / J_cf - 1.0)) < 2e-2


# --- 4. the owner of electron-ring's growth ---------------------------------------------------
def _sideband_distance(optics: dict) -> float:
    q1, _q2, q3 = optics["normal"]["tunes"]
    v = q1 + 3 * min(q3, 1 - q3)
    return v - round(v)


@pytest.mark.parametrize(
    ("f", "target", "grows"),
    [(1e-4, 0.0017, True), (-1.2e-3, -0.0087, False), (8e-4, 0.0073, False)],
    ids=["on-sideband", "below", "above"],
)
def test_the_default_rings_growth_follows_the_third_sideband(
    f: float, target: float, grows: bool
) -> None:
    """At fixed voltage, only the focusing quadrupoles scaled: the 6D-matched beam's ``<J_1>``
    after 600 turns is above 1.3 where ``Q_1 + 3 Q_s - 6 = +0.0017`` and below 1.01 at
    ``-0.0087`` and ``+0.0073``. The growth follows the betatron tune through the sideband, so
    the sideband owns it. (Its peak sits 0.0017 above the linear condition: measured, not
    localised.)"""
    sc = _scaled_qf(PRESETS["electron-ring"], f)
    r = _tracked(sc, 7e-4, 600, n=400)
    assert _sideband_distance(r["optics"]) == pytest.approx(target, abs=2e-4)
    eps = sc["beam"]["emit_x"]
    J_end = r["stats"][-1]["x"]["meanJ"] / eps
    if grows:
        assert J_end > 1.3
    else:
        assert J_end < 1.01


def test_the_page_names_the_sideband_where_it_is_and_only_there() -> None:
    """``sidebandNear`` (the page's warning) finds ``Q_1 + 3 Q_s`` 0.0009 from an integer on
    ``electron-ring`` and nothing within ``5e-3`` on ``wiggler-ring``."""
    cases = [
        {"scenario": PRESETS[k], "mode": "inject", "inj": {"n": 8}}
        for k in ("electron-ring", "wiggler-ring")
    ]
    script = (
        "const I=require(process.argv[1]);let t='';process.stdin.on('data',c=>t+=c);"
        "process.stdin.on('end',()=>{const cs=JSON.parse(t);"
        "process.stdout.write(JSON.stringify(cs.map(c=>I.sidebandNear(I.ringOptics(c.scenario)))))});"
    )
    assert NODE is not None
    out = json.loads(
        subprocess.run(
            [NODE, "-e", script, str(EDITOR / "inject.js")],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    er, wr = out
    assert er["plane"] == "x" and er["m"] == 3
    assert er["distance"] == pytest.approx(0.00086, abs=5e-5)
    assert wr is None
