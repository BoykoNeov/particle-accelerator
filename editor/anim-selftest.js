/*
 * Node harness for the animation's betatron gate. For every bundled preset that has a
 * closing, uncoupled periodic solution, it reconstructs a small matched beam with the
 * closed-form betatron motion and tracks it independently through the element matrices
 * for a few turns, reporting the largest disagreement. `tests/analytic/test_scenario.py`
 * runs this under Node and asserts the deviation is at the round-off floor — a wrong
 * sign, a flipped alpha or a missing turn-phase term would blow it up. Browser page
 * does not use this file.
 *
 *   node editor/anim-selftest.js
 */
"use strict";
const path = require("path");
const O = require(path.join(__dirname, "accsim-optics.js"));
const A = require(path.join(__dirname, "animate.js"));
const presets = require(path.join(__dirname, "presets.js"));

const TURNS = 3;
const beam = { emit_x: 1e-8, emit_y: 1e-8 }; // geometric; delta is frozen at 0 here

function run(scenario) {
  const out = { id: scenario.id, name: scenario.name };
  let a;
  try {
    // Same sampling step as the page (index.html recompute()), so the sampled `twiss` / `orbit`
    // arrays checked below are the ones the animation actually reads.
    const L = scenario.elements.reduce((s, e) => s + O.elementLength(e), 0);
    a = O.analyse(scenario, { plotStep: Math.max(0.02, L / 900) });
  } catch (e) {
    out.skipped = "analyse threw: " + e.message;
    return out;
  }
  if (a.errors.length || a.opticsError || !a.twissBoundaries || !a.periodic || !a.matrices) {
    out.skipped = a.opticsError ? a.opticsError.code : (a.errors[0] || "no periodic twiss");
    return out;
  }
  const parts = A.sampleBeam(24, 12345);
  let maxdev = 0;
  for (const p of parts) {
    maxdev = Math.max(maxdev, A.betatronConsistency(a.twissBoundaries, a.matrices, p, beam, TURNS));
  }
  out.maxdev = maxdev;

  // The gate above reads `twissBoundaries`; the page reads the *sampled* `twiss` and takes its
  // turn phase from the last sample. Hold the two together: the samples must reach s = L and end
  // on the boundary phase (else the lap-to-lap phase on screen is wrong while the gate still
  // passes), and mu must never decrease (a wrapped mu is invisible to cos() at the boundaries but
  // makes the linear interpolation in twissAt jump backwards between samples).
  const tw = a.twiss, last = tw[tw.length - 1], lastB = a.twissBoundaries[a.twissBoundaries.length - 1];
  out.sampleEndS = Math.abs(last.s - a.length) / a.length;
  out.sampleEndMu = Math.max(Math.abs(last.mu_x - lastB.mu_x), Math.abs(last.mu_y - lastB.mu_y));
  out.muMonotone = tw.every((p, k) => k === 0 || (p.mu_x >= tw[k - 1].mu_x && p.mu_y >= tw[k - 1].mu_y));

  // What each graph panel draws, through the same `panelCoord` the page calls. The Beam-size
  // panel's ±sigma band is the rms size about the closed orbit, so its particles must not move
  // with the orbit; the Orbit panel's must move by exactly the orbit. Swapping the two branches
  // of panelCoord fails both (sigmaSeesOrbit becomes the orbit, orbitPanelMiss becomes it too).
  if (a.orbit) {
    let sigmaSeesOrbit = 0, orbitPanelMiss = 0;
    const orbS = a.orbit.map((q) => q.s);
    for (let k = 0; k < tw.length; k += 7) {
      const o = A.orbitAt(a.orbit, orbS, tw[k].s);
      for (const p of parts.slice(0, 4)) {
        const w = A.particleOffset(tw[k], o, p, beam, 1e-3, 0, 0);
        const z = A.particleOffset(tw[k], [0, 0, 0, 0], p, beam, 1e-3, 0, 0);
        const sw = A.panelCoord(w, "sigma"), sz = A.panelCoord(z, "sigma");
        const ow = A.panelCoord(w, "orbit"), oz = A.panelCoord(z, "orbit");
        sigmaSeesOrbit = Math.max(sigmaSeesOrbit, Math.abs(sw[0] - sz[0]), Math.abs(sw[1] - sz[1]));
        orbitPanelMiss = Math.max(orbitPanelMiss,
          Math.abs(ow[0] - oz[0] - o[0]), Math.abs(ow[1] - oz[1] - o[2]));
      }
    }
    out.sigmaSeesOrbit = sigmaSeesOrbit;
    out.orbitPanelMiss = orbitPanelMiss;
    out.orbitMax = Math.max(...a.orbit.map((q) => Math.abs(q.o[0])));
  }

  // Coefficient-free floor-offset sign: a positive (outward) transverse displacement must point
  // away from the ring centroid, i.e. sign(totalAngle) · offset·(P − centroid) > 0. This is the
  // physics that makes a +delta particle (which sits at +D·delta) land outside the ring. It needs
  // no number, only the survey and the sign of the total bend.
  if (a.periodic && a.totalAngle) {
    const sv = a.surveyBoundaries, n = sv.X.length;
    let cx = 0, cz = 0;
    for (let k = 0; k < n; k++) { cx += sv.X[k]; cz += sv.Z[k]; }
    cx /= n; cz /= n;
    let acc = 0;
    for (let k = 1; k < n; k++) {
      const off = A.floorOffset(sv.theta[k], 1);
      acc += off[0] * (sv.X[k] - cx) + off[1] * (sv.Z[k] - cz);
    }
    out.floorOutward = Math.sign(a.totalAngle) * acc;
  }
  return out;
}

process.stdout.write(JSON.stringify(presets.map(run)));
