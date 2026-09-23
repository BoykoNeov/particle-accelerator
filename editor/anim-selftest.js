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
    a = O.analyse(scenario);
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
