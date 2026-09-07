/*
 * Node harness for the cross-check test: runs every bundled preset (or the scenario
 * JSON on stdin) through accsim-optics.js and prints one JSON document with every
 * number the Python side will hold it against. Not used by the browser page.
 *
 *   node editor/selftest.js            # all presets
 *   node editor/selftest.js - < s.json # one scenario from stdin
 */
"use strict";
const path = require("path");
const O = require(path.join(__dirname, "accsim-optics.js"));

function run(scenario) {
  const out = { name: scenario.name, id: scenario.id, scenario };
  const a = O.analyse(scenario, { slicesFor: () => 1 });
  out.errors = a.errors;
  if (a.errors.length) return out;
  out.ref = a.ref;
  out.elements = a.elements;
  out.length = a.length;
  out.totalAngle = a.totalAngle;
  out.matrices = a.matrices;
  out.kicks = a.kicks;
  out.oneTurn = a.oneTurn;
  out.oneTurnKick = a.oneTurnKick;
  out.survey = a.surveyBoundaries;
  out.coupled = a.coupled;
  out.periodic = a.periodic;
  out.opticsError = a.opticsError || null;
  out.twiss0 = a.twiss0 || null;
  out.twissBoundaries = a.twissBoundaries || null;
  out.tunes = a.tunes || null;
  out.chromaticity = a.chromaticity || null;
  out.naturalChromaticity = a.naturalChromaticity || null;
  out.alpha_c = a.alpha_c == null ? null : a.alpha_c;
  out.Qs = a.Qs == null ? null : a.Qs;
  out.closedOrbit0 = a.closedOrbit0 || null;
  out.orbit = a.orbit ? a.orbit.map((p) => p.o) : null;
  return out;
}

function main() {
  const arg = process.argv[2];
  if (arg === "-") {
    let text = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (c) => { text += c; });
    process.stdin.on("end", () => {
      process.stdout.write(JSON.stringify(run(JSON.parse(text))));
    });
    return;
  }
  const presets = require(path.join(__dirname, "presets.js"));
  process.stdout.write(JSON.stringify(presets.map(run)));
}
main();
