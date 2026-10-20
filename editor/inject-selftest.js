/*
 * Node harness for tests/analytic/test_injection.py: injects a beam with inject.js,
 * tracks it with accsim-track.js and prints what the Python side holds against the
 * closed forms and the package. Not used by the browser page.
 *
 *   node editor/inject-selftest.js < cases.json
 *
 * Each case is { scenario, mode, ... }:
 *   mode "inject": { inj }            -> the ring optics, the settings and the drawn states;
 *   mode "stats":  { inj, turns }     -> the injected beam tracked, beamStats after each turn
 *                                         (index 0 = before the first turn), and the deltas;
 *   mode "tbt":    { states, turns }  -> turn-by-turn (x, px, y, py) of the given particles
 *                                         (all six coordinates with `full: true`);
 *   mode "design": {}                 -> the design one-turn matrix, matched Twiss and affine 4D
 *                                         orbit the editor's analysis uses;
 *   mode "nf6":    { matrix }         -> normalForm6 of the given 6x6 matrix (no scenario);
 *   mode "nf4":    { matrix }         -> normalForm4 of the given 4x4 matrix (no scenario).
 *
 * On a ring with RF, `sampler: "4d"` draws the beam the way the tab did before the 6D match
 * (uncoupled Twiss + static dispersion, about the 4D orbit at zeta = delta = 0) and still
 * measures it in the 6D normal modes, about the 6D orbit — the control for
 * tests/analytic/test_injection_6d.py.
 *
 * On a ring without RF whose orbit is displaced, `sampler: "design"` draws AND reads the beam
 * with the design optics (uncoupled Twiss, static dispersion of the design matrix) about the
 * affine 4D orbit — the tab before it took its optics on the tracked orbit; the control for
 * tests/analytic/test_injection_4d_orbit.py.
 */
"use strict";
const path = require("path");
const T = require(path.join(__dirname, "accsim-track.js"));
const I = require(path.join(__dirname, "inject.js"));
const O = require(path.join(__dirname, "accsim-optics.js"));

function run(c) {
  try {
    if (c.mode === "nf6") return { normal: O.normalForm6(c.matrix) };
    if (c.mode === "nf4") return { normal: O.normalForm4(c.matrix) };
    if (c.mode === "design") {
      const a = O.analyse(c.scenario, { slicesFor: () => 1 });
      return { oneTurn: a.oneTurn, twiss0: a.twiss0 || null, closedOrbit0: a.closedOrbit0 || null };
    }
    if (c.mode === "tbt") {
      const m = T.build(c.scenario);
      return {
        tbt: c.states.map((s0) => {
          const s = s0.slice(), rows = [];
          for (let t = 0; t < c.turns; t++) { T.trackTurn(m, s); rows.push(c.full ? s.slice() : [s[0], s[1], s[2], s[3]]); }
          return rows;
        }),
      };
    }
    let optics = I.ringOptics(c.scenario);
    if (c.sampler === "design") {
      if (!optics.normal4 || optics.coupled) throw new Error("sampler 'design' is the control for an uncoupled ring without RF on a displaced orbit");
      optics = Object.assign({}, optics, { normal4: null, twiss: optics.twiss4, orbit0: optics.orbit4 });
    }
    const inj = Object.assign(I.defaults(c.scenario, optics), c.inj || {});
    let states;
    if (c.sampler === "4d") {
      if (!optics.normal || optics.coupled) throw new Error("sampler '4d' is the control for an uncoupled ring with a 6D normal form");
      const o4 = Object.assign({}, optics, { normal: null, twiss: optics.twiss4, orbit0: optics.orbit4 });
      const inj4 = Object.assign(I.defaults(c.scenario, o4), c.inj || {});
      states = I.sampleInjection(inj4, o4);
    } else states = I.sampleInjection(inj, optics);
    const out = { optics, inj, initial: states.map((s) => s.slice()) };
    if (c.mode === "inject") return out;
    const m = T.build(c.scenario);
    const lost = new Int32Array(states.length).fill(-1);
    const stats = [I.beamStats(states, lost, optics)];
    T.trackBunch(m, states, lost, c.turns, () => stats.push(I.beamStats(states, lost, optics)));
    out.stats = stats;
    out.deltas = states.map((s) => s[5]);
    out.final = states;
    return out;
  } catch (e) {
    return { error: e.message };
  }
}

let text = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => { text += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify(JSON.parse(text).map(run)));
});
