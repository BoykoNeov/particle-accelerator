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
 *                                         (all six coordinates with `full: true`).
 *
 * On a ring with RF, `sampler: "4d"` draws the beam the way the tab did before the 6D match
 * (uncoupled Twiss + static dispersion) and still measures it in the 6D normal modes — the
 * control for tests/analytic/test_injection_6d.py.
 */
"use strict";
const path = require("path");
const T = require(path.join(__dirname, "accsim-track.js"));
const I = require(path.join(__dirname, "inject.js"));

function run(c) {
  try {
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
    const optics = I.ringOptics(c.scenario);
    const inj = Object.assign(I.defaults(c.scenario, optics), c.inj || {});
    let states;
    if (c.sampler === "4d") {
      if (!optics.normal) throw new Error("sampler '4d' is the control for a ring with a 6D normal form");
      const o4 = Object.assign({}, optics, { normal: null, twiss: optics.twiss4 });
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
