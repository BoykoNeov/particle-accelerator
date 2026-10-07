/*
 * Node harness for tests/analytic/test_tracking_port.py: runs particles through
 * accsim-track.js and prints the result, for the Python side to hold against
 * accsim's own element-by-element tracker. Not used by the browser page.
 *
 *   node editor/track-selftest.js < cases.json
 *
 * Input: a JSON list of cases, each
 *   { scenario, states: [[x, px, y, py, zeta, delta], ...], turns, mode }
 * mode "elements": one pass, the state after EVERY element (localises a disagreement);
 * mode "turns":    the state after every turn, apertures ignored (the pure maps);
 * mode "losses":   track with apertures; returns the lost turn and element per particle.
 * Output: a JSON list, one entry per case: { error } or { states } / { lostTurn, lostAt, states }.
 */
"use strict";
const path = require("path");
const T = require(path.join(__dirname, "accsim-track.js"));

function run(c) {
  let m;
  try { m = T.build(c.scenario); } catch (e) { return { error: e.message }; }
  const states = c.states.map((s) => s.slice());
  if (c.mode === "elements") {
    const after = states.map(() => []);
    for (let i = 0; i < m.steps.length; i++) {
      states.forEach((s, p) => { m.steps[i](s); after[p].push(s.slice()); });
    }
    return { states: after };
  }
  if (c.mode === "turns") {
    const after = states.map(() => []);
    for (let t = 0; t < c.turns; t++) {
      states.forEach((s, p) => { for (const step of m.steps) step(s); after[p].push(s.slice()); });
    }
    return { states: after };
  }
  // losses
  const lostTurn = new Int32Array(states.length).fill(-1);
  const lostAt = new Int32Array(states.length).fill(-1);
  for (let t = 0; t < c.turns; t++) {
    states.forEach((s, p) => {
      if (lostTurn[p] >= 0) return;
      const at = T.trackTurn(m, s);
      if (at >= 0) { lostTurn[p] = t; lostAt[p] = at; }
    });
  }
  return { lostTurn: Array.from(lostTurn), lostAt: Array.from(lostAt), states };
}

let text = "";
process.stdin.setEncoding("utf-8");
process.stdin.on("data", (chunk) => { text += chunk; });
process.stdin.on("end", () => {
  const cases = JSON.parse(text);
  process.stdout.write(JSON.stringify(cases.map(run)));
});
