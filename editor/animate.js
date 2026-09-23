/*
 * animate.js — the *kinematics* of the accsim editor's beam animation.
 *
 * This file holds only the pure maths of "where is the beam right now":
 *
 *   - a seeded beam of particles, in normalised units (independent of the machine),
 *   - the closed-form betatron displacement of a particle at a Twiss sample,
 *   - the dispersive offset for an off-momentum particle,
 *   - the lab-frame offset of a transverse displacement on the survey (floor plan),
 *   - small interpolation helpers that read the sampled Twiss / orbit / survey at any s.
 *
 * It adds **no physics** beyond `accsim-optics.js`: the betatron motion is the exact
 * closed form of the machine's own Courant-Snyder optics,
 *
 *     x(s)  = sqrt(2 J_x beta_x(s)) cos(mu_x(s) + phi0)          (+ D_x(s) delta)
 *     px(s) = -sqrt(2 J_x / beta_x(s)) [sin(...) + alpha_x(s) cos(...)]
 *
 * evaluated from the Twiss functions already computed and cross-checked to 1e-9.
 * `betatronConsistency` is the discriminating gate: it tracks a particle through the
 * element matrices independently and holds the two against each other, so a wrong
 * sign, a flipped alpha, or a missing turn-phase term shows up (the plain
 * Courant-Snyder invariant cannot see any of those — it is an identity here).
 *
 * Coordinates: (x, px, y, py) in the Xsuite / MAD-X ordering. delta is frozen.
 *
 * Dependency-free; runs in a browser (window.AccsimAnim) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AccsimAnim = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const TWO_PI = 2.0 * Math.PI;

  // ---------------------------------------------------------------- seeded RNG
  /** mulberry32 — a tiny deterministic PRNG so the same seed gives the same beam,
   * and recomputing the machine never reshuffles the particles on screen. */
  function mulberry32(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /** One standard normal draw (Box-Muller) from a uniform generator. */
  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(TWO_PI * v);
  }

  /**
   * Sample `n` particles as a matched beam, in *normalised* units:
   *   nJx = J_x / emit_x  ~ Exp(mean 1)   (so <2 J_x beta> = emit_x beta = sigma^2),
   *   phix ~ U(0, 2pi),   likewise the y plane,
   *   ndelta ~ N(0, 1)    (multiply by sigma_delta for the momentum offset).
   * The units are deliberately machine-free: the same particle rides any lattice.
   */
  function sampleBeam(n, seed) {
    const rng = mulberry32(seed);
    const out = [];
    for (let i = 0; i < n; i++) {
      const ux = rng(), uy = rng();
      out.push({
        nJx: -Math.log(ux > 0 ? ux : 1e-12),
        phix: TWO_PI * rng(),
        nJy: -Math.log(uy > 0 ? uy : 1e-12),
        phiy: TWO_PI * rng(),
        ndelta: gaussian(rng),
      });
    }
    return out;
  }

  // ------------------------------------------------------- betatron kinematics
  /**
   * The betatron displacement (x, px, y, py) of one particle at a Twiss sample `tw`,
   * given the beam emittances and the accumulated whole-turn phase `turnPhaseX/Y`
   * (n * 2pi Q per lap). Betatron part only — add the closed orbit and D*delta outside.
   */
  function betatron(tw, part, beam, turnPhaseX, turnPhaseY) {
    const jx2 = 2.0 * beam.emit_x * part.nJx;
    const ampx = Math.sqrt(jx2 * tw.beta_x);
    const slopex = Math.sqrt(jx2 / tw.beta_x);
    const phx = tw.mu_x + turnPhaseX + part.phix;
    const jy2 = 2.0 * beam.emit_y * part.nJy;
    const ampy = Math.sqrt(jy2 * tw.beta_y);
    const slopey = Math.sqrt(jy2 / tw.beta_y);
    const phy = tw.mu_y + turnPhaseY + part.phiy;
    return {
      x: ampx * Math.cos(phx),
      px: -slopex * (Math.sin(phx) + tw.alpha_x * Math.cos(phx)),
      y: ampy * Math.cos(phy),
      py: -slopey * (Math.sin(phy) + tw.alpha_y * Math.cos(phy)),
    };
  }
  /** The dispersive offset D(s)*delta for a given momentum offset. */
  function dispersionOffset(tw, delta) {
    return {
      x: (tw.disp_x || 0) * delta, px: (tw.disp_px || 0) * delta,
      y: (tw.disp_y || 0) * delta, py: (tw.disp_py || 0) * delta,
    };
  }
  /**
   * Lab-frame offset of a transverse horizontal displacement `xOff` at survey heading
   * `theta`. This is the same yaw rotation W(theta) that accsim.geometry.survey applies
   * to a step vector: X gets +cos(theta)*xloc, Z gets -sin(theta)*xloc.
   */
  function floorOffset(theta, xOff) {
    return [Math.cos(theta) * xOff, -Math.sin(theta) * xOff];
  }

  // --------------------------------------------------------- sampled-array reads
  /** Bracket `s` in an ascending numeric array; returns {i0, i1, f} for lerp. */
  function locate(sArr, s) {
    const n = sArr.length;
    if (n === 0) return { i0: 0, i1: 0, f: 0 };
    if (s <= sArr[0]) return { i0: 0, i1: 0, f: 0 };
    if (s >= sArr[n - 1]) return { i0: n - 1, i1: n - 1, f: 0 };
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (sArr[m] < s) lo = m; else hi = m; }
    const span = sArr[hi] - sArr[lo];
    return { i0: lo, i1: hi, f: span > 0 ? (s - sArr[lo]) / span : 0 };
  }
  function lerp(a, b, f) { return a + (b - a) * f; }

  const TW_KEYS = ["beta_x", "alpha_x", "mu_x", "beta_y", "alpha_y", "mu_y",
    "disp_x", "disp_px", "disp_y", "disp_py"];
  /** Interpolate a Twiss sample at `s` from a sampled `pts` array (each carries `.s`). */
  function twissAt(pts, sArr, s) {
    const { i0, i1, f } = locate(sArr, s);
    const a = pts[i0], b = pts[i1], out = { s };
    for (const k of TW_KEYS) out[k] = lerp(a[k] || 0, b[k] || 0, f);
    return out;
  }
  /** Interpolate the 4D closed orbit vector `o` at `s` from a sampled orbit array. */
  function orbitAt(pts, sArr, s) {
    const { i0, i1, f } = locate(sArr, s);
    const a = pts[i0].o, b = pts[i1].o;
    return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f), lerp(a[3], b[3], f)];
  }
  /** Interpolate the survey pose (X, Z, theta) at `s`. */
  function surveyAt(sv, s) {
    const { i0, i1, f } = locate(sv.s, s);
    return {
      X: lerp(sv.X[i0], sv.X[i1], f),
      Z: lerp(sv.Z[i0], sv.Z[i1], f),
      theta: lerp(sv.theta[i0], sv.theta[i1], f),
    };
  }

  // -------------------------------------------------------------- the gate
  /** The transverse 4x4 block of a 6x6 element matrix. */
  function transverse4(M) {
    const T = [];
    for (let i = 0; i < 4; i++) { T.push([M[i][0], M[i][1], M[i][2], M[i][3]]); }
    return T;
  }
  function matvec4(T, v) {
    return [
      T[0][0] * v[0] + T[0][1] * v[1] + T[0][2] * v[2] + T[0][3] * v[3],
      T[1][0] * v[0] + T[1][1] * v[1] + T[1][2] * v[2] + T[1][3] * v[3],
      T[2][0] * v[0] + T[2][1] * v[1] + T[2][2] * v[2] + T[2][3] * v[3],
      T[3][0] * v[0] + T[3][1] * v[1] + T[3][2] * v[2] + T[3][3] * v[3],
    ];
  }
  /**
   * Discriminating consistency gate. Build (x, px, y, py) at boundary 0 from the closed
   * form, then push it through each element's transverse 4x4 matrix for `turns` laps,
   * comparing against the closed form (with the n*2pi Q turn phase) at every element
   * boundary. Returns the largest absolute deviation across all coordinates/boundaries.
   *
   * This is NOT the Courant-Snyder invariant (which is an identity for coordinates built
   * from the same formula): the two sides here propagate independently — matrix product
   * vs. Twiss functions — so a sign error, a flipped alpha, or a wrong turn term fails it.
   * Requires delta = 0 (frozen) and an uncoupled, closing lattice.
   */
  function betatronConsistency(twissBoundaries, matrices, part, beam, turns) {
    const last = twissBoundaries[twissBoundaries.length - 1];
    const dPhiX = last.mu_x, dPhiY = last.mu_y;
    const b0 = betatron(twissBoundaries[0], part, beam, 0, 0);
    let v = [b0.x, b0.px, b0.y, b0.py];
    let maxdev = 0;
    const N = matrices.length;
    for (let turn = 0; turn < turns; turn++) {
      for (let i = 0; i < N; i++) {
        v = matvec4(transverse4(matrices[i]), v);
        const f = betatron(twissBoundaries[i + 1], part, beam, turn * dPhiX, turn * dPhiY);
        maxdev = Math.max(maxdev,
          Math.abs(v[0] - f.x), Math.abs(v[1] - f.px),
          Math.abs(v[2] - f.y), Math.abs(v[3] - f.py));
      }
    }
    return maxdev;
  }

  return {
    mulberry32, gaussian, sampleBeam,
    betatron, dispersionOffset, floorOffset,
    locate, lerp, twissAt, orbitAt, surveyAt,
    transverse4, matvec4, betatronConsistency,
  };
});
