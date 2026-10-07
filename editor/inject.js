/*
 * inject.js — injecting a beam into the ring and watching it filament.
 *
 * The pure maths of the editor's injection view, with no page code in it:
 *
 *   - the injected beam: a Gaussian bunch with its OWN Twiss (beta, alpha), emittance,
 *     centroid offset and energy spread, placed on the ring's closed orbit and dispersion;
 *     drawn with exact moments ("moment-matched"), so its statistics are the requested
 *     ones to round-off rather than to 1/sqrt(N);
 *   - the turn-by-turn statistics at the injection point: the centroid (as a complex
 *     normalised amplitude, so it does not oscillate), the rms betatron emittance, and
 *     the mean Courant-Snyder action <J> measured with the RING's Twiss;
 *   - the closed forms the tracked beam is held against (and the page overlays):
 *       mismatch     <J> = eps_inj * Bmag,   Bmag = (beta gamma_i - 2 alpha alpha_i + gamma beta_i)/2
 *       offset       <J> += J(dx, dpx)  (the centroid's own action)
 *       filamented   eps -> <J>          (every particle keeps its J, the phases spread)
 *       decoherence  |centroid_n| / |centroid_0| = |(1/N) sum_p exp(i 2pi Q' delta_p n)|
 *                    (frozen delta, linear chromaticity — the sample's own, not a Gaussian)
 *
 * It adds no physics of its own: the motion is accsim-track.js (a port of the package's
 * element-by-element tracker), the optics are accsim-optics.js. Radiation is OFF — an
 * electron beam here filaments but never damps — and the page says so.
 *
 * Coordinates: (x, px, y, py, zeta, delta). Dependency-free beyond the two editor cores.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./accsim-optics.js"), require("./animate.js"));
  } else root.AccsimInject = factory(root.AccsimOptics, root.AccsimAnim);
})(typeof self !== "undefined" ? self : this, function (O, A) {
  "use strict";
  const X = 0, PX = 1, Y = 2, PY = 3, ZETA = 4, DELTA = 5;
  const TWO_PI = 2.0 * Math.PI;

  /**
   * The ring's optics at the injection point (the start of the lattice), from the editor's
   * own analysis — the same arrays the page draws. Throws if the ring does not close.
   */
  function ringOptics(scenario) {
    if (scenario.periodic === false) throw new Error("injection needs a ring (periodic) scenario");
    const a = O.analyse(scenario, { slicesFor: () => 1 });
    if (a.errors.length) throw new Error(a.errors[0]);
    if (a.opticsError) throw new Error(a.opticsError.message);
    if (a.coupled) throw new Error("the ring is x-y coupled; injection uses uncoupled Twiss");
    const tw = a.twiss0;
    return {
      ref: a.ref,
      length: a.length,
      tunes: a.tunes,
      chromaticity: a.chromaticity || null,
      Qs: a.Qs == null ? null : a.Qs,
      eta: a.eta == null ? null : a.eta,
      hasRF: (scenario.elements || []).some((e) => e.type === "RFCavity" && +e.voltage !== 0.0),
      orbit0: a.closedOrbit0 || [0.0, 0.0, 0.0, 0.0],
      twiss: {
        beta_x: tw.beta_x, alpha_x: tw.alpha_x, beta_y: tw.beta_y, alpha_y: tw.alpha_y,
        disp_x: tw.disp_x || 0.0, disp_px: tw.disp_px || 0.0,
        disp_y: tw.disp_y || 0.0, disp_py: tw.disp_py || 0.0,
      },
    };
  }

  // ------------------------------------------------------------ moment matching
  /** Cholesky factor (lower) of a symmetric positive-definite k x k matrix. */
  function cholesky(S) {
    const k = S.length, L = S.map(() => new Array(k).fill(0.0));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = S[i][j];
        for (let m = 0; m < j; m++) sum -= L[i][m] * L[j][m];
        if (i === j) {
          if (!(sum > 0.0)) throw new Error("moment matching: sample covariance is singular");
          L[i][i] = Math.sqrt(sum);
        } else L[i][j] = sum / L[j][j];
      }
    }
    return L;
  }
  /**
   * Whiten a sample in place: subtract its mean and solve by the Cholesky factor of its
   * (1/N) covariance, so the result has mean exactly 0 and covariance exactly I. This is
   * what makes the injected beam's emittance, offset and energy spread the requested ones
   * to round-off — the closed forms then gate the sampler, not the luck of the draw.
   */
  function whiten(rows) {
    const n = rows.length, k = rows[0].length;
    const mean = new Array(k).fill(0.0);
    for (const r of rows) for (let j = 0; j < k; j++) mean[j] += r[j] / n;
    for (const r of rows) for (let j = 0; j < k; j++) r[j] -= mean[j];
    const S = mean.map(() => new Array(k).fill(0.0));
    for (const r of rows) for (let i = 0; i < k; i++) for (let j = 0; j <= i; j++) S[i][j] += r[i] * r[j] / n;
    for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) S[j][i] = S[i][j];
    const L = cholesky(S);
    for (const r of rows) {
      // forward substitution: L z = r
      for (let i = 0; i < k; i++) {
        let v = r[i];
        for (let m = 0; m < i; m++) v -= L[i][m] * r[m];
        r[i] = v / L[i][i];
      }
    }
    return rows;
  }

  // ------------------------------------------------------------- the injected beam
  /**
   * The injection settings, filled in from the scenario's beam record and the ring's own
   * matched optics. A beam that is not told otherwise is MATCHED and centred: injecting it
   * changes nothing, which is the control.
   */
  function defaults(scenario, optics) {
    const b = scenario.beam || {};
    const t = optics.twiss;
    return {
      n: 1000, seed: 1,
      emit_x: +b.emit_x || 0.0, emit_y: +b.emit_y || 0.0, sigma_delta: +b.sigma_delta || 0.0,
      beta_x: t.beta_x, alpha_x: t.alpha_x, beta_y: t.beta_y, alpha_y: t.alpha_y,
      dx: 0.0, dpx: 0.0, dy: 0.0, dpy: 0.0, delta0: 0.0,
    };
  }
  /** The matched rms bunch length sigma_zeta = sigma_delta |eta| C / (2 pi Q_s), or 0. */
  function matchedBunchLength(optics, sigmaDelta) {
    if (!optics.hasRF || !(optics.Qs > 0) || optics.eta == null) return 0.0;
    return sigmaDelta * Math.abs(optics.eta) * optics.length / (TWO_PI * optics.Qs);
  }
  /**
   * Draw the injected bunch: `inj.n` particles as 6-arrays. Each transverse plane is
   *     x  = x_co + D delta + sqrt(eps beta_i) a + dx
   *     px = px_co + D' delta + sqrt(eps / beta_i) (b - alpha_i a) + dpx
   * with (a, b, ...) whitened unit normals, delta = delta0 + sigma_delta g, and zeta drawn at
   * the matched bunch length when the ring has RF (else 0 — without RF zeta only grows).
   */
  function sampleInjection(inj, optics) {
    const rng = A.mulberry32(inj.seed || 1);
    const sigZ = matchedBunchLength(optics, inj.sigma_delta);
    const k = sigZ > 0.0 ? 6 : 5;
    const rows = [];
    for (let p = 0; p < inj.n; p++) {
      const r = [];
      for (let j = 0; j < k; j++) r.push(A.gaussian(rng));
      rows.push(r);
    }
    whiten(rows);
    const t = optics.twiss, o = optics.orbit0;
    const sx = Math.sqrt(inj.emit_x * inj.beta_x), spx = Math.sqrt(inj.emit_x / inj.beta_x);
    const sy = Math.sqrt(inj.emit_y * inj.beta_y), spy = Math.sqrt(inj.emit_y / inj.beta_y);
    return rows.map((r) => {
      const delta = inj.delta0 + inj.sigma_delta * r[4];
      return [
        o[0] + t.disp_x * delta + sx * r[0] + inj.dx,
        o[1] + t.disp_px * delta + spx * (r[1] - inj.alpha_x * r[0]) + inj.dpx,
        o[2] + t.disp_y * delta + sy * r[2] + inj.dy,
        o[3] + t.disp_py * delta + spy * (r[3] - inj.alpha_y * r[2]) + inj.dpy,
        k === 6 ? sigZ * r[5] : 0.0,
        delta,
      ];
    });
  }

  // ------------------------------------------------------------------ statistics
  /** The betatron part of one particle in plane `pl` ("x" or "y"): the closed orbit and
   *  D delta removed, with the ring's dispersion at the injection point (never a fit). */
  function betatronPart(s, optics, pl) {
    const t = optics.twiss, o = optics.orbit0;
    if (pl === "x") return [s[X] - o[0] - t.disp_x * s[DELTA], s[PX] - o[1] - t.disp_px * s[DELTA]];
    return [s[Y] - o[2] - t.disp_y * s[DELTA], s[PY] - o[3] - t.disp_py * s[DELTA]];
  }
  /** Courant-Snyder action J = (gamma u^2 + 2 alpha u u' + beta u'^2)/2 for Twiss (beta, alpha). */
  function action(u, up, beta, alpha) {
    const gamma = (1.0 + alpha * alpha) / beta;
    return 0.5 * (gamma * u * u + 2.0 * alpha * u * up + beta * up * up);
  }
  /** Bmag = (beta gamma_i - 2 alpha alpha_i + gamma beta_i) / 2  (>= 1, = 1 iff matched). */
  function bmag(beta, alpha, betaI, alphaI) {
    const gamma = (1.0 + alpha * alpha) / beta, gammaI = (1.0 + alphaI * alphaI) / betaI;
    return 0.5 * (beta * gammaI - 2.0 * alpha * alphaI + gamma * betaI);
  }
  /** The closed-form <J> of the injected beam in one plane: eps Bmag + J(offset). */
  function expectedAction(inj, optics, pl) {
    const t = optics.twiss;
    if (pl === "x") {
      return inj.emit_x * bmag(t.beta_x, t.alpha_x, inj.beta_x, inj.alpha_x) +
        action(inj.dx, inj.dpx, t.beta_x, t.alpha_x);
    }
    return inj.emit_y * bmag(t.beta_y, t.alpha_y, inj.beta_y, inj.alpha_y) +
      action(inj.dy, inj.dpy, t.beta_y, t.alpha_y);
  }
  /**
   * Statistics of the surviving particles at the injection point, both planes:
   *   centroid  — |<z>| with z = (u - i (alpha u + beta u'))/sqrt(beta): the centroid's
   *               normalised amplitude sqrt(2 J_c), which does not oscillate turn to turn;
   *   emit      — rms betatron emittance sqrt(<u^2><u'^2> - <u u'>^2), central moments;
   *   meanJ     — <J> with the RING's Twiss (conserved by linear motion);
   *   m2        — the central second moments [<uu>, <uu'>, <u'u'>];
   *   m2ref     — the same about the closed orbit (u = 0), NOT about the centroid: these are
   *               the ones to average over turns. A finite bunch's centroid never settles to
   *               zero (|<z>|^2 ~ <2J>/N), so moments about it sit low by 1/N on average — a
   *               bias that more turns cannot remove; about the orbit they average to <J>.
   */
  function beamStats(states, lost, optics) {
    const t = optics.twiss;
    const out = { alive: 0 };
    for (const pl of ["x", "y"]) {
      const beta = pl === "x" ? t.beta_x : t.beta_y, alpha = pl === "x" ? t.alpha_x : t.alpha_y;
      let n = 0, mu = 0.0, mup = 0.0, J = 0.0;
      const us = [], ups = [];
      for (let p = 0; p < states.length; p++) {
        if (lost && lost[p] >= 0) continue;
        const [u, up] = betatronPart(states[p], optics, pl);
        us.push(u); ups.push(up);
        mu += u; mup += up; J += action(u, up, beta, alpha); n++;
      }
      if (n === 0) { out[pl] = null; continue; }
      mu /= n; mup /= n; J /= n;
      let suu = 0.0, sup = 0.0, spp = 0.0;
      for (let i = 0; i < n; i++) {
        const du = us[i] - mu, dp = ups[i] - mup;
        suu += du * du; sup += du * dp; spp += dp * dp;
      }
      suu /= n; sup /= n; spp /= n;
      const re = mu / Math.sqrt(beta), im = -(alpha * mu + beta * mup) / Math.sqrt(beta);
      out[pl] = {
        mean: [mu, mup], centroid: Math.hypot(re, im),
        emit: Math.sqrt(Math.max(0.0, suu * spp - sup * sup)), meanJ: J, m2: [suu, sup, spp],
        m2ref: [suu + mu * mu, sup + mu * mup, spp + mup * mup],
      };
      out.alive = n;
    }
    return out;
  }
  /**
   * The decoherence envelope for a FROZEN energy distribution and linear chromaticity:
   * |(1/N) sum_p exp(i 2 pi Qp delta_p n)|, from the sample's own deltas — so it carries no
   * Gaussian assumption and no 1/sqrt(N) noise. Blind to the sign of Qp (a modulus).
   */
  function chromaticEnvelope(deltas, Qp, n) {
    let re = 0.0, im = 0.0;
    for (const d of deltas) { const ph = TWO_PI * Qp * d * n; re += Math.cos(ph); im += Math.sin(ph); }
    return Math.hypot(re, im) / deltas.length;
  }

  return {
    ringOptics, defaults, sampleInjection, matchedBunchLength, whiten, cholesky,
    betatronPart, action, bmag, expectedAction, beamStats, chromaticEnvelope,
  };
});
