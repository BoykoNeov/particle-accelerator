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
 * On a ring with an RF cavity all of the above happens in the ring's 6D NORMAL MODES (the
 * normal form W, accsim-optics.js normalForm6, of the one-turn 6x6 map linearised about the
 * closed orbit, orbitMap): the matched beam is
 * Sigma = W diag(eps) W^T, the mismatch is against the mode Twiss, the energy offset rides the
 * dynamic dispersion, and every statistic reads the mode coordinates W^-1 (x - x_co). A ring
 * without RF has no 6D form and uses the uncoupled Twiss and the static dispersion, as before.
 * There x_co is the 6D closed orbit (closedOrbit6): with a kick and a cavity the beam closes
 * only at the momentum delta_co that keeps its longer path in step with the RF.
 * Gates: tests/analytic/test_injection.py (4D), tests/analytic/test_injection_6d.py (6D),
 * tests/analytic/test_injection_orbit6.py (the 6D centre), tests/analytic/test_injection_on_orbit.py
 * (the optics on that centre).
 *
 * It adds no physics of its own: the motion is accsim-track.js (a port of the package's
 * element-by-element tracker), the optics are accsim-optics.js. Radiation is OFF — an
 * electron beam here filaments but never damps — and the page says so.
 *
 * Coordinates: (x, px, y, py, zeta, delta). Dependency-free beyond the editor cores.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./accsim-optics.js"), require("./animate.js"), require("./accsim-track.js"));
  } else root.AccsimInject = factory(root.AccsimOptics, root.AccsimAnim, root.AccsimTrack);
})(typeof self !== "undefined" ? self : this, function (O, A, T) {
  "use strict";
  const X = 0, PX = 1, Y = 2, PY = 3, ZETA = 4, DELTA = 5;
  const TWO_PI = 2.0 * Math.PI;

  /**
   * The 6D closed orbit (x, px, y, py, zeta, delta) of the TRACKED turn (accsim-track.js,
   * radiation off): a port of accsim.orbit.closed_orbit_6d by the same route. Newton,
   *     x <- x - (J - I)^-1 (T(x) - x),   J by central differences,
   * started from the affine solve (I - M) x = k of the one-turn map, which misses the fixed
   * point by the orbit's second-order path length (5% of delta_co on a measured ring). With
   * no kick the fixed point is exactly zero and nothing is iterated. Without radiation the
   * answer has zeta_co = 0 and delta_co the momentum at which the kicked orbit's longer path
   * stays in step with the RF (closed_orbit_delta, N5). `a` is O.analyse(scenario).
   */
  function closedOrbit6(scenario, a) {
    const k = a.oneTurnKick;
    if (!k.some((v) => v !== 0.0)) return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const L = O._linalg, M = a.oneTurn;
    let x = L.solve(L.eye(6).map((row, i) => row.map((v, j) => v - M[i][j])), k);
    const machine = T.build(scenario);
    const turn = (s) => { const t = s.slice(); T.trackTurn(machine, t); return t; };
    const h = 1e-8, tol = 1e-14;
    let r = turn(x).map((v, i) => v - x[i]);
    for (let it = 0; it < 50; it++) {
      if (Math.max(...r.map(Math.abs)) < tol) return x;
      const A = [0, 1, 2, 3, 4, 5].map(() => new Array(6));
      for (let j = 0; j < 6; j++) {
        const p = x.slice(), q = x.slice();
        p[j] += h; q[j] -= h;
        const tp = turn(p), tq = turn(q);
        for (let i = 0; i < 6; i++) A[i][j] = (tp[i] - tq[i]) / (2.0 * h) - (i === j ? 1.0 : 0.0);
      }
      const dx = L.solve(A, r);
      x = x.map((v, i) => v - dx[i]);
      r = turn(x).map((v, i) => v - x[i]);
    }
    if (Math.max(...r.map(Math.abs)) < tol) return x;
    throw new Error(`the 6D closed orbit did not converge: one-turn residual ${Math.max(...r.map(Math.abs)).toPrecision(3)}`);
  }

  /**
   * The one-turn map linearised about the 6D closed orbit `x0`: the Jacobian of the tracked
   * turn (accsim-track.js, radiation off) by central differences at step 1e-7 — the optics a
   * beam on that orbit actually sees (I3; the package's linearised_one_turn_map, by a
   * whole-turn rather than a per-element route; the two agree to ~1e-10, which is the
   * package's own finite-difference floor). It moves at FIRST order in the orbit: an
   * off-centre quadrupole bends an off-momentum particle by k1l x_co / (1 + delta), so the
   * orbit makes dispersion of its own. Taken fresh, not from Newton's last step (that one sits
   * at the previous iterate). An orbit of exactly zero returns the design matrix itself, so a
   * kick-free ring is bit-identical to the design optics.
   */
  function orbitMap(scenario, a, x0) {
    if (!x0.some((v) => v !== 0.0)) return a.oneTurn;
    const machine = T.build(scenario);
    const turn = (s) => { const t = s.slice(); T.trackTurn(machine, t); return t; };
    const h = 1e-7;
    const J = [0, 1, 2, 3, 4, 5].map(() => new Array(6));
    for (let j = 0; j < 6; j++) {
      const p = x0.slice(), q = x0.slice();
      p[j] += h; q[j] -= h;
      const tp = turn(p), tq = turn(q);
      for (let i = 0; i < 6; i++) J[i][j] = (tp[i] - tq[i]) / (2.0 * h);
    }
    return J;
  }

  /**
   * The ring's optics at the injection point (the start of the lattice), from the editor's
   * own analysis — the same arrays the page draws. Throws if the ring does not close.
   * `orbit0` is the closed orbit as a 6-array: the 6D one where the beam is 6D-matched, else
   * the 4D one with zeta = delta = 0, which is kept as `orbit4` either way.
   */
  function ringOptics(scenario) {
    if (scenario.periodic === false) throw new Error("injection needs a ring (periodic) scenario");
    const a = O.analyse(scenario, { slicesFor: () => 1 });
    if (a.errors.length) throw new Error(a.errors[0]);
    if (a.opticsError) throw new Error(a.opticsError.message);
    if (a.coupled) throw new Error("the ring is x-y coupled; injection uses uncoupled Twiss");
    const tw = a.twiss0;
    const out = {
      ref: a.ref,
      length: a.length,
      tunes: a.tunes,
      chromaticity: a.chromaticity || null,
      Qs: a.Qs == null ? null : a.Qs,
      eta: a.eta == null ? null : a.eta,
      hasRF: (scenario.elements || []).some((e) => e.type === "RFCavity" && +e.voltage !== 0.0),
      orbit0: (a.closedOrbit0 || [0.0, 0.0, 0.0, 0.0]).concat([0.0, 0.0]),
      orbit4: (a.closedOrbit0 || [0.0, 0.0, 0.0, 0.0]).concat([0.0, 0.0]),
      twiss: {
        beta_x: tw.beta_x, alpha_x: tw.alpha_x, beta_y: tw.beta_y, alpha_y: tw.alpha_y,
        disp_x: tw.disp_x || 0.0, disp_px: tw.disp_px || 0.0,
        disp_y: tw.disp_y || 0.0, disp_py: tw.disp_py || 0.0,
      },
      normal: null,
      normalError: null,
    };
    // With a working RF cavity the ring has a 6D normal form, and the beam the one-turn map
    // leaves unchanged is drawn and measured in ITS coordinates: the transverse modes' own
    // Twiss (not the uncoupled one) and the dynamic dispersion (not the static one) replace
    // `twiss`, so every reading below is a normal-mode reading (docs/CONVENTIONS.md ->
    // "The 6D-matched beam"). The uncoupled Twiss is kept as `twiss4` for display. The centre
    // moves with it, to the 6D closed orbit. A ring where either fails keeps the 4D beam AND
    // the 4D centre: its readings strip the static dispersion, which a 6D centre (already
    // carrying D delta_co) would count twice. The normal form is of the map ON that orbit
    // (orbitMap), not of the design matrix: a displaced orbit sees its own dispersion.
    if (out.hasRF && out.Qs > 0) {
      try {
        const orbit = closedOrbit6(scenario, a);
        const J = orbitMap(scenario, a, orbit);
        const nf = O.normalForm6(J);
        out.orbit0 = orbit;
        out.orbitMap = J;
        out.normal = nf;
        out.twiss4 = out.twiss;
        out.twiss = {
          beta_x: nf.modeBeta[0], alpha_x: nf.modeAlpha[0], beta_y: nf.modeBeta[1], alpha_y: nf.modeAlpha[1],
          disp_x: nf.dispersion[0], disp_px: nf.dispersion[1], disp_y: nf.dispersion[2], disp_py: nf.dispersion[3],
        };
      } catch (e) {
        out.normalError = e.message;
      }
    }
    return out;
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
    if (optics.normal) return sampleInjection6(inj, optics);
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

  /** Courant-Snyder normalising block N(beta, alpha) = [[sqrt b, 0], [-a/sqrt b, 1/sqrt b]]. */
  function csBlock(beta, alpha) {
    const sb = Math.sqrt(beta);
    return [[sb, 0.0], [-alpha / sb, 1.0 / sb]];
  }
  /** Mode `k`'s longitudinal-mode-free share of the energy spread, W[5,2k]^2 + W[5,2k+1]^2. */
  function deltaShare(W, k) {
    return W[DELTA][2 * k] ** 2 + W[DELTA][2 * k + 1] ** 2;
  }
  /**
   * The injected bunch on a ring with RF: drawn in the 6D normal-mode coordinates u, then
   * x = x_co + W u. Per transverse mode k (Twiss beta_k, alpha_k = optics.twiss)
   *     (u_2k, u_2k+1) = sqrt(eps) N(beta_k, alpha_k)^-1 N(beta_i, alpha_i) (a, b) + N(beta_k, alpha_k)^-1 (du, dpu)
   * — the shape mismatch against the MODE Twiss, and the aiming error as a pure betatron
   * (mode-k) excursion, all about the 6D closed orbit. The longitudinal mode carries eps_3 = sigma_delta^2 / (W[5,4]^2 +
   * W[5,5]^2), its own energy spread (the transverse modes add theirs, ~1e-5 relative), and
   * the energy offset as the mode-3 point with zeta = 0, delta = delta0 (an offset from the
   * ring's own momentum delta_co, not from the design one) — which puts the
   * transverse centroid on the dynamic dispersion. With every setting at its default the
   * beam's covariance is W diag(eps) W^T: the one the one-turn map leaves unchanged.
   */
  function sampleInjection6(inj, optics) {
    const rng = A.mulberry32(inj.seed || 1);
    const rows = [];
    for (let p = 0; p < inj.n; p++) {
      const r = [];
      for (let j = 0; j < 6; j++) r.push(A.gaussian(rng));
      rows.push(r);
    }
    whiten(rows);
    const { W } = optics.normal, t = optics.twiss, o = optics.orbit0;
    const planes = [
      { eps: inj.emit_x, m: csBlock(t.beta_x, t.alpha_x), i: csBlock(inj.beta_x, inj.alpha_x), d: [inj.dx, inj.dpx] },
      { eps: inj.emit_y, m: csBlock(t.beta_y, t.alpha_y), i: csBlock(inj.beta_y, inj.alpha_y), d: [inj.dy, inj.dpy] },
    ].map((q) => {
      const mInv = [[q.m[1][1], -q.m[0][1]], [-q.m[1][0], q.m[0][0]]]; // det N = 1
      const shape = O._linalg.matmul(mInv, q.i).map((row) => row.map((v) => v * Math.sqrt(q.eps)));
      return { shape, offset: O._linalg.matvec(mInv, q.d) };
    });
    const sz = Math.sqrt(inj.sigma_delta * inj.sigma_delta / deltaShare(W, 2));
    // The mode-3 point with zeta = 0, delta = delta0: solve the 2x2 [[W44, W45], [W54, W55]].
    const det = W[ZETA][4] * W[DELTA][5] - W[ZETA][5] * W[DELTA][4];
    const z0 = [-W[ZETA][5] * inj.delta0 / det, W[ZETA][4] * inj.delta0 / det];
    return rows.map((r) => {
      const u = new Array(6);
      planes.forEach((q, k) => {
        u[2 * k] = q.shape[0][0] * r[2 * k] + q.shape[0][1] * r[2 * k + 1] + q.offset[0];
        u[2 * k + 1] = q.shape[1][0] * r[2 * k] + q.shape[1][1] * r[2 * k + 1] + q.offset[1];
      });
      u[4] = sz * r[4] + z0[0];
      u[5] = sz * r[5] + z0[1];
      const x = O._linalg.matvec(W, u);
      return [o[0] + x[0], o[1] + x[1], o[2] + x[2], o[3] + x[3], o[4] + x[4], o[5] + x[5]];
    });
  }

  // ------------------------------------------------------------------ statistics
  /** The betatron part of one particle in plane `pl` ("x" or "y"): the closed orbit and
   *  D delta removed, with the ring's dispersion at the injection point (never a fit).
   *  On a ring with a 6D normal form it is the mode-k pair of W^-1 (x - x_co), mapped back
   *  through N(beta_k, alpha_k): a betatron coordinate in metres whose Courant-Snyder action
   *  with the mode Twiss is the mode's own action, (u_2k^2 + u_2k+1^2) / 2, exactly. */
  function betatronPart(s, optics, pl) {
    const t = optics.twiss, o = optics.orbit0;
    if (optics.normal) {
      const Wi = optics.normal.Winv, k = pl === "x" ? 0 : 1;
      const d = [s[X] - o[0], s[PX] - o[1], s[Y] - o[2], s[PY] - o[3], s[ZETA] - o[4], s[DELTA] - o[5]];
      let u1 = 0.0, u2 = 0.0;
      for (let j = 0; j < 6; j++) { u1 += Wi[2 * k][j] * d[j]; u2 += Wi[2 * k + 1][j] * d[j]; }
      const beta = pl === "x" ? t.beta_x : t.beta_y, alpha = pl === "x" ? t.alpha_x : t.alpha_y;
      const sb = Math.sqrt(beta);
      return [sb * u1, -alpha / sb * u1 + u2 / sb];
    }
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

  /**
   * The nearest synchrotron sideband of an integer, Q_k + m Q_s (k = transverse mode, m =
   * +-1..+-3), within `tol` of an integer — or null. On a sideband the energy oscillation
   * drives the betatron motion resonantly and even a 6D-matched beam grows (measured on
   * `electron-ring`, 0.0009 from Q_1 + 3 Q_s = 6: <J> 1.48x after 600 turns). Mode tunes from
   * the normal form, so the distance is the one the growth follows.
   */
  function sidebandNear(optics, tol) {
    if (!optics.normal) return null;
    const lim = tol == null ? 5e-3 : tol, [q1, q2, q3] = optics.normal.tunes;
    const qs = Math.min(q3, 1.0 - q3); // the longitudinal mode turns backwards (q3 = 1 - Q_s)
    let best = null;
    [q1, q2].forEach((q, k) => {
      for (let m = -3; m <= 3; m++) {
        if (m === 0) continue;
        const v = q + m * qs, dist = v - Math.round(v);
        if (Math.abs(dist) < lim && (!best || Math.abs(dist) < Math.abs(best.distance))) best = { plane: k === 0 ? "x" : "y", m, distance: dist };
      }
    });
    return best;
  }

  return {
    ringOptics, closedOrbit6, orbitMap, defaults, sampleInjection, matchedBunchLength, whiten, cholesky,
    betatronPart, action, bmag, expectedAction, beamStats, chromaticEnvelope, sidebandNear,
  };
});
