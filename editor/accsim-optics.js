/*
 * accsim-optics.js — the linear-optics core of the accsim scenario editor.
 *
 * A faithful port of the *linear* half of the accsim Python package: each element's
 * 6x6 matrix and constant kick, the periodic (Courant-Snyder) match, Twiss
 * propagation, tunes, first-order chromaticity, momentum compaction, the small-
 * amplitude synchrotron tune, the 4D closed orbit and the planar survey. Every
 * formula here is the one in `src/accsim/...`, entry for entry, and
 * `tests/analytic/test_scenario.py` runs this file under Node and holds it against
 * the Python package to 1e-9 on every preset. Disagreement is a bug on one side or
 * the other, never a tolerance to loosen.
 *
 * Coordinates: (x, px, y, py, zeta, delta) in the Xsuite / MAD-X ordering, exactly
 * as `docs/CONVENTIONS.md` records. Units are SI throughout: m, rad, eV, V, Hz.
 *
 * The file is dependency-free and runs in a browser (window.AccsimOptics) and in
 * Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.AccsimOptics = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CLIGHT = 299792458.0;
  const ELECTRON_MASS_EV = 0.51099895069e6;
  const PROTON_MASS_EV = 938.27208816e6;
  const SPECIES = {
    electron: { mass_eV: ELECTRON_MASS_EV, charge: -1.0 },
    positron: { mass_eV: ELECTRON_MASS_EV, charge: +1.0 },
    proton: { mass_eV: PROTON_MASS_EV, charge: +1.0 },
  };
  const X = 0, PX = 1, Y = 2, PY = 3, ZETA = 4, DELTA = 5;
  const TWO_PI = 2.0 * Math.PI;
  const INV_4PI = 1.0 / (4.0 * Math.PI);

  // ---------------------------------------------------------------- linear algebra
  function eye(n) {
    const M = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(n).fill(0.0);
      row[i] = 1.0;
      M.push(row);
    }
    return M;
  }
  function matmul(A, B) {
    const n = A.length, m = B[0].length, k = B.length;
    const C = [];
    for (let i = 0; i < n; i++) {
      const row = new Array(m).fill(0.0);
      const Ai = A[i];
      for (let p = 0; p < k; p++) {
        const a = Ai[p];
        if (a === 0.0) continue;
        const Bp = B[p];
        for (let j = 0; j < m; j++) row[j] += a * Bp[j];
      }
      C.push(row);
    }
    return C;
  }
  function matvec(A, v) {
    return A.map((row) => row.reduce((acc, a, j) => acc + a * v[j], 0.0));
  }
  function vadd(a, b) {
    return a.map((x, i) => x + b[i]);
  }
  function transpose(A) {
    return A[0].map((_, j) => A.map((row) => row[j]));
  }
  function sub(M, rows, cols) {
    return rows.map((r) => cols.map((c) => M[r][c]));
  }
  function setBlock(M, rows, cols, B) {
    rows.forEach((r, i) => cols.forEach((c, j) => { M[r][c] = B[i][j]; }));
  }
  /** Solve A x = b by Gaussian elimination with partial pivoting; returns null if singular. */
  function solve(A, b) {
    const n = A.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-300) return null;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        if (f === 0.0) continue;
        for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map((row, i) => row[n] / row[i]);
  }
  /** Condition number proxy: max|entry| * max|entry of inverse| (Frobenius-ish bound). */
  function condProxy(A) {
    const n = A.length;
    const cols = [];
    for (let j = 0; j < n; j++) {
      const e = new Array(n).fill(0.0);
      e[j] = 1.0;
      const x = solve(A, e);
      if (x === null) return Infinity;
      cols.push(x);
    }
    let na = 0.0, ni = 0.0;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        na = Math.max(na, Math.abs(A[i][j]));
        ni = Math.max(ni, Math.abs(cols[j][i]));
      }
    return na * ni * n;
  }

  // ------------------------------------------------------------ reference particle
  function reference(spec) {
    const species = spec.species || "custom";
    const base = SPECIES[species] || {};
    const mass_eV = spec.mass_eV != null ? +spec.mass_eV : base.mass_eV;
    const charge = spec.charge != null ? +spec.charge : (base.charge != null ? base.charge : 1.0);
    if (!(mass_eV > 0)) throw new Error("reference particle needs a positive mass");
    let total;
    const mode = spec.energy_mode || "total";
    const E = +spec.energy_eV;
    if (!(E > 0)) throw new Error("reference particle needs a positive energy");
    if (mode === "total") total = E;
    else if (mode === "kinetic") total = mass_eV + E;
    else if (mode === "momentum") total = Math.hypot(E, mass_eV);
    else throw new Error("unknown energy mode " + mode);
    if (total < mass_eV) throw new Error("total energy below the rest mass");
    const gamma0 = total / mass_eV;
    const beta0 = Math.sqrt(1.0 - 1.0 / (gamma0 * gamma0));
    return {
      species, mass_eV, charge,
      total_energy_eV: total,
      gamma0, beta0,
      momentum_eV: Math.sqrt(total * total - mass_eV * mass_eV),
      kinetic_energy_eV: total - mass_eV,
      brho: Math.sqrt(total * total - mass_eV * mass_eV) / CLIGHT / Math.abs(charge || 1.0),
    };
  }

  // ------------------------------------------------------------- element building blocks
  /** 2x2 block of u'' + g u = 0 over length L (accsim.elements.quadrupole._focusing_block). */
  function focusingBlock(g, L) {
    if (g > 0.0) {
      const w = Math.sqrt(g), c = Math.cos(w * L), s = Math.sin(w * L);
      return [[c, s / w], [-w * s, c]];
    }
    if (g < 0.0) {
      const w = Math.sqrt(-g), ch = Math.cosh(w * L), sh = Math.sinh(w * L);
      return [[ch, sh / w], [w * sh, ch]];
    }
    return [[1.0, L], [0.0, 1.0]];
  }
  /** (c1, s1, c2) of accsim.elements.dipole._dispersion_integrals. */
  function dispersionIntegrals(K, L) {
    if (Math.abs(K) < 1e-9) {
      return [L * L / 2.0 - K * L ** 4 / 24.0, L - K * L ** 3 / 6.0, -(L ** 3) / 6.0 + K * L ** 5 / 120.0];
    }
    let cosWL, s1;
    if (K > 0.0) {
      const w = Math.sqrt(K);
      cosWL = Math.cos(w * L);
      s1 = Math.sin(w * L) / w;
    } else {
      const w = Math.sqrt(-K);
      cosWL = Math.cosh(w * L);
      s1 = Math.sinh(w * L) / w;
    }
    return [(1.0 - cosWL) / K, s1, (s1 - L) / K];
  }
  /** Hard-edge pole-face kick (accsim.elements.dipole._edge_matrix). */
  function edgeMatrix(h, e) {
    const E = eye(6);
    if (h === 0.0 || e === 0.0) return E;
    const t = h * Math.tan(e);
    E[PX][X] = t;
    E[PY][Y] = -t;
    return E;
  }
  /** Passive rotation about s (accsim.elements.alignment.s_rotation). */
  function sRotation(phi) {
    const M = eye(6);
    if (phi === 0.0) return M;
    const c = Math.cos(phi), s = Math.sin(phi);
    M[X][X] = M[Y][Y] = M[PX][PX] = M[PY][PY] = c;
    M[X][Y] = M[PX][PY] = s;
    M[Y][X] = M[PY][PX] = -s;
    return M;
  }
  function driftMatrix(L, ref) {
    const M = eye(6);
    M[X][PX] = L;
    M[Y][PY] = L;
    M[ZETA][DELTA] = L / (ref.gamma0 * ref.gamma0);
    return M;
  }
  function quadMatrix(L, k1, ref) {
    const M = eye(6);
    setBlock(M, [X, PX], [X, PX], focusingBlock(k1, L));
    setBlock(M, [Y, PY], [Y, PY], focusingBlock(-k1, L));
    M[ZETA][DELTA] = L / (ref.gamma0 * ref.gamma0);
    return M;
  }
  /** Bend body (no edges) — accsim.elements.dipole.Dipole._arc_matrix. */
  function bendBody(L, angle, k1, ref) {
    const g2 = ref.gamma0 * ref.gamma0;
    const M = eye(6);
    if (angle === 0.0 && k1 === 0.0) return driftMatrix(L, ref);
    const h = angle / L;
    if (k1 !== 0.0) {
      const Kx = h * h + k1;
      setBlock(M, [X, PX], [X, PX], focusingBlock(Kx, L));
      setBlock(M, [Y, PY], [Y, PY], focusingBlock(-k1, L));
      const [c1, s1, c2] = dispersionIntegrals(Kx, L);
      const r16 = h * c1, r26 = h * s1;
      M[X][DELTA] = r16;
      M[PX][DELTA] = r26;
      M[ZETA][X] = -r26;
      M[ZETA][PX] = -r16;
      M[ZETA][DELTA] = L / g2 + h * h * c2;
      return M;
    }
    const c = Math.cos(angle), s = Math.sin(angle);
    M[X][X] = c;
    M[X][PX] = s / h;
    M[X][DELTA] = (1.0 - c) / h;
    M[PX][X] = -h * s;
    M[PX][PX] = c;
    M[PX][DELTA] = s;
    M[Y][PY] = L;
    M[ZETA][X] = -s;
    M[ZETA][PX] = (c - 1.0) / h;
    M[ZETA][DELTA] = s / h - L + L / g2;
    return M;
  }
  function solenoidMatrix(L, ks, ref) {
    const K = 0.5 * ks;
    const u = K * L;
    const C = Math.cos(u);
    const Lsinc = u === 0.0 ? L : L * Math.sin(u) / u;
    const S = Math.sin(u);
    const CC = C * C, SC_K = Lsinc * C, SC = S * C, SS_K = Lsinc * S, K_SS = K * S * S;
    const M = eye(6);
    setBlock(M, [X, PX, Y, PY], [X, PX, Y, PY], [
      [CC, SC_K, SC, SS_K],
      [-K * SC, CC, -K_SS, SC],
      [-SC, -SS_K, CC, SC_K],
      [K_SS, -SC, -K * SC, CC],
    ]);
    M[ZETA][DELTA] = L / (ref.gamma0 * ref.gamma0);
    return M;
  }
  function skewQuadMatrix(L, k1s, ref) {
    const F = focusingBlock(k1s, L), D = focusingBlock(-k1s, L);
    const A = F.map((row, i) => row.map((v, j) => 0.5 * (v + D[i][j])));
    const B = F.map((row, i) => row.map((v, j) => 0.5 * (D[i][j] - v)));
    const M = eye(6);
    setBlock(M, [X, PX], [X, PX], A);
    setBlock(M, [Y, PY], [Y, PY], A);
    setBlock(M, [X, PX], [Y, PY], B);
    setBlock(M, [Y, PY], [X, PX], B);
    M[ZETA][DELTA] = L / (ref.gamma0 * ref.gamma0);
    return M;
  }
  /** Wiggler body of length L (accsim.elements.wiggler.Wiggler); exact for any L. */
  function wigglerMatrix(L, h0, period, ref) {
    const g2 = ref.gamma0 * ref.gamma0;
    const k = TWO_PI / period;
    const theta = h0 / k;
    const M = eye(6);
    setBlock(M, [X, PX], [X, PX], focusingBlock(0.0, L));
    setBlock(M, [Y, PY], [Y, PY], focusingBlock(0.5 * h0 * h0, L));
    M[ZETA][DELTA] = L / g2 + 0.25 * L * theta * theta * (2.0 + 1.0 / g2);
    return M;
  }
  function wigglerKick(L, h0, period) {
    const k = new Array(6).fill(0.0);
    const theta = h0 / (TWO_PI / period);
    k[ZETA] = -0.25 * L * theta * theta;
    return k;
  }
  function rfSlope(el, ref) {
    const krf = TWO_PI * el.frequency / (ref.beta0 * CLIGHT);
    return -(ref.charge * el.voltage * krf * Math.cos(el.phi_s || 0.0)) /
      (ref.beta0 * ref.beta0 * ref.total_energy_eV);
  }

  // --------------------------------------------------------------- element catalogue
  /**
   * The schema shared with `accsim.scenario` (Python). `params` lists the numeric
   * fields in constructor order; `thick` says whether a length is carried; `bends`
   * marks the one element that turns the reference frame; `align` says which
   * misalignment fields the element accepts.
   */
  const CATALOGUE = {
    Drift: { params: ["length"], thick: true, align: ["dx", "dy", "roll"] },
    Quadrupole: { params: ["length", "k1"], thick: true, align: ["dx", "dy", "roll"] },
    ThinQuadrupole: { params: ["k1l"], thick: false, align: ["dx", "dy", "roll"] },
    Dipole: { params: ["length", "angle", "k1", "e1", "e2"], defaults: { k1: 0.0, e1: 0.0, e2: 0.0 }, thick: true, bends: true, align: ["roll"], flags: ["fringe"] },
    Sextupole: { params: ["length", "k2"], thick: true, align: ["dx", "dy", "roll"] },
    ThinSextupole: { params: ["k2l"], thick: false, align: ["dx", "dy", "roll"] },
    Octupole: { params: ["length", "k3"], thick: true, align: ["dx", "dy", "roll"] },
    ThinOctupole: { params: ["k3l"], thick: false, align: ["dx", "dy", "roll"] },
    SkewQuadrupole: { params: ["length", "k1s"], thick: true, align: ["dx", "dy", "roll"] },
    ThinSkewQuadrupole: { params: ["k1sl"], thick: false, align: ["dx", "dy", "roll"] },
    Solenoid: { params: ["length", "ks"], thick: true, align: ["dx", "dy", "roll"] },
    RFCavity: { params: ["voltage", "frequency", "phi_s"], defaults: { frequency: 0.0, phi_s: 0.0 }, thick: false, align: [], optional: ["harmonic"] },
    Corrector: { params: ["kick_x", "kick_y"], defaults: { kick_x: 0.0, kick_y: 0.0 }, thick: false, align: ["dx", "dy", "roll"] },
    Wiggler: { params: ["period", "h0", "periods"], thick: true, align: ["dx", "dy", "roll"] },
    Aperture: { params: ["half_x", "half_y", "length"], defaults: { half_y: null, length: 0.0 }, thick: true, align: [], text: ["shape"] },
    Collimator: { params: ["half_x", "half_y", "length"], defaults: { half_y: null, length: 0.001 }, thick: true, align: [], text: ["shape"] },
  };
  function hasDefault(spec, p) {
    return spec.defaults && Object.prototype.hasOwnProperty.call(spec.defaults, p);
  }

  function elementLength(el) {
    if (el.type === "Wiggler") return (+el.period || 0) * (+el.periods || 0);
    if (el.type === "_WigglerSlice") return +el.length; // a wiggler of fractional length: its map is linear in L
    return CATALOGUE[el.type] && CATALOGUE[el.type].thick ? (+el.length || 0.0) : 0.0;
  }
  function elementAngle(el) {
    return el.type === "Dipole" ? (+el.angle || 0.0) : 0.0;
  }

  /**
   * Validate one element record. Returns a list of messages (empty when fine). The
   * rules are the ones the Python constructors enforce, so a scenario the editor
   * accepts is one accsim will load.
   */
  function validateElement(el) {
    const errs = [];
    const spec = CATALOGUE[el.type];
    if (!spec) return ["unknown element type " + el.type];
    for (const p of spec.params) {
      const missing = el[p] == null || el[p] === "";
      if (missing && hasDefault(spec, p)) continue;
      if (missing || !Number.isFinite(+el[p])) errs.push(`${p} must be a number`);
    }
    if (spec.thick && el.type !== "Wiggler" && +el.length < 0) errs.push("length must be >= 0");
    if (el.type === "Dipole") {
      if (+el.length === 0 && +el.angle !== 0) errs.push("a finite bend angle needs a positive length");
      if (+el.roll && +el.angle) errs.push("a rolled bend needs the curved rigid-body geometry (accsim K2); the editor does not model it — set roll to 0");
      if (el.dx || el.dy) errs.push("a bending dipole cannot be displaced (accsim refuses it); use a Corrector for its steering error");
    }
    if (el.type === "Wiggler") {
      if (!(+el.period > 0)) errs.push("period must be > 0");
      if (!(Number.isInteger(+el.periods) && +el.periods >= 1)) errs.push("periods must be a positive integer");
    }
    if (el.type === "RFCavity") {
      const hasH = el.harmonic != null && el.harmonic !== "";
      if (hasH && !(Number.isInteger(+el.harmonic) && +el.harmonic > 0)) errs.push("harmonic must be a positive integer");
      if (!hasH && !(+el.frequency >= 0)) errs.push("frequency must be >= 0 (or give a harmonic)");
    }
    if (el.type === "Aperture" || el.type === "Collimator") {
      if (!["circular", "elliptical", "rectangular"].includes(el.shape)) errs.push("shape must be circular, elliptical or rectangular");
      if (!(+el.half_x > 0)) errs.push("half_x must be > 0");
    }
    return errs;
  }

  /**
   * Resolve derived quantities that need the whole lattice: an RF cavity given by a
   * harmonic number gets its frequency from the circumference, `f = h beta0 c / C`.
   * Returns a shallow copy of each element with numbers coerced.
   */
  function resolve(elements, ref) {
    const C = elements.reduce((s, el) => s + elementLength(el), 0.0);
    return elements.map((el) => {
      const out = Object.assign({}, el);
      const spec = CATALOGUE[el.type] || { params: [], align: [] };
      for (const p of spec.params.concat(spec.align || [])) {
        if (out[p] != null && out[p] !== "") out[p] = +out[p];
        else if (hasDefault(spec, p)) out[p] = spec.defaults[p];
        else if ((spec.align || []).includes(p)) out[p] = 0.0;
      }
      if (el.type === "RFCavity" && el.harmonic != null && el.harmonic !== "") {
        out.harmonic = +el.harmonic;
        out.frequency = C > 0 ? out.harmonic * ref.beta0 * CLIGHT / C : 0.0;
      }
      return out;
    });
  }

  /** The element's own-frame matrix (accsim `_matrix_body`). */
  function bodyMatrix(el, ref) {
    const L = elementLength(el);
    switch (el.type) {
      case "Drift": case "Sextupole": case "Octupole": case "Aperture": case "Collimator":
        return driftMatrix(L, ref);
      case "Quadrupole": return quadMatrix(L, +el.k1, ref);
      case "ThinQuadrupole": { const M = eye(6); M[PX][X] = -el.k1l; M[PY][Y] = +el.k1l; return M; }
      case "Dipole": {
        const body = bendBody(L, +el.angle, +el.k1 || 0.0, ref);
        const e1 = +el.e1 || 0.0, e2 = +el.e2 || 0.0;
        if (e1 === 0.0 && e2 === 0.0) return body;
        const h = L > 0 ? el.angle / L : 0.0;
        return matmul(edgeMatrix(h, e2), matmul(body, edgeMatrix(h, e1)));
      }
      case "ThinSextupole": case "ThinOctupole": case "Corrector": return eye(6);
      case "SkewQuadrupole": return skewQuadMatrix(L, +el.k1s, ref);
      case "ThinSkewQuadrupole": { const M = eye(6); M[PX][Y] = +el.k1sl; M[PY][X] = +el.k1sl; return M; }
      case "Solenoid": return solenoidMatrix(L, +el.ks, ref);
      case "RFCavity": { const M = eye(6); M[DELTA][ZETA] = rfSlope(el, ref); return M; }
      case "Wiggler": case "_WigglerSlice": return wigglerMatrix(L, +el.h0, +el.period, ref);
      default: throw new Error("no matrix for element type " + el.type);
    }
  }
  function bodyKick(el, ref) {
    const k = new Array(6).fill(0.0);
    if (el.type === "Corrector") { k[PX] = +el.kick_x || 0.0; k[PY] = +el.kick_y || 0.0; }
    if (el.type === "Wiggler" || el.type === "_WigglerSlice") return wigglerKick(elementLength(el), +el.h0, +el.period);
    return k;
  }
  function offsetOf(el) {
    const d = new Array(6).fill(0.0);
    d[X] = +el.dx || 0.0;
    d[Y] = +el.dy || 0.0;
    return d;
  }
  /** Public matrix: body conjugated by the roll (accsim Element.matrix). */
  function elementMatrix(el, ref) {
    const body = bodyMatrix(el, ref);
    const roll = +el.roll || 0.0;
    if (roll === 0.0) return body;
    if (el.type === "Dipole" && +el.angle !== 0.0) throw new Error("rolled bend is not modelled");
    return matmul(sRotation(-roll), matmul(body, sRotation(roll)));
  }
  /** Public kick (accsim Element.kick), including the misalignment's constant part. */
  function elementKick(el, ref) {
    const k = bodyKick(el, ref);
    const roll = +el.roll || 0.0;
    const d = offsetOf(el);
    const misaligned = roll !== 0.0 || d[X] !== 0.0 || d[Y] !== 0.0;
    if (!misaligned) return k;
    if (el.type === "Dipole" && +el.angle !== 0.0) throw new Error("misaligned bend is not modelled");
    if (roll === 0.0) {
      const M = elementMatrix(el, ref);
      const IminusM = eye(6).map((row, i) => row.map((v, j) => v - M[i][j]));
      return vadd(k, matvec(IminusM, d));
    }
    const Min = sRotation(roll), kin = matvec(Min, d).map((v) => -v);
    const Mout = sRotation(-roll), kout = d;
    return vadd(matvec(Mout, vadd(matvec(bodyMatrix(el, ref), kin), k)), kout);
  }

  /**
   * Split a thick element into `n` exact sub-slices for plotting. Every element
   * here slices exactly (its map is the exponential of a constant generator), with
   * the pole faces of a bend kept on the first and last slice.
   */
  function sliceElement(el, n) {
    const L = elementLength(el);
    if (n <= 1 || L <= 0.0) return [el];
    const out = [];
    for (let i = 0; i < n; i++) {
      const s = Object.assign({}, el, { name: el.name });
      if (el.type === "Wiggler") {
        s.type = "_WigglerSlice";
        s.length = L / n;
      } else {
        s.length = L / n;
        if (el.type === "Dipole") {
          s.angle = (+el.angle || 0.0) / n;
          s.e1 = i === 0 ? (+el.e1 || 0.0) : 0.0;
          s.e2 = i === n - 1 ? (+el.e2 || 0.0) : 0.0;
        }
      }
      out.push(s);
    }
    return out;
  }
  // ------------------------------------------------------------------ lattice maps
  function transferMap(elements, ref) {
    let M = eye(6);
    let k = new Array(6).fill(0.0);
    for (const el of elements) {
      const Me = elementMatrix(el, ref);
      M = matmul(Me, M);
      k = vadd(matvec(Me, k), elementKick(el, ref));
    }
    return { M, k };
  }

  function blocks(M) {
    return [sub(M, [X, PX], [X, PX]), sub(M, [Y, PY], [Y, PY])];
  }
  function transverse4(M) {
    return sub(M, [X, PX, Y, PY], [X, PX, Y, PY]);
  }
  function dispersiveKick(M) {
    return [M[X][DELTA], M[PX][DELTA], M[Y][DELTA], M[PY][DELTA]];
  }
  function couplingNorm(M) {
    let m = 0.0;
    for (const r of [X, PX]) for (const c of [Y, PY]) m = Math.max(m, Math.abs(M[r][c]), Math.abs(M[c][r]));
    return m;
  }
  function matchedBlock(C) {
    const cosMu = 0.5 * (C[0][0] + C[1][1]);
    if (Math.abs(cosMu) >= 1.0) {
      const e = new Error(`unstable plane: |1/2 Tr| = ${Math.abs(cosMu).toPrecision(6)} >= 1`);
      e.code = "unstable";
      e.halfTrace = cosMu;
      throw e;
    }
    const sinMu = Math.sign(C[0][1] || 1) * Math.sqrt(1.0 - cosMu * cosMu);
    return { beta: C[0][1] / sinMu, alpha: 0.5 * (C[0][0] - C[1][1]) / sinMu };
  }
  function propagateBlock(C, beta, alpha) {
    const gamma = (1.0 + alpha * alpha) / beta;
    const B = [[beta, -alpha], [-alpha, gamma]];
    const B1 = matmul(C, matmul(B, transpose(C)));
    let dmu = Math.atan2(C[0][1], beta * C[0][0] - alpha * C[0][1]);
    if (dmu < 0.0) dmu += TWO_PI;
    return { beta: B1[0][0], alpha: -B1[0][1], dmu };
  }
  /** Periodic Courant-Snyder match (accsim.twiss.match_periodic). */
  function matchPeriodic(M) {
    const c = couplingNorm(M);
    if (c > 1e-9) {
      const e = new Error(`lattice is x-y coupled (off-block norm ${c.toPrecision(3)})`);
      e.code = "coupled";
      throw e;
    }
    const [cx, cy] = blocks(M);
    const hx = matchedBlock(cx), hy = matchedBlock(cy);
    const m4 = transverse4(M), d = dispersiveKick(M);
    const A = eye(4).map((row, i) => row.map((v, j) => v - m4[i][j]));
    const disp = solve(A, d);
    if (disp === null) { const e = new Error("integer tune: dispersion does not close"); e.code = "resonant"; throw e; }
    return {
      s: 0.0, beta_x: hx.beta, alpha_x: hx.alpha, mu_x: 0.0,
      beta_y: hy.beta, alpha_y: hy.alpha, mu_y: 0.0,
      disp_x: disp[0], disp_px: disp[1], disp_y: disp[2], disp_py: disp[3],
    };
  }
  /**
   * Twiss at every element boundary (accsim.twiss.propagate_twiss). With
   * `slicesFor(el)` > 1 the thick elements are sub-sliced for a smooth plot; the
   * boundary values are unchanged by the slicing. Each point carries `elem`, the
   * index of the element it ends.
   */
  function propagateTwiss(elements, ref, tw0, slicesFor) {
    const points = [Object.assign({ elem: -1 }, tw0)];
    let s = tw0.s || 0.0;
    let bx = tw0.beta_x, ax = tw0.alpha_x, mux = tw0.mu_x || 0.0;
    let by = tw0.beta_y, ay = tw0.alpha_y, muy = tw0.mu_y || 0.0;
    let disp = [tw0.disp_x || 0.0, tw0.disp_px || 0.0, tw0.disp_y || 0.0, tw0.disp_py || 0.0];
    elements.forEach((el, i) => {
      const pieces = slicesFor ? sliceElement(el, slicesFor(el)) : [el];
      for (const piece of pieces) {
        const M = elementMatrix(piece, ref);
        const [cx, cy] = blocks(M);
        const px = propagateBlock(cx, bx, ax), py = propagateBlock(cy, by, ay);
        bx = px.beta; ax = px.alpha; mux += px.dmu;
        by = py.beta; ay = py.alpha; muy += py.dmu;
        disp = vadd(matvec(transverse4(M), disp), dispersiveKick(M));
        s += elementLength(piece);
        points.push({ s, beta_x: bx, alpha_x: ax, mu_x: mux, beta_y: by, alpha_y: ay, mu_y: muy,
          disp_x: disp[0], disp_px: disp[1], disp_y: disp[2], disp_py: disp[3], elem: i });
      }
    });
    return points;
  }
  function tunes(elements, ref) {
    const { M } = transferMap(elements, ref);
    const tw0 = matchPeriodic(M);
    const pts = propagateTwiss(elements, ref, tw0);
    const end = pts[pts.length - 1];
    return [end.mu_x / TWO_PI, end.mu_y / TWO_PI];
  }

  // ---------------------------------------------------------------- chromaticity
  function dipoleChromaIntegrand(bx, ax, by, ay, dx, dpx, h, k1) {
    const gx = (1.0 + ax * ax) / bx, gy = (1.0 + ay * ay) / by;
    return [
      -bx * (k1 + h * h) + h * (gx * dx - 2.0 * ax * dpx) + 2.0 * h * k1 * bx * dx,
      by * k1 + gy * h * dx - h * k1 * by * dx,
    ];
  }
  /** accsim.twiss.natural_chromaticity, entry for entry. */
  function naturalChromaticity(elements, ref, slices) {
    slices = slices || 64;
    const { M } = transferMap(elements, ref);
    const tw0 = matchPeriodic(M);
    let bx = tw0.beta_x, ax = tw0.alpha_x, by = tw0.beta_y, ay = tw0.alpha_y;
    let disp = [tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py];
    let xix = 0.0, xiy = 0.0;
    const advance = (Mm) => {
      const [cx, cy] = blocks(Mm);
      const px = propagateBlock(cx, bx, ax), py = propagateBlock(cy, by, ay);
      bx = px.beta; ax = px.alpha; by = py.beta; ay = py.alpha;
      disp = vadd(matvec(transverse4(Mm), disp), dispersiveKick(Mm));
    };
    for (const el of elements) {
      const L = elementLength(el);
      if (el.type === "ThinQuadrupole") {
        xix += -INV_4PI * bx * el.k1l;
        xiy += +INV_4PI * by * el.k1l;
        advance(elementMatrix(el, ref));
      } else if (el.type === "Quadrupole" && el.k1 !== 0.0 && L > 0.0) {
        const ds = L / slices;
        const xb = focusingBlock(el.k1, ds), yb = focusingBlock(-el.k1, ds);
        const sub4 = transverse4(quadMatrix(ds, el.k1, ref));
        let ibx = 0.5 * bx, iby = 0.5 * by;
        for (let i = 0; i < slices; i++) {
          const px = propagateBlock(xb, bx, ax), py = propagateBlock(yb, by, ay);
          bx = px.beta; ax = px.alpha; by = py.beta; ay = py.alpha;
          disp = matvec(sub4, disp);
          const w = i === slices - 1 ? 0.5 : 1.0;
          ibx += w * bx; iby += w * by;
        }
        xix += -INV_4PI * el.k1 * ibx * ds;
        xiy += +INV_4PI * el.k1 * iby * ds;
      } else if (el.type === "Dipole" && L > 0.0 && (el.angle !== 0.0 || el.k1)) {
        const h = el.angle / L, k1 = el.k1 || 0.0;
        const t1 = h * Math.tan(el.e1 || 0.0);
        xix += INV_4PI * bx * t1; xiy += -INV_4PI * by * t1;
        advance(edgeMatrix(h, el.e1 || 0.0));
        const ds = L / slices;
        const subM = bendBody(ds, h * ds, k1, ref);
        const [cx, cy] = blocks(subM);
        const sub4 = transverse4(subM), subk = dispersiveKick(subM);
        let [ix, iy] = dipoleChromaIntegrand(bx, ax, by, ay, disp[0], disp[1], h, k1);
        let accx = 0.5 * ix, accy = 0.5 * iy;
        for (let i = 0; i < slices; i++) {
          const px = propagateBlock(cx, bx, ax), py = propagateBlock(cy, by, ay);
          bx = px.beta; ax = px.alpha; by = py.beta; ay = py.alpha;
          disp = vadd(matvec(sub4, disp), subk);
          [ix, iy] = dipoleChromaIntegrand(bx, ax, by, ay, disp[0], disp[1], h, k1);
          const w = i === slices - 1 ? 0.5 : 1.0;
          accx += w * ix; accy += w * iy;
        }
        xix += INV_4PI * accx * ds; xiy += INV_4PI * accy * ds;
        const t2 = h * Math.tan(el.e2 || 0.0);
        xix += INV_4PI * bx * t2; xiy += -INV_4PI * by * t2;
        advance(edgeMatrix(h, el.e2 || 0.0));
      } else {
        advance(elementMatrix(el, ref));
      }
    }
    return [xix, xiy];
  }
  /** accsim.twiss._sextupole_feeddown. */
  function sextupoleFeeddown(elements, ref, slices) {
    slices = slices || 64;
    const { M } = transferMap(elements, ref);
    const tw0 = matchPeriodic(M);
    let bx = tw0.beta_x, ax = tw0.alpha_x, by = tw0.beta_y, ay = tw0.alpha_y;
    let disp = [tw0.disp_x, tw0.disp_px, tw0.disp_y, tw0.disp_py];
    let xix = 0.0, xiy = 0.0;
    for (const el of elements) {
      const Mm = elementMatrix(el, ref);
      const L = elementLength(el);
      if (el.type === "ThinSextupole") {
        xix += +INV_4PI * bx * el.k2l * disp[0];
        xiy += -INV_4PI * by * el.k2l * disp[0];
      } else if (el.type === "Sextupole" && el.k2 !== 0.0 && L > 0.0) {
        const ds = L / slices;
        const db = focusingBlock(0.0, ds);
        let ix = 0.5 * bx * disp[0], iy = 0.5 * by * disp[0];
        for (let i = 0; i < slices; i++) {
          const px = propagateBlock(db, bx, ax), py = propagateBlock(db, by, ay);
          bx = px.beta; ax = px.alpha; by = py.beta; ay = py.alpha;
          disp[0] += disp[1] * ds; disp[2] += disp[3] * ds;
          const w = i === slices - 1 ? 0.5 : 1.0;
          ix += w * bx * disp[0]; iy += w * by * disp[0];
        }
        xix += +INV_4PI * el.k2 * ix * ds;
        xiy += -INV_4PI * el.k2 * iy * ds;
        continue;
      }
      const [cx, cy] = blocks(Mm);
      const px = propagateBlock(cx, bx, ax), py = propagateBlock(cy, by, ay);
      bx = px.beta; ax = px.alpha; by = py.beta; ay = py.alpha;
      disp = vadd(matvec(transverse4(Mm), disp), dispersiveKick(Mm));
    }
    return [xix, xiy];
  }
  function chromaticity(elements, ref, slices) {
    const [nx, ny] = naturalChromaticity(elements, ref, slices);
    const [fx, fy] = sextupoleFeeddown(elements, ref, slices);
    return [nx + fx, ny + fy];
  }

  // ------------------------------------------------------------ longitudinal
  /** accsim.twiss.momentum_compaction, the exact "identity" route. */
  function momentumCompaction(elements, ref) {
    const { M } = transferMap(elements, ref);
    const tw0 = matchPeriodic(M);
    const C = elements.reduce((s, el) => s + elementLength(el), 0.0);
    const slip = M[ZETA][X] * tw0.disp_x + M[ZETA][PX] * tw0.disp_px + M[ZETA][DELTA];
    return 1.0 / (ref.gamma0 * ref.gamma0) - slip / C;
  }
  function slipFactor(elements, ref) {
    return momentumCompaction(elements, ref) - 1.0 / (ref.gamma0 * ref.gamma0);
  }
  /** accsim.twiss.synchrotron_tune (lumped-cavity small-amplitude formula). */
  function synchrotronTune(elements, ref) {
    const cavities = elements.filter((el) => el.type === "RFCavity");
    if (cavities.length === 0) return null;
    const eta = slipFactor(elements, ref);
    const C = elements.reduce((s, el) => s + elementLength(el), 0.0);
    const r65 = cavities.reduce((s, c) => s + rfSlope(c, ref), 0.0);
    const half = 1.0 - 0.5 * r65 * eta * C;
    if (Math.abs(half) >= 1.0) {
      const e = new Error(`no stable RF bucket: 1/2 Tr(M_s) = ${half.toPrecision(4)}`);
      e.code = "rf-unstable";
      throw e;
    }
    return Math.acos(half) / TWO_PI;
  }

  // ---------------------------------------------------------------- closed orbit
  /** accsim.orbit.closed_orbit at delta = 0: the 4D fixed point of the affine map. */
  function closedOrbit(elements, ref) {
    const { M, k } = transferMap(elements, ref);
    const m4 = transverse4(M);
    const rhs = [k[X], k[PX], k[Y], k[PY]];
    if (!rhs.some((v) => v !== 0.0)) return [0.0, 0.0, 0.0, 0.0];
    const A = eye(4).map((row, i) => row.map((v, j) => v - m4[i][j]));
    const cond = condProxy(A);
    if (!Number.isFinite(cond) || cond > 1e12) {
      const e = new Error("no closed orbit: integer tune in at least one plane");
      e.code = "no-orbit";
      throw e;
    }
    return solve(A, rhs);
  }
  /** Orbit (x, px, y, py) at every (sliced) boundary from `orbit0`. */
  function propagateOrbit(elements, ref, orbit0, slicesFor) {
    const points = [{ s: 0.0, o: orbit0.slice(), elem: -1 }];
    let o = orbit0.slice();
    let s = 0.0;
    elements.forEach((el, i) => {
      const pieces = slicesFor ? sliceElement(el, slicesFor(el)) : [el];
      for (const piece of pieces) {
        const M = elementMatrix(piece, ref), k = elementKick(piece, ref);
        const m4 = transverse4(M);
        o = vadd(matvec(m4, o), [k[X], k[PX], k[Y], k[PY]]);
        s += elementLength(piece);
        points.push({ s, o: o.slice(), elem: i });
      }
    });
    return points;
  }

  // --------------------------------------------------------------------- survey
  function sinc(u) { return u === 0.0 ? 1.0 : Math.sin(u) / u; }
  /** accsim.geometry.survey — planar; returns N+1 boundary poses. */
  function survey(elements, slicesFor) {
    const s = [0.0], Xl = [0.0], Zl = [0.0], theta = [0.0], elem = [-1];
    let x = 0.0, z = 0.0, th = 0.0, ss = 0.0;
    elements.forEach((el, i) => {
      const pieces = slicesFor ? sliceElement(el, slicesFor(el)) : [el];
      for (const piece of pieces) {
        const L = elementLength(piece), a = elementAngle(piece);
        const half = 0.5 * a;
        const dvx = -L * half * sinc(half) ** 2, dvz = L * sinc(a);
        // W(theta) @ (dvx, 0, dvz) with the yaw rotation [[c,0,s],[0,1,0],[-s,0,c]]
        const c = Math.cos(th), sn = Math.sin(th);
        x += c * dvx + sn * dvz;
        z += -sn * dvx + c * dvz;
        th -= a;
        ss += L;
        s.push(ss); Xl.push(x); Zl.push(z); theta.push(th); elem.push(i);
      }
    });
    return { s, X: Xl, Z: Zl, theta, elem };
  }

  // ------------------------------------------------------------------ analysis
  /**
   * Everything the editor shows, in one call. Never throws for a physics reason:
   * each quantity is either a number or carries its own `error`.
   */
  function analyse(scenario, options) {
    options = options || {};
    const out = { errors: [], warnings: [] };
    let ref;
    try { ref = reference(scenario.reference || {}); }
    catch (e) { out.errors.push(e.message); return out; }
    out.ref = ref;
    const raw = scenario.elements || [];
    const perElement = raw.map(validateElement);
    perElement.forEach((errs, i) => errs.forEach((m) => out.errors.push(`${raw[i].name || raw[i].type} #${i + 1}: ${m}`)));
    if (out.errors.length) return out;
    const els = resolve(raw, ref);
    out.elements = els;
    out.length = els.reduce((s, el) => s + elementLength(el), 0.0);
    out.totalAngle = els.reduce((s, el) => s + elementAngle(el), 0.0);
    const slicesFor = options.slicesFor || ((el) => {
      const L = elementLength(el);
      if (L <= 0) return 1;
      return Math.max(2, Math.min(40, Math.ceil(L / (options.plotStep || 0.1))));
    });
    try {
      out.matrices = els.map((el) => elementMatrix(el, ref));
      out.kicks = els.map((el) => elementKick(el, ref));
    } catch (e) { out.errors.push(e.message); return out; }
    const { M, k } = transferMap(els, ref);
    out.oneTurn = M;
    out.oneTurnKick = k;
    out.survey = survey(els, slicesFor);
    out.surveyBoundaries = survey(els);
    out.coupled = couplingNorm(M) > 1e-9;
    out.halfTrace = { x: 0.5 * (M[X][X] + M[PX][PX]), y: 0.5 * (M[Y][Y] + M[PY][PY]) };
    const periodic = scenario.periodic !== false;
    out.periodic = periodic;
    let tw0 = null;
    if (periodic) {
      try { tw0 = matchPeriodic(M); }
      catch (e) { out.opticsError = { code: e.code || "error", message: e.message }; }
    } else {
      const t = scenario.initial_twiss || {};
      tw0 = { s: 0.0, beta_x: +t.beta_x || 10.0, alpha_x: +t.alpha_x || 0.0, mu_x: 0.0,
        beta_y: +t.beta_y || 10.0, alpha_y: +t.alpha_y || 0.0, mu_y: 0.0,
        disp_x: +t.disp_x || 0.0, disp_px: +t.disp_px || 0.0, disp_y: 0.0, disp_py: 0.0 };
      if (out.coupled) out.opticsError = { code: "coupled", message: "the line is x-y coupled; the uncoupled Twiss propagation would be wrong" };
    }
    if (tw0 && !out.opticsError) {
      out.twiss0 = tw0;
      out.twiss = propagateTwiss(els, ref, tw0, slicesFor);
      out.twissBoundaries = propagateTwiss(els, ref, tw0);
      const end = out.twissBoundaries[out.twissBoundaries.length - 1];
      out.tunes = [end.mu_x / TWO_PI, end.mu_y / TWO_PI];
      out.twissEnd = end;
      if (periodic) {
        try { out.chromaticity = chromaticity(els, ref); out.naturalChromaticity = naturalChromaticity(els, ref); }
        catch (e) { out.warnings.push("chromaticity: " + e.message); }
        try {
          out.alpha_c = momentumCompaction(els, ref);
          out.eta = out.alpha_c - 1.0 / (ref.gamma0 * ref.gamma0);
          out.gamma_t = out.alpha_c > 0 ? 1.0 / Math.sqrt(out.alpha_c) : null;
        } catch (e) { out.warnings.push("momentum compaction: " + e.message); }
        try { out.Qs = synchrotronTune(els, ref); }
        catch (e) { out.Qs = null; out.warnings.push(e.message); }
      }
    }
    // Closed orbit from correctors / misalignments (ring) or trajectory (line).
    const anyKick = k.slice(0, 4).some((v) => v !== 0.0);
    out.hasKicks = anyKick;
    try {
      if (periodic) {
        if (anyKick) {
          out.closedOrbit0 = closedOrbit(els, ref);
          out.orbit = propagateOrbit(els, ref, out.closedOrbit0, slicesFor);
        }
      } else if (anyKick) {
        out.orbit = propagateOrbit(els, ref, [0, 0, 0, 0], slicesFor);
      }
    } catch (e) { out.warnings.push("orbit: " + e.message); }
    return out;
  }

  return {
    CLIGHT, ELECTRON_MASS_EV, PROTON_MASS_EV, SPECIES, CATALOGUE,
    reference, validateElement, resolve, elementLength, elementAngle,
    elementMatrix, elementKick, transferMap, matchPeriodic, propagateTwiss, tunes,
    naturalChromaticity, sextupoleFeeddown, chromaticity, momentumCompaction, slipFactor,
    synchrotronTune, closedOrbit, propagateOrbit, survey, sliceElement, analyse,
    _linalg: { eye, matmul, matvec, solve },
  };
});
