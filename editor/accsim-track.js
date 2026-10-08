/*
 * accsim-track.js — turn-by-turn tracking for the accsim editor.
 *
 * A faithful port of accsim's element-by-element tracker (`Tracker.track_once`, i.e.
 * every element's `track()`), the path that is NOT the one-turn matrix: the exact
 * drift, the momentum-dependent thick quadrupole, the exact sector bend and the curved
 * combined-function body, the nonlinear multipole kicks, the solenoid, the wiggler's
 * second-power focusing, the RF cavity's sine, the pole-face fringes, and every
 * element's misalignment wrapper. Each map below names the Python function it ports;
 * the arithmetic is kept in the same order so the two agree to round-off, and
 * `tests/analytic/test_tracking_port.py` holds them together.
 *
 * Why this path and not the matrix: the one-turn matrix carries no momentum dependence,
 * so a particle tracked with it has the same tune at every energy and a beam never
 * filaments. The element-by-element maps carry the whole natural chromaticity (measured
 * 2026-10-07: tracked dQ/ddelta equals `chromaticity()` on every ring preset). That is
 * the physics an injection view exists to show.
 *
 * What it deliberately omits: radiation (no damping, no quantum excitation — an electron
 * beam here filaments but never shrinks), spin, and the tapered bend. Losses are the
 * Python `track_bunch_losses` rule: a particle outside an Aperture after it, or outside
 * a Collimator at either of its two faces, is lost and frozen.
 *
 * Coordinates: (x, px, y, py, zeta, delta), Xsuite / MAD-X ordering, as everywhere.
 * Dependency: accsim-optics.js (reference particle, catalogue, resolve).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./accsim-optics.js"));
  else root.AccsimTrack = factory(root.AccsimOptics);
})(typeof self !== "undefined" ? self : this, function (O) {
  "use strict";
  const X = 0, PX = 1, Y = 2, PY = 3, ZETA = 4, DELTA = 5;
  const CLIGHT = 299792458.0;
  const SKEW_ROLL = Math.PI / 4.0;

  /** E/E0 of a particle at momentum (1 + delta) p0 — `np.hypot(p0 (1+d), m) / E0`. */
  function eOverE0(ref, onePlus) {
    return Math.hypot(ref.momentum_eV * onePlus, ref.mass_eV) / ref.total_energy_eV;
  }
  function sinc(z) { return z === 0.0 ? 1.0 : Math.sin(z) / z; }

  // ------------------------------------------------------------------- drift
  /** accsim Drift._track_body — the exact field-free map. In place. */
  function driftExact(s, L, ref) {
    if (L === 0.0) return;
    const px = s[PX], py = s[PY], delta = s[DELTA];
    const onePlus = 1.0 + delta;
    const angleSq = px * px + py * py;
    const pz = Math.sqrt(onePlus * onePlus - angleSq);
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = delta * (2.0 + delta) / g2 - angleSq;
    s[X] = s[X] + L * px / pz;
    s[Y] = s[Y] + L * py / pz;
    s[ZETA] = s[ZETA] + L * slip / (pz * (pz + E));
  }

  // ------------------------------------------------------------- quadrupole
  /** accsim.elements.quadrupole._focusing_functions — (C, S) of u'' + g u = 0. */
  function focusingFunctions(g, L) {
    const u = Math.sqrt(Math.abs(g)) * L;
    if (u === 0.0) return [g >= 0.0 ? 1.0 : 1.0, L];
    if (g >= 0.0) return [Math.cos(u), L * (Math.sin(u) / u)];
    return [Math.cosh(u), L * (Math.sinh(u) / u)];
  }
  /** accsim.elements.quadrupole._path_lengthening. */
  function pathLengthening(g, u0, up0, L, C, S) {
    const T = 0.5 * (L - C * S);
    return 0.5 * (g * u0 * u0 * T - g * u0 * up0 * S * S + up0 * up0 * (L - T));
  }
  /** accsim.elements.quadrupole.thick_quadrupole_map (kinematic_slices = 0). In place. */
  function thickQuad(s, L, k1, ref) {
    if (L === 0.0) return;
    const delta = s[DELTA], onePlus = 1.0 + delta;
    const Kx = k1 / onePlus;
    const [Cx, Sx] = focusingFunctions(Kx, L);
    const [Cy, Sy] = focusingFunctions(-Kx, L);
    const x = s[X], y = s[Y];
    const xp = s[PX] / onePlus, yp = s[PY] / onePlus;
    s[X] = x * Cx + xp * Sx;
    s[PX] = (-Kx * x * Sx + xp * Cx) * onePlus;
    s[Y] = y * Cy + yp * Sy;
    s[PY] = (+Kx * y * Sy + yp * Cy) * onePlus;
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = L * delta * (2.0 + delta) / g2 / (onePlus * (onePlus + E));
    const path = pathLengthening(Kx, x, xp, L, Cx, Sx) + pathLengthening(-Kx, y, yp, L, Cy, Sy);
    s[ZETA] = s[ZETA] + slip - path * E / onePlus;
  }

  // ------------------------------------------------------------------ bends
  /** accsim.elements.dipole.exact_sector_bend_map. In place. */
  function exactSectorBend(s, L, h, ref) {
    if (L === 0.0) return;
    const x = s[X], px = s[PX], py = s[PY], delta = s[DELTA];
    const theta = h * L;
    const cosT = Math.cos(theta);
    const sincT = sinc(theta);
    const sh = sinc(0.5 * theta);
    const halfChord = 0.5 * L * L * (sh * sh);
    const onePlus = 1.0 + delta;
    const angleSq = px * px + py * py;
    const pz = Math.sqrt(onePlus * onePlus - angleSq);
    const u = (delta * (2.0 + delta) - angleSq) / (pz + 1.0);
    const C = u - h * x;
    const pxOut = px * cosT + C * h * L * sincT;
    const pzOut = Math.sqrt(onePlus * onePlus - pxOut * pxOut - py * py);
    const Q = px * h * halfChord - C * L * sincT;
    const xOut = x * cosT + px * L * sincT + Q * (px + pxOut) / (pzOut + pz) + u * h * halfChord;
    const invPPerp = 1.0 / Math.sqrt(onePlus * onePlus - py * py);
    const a = px * invPPerp, b = pxOut * invPPerp;
    const sigma = Math.sqrt(1.0 - a * a) + Math.sqrt(1.0 - b * b);
    const scale = 0.5 * sigma + 0.5 * (a + b) * (a + b) / sigma;
    const w = (a - b) * scale;
    const arcsinc = w === 0.0 ? 1.0 : Math.asin(w) / w;
    const dOverH = arcsinc * scale * invPPerp * Q;
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = L * delta * (2.0 + delta) / g2 / (onePlus * (onePlus + E));
    const path = delta * L / onePlus + dOverH;
    s[X] = xOut;
    s[PX] = pxOut;
    s[Y] = s[Y] + py * (L + dOverH);
    s[ZETA] = s[ZETA] + slip - path * E;
  }
  /** accsim.elements.dipole._cfd_path_integrals — (c2, t1), with the small-K series. */
  function cfdPathIntegrals(K, L, C, S) {
    const z = K * L * L;
    if (Math.abs(z) < 0.01) {
      const v = -z, w = 4.0 * v, L3 = L * L * L;
      const c2 = -L3 * (1.0 / 6.0 + v * (1.0 / 120.0 + v * (1.0 / 5040.0 + v * (1.0 / 362880.0 + v / 39916800.0))));
      const t1 = 2.0 * L3 * (1.0 / 6.0 + w * (1.0 / 120.0 + w * (1.0 / 5040.0 + w * (1.0 / 362880.0 + w / 39916800.0))));
      return [c2, t1];
    }
    return [(S - L) / K, (L - C * S) / (2.0 * K)];
  }
  /** accsim.elements.dipole.expanded_cfd_map — the curved gradient body. In place. */
  function expandedCfd(s, L, h, k1, ref) {
    if (L === 0.0) return;
    const x = s[X], px = s[PX], y = s[Y], py = s[PY], delta = s[DELTA];
    const onePlus = 1.0 + delta;
    const Kx = (h * h + k1) / onePlus;
    const Ky = -k1 / onePlus;
    const G = h * delta / onePlus;
    const [Cx, Sx] = focusingFunctions(Kx, L);
    const [Cy, Sy] = focusingFunctions(Ky, L);
    const sHalf = focusingFunctions(Kx, 0.5 * L)[1];
    const c1 = 2.0 * (sHalf * sHalf);
    const [c2, t1x] = cfdPathIntegrals(Kx, L, Cx, Sx);
    const t1y = cfdPathIntegrals(Ky, L, Cy, Sy)[1];
    const xp = px / onePlus, yp = py / onePlus;
    const A = -Kx * x + G, B = xp;
    const Cv = -Ky * y, D = yp;
    s[X] = x * Cx + xp * Sx + G * c1;
    s[PX] = (A * Sx + B * Cx) * onePlus;
    s[Y] = y * Cy + yp * Sy;
    s[PY] = (Cv * Sy + D * Cy) * onePlus;
    let path = h * (x * Sx + xp * c1 - G * c2);
    path += 0.5 * (A * A * t1x + A * B * Sx * Sx + B * B * (L - Kx * t1x));
    path += 0.5 * (Cv * Cv * t1y + Cv * D * Sy * Sy + D * D * (L - Ky * t1y));
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = L * delta * (2.0 + delta) / g2 / (onePlus * (onePlus + E));
    s[ZETA] = s[ZETA] + slip - path * E / onePlus;
  }
  /** accsim.elements.dipole.curvature_sextupole_kick. In place. */
  function curvatureSextupoleKick(s, hk1l) {
    const x = s[X], y = s[Y];
    s[PX] = s[PX] + hk1l * (y * y * 0.5 - x * x);
    s[PY] = s[PY] + hk1l * x * y;
  }
  /** accsim.elements.dipole._edge_matrix applied to the state (exactly linear). */
  function edgeKick(s, h, e) {
    if (h === 0.0 || e === 0.0) return;
    const t = h * Math.tan(e);
    s[PX] = t * s[X] + s[PX];
    s[PY] = -t * s[Y] + s[PY];
  }
  /** accsim.elements.dipole._arcsinc — with its small-argument series. */
  function arcsincSeries(t) {
    if (Math.abs(t) < 0.0001) { const t2 = t * t; return 1.0 + t2 * (1.0 / 6.0 + t2 * 3.0 / 40.0); }
    return Math.asin(t) / t;
  }
  /** accsim.elements.dipole.wedge_map. In place. */
  function wedgeMap(s, theta, h, ref) {
    if (theta === 0.0) return;
    const x = s[X], px = s[PX], py = s[PY], delta = s[DELTA];
    const c = Math.cos(theta), sn = Math.sin(theta);
    const onePlus = 1.0 + delta;
    const q2 = onePlus * onePlus - py * py;
    const pz = Math.sqrt(q2 - px * px);
    const newPx = px * c + (pz - h * x) * sn;
    const newPz = Math.sqrt(q2 - newPx * newPx);
    const npx0 = px * c + pz * sn;
    const npz0 = pz * c - px * sn;
    const newX = x * c + (x * px * (2.0 * sn * c) + sn * sn * (2.0 * x * pz - h * x * x)) / (newPz + npz0);
    const w = (px * newPz - newPx * pz) / q2;
    const r = (pz * newPz + px * newPx) / q2;
    const vOverH = x * sn * (px * (npx0 + newPx) / (newPz + npz0) + pz) / q2 * (c + sn * (sn - w) / (r + c));
    const arcOverH = vOverH * arcsincSeries(h * vOverH);
    const E = eOverE0(ref, onePlus);
    s[X] = newX;
    s[PX] = newPx;
    s[Y] = s[Y] + py * arcOverH;
    s[ZETA] = s[ZETA] - E * arcOverH;
  }
  /** accsim.elements.dipole.hard_edge_fringe_map. In place. */
  function hardEdgeFringe(s, h, ref) {
    if (h === 0.0) return;
    const x = s[X], px = s[PX], py = s[PY], delta = s[DELTA];
    const onePlus = 1.0 + delta;
    const d = onePlus * onePlus - px * px;
    const pz = Math.sqrt(d - py * py);
    const phi = h * px * pz / d;
    const dphiDpx = h * ((pz * pz - px * px) / (pz * d) + 2.0 * px * px * pz / (d * d));
    const dphiDpy = -h * px * py / (pz * d);
    const dphiDdelta = h * px * onePlus * (py * py - pz * pz) / (pz * d * d);
    const yNew = 2.0 * s[Y] / (1.0 + Math.sqrt(1.0 - 2.0 * dphiDpy * s[Y]));
    const ySq = yNew * yNew;
    const E = eOverE0(ref, onePlus);
    s[X] = x + 0.5 * dphiDpx * ySq;
    s[Y] = yNew;
    s[PY] = py - phi * yNew;
    s[ZETA] = s[ZETA] + 0.5 * (E / onePlus) * dphiDdelta * ySq;
  }
  /** accsim.elements.fringe.multipole_fringe_map. In place. */
  function multipoleFringe(s, k1, ref, exitFace) {
    if (k1 === 0.0) return;
    const x = s[X], y = s[Y];
    const rpp = 1.0 / (1.0 + s[DELTA]);
    const kappa = exitFace ? -k1 : k1;
    const x2 = x * x, y2 = y * y, xy = x * y;
    const fx = -kappa * (x2 * x + 3.0 * x * y2) / 12.0;
    const fy = kappa * (3.0 * x2 * y + y2 * y) / 12.0;
    const fxx = -kappa * (x2 + y2) * 0.25;
    const fxy = -kappa * xy * 0.5;
    const fyx = kappa * xy * 0.5;
    const fyy = kappa * (x2 + y2) * 0.25;
    const a = 1.0 - fxx * rpp, b = -fyx * rpp, c = -fxy * rpp, d = 1.0 - fyy * rpp;
    const det = a * d - b * c;
    const newPx = (d * s[PX] - b * s[PY]) / det;
    const newPy = (a * s[PY] - c * s[PX]) / det;
    const E = eOverE0(ref, 1.0 + s[DELTA]);
    s[X] = x - fx * rpp;
    s[Y] = y - fy * rpp;
    s[PX] = newPx;
    s[PY] = newPy;
    s[ZETA] = s[ZETA] + E * rpp * rpp * rpp * (newPx * fx + newPy * fy);
  }
  /** accsim.elements.fringe.quad_wedge_map. In place. */
  function quadWedge(s, theta, k1) {
    const strength = k1 * theta;
    if (strength === 0.0) return;
    const x = s[X], y = s[Y];
    s[PX] = s[PX] + strength * (y * y * 0.5 - x * x);
    s[PY] = s[PY] + strength * x * y;
  }
  /** accsim Dipole._face — one nonlinear pole face (fringe=True). In place. */
  function dipoleFace(s, e, h, k1, ref, exitFace) {
    if (exitFace) {
      wedgeMap(s, -e, h, ref);
      quadWedge(s, -e, k1);
      multipoleFringe(s, k1, ref, true);
      hardEdgeFringe(s, -h, ref);
      wedgeMap(s, e, 0.0, ref);
      return;
    }
    wedgeMap(s, e, 0.0, ref);
    hardEdgeFringe(s, h, ref);
    multipoleFringe(s, k1, ref, false);
    quadWedge(s, -e, k1);
    wedgeMap(s, -e, h, ref);
  }
  /** accsim Dipole._track_nominal (untapered). In place. */
  function dipole(s, el, ref) {
    const L = +el.length, k1 = +el.k1 || 0.0;
    const e1 = +el.e1 || 0.0, e2 = +el.e2 || 0.0;
    const h = L > 0.0 ? +el.angle / L : 0.0;
    const fringe = !!el.fringe;
    if (fringe) dipoleFace(s, e1, h, k1, ref, false);
    else edgeKick(s, h, e1);
    if (k1 === 0.0) {
      exactSectorBend(s, L, h, ref);
    } else {
      const half = 0.5 * L;
      expandedCfd(s, half, h, k1, ref);
      curvatureSextupoleKick(s, h * k1 * L);
      expandedCfd(s, half, h, k1, ref);
    }
    if (fringe) dipoleFace(s, e2, h, k1, ref, true);
    else edgeKick(s, h, e2);
  }

  // ------------------------------------------------------------- multipoles
  /** accsim.elements.sextupole._apply_kick. In place. */
  function sextupoleKick(s, k2l) {
    const x = s[X], y = s[Y];
    s[PX] = s[PX] - 0.5 * k2l * (x * x - y * y);
    s[PY] = s[PY] + k2l * (x * y);
  }
  /** accsim.elements.octupole._apply_kick. In place. */
  function octupoleKick(s, k3l) {
    const x = s[X], y = s[Y];
    s[PX] = s[PX] - k3l * (x * x * x - 3.0 * x * y * y) / 6.0;
    s[PY] = s[PY] + k3l * (3.0 * x * x * y - y * y * y) / 6.0;
  }
  /** Sextupole/Octupole._track_body: drift-kick-drift, n_slices = 1. In place. */
  function thickMultipole(s, L, kl, kick, ref) {
    if (kl === 0.0) { driftExact(s, L, ref); return; }
    const n = 1, half = 0.5 * L / n, klSlice = kl / n;
    for (let i = 0; i < n; i++) {
      driftExact(s, half, ref);
      kick(s, klSlice);
      driftExact(s, half, ref);
    }
  }

  // ------------------------------------------------------- rotation / solenoid
  /** accsim.elements.alignment.s_rotation(phi) applied to the state. In place. */
  function sRotate(s, phi) {
    if (phi === 0.0) return;
    const c = Math.cos(phi), sn = Math.sin(phi);
    const x = s[X], px = s[PX], y = s[Y], py = s[PY];
    s[X] = c * x + sn * y;
    s[PX] = c * px + sn * py;
    s[Y] = -sn * x + c * y;
    s[PY] = -sn * px + c * py;
  }
  /** accsim Solenoid._track_body. In place. */
  function solenoid(s, L, ks, ref) {
    if (L === 0.0) return;
    const delta = s[DELTA], onePlus = 1.0 + delta;
    const K = 0.5 * ks / onePlus;
    const u = K * L;
    const C = Math.cos(u), Lsinc = L * sinc(u), S = Math.sin(K * L);
    const CC = C * C, SC_K = Lsinc * C, SC = S * C, SS_K = Lsinc * S, K_SS = K * S * S;
    const x = s[X], y = s[Y];
    const xp = s[PX] / onePlus, yp = s[PY] / onePlus;
    s[X] = CC * x + SC_K * xp + SC * y + SS_K * yp;
    s[PX] = (-K * SC * x + CC * xp - K_SS * y + SC * yp) * onePlus;
    s[Y] = -SC * x - SS_K * xp + CC * y + SC_K * yp;
    s[PY] = (K_SS * x - SC * xp - K * SC * y + CC * yp) * onePlus;
    const xIn = xp + K * y, yIn = yp - K * x;
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = L * delta * (2.0 + delta) / g2 / (onePlus * (onePlus + E));
    const path = 0.5 * L * (xIn * xIn + yIn * yIn);
    s[ZETA] = s[ZETA] + slip - path * E / onePlus;
  }

  // ---------------------------------------------------------- wiggler / RF
  /** accsim Wiggler._track_body — K_y carries the SECOND power of (1 + delta). In place. */
  function wiggler(s, el, ref) {
    const period = +el.period, h0 = +el.h0;
    const L = period * (+el.periods);
    const delta = s[DELTA], onePlus = 1.0 + delta;
    const focusing = 0.5 * h0 * h0;
    const deflection = h0 / (2.0 * Math.PI / period);
    const Ky = focusing / (onePlus * onePlus);
    const [Cy, Sy] = focusingFunctions(Ky, L);
    const y = s[Y];
    const xp = s[PX] / onePlus, yp = s[PY] / onePlus;
    s[X] = s[X] + L * xp;
    s[Y] = y * Cy + yp * Sy;
    s[PY] = (-Ky * y * Sy + yp * Cy) * onePlus;
    const E = eOverE0(ref, onePlus);
    const g2 = ref.gamma0 * ref.gamma0;
    const slip = L * delta * (2.0 + delta) / g2 / (onePlus * (onePlus + E));
    const dfl = deflection / onePlus;
    const path = 0.5 * L * (xp * xp + yp * yp + Ky * y * y) + 0.25 * L * (dfl * dfl);
    s[ZETA] = s[ZETA] + slip - path * E / onePlus;
  }
  /** accsim RFCavity._track_body — the exact energy kick. In place. */
  function rfCavity(s, el, ref) {
    const f = +el.frequency || 0.0, phi = +el.phi_s || 0.0, V = +el.voltage || 0.0;
    const k = 2.0 * Math.PI * f / (ref.beta0 * CLIGHT);
    const amp = ref.charge * V / (ref.beta0 * ref.beta0 * ref.total_energy_eV);
    const onePD = 1.0 + s[DELTA];
    const dptau = ref.beta0 * (amp * (Math.sin(phi - k * s[ZETA]) - Math.sin(phi)));
    const eOverP0c = Math.hypot(onePD, ref.mass_eV / ref.momentum_eV);
    const q = dptau * (2.0 * eOverP0c + dptau);
    s[DELTA] = s[DELTA] + q / (Math.sqrt(onePD * onePD + q) + onePD);
  }

  // ----------------------------------------------------------- the elements
  /** The element's own-frame map (accsim `_track_body`), as an in-place step. */
  function bodyStep(el, ref) {
    const L = O.elementLength(el);
    switch (el.type) {
      case "Drift": return (s) => driftExact(s, L, ref);
      case "Quadrupole": { const k1 = +el.k1; return (s) => thickQuad(s, L, k1, ref); }
      case "ThinQuadrupole": {
        const k = +el.k1l;
        return (s) => { s[PX] = -k * s[X] + s[PX]; s[PY] = k * s[Y] + s[PY]; };
      }
      case "Dipole": return (s) => dipole(s, el, ref);
      case "Sextupole": { const kl = (+el.k2) * L; return (s) => thickMultipole(s, L, kl, sextupoleKick, ref); }
      case "ThinSextupole": { const kl = +el.k2l; return (s) => sextupoleKick(s, kl); }
      case "Octupole": { const kl = (+el.k3) * L; return (s) => thickMultipole(s, L, kl, octupoleKick, ref); }
      case "ThinOctupole": { const kl = +el.k3l; return (s) => octupoleKick(s, kl); }
      case "SkewQuadrupole": {
        const k1s = +el.k1s;
        return (s) => { sRotate(s, -SKEW_ROLL); thickQuad(s, L, k1s, ref); sRotate(s, +SKEW_ROLL); };
      }
      case "ThinSkewQuadrupole": {
        const k = +el.k1sl;
        return (s) => { const x = s[X], y = s[Y]; s[PX] = k * y + s[PX]; s[PY] = k * x + s[PY]; };
      }
      case "Solenoid": { const ks = +el.ks; return (s) => solenoid(s, L, ks, ref); }
      case "RFCavity": return (s) => rfCavity(s, el, ref);
      case "Corrector": {
        const kx = +el.kick_x || 0.0, ky = +el.kick_y || 0.0;
        return (s) => { s[PX] = s[PX] + kx; s[PY] = s[PY] + ky; };
      }
      case "Wiggler": return (s) => wiggler(s, el, ref);
      // Acceptance boundaries (accsim AcceptanceElement): no field, so the exact drift of
      // their length (a no-op when thin); their physics is the survival predicate below.
      case "Aperture": case "Collimator": return (s) => driftExact(s, L, ref);
      default: throw new Error("tracking: no map for element type " + el.type);
    }
  }
  /** accsim Element._track_placed: the body wrapped in the element's alignment. */
  function elementStep(el, ref) {
    const body = bodyStep(el, ref);
    const dx = +el.dx || 0.0, dy = +el.dy || 0.0, roll = +el.roll || 0.0;
    if (dx === 0.0 && dy === 0.0 && roll === 0.0) return body;
    // The editor's optics refuse a rolled or displaced *bending* Dipole (a rolled bend needs
    // K2's curved-body geometry); a straight one (angle 0) is rolled like any magnet.
    if (el.type === "Dipole" && +el.angle !== 0.0) throw new Error("tracking: a misaligned bend is not modelled");
    if (roll === 0.0) {
      return (s) => { s[X] = s[X] - dx; s[Y] = s[Y] - dy; body(s); s[X] = s[X] + dx; s[Y] = s[Y] + dy; };
    }
    // body_in = R(roll) s - R(roll) d;  out = R(-roll) body(body_in) + d.
    const c = Math.cos(roll), sn = Math.sin(roll);
    const kinX = -(c * dx + sn * dy), kinY = -(-sn * dx + c * dy);
    return (s) => {
      sRotate(s, roll);
      s[X] = s[X] + kinX; s[Y] = s[Y] + kinY;
      body(s);
      sRotate(s, -roll);
      s[X] = s[X] + dx; s[Y] = s[Y] + dy;
    };
  }
  /** accsim Aperture.survives, or null for an element that is not an acceptance boundary. */
  function survivalTest(el) {
    if (el.type !== "Aperture" && el.type !== "Collimator") return null;
    const shape = el.shape || "rectangular";
    const hx = +el.half_x;
    const hy = shape === "circular" ? hx : +el.half_y;
    if (shape === "rectangular") return (s) => Math.abs(s[X]) <= hx && Math.abs(s[Y]) <= hy;
    return (s) => { const u = s[X] / hx, v = s[Y] / hy; return u * u + v * v <= 1.0; };
  }

  /**
   * Compile a scenario into a tracker. Throws on anything the editor's analysis would
   * refuse (an invalid element, a design tilt, a misaligned bend).
   */
  function build(scenario) {
    const ref = O.reference(scenario.reference || {});
    const raw = scenario.elements || [];
    raw.forEach((el, i) => {
      const errs = O.validateElement(el);
      if (errs.length) throw new Error(`${el.name || el.type} #${i + 1}: ${errs[0]}`);
      if (+el.tilt) throw new Error("tracking: a design tilt is not modelled in the editor");
    });
    const elements = O.resolve(raw, ref);
    const steps = elements.map((el) => elementStep(el, ref));
    const survive = elements.map(survivalTest);
    const thick = elements.map((el, i) => survive[i] !== null && O.elementLength(el) > 0.0);
    return { ref, elements, steps, survive, thick };
  }

  /** One pass of one particle (a 6-array, updated in place). Returns the index of the
   *  element where it was lost, or -1 if it survived the turn. A thick boundary (a
   *  Collimator) is tested at its entry face — a particle lost there is frozen before the
   *  jaw moves it — and at its exit face; a thin one once. Exact for a drift through a
   *  convex opening, as in `track_bunch_losses`. */
  function trackTurn(machine, s) {
    const { steps, survive, thick } = machine;
    for (let i = 0; i < steps.length; i++) {
      const t = survive[i];
      if (thick[i] && !t(s)) return i;
      steps[i](s);
      if (t !== null && !t(s)) return i;
    }
    return -1;
  }

  /**
   * Track a bunch for `nTurns`. `states` is an array of 6-arrays (updated in place), `lost`
   * an Int32Array of the turn each particle was lost on (-1 = alive) — a lost particle is
   * frozen where it was lost, exactly as `track_bunch_losses` does. `onTurn(turn)` is
   * called after every turn (for turn-by-turn statistics).
   */
  function trackBunch(machine, states, lost, nTurns, onTurn, turn0) {
    const t0 = turn0 || 0;
    for (let t = 0; t < nTurns; t++) {
      for (let p = 0; p < states.length; p++) {
        if (lost[p] >= 0) continue;
        if (trackTurn(machine, states[p]) >= 0) lost[p] = t0 + t;
      }
      if (onTurn) onTurn(t0 + t);
    }
  }

  return {
    build, trackTurn, trackBunch,
    _maps: {
      driftExact, thickQuad, exactSectorBend, expandedCfd, wedgeMap, hardEdgeFringe,
      multipoleFringe, solenoid, wiggler, rfCavity, sRotate,
    },
  };
});
