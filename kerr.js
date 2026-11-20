// Kerr black hole physics in Cartesian Kerr–Schild coordinates (units G = c = M = 1, spin along +z).
// Used by the page (camera frame, uniforms) and by tests/gargantua.test.mjs (the 64-bit reference the GPU shader
// is checked against). The shader in shader.js implements the same equations; keep the two in step.
//
//   g_μν = η_μν + f l_μ l_ν      f = 2r³ / (r⁴ + a²z²)      l_μ = (1, (rx + ay)/(r² + a²), (ry − ax)/(r² + a²), z/r)
//   g^μν = η^μν − f l^μ l^ν      (exact inverse, because l is null)
//
// Nothing here is singular at the horizons, so the same code works outside, across and inside the hole.
// Light obeys H = ½ g^μν p_μ p_ν = 0: dx^μ/dλ = ∂H/∂p_μ, dp_μ/dλ = −∂H/∂x^μ. The metric is stationary, so p_t = −E
// is constant and only x, y, z and p_x, p_y, p_z are integrated.

// --- Dual numbers [value, ∂x, ∂y, ∂z]: forward-mode automatic differentiation for the metric gradients ---
const d = (v, gx = 0, gy = 0, gz = 0) => [v, gx, gy, gz];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s, a[3] * s];
const mul = (a, b) => [a[0] * b[0], a[0] * b[1] + b[0] * a[1], a[0] * b[2] + b[0] * a[2], a[0] * b[3] + b[0] * a[3]];
const div = (a, b) => { const q = a[0] / b[0], i = 1 / b[0]; return [q, (a[1] - q * b[1]) * i, (a[2] - q * b[2]) * i, (a[3] - q * b[3]) * i]; };
const sqrtD = (a) => { const s = Math.sqrt(a[0]), k = 0.5 / s; return [s, a[1] * k, a[2] * k, a[3] * k]; };

/** Boyer–Lindquist-like radius r of a Cartesian point (the root of x²+y²)/(r²+a²) + z²/r² = 1). */
export function radius(x, y, z, a) {
  const w = x * x + y * y + z * z - a * a;
  return Math.sqrt(0.5 * w + Math.sqrt(0.25 * w * w + a * a * z * z));
}

/** f and the null vector l (covariant spatial part; l_t = 1) at a point, plain numbers. */
export function metricAt(x, y, z, a) {
  const r = radius(x, y, z, a), r2 = r * r, den = r2 + a * a;
  return { r, f: (2 * r2 * r) / (r2 * r2 + a * a * z * z), l: [1, (r * x + a * y) / den, (r * y - a * x) / den, z / r] };
}

/** f, l and their gradients, by automatic differentiation. */
export function metricDual(x, y, z, a) {
  const X = d(x, 1), Y = d(y, 0, 1), Z = d(z, 0, 0, 1), A2 = a * a;
  const w = sub(add(add(mul(X, X), mul(Y, Y)), mul(Z, Z)), d(A2));
  const r2 = add(scl(w, 0.5), sqrtD(add(scl(mul(w, w), 0.25), scl(mul(Z, Z), A2))));
  const r = sqrtD(r2);
  const f = div(scl(mul(r2, r), 2), add(mul(r2, r2), scl(mul(Z, Z), A2)));
  const den = add(r2, d(A2));
  const lx = div(add(mul(r, X), scl(Y, a)), den);
  const ly = div(sub(mul(r, Y), scl(X, a)), den);
  const lz = div(Z, r);
  return { r: r[0], f, lx, ly, lz };
}

/** Right-hand side of Hamilton's equations. s = [x, y, z, px, py, pz]; pt = p_t (constant). */
export function deriv(s, pt, a) {
  const [x, y, z, px, py, pz] = s;
  const { f, lx, ly, lz } = metricDual(x, y, z, a);
  const L = -pt + lx[0] * px + ly[0] * py + lz[0] * pz; // l^μ p_μ with l^t = −1
  const fL = f[0] * L;
  const out = [px - fL * lx[0], py - fL * ly[0], pz - fL * lz[0], 0, 0, 0];
  for (let i = 1; i <= 3; i++) out[2 + i] = 0.5 * f[i] * L * L + fL * (px * lx[i] + py * ly[i] + pz * lz[i]);
  return out;
}

/** dt/dλ (not integrated, but needed for light travel time and checks). */
export function dtdl(s, pt, a) {
  const { f, l } = metricAt(s[0], s[1], s[2], a);
  const L = -pt + l[1] * s[3] + l[2] * s[4] + l[3] * s[5];
  return -pt + f * L;
}

/** H = ½ g^μν p_μ p_ν (must stay 0 for light). */
export function hamiltonian(s, pt, a) {
  const { f, l } = metricAt(s[0], s[1], s[2], a);
  const L = -pt + l[1] * s[3] + l[2] * s[4] + l[3] * s[5];
  return 0.5 * (-pt * pt + s[3] * s[3] + s[4] * s[4] + s[5] * s[5] - f * L * L);
}

/** Axial angular momentum L_z = x p_y − y p_x (conserved: the metric is symmetric about z). */
export const angularMomentum = (s) => s[0] * s[4] - s[1] * s[3];

// --- Vectors: 4-vectors are [t, x, y, z] ---

/** g(u, v) at a point. */
export function dot4(m, u, v) {
  const lu = m.l[0] * u[0] + m.l[1] * u[1] + m.l[2] * u[2] + m.l[3] * u[3];
  const lv = m.l[0] * v[0] + m.l[1] * v[1] + m.l[2] * v[2] + m.l[3] * v[3];
  return -u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + u[3] * v[3] + m.f * lu * lv;
}

/** Lower an index: v_μ = g_μν v^ν. */
export function lower(m, v) {
  const lv = m.l[0] * v[0] + m.l[1] * v[1] + m.l[2] * v[2] + m.l[3] * v[3];
  return [-v[0] + m.f * m.l[0] * lv, v[1] + m.f * m.l[1] * lv, v[2] + m.f * m.l[2] * lv, v[3] + m.f * m.l[3] * lv];
}

/**
 * Orthonormal frame (tetrad) of an observer with 4-velocity u at position pos, with its forward / right / up axes
 * as close as possible to the given Cartesian directions (metric-aware Gram–Schmidt; forward kept exact first).
 * Returns contravariant vectors { e0, fwd, right, up }.
 */
export function tetrad(pos, u, fwd, right, up, a) {
  const m = metricAt(pos[0], pos[1], pos[2], a);
  const basis = [];
  for (const w of [fwd, right, up]) {
    let v = [0, w[0], w[1], w[2]];
    const c0 = dot4(m, v, u); // remove the time part: v + g(v,u) u   (g(u,u) = −1)
    v = v.map((x, i) => x + c0 * u[i]);
    for (const e of basis) { const c = dot4(m, v, e); v = v.map((x, i) => x - c * e[i]); }
    const n = Math.sqrt(dot4(m, v, v));
    basis.push(v.map((x) => x / n));
  }
  return { e0: u, fwd: basis[0], right: basis[1], up: basis[2] };
}

/** 4-velocity of an observer at rest in these coordinates (exists outside the ergosphere, where f < 1). */
export function staticObserver(pos, a) {
  const { f } = metricAt(pos[0], pos[1], pos[2], a);
  if (f >= 1) throw new Error('no static observer inside the ergosphere');
  return [1 / Math.sqrt(1 - f), 0, 0, 0];
}

/**
 * The locally non-rotating observer (ZAMO / "FIDO", Bardeen 1972), the frame the film's camera was placed in
 * (James et al. 2015). Its 4-velocity is normal to the Boyer–Lindquist time slices: u_μ = −α ∂_μ t_BL. In these
 * (ingoing Kerr–Schild) coordinates t_BL = t − ∫ 2r/Δ dr, so dt_BL = dt − (2r/Δ) dr. Unlike the static observer it
 * exists everywhere outside the horizon, including the ergosphere. α (the lapse) is how fast its clock runs
 * against a clock far away: dτ/dt_BL = α.
 * Returns { u (contravariant), alpha }.
 */
export function zamo(pos, a) {
  const [x, y, z] = pos, m = metricAt(x, y, z, a), r = m.r, rho2 = x * x + y * y + z * z;
  // gradient of r from r⁴ − (ρ² − a²) r² − a² z² = 0
  const dF = 4 * r * r * r - 2 * (rho2 - a * a) * r;
  const dr = [2 * x * r * r / dF, 2 * y * r * r / dF, (2 * z * r * r + 2 * a * a * z) / dF];
  const delta = r * r - 2 * r + a * a;
  if (!(delta > 0)) throw new Error('no ZAMO at or inside the horizon');
  const k = 2 * r / delta, tau = [1, -k * dr[0], -k * dr[1], -k * dr[2]]; // dt_BL as a covector
  // raise with g^μν = η^μν − f l^μ l^ν, l^μ = (−1, l_x, l_y, l_z)
  const lu = [-1, m.l[1], m.l[2], m.l[3]], lt = lu[0] * tau[0] + lu[1] * tau[1] + lu[2] * tau[2] + lu[3] * tau[3];
  const up = [-tau[0] - m.f * lu[0] * lt, tau[1] - m.f * lu[1] * lt, tau[2] - m.f * lu[2] * lt, tau[3] - m.f * lu[3] * lt];
  const n2 = -(up[0] * tau[0] + up[1] * tau[1] + up[2] * tau[2] + up[3] * tau[3]);
  const alpha = 1 / Math.sqrt(n2);
  return { u: up.map((v) => -alpha * v), alpha };
}

/** Twist of the Kerr–Schild azimuth against Boyer–Lindquist: φ_KS = φ_BL + twist(r), 0 far away (∫ a/Δ dr). */
export function twist(r, a) {
  if (a === 0) return 0;
  const s = Math.sqrt(1 - a * a), rp = 1 + s, rm = 1 - s;
  return s > 1e-9 ? (a / (rp - rm)) * Math.log((r - rp) / (r - rm)) : -a / (r - 1);
}

/**
 * The ZAMO's local x, y, z axes (contravariant 4-vectors), tied to the distant sky: its orthonormal frame along the
 * Boyer–Lindquist directions ∂r, ∂θ, ∂φ (Bardeen 1972; these are mutually orthogonal and orthogonal to u), turned
 * into Cartesian-like axes with the Boyer–Lindquist angles: the standard frame, as in James et al. 2015. (The first
 * version Gram–Schmidt'ed the Kerr–Schild x, y, z directions. Suspected of turning the view while flying in, it was
 * measured: the shadow lands within 0.5° of where this frame puts it at 5 M, a = 0.6. The swirl of the lensed sky
 * the owner saw while zooming is real frame dragging.)
 * In these coordinates: ∂φ = (0, −y, x, 0); ∂θ = (0, x cotθ, y cotθ, −r sinθ);
 * ∂r|BL = ∂r|KS + (2r/Δ) ∂t + (a/Δ) ∂φ, with ∂r|KS = (0, cosφ_KS sinθ, sinφ_KS sinθ, cosθ).
 */
export function zamoAxes(pos, u, a) {
  const [x, y, z] = pos, m = metricAt(x, y, z, a), r = m.r, cos = z / r, sin = Math.sqrt(Math.max(1 - cos * cos, 0));
  if (sin < 1e-6) { // on the spin axis the polar directions degenerate: the coordinate axes are fine there
    const L = tetrad(pos, u, [1, 0, 0], [0, 1, 0], [0, 0, 1], a);
    return [L.fwd, L.right, L.up];
  }
  const D = r * r - 2 * r + a * a, phiKS = Math.atan2(y, x) - Math.atan2(a, r), cpK = Math.cos(phiKS), spK = Math.sin(phiKS);
  const dPhi = [0, -y, x, 0], dTh = [0, x * cos / sin, y * cos / sin, -r * sin];
  const dR = [2 * r / D, cpK * sin - (a / D) * y, spK * sin + (a / D) * x, cos];
  const unit = (w) => { const n = Math.sqrt(dot4(m, w, w)); return w.map((c) => c / n); };
  const er = unit(dR), et = unit(dTh), ep = unit(dPhi);
  const phi = phiKS - twist(r, a), cp = Math.cos(phi), sp = Math.sin(phi);
  const comb = (cr, ct, cf) => [0, 1, 2, 3].map((i) => cr * er[i] + ct * et[i] + cf * ep[i]);
  return [comb(sin * cp, cos * cp, -sp), comb(sin * sp, cos * sp, cp), comb(cos, -sin, 0)];
}

/**
 * Frame of a camera moving with velocity v (fraction of c, components along the ZAMO's local x, y, z axes) relative
 * to the ZAMO at pos, looking along the local directions right / up / fwd (unit 3-vectors in the same axes).
 * The ZAMO's spatial axes are zamoAxes(); the camera frame is that frame Lorentz-boosted by v (pure boost), then
 * turned to the look direction. Returns the contravariant
 * tetrad { e0, right, up, fwd } plus gamma and alpha (clock rate against far away = alpha / gamma).
 */
export function cameraTetrad(pos, v, right, up, fwd, a) {
  const Z = zamo(pos, a), E = zamoAxes(pos, Z.u, a);
  const sp = Math.hypot(v[0], v[1], v[2]), gamma = 1 / Math.sqrt(1 - sp * sp);
  let e0 = Z.u, axes = E;
  if (sp > 1e-12) {
    const nh = v.map((c) => c / sp);
    const En = [0, 1, 2, 3].map((i) => nh[0] * E[0][i] + nh[1] * E[1][i] + nh[2] * E[2][i]);
    e0 = [0, 1, 2, 3].map((i) => gamma * (Z.u[i] + sp * En[i]));
    axes = E.map((e, k) => [0, 1, 2, 3].map((i) => e[i] + nh[k] * ((gamma - 1) * En[i] + gamma * sp * Z.u[i])));
  }
  const comb = (w) => [0, 1, 2, 3].map((i) => w[0] * axes[0][i] + w[1] * axes[1][i] + w[2] * axes[2][i]);
  return { e0, right: comb(right), up: comb(up), fwd: comb(fwd), gamma, alpha: Z.alpha };
}

// --- Horizons and orbits ---
export const outerHorizon = (a) => 1 + Math.sqrt(1 - a * a);
export const innerHorizon = (a) => 1 - Math.sqrt(1 - a * a);
/** Innermost stable circular orbit, prograde (Bardeen, Press & Teukolsky 1972). */
export function isco(a) {
  const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a * a + z1 * z1);
  return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}
/** Prograde circular photon orbit (photon sphere radius in the equator). */
export const photonOrbit = (a) => 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
/** Keplerian angular velocity of a prograde circular orbit, Ω = 1 / (r^{3/2} + a). */
export const orbitalOmega = (r, a) => 1 / (Math.pow(r, 1.5) + a);
/** Retrograde circular photon orbit (the outer edge of the equatorial photon shell). */
export const photonOrbitRetro = (a) => 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
/** dt/dτ of a prograde circular equatorial orbit (Bardeen, Press & Teukolsky 1972): how much slower its clock runs. */
export const orbitUt = (r, a) => (Math.pow(r, 1.5) + a) / (Math.pow(r, 0.75) * Math.sqrt(Math.pow(r, 1.5) - 3 * Math.sqrt(r) + 2 * a));
/**
 * Frequency shift g = ν_seen / ν_emitted of light from the disk (gas on prograde circular orbits at Boyer–Lindquist r)
 * reaching the camera along a ray with E = −p_t, L = p_φ (energy 1 in the camera's frame; both conserved, the same
 * in Kerr–Schild and Boyer–Lindquist). The gas has u = u^t (∂t + Ω ∂φ), so −p·u = u^t (E − Ω L): Doppler shift and
 * gravitational redshift together. SHADE (Reality mode) uses the same formula.
 */
export const diskShift = (r, a, E, L) => 1 / (orbitUt(r, a) * (E - orbitalOmega(r, a) * L));

// --- Capture test (exact, from the constants of motion) ---
// A photon's radial motion in Kerr is governed by R(r) = [E(r²+a²) − aL]² − Δ[(L − aE)² + Q] ≥ 0 (Carter 1968),
// with E = −p_t, L = p_φ and Carter's constant Q. Kerr–Schild and Boyer–Lindquist coordinates differ only by
// functions of r in t and φ, so r, θ, p_θ and p_φ are the same in both. A photon moving inward at r can only turn
// back at a root of R; if R > 0 on all of (r₊, r), nothing can stop it: it is captured.

/** Constants of motion (E, L, Q) of a ray state s = [x, y, z, px, py, pz] with p_t = pt. */
export function constants(s, pt, a) {
  const [x, y, z, px, py, pz] = s;
  const r = radius(x, y, z, a), cos = z / r, sin2 = Math.max(1 - cos * cos, 1e-12), sin = Math.sqrt(sin2);
  const L = x * py - y * px;
  // x = (r cosφ − a sinφ) sinθ, y = (r sinφ + a cosφ) sinθ, z = r cosθ  ⇒  ∂x/∂θ = x cotθ, ∂y/∂θ = y cotθ, ∂z/∂θ = −r sinθ
  const pTh = (cos / sin) * (x * px + y * py) - r * sin * pz;
  const E = -pt;
  return { E, L, Q: pTh * pTh + cos * cos * (L * L / sin2 - a * a * E * E) };
}
/** Radial potential R(r) (M = 1). */
export function radialPotential(r, { E, L, Q }, a) {
  const A = E * (r * r + a * a) - a * L, D = r * r - 2 * r + a * a, B = L - a * E;
  return A * A - D * (B * B + Q);
}
/**
 * True if a photon moving inward at radius r can never turn back: R > 0 on (r₊, r). R is a quartic, so it has at
 * most two dips there: sample it and refine the lowest sample by golden-section search. (Only a ray whose dip
 * just touches zero, i.e. one that would orbit the hole almost forever, can be misjudged; it would use up its
 * step budget anyway.) The shader uses the same samples.
 */
export const CAPTURE_SAMPLES = 24;
export function captured(r, c, a) {
  const rH = outerHorizon(a), n = CAPTURE_SAMPLES, at = (i) => rH + (r - rH) * (i / n);
  let best = 0, bestR = Infinity;
  for (let i = 1; i < n; i++) { const v = radialPotential(at(i), c, a); if (v < bestR) { bestR = v; best = i; } }
  if (bestR <= 0) return false;
  let lo = at(best - 1), hi = at(best + 1);
  for (let k = 0; k < 12; k++) {
    const m1 = hi - 0.618 * (hi - lo), m2 = lo + 0.618 * (hi - lo);
    if (radialPotential(m1, c, a) < radialPotential(m2, c, a)) hi = m2; else lo = m1;
  }
  return radialPotential((lo + hi) / 2, c, a) > 0;
}

/** Below this radius a ray heading inward is checked for capture: just outside the retrograde photon orbit. */
export const captureRadius = (a) => 1.3 * 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));

// --- Ray tracing (reference integrators) ---

/** Initial state of a light ray reaching an observer (tetrad T) from local direction dir = [right, up, forward]. */
export function rayFromCamera(pos, T, dir, a) {
  const m = metricAt(pos[0], pos[1], pos[2], a);
  // The photon travels along −dir: p^μ = e0 − dir·e (energy 1 in the camera's frame).
  const p = [0, 1, 2, 3].map((i) => T.e0[i] - dir[0] * T.right[i] - dir[1] * T.up[i] - dir[2] * T.fwd[i]);
  const pl = lower(m, p);
  return { s: [pos[0], pos[1], pos[2], pl[1], pl[2], pl[3]], pt: pl[0] };
}

/** One RK4 step of Hamilton's equations (h > 0 forward in time, < 0 backward). Light or matter alike. */
export function stepGeodesic(s, pt, a, h) { return rk4(s, pt, a, h); }

/** Raise an index: p^μ = g^μν p_ν, with g^μν = η^μν − f l^μ l^ν and l^μ = (−1, l_x, l_y, l_z). */
export function raise(m, p) {
  const L = -p[0] + m.l[1] * p[1] + m.l[2] * p[2] + m.l[3] * p[3];
  return [-p[0] + m.f * L, p[1] - m.f * m.l[1] * L, p[2] - m.f * m.l[2] * L, p[3] - m.f * m.l[3] * L];
}

/**
 * A falling camera (stage 3): advance its timelike geodesic by proper time dtau. fall = { s, pt } with
 * s = [x, y, z, p_x, p_y, p_z] and p_t = pt, p the 4-velocity lowered (so p·p = −1 and λ is proper time).
 * Substeps keep each RK4 step under 2% of r. Returns the new fall state; nothing here is singular at either horizon.
 */
export function fallStep(fall, a, dtau) {
  let s = fall.s, left = dtau;
  while (left > 1e-12) {
    const r = radius(s[0], s[1], s[2], a), h = Math.min(left, 0.02 * Math.max(r, 0.05));
    s = rk4(s, fall.pt, a, h); left -= h;
  }
  return { s, pt: fall.pt };
}

function rk4(s, pt, a, h) {
  const k1 = deriv(s, pt, a);
  const k2 = deriv(s.map((x, i) => x + 0.5 * h * k1[i]), pt, a);
  const k3 = deriv(s.map((x, i) => x + 0.5 * h * k2[i]), pt, a);
  const k4 = deriv(s.map((x, i) => x + h * k3[i]), pt, a);
  return s.map((x, i) => x + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

/**
 * The step policy the shader uses: the spatial distance to cover in one RK4 step, from r. Tuned against the
 * precise integrator (tests): shadow edge within 0.1 px at 900 px tall for a ≤ 0.6, ~1 px at a = 0.999.
 * budget is the most steps a ray may take (the shader's loop bound); stepBudget(a) is what a given spin uses.
 */
export const STEP = { near: 0.05, far: 0.2, horizon: 1, budget: 1200 };
/**
 * Steps per ray for spin a: 400 up to a = 0.9, rising to 1200 at 0.999. Near-maximal spin needs more: rays near the
 * prograde edge circle the hole many times before escaping (with 400 the a = 0.999 prograde edge sat 9.7 px off,
 * 1.2 px with 1200). At lower spin the extra steps only went to rays a hair from the critical curve, and made the
 * refinement 2× slower at a = 0 for nothing. (Captured rays end early, captured(), so they never use the budget.)
 */
export function stepBudget(a) {
  const t = Math.min(Math.max((Math.abs(a) - 0.9) / 0.099, 0), 1);
  return Math.round(400 + (STEP.budget - 400) * t * t * (3 - 2 * t));
}
export function stepLength(r, rH, rM = 0) {
  const k = STEP.near + STEP.far * Math.min(Math.max((r - 8) / 60, 0), 1);
  // small steps near either horizon (inside the hole the camera sits between them); outside, r − rH as before
  return Math.min(k * r, STEP.horizon * Math.abs(r - rH) + 0.005, rM > 0 ? Math.abs(r - rM) + 0.005 : Infinity);
}

/**
 * Trace a ray backward in time from the camera until it falls into the hole, escapes, or runs out of steps.
 * mode 'shader' copies the GPU policy (fixed RK4 steps from stepLength); mode 'precise' uses many small steps.
 * Returns { fate: 'hole' | 'sky' | 'past' | 'steps', s, steps, maxDriftH }.
 *
 * Camera inside the hole (stage 3): between the horizons r itself is a time, and traced back in time every ray
 * climbs outward. Light that fell in from outside crosses the outer horizon (back out, in the trace) and goes on
 * to the disk or the sky. The rest creeps up to the outer horizon from inside without ever crossing it: in the
 * ideal eternal hole it would come from a white hole; for a real hole, from the collapsing star and whatever fell
 * in earlier. Neither is modelled, so it is 'past' (drawn black): a ray within 0.2% under the horizon whose outward
 * speed has almost stopped (dr per unit of path < 0.02) for 3 steps in a row. A crossing ray keeps a finite radial
 * speed there; a creeping one slows in proportion to its distance from the horizon. (Step-size independent, so the
 * precise integrator agrees.)
 */
export function trace(start, a, { mode = 'shader', maxSteps = stepBudget(a), rFar = 600, accuracy = 1 } = {}) {
  const rH = outerHorizon(a), rM = innerHorizon(a);
  const r0 = radius(start.s[0], start.s[1], start.s[2], a), inside = r0 < rH;
  // shader mode also ends captured rays early (exact test, once, when the ray first heads inward inside the photon
  // region); precise mode doesn't, so it stays an independent reference. Not for a camera inside the hole.
  const c = mode === 'shader' && !inside ? constants(start.s, start.pt, a) : null, rCheck = captureRadius(a);
  let s = start.s.slice(), steps = 0, maxDriftH = 0, rPrev = r0, checked = false, wasOut = !inside, band = 0, lenPrev = 1;
  for (; steps < maxSteps; steps++) {
    const r = radius(s[0], s[1], s[2], a);
    if (!(r > 0)) return { fate: 'hole', s, steps, maxDriftH };
    if (!wasOut) {
      if (r > rH * 1.02) wasOut = true;
      band = r < rH && r > rH * (1 - 2e-3) && (r - rPrev) / lenPrev < 0.02 ? band + 1 : 0;
      if (band >= 3) return { fate: 'past', s, steps, maxDriftH };
    }
    if (wasOut && !(r >= rH * 1.01)) return { fate: 'hole', s, steps, maxDriftH };
    if (c && !checked && r < rCheck && r < rPrev) { checked = true; if (captured(r, c, a)) return { fate: 'hole', s, steps, maxDriftH }; }
    rPrev = r;
    const v = deriv(s, start.pt, a);
    const out = -(s[0] * v[0] + s[1] * v[1] + s[2] * v[2]) > 0; // backward steps move along −v
    if (r > rFar && out) return { fate: 'sky', s, steps, maxDriftH };
    const len = stepLength(r, rH, rM) * (mode === 'precise' ? 0.05 / accuracy : 1);
    lenPrev = len;
    const h = len / Math.hypot(v[0], v[1], v[2]);
    s = rk4(s, start.pt, a, -h);
    maxDriftH = Math.max(maxDriftH, Math.abs(hamiltonian(s, start.pt, a)));
  }
  return { fate: 'steps', s, steps, maxDriftH };
}
