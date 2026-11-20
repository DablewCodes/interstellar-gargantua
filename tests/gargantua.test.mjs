// Gargantua: the light-ray physics (site/tools/gargantua/kerr.js) against exact results of general relativity, and
// the GPU step policy (the shader's integrator, mirrored by trace() mode 'shader') against the precise integrator.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as K from '../kerr.js';
import { geodesicSource } from '../shader.js';
import { blackbodyRGB, blackbodyLUT, LUT_SIZE } from '../colour.js';

const PIXEL = (45 / 900) * (Math.PI / 180); // one pixel at 900 px tall, 45° field of view

function camera(r, a, z = 0) {
  const pos = [Math.sqrt(r * r + a * a), 0, z];
  return { pos, T: K.tetrad(pos, K.staticObserver(pos, a), [-1, 0, 0], [0, 1, 0], [0, 0, 1], a) };
}
/** Angle (from "towards the hole") of the shadow edge in the equatorial plane, by bisection. side −1 = prograde. */
function shadowEdge({ pos, T }, a, side, mode, iterations = 30) {
  let lo = 0, hi = 0.6;
  for (let i = 0; i < iterations; i++) {
    const m = (lo + hi) / 2;
    const ray = K.rayFromCamera(pos, T, [side * Math.sin(m), 0, Math.cos(m)], a);
    if (K.trace(ray, a, { mode, maxSteps: mode === 'precise' ? 400000 : K.stepBudget(a) }).fate !== 'sky') lo = m; else hi = m;
  }
  return lo;
}
const impact = (cam, a, side, angle) => {
  const ray = K.rayFromCamera(cam.pos, cam.T, [side * Math.sin(angle), 0, Math.cos(angle)], a);
  return Math.abs(K.angularMomentum(ray.s) / ray.pt);
};
// Critical impact parameter of equatorial photon orbits (Bardeen 1973): b = 6 cos(acos(∓a)/3) ∓ a
const bExact = (a, prograde) => (prograde ? 6 * Math.cos(Math.acos(-a) / 3) - a : 6 * Math.cos(Math.acos(a) / 3) + a);

test('horizons, ISCO and photon orbits match the textbook values', () => {
  assert.equal(K.outerHorizon(0), 2);
  assert.equal(K.innerHorizon(0), 0);
  assert.ok(Math.abs(K.isco(0) - 6) < 1e-12);
  assert.ok(Math.abs(K.isco(0.999999) - 1) < 0.02, 'ISCO → M for a maximally spinning hole');
  assert.ok(Math.abs(K.photonOrbit(0) - 3) < 1e-12);
  assert.ok(Math.abs(K.photonOrbit(1) - 1) < 1e-9);
  assert.ok(Math.abs(K.outerHorizon(0.6) - 1.8) < 1e-12 && Math.abs(K.innerHorizon(0.6) - 0.2) < 1e-12);
});

test('metric gradients by automatic differentiation match finite differences', () => {
  for (const [x, y, z, a] of [[7, -3, 2, 0.6], [2.5, 1, -0.4, 0.999], [30, 4, 1, 0]]) {
    const dual = K.metricDual(x, y, z, a), e = 1e-6;
    for (let i = 1; i <= 3; i++) {
      const pt = [x, y, z], mt = [x, y, z];
      pt[i - 1] += e; mt[i - 1] -= e;
      const P = K.metricAt(...pt, a), M = K.metricAt(...mt, a);
      assert.ok(Math.abs((P.f - M.f) / (2 * e) - dual.f[i]) < 1e-6, `∂f/∂x${i}`);
      assert.ok(Math.abs((P.l[1] - M.l[1]) / (2 * e) - dual.lx[i]) < 1e-6, `∂lx/∂x${i}`);
      assert.ok(Math.abs((P.l[3] - M.l[3]) / (2 * e) - dual.lz[i]) < 1e-6, `∂lz/∂x${i}`);
    }
  }
});

test('camera frame is orthonormal and the ray starts as light (H = 0)', () => {
  const a = 0.6, cam = camera(12, a, 1.5), m = K.metricAt(...cam.pos, a);
  const E = [cam.T.e0, cam.T.right, cam.T.up, cam.T.fwd];
  E.forEach((u, i) => E.forEach((v, j) => {
    const want = i === j ? (i === 0 ? -1 : 1) : 0;
    assert.ok(Math.abs(K.dot4(m, u, v) - want) < 1e-12, `g(e${i}, e${j})`);
  }));
  const ray = K.rayFromCamera(cam.pos, cam.T, [0.3, -0.2, Math.sqrt(1 - 0.13)], a);
  assert.ok(Math.abs(K.hamiltonian(ray.s, ray.pt, a)) < 1e-12);
});

test('ZAMO: unit timelike, zero angular momentum, the static observer at a = 0, the exact Kerr lapse', () => {
  for (const [pos, a] of [[[12, 3, 2], 0.6], [[1.9, 0.4, 0.2], 0.999], [[40, -5, 9], 0], [[1.5, 0, 0.1], 0.999]]) {
    const m = K.metricAt(...pos, a), Z = K.zamo(pos, a);
    assert.ok(Math.abs(K.dot4(m, Z.u, Z.u) + 1) < 1e-10, 'u·u = −1');
    const ul = K.lower(m, Z.u);
    assert.ok(Math.abs(pos[0] * ul[2] - pos[1] * ul[1]) < 1e-10, 'u_φ = 0 (zero angular momentum)');
    // exact Boyer–Lindquist lapse α² = ΔΣ / ((r² + a²)² − a²Δ sin²θ)
    const r = m.r, cos = pos[2] / r, sin2 = 1 - cos * cos, D = r * r - 2 * r + a * a, S = r * r + a * a * cos * cos;
    const alpha = Math.sqrt(D * S / ((r * r + a * a) ** 2 - a * a * D * sin2));
    assert.ok(Math.abs(Z.alpha - alpha) < 1e-10, `α ${Z.alpha} vs ${alpha}`);
    if (a === 0) { const st = K.staticObserver(pos, a); st.forEach((c, i) => assert.ok(Math.abs(c - Z.u[i]) < 1e-10, 'static')); }
  }
});

test('ZAMO axes: orthonormal, and the plain coordinate axes far away', () => {
  for (const [pos, a] of [[[12, 3, 2], 0.6], [[2.2, 0.4, 0.3], 0.999], [[3, -4, 1], 0]]) {
    const m = K.metricAt(...pos, a), Z = K.zamo(pos, a), E = [Z.u, ...K.zamoAxes(pos, Z.u, a)];
    E.forEach((p, i) => E.forEach((q, j) => {
      const want = i === j ? (i === 0 ? -1 : 1) : 0;
      assert.ok(Math.abs(K.dot4(m, p, q) - want) < 1e-10, `g(e${i}, e${j}) at a = ${a}`);
    }));
  }
  // far away the axes are the plain coordinate axes (where Kerr–Schild and Boyer–Lindquist agree)
  for (const a of [0, 0.6, 0.999]) {
    const pos = [3000, 1200, 800], E = K.zamoAxes(pos, K.zamo(pos, a).u, a);
    E.forEach((e, k) => [1, 2, 3].forEach((i) => assert.ok(Math.abs(e[i] - (i === k + 1 ? 1 : 0)) < 2e-3, `far axis ${k} at a = ${a}`)));
  }
});

test('moving camera: boosted frame is orthonormal, speed v relative to the ZAMO, clock rate α/γ', () => {
  const a = 0.6, pos = [9, 2, 1.5], m = K.metricAt(...pos, a), Z = K.zamo(pos, a);
  const v = [0.3, -0.5, 0.4], sp = Math.hypot(...v);
  // an orthonormal look direction (right, up, fwd) in the ZAMO's local axes
  const fwd = [0.6, 0, 0.8], right = [0, 1, 0], up = [-0.8, 0, 0.6];
  const C = K.cameraTetrad(pos, v, right, up, fwd, a), E = [C.e0, C.right, C.up, C.fwd];
  E.forEach((p, i) => E.forEach((q, j) => {
    const want = i === j ? (i === 0 ? -1 : 1) : 0;
    assert.ok(Math.abs(K.dot4(m, p, q) - want) < 1e-10, `g(e${i}, e${j})`);
  }));
  const gamma = -K.dot4(m, C.e0, Z.u);                         // γ = −u_cam · u_ZAMO
  assert.ok(Math.abs(gamma - 1 / Math.sqrt(1 - sp * sp)) < 1e-10, 'γ from the relative speed');
  assert.ok(Math.abs(C.alpha / C.gamma - Z.alpha * Math.sqrt(1 - sp * sp)) < 1e-12, 'clock rate');
  const far = K.cameraTetrad([5000, 0, 0], [0, 0, 0], right, up, fwd, 0);
  assert.ok(Math.abs(far.alpha - 1) < 1e-3, 'far away the clock runs at Earth rate');
});

test('aberration: flying at the hole shrinks the shadow exactly as special relativity says', () => {
  // camera on the x axis, looking at the hole (−x); local axes: right = +y, up = +z, fwd = −x
  const a = 0, pos = [30, 0, 0], right = [0, 1, 0], up = [0, 0, 1], fwd = [-1, 0, 0];
  const edge = (v) => {
    const T = K.cameraTetrad(pos, [-v, 0, 0], right, up, fwd, a); // velocity towards the hole
    let lo = 0, hi = 0.8;
    for (let i = 0; i < 40; i++) {
      const m = (lo + hi) / 2;
      const fate = K.trace(K.rayFromCamera(pos, T, [Math.sin(m), 0, Math.cos(m)], a), a, { mode: 'precise', maxSteps: 400000 }).fate;
      if (fate !== 'sky') lo = m; else hi = m;
    }
    return lo;
  };
  const th0 = edge(0), v = 0.5, th = edge(v);
  // light arriving at angle θ from the direction of motion appears at θ' with cos θ' = (cos θ + v) / (1 + v cos θ)
  const want = Math.acos((Math.cos(th0) + v) / (1 + v * Math.cos(th0)));
  assert.ok(Math.abs(th - want) < 1e-6, `shadow radius at ${v} c: ${th} rad, special relativity: ${want}`);
  assert.ok(th < th0 * 0.6, 'and it is clearly smaller');
});

// --- stage 3: the fall ---
function dropFrom(r0, a, theta = Math.PI / 2) {               // at rest relative to the ZAMO, then let go
  const pos = [Math.sqrt(r0 * r0 + a * a) * Math.sin(theta), 0, r0 * Math.cos(theta)];
  const pl = K.lower(K.metricAt(...pos, a), K.zamo(pos, a).u);
  return { s: [...pos, pl[1], pl[2], pl[3]], pt: pl[0] };
}
const rOf = (f, a) => K.radius(f.s[0], f.s[1], f.s[2], a);

test('the fall: dropped from rest at 6 M, the proper time to the horizon and on is the exact cycloid (a = 0)', () => {
  // r = R (1 + cos η) / 2, τ = √(R³ / 2M) (η + sin η) / 2
  const R = 6, exact = (r) => { const eta = Math.acos(2 * r / R - 1); return Math.sqrt(R ** 3 / 2) * (eta + Math.sin(eta)) / 2; };
  let f = dropFrom(R, 0), tau = 0;
  for (const target of [2, 0.5]) {
    while (rOf(f, 0) > target) { f = K.fallStep(f, 0, 0.002); tau += 0.002; }
    // interpolate the crossing within the last step
    assert.ok(Math.abs(tau - exact(rOf(f, 0))) < 2e-3, `r = ${rOf(f, 0).toFixed(4)}: τ ${tau.toFixed(4)} vs exact ${exact(rOf(f, 0)).toFixed(4)}`);
    assert.ok(Math.abs(K.hamiltonian(f.s, f.pt, 0) + 0.5) < 1e-8, 'stays a unit timelike path (H = −½) through the horizon');
  }
});

test('the fall (a = 0.6): through both horizons smoothly, still a unit timelike path', () => {
  const a = 0.6; let f = dropFrom(2.9, a, 84 * Math.PI / 180), n = 0;
  while (rOf(f, a) > K.innerHorizon(a) * 1.05) { f = K.fallStep(f, a, 0.004); n++; assert.ok(n < 5000, 'reaches the inner horizon'); }
  assert.ok(Math.abs(K.hamiltonian(f.s, f.pt, a) + 0.5) < 1e-4, `H = ${K.hamiltonian(f.s, f.pt, a)}`);
  assert.ok(f.s.every(Number.isFinite));
});

function insideCamera(a, rTarget) {
  let f = dropFrom(2.9, a, 84 * Math.PI / 180);
  while (rOf(f, a) > rTarget) f = K.fallStep(f, a, 0.004);
  const pos = f.s.slice(0, 3), m = K.metricAt(...pos, a), u = K.raise(m, [f.pt, f.s[3], f.s[4], f.s[5]]);
  const n = Math.hypot(...pos), back = pos.map((x) => x / n), side = [-pos[1], pos[0], 0].map((x) => x / Math.hypot(pos[0], pos[1]));
  const up = [back[1] * side[2] - back[2] * side[1], back[2] * side[0] - back[0] * side[2], back[0] * side[1] - back[1] * side[0]];
  return { pos, T: K.tetrad(pos, u, back, side, up, a) };   // looking straight back, away from the hole
}

test('inside the hole: the GPU step policy agrees with the precise integrator', () => {
  const a = 0.6, cam = insideCamera(a, 1.0);
  let agree = 0, total = 0, worst = 0;
  for (let k = 0; k < 40; k++) {
    const A = (k / 40) * 2.6, ph = k * 2.39996;                  // spread over most of the sky around "back"
    const ray = K.rayFromCamera(cam.pos, cam.T, [Math.sin(A) * Math.cos(ph), Math.sin(A) * Math.sin(ph), Math.cos(A)], a);
    const g = K.trace(ray, a), p = K.trace(ray, a, { mode: 'precise', maxSteps: 400000 });
    total++;
    if (g.fate === p.fate) agree++;
    if (g.fate === 'sky' && p.fate === 'sky') {
      const u = K.deriv(g.s, ray.pt, a), v = K.deriv(p.s, ray.pt, a);
      const c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / Math.hypot(u[0], u[1], u[2]) / Math.hypot(v[0], v[1], v[2]);
      worst = Math.max(worst, Math.acos(Math.min(1, c)) / PIXEL);
    }
  }
  assert.ok(agree >= total - 2, `fates agree on ${agree} of ${total}`);
  assert.ok(worst < 1, `escape direction within 1 px (worst ${worst.toFixed(2)})`);
});

test('inside the hole: the sky straight back is redshifted, the rim of the view blueshifted', () => {
  const a = 0.6, cam = insideCamera(a, 1.2), g = (deg) => {
    const A = deg * Math.PI / 180, ray = K.rayFromCamera(cam.pos, cam.T, [Math.sin(A), 0, Math.cos(A)], a);
    assert.equal(K.trace(ray, a).fate, 'sky', `${deg}° reaches the sky`);
    return 1 / -ray.pt;                                         // observed / emitted (the camera measures energy 1)
  };
  assert.ok(g(0) < 0.8, `straight back g = ${g(0)}`);
  assert.ok(g(90) > 1.3, `90° off g = ${g(90)}`);
});

test('non-spinning shadow: critical impact parameter is √27 M', () => {
  const cam = camera(1000, 0);
  for (const side of [1, -1]) {
    const b = impact(cam, 0, side, shadowEdge(cam, 0, side, 'precise'));
    assert.ok(Math.abs(b - Math.sqrt(27)) < 1e-4, `b = ${b}`);
  }
});

test('spinning shadow (a = 0.6): both equatorial edges match the exact Kerr values', () => {
  const a = 0.6, cam = camera(30, a);
  for (const side of [1, -1]) {
    const b = impact(cam, a, side, shadowEdge(cam, a, side, 'precise'));
    const want = bExact(a, side < 0);
    assert.ok(Math.abs(b - want) / want < 2e-4, `${side < 0 ? 'prograde' : 'retrograde'} b = ${b}, exact ${want}`);
  }
});

test('energy, angular momentum and the light condition stay conserved along rays', () => {
  const a = 0.6, cam = camera(30, a, 3);
  for (const dir of [[0.12, 0.05], [-0.2, 0.1], [0.05, -0.3], [0.3, 0.3]]) {
    const ray = K.rayFromCamera(cam.pos, cam.T, [dir[0], dir[1], Math.sqrt(1 - dir[0] ** 2 - dir[1] ** 2)], a);
    const L0 = K.angularMomentum(ray.s), res = K.trace(ray, a, { mode: 'precise', maxSteps: 200000 });
    assert.ok(res.maxDriftH < 1e-6, `H drift ${res.maxDriftH}`);
    assert.ok(Math.abs(K.angularMomentum(res.s) - L0) < 1e-6 * Math.max(1, Math.abs(L0)), 'L_z drift');
  }
});

test('Carter constant Q is conserved along rays (and computed the same in Kerr–Schild coordinates)', () => {
  const a = 0.999, cam = camera(10, a, 2.5);
  for (const dir of [[0.25, 0.1], [-0.3, 0.2], [0.1, -0.35]]) {
    const ray = K.rayFromCamera(cam.pos, cam.T, [dir[0], dir[1], Math.sqrt(1 - dir[0] ** 2 - dir[1] ** 2)], a);
    const c0 = K.constants(ray.s, ray.pt, a), res = K.trace(ray, a, { mode: 'precise', maxSteps: 200000 });
    const c1 = K.constants(res.s, ray.pt, a);
    assert.ok(Math.abs(c1.Q - c0.Q) < 1e-5 * (Math.abs(c0.Q) + c0.L ** 2 + 1), `Q ${c0.Q} → ${c1.Q}`);
  }
});

test('capture test agrees with the precise integrator: rays it calls captured fall in, the others escape', () => {
  for (const a of [0, 0.6, 0.999]) {
    const cam = camera(8, a, 1), rH = K.outerHorizon(a), rCheck = K.captureRadius(a);
    let checked = 0;
    for (let k = 0; k < 40; k++) {
      const dx = -0.5 + k / 39, dy = 0.3 * Math.sin(k * 1.7);
      const ray = K.rayFromCamera(cam.pos, cam.T, [dx, dy, Math.sqrt(1 - dx * dx - dy * dy)], a);
      const c = K.constants(ray.s, ray.pt, a);
      // follow the precise ray (one step per call) to where the shader would first check, then let it finish
      let s = ray.s.slice(), rPrev = Infinity, verdict = null;
      for (let n = 0; n < 100000 && verdict === null; n++) {
        const r = K.radius(s[0], s[1], s[2], a);
        if (r < rH * 1.01) break;
        if (r < rCheck && r < rPrev) verdict = K.captured(r, c, a);
        rPrev = r;
        s = K.trace({ s, pt: ray.pt }, a, { mode: 'precise', maxSteps: 1 }).s;
      }
      if (verdict === null) continue;
      const fate = K.trace({ s, pt: ray.pt }, a, { mode: 'precise', maxSteps: 400000 }).fate;
      if (fate === 'steps') continue;
      checked++;
      assert.equal(verdict, fate === 'hole', `a = ${a}, ray ${k}: captured() says ${verdict}, precise says ${fate}`);
    }
    assert.ok(checked > 10, `a = ${a}: ${checked} rays checked`);
  }
});

test('fast spin (a = 0.999): GPU step policy puts both shadow edges within 1.5 px', () => {
  const a = 0.999, cam = camera(30, a);
  for (const side of [1, -1]) {
    const off = Math.abs(shadowEdge(cam, a, side, 'shader', 22) - shadowEdge(cam, a, side, 'precise', 22)) / PIXEL;
    assert.ok(off < 1.5, `${side < 0 ? 'prograde' : 'retrograde'}: ${off.toFixed(2)} px`);
  }
});

test('GPU step policy puts the shadow edge within 0.1 px of the precise answer (a ≤ 0.6)', () => {
  for (const a of [0, 0.6]) {
    const cam = camera(30, a);
    for (const side of [1, -1]) {
      const off = Math.abs(shadowEdge(cam, a, side, 'shader', 22) - shadowEdge(cam, a, side, 'precise', 22)) / PIXEL;
      assert.ok(off < 0.1, `a = ${a} ${side < 0 ? 'prograde' : 'retrograde'}: ${off.toFixed(3)} px`);
    }
  }
});

test('shader source has every constant filled in', () => {
  const src = geodesicSource(K.STEP);
  assert.ok(!src.includes('__'), 'unreplaced placeholder');
  assert.match(src, /const int BUDGET = \d+;/);
});

test('blackbody colours: 6500 K is white, cooler is redder, hotter is bluer', () => {
  const w = blackbodyRGB(6500);
  assert.ok(Math.max(...w) / Math.min(...w) < 1.06, `6500 K = ${w}`);
  const [r1, g1, b1] = blackbodyRGB(2000), [r2, , b2] = blackbodyRGB(20000);
  assert.ok(r1 > g1 && g1 > b1, '2000 K is orange-red');
  assert.ok(b2 > r2, '20000 K is blue-white');
  const lut = blackbodyLUT();
  assert.equal(lut.length, LUT_SIZE * 4);
  assert.ok(lut.every(Number.isFinite));
});

test('Reality mode: the disk gas is a unit timelike orbit and its light shift is right (gravity + Doppler)', () => {
  for (const a of [0, 0.6, 0.999]) {
    for (const r of [K.isco(a) * 1.01, 6, 20]) {
      // the gas 4-velocity u^t (∂t + Ω ∂φ), in Kerr–Schild Cartesian components (∂φ = −y ∂x + x ∂y), at φ = 0
      const pos = [r, a, 0], ut = K.orbitUt(r, a), W = K.orbitalOmega(r, a);
      const u = [ut, -W * pos[1] * ut, W * pos[0] * ut, 0];
      assert.ok(Math.abs(K.dot4(K.metricAt(...pos, a), u, u) + 1) < 1e-9, `u·u = −1 at a=${a}, r=${r}`);
    }
  }
  // a = 0, seen face-on from far away (L = 0, E = 1): only time dilation and gravity, g = √(1 − 3/r)
  for (const r of [6, 10, 40]) assert.ok(Math.abs(K.diskShift(r, 0, 1, 0) - Math.sqrt(1 - 3 / r)) < 1e-12);
  // seen edge-on from a far camera (looking along −x, the gas turning counter-clockwise about +z): the side coming
  // towards the camera (y < 0, to its left) is blueshifted, the far side's redshift is stronger than time dilation
  const a = 0.6, cam = camera(1000, a);
  const g = (side) => {
    const ray = K.rayFromCamera(cam.pos, cam.T, [side * Math.sin(0.01), 0, Math.cos(0.01)], a);
    return K.diskShift(10, a, -ray.pt, K.angularMomentum(ray.s));
  };
  assert.ok(g(-1) > 1.15 && g(1) < 0.75, `approaching ${g(-1)}, receding ${g(1)}`);
});
