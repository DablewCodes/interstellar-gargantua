// Interstellar's wormhole (site/tools/gargantua/wormhole/worm.js): the shape against the paper's formula, and the
// light rays against exact properties (conservation, the exact edge of the disk, straight lines far away).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as W from '../wormhole/worm.js';

const p = W.params();

test('wormhole shape: the film\'s parameters, r(ℓ) continuous, flat inside, |ℓ|-like far away', () => {
  assert.ok(Math.abs(p.M * W.W_PER_M - 0.05) < 1e-12 && p.a === 0.005 && p.rho === 1);
  assert.equal(W.radius(0, p), 1);
  assert.equal(W.radius(0.005, p), 1);
  assert.ok(Math.abs(W.radius(0.005 + 1e-9, p) - 1) < 1e-9, 'continuous at the end of the interior');
  for (const l of [-50, -3, -0.2, 0.2, 3, 50]) assert.ok(W.radius(l, p) >= 1 && W.radius(l, p) === W.radius(-l, p));
  assert.ok(Math.abs(W.dradius(400, p) - 1) < 1e-3, 'flat space far away: dr/dℓ → 1');
  // the closed form (eq. 5b) is the integral of eq. 5a: dr/dℓ = (2/π) arctan(2(|ℓ|−a)/(πM)); check by finite differences
  for (const l of [-2, -0.04, 0.01, 0.06, 1, 9]) {
    const fd = (W.radius(l + 1e-6, p) - W.radius(l - 1e-6, p)) / 2e-6;
    assert.ok(Math.abs(fd - W.dradius(l, p)) < 1e-6, `dr/dℓ at ${l}`);
  }
});

test('light rays keep p_ℓ² + b²/r² = 1 (they stay light-like)', () => {
  for (const l0 of [-8, -2, 3]) for (const psi of [0.05, 0.4, 1.2, 2, 3]) {
    const r = W.trace(l0, psi, p);
    assert.ok(r.side !== 0 && r.maxH < 1e-5, `ℓ0 ${l0}, ψ ${psi}: drift ${r.maxH}`);
  }
});

test('the disk\'s edge is exact: rays with b < ρ go through to the other side, b > ρ turn back', () => {
  for (const l0 of [-8, -3, -1.2]) {
    const edge = W.diskAngle(l0, p);
    assert.equal(W.trace(l0, edge * 0.995, p).side, 1, `ℓ0 ${l0}: just inside the edge goes through`);
    assert.equal(W.trace(l0, edge * 1.005, p).side, -1, `ℓ0 ${l0}: just outside stays on Saturn's side`);
    assert.equal(W.trace(l0, 0, p).side, 1);
    assert.equal(W.trace(l0, Math.PI, p).side, -1);
  }
  // and from the far side, symmetrically
  assert.equal(W.trace(5, Math.PI - W.diskAngle(5, p) * 0.995, p).side, -1);
});

test('far from the wormhole light goes straight', () => {
  // looking straight away from it: you see the sky in your own direction (Φ = 0)
  const away = W.trace(-8, Math.PI, p);
  assert.equal(away.side, -1); assert.ok(Math.abs(away.Phi) < 1e-9);
  // from 2000 ρ, a ray sideways passes ~2000 ρ from the throat: it ends at 90° around, bent by less than 0.1°
  const side = W.trace(-2000, Math.PI / 2, p, { far: 8000 });
  assert.ok(Math.abs(side.Phi - Math.PI / 2) < 0.1 * Math.PI / 180, `bent by ${((side.Phi - Math.PI / 2) * 180 / Math.PI).toFixed(4)}°`);
});

test('the step rule agrees with 10× smaller steps (away from the critical rays that circle the throat)', () => {
  let worst = 0;
  for (const l0 of [-8, -3]) {
    const edge = W.diskAngle(l0, p);
    for (const f of [0.2, 0.6, 0.9, 1.1, 1.5, 3, 8]) {
      const psi = Math.min(Math.PI - 0.01, edge * f);
      const a = W.trace(l0, psi, p), b = W.trace(l0, psi, p, { k: 0.1, maxSteps: 60000 });
      assert.equal(a.side, b.side);
      worst = Math.max(worst, Math.abs(a.Phi - b.Phi));
    }
  }
  assert.ok(worst < 2e-4, `worst sky direction error ${worst} rad`);
});
