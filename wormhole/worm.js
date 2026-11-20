// Interstellar's wormhole: the physics, 64-bit. Shared by the page (it mirrors this in worm-shader.js) and the tests.
//
// O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, "Visualizing Interstellar's Wormhole", American Journal
// of Physics 83, 486 (2015), arXiv:1502.03809. The "Dneg" wormhole, gravity off (their eq. 1):
//   ds² = −dt² + dℓ² + r(ℓ)² (dθ² + sin²θ dφ²)
// ℓ runs from −∞ (Saturn's side) through the throat to +∞ (the far side: Gargantua's galaxy). The radius of the
// spheres of constant ℓ (eq. 5):
//   r = ρ                                     for |ℓ| ≤ a     (a cylindrical interior of length 2a)
//   r = ρ + M [x arctan x − ½ ln(1 + x²)]     for |ℓ| > a,    x = 2(|ℓ| − a) / (π M)
// with the lensing width W = 1.42953 M (eq. 7). The film: throat radius ρ = 1 km, 2a = 0.01 ρ, W = 0.05 ρ.
// Lengths here are in units of ρ.

export const FILM = { rho: 1, a: 0.005, W: 0.05 };
export const W_PER_M = 1.42953;
export const params = ({ rho, a, W } = FILM) => ({ rho, a, M: W / W_PER_M });

/** r(ℓ), the radius of the sphere at ℓ. */
export function radius(l, p) {
  const d = Math.abs(l) - p.a;
  if (d <= 0) return p.rho;
  const x = (2 * d) / (Math.PI * p.M);
  return p.rho + p.M * (x * Math.atan(x) - 0.5 * Math.log(1 + x * x));
}
/** dr/dℓ. */
export function dradius(l, p) {
  const d = Math.abs(l) - p.a;
  return d <= 0 ? 0 : Math.sign(l) * (2 / Math.PI) * Math.atan((2 * d) / (Math.PI * p.M));
}

/**
 * Step length along the ray (affine parameter = time, p_t = −1): short near the throat, where r(ℓ) bends over a
 * length ~M, growing with distance. The shader uses the same rule.
 */
export const STEP = { near: 0.004, grow: 0.04 };
export const stepLength = (l, p, k = 1) => k * (STEP.near + STEP.grow * Math.max(0, Math.abs(l) - p.a));

/**
 * One light ray, traced from a camera at ℓ0 leaving at angle ψ from the direction of increasing ℓ (backward in time,
 * so it ends where the light came from). By spherical symmetry it stays in one plane: coordinates ℓ and the angle φ
 * around the wormhole in that plane, momentum p_ℓ = cos ψ, and the conserved b = p_φ = r(ℓ0) sin ψ (eqs. 16, 18 of the
 * paper, in the ray's own plane). Equations: dℓ/dt = p_ℓ, dφ/dt = b / r², dp_ℓ/dt = b² r′ / r³.
 * Ends once far out (|ℓ| > far) and moving away, with the direction it is then heading: Φ = φ + atan(b / (r |p_ℓ|)).
 * Returns { side: −1 Saturn's / +1 the far side / 0 out of steps, Phi, steps, maxH (largest |p_ℓ² + b²/r² − 1|) }.
 */
export function trace(l0, psi, p, { far = 0, maxSteps = 6000, k = 1 } = {}) {
  const fr = Math.max(far, 4 * Math.abs(l0), 50);
  const b = radius(l0, p) * Math.sin(psi);
  let l = l0, ph = 0, pl = Math.cos(psi), maxH = 0;
  const f = (l, pl) => { const r = radius(l, p); return [pl, b / (r * r), (b * b * dradius(l, p)) / (r * r * r)]; };
  for (let i = 0; i < maxSteps; i++) {
    const r = radius(l, p);
    maxH = Math.max(maxH, Math.abs(pl * pl + (b * b) / (r * r) - 1));
    if (Math.abs(l) > fr && pl * Math.sign(l) > 0) return { side: Math.sign(l), Phi: ph + Math.atan2(b / r, Math.abs(pl)), steps: i, maxH };
    const h = stepLength(l, p, k);
    const k1 = f(l, pl), k2 = f(l + 0.5 * h * k1[0], pl + 0.5 * h * k1[2]);
    const k3 = f(l + 0.5 * h * k2[0], pl + 0.5 * h * k2[2]), k4 = f(l + h * k3[0], pl + h * k3[2]);
    l += (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    ph += (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    pl += (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]);
  }
  return { side: 0, Phi: ph, steps: maxSteps, maxH };
}

/** The angular radius of the wormhole's disk seen from ℓ0: rays with b < ρ go through (b = r sin ψ). */
export const diskAngle = (l0, p) => Math.asin(Math.min(1, p.rho / radius(l0, p)));
