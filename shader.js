// GPU renderer (WebGL2). The picture is built in passes:
//   GEODESIC  traces one light ray per pixel through the Kerr spacetime (same equations as kerr.js, whose tests
//             check this step policy) and stores where it went: up to two disk crossings (r, φ) and the final sky
//             direction. Expensive, so it only runs when the camera moves, and refines to full resolution while
//             the camera is still (the "geodesic cache").
//   SHADE     every frame: turns the cached hits into light: animated disk (noise texture, blackbody colours),
//             lensed point stars and the real Milky Way. Writes HDR.
//   BLOOM     down/up-sampling chain over the HDR image: the glow the film's IMAX lenses gave every bright edge.
//   COMPOSITE exposure, ACES tone mapping, film grade, vignette, grain.
// NOISE runs once to make the tileable noise texture the disk is painted with.

import { CAPTURE_SAMPLES } from './kerr.js';

export const VERTEX = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

export const HASH = `
uint hashU(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v.x ^ v.y ^ v.z;
}
vec3 hash3(vec3 c) {
  uvec3 u = uvec3(ivec3(floor(c)) + 32768);
  return vec3(hashU(u), hashU(u + 7919u), hashU(u + 104729u)) / 4294967295.0;
}`;

// ------------------------------------------------------------------------------------------------ GEODESIC
// Compiled three times:
//   RING_PASS 0 writes the cache (centre rays only; photon-ring pixels are flagged, ring.w = 1).
//   RING_PASS 1 is the edge refinement: centre rays at full screen resolution, but only on the pixels REFINE_MASK
//     marked (stencil) as places where interpolating the (coarser) cache can't work: where neighbouring cache rays
//     disagree about what they hit (disk / hole / sky edges, thin disk images) or where light looped the hole.
//     Output = the same three targets as the cache, with ring.w = 0 not refined, 1 refined, 2 photon-ring pixel.
//   RING_PASS 2 is the ray bundle: only on the ring pixels pass 1 found (RING_MASK, stencil 2), uSubN more rays
//     spread over the pixel; writes the ring target only (its average light, mean radius, sky seen through).
// Passes 1 and 2 are split so the GPU's lockstep groups hold pixels of one kind: in one pass, the few 17-ray ring
// pixels held up the 1-ray edge pixels sharing their group (bundles cost ~3× the whole refinement, measured).
const GEODESIC = `#version 300 es
#define RING_PASS __RING_PASS__
precision highp float;
precision highp int;
layout(location = 0) out vec4 outHits;   // (r1, φ1, r2, φ2) of the first two disk crossings, r = 0 if none
layout(location = 1) out vec4 outSky;    // (sky direction, 1 if the ray escaped / 0 if it fell in)
layout(location = 2) out vec4 outRing;   // photon-ring pixels: (mean light, light·r sum, sky-through, flag), else 0

uniform vec2 uRes;          // size of the buffer being written
uniform int uBudget;        // steps per ray for this spin (kerr.js stepBudget), at most BUDGET
uniform int uSubN;          // rays per photon-ring pixel in the bundle pass: 16 sharp, 4 while moving
uniform int uBundle;        // 0 (lite quality): no ring bundle; ring pixels are refined like edges, one ray each
uniform float uRingSweep;   // a ray that swept more than this angle (rad) around the hole marks a photon-ring pixel
// Crossings after the light has swept more than this around the hole are higher-order disk images ("looped"): on
// ring pixels their average replaces them. (A camera's own direct view sweeps under half a turn.)
const float LOOP_SWEEP = 1.35 * 3.14159265;
uniform vec4 uRect;         // this buffer covers the camera view rectangle (x0, y0, x1, y1) in [0, 1]
uniform float uSpin, uRH, uRM, uRin, uRout, uRFar; // uRH / uRM: outer / inner horizon
uniform float uRCheck;      // below this radius a ray heading inward is tested for capture (kerr.js captured())
uniform vec3 uCam;
uniform vec4 uE0, uER, uEU, uEF;  // camera tetrad, lowered (t, x, y, z): time, right, up, forward
uniform vec2 uTan;

const int BUDGET = __BUDGET__;
const int CAPTURE_SAMPLES = __CAPTURE_SAMPLES__;
const float STEP_NEAR = __STEP_NEAR__, STEP_FAR = __STEP_FAR__, STEP_HORIZON = __STEP_HORIZON__;

vec4 dmul(vec4 a, vec4 b) { return vec4(a.x * b.x, a.x * b.yzw + b.x * a.yzw); }
vec4 ddiv(vec4 a, vec4 b) { float q = a.x / b.x; return vec4(q, (a.yzw - q * b.yzw) / b.x); }
vec4 dsqrt(vec4 a) { float s = sqrt(a.x); return vec4(s, a.yzw * (0.5 / s)); }

float radiusKS(vec3 q) {
  float a2 = uSpin * uSpin, w = dot(q, q) - a2;
  return sqrt(0.5 * w + sqrt(0.25 * w * w + a2 * q.z * q.z));
}

// Hamilton's equations for light, H = ½ g^μν p_μ p_ν, metric gradients by dual numbers (automatic differentiation)
void geodesic(vec3 q, vec3 p, float pt, out vec3 dq, out vec3 dp) {
  float a = uSpin, a2 = a * a;
  vec4 X = vec4(q.x, 1.0, 0.0, 0.0), Y = vec4(q.y, 0.0, 1.0, 0.0), Z = vec4(q.z, 0.0, 0.0, 1.0);
  vec4 w = dmul(X, X) + dmul(Y, Y) + dmul(Z, Z) - vec4(a2, 0.0, 0.0, 0.0);
  vec4 r2 = 0.5 * w + dsqrt(0.25 * dmul(w, w) + a2 * dmul(Z, Z));
  vec4 r = dsqrt(r2);
  vec4 f = ddiv(2.0 * dmul(r2, r), dmul(r2, r2) + a2 * dmul(Z, Z));
  vec4 den = r2 + vec4(a2, 0.0, 0.0, 0.0);
  vec4 lx = ddiv(dmul(r, X) + a * Y, den);
  vec4 ly = ddiv(dmul(r, Y) - a * X, den);
  vec4 lz = ddiv(Z, r);
  float L = -pt + lx.x * p.x + ly.x * p.y + lz.x * p.z;
  float fL = f.x * L;
  dq = p - fL * vec3(lx.x, ly.x, lz.x);
  dp = 0.5 * f.yzw * L * L + fL * (p.x * lx.yzw + p.y * ly.yzw + p.z * lz.yzw);
}

float stepLength(float r) {                                    // kerr.js stepLength
  float k = STEP_NEAR + STEP_FAR * clamp((r - 8.0) / 60.0, 0.0, 1.0);
  float l = min(k * r, STEP_HORIZON * abs(r - uRH) + 0.005);
  return uRM > 0.0 ? min(l, abs(r - uRM) + 0.005) : l;
}

// Radial potential R(r) of a photon with constants (E, L, Q), M = 1 (Carter 1968; kerr.js radialPotential)
float radialR(float r, vec3 c) {
  float a = uSpin, A = c.x * (r * r + a * a) - a * c.y, D = r * r - 2.0 * r + a * a, B = c.y - a * c.x;
  return A * A - D * (B * B + c.z);
}
// Captured: moving inward at r, and R > 0 on (r+, r), so nothing can turn it back (kerr.js captured(), same samples)
bool captured(float r, vec3 c) {
  float best = 0.0, bestR = 1e30, n = float(CAPTURE_SAMPLES);
  for (int i = 1; i < CAPTURE_SAMPLES; i++) {
    float v = radialR(uRH + (r - uRH) * float(i) / n, c);
    if (v < bestR) { bestR = v; best = float(i); }
  }
  if (bestR <= 0.0) return false;
  float lo = uRH + (r - uRH) * (best - 1.0) / n, hi = uRH + (r - uRH) * (best + 1.0) / n;
  for (int k = 0; k < 12; k++) {
    float m1 = hi - 0.618 * (hi - lo), m2 = lo + 0.618 * (hi - lo);
    if (radialR(m1, c) < radialR(m2, c)) hi = m2; else lo = m1;
  }
  return radialR(0.5 * (lo + hi), c) > 0.0;
}

// The disk's average look for the ring (no texture: the photon-ring images are far too thin to show any).
// Same profile as SHADE's disk(): returns (opacity, light without colour/brightness).
vec2 diskAverage(float r) {
  float x = r / 6.0;                                            // DISK_REF (see SHADE)
  float dens = 0.69 * smoothstep(0.92, 1.12, r / uRin) * (1.0 - smoothstep(uRout * 0.45, uRout, r));
  float alpha = clamp(dens * 2.2, 0.0, 0.97);
  return vec2(alpha, alpha * dens * pow(max(x, 0.85), -2.3));
}

// Sample 0 is the pixel's centre ray, stored as cache hits for the textured shading. If that ray wound around the hole
// (swept angle, see uRingSweep) like light near the
// photon sphere, the pixel is on the photon ring, whose higher-order images of the disk are thinner than a pixel:
// one ray hits or misses them at random (dotted, flickering ring). Those pixels trace uSubN more rays spread over
// the pixel (a real-time version of DNGR's ray bundles) and store their average: ring light, its mean radius
// (for the colour), and how much sky shows through. One integrator call site, inside the sample loop.
void main() {
  // In RING_PASS 1 every pixel whose centre ray has a looped crossing becomes a ring pixel too, so ALL light that
  // went around the hole comes from screen-resolution rays, none from the coarse cache.
  bool loopedHit = false;
  float a2 = uSpin * uSpin;
  vec4 hits = vec4(0.0);
  outSky = vec4(0.0, 0.0, 1.0, 0.0);
  vec4 ring = vec4(0.0);           // (light, light-weighted radius sum, sky-through sum, 1 if ring pixel)
  int samples = RING_PASS == 2 ? 1 + uSubN : 1;
  for (int s = RING_PASS == 2 ? 1 : 0; s < 17; s++) {
    if (s >= samples) break;
    vec2 jit = s == 0 ? vec2(0.0) : (vec2(float((s - 1) % 4), float((s - 1) / 4)) + 0.5) / 4.0 - 0.5;
    if (s > 0 && uSubN == 4) jit = (vec2(float((s - 1) % 2), float((s - 1) / 2)) + 0.5) / 2.0 - 0.5;
    if (uSubN == 1) jit = vec2(0.0);                             // (moving, when the GPU is short of time)
    vec2 uv = mix(uRect.xy, uRect.zw, (gl_FragCoord.xy + jit) / uRes);
    vec3 dir = normalize(vec3((uv * 2.0 - 1.0) * uTan, 1.0));  // camera frame: right, up, forward
    vec4 pc = uE0 - dir.x * uER - dir.y * uEU - dir.z * uEF;   // the photon travels along −dir
    float pt = pc.x;
    vec3 q = uCam, p = pc.yzw;
    int nHits = 0;
    float sweep = 0.0, light = 0.0, lightR = 0.0, trans = 1.0, escaped = 0.0; // sweep: angle travelled around the hole
    // constants of motion (E, L, Q) for the capture test (kerr.js constants())
    float r0 = radiusKS(q), cth = q.z / r0, sin2 = max(1.0 - cth * cth, 1e-12);
    float Lz = q.x * p.y - q.y * p.x, pTh = cth / sqrt(sin2) * (q.x * p.x + q.y * p.y) - r0 * sqrt(sin2) * p.z;
    vec3 cst = vec3(-pt, Lz, pTh * pTh + cth * cth * (Lz * Lz / sin2 - a2 * pt * pt));
    // The capture verdict, decided once per ray before the loop (a call inside the step loop made the compiled loop
    // heavy: refinement 2× slower at a = 0). Equivalent to testing when the ray first heads inward inside uRCheck:
    // R > 0 on (r+, rC) with rC = min(camera radius, uRCheck), and a ray moving inward below rC passed only through
    // allowed radii to get there, so if R > 0 below it nothing can turn it back.
    float rC = min(r0, uRCheck);
    bool capt = r0 > uRH * 1.01 ? captured(rC, cst) : false, doomed = false; // (no capture test from inside the hole)
    float rPrev = 1e30, doomSweep = 0.0;                        // angle travelled since the ray was found captured
    // Camera inside the hole (kerr.js trace(), same rule): traced back in time every ray climbs outward. Light that
    // fell in from outside crosses the outer horizon and goes on to the disk or sky; the rest creeps up to the horizon
    // from inside without crossing (light from the hole's own past, not modelled: black).
    bool wasOut = r0 > uRH;
    int band = 0;
    float rLast = r0, lenPrev = 1.0;

    // bundle rays (s > 0) get at most 400 steps: 16 of them × the full budget per pixel is the heaviest work there
    // is; one that runs out only counts as dark, in the faintest, innermost rings
    int budget = s == 0 ? uBudget : min(uBudget, 400);
    for (int i = 0; i < BUDGET; i++) {
      if (i >= budget) break;
      float r = radiusKS(q);
      if (!(r > 0.0)) break;                                    // (NaN)
      if (!wasOut) {
        if (r > uRH * 1.02) wasOut = true;
        band = (r < uRH && r > uRH * 0.998 && (r - rLast) / lenPrev < 0.02) ? band + 1 : 0;
        if (band >= 3) break;                                   // creeping up to the horizon from inside: the past
      }
      rLast = r;
      if (wasOut && r < uRH * 1.01) break;                      // into the hole
      // Captured rays end once nothing more can be seen along them: inside the disk's inner edge, (centre ray) after
      // both recorded crossings, (bundle ray) once the disk images in front have covered it, or half a turn after
      // capture: past that only slivers of disk seen through the zone just outside the horizon remain, while any
      // direct view of the inner disk comes well before. Without this, at fast spin captured rays circled there until
      // the step budget ran out (35% of rays at 8 M; then still 263 steps each with only the first three endings).
      if (capt && !doomed && r < rC && r < rPrev) doomed = true;
      if (doomed && (r < uRin * 0.9 || (s == 0 ? nHits >= 2 : trans < 0.02) || doomSweep > 3.14159265)) break;
      rPrev = r;
      vec3 k1q, k1p; geodesic(q, p, pt, k1q, k1p);
      if (r > uRFar && dot(q, k1q) < 0.0) {
        escaped = 1.0;
        // w = 1 + g, g = the starlight's frequency shift (observed / emitted): the camera measures the photon's
        // energy as 1 (its tetrad), a source at rest far away as E = −p_t
        if (s == 0) outSky = vec4(normalize(-k1q), 1.0 + (pt < 0.0 ? 1.0 / (-pt) : 1.0));
        break;
      }
      float len = stepLength(r);
      lenPrev = len;
      float h = -len / length(k1q);
      vec3 k2q, k2p; geodesic(q + 0.5 * h * k1q, p + 0.5 * h * k1p, pt, k2q, k2p);
      vec3 k3q, k3p; geodesic(q + 0.5 * h * k2q, p + 0.5 * h * k2p, pt, k3q, k3p);
      vec3 k4q, k4p; geodesic(q + h * k3q, p + h * k3p, pt, k4q, k4p);
      vec3 nq = q + h / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
      vec3 np = p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
      // once captured, the winding just outside the horizon no longer counts: it is not the photon ring, and at fast
      // spin it flagged the whole shadow (its rays circle there before falling in)
      float dSweep = atan(length(cross(q, nq)), dot(q, nq));
      if (doomed) doomSweep += dSweep; else sweep += dSweep;
      if (q.z * nq.z <= 0.0 && q.z != nq.z) {                   // crossed the equatorial plane
        vec3 c = mix(q, nq, q.z / (q.z - nq.z));
        float rc = sqrt(max(dot(c.xy, c.xy) - a2, 0.0));
        if (rc > uRin * 0.9 && rc < uRout) {
          bool looped = sweep > LOOP_SWEEP;                     // a higher-order image (light went around the hole)
          if (s == 0 && looped) loopedHit = true;
          if (s == 0 && nHits < 2) {
            float phi = atan(c.y, c.x) - atan(uSpin, rc);       // Kerr–Schild → Boyer–Lindquist azimuth
            vec2 hv = vec2(looped ? -rc : rc, phi);             // r < 0 marks "looped" (replaced by the average on ring pixels)
            if (nHits == 0) hits.xy = hv; else hits.zw = hv;
          }
          nHits++;
          if (looped) {
            vec2 da = diskAverage(rc);
            light += trans * da.y; lightR += trans * da.y * rc;
            trans *= 1.0 - da.x;
          }
        }
      }
      q = nq; p = np;
    }
    if (s == 0) {
      if (sweep > uRingSweep || (RING_PASS == 1 && loopedHit)) ring.w = 1.0;   // a photon-ring pixel
    } else {
      ring.xyz += vec3(light, lightR, trans * escaped);
    }
  }
  outHits = hits;
  outRing = ring;
#if RING_PASS == 1
  outRing.w = ring.w > 0.0 && uBundle == 1 ? 2.0 : 1.0;         // refined (ring) pixel; pass 2 fills in its light
#elif RING_PASS == 2
  outRing = vec4(ring.xyz / float(max(uSubN, 1)), 2.0);
#endif
}`;

// ------------------------------------------------------------------------------------------------ RING_MASK
// Marks (stencil = 2) the refined pixels that pass 1 found on the photon ring, for the bundle pass.
export const RING_MASK = `#version 300 es
precision highp float;
out vec4 o0;
uniform highp sampler2D uRefRing;
void main() {
  if (texelFetch(uRefRing, ivec2(gl_FragCoord.xy), 0).w < 1.5) discard;
  o0 = vec4(0.0);
}`;

// ------------------------------------------------------------------------------------------------ REFINE_MASK
// Marks (stencil = 1, by not discarding) the screen pixels the edge refinement must re-trace: a looped crossing or
// ring texel, or any disagreement in class (either crossing, escaped or not), in the 3×3 cache neighbourhood.
// Why a separate pass: the GPU runs pixels in lockstep groups (~64). When the refinement itself decided per pixel,
// the ~1% of pixels that trace (17 rays each on the ring) were scattered over almost every group, so nearly the
// whole screen waited on them (72 ms for 0.6% of the pixels). With the stencil test, the GPU drops unmarked
// pixels before the shader starts and packs only marked ones into groups.
export const REFINE_MASK = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
uniform highp sampler2D uCacheRing, uCacheHits, uCacheSky;
uniform vec2 uCacheSize, uRes;
void main() {
  ivec2 cc = ivec2(gl_FragCoord.xy * uCacheSize / uRes), hi = ivec2(uCacheSize) - 1;
  vec4 h0 = texelFetch(uCacheHits, clamp(cc, ivec2(0), hi), 0);
  float e0 = texelFetch(uCacheSky, clamp(cc, ivec2(0), hi), 0).w;
  bool near = false;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    ivec2 c = clamp(cc + ivec2(i, j), ivec2(0), hi);
    vec4 ch = texelFetch(uCacheHits, c, 0);
    float ce = texelFetch(uCacheSky, c, 0).w;
    if (texelFetch(uCacheRing, c, 0).w > 0.5 || ch.x < 0.0 || ch.z < 0.0) near = true;            // looped light
    if (sign(ch.x) != sign(h0.x) || sign(ch.z) != sign(h0.z) || (ce > 0.5) != (e0 > 0.5)) near = true; // an edge
  }
  if (!near) discard;
  o0 = vec4(0.0); o1 = vec4(0.0); o2 = vec4(0.0);             // colour writes are masked off: stencil only
}`;

/** The geodesic pass with the step policy from kerr.js filled in. pass: 0 cache, 1 edge refinement, 2 ring bundle. */
export function geodesicSource({ budget, near, far, horizon }, pass = 0) {
  const num = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
  return GEODESIC.replace('__RING_PASS__', String(pass)).replace('__BUDGET__', String(budget)).replace('__CAPTURE_SAMPLES__', String(CAPTURE_SAMPLES)).replace('__STEP_NEAR__', num(near))
    .replace('__STEP_FAR__', num(far)).replace('__STEP_HORIZON__', num(horizon));
}

// ------------------------------------------------------------------------------------------------ NOISE
// Tileable noise (period = texture size) in four channels: R fine fbm, G ridged filaments, B soft clumps, A dust.
export const NOISE = `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform float uSize;
${HASH}
float lattice(vec2 i, float period, float seed) { return hash3(vec3(mod(i, period), seed)).x; }
float vnoise(vec2 x, float period, float seed) {
  vec2 i = floor(x), f = fract(x), u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float a = lattice(i, period, seed), b = lattice(i + vec2(1, 0), period, seed);
  float c = lattice(i + vec2(0, 1), period, seed), d = lattice(i + vec2(1, 1), period, seed);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 uv, float base, float seed, int oct) {
  float s = 0.0, amp = 0.5, norm = 0.0, period = base;
  for (int i = 0; i < 9; i++) {
    if (i >= oct) break;
    s += amp * vnoise(uv * period, period, seed + float(i));
    norm += amp; amp *= 0.55; period *= 2.0;
  }
  return s / norm;
}
float ridged(vec2 uv, float base, float seed) {
  float s = 0.0, amp = 0.5, norm = 0.0, period = base;
  for (int i = 0; i < 7; i++) {
    float n = 1.0 - abs(vnoise(uv * period, period, seed + float(i)) * 2.0 - 1.0);
    s += amp * n * n; norm += amp; amp *= 0.5; period *= 2.0;
  }
  return s / norm;
}
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  outColor = vec4(fbm(uv, 8.0, 1.0, 9), ridged(uv, 6.0, 20.0), fbm(uv, 4.0, 40.0, 5), fbm(uv, 6.0, 60.0, 7));
}`;

// ------------------------------------------------------------------------------------------------ SHADE
const SHADE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 outColor;

uniform sampler2D uHits, uSky, uRing; // geodesic cache
uniform sampler2D uRHits, uRSky, uRRing; // edge refinement at screen resolution (ring.w: 0 none, 1 edge, 2 ring)
uniform float uRefScale;         // refinement pixels per pixel of this pass (2 for the half-res moving sky)
uniform float uGradK;            // disk texture detail: 1, or < 1 while moving (sharper; see app.js)
uniform vec2 uJit;               // where this cache's rays sit inside their texels (moving: jittered per frame, see TAA)
uniform vec2 uGSize;             // cache size (pixels)
uniform vec2 uOutSize;           // output size
uniform sampler2D uNoise;        // tileable disk noise
uniform sampler2D uMilkyWay;     // NASA SVS Deep Star Maps 2020 (galactic, equirectangular)
uniform sampler2D uBlackbody;    // colour.js lookup table
uniform mat3 uGalaxy;            // world → galactic frame
uniform float uTime, uSpin, uRin, uRout, uPix, uMWBright, uStarBright, uDiskBright;
uniform vec2 uDiskTemp;          // disk temperature at 6 M and its cap (K): the film look's palette (LOOK.diskTemp)
uniform float uDust;             // how dark the dust lanes are, 0..1 (LOOK.dust)
uniform float uFalloff, uDensPow; // brightness ∝ x^−uFalloff; gas density contrast (power > 1: more dark gaps)
uniform int uSkyMode;            // 0 real sky, 1 checkerboard
uniform float uShift;            // 0..1: how much of the starlight's frequency shift g to show (the film shows none;
                                 // falling in, it fades in: the light from outside turns bluer and brighter)
const int uPass = __PASS__;      // 1 sky colour (cached), 2 disk over the cached sky: two separate programs, so
                                 // the per-frame one carries no sky code (a uniform branch was flattened = 5× slower)
uniform float uShiftCap;         // brightness cap of the shift (g⁴)
uniform float uReality;          // 0 film (no Doppler shift on the disk, as the film chose), 1 Reality (the true shift g)
uniform vec3 uCam;               // the camera (the same as the geodesic pass's): each pixel's ray gives its own
uniform vec4 uE0, uER, uEU, uEF; // E = −p_t and L = p_φ, both conserved, so the disk's shift needs no extra cache
uniform vec2 uTan;
uniform sampler2D uSkyCol;       // the cached sky colour (pass 2)
uniform vec2 uSkyScale;          // its resolution relative to this pass (0.5 while moving)
${HASH}

const float PI = 3.14159265, TAU = 6.2831853;
const float LUT_MIN = 1000.0, LUT_MAX = 40000.0;

vec3 blackbody(float T) {
  float x = log(clamp(T, LUT_MIN, LUT_MAX) / LUT_MIN) / log(LUT_MAX / LUT_MIN);
  return texture(uBlackbody, vec2((x * 255.0 + 0.5) / 256.0, 0.5)).rgb;
}

// --- reading the cache: bilinear when the 4 neighbours agree (same hits, same fate), nearest otherwise ---
// hits: r > 0 a direct disk crossing, r < 0 one after the ray looped the photon sphere (on ring pixels the ring
// average replaces those), 0 none. ring.w = 1 on photon-ring pixels.
struct Px { vec4 hits; vec4 sky; vec4 ring; };
Px fetchPx(ivec2 c) {
  c = clamp(c, ivec2(0), ivec2(uGSize) - 1);
  return Px(texelFetch(uHits, c, 0), texelFetch(uSky, c, 0), texelFetch(uRing, c, 0));
}
float unwrap(float ref, float a) { return a - TAU * floor((a - ref) / TAU + 0.5); }
bool sameClass(Px a, Px b) {
  return sign(a.hits.x) == sign(b.hits.x) && sign(a.hits.z) == sign(b.hits.z) && (a.sky.w > 0.5) == (b.sky.w > 0.5);
}
Px lerpPx(Px p00, Px p10, Px p01, Px p11, vec2 t) {
  float w00 = (1.0 - t.x) * (1.0 - t.y), w10 = t.x * (1.0 - t.y), w01 = (1.0 - t.x) * t.y, w11 = t.x * t.y;
  Px o;
  o.sky = vec4(normalize(p00.sky.xyz * w00 + p10.sky.xyz * w10 + p01.sky.xyz * w01 + p11.sky.xyz * w11), p00.sky.w);
  vec4 h10 = p10.hits, h01 = p01.hits, h11 = p11.hits;
  h10.y = unwrap(p00.hits.y, h10.y); h01.y = unwrap(p00.hits.y, h01.y); h11.y = unwrap(p00.hits.y, h11.y);
  h10.w = unwrap(p00.hits.w, h10.w); h01.w = unwrap(p00.hits.w, h01.w); h11.w = unwrap(p00.hits.w, h11.w);
  o.hits = p00.hits * w00 + h10 * w10 + h01 * w01 + h11 * w11;
  o.ring = p00.ring * w00 + p10.ring * w10 + p01.ring * w01 + p11.ring * w11;
  return o;
}

// --- the disk (film: thin, not accreting, artist-textured; colour from temperature) ---
float diskTemp(float x) { return clamp(uDiskTemp.x * pow(x, -0.8), 1500.0, uDiskTemp.y); } // hotter inside: white → orange → brown
// The disk's look (temperature, brightness, texture scale, dust, wisps) is laid out in units of DISK_REF = 6 M, the
// inner edge of a non-spinning hole's disk, the look the owner approved. Spin only moves the inner edge (the ISCO,
// 6 M → 3.8 M at the film's a = 0.6 → 1.2 M at 0.999): the disk reaches closer and hotter, and the rest of it keeps
// its look. (Scaling everything by the ISCO turned a fast-spinning hole's whole disk into a smooth orange blob.)
// Brightness keeps rising inward only to 0.85 DISK_REF (at most ~1.5× the approved inner edge): with 0.6 (3.2×) the
// inner disk of a fast-spinning hole washed out into the shadow.
const float DISK_REF = 6.0;

// Reality mode: frequency shift g = ν_seen / ν_emitted of the gas on its circular orbit at r (kerr.js diskShift):
// g = 1 / (u^t (E − Ω L)), Doppler and gravitational shift together. The light seen is a blackbody at g·T, g⁴ as
// bright (Liouville: I_ν / ν³ is constant along a ray; integrated over frequency, g⁴), capped (the tone mapping
// turns the rest into a white-out). E, L: this pixel's ray (set in main()).
float rayE = 1.0, rayL = 0.0;
float diskShift(float r) {
  float s = sqrt(r), r15 = r * s;
  float ut = (r15 + uSpin) / (sqrt(r15) * sqrt(max(r15 - 3.0 * s + 2.0 * uSpin, 1e-6)));
  return mix(1.0, 1.0 / max(ut * (rayE - rayL / (r15 + uSpin)), 1e-3), uReality);
}
vec3 shifted(float T, float g) { return blackbody(T * g) * min(g * g * g * g, uShiftCap); }

// returns (light, opacity). dR, dPhi: how r and φ change per output pixel (x, y), from neighbouring cached rays
// (explicit, because GPU screen derivatives are undefined inside branches); they pick the texture's blur level.
vec4 disk(float r, float phi, vec2 dR, vec2 dPhi, float t) {
  float omega = 1.0 / (pow(r, 1.5) + uSpin);                 // Kepler (Kerr prograde) angular velocity
  float dOmega = -1.5 * sqrt(r) * omega * omega;              // dΩ/dr: the shear
  const float P = 80.0;                                       // two texture phases, cross-faded: no endless shear
  float t1 = mod(t, P), t2 = mod(t + 0.5 * P, P), wgt = abs(t1 / P - 0.5) * 2.0;
  float lr = log(r / DISK_REF);
  vec2 dr = dR / r;
  vec4 n = vec4(0.0);
  for (int k = 0; k < 2; k++) {
    float tk = k == 0 ? t1 : t2;
    float ph = (phi - omega * tk) / TAU + (k == 0 ? 0.0 : 0.37);
    vec2 uvA = vec2(ph * 2.0, lr * 1.2);                      // big structure
    vec2 uvB = vec2(ph * 3.0, lr * 7.0);                      // long thin filaments along the orbits
    vec2 uvC = vec2(ph * 6.0, lr * 22.0);                     // fine grain
    vec2 dA = (dPhi - dOmega * tk * dR) / TAU;
    vec4 a = textureGrad(uNoise, uvA, vec2(dA.x * 2.0, dr.x * 1.2), vec2(dA.y * 2.0, dr.y * 1.2));
    vec4 b = textureGrad(uNoise, uvB, vec2(dA.x * 3.0, dr.x * 7.0), vec2(dA.y * 3.0, dr.y * 7.0));
    vec4 c = textureGrad(uNoise, uvC, vec2(dA.x * 6.0, dr.x * 22.0), vec2(dA.y * 6.0, dr.y * 22.0));
#if LITE
    vec4 m = vec4(b.g, c.r, a.b, a.a);                          // lite: no cloud layer (3 texture reads, not 4)
#else
    vec2 uvD = vec2(ph * 40.0, lr * 30.0);                    // turbulent clouds, only mildly stretched
    vec4 e = textureGrad(uNoise, uvD, vec2(dA.x * 40.0, dr.x * 30.0), vec2(dA.y * 40.0, dr.y * 30.0));
    vec4 m = vec4(b.g, c.r * 0.5 + e.r * 0.5, a.b, a.a * 0.55 + e.a * 0.45); // filaments, grain/clouds, clumps, dust
#endif
    m = clamp((m - 0.5) * vec4(1.3, 2.2, 2.0, 2.4) + 0.5, 0.0, 1.0); // fbm is too even around 0.5: stretch contrast
    n += m * (k == 0 ? 1.0 - wgt : wgt);
  }
  float x = r / DISK_REF;
  float inner = smoothstep(0.92, 1.12, r / uRin);
  float edge = smoothstep(uRout, uRout * (0.15 + 0.45 * n.z), r);           // ragged outer edge
  float filaments = smoothstep(0.12, 0.92, n.x);
  float grain = 0.7 + 0.6 * n.y;
  float clumps = smoothstep(0.3, 0.8, n.z);
  float dust = smoothstep(0.5, 0.7, n.w) * smoothstep(1.5, 3.2, x);         // dark lanes, not right at the edge
  float outerPart = smoothstep(1.8, 4.5, x);
  float dens = inner * edge * (0.45 + 0.55 * filaments) * grain * (0.6 + 0.7 * clumps);
  dens *= mix(1.0, 0.15 + 1.5 * clumps * filaments, outerPart);            // outside: patchy wisps, not a sheet
  float T = diskTemp(x);
  dens = pow(dens, uDensPow);
  float glow = uDiskBright * pow(max(x, 0.85), -uFalloff) * (1.0 + 4.0 * pow(clumps * filaments, 6.0));
  vec3 light = shifted(T, diskShift(r)) * glow * dens * (1.0 - uDust * dust);
  float alpha = clamp(dens * 2.2 + dust * 0.6 * inner * edge, 0.0, 0.97) * mix(1.0, 0.6, smoothstep(2.5, uRout / DISK_REF, x));
  return vec4(light, alpha);
}

// --- sky ---
vec3 starColor(float k, float g) { return blackbody(mix(3200.0, 14000.0, k * k) * g); } // observed temperature g·T

// Point stars, drawn through the lensed footprint J of this pixel (ray differentials; see engineering log)
vec3 stars(vec3 d, vec3 Jx, vec3 Jy, float shift) {
  vec3 ad = abs(d), col = vec3(0.0);
  float face; vec2 uv;
  if (ad.x >= ad.y && ad.x >= ad.z) { face = d.x > 0.0 ? 0.0 : 1.0; uv = d.yz / ad.x; }
  else if (ad.y >= ad.z) { face = d.y > 0.0 ? 2.0 : 3.0; uv = d.xz / ad.y; }
  else { face = d.z > 0.0 ? 4.0 : 5.0; uv = d.xy / ad.z; }
  float gxx = dot(Jx, Jx), gxy = dot(Jx, Jy), gyy = dot(Jy, Jy), det = gxx * gyy - gxy * gxy;
  if (det < 1e-30) return col;
  float mu = min(uPix * uPix / max(length(cross(Jx, Jy)), 1e-20), 60.0);
  for (int layer = 0; layer < (LITE == 1 ? 2 : 3); layer++) {    // lite: the two brighter layers only
    float N = layer == 0 ? 30.0 : (layer == 1 ? 90.0 : 260.0);
    vec2 g = (uv * 0.5 + 0.5) * N, c0 = floor(g - 0.5);         // the 2×2 cells nearest this point
    for (int j = 0; j <= 1; j++) for (int i = 0; i <= 1; i++) {
      vec2 c = c0 + vec2(i, j);
      vec3 h = hash3(vec3(c, face * 31.0 + float(layer) * 7.0));
      vec2 sp = ((c + h.xy) / N) * 2.0 - 1.0;
      vec3 sd = face < 2.0 ? vec3(face < 1.0 ? 1.0 : -1.0, sp) : (face < 4.0 ? vec3(sp.x, face < 3.0 ? 1.0 : -1.0, sp.y) : vec3(sp, face < 5.0 ? 1.0 : -1.0));
      vec3 o = normalize(sd) - d;
      float ox = dot(o, Jx), oy = dot(o, Jy);
      float px = (gyy * ox - gxy * oy) / det, py = (gxx * oy - gxy * ox) / det;
      float flux = pow(h.z, 28.0 + float(layer) * 12.0) * (layer == 0 ? 8.0 : 3.0);
      col += starColor(fract(h.z * 91.7), shift) * flux * mu * exp(-(px * px + py * py) / 0.45);
    }
  }
  return col * uStarBright;
}

vec3 milkyWay(vec3 d, vec3 Jx, vec3 Jy) {
  vec3 g = uGalaxy * d;
  vec2 uv = vec2(0.5 - atan(g.y, g.x) / TAU, 0.5 - asin(clamp(g.z, -1.0, 1.0)) / PI);
  // texture gradients from the lensed footprint, so the image stays sharp where lensing magnifies it
  vec3 gx = uGalaxy * Jx, gy = uGalaxy * Jy;
  float rho2 = max(g.x * g.x + g.y * g.y, 1e-6);
  vec2 dux = vec2(-(g.x * gx.y - g.y * gx.x) / (TAU * rho2), -gx.z / (PI * sqrt(rho2)));
  vec2 duy = vec2(-(g.x * gy.y - g.y * gy.x) / (TAU * rho2), -gy.z / (PI * sqrt(rho2)));
  vec3 c = pow(textureGrad(uMilkyWay, uv, dux, duy).rgb, vec3(2.2));
  return max(c - 0.05, 0.0) * uMWBright;                       // black point: drop the faint all-sky haze
}

vec3 checker(vec3 d, float s) {
  float th = acos(clamp(d.z, -1.0, 1.0)), ph = atan(d.y, d.x) + PI;
  float step10 = PI / 18.0;
  vec2 cell = floor(vec2(ph, th) / step10);
  vec3 h = hash3(vec3(cell, 5.0));
  vec3 base = mix(vec3(0.25, 0.1, 0.35), vec3(0.15, 0.55, 0.45), h.x) + 0.25 * h.yzx;
  vec2 fr = abs(fract(vec2(ph, th) / step10) - 0.5) * step10;
  float edge = min(step10 * 0.5 - fr.x, step10 * 0.5 - fr.y);
  float line = 1.0 - smoothstep(0.0, max(s * 1.5, 0.002), edge);
  return mix(base * 0.25, vec3(0.6), line);
}

// per output pixel change of (r1, φ1, r2, φ2) between two neighbouring cached rays (0 where a neighbour has no hit)
vec4 hitDiff(Px a, Px b, float s) {
  vec4 d = vec4(0.0);
  if (a.hits.x != 0.0 && b.hits.x != 0.0) d.xy = vec2(abs(a.hits.x) - abs(b.hits.x), unwrap(0.0, a.hits.y - b.hits.y));
  if (a.hits.z != 0.0 && b.hits.z != 0.0) d.zw = vec2(abs(a.hits.z) - abs(b.hits.z), unwrap(0.0, a.hits.w - b.hits.w));
  return d * 0.5 * s;
}

// light from the disk crossings of one ray (rgb) and how much of what's behind still shows through (a).
// On photon-ring pixels the looped crossings are replaced by the ring average (light, then sky-through).
vec4 diskPart(Px px, vec4 ring, vec4 Hx, vec4 Hy, float t, vec3 skyRaw) {
  vec3 col = vec3(0.0);
  float trans = 1.0;
  px.ring = ring;
  bool ringPx = ring.w > 0.5;
  float r1 = px.hits.x, r2 = px.hits.z;
  if (r1 != 0.0 && !(ringPx && r1 < 0.0)) { vec4 d1 = disk(abs(r1), px.hits.y, vec2(Hx.x, Hy.x), vec2(Hx.y, Hy.y), t); col += trans * d1.a * d1.rgb; trans *= 1.0 - d1.a; }
  if (r2 != 0.0 && !(ringPx && r2 < 0.0)) { vec4 d2 = disk(abs(r2), px.hits.w, vec2(Hx.z, Hy.z), vec2(Hx.w, Hy.w), t); col += trans * d2.a * d2.rgb; trans *= 1.0 - d2.a; }
  if (ringPx) {
    float rMean = px.ring.y / max(px.ring.x, 1e-6);
    col += trans * shifted(diskTemp(rMean / DISK_REF), diskShift(rMean)) * uDiskBright * px.ring.x;
    col += trans * px.ring.z * skyRaw;
    return vec4(col, 0.0);                                     // sky already added
  }
  return vec4(col, trans);
}
// g: frequency shift of the starlight. Brightness ∝ g⁴ (Liouville: I_ν/ν³ is constant along a ray, integrated over
// frequency), capped (tone mapping turns the rest into a white-out); the Milky Way takes a sun-like source's colour
// change, the stars their own temperatures × g.
vec3 skyColor(vec3 d, vec3 Jx, vec3 Jy, float g) {
  if (uSkyMode == 1) return checker(d, max(length(Jx), uPix));
  float I = min(pow(g, 4.0), 40.0);
  vec3 tint = blackbody(5800.0 * g) / max(blackbody(5800.0), vec3(1e-3));
  return I * (milkyWay(d, Jx, Jy) * tint + stars(d, Jx, Jy, g));
}

void main() {
  vec2 scale = uGSize / uOutSize;
  {                                                            // this pixel's ray: E and L (GEODESIC main())
    vec3 dir = normalize(vec3((gl_FragCoord.xy / uOutSize * 2.0 - 1.0) * uTan, 1.0));
    vec4 pc = uE0 - dir.x * uER - dir.y * uEU - dir.z * uEF;
    rayE = -pc.x; rayL = uCam.x * pc.z - uCam.y * pc.y;
  }
  vec2 gp = gl_FragCoord.xy * scale;
  vec2 f = gp - 0.5 - uJit, fi = floor(f), tt = f - fi;
  ivec2 c = ivec2(fi);
  Px p00 = fetchPx(c), p10 = fetchPx(c + ivec2(1, 0)), p01 = fetchPx(c + ivec2(0, 1)), p11 = fetchPx(c + ivec2(1, 1));
  bool uniformClass = sameClass(p00, p10) && sameClass(p00, p01) && sameClass(p00, p11);
  // the sky's footprint per output pixel, from the cache (explicit ray differentials)
  ivec2 gc = ivec2(floor(gp - uJit));
  Px nx1 = fetchPx(gc + ivec2(1, 0)), nx0 = fetchPx(gc - ivec2(1, 0)), ny1 = fetchPx(gc + ivec2(0, 1)), ny0 = fetchPx(gc - ivec2(0, 1));
  vec3 Jx = (nx1.sky.xyz - nx0.sky.xyz) * 0.5 * scale.x, Jy = (ny1.sky.xyz - ny0.sky.xyz) * 0.5 * scale.y;
  vec4 Hx = hitDiff(nx1, nx0, scale.x) * uGradK, Hy = hitDiff(ny1, ny0, scale.y) * uGradK;
  float lim = 30.0 * uPix;                                     // across an edge (disk / hole) the difference is meaningless
  if (length(Jx) > lim) Jx *= lim / length(Jx);
  if (length(Jy) > lim) Jy *= lim / length(Jy);

  // disk clock, M per second: 8 makes the orbits visible (inner edge ~6 s per turn at a = 0.6, 10 M ~25 s); at 2
  // the disk looked frozen, because its streaks run along the direction it moves (owner, 2026-11-12)
  float t = uTime * 8.0;
  // Smooth area: interpolate the rays and shade once. Edge (disk / hole / sky meet): shade the four rays and blend
  // the colours (anti-aliased edges). One call site in a loop keeps the compiled shader small.
  // uPass 1 = sky only (static: run when the cache changes, written to a texture);
  // uPass 2 = every frame: the disk, with the cached sky colour behind it.
  // Refined pixel (edge or photon ring, traced at screen resolution by the refinement pass): use its own ray.
  ivec2 rc = ivec2(gl_FragCoord.xy * uRefScale);
  Px R = Px(texelFetch(uRHits, rc, 0), texelFetch(uRSky, rc, 0), texelFetch(uRRing, rc, 0));
  bool refined = R.ring.w > 0.5;
  R.ring = R.ring.w > 1.5 ? vec4(R.ring.xyz, 1.0) : vec4(0.0);
  vec3 skyRaw = vec3(0.0);
  if (uPass == 2) skyRaw = uSkyScale.x >= 1.0 ? texelFetch(uSkyCol, ivec2(gl_FragCoord.xy), 0).rgb
                                              : texture(uSkyCol, gl_FragCoord.xy * uSkyScale / vec2(textureSize(uSkyCol, 0))).rgb;
  vec3 col = vec3(0.0);
  float wsum = 0.0;
  Px lerped = lerpPx(p00, p10, p01, p11, tt);
  bool one = refined || uniformClass;
  int n = one ? 1 : 4;
  for (int i = 0; i < 4; i++) {
    if (i >= n) break;
    Px q = lerped;                                             // (WebGL forbids ?: on structs)
    if (refined) q = R;
    else if (!uniformClass) { if (i == 0) q = p00; else if (i == 1) q = p10; else if (i == 2) q = p01; else q = p11; q.ring = vec4(0.0); }
    else q.ring = vec4(0.0);
    float w = one ? 1.0 : (i == 0 ? (1.0 - tt.x) * (1.0 - tt.y) : (i == 1 ? tt.x * (1.0 - tt.y) : (i == 2 ? (1.0 - tt.x) * tt.y : tt.x * tt.y)));
    float esc = q.sky.w > 0.5 ? 1.0 : 0.0;
    if (uPass == 1) {
      if (esc > 0.0) { col += w * skyColor(q.sky.xyz, Jx, Jy, mix(1.0, q.sky.w - 1.0, uShift)); wsum += w; }
    } else {
      vec4 dp = diskPart(q, q.ring, Hx, Hy, t, skyRaw);
      col += w * (dp.rgb + dp.a * esc * skyRaw);
    }
  }
  if (uPass == 1) col /= max(wsum, 1e-6);                       // the sky seen by this pixel's escaping rays
  outColor = vec4(col, 1.0);
}`;

/** SHADE as the sky pass (1) or the per-frame disk pass (2). */
export const shadeSource = (pass, lite = false) => SHADE.replace('__PASS__', String(pass)).replace('#version 300 es\n', `#version 300 es\n#define LITE ${lite ? 1 : 0}\n`);

// ------------------------------------------------------------------------------------------------ BLOOM
// Down: 13-tap filter (Jimenez 2014, "Next generation post processing in Call of Duty"), first level Karis-averaged
// so single bright pixels don't flicker. Up: 3×3 tent, added onto the next bigger level.
export const BLOOM_DOWN = `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;      // 1 / source size
uniform int uFirst;
vec3 s(vec2 uv) { return texture(uSrc, uv).rgb; }
float karis(vec3 c) { return 1.0 / (1.0 + max(max(c.r, c.g), c.b)); }
void main() {
  vec2 t = uTexel, uv = (floor(gl_FragCoord.xy) * 2.0 + 1.0) * t;  // centre of the 2×2 source block
  vec3 a = s(uv + t * vec2(-2, 2)), b = s(uv + t * vec2(0, 2)), c = s(uv + t * vec2(2, 2));
  vec3 d = s(uv + t * vec2(-2, 0)), e = s(uv), f = s(uv + t * vec2(2, 0));
  vec3 g = s(uv + t * vec2(-2, -2)), h = s(uv + t * vec2(0, -2)), i = s(uv + t * vec2(2, -2));
  vec3 j = s(uv + t * vec2(-1, 1)), k = s(uv + t * vec2(1, 1)), l = s(uv + t * vec2(-1, -1)), m = s(uv + t * vec2(1, -1));
  vec3 o;
  if (uFirst == 1) {
    vec3 g0 = (j + k + l + m) * 0.25, g1 = (a + b + d + e) * 0.25, g2 = (b + c + e + f) * 0.25;
    vec3 g3 = (d + e + g + h) * 0.25, g4 = (e + f + h + i) * 0.25;
    float w0 = karis(g0) * 0.5, w1 = karis(g1) * 0.125, w2 = karis(g2) * 0.125, w3 = karis(g3) * 0.125, w4 = karis(g4) * 0.125;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  outColor = vec4(o, 1.0);
}`;

export const BLOOM_UP = `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uSrc;
uniform vec2 uTexel;      // 1 / source size
uniform vec2 uDstSize;
uniform float uWeight;
vec3 s(vec2 uv) { return texture(uSrc, uv).rgb; }
void main() {
  vec2 uv = gl_FragCoord.xy / uDstSize, t = uTexel;
  vec3 o = s(uv) * 4.0 + (s(uv + vec2(t.x, 0)) + s(uv - vec2(t.x, 0)) + s(uv + vec2(0, t.y)) + s(uv - vec2(0, t.y))) * 2.0
         + s(uv + t) + s(uv - t) + s(uv + vec2(t.x, -t.y)) + s(uv + vec2(-t.x, t.y));
  outColor = vec4(o / 16.0 * uWeight, 1.0);
}`;

// ------------------------------------------------------------------------------------------------ COMPOSITE
// ------------------------------------------------------------------------------------------------ TAA
// Moving frames: the low-resolution rays move inside their texels from frame to frame (a Halton sequence), and each
// frame is blended with the history, clamped to the range of its 3×3 neighbourhood in the new frame (so a turn or a
// new feature can't leave a smear): over a few frames the picture gathers the detail of several ray positions.
export const TAA = `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uCur, uHist;
uniform float uAlpha;     // weight of the history (0: just the new frame)
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy), hi = textureSize(uCur, 0) - 1;
  vec3 cur = texelFetch(uCur, c, 0).rgb, mn = cur, mx = cur;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec3 v = texelFetch(uCur, clamp(c + ivec2(i, j), ivec2(0), hi), 0).rgb;
    mn = min(mn, v); mx = max(mx, v);
  }
  vec3 h = clamp(texelFetch(uHist, c, 0).rgb, mn, mx);
  outColor = vec4(mix(cur, h, uAlpha), 1.0);
}`;

export const COMPOSITE = `#version 300 es
precision highp float;
out vec4 outColor;
uniform sampler2D uHdr, uBloom, uPrev, uHazeTex;
uniform vec2 uOutSize;
uniform float uExposure, uBloomAmt, uBloomNorm, uTime, uVignette, uGrain;
// Veiling glare (stage 4, the film look): a camera lens scatters a little of a very bright source's light over the
// whole frame, a wide warm haze that lifts the dark around the disk (the film's frames have it; space doesn't).
// uHaze: its amount, from a coarse bloom level (several hundred pixels wide); uHazeTint its colour.
uniform float uHaze;
uniform vec3 uHazeTint;
uniform float uSaturation;  // the film's grade is less saturated than raw blackbody colours (1 = unchanged)
uniform float uFade;      // 0..1: how much of the frame before a cache switch still shows (smooth hand-over)
${HASH}
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
void main() {
  vec2 uv = gl_FragCoord.xy / uOutSize;
  vec3 hdr = mix(texture(uHdr, uv).rgb, texture(uPrev, uv).rgb, uFade);
  vec3 bloom = texture(uBloom, uv).rgb * uBloomNorm;          // the up-chain sums every level: average them
  vec3 c = mix(hdr, bloom, uBloomAmt) * uExposure;            // energy-conserving glow (light spread, not added)
  c += uHaze * uHazeTint * texture(uHazeTex, uv).rgb * uExposure;
  float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = max(mix(vec3(lum), c, uSaturation), 0.0);
  c = aces(c);
  c = pow(c, vec3(1.0 / 2.2));
  vec2 v = uv - 0.5;
  c *= 1.0 - uVignette * dot(v, v) * 1.6;
  float n = hash3(vec3(gl_FragCoord.xy, floor(uTime * 24.0))).x - 0.5;
  c += n * uGrain;
  outColor = vec4(c, 1.0);
}`;
