// The wormhole's GPU passes.
//   RAYS   one row: the light ray for every angle ψ from the direction of increasing ℓ (worm.js trace(), same
//          equations and step rule), recomputed when the camera's ℓ changes. By spherical symmetry that one angle is
//          all a pixel's ray depends on, so 2048 rays serve the whole screen (cheap enough for phones).
//   VIEW   per pixel: its ψ, the ray from the table, the sky it ends on (Saturn's side: the Milky Way and Saturn;
//          the far side: Gargantua's galaxy), tone mapping.
import { HASH } from '../shader.js';
import { STEP } from './worm.js';

const num = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

export const RAYS = `#version 300 es
precision highp float;
out vec4 outRay;          // (Φ, side −1 Saturn / +1 far / 0 out of steps, steps / 1000, 0)
uniform float uL0, uRho, uA, uM, uN;
const float PI = 3.14159265;
float radius(float l) {
  float d = abs(l) - uA;
  if (d <= 0.0) return uRho;
  float x = 2.0 * d / (PI * uM);
  return uRho + uM * (x * atan(x) - 0.5 * log(1.0 + x * x));
}
float dradius(float l) { float d = abs(l) - uA; return d <= 0.0 ? 0.0 : sign(l) * (2.0 / PI) * atan(2.0 * d / (PI * uM)); }
vec3 f(float l, float pl, float b) { float r = radius(l); return vec3(pl, b / (r * r), b * b * dradius(l) / (r * r * r)); }
void main() {
  float psi = PI * (gl_FragCoord.x - 0.5) / (uN - 1.0);   // table entry i ↔ ψ = π i / (N − 1): both ends exact
  float b = radius(uL0) * sin(psi), far = max(4.0 * abs(uL0), 50.0);
  float l = uL0, ph = 0.0, pl = cos(psi);
  outRay = vec4(ph, 0.0, 6.0, 0.0);
  for (int i = 0; i < 3000; i++) {                            // (rays circling the throat 2+ times: a hair-thin ring, left dark)
    float r = radius(l);
    if (abs(l) > far && pl * sign(l) > 0.0) { outRay = vec4(ph + atan(b / r, abs(pl)), sign(l), float(i) / 1000.0, 0.0); return; }
    float h = ${num(STEP.near)} + ${num(STEP.grow)} * max(0.0, abs(l) - uA);
    vec3 k1 = f(l, pl, b), k2 = f(l + 0.5 * h * k1.x, pl + 0.5 * h * k1.z, b);
    vec3 k3 = f(l + 0.5 * h * k2.x, pl + 0.5 * h * k2.z, b), k4 = f(l + h * k3.x, pl + h * k3.z, b);
    vec3 d = h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    l += d.x; ph += d.y; pl += d.z;
  }
  outRay = vec4(ph, 0.0, 6.0, 0.0);
}`;

export const VIEW = `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform vec2 uRes, uTan;
uniform vec3 uU, uF, uR, uUp;      // the camera's place on the sphere (u = direction of increasing ℓ) and its axes,
                                   // all in the same local frame (see wormhole.js)
uniform sampler2D uRays;           // RAYS
uniform float uN, uPix, uTime, uExposure, uStars;
uniform sampler2D uMilkyWay;
uniform mat3 uGalaxy;              // Saturn-side sky → galactic frame
uniform vec3 uSat, uSatPole, uSun; // Saturn's direction on its sky, its pole, the direction of the Sun
uniform float uSatR;               // Saturn's angular radius (rad)
const float PI = 3.14159265, TAU = 6.2831853;
${HASH}

// --- point stars through the pixel's footprint J on the sky (as in Gargantua's shader) ---
vec3 starColor(float k) { return mix(vec3(1.0, 0.72, 0.45), vec3(0.65, 0.8, 1.0), k); }
vec3 stars(vec3 d, vec3 Jx, vec3 Jy, float seed, float density) {
  vec3 ad = abs(d), col = vec3(0.0);
  float face; vec2 uv;
  if (ad.x >= ad.y && ad.x >= ad.z) { face = d.x > 0.0 ? 0.0 : 1.0; uv = d.yz / ad.x; }
  else if (ad.y >= ad.z) { face = d.y > 0.0 ? 2.0 : 3.0; uv = d.xz / ad.y; }
  else { face = d.z > 0.0 ? 4.0 : 5.0; uv = d.xy / ad.z; }
  float gxx = dot(Jx, Jx), gxy = dot(Jx, Jy), gyy = dot(Jy, Jy), det = gxx * gyy - gxy * gxy;
  if (det < 1e-30) return col;
  float mu = min(uPix * uPix / max(length(cross(Jx, Jy)), 1e-20), 60.0);   // lensing magnification
  for (int layer = 0; layer < 2; layer++) {
    float N = layer == 0 ? 40.0 : 140.0;
    vec2 g = (uv * 0.5 + 0.5) * N, c0 = floor(g - 0.5);
    for (int j = 0; j <= 1; j++) for (int i = 0; i <= 1; i++) {
      vec2 c = c0 + vec2(i, j);
      vec3 h = hash3(vec3(c, face * 31.0 + float(layer) * 7.0 + seed));
      vec2 sp = ((c + h.xy) / N) * 2.0 - 1.0;
      vec3 sd = face < 2.0 ? vec3(face < 1.0 ? 1.0 : -1.0, sp) : (face < 4.0 ? vec3(sp.x, face < 3.0 ? 1.0 : -1.0, sp.y) : vec3(sp, face < 5.0 ? 1.0 : -1.0));
      vec3 o = normalize(sd) - d;
      float ox = dot(o, Jx), oy = dot(o, Jy);
      float px = (gyy * ox - gxy * oy) / det, py = (gxx * oy - gxy * ox) / det;
      float flux = pow(h.z, 24.0 / density + float(layer) * 10.0) * (layer == 0 ? 6.0 : 2.5);
      col += starColor(fract(h.z * 91.7)) * flux * mu * exp(-(px * px + py * py) / 0.45);
    }
  }
  return col * uStars;
}

// --- Saturn's side: the Milky Way (NASA SVS Deep Star Maps) and Saturn ---
vec3 milkyWay(vec3 d, vec3 Jx, vec3 Jy) {
  vec3 g = uGalaxy * d;
  vec2 uv = vec2(0.5 - atan(g.y, g.x) / TAU, 0.5 - asin(clamp(g.z, -1.0, 1.0)) / PI);
  vec3 gx = uGalaxy * Jx, gy = uGalaxy * Jy;
  float rho2 = max(g.x * g.x + g.y * g.y, 1e-6);
  vec2 dux = vec2(-(g.x * gx.y - g.y * gx.x) / (TAU * rho2), -gx.z / (PI * sqrt(rho2)));
  vec2 duy = vec2(-(g.x * gy.y - g.y * gy.x) / (TAU * rho2), -gy.z / (PI * sqrt(rho2)));
  return max(pow(textureGrad(uMilkyWay, uv, dux, duy).rgb, vec3(2.2)) - 0.05, 0.0) * 0.22;
}
// Saturn as a lit ball with its rings, far away (on the sky): the line of sight d from the origin meets a sphere of
// radius sin(R) at distance 1 in direction uSat, and the ring plane through it. Edges softened over a pixel.
vec4 saturn(vec3 d) {
  float R = sin(uSatR), t = dot(d, uSat), d2 = max(1.0 - t * t, 0.0), aa = uPix * 1.5;
  vec4 col = vec4(0.0);
  float tp = 1e9;
  if (t > 0.0 && d2 < R * R * 1.02) {
    float th = t - sqrt(max(R * R - d2, 0.0));
    vec3 n = normalize(d * th - uSat);
    float lat = dot(n, uSatPole);
    float bands = 0.5 + 0.5 * sin(lat * 24.0 + 1.3 * sin(lat * 9.0)) * 0.5 + 0.25 * sin(lat * 61.0);
    vec3 base = mix(vec3(0.62, 0.48, 0.3), vec3(0.86, 0.76, 0.56), bands) * mix(0.75, 1.0, smoothstep(0.9, 0.2, abs(lat)));
    float light = max(dot(n, uSun), 0.0);
    float edge = 1.0 - smoothstep(R - aa, R, sqrt(d2));
    col = vec4(base * (0.02 + 1.6 * light), edge);
    tp = th;
  }
  float dn = dot(d, uSatPole);
  if (abs(dn) > 1e-5) {
    float tr = dot(uSat, uSatPole) / dn;
    vec3 q = d * tr - uSat;
    float rr = length(q) / R;
    if (tr > 0.0 && rr > 1.2 && rr < 2.3) {
      // C ring (faint), B ring (bright), the Cassini division, A ring; brightness by radius
      float dens = rr < 1.52 ? 0.25 : rr < 1.95 ? 0.95 : rr < 2.03 ? 0.08 : 0.6;
      dens *= 0.75 + 0.25 * sin(rr * 140.0) * sin(rr * 37.0);
      dens *= smoothstep(1.2, 1.24, rr) * (1.0 - smoothstep(2.26, 2.3, rr));
      // the planet's shadow on the rings
      vec3 qp = d * tr; float s = dot(uSat - qp, uSun);
      float shadow = s > 0.0 && length(uSat - qp - s * uSun) < R ? 0.08 : 1.0;
      vec3 ring = vec3(0.82, 0.74, 0.6) * 1.3 * shadow * (0.3 + 0.7 * abs(dot(uSun, uSatPole)) * 3.0);
      float a = clamp(dens, 0.0, 1.0) * 0.9;
      if (tr < tp) col = vec4(mix(col.rgb * col.a, ring, a), max(col.a, a));   // rings in front of the planet
      else if (col.a < 0.99) col = vec4(mix(ring, col.rgb, col.a), max(col.a, a)); // behind: where the planet isn't
    }
  }
  return col;
}

// --- the far side: Gargantua's galaxy (artistic: a starfield and glowing gas, no real map of it exists) ---
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x), u = f * f * (3.0 - 2.0 * f);
  float n000 = hash3(i).x, n100 = hash3(i + vec3(1, 0, 0)).x, n010 = hash3(i + vec3(0, 1, 0)).x, n110 = hash3(i + vec3(1, 1, 0)).x;
  float n001 = hash3(i + vec3(0, 0, 1)).x, n101 = hash3(i + vec3(1, 0, 1)).x, n011 = hash3(i + vec3(0, 1, 1)).x, n111 = hash3(i + vec3(1, 1, 1)).x;
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
float fbm(vec3 x) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(x); x *= 2.03; a *= 0.5; } return s; }
vec3 farSky(vec3 d) {
  vec3 band = normalize(vec3(0.04, 0.85, 0.5));               // the plane of that galaxy, across its sky (passing behind the throat)
  float inBand = exp(-pow(dot(d, band) / 0.22, 2.0));
  float n = fbm(d * 3.0 + 4.0), m = fbm(d * 7.0 - 2.0);
  vec3 gas = mix(vec3(0.05, 0.12, 0.3), vec3(0.45, 0.12, 0.35), smoothstep(0.35, 0.7, n)) * smoothstep(0.3, 0.75, m);
  vec3 glow = vec3(0.9, 0.75, 0.6) * pow(inBand, 1.5) * (0.4 + 0.6 * n);
  return (gas * (0.25 + 0.9 * inBand) + glow * 0.35) * 0.5;
}

vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }

void main() {
  vec2 ndc = gl_FragCoord.xy / uRes * 2.0 - 1.0;
  vec3 n = normalize(uF + ndc.x * uTan.x * uR + ndc.y * uTan.y * uUp);   // this pixel's direction, local frame
  float c = clamp(dot(n, uU), -1.0, 1.0), psi = acos(c);
  // the ray for this ψ: interpolated where both table neighbours end on the same side
  float x = psi / PI * (uN - 1.0), i0 = floor(x), fr = x - i0;
  vec4 r0 = texelFetch(uRays, ivec2(int(i0), 0), 0), r1 = texelFetch(uRays, ivec2(min(int(i0) + 1, int(uN) - 1), 0), 0);
  vec4 ray = r0.y == r1.y ? mix(r0, r1, fr) : (fr < 0.5 ? r0 : r1);
  vec3 tq = n - c * uU;
  float tl = length(tq);
  vec3 t = tl > 1e-6 ? tq / tl : normalize(cross(uU, abs(uU.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0)));
  vec3 d = cos(ray.x) * uU + sin(ray.x) * t;                  // where the light came from, on that side's sky
  // the sky's footprint per pixel (top level, every pixel of the quad runs it), limited across the disk's edge
  vec3 Jx = dFdx(d), Jy = dFdy(d);
  float lim = 40.0 * uPix;
  if (length(Jx) > lim) Jx *= lim / length(Jx);
  if (length(Jy) > lim) Jy *= lim / length(Jy);
  vec3 col;
  if (ray.y < -0.5) {
    vec4 sat = saturn(d);
    col = mix(milkyWay(d, Jx, Jy) + stars(d, Jx, Jy, 0.0, 1.0), sat.rgb, sat.a);
  } else if (ray.y > 0.5) {
    col = farSky(d) + stars(d, Jx, Jy, 50.0, 1.6);
  } else col = vec3(0.0);                                      // circled the throat until out of steps
  col = aces(col * uExposure);
  vec2 q = gl_FragCoord.xy / uRes - 0.5;
  col *= 1.0 - 0.45 * dot(q, q) * 2.0;                         // vignette
  col += (hash3(vec3(gl_FragCoord.xy, floor(uTime * 24.0))).x - 0.5) * 0.012;
  outColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}`;
