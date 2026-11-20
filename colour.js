// Blackbody colour: Planck's law integrated against the CIE 1931 colour-matching functions (analytic multi-lobe
// fit of Wyman, Sloan & Shirley 2013, JCGT 2(2)), converted to linear sRGB. So "4500 K" is a computed colour.
// Shared by the page (lookup table texture) and tests/gargantua.test.mjs.

const lobe = (x, mu, s1, s2) => { const t = (x - mu) / (x < mu ? s1 : s2); return Math.exp(-0.5 * t * t); };
export function cie1931(nm) {
  return [
    1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2),
    0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1),
    1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8),
  ];
}

/** Spectral radiance of a blackbody (arbitrary units), λ in nm. */
export const planck = (nm, T) => 1 / (Math.pow(nm, 5) * (Math.exp(1.4388e7 / (nm * T)) - 1));

/** CIE XYZ of a blackbody at temperature T (K), unnormalised (Y is its visible luminance). */
export function blackbodyXYZ(T) {
  let X = 0, Y = 0, Z = 0;
  for (let nm = 380; nm <= 780; nm += 2) {
    const b = planck(nm, T), [x, y, z] = cie1931(nm);
    X += b * x; Y += b * y; Z += b * z;
  }
  return [X, Y, Z];
}

/** XYZ -> linear sRGB (D65). */
export const xyzToRgb = ([X, Y, Z]) => [
  3.2406 * X - 1.5372 * Y - 0.4986 * Z,
  -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
  0.0557 * X - 0.2040 * Y + 1.0570 * Z,
];

/** Linear sRGB colour of a blackbody with luminance Y = 1 (out-of-gamut parts clipped to 0). */
export function blackbodyRGB(T) {
  const xyz = blackbodyXYZ(T);
  return xyzToRgb(xyz.map((v) => v / xyz[1])).map((v) => Math.max(v, 0));
}

export const LUT_MIN = 1000, LUT_MAX = 40000, LUT_SIZE = 256;
/** Lookup table for the shader: RGBA per entry, log-spaced in T. RGB = colour at Y = 1, A = log10(Y(T) / Y(6500 K)). */
export function blackbodyLUT() {
  const out = new Float32Array(LUT_SIZE * 4), Y65 = blackbodyXYZ(6500)[1];
  for (let i = 0; i < LUT_SIZE; i++) {
    const T = LUT_MIN * Math.pow(LUT_MAX / LUT_MIN, i / (LUT_SIZE - 1));
    const rgb = blackbodyRGB(T);
    out.set([rgb[0], rgb[1], rgb[2], Math.log10(blackbodyXYZ(T)[1] / Y65)], i * 4);
  }
  return out;
}
