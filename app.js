// Gargantua: the page. Runs the render passes in shader.js and builds the camera's frame from kerr.js every frame.
//
// Geodesic cache: tracing light is the expensive part, and it only depends on the camera. So rays are traced into
// a cache (where each ray hit the disk, where it ended on the sky) only when the camera moves, at reduced
// resolution while it moves. Then, once it stops, again at full resolution in horizontal bands over a few frames.
// Every frame only re-colours the cache (animated disk, stars, Milky Way, glow), so the picture is always sharp
// and smooth.
import * as K from './kerr.js';
import { blackbodyLUT, LUT_SIZE } from './colour.js';
import { VERTEX, geodesicSource, REFINE_MASK, RING_MASK, NOISE, shadeSource, BLOOM_DOWN, BLOOM_UP, COMPOSITE, TAA } from './shader.js';

document.getElementById('solarSwitch')?.addEventListener('click', (e) => { e.preventDefault(); window.PortfolioView?.switchTo('solar'); });

const canvas = document.getElementById('sim');
const hud = document.getElementById('hud');
const failBox = document.getElementById('simFail');
const tipBox = document.getElementById('simTip');
const fail = (msg) => { failBox.textContent = msg; failBox.hidden = false; throw new Error(msg); };

// The look. 'film' (stage 4, the default): the owner preferred the bright, punchy stage 1b look over the darker
// film-matched one (tuned against a frame of the film), so it is crisp's brightness and full star field with a
// slightly warmer, more saturated disk and only a faint lens glow (veiling glare). 'crisp' is the stage 1b look:
// ?look=crisp. Live tweaks: __gargantua.LOOK.
const LOOK_PRESETS = {
  film: {
    exposure: 1.1, bloom: 0.32, vignette: 0.45, grain: 0.018, milkyWay: 0.16, stars: 1.0, disk: 3.0,
    diskTemp: [6400, 7400], dust: 0.95, haze: 0.05, hazeTint: [1, 0.8, 0.6], falloff: 2.3, densPow: 1, saturation: 1.15, shiftCap: 4,
  },
  crisp: {
    exposure: 1.1, bloom: 0.3, vignette: 0.45, grain: 0.018, milkyWay: 0.16, stars: 1.0, disk: 3.0,
    diskTemp: [6800, 7500], dust: 0.95, haze: 0, hazeTint: [1, 0.8, 0.6], falloff: 2.3, densPow: 1, saturation: 1, shiftCap: 4,
  },
};
const LOOK = { ...(LOOK_PRESETS[new URLSearchParams(location.search).get('look')] || LOOK_PRESETS.film) };
// Two looks (stage 5). Film: the film's spin a/M = 0.6 and no Doppler shift on the disk, as the film chose (James
// et al. 2015; the story's time dilation needs a ≈ 1, judged too confusing a shadow for a mass audience). Reality:
// the true light shift g of disk and stars, and a spin slider (default 0.999, near the maximum).
// ?mode=reality, or ?spin=0…0.999 (Reality at that spin), opens in Reality.
const FILM_SPIN = 0.6;
const params = new URLSearchParams(location.search);
const spinParam = parseFloat(params.get('spin'));
const startReality = params.get('mode') === 'reality' || Number.isFinite(spinParam);
const state = {
  mode: startReality ? 'reality' : 'film',
  realitySpin: Number.isFinite(spinParam) ? Math.min(0.999, Math.max(0, spinParam)) : 0.999,
  spin: FILM_SPIN,
  cine: null,         // the Cinematic trip while it plays (see "Cinematic")
  pos: [90, 0, 0],    // camera position, Kerr–Schild x, y, z (M); see "camera and free flight"
  vel: [0, 0, 0],     // velocity relative to the local ZAMO, fraction of c, in its local x, y, z axes
  wheel: 0,           // speed added by the mouse wheel, dies away
  yaw: 0, pitch: 0,   // look direction in the same local axes (z = spin axis)
  r: 90,              // camera radius (from pos)
  fovY: 45 * Math.PI / 180,
  sky: 0,             // 0 real sky, 1 checkerboard
};
if (startReality) state.spin = state.realitySpin;
const R_MAX = 1000, R_START = 90, DISK_OUT = 26, INCL = 84 * Math.PI / 180;
// Gargantua is 10⁸ solar masses (Thorne, The Science of Interstellar): 1 M = GM/c² = 1.477 km × 10⁸ = 1.477e8 km
const KM_PER_M = 1.477e8, KM_PER_AU = 1.496e8;

// ---------- WebGL2 ----------
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: false });
if (!gl) fail('This simulation needs WebGL2. Try a recent Chrome, Edge, Firefox or Safari on a computer.');
if (!gl.getExtension('EXT_color_buffer_float')) fail('This simulation needs floating-point render targets (EXT_color_buffer_float), which this browser or GPU does not offer.');
const aniso = gl.getExtension('EXT_texture_filter_anisotropic');

// Hardware acceleration off = Chrome/Edge emulate the GPU on the processor (SwiftShader): show a tip, render lighter.
const dbg = gl.getExtension('WEBGL_debug_renderer_info');
const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
const software = /swiftshader|llvmpipe|softpipe|software|basic render/i.test(String(renderer));
// ?quality=light previews the light mode on any machine (it only lowers the resolution; for the real speed of a
// machine without GPU, turn off hardware acceleration in the browser instead)
const forceLight = new URLSearchParams(location.search).get('quality') === 'light';
const light = software || forceLight;
if (software) tipBox.hidden = false;
else if (forceLight) { tipBox.textContent = 'Light mode preview (?quality=light): how computers without graphics acceleration see this page.'; tipBox.hidden = false; }
// Phones and tablets (touch only, no mouse): a lighter setting for their GPUs (stage 6): 60% resolution when still, a
// lower moving resolution and device pixels capped at 1.5×. ?quality=phone previews it on a computer.
const phone = (matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches) || new URLSearchParams(location.search).get('quality') === 'phone';
const QUALITY = light ? { full: 0.4, lowMax: 0.25, lowMin: 0.12, refMin: 0.4, dpr: 1 }
  : phone ? { full: 0.6, lowMax: 0.45, lowMin: 0.2, refMin: 0.3, dpr: 1.5 }
  : { full: 1, lowMax: 0.7, lowMin: 0.3, refMin: 0.4, dpr: 2 };

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}
const vs = compile(gl.VERTEX_SHADER, VERTEX);
function program(fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.bindAttribLocation(p, 0, 'aPos');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
  return { p, u };
}
const P = {
  geo: program(geodesicSource(K.STEP)), geoRing: program(geodesicSource(K.STEP, 1)), geoBundle: program(geodesicSource(K.STEP, 2)), mask: program(REFINE_MASK), ringMask: program(RING_MASK), noise: program(NOISE), sky: program(shadeSource(1)), disk: program(shadeSource(2)),
  down: program(BLOOM_DOWN), up: program(BLOOM_UP), comp: program(COMPOSITE), taa: program(TAA),
};

// lite shading (weak GPUs, phones; see "automatic quality"): compiled on first use
const liteProgs = () => (P.skyLite ||= program(shadeSource(1, true)), P.diskLite ||= program(shadeSource(2, true)), { sky: P.skyLite, disk: P.diskLite });

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW); // one big triangle
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

function texture(w, h, internal, format, type, { filter = gl.LINEAR, wrapS = gl.CLAMP_TO_EDGE, wrapT = gl.CLAMP_TO_EDGE, data = null } = {}) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter === gl.NEAREST ? gl.NEAREST : gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrapS);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrapT);
  return t;
}
function framebuffer(texs, stencil = null) {
  const f = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
  if (stencil) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencil);
  gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete');
  return f;
}
function bindTex(unit, tex, prog, name) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(prog.u[name], unit);
}
const draw = () => gl.drawArrays(gl.TRIANGLES, 0, 3);

// ---------- static textures ----------
// Disk noise: tileable, made once on the GPU, mipmapped (+ anisotropic filtering: the texture is stretched along orbits;
// 4× looks the same as 16× here and halves the per-frame disk colouring on integrated GPUs, measured 20 → 9 ms)
const NOISE_SIZE = 1024;
const noiseTex = texture(NOISE_SIZE, NOISE_SIZE, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, { filter: gl.LINEAR_MIPMAP_LINEAR, wrapS: gl.REPEAT, wrapT: gl.REPEAT });
{
  const f = framebuffer([noiseTex]);
  gl.viewport(0, 0, NOISE_SIZE, NOISE_SIZE);
  gl.useProgram(P.noise.p);
  gl.uniform1f(P.noise.u.uSize, NOISE_SIZE);
  draw();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(f);
  gl.bindTexture(gl.TEXTURE_2D, noiseTex);
  gl.generateMipmap(gl.TEXTURE_2D);
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(4, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
}
const lutTex = texture(LUT_SIZE, 1, gl.RGBA16F, gl.RGBA, gl.FLOAT, { data: blackbodyLUT() });

// Milky Way: NASA SVS Deep Star Maps 2020 (Gaia DR2), galactic equirectangular; black until it loads
let mwTex = texture(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, { data: new Uint8Array([0, 0, 0, 255]) });
let mwVersion = 0; // bumps when the image arrives, so the cached sky colour is redone
const img = new Image();
img.onload = () => {
  const t = texture(img.width, img.height, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, { filter: gl.LINEAR_MIPMAP_LINEAR, wrapS: gl.REPEAT });
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.generateMipmap(gl.TEXTURE_2D);
  if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
  mwTex = t; mwVersion++;
};
img.src = new URL('./sky/milkyway.webp', import.meta.url).href; // (relative: the same file runs in the open-source copy)

// World → galactic frame: the galactic centre sits just behind and above the hole from the starting view, so its
// bright bulge is lensed around the shadow; the band crosses the sky at a tilt.
const galaxy = (() => {
  const norm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const C = norm([-1, 0.18, 0.28]);
  const tilt = 32 * Math.PI / 180;
  let N = norm(cross(cross(C, [0, 0, 1]), C));                 // world "up", made perpendicular to C
  const Y0 = cross(N, C);
  N = norm(N.map((x, i) => x * Math.cos(tilt) + Y0[i] * Math.sin(tilt)));
  const Y = cross(N, C);
  return new Float32Array([C[0], Y[0], N[0], C[1], Y[1], N[1], C[2], Y[2], N[2]]); // rows C, Y, N (column-major)
})();

// ---------- size-dependent buffers ----------
let out = null; // { w, h, hdr, hdrFbo, full, low, mips }
function cacheSet(w, h, withStencil = false) {
  const hits = texture(w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT, { filter: gl.NEAREST });
  const sky = texture(w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT, { filter: gl.NEAREST });
  const ring = texture(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, { filter: gl.NEAREST });
  let stencil = null;
  if (withStencil) {
    stencil = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, stencil);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, w, h);
  }
  const stencilFbo = stencil ? framebuffer([], stencil) : null; // stencil only: the ring mask reads ring
  return { w, h, hits, sky, ring, stencil, stencilFbo, fbo: framebuffer([hits, sky, ring], stencil), used: [w, h] };
}
function freeCache(c) {
  [c.hits, c.sky, c.ring].forEach((t) => gl.deleteTexture(t)); gl.deleteFramebuffer(c.fbo);
  if (c.stencil) { gl.deleteRenderbuffer(c.stencil); gl.deleteFramebuffer(c.stencilFbo); }
}
function buildBuffers(w, h) {
  if (out) {
    freeCache(out.full); freeCache(out.low);
    freeCache(out.ref);
    [out.hdr, out.prev, out.skyCol, ...out.hist].forEach((t) => gl.deleteTexture(t));
    [out.hdrFbo, out.prevFbo, out.skyFbo, ...out.histFbo].forEach((f) => gl.deleteFramebuffer(f));
    out.mips.forEach((m) => { gl.deleteTexture(m.tex); gl.deleteFramebuffer(m.fbo); });
  }
  const hdr = texture(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
  const mips = [];
  let mw = w, mh = h;
  while (mips.length < 7 && mw > 16 && mh > 16) {
    mw = Math.max(1, mw >> 1); mh = Math.max(1, mh >> 1);
    const tex = texture(mw, mh, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
    mips.push({ w: mw, h: mh, tex, fbo: framebuffer([tex]) });
  }
  const prev = texture(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);   // last frame before a cache switch (cross-fade)
  const hist = [0, 1].map(() => texture(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT));
  const skyCol = texture(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT); // cached sky colour (linear: upscaled while moving)
  out = {
    w, h, hdr, hdrFbo: framebuffer([hdr]), mips,
    prev, prevFbo: framebuffer([prev]), skyCol, skyFbo: framebuffer([skyCol]),
    hist, histFbo: hist.map((t) => framebuffer([t])), histIdx: 0, histValid: false, // TAA history (ping-pong)
    show: hdr, showFbo: null,                                                       // what post-processing reads
    ref: cacheSet(w, h, true), // edge refinement: edges + photon ring re-traced at screen resolution (+ stencil mask)
    full: cacheSet(Math.max(1, Math.round(w * QUALITY.full)), Math.max(1, Math.round(h * QUALITY.full))),
    low: cacheSet(Math.max(1, Math.round(w * QUALITY.lowMax)), Math.max(1, Math.round(h * QUALITY.lowMax))),
  };
}

// ---------- the fall (stage 3) ----------
// At the inner flight limit, keep pushing in and the engine lets go: from then on the camera is in free fall, its
// own timelike geodesic (kerr.js fallStep(), proper time), starting with the 4-velocity it had. Nothing is scripted:
// it crosses the horizon because the equations say so, and inside, every path leads to smaller r (r is a time
// there). Looking around still works; flying doesn't. It ends at the inner horizon (no spin: near the centre) with
// the note on where known physics stops. R starts again (the simulation resets; no real ship could leave).
// Gargantua is 10⁸ solar masses: 1 M of time = GM/c³ = 492 s.
const SEC_PER_M = 492;
const card = document.getElementById('simCard');
let cardTimer = 0;
function showCard(title, body, ms = 0) {
  card.innerHTML = '';
  const t = document.createElement('strong'); t.textContent = title; card.append(t);
  for (const [i, line] of body.entries()) { const e = document.createElement(i === body.length - 1 && body.length > 1 ? 'small' : 'span'); e.textContent = line; card.append(e); }
  card.classList.add('show');
  clearTimeout(cardTimer);
  if (ms) cardTimer = setTimeout(() => card.classList.remove('show'), ms);
}
const hideCard = () => { clearTimeout(cardTimer); card.classList.remove('show'); };
const clock = (tauM) => { const s = Math.round(tauM * SEC_PER_M), m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min ${String(s % 60).padStart(2, '0')} s`; };

// pace: proper time (M) per real second; slower near the end so the last moments can be watched
function fallRate(r) {
  const rM = K.innerHorizon(state.spin), end = fallEnd();
  return 0.06 + 0.22 * Math.min(1, Math.max(0, (r - end) / (K.outerHorizon(state.spin) - rM + 0.5)));
}
const fallEnd = () => { const rM = K.innerHorizon(state.spin); return rM > 0.01 ? rM * 1.04 + 0.01 : 0.12; };

function startFall() {
  const a = state.spin, { fwd, right, up } = lookAxes();
  const T = K.cameraTetrad(state.pos, state.vel, right, up, fwd, a), m = K.metricAt(...state.pos, a);
  const pl = K.lower(m, T.e0);
  const sp = (v) => norm([v[1], v[2], v[3]]);                // the camera's axes as Kerr–Schild directions
  state.fall = {
    s: [...state.pos, pl[1], pl[2], pl[3]], pt: pl[0], tau: 0, inside: false, ended: false, t0: performance.now(),
    seed: { F: sp(T.fwd), R: sp(T.right), U: sp(T.up) }, yaw0: state.yaw, pitch0: state.pitch,
  };
  state.vel = [0, 0, 0]; held.clear();
  showCard('Engines off', ['You are falling freely now. Past the horizon, nothing could stop it. The view turns to look back at where you came from.',
    'Look around yourself with the mouse or ← / →. R starts again.'], 6500);
}

// Ahead (into the hole) there is no light from outside: that direction is the hole's own past (black). The sight is
// behind: the universe you left. Measured with kerr.js (a = 0.6, falling from rest at 2.9 M): straight back it is
// redshifted (g = 0.9 at 2 M, 0.4 near the inner horizon: you are racing away from the light falling in after you),
// and around the edge of that view, 60–130° from straight back, blueshifted 2–8× into a bright rim (as in
// Hamilton's black hole flight simulator). So the view turns around by itself (from 3.5 s, over 8 s) and widens
// from 45° to 90° to take in the rim, until you look around yourself.
// turn: 120°, not straight back: the view then holds the edge, the universe you left with its bright blueshifted
// rim (60–130° from straight back) on one side, the darkness ahead on the other. (Straight back framed only the
// dim, redshifted middle.)
const LOOK_BACK = { delay: 3500, ms: 8000, turn: 2.1 };   // starts once the "engines off" note has been read
let lookTouched = 0;
function fallTick(dt) {
  const f = state.fall, a = state.spin;
  if (held.has('turnL')) { state.yaw += 0.8 * dt / 1000; lookTouched = performance.now(); }
  if (held.has('turnR')) { state.yaw -= 0.8 * dt / 1000; lookTouched = performance.now(); }
  if (lookTouched < f.t0) {
    const t = Math.min(1, Math.max(0, (performance.now() - f.t0 - LOOK_BACK.delay) / LOOK_BACK.ms));
    state.yaw = f.yaw0 + LOOK_BACK.turn * t * t * (3 - 2 * t);
  }
  const tf = Math.min(1, Math.max(0, (performance.now() - f.t0 - LOOK_BACK.delay) / LOOK_BACK.ms));
  state.fovY = (45 + 45 * tf * tf * (3 - 2 * tf)) * Math.PI / 180;
  if (f.ended) return;
  const dtau = fallRate(state.r) * dt / 1000;
  const next = K.fallStep(f, a, dtau);
  f.s = next.s; f.tau += dtau;
  state.pos = f.s.slice(0, 3); state.r = K.radius(...state.pos, a);
  if (!f.inside && state.r < K.outerHorizon(a)) {
    f.inside = true;
    showCard('The event horizon', ['Past this point light can’t get out, and neither can you. Light from outside still falls in after you.',
      'R restarts the simulation. No real ship could leave.'], 7000);
  }
  if (state.r <= fallEnd()) {
    f.ended = true;
    const spin = a > 0.01;
    showCard(spin ? 'The inner horizon: where known physics stops' : 'The centre: where known physics stops', spin
      ? ['In the ideal solution space goes on past here, to a ring singularity and even other universes. A real black hole is different: near this horizon, matter and light that fell in at different times stream through each other at almost the speed of light, and their energy grows without limit ("mass inflation", Poisson & Israel 1990; Hamilton & Avelino 2010). What happens next needs a theory of quantum gravity, which nobody has yet.',
        `Your clock: ${clock(f.tau)} since the engines stopped. Press R to start again.`]
      : ['Without spin there is no inner horizon: every path ends at the centre, where the equations give infinite density and tides tear anything apart. What really happens there needs a theory of quantum gravity, which nobody has yet.',
        `Your clock: ${clock(f.tau)} since the engines stopped. Press R to start again.`]);
  }
}

// the falling camera's frame: its 4-velocity from the geodesic, its axes the ones it had when the engine stopped,
// turned by how far you have looked around since (metric Gram–Schmidt makes them orthonormal again)
function fallFrame() {
  const f = state.fall, a = state.spin, m = K.metricAt(...state.pos, a);
  const u = K.raise(m, [f.pt, f.s[3], f.s[4], f.s[5]]);
  const dy = state.yaw - f.yaw0, dp = state.pitch - f.pitch0, { F, R, U } = f.seed;
  const fwd = norm([0, 1, 2].map((i) => Math.cos(dp) * (Math.cos(dy) * F[i] - Math.sin(dy) * R[i]) + Math.sin(dp) * U[i]));
  const right = norm(cross(fwd, U)), up = cross(right, fwd);
  const T = K.tetrad(state.pos, u, fwd, right, up, a);
  return { pos: state.pos, low: { e0: K.lower(m, T.e0), right: K.lower(m, T.right), up: K.lower(m, T.up), fwd: K.lower(m, T.fwd) } };
}

// ---------- camera and free flight ----------
// The camera moves relative to the local non-rotating observer (ZAMO, kerr.js zamo()), as the film's did. Its
// state: a position (Kerr–Schild x, y, z), a look direction (yaw, pitch) and a velocity (fraction of c), both in
// the ZAMO's local x, y, z axes (z = the spin axis). The rays start from the camera's own frame, the ZAMO frame
// Lorentz-boosted by that velocity (kerr.js cameraTetrad()), so speed really bends what you see (aberration).
// The "magic engine": the camera may hover and fly anywhere outside the horizon at the speed you choose, and time
// is compressed in proportion to the distance (TIME_SCALE), so crossing 1000 M takes seconds, not days.
const norm = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const TIME_SCALE = 2.2;              // coordinate time (M) per real second, per M of distance
const SPEED = { cruise: 0.3, fast: 0.8, max: 0.95 }; // fraction of c (Shift = fast)
const rMin = () => 1.6 * K.outerHorizon(state.spin); // stage 3 opens the horizon; until then keep out of it
const lookDir = () => {
  const cp = Math.cos(state.pitch);
  return [cp * Math.cos(state.yaw), cp * Math.sin(state.yaw), Math.sin(state.pitch)];
};
// "Away from the hole" in the ZAMO's local axes (kerr.js zamoAxes: tied to Boyer–Lindquist angles). Not
// norm(pos): near the hole the Kerr–Schild azimuth is twisted against those axes (19° at the inner flight limit,
// a = 0.6), and steering judged by it pushed the camera sideways and off the limit.
// Boyer–Lindquist r, θ, φ of a Kerr–Schild point, and back (x + iy = (r + ia) sinθ e^{iφ_KS}, z = r cosθ,
// φ_KS = φ_BL + twist(r))
function blOf(pos) {
  const a = state.spin, r = K.radius(pos[0], pos[1], pos[2], a);
  return { r, th: Math.acos(Math.max(-1, Math.min(1, pos[2] / r))), ph: Math.atan2(pos[1], pos[0]) - Math.atan2(a, r) - K.twist(r, a) };
}
function ksOf(r, th, ph) {
  const a = state.spin, phK = ph + K.twist(r, a), st = Math.sin(th);
  return [st * (r * Math.cos(phK) - a * Math.sin(phK)), st * (r * Math.sin(phK) + a * Math.cos(phK)), r * Math.cos(th)];
}
function localRadial(pos) {
  const a = state.spin, r = K.radius(pos[0], pos[1], pos[2], a), c = pos[2] / r, sn = Math.sqrt(Math.max(0, 1 - c * c));
  const phi = Math.atan2(pos[1], pos[0]) - Math.atan2(a, r) - K.twist(r, a);
  return [sn * Math.cos(phi), sn * Math.sin(phi), c];
}
function lookAxes() {
  const fwd = lookDir();
  let right = cross(fwd, [0, 0, 1]);
  right = Math.hypot(...right) < 1e-6 ? [0, -1, 0] : norm(right);
  return { fwd, right, up: cross(right, fwd) };
}
// place the camera at distance r on the starting line of sight (inclination 84°, just above the disk, like the
// film), looking at the hole, at rest
function placeAt(r) {
  const a = state.spin, rr = Math.sqrt(r * r + a * a);
  state.pos = [rr * Math.sin(INCL), 0, r * Math.cos(INCL)];
  const f = localRadial(state.pos).map((x) => -x);
  state.yaw = Math.atan2(f[1], f[0]); state.pitch = Math.asin(f[2]);
  state.vel = [0, 0, 0]; state.wheel = 0;
  state.r = r;
  state.fall = null; pushT = 0; hinted = false; hideCard(); state.fovY = 45 * Math.PI / 180;
}
let pushT = 0, hinted = false;   // how long you've pushed in at the inner limit; whether the hint was shown
const ENGINE_OFF_MS = 2000;
placeAt(R_START);

let camInfo = { gamma: 1, alpha: 1, speed: 0 };
function cameraFrame() {
  if (state.fall) return fallFrame();
  const a = state.spin, { fwd, right, up } = lookAxes();
  const T = K.cameraTetrad(state.pos, state.vel, right, up, fwd, a);
  camInfo = { gamma: T.gamma, alpha: T.alpha, speed: Math.hypot(...state.vel), e0: T.e0 };
  const m = K.metricAt(state.pos[0], state.pos[1], state.pos[2], a);
  return { pos: state.pos, low: { e0: K.lower(m, T.e0), right: K.lower(m, T.right), up: K.lower(m, T.up), fwd: K.lower(m, T.fwd) } };
}

// ---------- input ----------
let dragging = false, lastX = 0, lastY = 0, lastDrag = 0;
canvas.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const k = state.fovY / canvas.clientHeight;
  state.yaw -= (e.clientX - lastX) * k;
  lastDrag = performance.now();
  lookTouched = performance.now();
  state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + (e.clientY - lastY) * k));
  lastX = e.clientX; lastY = e.clientY;
});
// the wheel flies forward / back: each notch adds speed along the view that dies away in ~0.3 s
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (state.cine) { scrub(e.deltaY > 0 ? -4 : 4); return; }   // Cinematic: the wheel scrubs along the trip
  state.wheel = Math.max(-SPEED.max, Math.min(SPEED.max, state.wheel - e.deltaY * 0.0025));
}, { passive: false });
// Keyboard (free flight): W / S or ↑ / ↓ forward / back, A / D strafe, Q / E down / up, ← / → turn, Shift faster.
// (No Ctrl for "slower", as first planned: Ctrl+W closes the tab and Ctrl+S opens a save dialog.)
// Held keys act every frame, scaled by the frame time. Arrows don't scroll the page while the simulation is in view.
const held = new Set();
const MOVE_KEYS = {
  w: 'fwd', arrowup: 'fwd', s: 'back', arrowdown: 'back', a: 'left', d: 'right', q: 'down', e: 'up',
  arrowleft: 'turnL', arrowright: 'turnR', shift: 'fast',
};
const simInView = () => window.scrollY < canvas.clientHeight * 0.5;
window.addEventListener('keydown', (e) => {
  if (e.target.closest && e.target.closest('input, textarea, select')) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest && e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  const k = e.key.toLowerCase();
  if (k === 'escape' && ui.start.hidden) { backToStart(); return; }
  if (!ui.start.hidden) { if (MOVE_KEYS[k] && k !== 'shift') choose(false); else return; } // a flight key: fly yourself
  if (k === 'c') setSky(1 - state.sky);
  if (k === 'm') toggleSound();
  if (k === 'r') { if (state.cine) startTrip(); else placeAt(R_START); }
  if (state.cine && k === ' ' && simInView()) { e.preventDefault(); state.cine.paused = !state.cine.paused; return; }
  if (MOVE_KEYS[k] && simInView()) { held.add(MOVE_KEYS[k]); if (k !== 'shift') e.preventDefault(); }
});
window.addEventListener('keyup', (e) => { const m = MOVE_KEYS[e.key.toLowerCase()]; if (m) held.delete(m); });
window.addEventListener('blur', () => held.clear());

// one frame of flight: steer the velocity towards what the keys ask for, then move
function fly(dt) {
  const s = dt / 1000, { fwd, right, up } = lookAxes(), held0 = { pos: state.pos, r: state.r };
  if (held.has('turnL')) state.yaw += 0.8 * s;
  if (held.has('turnR')) state.yaw -= 0.8 * s;
  const f = (held.has('fwd') ? 1 : 0) - (held.has('back') ? 1 : 0);
  const st = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
  const u = (held.has('up') ? 1 : 0) - (held.has('down') ? 1 : 0);
  let cmd = [0, 1, 2].map((i) => f * fwd[i] + st * right[i] + u * up[i]);
  const n = Math.hypot(...cmd), vmax = held.has('fast') ? SPEED.fast : SPEED.cruise;
  cmd = n > 0 ? cmd.map((c) => (c / n) * vmax) : [0, 0, 0];
  state.wheel *= Math.exp(-dt / 300);
  cmd = cmd.map((c, i) => c + state.wheel * fwd[i]);
  const cn = Math.hypot(...cmd);
  if (cn > SPEED.max) cmd = cmd.map((c) => (c * SPEED.max) / cn);
  const k = 1 - Math.exp(-dt / 350);                          // the engine reaches a new speed in ~1 s
  state.vel = state.vel.map((v, i) => v + (cmd[i] - v) * k);
  if (Math.hypot(...state.vel) < 1e-4 && n === 0 && Math.abs(state.wheel) < 1e-4) { state.vel = [0, 0, 0]; return; }
  // Move. The engine holds the ship against the hole's frame dragging: "at rest" means at rest relative to the
  // distant stars, and straight means straight. The local observers (ZAMOs) the velocity is measured against are
  // themselves carried round the hole (dφ/dt = ω), and with time compressed TIME_SCALE·r-fold that swept a ship
  // flying straight at the hole round and past it (22° off by the inner limit at a = 0.6, 58° at 0.999; owner
  // 2026-11-13: "it tilts my camera and moves me sideways"). (A real ship could do the same with thrust outside the
  // ergosphere; inside it, at fast spin, this is the magic part of the magic engine.)
  // So the ship moves in Boyer–Lindquist r, θ, φ (the angles the ZAMO's axes are tied to), by its velocity along the
  // ZAMO's e_r, e_θ, e_φ without the ω term: dr = α v_r √(Δ/Σ) dT, dθ = α v_θ / √Σ dT, dφ = α v_φ / ϖ dT, with
  // ϖ = sinθ √(A/Σ), A = (r² + a²)² − a²Δ sin²θ. The inner and outer limits are then limits on r.
  const a = state.spin, dT = TIME_SCALE * state.r * s;
  let { r, th, ph } = blOf(state.pos);
  const sT = Math.sin(th), cT = Math.cos(th), sP = Math.sin(ph), cP = Math.cos(ph), v = state.vel;
  const vr = v[0] * sT * cP + v[1] * sT * sP + v[2] * cT;
  const vt = v[0] * cT * cP + v[1] * cT * sP - v[2] * sT;
  const vp = -v[0] * sP + v[1] * cP;
  const D = r * r - 2 * r + a * a, S = r * r + a * a * cT * cT, A = (r * r + a * a) ** 2 - a * a * D * sT * sT;
  const alpha = Math.sqrt(D * S / A), varpi = sT * Math.sqrt(A / S);
  r += alpha * vr * Math.sqrt(D / S) * dT;
  th = Math.min(Math.PI - 1e-3, Math.max(1e-3, th + alpha * vt / Math.sqrt(S) * dT));
  ph += alpha * vp / varpi * dT;
  const lo = rMin(), hi = R_MAX;
  const atLimit = r < lo;
  if (r < lo || r > hi) {
    // at a limit, drop the velocity's part across it
    if ((r < lo && vr < 0) || (r > hi && vr > 0)) { const rad = localRadial(state.pos); state.vel = v.map((c, i) => c - vr * rad[i]); }
    r = Math.min(hi, Math.max(lo, r));
  }
  const pos = ksOf(r, th, ph);
  state.pos = pos; state.r = r;
  // At the inner limit the engine holds you. Pushing on for 2 s switches it off (the fall), with a countdown in the
  // note; letting go cancels. (First 0.7 s: holding W all the way in, the note showed for under a second before the
  // fall began, so the turn-around came with no warning, owner 2026-11-13.)
  if (r > lo * 1.3) hinted = false;
  // "pushing in": within 3% of the limit and steering inward (at the limit the engine drops the inward part of the
  // velocity and the camera slides along it, a hair above it now and then, so touching it exactly isn't enough)
  const radialDir = localRadial(state.pos), inward = cmd[0] * radialDir[0] + cmd[1] * radialDir[1] + cmd[2] * radialDir[2] < -0.02;
  const wasPushing = pushT > 0;
  pushT = r < lo * 1.03 && inward ? pushT + dt : 0;
  // While you push in at the limit the engine holds you exactly where you are. Else the camera slid along the
  // limit until the push no longer pointed inward and the countdown reset: frame dragging sweeps a ship flying
  // "straight" in round the hole, so it arrives with the hole up to ~60° off its view (measured, a = 0.6).
  if (pushT > 0) { state.vel = [0, 0, 0]; state.pos = held0.pos; state.r = held0.r; }
  if (pushT > 0) {
    const left = Math.max(0, (ENGINE_OFF_MS - pushT) / 1000);
    showCard('The closest the engine can hover', [`Keep pushing in to switch it off and fall into the hole: ${left.toFixed(1)} s`,
      'Let go to stay outside.']);
    hinted = true;
  } else if (wasPushing) hideCard();
  else if (atLimit && !hinted) {
    hinted = true;
    showCard('The closest the engine can hover', ['Keep pushing in to switch it off and fall into the hole.'], 4500);
  }
  if (pushT > ENGINE_OFF_MS) startFall();
}

// ---------- start screen, sound, grid sky ----------
// The visitor picks once on the start screen: take the trip or fly yourself; refreshing the page is how to choose
// again (owner, 2026-11-14: no mode menu, a clean screen). Film look; Reality (the disk's true Doppler shift, a spin
// to 0.999) stays reachable for checks with ?mode=reality or ?spin=… and from the console (__gargantua.setMode).
const ui = {
  start: document.getElementById('simStart'), keys: document.getElementById('simKeys'), mute: document.getElementById('simMute'),
  touch: document.getElementById('simTouch'), pause: document.getElementById('touchPause'),
  fly: document.querySelectorAll('#simTouch [data-hold]'),
};
// Touch screens: drag to look (pointer events, as with a mouse), on-screen buttons for the rest: hold ▲ / ▼ to fly
// forward / back, pause (trip), back to the start screen (= Esc). Shown only after the choice, on touch devices.
const KEYS_TOUCH_FREE = 'drag to look · hold ▲ / ▼ to fly';
const KEYS_TOUCH_TRIP = 'drag to look';
ui.fly.forEach((b) => {
  const on = (e) => { e.preventDefault(); held.add(b.dataset.hold); b.setPointerCapture?.(e.pointerId); };
  const off = () => held.delete(b.dataset.hold);
  b.addEventListener('pointerdown', on);
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach((ev) => b.addEventListener(ev, off));
  b.addEventListener('contextmenu', (e) => e.preventDefault());
});
ui.pause.addEventListener('click', () => { if (state.cine) state.cine.paused = !state.cine.paused; syncUI(); });
document.getElementById('touchBack').addEventListener('click', () => backToStart());
const KEYS_FREE = 'W / S or ↑ / ↓ fly · A / D strafe · Q / E down / up · Shift faster · drag or ← / → to look · wheel to fly · C grid sky · M sound · R reset · Esc back';
const KEYS_TRIP = 'Space pause · ← / → or wheel rewind / skip · drag to look · C grid sky · M sound · R restart · Esc back';
function syncUI() {
  const trip = state.cine && !state.fall;
  const keys = phone ? (trip ? KEYS_TOUCH_TRIP : state.fall ? '' : KEYS_TOUCH_FREE) : trip ? KEYS_TRIP : KEYS_FREE;
  if (ui.keys.textContent !== keys) ui.keys.textContent = keys;
  ui.touch.hidden = !phone || !ui.start.hidden;
  ui.fly.forEach((b) => { b.hidden = !!state.cine || !!state.fall; });
  ui.pause.hidden = !trip;
  const paused = !!state.cine?.paused;
  if (ui.pause.dataset.paused !== String(paused)) {
    ui.pause.dataset.paused = String(paused);
    ui.pause.setAttribute('aria-label', paused ? 'Play' : 'Pause');
    ui.pause.innerHTML = `<i class="bi ${paused ? 'bi-play-fill' : 'bi-pause-fill'}" aria-hidden="true"></i>`;
  }
}
// a new spin keeps the camera at the same Boyer–Lindquist r, θ, φ (and outside the new inner flight limit)
function setSpin(a) {
  if (state.fall || a === state.spin) return;
  const { r, th, ph } = blOf(state.pos);
  state.spin = a;
  const r2 = Math.min(R_MAX, Math.max(rMin(), r));
  state.pos = ksOf(r2, th, ph); state.r = r2;
  if (state.cine) state.cine.trip = buildTrip();
}
function setMode(mode) {
  if (state.fall) return;
  state.mode = mode;
  setSpin(mode === 'reality' ? state.realitySpin : FILM_SPIN);
}
function setSky(v) {
  state.sky = v;
  if (v) showCard('The grid sky', ['The stars are swapped for a grid of 10° squares all around you. Every line of it is straight: each bend you see is the hole bending light, traced ray by ray through the Kerr spacetime.',
    'C again for the real sky.'], 9000);
  else hideCard();
}

// Sound: NASA's sonification of a black hole (~20 s), looped without a seam: each copy fades in over the last
// seconds of the one before (Web Audio; an <audio loop> leaves a gap and a click at the joint). Browsers only allow
// sound after a click or key press, so it starts with the start-screen choice. M (or the button) mutes it.
const sound = { ctx: null, gain: null, buf: null, on: true, next: 0, timer: 0 };
const SOUND_VOL = 0.8, XFADE = 2.5;
// Louder the closer you are: full volume at 4 M and inside, falling with log(distance) to 0.1 of it at 300 M and beyond
// (the start, at 90 M, is ~0.3). Followed smoothly (setTargetAtTime), so flying in swells the sound.
const nearness = (r) => Math.min(1, Math.max(0.1, 1 - Math.log(Math.max(r, 4) / 4) / Math.log(300 / 4)));
let lastVolT = 0;
async function startSound() {
  if (sound.ctx) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    sound.ctx = new Ctx();
    sound.gain = sound.ctx.createGain();
    sound.gain.gain.value = 0;
    sound.gain.connect(sound.ctx.destination);
    const data = await (await fetch(new URL('./sound/blackhole.mp3', import.meta.url))).arrayBuffer();
    sound.buf = await sound.ctx.decodeAudioData(data);
    sound.next = sound.ctx.currentTime + 0.05;
    const schedule = () => {
      const { ctx, buf } = sound;
      while (sound.next < ctx.currentTime + 4) {
        const src = ctx.createBufferSource(), g = ctx.createGain(), t = sound.next, d = buf.duration;
        src.buffer = buf; src.connect(g); g.connect(sound.gain);
        g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(1, t + XFADE);
        g.gain.setValueAtTime(1, t + d - XFADE); g.gain.linearRampToValueAtTime(0, t + d);
        src.start(t); src.stop(t + d);
        sound.next = t + d - XFADE;
      }
    };
    schedule();
    sound.timer = setInterval(schedule, 1000);
    setVolume();
  } catch (err) { console.warn('sound unavailable', err); }
}
const setGain = () => { if (sound.gain) sound.gain.gain.setTargetAtTime(sound.on ? SOUND_VOL * nearness(state.r) : 0, sound.ctx.currentTime, 0.6); };
function setVolume() {
  setGain();
  ui.mute.setAttribute('aria-pressed', String(!sound.on));
  ui.mute.setAttribute('aria-label', sound.on ? 'Mute sound' : 'Turn sound on');
  ui.mute.innerHTML = `<i class="bi ${sound.on ? 'bi-volume-up' : 'bi-volume-mute'}" aria-hidden="true"></i>`;
}
function toggleSound() { sound.on = !sound.on; if (sound.on) startSound(); setVolume(); }
ui.mute.addEventListener('click', () => { toggleSound(); ui.mute.blur(); });
document.addEventListener('visibilitychange', () => { if (sound.ctx) (document.hidden ? sound.ctx.suspend() : sound.ctx.resume()); });
// start screen: take the trip or fly yourself (the picture already renders behind it)
function choose(trip) {
  if (ui.start.hidden) return;
  ui.start.hidden = true; ui.mute.hidden = false;
  startSound();
  if (trip) startTrip(); else placeAt(R_START);
  syncUI();
}
// Esc, from either mode: back to the start screen to choose again (the sound keeps playing)
function backToStart() {
  state.cine = null; held.clear(); placeAt(R_START);
  ui.start.hidden = false; syncUI();
}
document.querySelectorAll('[data-start]').forEach((b) => b.addEventListener('click', () => choose(b.dataset.start === 'trip')));

// ---------- Cinematic ----------
// A fixed trip from far away down to the closest the engine can hover, then the engines go off and the real fall
// (the same free fall as in free flight) takes over: through the horizon to the inner horizon and the ending note.
// The path: keyframes in Boyer–Lindquist r (log), θ, φ and a look offset from "at the hole", joined by a Hermite
// curve in time (Catmull–Rom tangents). The camera moves along it under the same "magic engine" rules as free flight
// (time compressed TIME_SCALE·r-fold, motion in r, θ, φ without the frame dragging): its velocity relative to the
// local observer comes from the path and stays far below light, so aberration is as real as when you fly.
// The inner stops depend on the spin: the photon shell, the last stable orbit and Miller's planet move with it, and
// those below the closest the engine can hover are explained from just above.
const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
const fmt = (x) => x.toFixed(2);
function buildTrip() {
  const a = state.spin, lo = rMin(), isc = K.isco(a), pPro = K.photonOrbit(a), pRet = K.photonOrbitRetro(a);
  // the stops (pose + note); timed below
  const keys = [
    { r: 90, th: 84, ph: 0, dy: 0, dp: 0 },
    { r: 48, th: 85, ph: 4, dy: 0, dp: 0 },
    { r: 28, th: 87, ph: 10, dy: 0, dp: 0 },
    { r: 18, th: 84, ph: 24, dy: 0.15, dp: 0.03 },
    { r: 12, th: 76, ph: 60, dp: 0.05 },
    { r: 8.5, th: 72, ph: 95, dp: 0.04 },
  ];
  const rMiller = Math.max(isc * 1.25, lo * 1.25), ut = K.orbitUt(Math.max(rMiller, K.isco(a) * 1.001), a);
  const stops = [
    { r: rMiller, title: 'Miller’s planet', body: () => [
      'In the film it orbits about here, where 1 hour is 7 years on Earth. That takes a spin within a hair of the maximum (0.99999999999999, Thorne, The Science of Interstellar).',
      `At this spin (${a.toFixed(3)}), 1 hour on an orbit here is ${earthTime(ut)} on Earth.`] },
    { r: Math.max(isc * 1.03, lo * 1.1), title: 'The last stable orbit', body: () => [isc > lo * 1.1
      ? `At ${fmt(isc)} M. Closer in, gas can’t stay in orbit and spirals in: this is where the disk ends.`
      : `It is at ${fmt(isc)} M, closer than the engine can hover. Spin lets gas orbit nearer the hole, so the disk reaches further in (6 M without spin).`] },
    { r: pRet > lo * 1.1 ? Math.max(lo * 1.05, pRet * 0.96) : lo * 1.05, title: 'The photon shell', body: () => [
      `${pRet > lo * 1.1 ? 'Between' : 'Just below you, between'} ${fmt(pPro)} and ${fmt(pRet)} M, light itself can circle the hole. The thin bright ring hugging the shadow is light that went round it once or more on its way to you.`] },
  ].sort((x, y) => y.r - x.r);
  let r = 8.5, ph = 95;
  for (const st of stops) {
    r = Math.max(lo * 1.03, Math.min(st.r, r / 1.06)); ph += 20;
    keys.push({ r, th: 80, ph, dp: 0.03, note: st });
  }
  keys.push({ r: lo * 1.01, th: 84, ph: ph + 15, dp: 0 });
  // Close in, the shadow fills most of the sky (from 4 M over 140° of it): looking at the hole shows only black. These
  // keys look at the shadow's edge instead, where the disk and the photon ring are, with a wider view. Edge: the
  // critical impact parameter b_c (mean of the two equatorial ones, Bardeen 1973) seen by the local observer,
  // sin ψ = b_c α / √(r² + a²) (exact at a = 0: sin ψ = √27 √(1 − 2/r) / r).
  const bc = 3 * (Math.cos(Math.acos(-a) / 3) + Math.cos(Math.acos(a) / 3));
  for (const k of keys) {
    k.fov = 45 + 20 * Math.min(1, Math.max(0, (9 - k.r) / 5));
    if (k.dy !== undefined) continue;
    const R2 = k.r * k.r + a * a, alpha = Math.sqrt(Math.max(0, (k.r * k.r - 2 * k.r + a * a) / R2));
    const psi = Math.asin(Math.min(0.999, bc * alpha / Math.sqrt(R2)));
    k.dy = Math.min(1.25, 0.85 * psi + 0.35 * (k.fov - 45) * Math.PI / 180);
  }
  // Timing: glide from stop to stop (TRIP.move s, easing in and out), then hold still at each (TRIP.hold s) while its
  // note shows. Holding still is what lets the picture sharpen: while the camera moves, every frame is the
  // low-resolution moving picture (owner, 2026-11-14: "stays blurry all the way" when it never stopped).
  const timed = [];
  let t = 0;
  keys.forEach((k, i) => {
    if (i > 0) t += TRIP.move;
    timed.push({ ...k, t });
    if (i > 0 && i < keys.length - 1) { k.at = t; t += TRIP.hold; timed.push({ ...k, t }); }
  });
  const end = t;
  const doppler = () => (state.mode === 'reality'
    ? ['The gas coming towards you looks bluer and far brighter, the side moving away dimmer and redder (Doppler beaming). Near the hole its clock runs slow too: gravity reddens all of it.',
      'This is Reality mode. Film shows the disk as the film did.']
    : ['In the film, and here, both sides of the disk look alike. In reality the side coming towards you would blaze and the other fade (Doppler shift); the film’s makers left that out so audiences wouldn’t be confused (James et al. 2015).']);
  const farNotes = [
    { title: 'The sky starts to bend', body: () => ['Gravity bends the light of the stars behind the hole around it: near its edge they are pushed outward and seen twice.'] },
    { title: 'The Einstein ring', body: () => ['Light from straight behind the hole is bent the same way on every side and arrives as a ring around the shadow.', 'Press C for a grid sky that shows the bending.'] },
    { title: 'The far side of the disk', body: () => ['The disk is flat. The part behind the hole shows above and below it: its light curves over the top and under the bottom to reach you.'] },
    { title: 'Why both sides look the same', body: doppler },
    { title: 'The shadow', body: () => ['The black disk is the hole’s shadow, about two and a half times as wide as its horizon: light that comes closer than its edge falls in.'] },
  ];
  // far stops 1-5 carry the far notes, the inner ones their own; each shows as the camera arrives
  const notes = [];
  keys.forEach((k, i) => { const n = k.note || farNotes[i - 1]; if (n && k.at !== undefined) notes.push({ ...n, t: k.at + 0.2 }); });
  return { keys: timed, notes, end };
}
const TRIP = { move: 6, hold: 3.5 };  // 8 stops: ~1:22 of path + the fall ≈ 1:47 (owner: under 2 minutes)
// position on the path at time t: Hermite curve through the keys with zero tangents (each leg eases in and out)
function tripPose(trip, t) {
  const k = trip.keys, n = k.length;
  t = Math.min(Math.max(t, 0), trip.end);
  let i = 0;
  while (i < n - 2 && t > k[i + 1].t) i++;
  const val = (j, f) => (f === 'r' ? Math.log(k[j].r) : k[j][f]);
  const h = k[i + 1].t - k[i].t, u = (t - k[i].t) / h, w = u * u * (3 - 2 * u);
  const H = (f) => val(i, f) + (val(i + 1, f) - val(i, f)) * w;
  const D = Math.PI / 180;
  return { r: Math.max(rMin(), Math.exp(H('r'))), th: H('th') * D, ph: H('ph') * D, dy: H('dy'), dp: H('dp'), fov: H('fov') };
}
// the velocity (fraction of c, local axes) that moves the camera from pose p to q in dts seconds (fly()'s rules, inverted)
function tripVelocity(p, q, dts) {
  const a = state.spin, { r, th } = p, dT = TIME_SCALE * r * dts;
  const sT = Math.sin(th), cT = Math.cos(th), sP = Math.sin(p.ph), cP = Math.cos(p.ph);
  const D = r * r - 2 * r + a * a, S = r * r + a * a * cT * cT, A = (r * r + a * a) ** 2 - a * a * D * sT * sT;
  const alpha = Math.sqrt(D * S / A), varpi = sT * Math.sqrt(A / S);
  const vr = (q.r - r) / (alpha * Math.sqrt(D / S) * dT), vt = (q.th - th) * Math.sqrt(S) / (alpha * dT), vp = (q.ph - p.ph) * varpi / (alpha * dT);
  let v = [0, 1, 2].map((i) => vr * [sT * cP, sT * sP, cT][i] + vt * [cT * cP, cT * sP, -sT][i] + vp * [-sP, cP, 0][i]);
  const n = Math.hypot(...v);
  if (n > 0.9) v = v.map((c) => (c * 0.9) / n);
  return v;
}
function startTrip() {
  placeAt(R_START);
  ui.start.hidden = true;
  state.cine = { t: 0, paused: false, trip: buildTrip(), offY: 0, offP: 0, yawSet: state.yaw, pitchSet: state.pitch };
  syncUI();
}
function stopTrip() { state.cine = null; syncUI(); }
function scrub(ds) {
  const c = state.cine;
  if (state.fall) {
    if (ds > 0) return;                                        // the fall can't be skipped through, only left
    placeAt(R_START);                                          // (resets the fall, the view and the cards)
    c.t = Math.max(0, c.trip.end + ds);
    c.yawSet = state.yaw; c.pitchSet = state.pitch; c.offY = 0; c.offP = 0;
  } else c.t = Math.min(c.trip.end, Math.max(0, c.t + ds));
}
function cineTick(dt) {
  const c = state.cine, s = dt / 1000;
  if (state.fall) { if (!c.paused) c.t += s; return; }          // the fall runs itself (fallTick)
  const ff = (held.has('turnR') ? 6 : 0) - (held.has('turnL') ? 6 : 0); // ← / → held: rewind / skip at 6×
  const t0 = c.t;
  c.t = Math.min(c.trip.end, Math.max(0, c.t + (ff || (c.paused ? 0 : 1)) * s));
  if (c.t > t0) for (const n of c.trip.notes) if (n.t > t0 && n.t <= c.t) showCard(n.title, n.body(), 7500);
  // look: at the hole plus the path's offset plus where you dragged the view (eases back to the path in ~4 s)
  c.offY += state.yaw - c.yawSet; c.offP += state.pitch - c.pitchSet;
  if (performance.now() - lastDrag > 1500) { const d = Math.exp(-dt / 4000); c.offY *= d; c.offP *= d; }
  const p = tripPose(c.trip, c.t);
  state.pos = ksOf(p.r, p.th, p.ph); state.r = p.r;
  state.vel = c.t > t0 && !ff ? tripVelocity(p, tripPose(c.trip, c.t + 0.05), 0.05) : [0, 0, 0];
  const f = localRadial(state.pos).map((x) => -x);
  state.yaw = Math.atan2(f[1], f[0]) + p.dy + c.offY;
  state.fovY = p.fov * Math.PI / 180;
  state.pitch = Math.max(-1.5, Math.min(1.5, Math.asin(f[2]) + p.dp + c.offP));
  c.yawSet = state.yaw; c.pitchSet = state.pitch;
  if (c.t >= c.trip.end && !c.paused && !ff) { startFall(); syncUI(); }
}

// ---------- passes ----------
// The cache only flags photon-ring pixels (subN 0): the refinement re-traces every one of them at screen resolution
// with the ray bundle, and shading never reads the cache's own ring average. (Bundling here as well cost up to half
// of the trace, measured, for nothing.)
function traceInto(cache, w, h, rowFrom, rowTo, cam, tanX, tanY, jit = [0, 0]) {
  const g = P.geo;
  traceJit = jit; cache.jit = jit;
  gl.bindFramebuffer(gl.FRAMEBUFFER, cache.fbo);
  gl.viewport(0, 0, w, h);
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(0, rowFrom, w, rowTo - rowFrom);
  gl.useProgram(g.p);
  geoUniforms(g, w, h, cam, tanX, tanY, 0);
  draw();
  traceJit = [0, 0];
  gl.disable(gl.SCISSOR_TEST);
  cache.used = [w, h];
}

// Edge refinement: at screen resolution, only the pixels where the cache can't be interpolated (edges, thin disk
// images) or where light looped the hole (photon ring, supersampled). See GEODESIC, RING_PASS 1.
// REFINE_MASK marks the pixels to refine in the stencil buffer (1); the tracer runs only on those (the GPU skips
// unmarked pixels before the shader starts, so the few traced pixels are packed together); RING_MASK marks the ring
// pixels it found (2) and the bundle pass adds their extra rays.
// strips > 1 splits the tracing into horizontal strips, separate draws with a flush between: the sharp refinement
// at a = 0.999 close up is ~0.85 s of GPU work, and one draw running past ~2 s makes Windows reset the GPU
// (context lost), which a slower GPU than the owner's could reach.
// scale < 1 (moving only): refine at that fraction of screen resolution, into the corner of the buffer (the shading
// reads it scaled, out.refScale). Close in at fast spin the camera sits inside the photon shell and ~12% of the
// screen is photon ring: refining all of it at screen resolution took 90-250 ms per moving frame at a = 0.999
// (measured on the Cinematic path), and the trip never stops moving. tune() sizes it from the measured GPU time.
// from / to: run only those strips (lite quality spreads the sharp refinement over frames); prepare: clear and build
// the stencil mask first (false when continuing with later strips).
function traceRefine(cache, cam, tanX, tanY, subN, strips = 1, scale = 1, from = 0, to = strips, prepare = true) {
  const rw = Math.max(1, Math.round(out.w * scale)), rh = Math.max(1, Math.round(out.h * scale));
  out.refScale = rw / out.w;
  gl.bindFramebuffer(gl.FRAMEBUFFER, out.ref.fbo);
  gl.viewport(0, 0, rw, rh);
  if (prepare) {
  gl.clearColor(0, 0, 0, 0); gl.clearStencil(0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
  gl.enable(gl.STENCIL_TEST);
  const m = P.mask;
  gl.useProgram(m.p);
  gl.colorMask(false, false, false, false);
  gl.stencilFunc(gl.ALWAYS, 1, 0xff);
  gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
  bindTex(0, cache.ring, m, 'uCacheRing');
  bindTex(1, cache.hits, m, 'uCacheHits');
  bindTex(2, cache.sky, m, 'uCacheSky');
  gl.uniform2f(m.u.uCacheSize, ...cache.used);
  gl.uniform2f(m.u.uRes, rw, rh);
  draw();
  gl.colorMask(true, true, true, true);
  }
  gl.enable(gl.STENCIL_TEST);
  if (strips > 1) gl.enable(gl.SCISSOR_TEST);
  for (let k = from; k < to; k++) {
    const y0 = Math.floor(rh * k / strips), y1 = Math.floor(rh * (k + 1) / strips);
    if (strips > 1) gl.scissor(0, y0, rw, y1 - y0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.ref.fbo);
    gl.stencilFunc(gl.EQUAL, 1, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    const g = P.geoRing;
    gl.useProgram(g.p);
    geoUniforms(g, rw, rh, cam, tanX, tanY, 0);
    draw();
    if (subN > 0) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.ref.stencilFbo);
      const rm = P.ringMask;
      gl.useProgram(rm.p);
      gl.stencilFunc(gl.ALWAYS, 2, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      bindTex(0, out.ref.ring, rm, 'uRefRing');
      draw();
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.ref.fbo);
      gl.drawBuffers([gl.NONE, gl.NONE, gl.COLOR_ATTACHMENT2]); // the bundle only writes the ring target
      gl.stencilFunc(gl.EQUAL, 2, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
      const b = P.geoBundle;
      gl.useProgram(b.p);
      geoUniforms(b, rw, rh, cam, tanX, tanY, subN);
      draw();
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
    }
    if (strips > 1) gl.flush();
  }
  gl.disable(gl.SCISSOR_TEST);
  gl.disable(gl.STENCIL_TEST);
}

function geoUniforms(g, w, h, cam, tanX, tanY, subN) {
  const a = state.spin;
  gl.uniform2f(g.u.uRes, w, h);
  const jx = traceJit[0] / w, jy = traceJit[1] / h;
  gl.uniform4f(g.u.uRect, jx, jy, 1 + jx, 1 + jy);
  gl.uniform1f(g.u.uSpin, a);
  gl.uniform1f(g.u.uRH, K.outerHorizon(a));
  gl.uniform1f(g.u.uRM, K.innerHorizon(a));
  gl.uniform1f(g.u.uRin, K.isco(a));
  gl.uniform1f(g.u.uRout, DISK_OUT);
  gl.uniform1f(g.u.uRFar, Math.max(600, state.r * 1.05));
  gl.uniform1f(g.u.uRCheck, K.captureRadius(a));
  gl.uniform1i(g.u.uBudget, Math.min(K.stepBudget(a), budgetCap));
  gl.uniform3f(g.u.uCam, ...cam.pos);
  gl.uniform4f(g.u.uE0, ...cam.low.e0); gl.uniform4f(g.u.uER, ...cam.low.right);
  gl.uniform4f(g.u.uEU, ...cam.low.up); gl.uniform4f(g.u.uEF, ...cam.low.fwd);
  gl.uniform2f(g.u.uTan, tanX, tanY);
  // Photon-ring pixels: light that wound around the hole. The angle a ray sweeps around the hole grows by ~1 rad per
  // e-fold closer to the critical ray (the photon sphere's instability), measured with kerr.js at a = 0 and 0.999,
  // 12 and 30 M. A pixel spans Δb ≈ r · (pixel angle), so threshold = 3.0 − ln Δb_pixel flags the pixels within
  // ~2-3 px of the ring. Bigger pixels (far away, moving, light mode) lower it, so the ring is never missed. (The
  // first test, time spent inside a fixed band around the photon orbits, flagged over half the screen at a = 0.999,
  // whose band reaches out to 5 M: the refinement then took 1.7 s per draw and Windows reset the GPU.)
  const dbPixel = state.r * (2 * tanY / h);
  gl.uniform1i(g.u.uSubN, subN);
  if (g.u.uBundle) gl.uniform1i(g.u.uBundle, lite ? 0 : 1);
  gl.uniform1f(g.u.uRingSweep, Math.max(1.6 * Math.PI, 3.0 - Math.log(dbPixel)));
}

// skyPass: recolour the (static) sky into out.skyCol: only when the cache shown changed or the sky mode changed
// skyScale: the sky pass runs at half resolution while moving (it is redone on every moving frame), full when still
function shadeAndPost(cache, now, tanY, skyPass, fade, skyScale) {
  const passes = skyPass ? [1, 2] : [2];
  for (const pass of passes) {
    const progs = lite ? liteProgs() : P, s = pass === 1 ? progs.sky : progs.disk;
    gl.useProgram(s.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pass === 1 ? out.skyFbo : out.hdrFbo);
    const ow = pass === 1 ? Math.round(out.w * skyScale) : out.w, oh = pass === 1 ? Math.round(out.h * skyScale) : out.h;
    if (pass === 1) out.skySize = [ow, oh];
    gl.viewport(0, 0, ow, oh);
    shadeUniforms(s, cache, now, tanY, ow, oh);
    // pass 1 writes skyCol, so it must not also be bound for reading (WebGL refuses the draw: feedback loop)
    bindTex(5, pass === 1 ? lutTex : out.skyCol, s, 'uSkyCol');
    draw();
  }
  // TAA (moving, not lite): blend this frame with the history into the other history buffer, which is then shown
  if (taa) {
    const a = out.histIdx, b = 1 - a;
    if (!out.histValid) {                                      // no history yet: start from this frame
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, out.hdrFbo); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, out.histFbo[a]);
      gl.blitFramebuffer(0, 0, out.w, out.h, 0, 0, out.w, out.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      out.histValid = true;
    }
    gl.useProgram(P.taa.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.histFbo[b]);
    gl.viewport(0, 0, out.w, out.h);
    bindTex(0, out.hdr, P.taa, 'uCur'); bindTex(1, out.hist[a], P.taa, 'uHist');
    gl.uniform1f(P.taa.u.uAlpha, TAA_ALPHA);
    draw();
    out.histIdx = b; out.show = out.hist[b]; out.showFbo = out.histFbo[b];
  } else { out.show = out.hdr; out.showFbo = out.hdrFbo; out.histValid = false; }
  postProcess(now, fade);
}

function shadeUniforms(s, cache, now, tanY, ow, oh) {
  bindTex(0, cache.hits, s, 'uHits'); bindTex(1, cache.sky, s, 'uSky'); bindTex(6, cache.ring, s, 'uRing');
  bindTex(7, out.ref.hits, s, 'uRHits'); bindTex(8, out.ref.sky, s, 'uRSky'); bindTex(9, out.ref.ring, s, 'uRRing');
  gl.uniform1f(s.u.uRefScale, (out.refScale ?? 1) * out.w / ow);
  if (s.u.uJit) gl.uniform2f(s.u.uJit, ...(cache.jit || [0, 0]));
  if (s.u.uGradK) gl.uniform1f(s.u.uGradK, fullReady ? 1 : MOVE_GRAD);
  bindTex(2, noiseTex, s, 'uNoise'); bindTex(3, mwTex, s, 'uMilkyWay'); bindTex(4, lutTex, s, 'uBlackbody');
  gl.uniform2f(s.u.uGSize, ...cache.used);
  gl.uniform2f(s.u.uOutSize, ow, oh);
  if (s.u.uSkyScale) gl.uniform2f(s.u.uSkyScale, (out.skySize?.[0] ?? ow) / ow, (out.skySize?.[1] ?? oh) / oh);
  gl.uniformMatrix3fv(s.u.uGalaxy, false, galaxy);
  gl.uniform1f(s.u.uTime, now / 1000);
  gl.uniform1f(s.u.uSpin, state.spin);
  gl.uniform1f(s.u.uRin, K.isco(state.spin));
  gl.uniform1f(s.u.uRout, DISK_OUT);
  gl.uniform1f(s.u.uPix, 2 * tanY / oh);
  // In the fall the camera re-exposes for the sky (the bright disk is behind it): the film look's dim sky eases to
  // the crisp level, together with the starlight's frequency shift. Else the blueshifted rim was barely visible.
  const fallK = state.fall ? Math.min(1, (performance.now() - state.fall.t0) / 2500) : 0;
  gl.uniform1f(s.u.uMWBright, LOOK.milkyWay + (LOOK_PRESETS.crisp.milkyWay - LOOK.milkyWay) * Math.max(0, fallK));
  gl.uniform1f(s.u.uStarBright, LOOK.stars + (LOOK_PRESETS.crisp.stars - LOOK.stars) * Math.max(0, fallK));
  gl.uniform1f(s.u.uDiskBright, LOOK.disk);
  gl.uniform2f(s.u.uDiskTemp, ...LOOK.diskTemp);
  gl.uniform1f(s.u.uDust, LOOK.dust);
  gl.uniform1f(s.u.uFalloff, LOOK.falloff);
  gl.uniform1f(s.u.uDensPow, LOOK.densPow);
  gl.uniform1i(s.u.uSkyMode, state.sky);
  // the starlight's frequency shift: always in Reality; in Film it fades in over 2.5 s falling in (the film shows none)
  const reality = state.mode === 'reality';
  if (s.u.uShift) gl.uniform1f(s.u.uShift, reality ? 1 : state.fall ? Math.min(1, (performance.now() - state.fall.t0) / 2500) : 0);
  if (s.u.uReality) {
    // the disk's shift needs each pixel's ray: the camera of this frame's trace
    gl.uniform1f(s.u.uReality, reality ? 1 : 0);
    gl.uniform1f(s.u.uShiftCap, LOOK.shiftCap ?? 40);
    gl.uniform3f(s.u.uCam, ...shadeCam.cam.pos);
    gl.uniform4f(s.u.uE0, ...shadeCam.cam.low.e0); gl.uniform4f(s.u.uER, ...shadeCam.cam.low.right);
    gl.uniform4f(s.u.uEU, ...shadeCam.cam.low.up); gl.uniform4f(s.u.uEF, ...shadeCam.cam.low.fwd);
    gl.uniform2f(s.u.uTan, ...shadeCam.tan);
  }
}

function postProcess(now, fade) {
  // bloom: down the chain, then back up (additive)
  let src = out.show, sw = out.w, sh = out.h;
  gl.useProgram(P.down.p);
  out.mips.forEach((m, i) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, m.fbo);
    gl.viewport(0, 0, m.w, m.h);
    bindTex(0, src, P.down, 'uSrc');
    gl.uniform2f(P.down.u.uTexel, 1 / sw, 1 / sh);
    gl.uniform1i(P.down.u.uFirst, i === 0 ? 1 : 0);
    draw();
    src = m.tex; sw = m.w; sh = m.h;
  });
  gl.useProgram(P.up.p);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  for (let i = out.mips.length - 1; i > 0; i--) {
    const from = out.mips[i], to = out.mips[i - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, to.fbo);
    gl.viewport(0, 0, to.w, to.h);
    bindTex(0, from.tex, P.up, 'uSrc');
    gl.uniform2f(P.up.u.uTexel, 1 / from.w, 1 / from.h);
    gl.uniform2f(P.up.u.uDstSize, to.w, to.h);
    gl.uniform1f(P.up.u.uWeight, 1.0);
    draw();
  }
  gl.disable(gl.BLEND);

  const c = P.comp;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, out.w, out.h);
  gl.useProgram(c.p);
  bindTex(0, out.show, c, 'uHdr');
  bindTex(1, out.mips[0].tex, c, 'uBloom');
  bindTex(2, out.prev, c, 'uPrev');
  bindTex(3, out.mips[Math.min(4, out.mips.length - 1)].tex, c, 'uHazeTex'); // levels 4+ summed by the up-chain
  gl.uniform1f(c.u.uHaze, LOOK.haze);
  gl.uniform3f(c.u.uHazeTint, ...LOOK.hazeTint);
  gl.uniform1f(c.u.uSaturation, LOOK.saturation);
  gl.uniform1f(c.u.uFade, fade);
  gl.uniform2f(c.u.uOutSize, out.w, out.h);
  gl.uniform1f(c.u.uExposure, LOOK.exposure);
  gl.uniform1f(c.u.uBloomAmt, LOOK.bloom);
  gl.uniform1f(c.u.uBloomNorm, 1 / out.mips.length);
  gl.uniform1f(c.u.uTime, now / 1000);
  gl.uniform1f(c.u.uVignette, LOOK.vignette);
  gl.uniform1f(c.u.uGrain, LOOK.grain);
  draw();
}

// How long 1 hour of the camera's time is on Earth (far away), given that ratio: readable units.
function earthTime(ratio) {
  const h = ratio;
  if (h < 10) return `${h.toFixed(h < 1.1 ? 3 : 2)} h`;
  if (h < 48) return `${h.toFixed(0)} h`;
  if (h < 24 * 365 * 2) return `${(h / 24).toFixed(0)} days`;
  return `${(h / 24 / 365.25).toFixed(1)} years`;
}

// ---------- frame loop ----------
// Time budget, aiming at 60 fps. Colouring the cache (shade) costs about the same every frame; the ray tracing is
// sized to fit in what is left: the moving resolution, and the rows traced per frame while sharpening.
// The times come from the GPU itself (EXT_disjoint_timer_query_webgl2, Chrome / Edge): each pass is wrapped in a
// timer query, read back a few frames later. (The first version guessed the shading cost from frame times; the
// browser delivers frames in bursts, the guess collapsed to 0.2 ms while the real cost was 17 ms, and sharpening
// crawled at its minimum band.) Without the timer (Firefox, Safari), see the fallback in frame().
const FRAME_MS = 1000 / 60;
const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
const queries = [];
function timed(kind, meta, fn) {
  if (!timer || queries.length > 40) { fn(); return; }
  const q = gl.createQuery();
  gl.beginQuery(timer.TIME_ELAPSED_EXT, q);
  fn();
  gl.endQuery(timer.TIME_ELAPSED_EXT);
  queries.push({ q, kind, meta });
}
// measured GPU costs: shading and refinement (ms per frame), tracing (ms per traced pixel, moving / sharpening)
const gpu = { shade: 12, refine: 8, lowRate: 0, bandRate: 0 };
function readTimers() {
  const disjoint = gl.getParameter(timer.GPU_DISJOINT_EXT);
  while (queries.length && gl.getQueryParameter(queries[0].q, gl.QUERY_RESULT_AVAILABLE)) {
    const { q, kind, meta } = queries.shift();
    const ms = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
    gl.deleteQuery(q);
    if (disjoint || !(ms > 0)) continue;
    const ema = (old, v, k) => (old ? old + (v - old) * k : v);
    if (kind === 'shade') gpu.shade = ema(gpu.shade, ms, 0.2);
    else if (kind === 'refine') gpu.refine = ema(gpu.refine, ms, 0.3);
    else if (kind === 'low') gpu.lowRate = ema(gpu.lowRate, ms / meta, 0.4);
    else if (kind === 'band') gpu.bandRate = ema(gpu.bandRate, ms / meta, 0.4);
  }
}
// Sizes from the measured costs: the moving trace gets what shading and refinement leave of the frame (at least
// 3 ms); the sharpening bands at least 8 ms (sharp soon after a stop matters more than a few frames at 60).
// The trip aims at 40 fps instead of 60 while it moves: a sharper moving picture between its stops.
const frameMs = () => (state.cine && !state.fall ? 1000 / 40 : FRAME_MS);
function tune() {
  const FRAME_MS = frameMs();
  if (gpu.lowRate) {
    const px = Math.max(FRAME_MS - gpu.shade - gpu.refine, 3) / gpu.lowRate;
    const s = Math.sqrt(px / (out.w * out.h));
    lowScale = Math.min(QUALITY.lowMax, Math.max(QUALITY.lowMin, lowScale + (s - lowScale) * 0.5));
  }
  // moving refinement: at most ~35% of what shading leaves of the frame (the moving trace needs the rest)
  if (gpu.refine) {
    const want = refScale * Math.sqrt(Math.max(4, (FRAME_MS - gpu.shade) * 0.35) / gpu.refine);
    refScale = Math.min(1, Math.max(QUALITY.refMin, refScale + (want - refScale) * 0.5));
    // still too slow at the lowest scale (fast spin, inside the photon shell, the bundle is ~3/4 of it): 1 ray per
    // ring pixel instead of 4 while moving; back to 4 once there is room for ~3× the cost
    const target = Math.max(4, (FRAME_MS - gpu.shade) * 0.35);
    if (subMove === 4 && refScale <= QUALITY.refMin + 0.01 && gpu.refine > 1.5 * target) subMove = 1;
    else if (subMove === 1 && gpu.refine < 0.3 * target) subMove = 4;
  }
  if (gpu.bandRate) band = Math.max(16, Math.round(Math.max(FRAME_MS - gpu.shade, 8) / gpu.bandRate / out.full.w));
}
let refScale = 1, subMove = 4;
// Moving frames trace at most MOVING_BUDGET steps per ray. Fast spin needs up to 1200 for a sharp shadow edge
// (kerr.js stepBudget), but while moving the picture is low resolution anyway; the still picture gets the full budget.
const MOVING_BUDGET = 600;
let budgetCap = Infinity;
let lastKey = '', lowScale = QUALITY.lowMax * 0.7, band = 96, fullRow = 0, fullReady = false, lowValid = false;
let shadeCam = null;
// While moving: the sky at 3/4 resolution (at half, stars turned into soft blobs; full cost ~5 ms more per frame on the
// owner's integrated GPU), the disk texture read 1/0.6 finer
// (its detail softened; at 0.35 the dust lanes broke into blocks), and TAA (see shader.js TAA).
let MOVE_SKY = 0.75, MOVE_GRAD = 0.6;
const TAA_ALPHA = 0.75, HALTON = [...Array(8)].map((_, i) => {
  const h = (n, b) => { let f = 1, r = 0; while (n > 0) { f /= b; r += f * (n % b); n = Math.floor(n / b); } return r; };
  return [h(i + 1, 2) - 0.5, h(i + 1, 3) - 0.5];
});
let TAA_ON = true, traceJit = [0, 0], jitIdx = 0, taa = false, lastCamPos = null, lastYaw = 0;
const LITE_STRIPS = 8;
let refStep = 0;
let lastNow = performance.now(), frames = 0, lastFps = lastNow, fps = 0, traced = 'none', dtAvg = FRAME_MS;
const FADE_MS = 300;
let fadeStart = -1e9, lastSky = -1, lastMw = -1;

// ---------- automatic quality (weak computers, phones; owner 2026-11-18: 3-4 fps on a phone) ----------
// Measured with a software GPU standing in for a weak one (960×600): the sharp picture's ring bundle took 18 s in one
// frame, the moving edge refinement 0.2-2.4 s, the disk pass 0.1-0.4 s per frame. Two levers, both automatic:
// - lite: the disk without its cloud layer and two star layers instead of three, no ring bundle (ring pixels get one
//   ray, like edges), no edge refinement while moving. On from the start on phones and without a GPU; on any other
//   device once frames run slow. It stays on (switching back and forth would flicker the look).
// - renderScale: the whole picture's resolution (the browser scales the canvas up), 0.4-1. Every 1.5 s the median
//   frame time decides: under 22 fps twice in a row → 0.8× (or lite first), over ~50 fps for 8 s after the last step
//   down → 1.15×. Counted only after the start-screen choice. (At 26 fps once, the trip's brief dips switched a fast PC
//   to lite.)
let lite = phone || light, renderScale = phone || light ? 0.75 : 1;
const SCALE_MIN = 0.4, perf = { win: [], since: 0, lastDown: -1e9, slow: 0, from: Infinity };
function autoQuality(now, dt) {
  if (!ui.start.hidden) perf.from = Infinity; else if (perf.from === Infinity) perf.from = now + 3000;
  if (document.hidden || now < perf.from || !out) { perf.win = []; perf.since = now; return; }
  perf.win.push(dt);
  if (now - perf.since < 1500 || perf.win.length < 8) return;
  const med = perf.win.sort((a, b) => a - b)[perf.win.length >> 1];
  perf.win = []; perf.since = now;
  perf.slow = med > 1000 / 22 ? perf.slow + 1 : 0;
  if (perf.slow >= 2) {
    perf.slow = 0;
    if (!lite) lite = true;
    else renderScale = Math.max(SCALE_MIN, renderScale * 0.8);
    perf.lastDown = now; lastKey = '';
  } else if (med < 1000 / 50 && renderScale < 1 && now - perf.lastDown > 8000) {
    renderScale = Math.min(1, renderScale * 1.15);
  }
}
function clearRefine() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, out.ref.fbo);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  out.refScale = 1;
}

function frame(now) {
  const dt = Math.min(Math.max(now - lastNow, 0), 250); lastNow = now; // (the first rAF time can precede lastNow)
  if (timer) { readTimers(); if (out) tune(); }
  else {
    // Fallback, from frame times: the moving resolution steers the whole frame towards 60 fps (if the GPU can't
    // get there, it settles at the lowest resolution: the fastest it can do), and sharpening bands grow while a
    // frame stays under 30 ms. No estimate of the shading cost is needed (one measured synchronously read 14 ms
    // in one run and 35 in the next). Frame times come in bursts (6 ms, 90 ms, ...) and lag the work by 2-3
    // frames, so the moving scale follows a smoothed frame time in small steps; else it swings between its limits.
    dtAvg += (dt - dtAvg) * 0.25;
    if (traced === 'low' && lowValid) {
      lowScale = Math.min(QUALITY.lowMax, Math.max(QUALITY.lowMin, lowScale * Math.min(1.04, Math.max(0.96, Math.sqrt(FRAME_MS / Math.max(dtAvg, 1))))));
    } else if (traced === 'band') band = Math.max(16, Math.round(band * Math.min(1.5, Math.max(0.5, 30 / Math.max(dt, 1)))));
  }
  traced = 'none';
  if (state.cine) cineTick(dt);
  if (sound.gain && now - lastVolT > 150) { lastVolT = now; setGain(); }
  if (state.fall) { if (!state.cine?.paused) fallTick(dt); } else if (!state.cine) fly(dt);
  // device pixels, capped at ~2560×1440 worth (the caches hold ~70 bytes per pixel)
  autoQuality(now, dt);
  let dpr = Math.min(window.devicePixelRatio || 1, QUALITY.dpr);
  dpr = Math.min(dpr, Math.sqrt(3.7e6 / Math.max(1, canvas.clientWidth * canvas.clientHeight))) * renderScale;
  const w = Math.max(32, Math.round(canvas.clientWidth * dpr)), h = Math.max(32, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h || !out) {
    canvas.width = w; canvas.height = h;
    buildBuffers(w, h);
    lastKey = ''; lowValid = false;
  }
  const tanY = Math.tan(state.fovY / 2), tanX = tanY * w / h;
  const key = [...state.pos, ...state.vel, state.yaw, state.pitch, state.spin, w, h].join();
  const cam = cameraFrame();
  shadeCam = { cam, tan: [tanX, tanY] };

  let skyPass = state.sky !== lastSky || mwVersion !== lastMw;
  lastSky = state.sky; lastMw = mwVersion;
  if (key !== lastKey || !lowValid) {
    // moving: trace the whole view at reduced resolution, then refine edges and the ring at screen resolution
    const lw = Math.max(16, Math.round(w * lowScale)), lh = Math.max(16, Math.round(h * lowScale));
    budgetCap = MOVING_BUDGET;
    // TAA while moving (not lite): jittered rays; history dropped after a jump (reset, a big turn or move)
    const jump = !lastCamPos || Math.hypot(...state.pos.map((x, i) => x - lastCamPos[i])) > 0.03 * state.r || Math.abs(state.yaw - lastYaw) > 0.15;
    taa = !lite && TAA_ON; if (jump) out.histValid = false;
    const jit = taa ? HALTON[jitIdx++ % HALTON.length] : [0, 0];
    timed('low', lw * lh, () => traceInto(out.low, lw, lh, 0, lh, cam, tanX, tanY, jit));
    if (lite) clearRefine();                                   // lite: no edge refinement while moving
    else {                                                     // (refine rays jittered too: TAA smooths the ring's edge)
      traceJit = jit;
      timed('refine', 0, () => traceRefine(out.low, cam, tanX, tanY, subMove, 1, refScale));
      traceJit = [0, 0];
    }
    budgetCap = Infinity;
    traced = 'low'; lowValid = true; lastKey = key; fullRow = 0; fullReady = false; skyPass = true;
    lastCamPos = [...state.pos]; lastYaw = state.yaw;
  } else if (!fullReady) {
    // still: trace at full resolution, a band of rows per frame
    if (fullRow < out.full.h) {
      const to = Math.min(out.full.h, fullRow + band);
      timed('band', (to - fullRow) * out.full.w, () => traceInto(out.full, out.full.w, out.full.h, fullRow, to, cam, tanX, tanY));
      traced = 'band'; fullRow = to; refStep = 0;
    } else if (lite && refStep < LITE_STRIPS) {
      // lite: the sharp refinement one strip per frame (in one frame it froze a weak GPU for ~0.3 s)
      traceRefine(out.full, cam, tanX, tanY, 0, LITE_STRIPS, 1, refStep, refStep + 1, refStep === 0);
      refStep++;
    }
    if (fullRow >= out.full.h && (!lite || refStep >= LITE_STRIPS)) {
      // hand over from the low-res picture to the sharp one with a short cross-fade: near the rings, where light
      // bends hardest, the interpolated low-res picture puts features slightly off, and a hard switch would jump
      fullReady = true; skyPass = true; taa = false;
      if (!lite) traceRefine(out.full, cam, tanX, tanY, 16, 6);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, out.showFbo || out.hdrFbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, out.prevFbo);
      gl.blitFramebuffer(0, 0, out.w, out.h, 0, 0, out.w, out.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      fadeStart = now;
    }
  }
  const fade = Math.max(0, 1 - (now - fadeStart) / FADE_MS);
  timed('shade', 0, () => shadeAndPost(fullReady ? out.full : out.low, now, tanY, skyPass, fullReady ? fade : 0, fullReady ? 1 : lite ? 0.5 : MOVE_SKY));

  frames++;
  if (now - lastFps > 500) {
    fps = frames * 1000 / (now - lastFps); frames = 0; lastFps = now;
    syncUI();
    const q = fullReady ? 'sharp' : (fullRow > 0 ? `sharpening ${Math.round(100 * fullRow / out.full.h)}%` : 'moving');
    const au = state.r * KM_PER_M / KM_PER_AU;
    const f = state.fall;
    const c = state.cine, trip = c ? `trip ${mmss(c.t)}${f ? "" : ` / ${mmss(c.trip.end)}`}${c.paused ? ' (paused)' : ''} · ` : '';
    hud.textContent = trip + (f
      ? `${f.inside ? 'inside the horizon' : 'falling freely'} · r ${state.r.toFixed(2)} M · your clock +${clock(f.tau)} · ${fps.toFixed(0)} fps · ${q}`
      : `distance ${state.r.toFixed(1)} M (${au < 10 ? au.toFixed(1) : Math.round(au)} AU) · speed ${camInfo.speed.toFixed(2)} c`
        + ` · 1 h here = ${earthTime(camInfo.gamma / camInfo.alpha)} on Earth · ${fps.toFixed(0)} fps · ${q}`);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Profiler (console: __gargantua.profile()): GPU time of each pass at the current view, median of n runs.
// Each pass is followed by a 1-pixel readPixels, which waits for the GPU to finish it.
function profile(n = 7) {
  const px = new Uint8Array(4), cam = cameraFrame();
  const tanY = Math.tan(state.fovY / 2), tanX = tanY * out.w / out.h;
  shadeCam = { cam, tan: [tanX, tanY] };
  const sync = () => { gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); };
  const time = (fn) => {
    const t = [];
    for (let i = 0; i < n; i++) { sync(); const t0 = performance.now(); fn(); sync(); t.push(performance.now() - t0); }
    return +t.sort((x, y) => x - y)[n >> 1].toFixed(2);
  };
  const lw = Math.round(out.w * 0.5), lh = Math.round(out.h * 0.5);
  const res = {
    size: `${out.w}x${out.h}`, r: state.r, spin: state.spin,
    traceFull: time(() => traceInto(out.full, out.full.w, out.full.h, 0, out.full.h, cam, tanX, tanY)),
    refineFull: time(() => traceRefine(out.full, cam, tanX, tanY, 16)),
    traceHalf: time(() => traceInto(out.low, lw, lh, 0, lh, cam, tanX, tanY)),
    refineHalf: time(() => traceRefine(out.low, cam, tanX, tanY, 4)),
  };
  // the moving refinement as frame() runs it (capped budget, scale 0.4), with and without the ring bundle
  budgetCap = MOVING_BUDGET;
  const ls = Math.round(out.w * 0.3), lsh = Math.round(out.h * 0.3);
  traceInto(out.low, ls, lsh, 0, lsh, cam, tanX, tanY);
  res.traceMove = time(() => traceInto(out.low, ls, lsh, 0, lsh, cam, tanX, tanY));
  res.refineMove = time(() => traceRefine(out.low, cam, tanX, tanY, 4, 1, 0.4));
  res.refineMoveNoBundle = time(() => traceRefine(out.low, cam, tanX, tanY, 0, 1, 0.4));
  budgetCap = Infinity;
  // leave a consistent state: the full cache + its refinement shown
  traceInto(out.full, out.full.w, out.full.h, 0, out.full.h, cam, tanX, tanY);
  traceRefine(out.full, cam, tanX, tanY, 16);
  const now = performance.now();
  res.skyPass = time(() => { gl.useProgram(P.sky.p); gl.bindFramebuffer(gl.FRAMEBUFFER, out.skyFbo); gl.viewport(0, 0, out.w, out.h); out.skySize = [out.w, out.h]; shadeUniforms(P.sky, out.full, now, tanY, out.w, out.h); bindTex(5, lutTex, P.sky, 'uSkyCol'); draw(); });
  res.diskPass = time(() => { gl.useProgram(P.disk.p); gl.bindFramebuffer(gl.FRAMEBUFFER, out.hdrFbo); gl.viewport(0, 0, out.w, out.h); shadeUniforms(P.disk, out.full, now, tanY, out.w, out.h); bindTex(5, out.skyCol, P.disk, 'uSkyCol'); draw(); });
  res.post = time(() => postProcess(now, 0));
  // share of screen pixels the refinement re-traced (edge) and supersampled (ring), for the sharp picture
  {
    const buf = new Float32Array(out.w * out.h * 4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, out.ref.fbo); gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(0, 0, out.w, out.h, gl.RGBA, gl.FLOAT, buf);
    gl.readBuffer(gl.COLOR_ATTACHMENT0); gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    let e = 0, rg = 0;
    for (let i = 3; i < buf.length; i += 4) { if (buf[i] > 1.5) rg++; else if (buf[i] > 0.5) e++; }
    res.refinedPct = +(100 * (e + rg) / (out.w * out.h)).toFixed(1); res.ringPct = +(100 * rg / (out.w * out.h)).toFixed(1);
  }
  lastKey = ''; // re-trace normally afterwards
  return res;
}
// tuner: what the frame budget measured (GPU ms per pass, or per traced pixel) and what it chose
window.__gargantua = {
  setMove(sky, grad, t = true) { MOVE_SKY = sky; MOVE_GRAD = grad; TAA_ON = t; },
  state, LOOK, profile, renderer, placeAt, setMode, setSpin, startTrip, stopTrip, choose, sound,
  get camera() { return { ...camInfo, r: state.r, fall: state.fall && { tau: state.fall.tau, inside: state.fall.inside, ended: state.fall.ended } }; },
  startFall,
  get ready() { return fullReady; },
  get lowScale() { return lowScale; },
  get tuner() { return { timer: !!timer, gpu: { ...gpu }, lowScale, refScale, subMove, band, lite, renderScale }; },
};
