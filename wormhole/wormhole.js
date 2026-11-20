// The wormhole from Interstellar: the page. Physics in worm.js, GPU passes in worm-shader.js.
//
// The camera's state: ℓ (signed: < 0 Saturn's side, > 0 the far side) and its place on the sphere of constant ℓ, the
// unit vector u. Its look direction and axes are vectors in the local orthonormal frame (e_ℓ, e_θ, e_φ), written as
// 3-vectors with e_ℓ ↦ u and the two tangent directions ↦ the plane perpendicular to u. That frame is continuous
// through the throat, so flying straight in on Saturn's side comes out flying straight away on the far side with no
// special case. Moving sideways turns u (by distance / r(ℓ)), and the camera axes are carried along with it.
import * as W from './worm.js';
import { VERTEX } from '../shader.js';
import { RAYS, VIEW } from './worm-shader.js';

document.getElementById('solarSwitch')?.addEventListener('click', (e) => { e.preventDefault(); window.PortfolioView?.switchTo('solar'); });

const canvas = document.getElementById('sim');
const hud = document.getElementById('hud');
const failBox = document.getElementById('simFail');
const fail = (msg) => { failBox.textContent = msg; failBox.hidden = false; throw new Error(msg); };
const P = W.params();                 // the film's wormhole (units of ρ = 1 km)
const L_MAX = 40, L_START = -9;

// ---------- small vector helpers ----------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => scale(a, 1 / Math.hypot(...a));
const rotate = (v, axis, ang) => { // Rodrigues
  const c = Math.cos(ang), s = Math.sin(ang), k = axis;
  return add(add(scale(v, c), cross(k, v), s), k, dot(k, v) * (1 - c));
};

// ---------- WebGL ----------
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' });
if (!gl) fail('This simulation needs WebGL2. Try a recent Chrome, Edge, Firefox or Safari.');
if (!gl.getExtension('EXT_color_buffer_float')) fail('This simulation needs floating-point render targets (EXT_color_buffer_float), which this browser or GPU does not offer.');
function program(fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.bindAttribLocation(p, 0, 'aPos'); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) { const n = gl.getActiveUniform(p, i).name; u[n] = gl.getUniformLocation(p, n); }
  return { p, u };
}
const rays = program(RAYS), view = program(VIEW);
gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
const draw = () => gl.drawArrays(gl.TRIANGLES, 0, 3);

// the ray table: one row, N rays from ψ = 0 to π
const N = 2048;
const rayTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, rayTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, N, 1, 0, gl.RGBA, gl.FLOAT, null);
for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
const rayFbo = gl.createFramebuffer();
gl.bindFramebuffer(gl.FRAMEBUFFER, rayFbo);
gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, rayTex, 0);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
// fewer rays at lower resolutions (weak GPUs: the table, not the pixels, is then most of the frame)
let rayL = NaN, nRays = N;
function traceRays(l) {
  const n = renderScale < 0.6 ? 512 : renderScale < 0.85 ? 1024 : N;
  if (l === rayL && n === nRays) return;
  rayL = l; nRays = n;
  gl.bindFramebuffer(gl.FRAMEBUFFER, rayFbo); gl.viewport(0, 0, nRays, 1);
  gl.useProgram(rays.p);
  gl.uniform1f(rays.u.uL0, l); gl.uniform1f(rays.u.uRho, P.rho); gl.uniform1f(rays.u.uA, P.a); gl.uniform1f(rays.u.uM, P.M); gl.uniform1f(rays.u.uN, nRays);
  draw();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// the Milky Way on Saturn's side (NASA SVS Deep Star Maps 2020, shared with Gargantua)
let mwTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, mwTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
const img = new Image();
img.onload = () => {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.generateMipmap(gl.TEXTURE_2D);
  const an = gl.getExtension('EXT_texture_filter_anisotropic'); if (an) gl.texParameterf(gl.TEXTURE_2D, an.TEXTURE_MAX_ANISOTROPY_EXT, 8);
  mwTex = t;
};
img.src = new URL('../sky/milkyway.webp', import.meta.url).href;
// Saturn's-side sky → galactic frame: the bright galactic centre just behind the wormhole as seen from the start, so
// the Einstein ring around it is the Milky Way's core stretched round the edge (the sky straight behind was dark)
const galaxy = (() => {
  const C = norm([-1, -0.12, 0.08]), Nn = norm(cross(cross(C, [0.2, 0.3, 1]), C)), Y = cross(Nn, C);
  return new Float32Array([C[0], Y[0], Nn[0], C[1], Y[1], Nn[1], C[2], Y[2], Nn[2]]);
})();
// Saturn: on its side's sky behind the wormhole as seen from the start (so its light also bends around the throat)
const SAT = (() => {
  const dir = norm([-1, 0.32, 0.36]), across = norm([dir[2], 0, -dir[0]]);   // perpendicular to the line of sight
  const B = 24 * Math.PI / 180;                                             // ring opening angle as seen from here
  return { dir, pole: norm(add(scale(across, Math.cos(B)), dir, -Math.sin(B))), sun: norm(add(scale(dir, -1), [0, 0.7, 0.25])), r: 8 * Math.PI / 180 };
})();

// ---------- camera ----------
const state = { l: L_START, u: [1, 0, 0], F: [1, 0, 0], Up: [0, 0, 1], fovY: 50 * Math.PI / 180, cine: null, held: new Set(), wheel: 0 };
const right = () => norm(cross(state.F, state.Up));
function placeStart() {
  state.l = L_START; state.u = [1, 0, 0];
  // look at the throat (increasing ℓ), tilted up a little towards Saturn
  state.F = norm(rotate([1, 0, 0], [0, -1, 0], 0.16)); state.Up = norm(cross(cross(state.F, [0, 0, 1]), state.F));
  state.wheel = 0; hideCard();
}
function turn(yaw, pitch) {
  const R = right();
  state.F = norm(rotate(state.F, state.Up, yaw));
  state.F = norm(rotate(state.F, R, pitch)); state.Up = norm(rotate(state.Up, R, pitch));
}
// move by d (local frame): ℓ by its e_ℓ part, the place on the sphere by the rest (arc length / r), axes carried along
function move(d) {
  const dl = dot(d, state.u), dt = add(d, state.u, -dl), dn = Math.hypot(...dt);
  state.l = Math.max(-L_MAX, Math.min(L_MAX, state.l + dl));
  if (dn > 1e-9) {
    const axis = norm(cross(state.u, dt)), ang = dn / W.radius(state.l, P);
    state.u = norm(rotate(state.u, axis, ang)); state.F = norm(rotate(state.F, axis, ang)); state.Up = norm(rotate(state.Up, axis, ang));
  }
  state.Up = norm(add(state.Up, state.F, -dot(state.Up, state.F)));
}

// ---------- cards, sound, start screen ----------
const card = document.getElementById('simCard');
let cardTimer = 0;
function showCard(title, body, ms = 0) {
  card.innerHTML = '';
  const t = document.createElement('strong'); t.textContent = title; card.append(t);
  body.forEach((line, i) => { const e = document.createElement(i === body.length - 1 && body.length > 1 ? 'small' : 'span'); e.textContent = line; card.append(e); });
  card.classList.add('show'); clearTimeout(cardTimer);
  if (ms) cardTimer = setTimeout(() => card.classList.remove('show'), ms);
}
function hideCard() { clearTimeout(cardTimer); card.classList.remove('show'); }
placeStart();

// sound: the same NASA black hole sonification as Gargantua, looped seamlessly, louder near the throat
const ui = { start: document.getElementById('simStart'), keys: document.getElementById('simKeys'), mute: document.getElementById('simMute'),
  touch: document.getElementById('simTouch'), pause: document.getElementById('touchPause'), fly: document.querySelectorAll('#simTouch [data-hold]') };
const sound = { ctx: null, gain: null, on: true, next: 0 };
async function startSound() {
  if (sound.ctx) return;
  try {
    sound.ctx = new (window.AudioContext || window.webkitAudioContext)();
    sound.gain = sound.ctx.createGain(); sound.gain.gain.value = 0; sound.gain.connect(sound.ctx.destination);
    const buf = await sound.ctx.decodeAudioData(await (await fetch(new URL('../sound/blackhole.mp3', import.meta.url))).arrayBuffer());
    sound.next = sound.ctx.currentTime + 0.05;
    const XF = 2.5, schedule = () => {
      while (sound.next < sound.ctx.currentTime + 4) {
        const src = sound.ctx.createBufferSource(), g = sound.ctx.createGain(), t = sound.next, d = buf.duration;
        src.buffer = buf; src.connect(g); g.connect(sound.gain);
        g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(1, t + XF); g.gain.setValueAtTime(1, t + d - XF); g.gain.linearRampToValueAtTime(0, t + d);
        src.start(t); src.stop(t + d); sound.next = t + d - XF;
      }
    };
    schedule(); setInterval(schedule, 1000); setVolume();
  } catch (err) { console.warn('sound unavailable', err); }
}
const setGain = () => { if (sound.gain) sound.gain.gain.setTargetAtTime(sound.on ? 0.8 * Math.max(0.12, 1 - Math.abs(state.l) / 25) : 0, sound.ctx.currentTime, 0.6); };
function setVolume() {
  setGain();
  ui.mute.setAttribute('aria-pressed', String(!sound.on)); ui.mute.setAttribute('aria-label', sound.on ? 'Mute sound' : 'Turn sound on');
  ui.mute.innerHTML = `<i class="bi ${sound.on ? 'bi-volume-up' : 'bi-volume-mute'}" aria-hidden="true"></i>`;
}
const toggleSound = () => { sound.on = !sound.on; if (sound.on) startSound(); setVolume(); };
ui.mute.addEventListener('click', () => { toggleSound(); ui.mute.blur(); });
document.addEventListener('visibilitychange', () => { if (sound.ctx) (document.hidden ? sound.ctx.suspend() : sound.ctx.resume()); });

const phone = matchMedia('(pointer: coarse)').matches && !matchMedia('(any-pointer: fine)').matches;
const KEYS_FREE = 'W / S or ↑ / ↓ fly · A / D strafe · Q / E down / up · Shift faster · drag or ← / → to look · wheel to fly · M sound · R reset · Esc back';
const KEYS_TRIP = 'Space pause · drag to look · M sound · R restart · Esc back';
function syncUI() {
  const keys = phone ? (state.cine ? 'drag to look' : 'drag to look · hold ▲ / ▼ to fly') : state.cine ? KEYS_TRIP : KEYS_FREE;
  if (ui.keys.textContent !== keys) ui.keys.textContent = keys;
  ui.touch.hidden = !phone || !ui.start.hidden;
  ui.fly.forEach((b) => { b.hidden = !!state.cine; });
  ui.pause.hidden = !state.cine;
  const paused = !!state.cine?.paused;
  ui.pause.setAttribute('aria-label', paused ? 'Play' : 'Pause');
  ui.pause.innerHTML = `<i class="bi ${paused ? 'bi-play-fill' : 'bi-pause-fill'}" aria-hidden="true"></i>`;
}
function choose(trip) {
  if (ui.start.hidden) return;
  ui.start.hidden = true; ui.mute.hidden = false; startSound();
  placeStart();
  state.cine = trip ? { t: 0, paused: false, offY: 0, offP: 0 } : null;
  syncUI();
}
function backToStart() { state.cine = null; state.held.clear(); placeStart(); ui.start.hidden = false; syncUI(); }
document.querySelectorAll('[data-start]').forEach((b) => b.addEventListener('click', () => choose(b.dataset.start === 'trip')));
document.getElementById('touchBack').addEventListener('click', backToStart);
ui.pause.addEventListener('click', () => { if (state.cine) state.cine.paused = !state.cine.paused; syncUI(); });
ui.fly.forEach((b) => {
  b.addEventListener('pointerdown', (e) => { e.preventDefault(); state.held.add(b.dataset.hold); b.setPointerCapture?.(e.pointerId); });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach((ev) => b.addEventListener(ev, () => state.held.delete(b.dataset.hold)));
  b.addEventListener('contextmenu', (e) => e.preventDefault());
});

// ---------- input ----------
let dragging = false, lastX = 0, lastY = 0, lastDrag = 0;
canvas.addEventListener('pointerdown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', (e) => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const k = state.fovY / canvas.clientHeight;
  if (state.cine) { state.cine.offY -= (e.clientX - lastX) * k; state.cine.offP = Math.max(-1.4, Math.min(1.4, state.cine.offP + (e.clientY - lastY) * k)); }
  else turn(-(e.clientX - lastX) * k, -(e.clientY - lastY) * k);
  lastDrag = performance.now(); lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', (e) => { e.preventDefault(); if (!state.cine) state.wheel = Math.max(-1.5, Math.min(1.5, state.wheel - e.deltaY * 0.004)); }, { passive: false });
const KEYMAP = { w: 'fwd', arrowup: 'fwd', s: 'back', arrowdown: 'back', a: 'left', d: 'right', q: 'down', e: 'up', arrowleft: 'turnL', arrowright: 'turnR', shift: 'fast' };
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest && e.target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
  const k = e.key.toLowerCase();
  if (k === 'escape' && ui.start.hidden) { backToStart(); return; }
  if (!ui.start.hidden) { if (KEYMAP[k] && k !== 'shift') choose(false); else return; }
  if (k === 'm') toggleSound();
  if (k === 'r') { placeStart(); if (state.cine) state.cine = { t: 0, paused: false, offY: 0, offP: 0 }; }
  if (state.cine && k === ' ') { e.preventDefault(); state.cine.paused = !state.cine.paused; syncUI(); return; }
  if (KEYMAP[k]) { state.held.add(KEYMAP[k]); if (k !== 'shift') e.preventDefault(); }
});
window.addEventListener('keyup', (e) => state.held.delete(KEYMAP[e.key.toLowerCase()]));
window.addEventListener('blur', () => state.held.clear());

// Free flight: speed in proportion to the size of the sphere you are on (r(ℓ): 1 km at the throat), so the
// approach takes seconds and the throat can still be crossed slowly.
function fly(dt) {
  const s = dt / 1000, h = state.held;
  if (h.has('turnL')) turn(0.8 * s, 0);
  if (h.has('turnR')) turn(-0.8 * s, 0);
  const f = (h.has('fwd') ? 1 : 0) - (h.has('back') ? 1 : 0), st = (h.has('right') ? 1 : 0) - (h.has('left') ? 1 : 0), up = (h.has('up') ? 1 : 0) - (h.has('down') ? 1 : 0);
  state.wheel *= Math.exp(-dt / 300);
  const v = 0.45 * W.radius(state.l, P) * (h.has('fast') ? 3 : 1);
  let d = add(add(scale(state.F, f + state.wheel), right(), st), state.Up, up);
  if (Math.hypot(...d) < 1e-6) return;
  move(scale(d, v * s));
}

// ---------- the trip (~1 min) ----------
// ℓ along a straight line through the throat (eased legs), a turn to look back at the end, a note at each stop.
const TRIP = [
  { t: 0, l: -9 }, { t: 9, l: -4.2 }, { t: 18, l: -1.8 }, { t: 27, l: -0.25 }, { t: 33, l: 0.25 }, { t: 41, l: 3.2 }, { t: 50, l: 6 }, { t: 64, l: 6 },
];
const TRIP_END = 64, TURN = { from: 50, to: 58 };
const NOTES = [
  { t: 1, title: 'A wormhole', body: ['A sphere that shows another place: the light coming through it left a galaxy far away, Gargantua’s. In the film it is 1 km wide.', 'Space pauses · drag to look · Esc for the start screen'] },
  { t: 10, title: 'Saturn, twice', body: ['Light from Saturn passing close to the sphere bends around it: a second, distorted Saturn shows on the far edge, and the stars around the rim stream into a ring (an Einstein ring).'] },
  { t: 19, title: 'Its lensing width', body: ['How hard the edge bends light is set by one number, its lensing width. The film chose 5% of the throat’s radius: a sharp edge (James et al. 2015).'] },
  { t: 28, title: 'The throat', body: ['Inside, a short tunnel 1% of its width long, as in the film. Both skies wrap around you: Saturn’s behind, the other galaxy ahead.'] },
  { t: 38, title: 'The other side', body: ['Gargantua’s galaxy. Nobody has a map of its stars, so this sky is an artistic one; how the wormhole bends it is computed.'] },
  { t: 52, title: 'Looking back', body: ['Saturn’s side, now through the wormhole: the whole of our sky squeezed into one sphere.', 'R to watch again · Esc for the start screen'] },
];
const ease = (x) => x * x * (3 - 2 * x);
function tripTick(dt) {
  const c = state.cine, t0 = c.t;
  if (!c.paused) c.t = Math.min(TRIP_END, c.t + dt / 1000);
  for (const n of NOTES) if (n.t > t0 && n.t <= c.t) showCard(n.title, n.body, 8000);
  let i = 0; while (i < TRIP.length - 2 && c.t > TRIP[i + 1].t) i++;
  const a = TRIP[i], b = TRIP[i + 1], k = ease(Math.min(1, Math.max(0, (c.t - a.t) / (b.t - a.t))));
  state.l = a.l + (b.l - a.l) * k; state.u = [1, 0, 0];
  const back = ease(Math.min(1, Math.max(0, (c.t - TURN.from) / (TURN.to - TURN.from))));
  if (performance.now() - lastDrag > 1500) { const d = Math.exp(-dt / 3000); c.offY *= d; c.offP *= d; }
  // forward (increasing ℓ), tilted up towards Saturn at first, then turned round to look back
  const tilt = 0.16 * (1 - Math.min(1, c.t / 20));
  state.F = [1, 0, 0]; state.Up = [0, 0, 1];
  turn(Math.PI * back + c.offY, tilt + c.offP);
}

// ---------- automatic resolution (weak GPUs, phones) ----------
// Every 1.5 s after the start-screen choice, the median frame time decides: under 28 fps twice in a row → 0.8× the
// resolution (down to 0.35), over 50 fps for 6 s after the last step down → 1.15× (up to 1). The browser scales the
// canvas up. Phones start at 0.8.
let renderScale = phone ? 0.8 : 1;
const perf = { win: [], since: 0, slow: 0, lastDown: -1e9 };
function autoScale(now, dt) {
  if (!ui.start.hidden || document.hidden) { perf.win = []; perf.since = now + 2000; return; }
  if (now < perf.since) return;
  perf.win.push(dt);
  if (now - perf.since < 1500 || perf.win.length < 8) return;
  const med = perf.win.sort((a, b) => a - b)[perf.win.length >> 1];
  perf.win = []; perf.since = now;
  perf.slow = med > 1000 / 28 ? perf.slow + 1 : 0;
  if (perf.slow >= 2) { perf.slow = 0; renderScale = Math.max(0.35, renderScale * 0.8); perf.lastDown = now; }
  else if (med < 1000 / 50 && renderScale < 1 && now - perf.lastDown > 6000) renderScale = Math.min(1, renderScale * 1.15);
}

// ---------- frame ----------
let lastNow = performance.now(), frames = 0, lastFps = lastNow, fps = 0;
function frame(now) {
  const dt = Math.min(Math.max(now - lastNow, 0), 250); lastNow = now;
  if (state.cine) tripTick(dt); else fly(dt);
  autoScale(now, dt);
  const dpr = Math.min(window.devicePixelRatio || 1, phone ? 1.25 : 1.75) * renderScale;
  const w = Math.max(32, Math.round(canvas.clientWidth * dpr)), h = Math.max(32, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  traceRays(state.l);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h);
  gl.useProgram(view.p);
  const tanY = Math.tan(state.fovY / 2), u = view.u;
  gl.uniform2f(u.uRes, w, h); gl.uniform2f(u.uTan, tanY * w / h, tanY);
  gl.uniform3f(u.uU, ...state.u); gl.uniform3f(u.uF, ...state.F); gl.uniform3f(u.uR, ...right()); gl.uniform3f(u.uUp, ...state.Up);
  gl.uniform1f(u.uN, nRays); gl.uniform1f(u.uPix, 2 * tanY / h); gl.uniform1f(u.uTime, now / 1000);
  gl.uniform1f(u.uExposure, 1.1); gl.uniform1f(u.uStars, 1);
  gl.uniformMatrix3fv(u.uGalaxy, false, galaxy);
  gl.uniform3f(u.uSat, ...SAT.dir); gl.uniform3f(u.uSatPole, ...SAT.pole); gl.uniform3f(u.uSun, ...SAT.sun); gl.uniform1f(u.uSatR, SAT.r);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rayTex); gl.uniform1i(u.uRays, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mwTex); gl.uniform1i(u.uMilkyWay, 1);
  draw();
  if (sound.gain && frames % 9 === 0) setGain();
  frames++;
  if (now - lastFps > 500) {
    fps = frames * 1000 / (now - lastFps); frames = 0; lastFps = now; syncUI();
    const where = Math.abs(state.l) <= P.a + 0.02 ? 'inside the throat' : state.l < 0 ? 'Saturn’s side' : 'the far side (Gargantua’s galaxy)';
    const km = Math.max(0, Math.abs(state.l) - P.a);
    hud.textContent = (state.cine ? `trip ${Math.floor(state.cine.t / 60)}:${String(Math.floor(state.cine.t % 60)).padStart(2, '0')} · ` : '')
      + `${where} · ${km.toFixed(km < 10 ? 2 : 1)} km from the throat · ${fps.toFixed(0)} fps`;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

window.__wormhole = { state, P, choose, placeStart, get fps() { return fps; }, get renderScale() { return renderScale; } };
