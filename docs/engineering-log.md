> Copied from the portfolio site this simulation was built in. Paths like `site/tools/gargantua/` mean this
> repo's root; mentions of "the owner" are the author's review notes.

# Gargantua (black hole simulation): engineering log

The plan, decisions and sources are in [plans-prds/gargantua-black-hole.md](../plans-prds/gargantua-black-hole.md).
This file is the build log: what each part does, the hard problems, the options, what was chosen and the measured
numbers. It is the source for the "How it works" page and the interview cheat sheet (stage 6).

**Status:** stages 1–5 done and live; stage 6 (phone, "How it works" page, launch) built 2026-11-17. Listed in
`tools.json` (first card) and the sitemap, with its explainer page `/tools/gargantua/how-it-works/`. Interview cheat
sheet: [gargantua-interview.md](gargantua-interview.md).

## Files

| File | What it is |
| --- | --- |
| `site/tools/gargantua/kerr.js` | The physics, 64-bit: Kerr–Schild metric, dual-number gradients, Hamilton's equations, camera tetrad, horizons / ISCO / photon orbit, reference integrators. Shared by the page and the tests. |
| `site/tools/gargantua/shader.js` | The GPU passes: GEODESIC (the same equations in GLSL; cache, edge refinement and ring-bundle variants), REFINE_MASK / RING_MASK (stencil masks that pack the few pixels to re-trace, see 6e), NOISE (disk texture, once), SHADE (disk, stars, Milky Way), BLOOM down/up, COMPOSITE (tone map, grade). |
| `site/tools/gargantua/colour.js` | Blackbody colour: Planck × CIE 1931 (Wyman 2013 fit) → linear sRGB; the lookup table the shader reads. |
| `site/tools/gargantua/app.js` | WebGL setup, the geodesic cache (low-res while moving, full-res bands while still), passes, GPU-timed frame budget, camera frame (from kerr.js), input, hardware-acceleration check, `__gargantua.profile()` (GPU time per pass). |
| `site/tools/gargantua/sky/milkyway.webp` | The Milky Way: NASA SVS Deep Star Maps 2020 (`milkyway_2020_4k_gal.exr`, Gaia DR2), 4096×2048, 2.7 MB. Credit on the page. Remade with `scripts/gargantua/make-sky.py` (instructions inside). |
| `site/tools/gargantua/index.html`, `style.css` | The tool page (site navbar + footer, full-screen canvas). |
| `tests/gargantua.test.mjs` | Physics vs exact results, GPU step policy vs the precise integrator. |

## How a pixel is computed

1. **Camera frame.** The camera is an observer with a 4-velocity u (stage 1: at rest, u ∝ ∂t, which exists outside
   the ergosphere). `tetrad()` builds its orthonormal frame: forward, right, up, made orthonormal with the
   *spacetime* metric (Gram–Schmidt with g, not the flat dot product). The frame is the same for every pixel, so
   it is computed once per frame in JavaScript and sent lowered (covariant) to the shader.
2. **The ray.** For a pixel direction d in the camera frame, the photon arriving there has p = e₀ − d·e (energy 1
   in the camera's frame). It is lowered with the metric, so p_t = −E is constant and the shader integrates
   x, y, z, p_x, p_y, p_z.
3. **Integration, backward in time.** H = ½ g^μν p_μ p_ν, with dx/dλ = ∂H/∂p and dp/dλ = −∂H/∂x. RK4 with a
   negative step.
4. **Disk.** Every crossing of z = 0 between the ISCO and the outer edge adds the disk's light with its opacity,
   and the ray goes on, so disk images behind thin parts show through.
5. **End.** The ray falls into the hole (black), runs out of steps (black: those rays are skimming the photon
   sphere), or escapes past r_far moving outward. It then looks up the sky in its final direction.
6. **Stars** are drawn from the pixel's lensed footprint (see below), then ACES tone mapping and a dither.

## Hard problems (interview stories)

### 1. Coordinates that don't break at the horizon
- **Problem:** the usual Boyer–Lindquist coordinates are singular at the horizon. Most renderers stop there, and
  the owner wants to fall in.
- **Options:** Boyer–Lindquist plus a special case at the horizon; Kerr–Schild; Doran (river) coordinates.
- **Chosen: Cartesian Kerr–Schild.**
  - g = η + f l⊗l, with l null.
  - The inverse is then exact and cheap, g^μν = η^μν − f l^μ l^ν, so there's no matrix inversion per step.
  - It's smooth through both horizons, and Cartesian coordinates also avoid the pole problems of spherical ones.
- **Cost:** r has to be recovered from x, y, z by a closed-form root. Hamilton's equations also need the metric's
  gradient, which is problem 2.

### 2. Exact metric gradients on the GPU
- **Problem:** dp/dλ needs ∂f/∂x^i and ∂l/∂x^i.
  - Hand-derived formulas are long and error-prone.
  - Finite differences are noisy in 32-bit floats.
- **Chosen: forward-mode automatic differentiation.** A dual number type in GLSL, vec4 = (value, ∂x, ∂y, ∂z),
  with mul, div and sqrt. The metric code is written once, as if it were plain numbers, and the exact gradient
  comes out with it.
- **Checked:** the JavaScript copy is tested against central finite differences (agreement < 10⁻⁶).

### 3. Step size: accuracy vs frame rate
- **Problem:** every pixel integrates up to hundreds of RK4 steps (each step = 4 metric evaluations with
  derivatives). Too-large steps blur the shadow edge, and too-small ones kill the frame rate.
- **Approach:** tune the GPU step rule against a precise 64-bit integrator (20× smaller steps), measuring the
  shadow-edge error in **screen pixels**, which is what a viewer sees.
- **Findings:**
  - At first the rule shrank steps near the horizon, k·(r − r_H). It ran out of step budget and *looked* like a
    22% error at a = 0.999. These coordinates are smooth at the horizon, so steps don't need to shrink there.
    Only a gentle cap remains, (r − r_H) + 0.005.
  - Removing the cap entirely made a = 0 unstable: a step could jump inside the horizon toward the r = 0
    singularity.

  | Rule (near, cap, budget) | Worst shadow edge a ≤ 0.6 | a = 0.999 prograde | Mean steps / ray |
  | --- | --- | --- | --- |
  | 0.035·r, 0.35(r−r_H), 400 | 0.01 px | 23% (budget) | 127 |
  | 0.05·r, 1·(r−r_H), 400 (**chosen**) | 0.01 px | 8.7 px | 119 |
  | 0.05·r, 1·(r−r_H), 250 | 0.81 px | 20.6 px | 105 |

- **Open for stage 2:** at a = 0.999 the prograde edge is still ~9 px off. Near-extremal spin traps edge rays in
  the thin region just outside the horizon for several turns, and they run out of steps. Ideas: a bigger budget
  only for rays that enter r < 1.5, or a step limit by angle instead of length.

### 4. Stars that don't sparkle (DNGR's ray bundles, in real time)
- **Problem:** DNGR traced a tiny beam per pixel (geodesic deviation, Pineault & Roeder 1977) so stars wouldn't
  flicker. The first version here drew each star as a Gaussian blob as wide as the pixel's footprint. Where lensing
  stretches the sky in one direction, that made big smeared ovals.
- **Chosen: ray differentials.** The shader takes the screen-space derivatives of the final sky direction (dFdx,
  dFdy): two vectors J_x, J_y saying how far the sky moves per pixel.
  - A star's offset on the sky is mapped back to *screen pixels* by solving the 2×2 system with J. The star is
    then drawn as a sharp 1-pixel point exactly where lensing puts it, however the sky is stretched.
  - Its brightness is multiplied by the magnification μ = (unlensed pixel solid angle) / |J_x × J_y|, so a star's
    total light is conserved: magnified images are brighter and squeezed ones dimmer, as in reality (capped at
    60×).
- **Gotcha:** derivatives in GLSL are undefined inside divergent control flow (the ray loop ends at different steps
  per pixel). So the derivatives are taken at the top level after the loop, where every pixel of the 2×2 quad
  runs them.

### 5. Disk texture that doesn't wind up
- **Problem:** gas closer in orbits faster (Ω = 1/(r^{3/2} + a)). Rotating a texture by that for minutes shears it
  into ever thinner stripes, until it aliases.
- **Chosen: flow-noise cross-fade.** Two copies of the texture, advected with phases half a period (90 s) apart,
  each faded out before it shears too far.

### 6. Full resolution in real time: the geodesic cache (stage 1b)
- **Problem:** at full resolution the ray tracing ran at ~10 fps, so stage 1 rendered at ~45% and stretched it
  up. The owner: "looks cheap", especially next to film frames.
- **Observation:** the light paths depend only on the camera. The disk's animation, the stars and the colours
  don't change where rays go, only what they find there.
- **Chosen:** split the render.
  1. A **geodesic pass** writes a cache: two RGBA32F render targets (MRT) holding the first two disk crossings
     (r, φ) and the final sky direction plus escaped/captured.
  2. It runs only when the camera changes, at reduced resolution while moving (adapted to ~45 fps). When the
     camera stops, it fills a full-resolution cache in horizontal bands (scissor), sized to the frame time.
  3. Every frame, a cheap **shade pass** re-colours the cache at full resolution: the disk orbiting, the stars,
     the Milky Way.
  - **Result:** the picture is always full-resolution and animated, and the expensive part is only paid when
    the view changes.
- **Upscaling the moving cache:** the 4 neighbouring cache texels are interpolated (with φ unwrapped across the
  2π seam) when they agree on what they hit. At an edge (disk / hole / sky) the four rays are shaded separately
  and their *colours* blended, which also anti-aliases the shadow edge.
- **Gotchas found:**
  - WebGL forbids `?:` on structs.
  - Calling the shading function from five places made the compiled shader so big that the frame rate fell from
    35 to 5 fps. One call site inside a loop fixed it.
  - GPU screen derivatives (dFdx) are undefined inside branches. So the texture's blur level now comes from
    explicit differences between neighbouring cached rays, including the shear term dΩ/dr·t of the orbiting gas.

### 6b. Smooth motion and a frame budget that measures the right thing (owner feedback, 2026-11-11)
- **Symptom 1: blocky, jumbled rings while dragging.**
  - **Cause:** the auto-tuner sized the moving resolution by the *whole* frame time. On the owner's GPU,
    colouring alone took ~28 ms, so it kept cutting the ray resolution down to 15% without ever getting faster.
  - **Fix:** measure the colouring-only time (shadeMs, on frames that trace nothing, plus 4 measuring frames at
    start). Size the ray tracing by the time *above* that, aiming at 16 ms. Area ∝ scale², so the scale moves by
    √(budget / spent). Moving resolution is now 30–70% instead of 15%.
- **Symptom 2: slow sharpening after a zoom.** It had the same cause, through the band size. With the fix, the
  picture is sharp ~0.6 s after a zoom stops.
- **Symptom 3: the ring pauses and jumps back when a zoom ends.**
  - **Cause, part one:** near the photon ring, light bends so hard that interpolating the low-res rays places
    features slightly wrong, so the hard switch to the sharp cache made them jump.
  - **Fix:** a 300 ms cross-fade. The last low-res frame is blitted to a texture and faded out.
  - **Cause, part two:** the zoom easing was per frame. It is now time-based, `1 − e^(−dt/120 ms)`.
- **Sky cache:** the sky (stars + Milky Way) only changes when the cache does, but it was recoloured every frame.
  - It now goes to its own texture only when the cache changes: at half resolution while moving, full when still.
    Every frame only colours the disk over it.
  - **Gotcha:** a single program with a `uniform int` pass switch was *still* slow, because the D3D compiler
    flattened the branch and ran the star code anyway. Two programs from one source (a compile-time constant)
    fixed it.
  - **Gotcha:** the sky pass at first read the texture it was writing (still bound on a sampler). WebGL silently
    refuses that draw, and the sky went black.

  | Owner's PC, 1600×900, full resolution | Before | After |
  | --- | --- | --- |
  | Still (sharp), colouring time per frame | 36–43 ms (23–35 fps) | 6 ms (~140 fps) |
  | Dragging | 12–16 fps at 15–30% res | 26–36 fps at 30–70% res |
  | Zoom end → sharp | several s | ~0.6 s |

### 6c. The photon ring: ray bundles where they matter (owner feedback, 2026-11-11)
- **Symptom:** the thin bright line hugging the shadow was dotted, and flickered while zooming.
- **Cause:** that line is the disk seen after the light looped the hole once or more (higher-order images). It
  is thinner than a pixel, and it gets exponentially thinner with each extra loop. One ray per pixel hits it or
  misses it at random, and the pattern reshuffles with every camera move.
- **Fix: adaptive supersampling.** DNGR solved this for the whole frame with ray bundles. We only need it on the
  ring.
  1. While tracing a pixel's centre ray, the shader measures how long the ray travels inside the photon-sphere
     band, from just inside the prograde to just outside the retrograde photon orbit.
  2. If that "linger" passes a threshold, the pixel is on the ring. It traces 16 more rays spread over the pixel
     (4 while moving).
  3. They store the *average* of the looped crossings: light, light-weighted radius (for the colour), and sky
     seen through. The disk's average profile is used, because no texture survives at that thinness.
  4. Crossings before the loop (the near side of the disk passing in front) keep their full texture: the centre
     ray marks looped crossings with a negative r, and shading only replaces those.
- **Threshold, measured** with kerr.js (r = 28 M, a = 0), as linger against distance from the shadow edge:

  | Offset (px) | −8 | −2 | −0.5 | +0.5 | +2 | +8 | +20 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Linger (M) | 6.5 | 10.4 | 14.3 | 16.8 | 12.4 | 7.0 | 0 |

  - It grows like −ln(distance to the critical impact parameter b_c): linger ≈ 0.15 − 4.2 ln Δb. That is the
    logarithmic divergence of the winding near the unstable photon orbit.
  - A pixel spans Δb ≈ r · (pixel angle), so the threshold is 0.15 − 4.2 ln(3 Δb_pixel): rays within ~3 pixels
    get the bundle.
  - A first, fixed threshold missed ring pixels when pixels were bigger (far away, light mode) and gave a dashed
    ring. The derived one adapts.
- **Result:** a continuous, steady line, sharp or mid-zoom. Only the few pixels along the ring pay for the extra
  rays.
- **Then the owner found it blocky in light mode.** The cache there is 40% (12–25% while moving), so each ring
  sample covers ~2.5 screen pixels: a thick, stair-stepped ring.
  - **Fix: a second geodesic program** (same source, `RING_PASS 1`) traces at *screen* resolution, but returns
    immediately unless a 3×3 neighbourhood of cache texels contains a ring pixel or a looped crossing.
  - In that pass, every pixel whose centre ray has a looped crossing is a ring pixel, so *all* light that went
    around the hole comes from screen-resolution rays and none from the coarse cache.
  - It runs once per cache update: every moving frame with 4 rays, once with 16 when the sharp cache completes.
  - **Result:** light mode's still picture matches full quality around the ring.
- **Then blocky edges while moving** (owner screenshot): wherever neighbouring cache rays disagree (disk / hole
  / sky edges, the thin disk images near the ring, the disk passing in front of the shadow), interpolation can't
  know what lies between two coarse rays.
  - **Fix: the ring pass became a general edge refinement.** At screen resolution, a pixel is re-traced if its
    3×3 cache neighbourhood has any class disagreement (sign of either crossing, escaped or not) or any looped
    light.
  - It writes the same three targets as the cache. ring.w is 0 for not refined, 1 for refined, 2 for a
    supersampled ring pixel.
  - Shading uses a refined pixel's own ray directly and interpolates the cache everywhere else.
  - The cost follows the length of the edges, not the screen area: turning at full quality ~58 fps, light mode
    ~98 fps (owner's PC).
  - **Memory:** caches are ~70 bytes per pixel, so the drawing size is capped at ~3.7 MP (2560×1440 worth).

### 6d. Controls and the light-mode preview
- **Keyboard:**
  - W / S or ↑ / ↓ move towards / away from the hole (exponential in distance, 0.9 e-folds per second).
  - A / D or ← / → turn the view (0.8 rad/s).
  - Held keys act every frame, scaled by the frame time. Arrows don't scroll the page while the simulation is in
    view.
- **`?quality=light`** forces the light mode (40% sharp cache, 25% moving, DPR 1) with a "preview" note, to see
  what machines without graphics acceleration get. It only lowers the resolution. For the real speed of such a
  machine, turn off hardware acceleration in the browser.

### 6e. Performance pass (stage 2 start, 2026-11-12)
The owner asked why browser ports of old games run smoothly and this didn't. Short answer: a game does a few
hundred operations per pixel; here each pixel integrates a curved light path (up to 400 RK4 steps, 4 metric
evaluations each), ~1000× more, on an integrated GPU. But measuring showed most of the time went elsewhere.
- **Tooling first:**
  - `__gargantua.profile()` times every pass at the current view (median of 7 runs, a 1-pixel readPixels after
    each, which waits for the GPU).
  - Reports the share of pixels refined / supersampled.
  - Scratch scripts compare the picture against the pushed version, with the clock frozen so the disk is in the
    same place.
- **Finding 1: lockstep groups.** The GPU runs pixels in groups of ~64 that execute together. The edge
  refinement decided per pixel whether to trace, and only 0.6% did. They were scattered over nearly every group,
  so the whole screen waited on them: 72 ms for 0.6% of the pixels, about what tracing *every* pixel cost.
  - **Fix: a stencil mask.** A cheap pass (REFINE_MASK) marks those pixels in the stencil buffer, and the tracer
    runs with the stencil test on. The GPU drops unmarked pixels before the shader starts and packs the marked
    ones into full groups. Refinement 3× faster.
- **Finding 2: bundles nobody read.** Both cache traces still shot the 17-ray photon-ring bundle. Shading never
  reads the cache's ring average, because the refinement re-traces every ring pixel at screen resolution anyway.
  - The cache now only *flags* ring pixels. Sharp trace 3-6× faster (30 M: 235 → 60 ms, 12 M: 533 → 79 ms).
- **Finding 3: the same lockstep problem one level down.** Inside the refinement, the 5- or 17-ray ring pixels
  held up the 1-ray edge pixels in their groups.
  - **Fix:** split into a centre-ray pass and a bundle pass (RING_PASS 2). A second stencil value (RING_MASK,
    stencil 2, from a stencil-only framebuffer, because it reads the refinement's ring target) packs only ring
    pixels for the bundle. Refinement while moving: 32 → 18 ms at 30 M.
- **Finding 4: anisotropic filtering.** 16× on the disk texture cost half of the per-frame disk colouring. 4×
  looks the same (side-by-side crops; 1× is visibly softer), so 20 → 9 ms.
- **Finding 5: the frame budget measured the wrong thing (bug).** It guessed the shading cost from frame times,
  but the browser delivers frames in bursts, and the guess collapsed to 0.2 ms (real: 17 ms). Sharpening then
  crawled at its minimum band (1.2 s to sharpen).
  - **Fix:** GPU timer queries (EXT_disjoint_timer_query_webgl2) around the trace, refinement and shading,
    read back a few frames later. The moving scale and the band size are computed from the measured cost per
    pixel.
  - **Without the timer** (Firefox, Safari): steer the whole frame time towards 60 fps on a smoothed frame time
    in small steps. Frame times lag 2-3 frames, and bigger steps made the scale swing between its limits (logged
    per frame). A one-off synchronous shading measurement was tried and dropped: it read 14 ms in one run and
    35 in the next.
- **Tried and rejected: a smarter integrator.** Benchmarked against the 64-bit reference (880 rays per view,
  error in pixels at 900 px tall):
  - Dormand–Prince 5(4) with error control (MATLAB's ode45) saved at most ~25% of the work for the same
    accuracy, and was worse close up. The hand-tuned RK4 rule is already near optimal.
  - Uniformly bigger RK4 steps fail fast (error grows with the 4th power of the step).
  - Today's rule is far more accurate than needed (99% of rays within 0.01 px), but only ~2× in step size is
    available before p99 passes 0.2 px.
- **Found for later (a = 0.999, 8 M):** 307 of 880 rays exhaust the 400-step budget, spiralling near the
  horizon with frame dragging. The "inside the innermost photon orbit and moving inward = captured" exit is
  exact but doesn't help there (that orbit is at 1.09 M, the horizon at 1.045 M). Stage 2 problem.

  | Owner's PC (AMD Radeon integrated), 1600×900, full quality | Before | After |
  | --- | --- | --- |
  | Moving, 90 / 30 / 12 M | 24-29 / 9-12 / 8 fps | 37-51 / 38-55 / 41-57 fps |
  | Worst single frame while moving | 157-485 ms | 24-85 ms |
  | Still (sharp) | 36-95 fps | 59-133 fps |
  | Sharp trace, 12 M (one-off, banded) | 533 ms | 79 ms |
  | Refinement while moving, 12 M | 162 ms | 36 ms |
  | Light mode, moving, 30 M | n/a | 31 fps |
  | No GPU timer (fallback), moving, 30 M | n/a | 29 fps |

  Ranges are two runs. The owner's own Chrome shares the GPU, so absolute numbers move between runs; the ratios
  hold.

### 6f. Spin up to a = 0.999 (stage 2, 2026-11-12)
The default is now the film's a = 0.6; `?spin=0…0.999` previews others until Reality mode's slider (stage 5).
Turning spin up exposed four problems, each fixed and measured.
- **Captured rays crawling at the horizon.** At a = 0.999 and 8 M, 35% of rays used up their 400 steps circling
  just outside the horizon (r ≈ 1.13 M, horizon 1.045 M), dragged around by the spin.
  - **Fix: an exact capture test.** From the ray's constants of motion (energy, angular momentum and Carter's
    constant Q), the radial potential R(r) = [E(r²+a²) − aL]² − Δ[(L − aE)² + Q] says where a photon can turn.
    Moving inward with R > 0 all the way to the horizon, it is captured (`kerr.js` `constants()`, `captured()`).
  - Q in Kerr–Schild coordinates: they differ from Boyer–Lindquist only by functions of r in t and φ, so p_θ and
    p_φ are the same. Tested: Q stays fixed to ~1e-6 along precise rays.
  - Checked against the precise integrator on 1,877 rays (3 spins × 3 camera positions): no disagreement.
  - On the GPU it runs once per ray before the step loop (the test only depends on min(camera radius, check
    radius)). A captured ray then ends inside the disk's inner edge, after its 2 recorded crossings, or half a
    turn after capture. Before the half-turn rule, captured rays still took 263 steps each; after it, 53.
- **The prograde shadow edge 9.7 px off at a = 0.999.** Rays near that edge circle the hole many times before
  escaping (near-extremal prograde orbits are only weakly unstable), and 400 steps weren't enough. The step budget
  now depends on spin: 400 up to a = 0.9, rising to 1200 at 0.999 (`stepBudget()`). Edge error 9.7 → 1.2 px
  (test: < 1.5 px). The step-size limit near the horizon made no difference (measured).
- **The GPU reset (context lost) at a = 0.999 close up,** in the pushed version too.
  - The photon-ring test (time spent in a band around the photon orbits) flagged 52% of the screen at 15 M: that
    band reaches 5 M at high spin, so even direct views of the inner disk counted. One refinement draw took 1.7 s,
    and Windows resets the GPU when a draw runs past ~2 s.
  - **Fix: the angle light sweeps around the hole.** A straight pass sweeps under half a turn; near the critical
    curve it grows by ~1 rad per e-fold closer (the photon sphere's instability). Measured against distance from
    the shadow edge in pixels at a = 0 and 0.999, 12 and 30 M; threshold 3.0 − ln Δb_pixel flags the pixels within
    ~2–3 px, as before. "Looped" disk images = crossings after 1.35π. A captured ray's winding stops counting.
  - Result: ring pixels at a = 0.999 are 1–10% of the screen (a = 0 unchanged at 0.1–1.6%).
  - Safety net: the final sharp refinement is drawn in 6 strips with a flush between, and ring-bundle rays get at
    most 400 steps.
- **The disk turned into a smooth orange blob.** Its look (temperature, brightness, texture scale) was laid out in
  units of the ISCO, which moves from 6 M to 1.2 M with spin. It now uses a fixed 6 M (the look the owner approved);
  spin only moves the inner edge. Brightness rises inward to 0.85 × 6 M at most (0.6 washed out the shadow).

### 6g. The camera: a real moving observer, free flight, readouts (stage 2, 2026-11-12)
- **The film's observer.** The camera now moves relative to the locally non-rotating observer (ZAMO, Bardeen 1972),
  as in James et al. 2015. The static observer it replaced can't exist close to a spinning hole (inside the
  ergosphere nothing can stay still). `kerr.js` `zamo()`: u_μ = −α ∂_μ t_BL, with dt_BL = dt − (2r/Δ) dr in these
  coordinates. Tested: unit timelike, zero angular momentum, the exact Kerr lapse α, the static observer at a = 0.
- **Speed is real.** The camera's frame is the ZAMO frame Lorentz-boosted by its velocity (`cameraTetrad()`), so
  aberration bends the view. Tested exactly: flying at the hole at 0.5 c shrinks the shadow to the angle special
  relativity gives, cos θ′ = (cos θ + v)/(1 + v cos θ), within 1e-6 rad.
- **Free flight** (`app.js` `fly()`): W/S or ↑/↓ forward/back, A/D strafe, Q/E down/up, Shift faster (0.3 c →
  0.8 c), wheel adds speed that dies away, ←/→ or drag to look, R reset. The engine reaches a new speed in ~1 s
  and brakes to rest when keys are released.
  - The "magic engine": time is compressed in proportion to the distance (2.2 M of coordinate time per second per
    M), so 1000 M takes seconds. Position moves by the camera's own coordinate velocity u^i/u^t.
  - Limits: 1000 M out; 1.6 × the horizon in, until stage 3 opens the fall. At a limit, the part of the velocity
    across it is dropped.
  - No Ctrl for "slower", as first planned: Ctrl+W closes the tab and Ctrl+S opens a save dialog.
- **Readouts** (HUD): distance (M and AU), speed relative to the local ZAMO, and "1 h here = X on Earth",
  X = γ/α (Earth taken as far away). At 90 M at rest: 1.011 h; at 2.9 M and 0.53 c: 1.88 h.
- **Owner review: "light and stars rotate while zooming, nothing moves when I stop, I don't see it spinning".**
  - Suspected: the camera's axes came from the Kerr–Schild x, y, z directions, and that coordinate azimuth twists
    against Boyer–Lindquist's (19° between 90 M and 2.9 M at a = 0.6). Measured instead of assumed: the shadow
    lands within 0.5° of where the standard Boyer–Lindquist-aligned ZAMO frame (`zamoAxes()`) puts it, so the
    twist is in where points are labelled, not in the view. The standard frame is kept anyway (it's the paper's).
  - So the swirl is real: a spinning hole drags space around, and the lensed sky twists around the shadow as you
    move (without spin it only streams straight outwards). It stops when you stop, because the light paths only
    change when the camera does. A black hole's spin has no surface to watch; it shows in that swirl, the
    lopsided shadow and the disk.
  - The disk *was* orbiting, just too slowly to notice (its streaks run along its motion): its clock went from
    2 to 8 M per second (inner edge ~6 s per turn, 10 M ~25 s).
- Frame rate flying in from 90 to 9 M (owner's PC): ~28–31 fps (a = 0.6), 23 at a = 0.999; turning in place:
  ~40–55. The limit is now the screen-resolution refinement and shading, not the trace (light mode is no faster).

### 6h. Falling in (stage 3, 2026-11-13)
- **Engines off.** At the inner flight limit (1.6 × the horizon) a card says to keep flying in; pushing in for
  0.7 s (within 3% of the limit, steering inward) switches the engine off. From then on the camera is its own
  timelike geodesic (`kerr.js` `fallStep()`: the same Hamiltonian, p·p = −1, proper time), starting with the
  4-velocity it had. It crosses the horizon because the equations say so; inside, r is a time and only decreases.
  Flying stops; looking around still works. R starts again (a note says no real ship could leave).
  - Tested exactly: dropped from rest at 6 M (a = 0), the proper time to the horizon and on follows the textbook
    cycloid τ = √(R³/2M)(η + sin η)/2 within 0.002 M, and H stays −½ through the horizon. At a = 0.6 it passes
    both horizons with H within 1e-4.
  - Pace: ~0.06–0.28 M of proper time per second, slower near the end: ~25 s from engines off to the end.
    The HUD shows your clock (1 M = 492 s for Gargantua's 10⁸ solar masses): ~50 min from 2.9 M to the inner
    horizon at a = 0.6.
- **Rays from inside.** Traced back in time, every ray climbs outward. Light that fell in from outside crosses
  the outer horizon (back out, in the trace) and reaches the disk or sky. The rest creeps up to the horizon from
  inside and never crosses: in the ideal eternal hole it would come from a white hole, in a real one from the
  collapsing star and what fell in before. Not modelled, drawn black ("past").
  - Rule, the same in kerr.js and the shader: within 0.2% under the horizon, with the outward speed almost stopped
    (dr per unit path < 0.02) for 3 steps. A first, step-count rule made the precise integrator (tiny steps) call
    crossing rays "past"; this one doesn't depend on step size.
  - Also needed: step limits near both horizons use |r − r±| (inside, r − r₊ is negative); the capture test is
    off for a camera inside; "into the hole" only counts for rays that have been outside.
  - Found on the way: the JS reference's capture check ran on a ray's very first step, before knowing whether
    it was heading inward (no earlier test had a camera close enough). The shader didn't have that bug.
  - Tested: from a camera at 1 M inside (a = 0.6), the GPU step policy and the precise integrator agree on the
    fate of every ray but at most 2 of 40, and escape directions within 1 px.
- **What you see, measured (not assumed).** The plan expected "the light behind you blueshifts more and more".
  For the free-faller these coordinates follow, it's subtler: straight back the outside universe is *redshifted*
  (g = 0.9 at 2 M, 0.4 near the inner horizon: you race away from the light falling in after you), and around the
  edge of that view, 60–130° off straight back, it is blueshifted 2–8× into a bright rim. Ahead there's only the
  black "past". (Tested: g < 0.8 straight back and > 1.3 at 90° from inside at 1.2 M.)
  - So the view turns around by itself after engines-off (8 s) and widens from 45° to 90° to take in the rim,
    until you look around yourself.
  - Starlight shift, shown only in the fall (the film shows none; it fades in over 2.5 s): g = 1/(−p_t) for each
    escaping ray (the camera measures the photon's energy as 1), stored in the cache's sky.w = 1 + g. Brightness
    × g⁴ (Liouville, capped at 40), Milky Way tinted like a sun-like source at g·5800 K, stars at g·T.
- **The ending.** At the inner horizon (no spin: near the centre), the camera stops and a note says this is
  where known physics stops: in the ideal solution space goes on (ring singularity, other universes); for a real
  hole, streams that fell in at different times pass through each other near this horizon and their energy grows
  without limit (mass inflation, Poisson & Israel 1990; Hamilton & Avelino 2010); what comes next needs quantum
  gravity. No scenery is invented past it.
- **Owner review (2026-11-13): "the camera tilts back or sideways near the horizon, but no note appears".**
  - The tilt was the fall starting and the view turning round; the notes didn't show. Reproduced: holding W all the
    way in, the limit note showed for under a second, then "engines off" 0.7 s later and the turn 1.5 s after
    that. A stale cached stylesheet (the dev server sent no cache headers) would also hide the notes behind the
    canvas. Fixes: the dev server sends `Cache-Control: no-cache` like production; engines-off now needs 2 s of
    pushing at the limit, with a countdown in the note (letting go cancels); the turn waits 3.5 s; notes sit
    above the canvas (z-index).
  - Found while fixing it: the ship reached the limit with the hole up to ~60° off its view (the sideways drift). The "pushing inward" test had
    compared steering in the ZAMO's sky-aligned axes with the Kerr–Schild direction to the hole (19° apart there),
    and at the limit the camera slid along it until the countdown reset. Now: the true radial direction in the
    ZAMO's axes (`localRadial()`), and the engine holds the camera still while you push.
- **Owner review, round 2: "as I reach the dark centre it tilts my camera and moves me up sideways; I can't get
  inside".** Cause: the time compression. The velocity is measured against the local ZAMOs, which are themselves
  carried round the hole by frame dragging (dφ/dt = ω); with time compressed TIME_SCALE·r-fold, that swept a ship
  flying straight at the hole round and past it: 22° off by the inner limit at a = 0.6, 58° at 0.999 (measured
  in the page, hold W from 90 M). Over real time the drift would be tiny.
  - **Fix: the engine holds against the dragging** (what hovering means; a real ship could, outside the
    ergosphere). The ship now moves in Boyer–Lindquist r, θ, φ by its velocity along the ZAMO's e_r, e_θ, e_φ
    without the ω term: dr = α v_r √(Δ/Σ) dT, dθ = α v_θ/√Σ dT, dφ = α v_φ/ϖ dT. "At rest" is at rest against the
    stars, the limits are plain limits on r, and the hole stays dead ahead: 0° off at both spins.
  - The camera's frame (aberration, clock readout) still uses the velocity relative to the ZAMO.
- **Owner question: "why is this one circular line separate?"** That thin line at the shadow's edge is the photon
  ring: light from the disk that went around the hole once or more. The darker gap between it and the disk is real:
  the disk ends at its inner edge (the ISCO; gas closer in plunges), so nothing glows there. It's widest on the
  left, where the spin flattens the shadow.
- Frame rate during the fall (owner's PC): 37–46 fps (a = 0 / 0.6), 34 at a = 0.999 (one 0.5 s hitch when the
  final picture sharpens there).

### 6i. The film look (stage 4, 2026-11-13)
- Tuned side by side against the owner's film frame (`ref1.jpg`, the Miller's planet shot; reference only, never
  committed or shipped), with the clock frozen so frames are comparable, in four rounds of three variants.
- What made it read as film, and what changed (all in `LOOK`, presets in `app.js`):
  - **Veiling glare:** a lens scatters a bright source's light over the frame, a wide warm haze around the disk
    that lifts the dark. New composite term from a coarse bloom level (mip 4+), `haze` 0.3, warm tint.
  - **Exposure for the disk:** exposure 1.1 → 0.8, stars 1 → 0.35 and Milky Way 0.16 → 0.08 (the film's camera,
    exposed for the disk, shows almost none; kept at a third because the owner liked the star field).
  - **Disk palette:** temperature 6800 → 4500 K at 6 M (cap 6500 K): amber, white only near the hole; light gathers
    nearer the hole (brightness ∝ x^−2.7, was −2.3); denser dust gaps (density^1.3); saturation 0.65 in the grade.
  - No horizontal lens streak: the film frame has none; its "flare" is the glare and glow.
- **Owner review (2026-11-14): too dark.** The owner preferred the brighter, punchier stage 1b look, so the default
  went back toward it: crisp's exposure 1.1, full stars and Milky Way, disk 6400 K (cap 7400 K), falloff 2.3,
  saturation 1.15 (more colour than crisp), haze only 0.05 (a faint glow). The film-matched values above are history.
- `?look=crisp` keeps the stage 1b look for comparison. In the fall the sky eases back to the crisp brightness
  (the camera re-exposes once the disk is behind it); without it the blueshifted rim was barely visible.
- The fall's turn is now 120°, not straight back: that frames the edge, the universe with its bright blueshifted
  rim on one side and the darkness ahead on the other (straight back held only the dim, redshifted middle).

### 6j. Film / Reality, Cinematic, start screen (stage 5, 2026-11-14)
- **Menu** (top right): Cinematic / Free flight, Film / Reality, a spin slider (Reality only), How it works (the grid
  sky, also C). A **start screen** offers "Take the trip" or "Fly yourself" (the picture renders behind it).
  `?mode=reality`, or `?spin=0…0.999` (Reality at that spin), opens in Reality.
- **Reality mode:** the true frequency shift of the disk's light, g = 1 / (u^t (E − Ω L)) (`kerr.js diskShift`):
  Doppler and gravitational shift together, for gas on prograde circular orbits. E = −p_t and L = p_φ are conserved
  along the ray, so the shading pass gets them from each pixel's own direction and the camera tetrad (the same
  formula as the tracer's first step), with no extra cache channel. Seen colour: a blackbody at g·T; brightness g⁴
  (Liouville), capped at 4 (`LOOK.shiftCap`: past that the camera saturates anyway; with 40, the inner disk of a
  fast-spinning hole washed the frame out white). The starlight's shift is on all the time in Reality, not only in
  the fall. Spin slider 0–0.999, default 0.999; changing it keeps the camera at the same Boyer–Lindquist r, θ, φ.
  Film keeps a = 0.6 and no shift on the disk, as the film did. Tested: the gas 4-velocity is unit timelike, the
  face-on a = 0 shift is √(1 − 3/r), and the side coming towards a far camera is blueshifted (g > 1.15 at 10 M) and
  the far side redshifted (g < 0.75).
- **Cinematic:** a fixed ~3.3-minute path from 90 M down to the closest the engine can hover, then the engines go
  off and the real fall (stage 3) takes over to the inner horizon. Keyframes in Boyer–Lindquist (log r, θ, φ) and a
  look offset, joined by a Hermite curve in time (Catmull–Rom tangents). The camera moves along it under free
  flight's rules (time compressed, no frame dragging), its velocity from the path (≤ 0.03 c), so aberration is real.
  Notes at each stop: the bending sky, the Einstein ring, the far side of the disk, Film vs Reality (the text
  depends on the mode), the shadow, then Miller's planet, the last stable orbit and the photon shell. Those three
  move with the spin and are sorted by radius; ones below the closest hover point (fast spin) are explained from
  just above. Close in the shadow fills most of the sky (over 140° of it from 4 M), so those keys look at the
  shadow's edge (sin ψ = b_c α / √(r² + a²), exact at a = 0) with a wider view (up to 65°).
  Controls: Space pauses, ← / → (held, 6×) or the wheel rewind / skip, drag looks around (eases back to the path),
  Esc or a flight key leaves to free flight from where you are, R restarts. In the fall the wheel back leaves it
  and returns to the path; nothing skips forward through it.
- **Speed while moving** (the trip never stops, so it is always the moving picture): close in at a = 0.999 the
  camera sits inside the photon shell and ~12% of the screen is photon ring; the moving refinement cost 90–250 ms
  per frame (2–8 fps). Now, only while moving: the refinement runs at a resolution sized from its measured GPU time
  (`refScale`, 0.4–1, into the corner of its buffer, read back scaled), traced rays stop at 600 steps
  (`MOVING_BUDGET`; the still picture keeps up to 1200), and when even that is too slow the ring bundle drops from
  4 rays to 1. The bundle was ~3/4 of the cost (32 of 43 ms). Trip frame rate: Film 39–56 fps, Reality (a = 0.999)
  17–53 fps (was 2–44). `__gargantua.profile()` now also reports `traceMove`, `refineMove`, `refineMoveNoBundle`.
- **Owner review (2026-11-14): a clean screen.** The menu went: the visitor chooses once on the start screen (take
  the trip or fly yourself) and refreshes the page to choose again. The look is Film; Reality stays in the code
  for checks (`?mode=reality`, `?spin=…`, `__gargantua.setMode`). In the trip, Esc / flight keys no longer leave it.
  The trip was too long (~3.3 min + the fall): now 1:20 of path (notes every ~9 s, the intro note dropped) + the
  fall ≈ 1:45 (measured), under the owner's 2-minute limit.
- **Owner review 2 (2026-11-14):** Esc from either mode returns to the start screen. The trip "stayed blurry all the
  way": the camera never stopped, so every frame was the low-resolution moving picture. Now it glides between its 8
  stops (6 s each, easing in and out: Hermite with zero tangents) and holds still 3.5 s at each while its note shows;
  measured, every stop is sharp within ~1 s and stays sharp to the end of the hold. Between stops it aims at 40 fps
  instead of 60 (more resolution; measured ~37 fps). Path 1:22 + the fall ≈ 1:47.
- **Sound:** the owner's NASA black hole sonification (`sound/blackhole.mp3`, 30 s, 470 KB), looped with Web Audio:
  each copy fades in over the last 2.5 s of the one before, so there is no gap or click at the joint (an
  `<audio loop>` leaves both). Starts with the start-screen choice (browsers allow sound only after a click or key);
  M or the round button bottom right mutes; paused while the tab is hidden. Credit: "Black hole sound: NASA".
  Volume follows the distance (owner, 2026-11-14): full at 4 M and inside, falling with log r to a tenth at 300 M
  (measured gain: 0.08 at 600 M, 0.22 at the 90 M start, 0.43 at 30 M, 0.60 at 12 M, 0.80 at 4 M), eased over ~1 s.

### 6k. Phone, "How it works", launch (stage 6, 2026-11-17)
- **Phones / tablets** (touch only, `(pointer: coarse)` and no fine pointer; `?quality=phone` previews it): 60% resolution
  when still, a lower moving resolution, device pixels capped at 1.5×. Drag looks around (pointer events, as with a
  mouse). Round buttons bottom right, only on touch devices: hold ▲ / ▼ to fly, pause (trip), back to the start screen.
  At phone width the HUD is smaller and the notes sit at the top (they covered the readout at the bottom). Tested in
  phone emulation (390×844, touch): flying, pause, back, no sideways scroll. Not yet on a real phone GPU.
- **"How it works" page** `site/tools/gargantua/how-it-works/` (the case-study layout, `case-studies.css`, plus its
  `style.css`): four checked numbers, eight short chapters (spacetime, light, camera, disk with Film vs Reality images,
  sky, falling in, real time with the speed table, the checks table), sources. Its three images are frames of the
  simulation at 26 M (film, reality at a = 0.999, grid sky), 1200×675 JPEG ~95 KB each. Linked from the start screen.
- **Launch:** first entry in `tools.json` (icon `bi-hurricane`, a gold gradient), both pages in the sitemap. The
  browser smoke test now opens Gargantua with a software GPU (start screen, then "Fly yourself" renders) and the
  explainer page (sections and images load).

### 6l. Weak computers and phones (stage 7, 2026-11-18)
- **Owner: 3-4 fps on a phone.** Measured with a software GPU (SwiftShader, 960×600) standing in for a weak one:
  2-3 fps; the sharp picture's ring bundle took 18 s in one frame, the moving edge refinement 0.2-2.4 s, the disk
  pass 0.1-0.4 s, the bloom ~50 ms. (The owner's integrated AMD GPU: 97-133 fps still.)
- **Automatic quality** (`autoQuality` in app.js), two levers:
  - **lite**: the disk without its cloud texture layer (3 reads, not 4 per phase), two star layers instead of three,
    no ring bundle (`uBundle` 0: ring pixels get one ray, like edges), no edge refinement while moving, and the
    sharp refinement spread over 8 frames (one strip each; in one frame it froze a weak GPU ~0.3 s). On from the
    start on phones and without a GPU; on elsewhere after two 1.5 s windows in a row under 22 fps. It stays on.
    (At a single window under 26 fps, the trip's brief dips switched the owner's fast PC to lite.)
  - **renderScale** 0.4-1: the whole picture's resolution, the browser scales the canvas up. Phones start at 0.75.
    Slow → 0.8×; over 50 fps for 8 s after the last step down → 1.15×. Counted only after the start-screen choice.
- Software GPU: 2-3 → 8-15 fps (lite, scale 0.4), no more multi-second freezes. Owner's PC: unchanged (full quality,
  never switched to lite in a full trip). A real phone GPU is far faster than the software stand-in.

### 6m. A sharper picture while moving (stage 7, 2026-11-18)
- What went soft while moving (side by side with the still picture): stars became blobs (the sky pass ran at half
  resolution), the disk's fine streaks blurred, the photon ring's edge was blocky (refined at 0.4 scale, one ray).
- **Sky at 3/4 resolution while moving** (`MOVE_SKY`): crisp stars; full resolution cost ~5 ms more per moving
  frame on the owner's integrated GPU, 3/4 about 3. Lite keeps 1/2.
- **Disk texture read finer while moving** (`MOVE_GRAD` 0.6, uniform `uGradK` on the texture derivatives): 0.35
  broke the dust lanes into blocks.
- **TAA** (shader.js `TAA`, not in lite): each moving frame's low-resolution rays (and the edge-refinement rays) sit
  at a different spot inside their texels (Halton 2,3, 8 positions, via `uRect` in the tracer and `uJit` in the
  shading), and each frame is blended with the history (weight 0.75), clamped to its 3×3 neighbourhood's range in
  the new frame so turns and new features don't smear. Two ping-pong history buffers; post-processing reads
  `out.show`. History dropped after a jump (move > 3% of r or turn > 0.15 rad in a frame). ~1 ms per frame.
- Result: the moving picture keeps the disk's streaks and the stars almost like the still one; the ring's edge is
  smoother. Cost on the owner's PC: the trip's glides ~25–30 fps (were 31–36); stops unchanged (50–64).

### 6n. The wormhole, Gargantua's sibling (stage 7, 2026-11-19)
- **Where:** `site/tools/gargantua/wormhole/` (`worm.js` physics, `worm-shader.js` GPU passes, `wormhole.js` page,
  `index.html`), not a separate tool: linked from Gargantua's start screen ("The wormhole") and back, a chapter on the
  "How it works" page (`#wormhole`), its own sitemap line, the same Tools card. Same look, controls, sound, Esc, touch
  buttons as Gargantua (shares its `style.css`; no footer).
- **Physics:** James, von Tunzelmann, Franklin & Thorne 2015, "Visualizing Interstellar's Wormhole" (Am. J. Phys. 83,
  486, arXiv:1502.03809): the Dneg wormhole with gravity off, ds² = −dt² + dℓ² + r(ℓ)²dΩ², r = ρ inside |ℓ| ≤ a, else
  ρ + M[x arctan x − ½ ln(1+x²)], x = 2(|ℓ|−a)/(πM), W = 1.42953 M. The film's numbers: ρ = 1 km, 2a = 0.01ρ, W = 0.05ρ.
  ℓ < 0 is Saturn's side, ℓ > 0 Gargantua's galaxy (as in the paper).
- **One angle per pixel:** spherical symmetry keeps each ray in a plane, and from a given ℓ it depends only on the angle
  ψ between its direction and e_ℓ. A GPU pass traces a table of 2048 rays (RK4 in ℓ, φ, p_ℓ with b = r(ℓ0) sin ψ
  conserved; step 0.004 + 0.04·(|ℓ| − a)), redone only when ℓ changes; each pixel looks its ray up. 120-165 fps on the
  owner's integrated GPU.
- **Camera frame continuous through the throat:** directions are kept in the local frame (e_ℓ, e_θ, e_φ) written as
  3-vectors (e_ℓ ↦ u, the camera's place on the sphere). Flying straight in on Saturn's side comes out flying
  straight away on the far side with no special case; moving sideways turns u by distance / r(ℓ) and carries the axes.
- **Skies:** Saturn's side: the NASA Milky Way (its bright core placed behind the wormhole from the start, so the
  Einstein ring around the edge is bright; the sky straight behind was dark at first) and a procedural Saturn (lit
  ball with bands, rings at a 24° opening angle with the Cassini division, the planet's shadow on the rings), placed on
  the sky (treated as far away). The far side: an artistic starfield and glowing gas (no map of that galaxy exists;
  the card says so).
- **Trip** ~1 min: 9 km → through the throat → 6 km on the far side → turn to look back, 6 notes. Free flight: speed
  ∝ r(ℓ).
- **Weak GPUs:** software GPU 5 fps at first: the ray table was most of the frame. Now the resolution steps down
  automatically (0.35-1, like Gargantua's) and the table shrinks with it (2048 → 1024 → 512 rays); rays circling the
  throat more than ~twice stop at 3000 steps (a hair-thin ring, left dark). Software GPU: ~20 fps.
- **Tests** (`tests/wormhole.test.mjs`): the shape (film parameters, continuity, dr/dℓ vs finite differences, → 1 far
  out), light-like rays (p_ℓ² + b²/r² = 1 to 10⁻⁵), the exact edge (b < ρ goes through, b > ρ turns back, both sides),
  straight lines far away (< 0.1° from 2000 ρ), the step rule vs 10× smaller steps (< 2·10⁻⁴ rad).

### 7. The look (stage 1b)
- **HDR + bloom:** the scene renders to RGBA16F. A 7-level down/up chain (13-tap downsample, Jimenez 2014
  "Next generation post processing in Call of Duty", Karis-averaged first level so single bright pixels don't
  flicker; 3×3 tent upsample) gives the soft glow the film's IMAX lenses put around every bright edge. It is
  mixed in energy-conserving: light is spread, not added.
- **Colour from physics:** the disk's colour is the blackbody colour of its temperature, from colour.js (Planck's
  law integrated against the CIE 1931 colour-matching functions). The temperature falls outward (~6800 K × x^-0.8,
  x = r/ISCO): white-hot inside, orange, then brown, as in ref1. The film used one 4500 K for the whole disk, but
  its frames read as a gradient because of exposure.
- **Detail:**
  - The disk is painted from a 1024² tileable noise texture made once on the GPU: fine fbm, ridged filaments,
    clumps and dust, mipmapped with 16× anisotropic filtering, because the texture is stretched along the orbits.
  - It is sampled at four scales with explicit gradients.
  - Contrast is stretched, because fbm sits too evenly around 0.5.
  - The outer disk breaks up into patchy wisps with dark dust lanes.
- **Real sky:**
  - NASA SVS Deep Star Maps 2020 Milky Way (Gaia DR2) in galactic coordinates, rotated so the galactic centre
    sits just behind the hole and gets lensed into a ring around the shadow.
  - It is sampled with `textureGrad` from the lensed footprint, so it is sharp where lensing magnifies it.
  - A black point removes the faint all-sky haze.
- **Grade:** ACES tone mapping, vignette, light film grain.

### 8. Hardware acceleration
- WebGL *is* the GPU path. If the visitor turned off "Use graphics acceleration when available", Chrome/Edge
  emulate the GPU on the processor (SwiftShader) and it crawls.
- The page reads the renderer name (`WEBGL_debug_renderer_info`). On SwiftShader / llvmpipe it shows a tip on how
  to turn acceleration on and renders lighter (40% full cache, 25% moving cache, DPR 1). Tested with
  `--use-angle=swiftshader`: the tip shows and the picture still renders, slowly.

## Measurements

- **Shadow (non-spinning):** critical impact parameter 5.19615 M = √27 M (precise integrator, both sides).
- **Shadow (a = 0.6), equatorial edges:** 3.83849 (prograde) and 6.31565 (retrograde). The exact Kerr values are
  3.83850 and 6.31568.
- **On screen** (r = 30 M, 45° field of view, 900 px tall): the shadow measures ~370 px across. The formula
  (sin α = √27/r · √(1 − 2/r)) predicts 369 px.
- **Frame rate** (owner's PC, Chrome, 1600×900 window, render scale auto): ~32–50 fps at 43–51% resolution, and
  56 fps at 62% when far away.

## Stage 1 checklist

- [x] Kerr–Schild Hamiltonian integrator with dual-number gradients (spin-ready; stage 1 runs at a = 0).
- [x] Shadow = √27 M test.
- [x] Thin disk (4500 K, streaky, ISCO inner edge, ragged outer edge).
- [x] Procedural star field with lensed point stars; faint Milky Way band; checkerboard sky (C).
- [x] Mouse look, wheel distance (3.2–1000 M), adaptive resolution.
- [x] Owner review: "looks cheap" → stage 1b.

## Stage 1b checklist (look foundation)

- [x] Geodesic cache (MRT RGBA32F), low-res while moving, full-res bands while still; edge-aware upscaling.
- [x] HDR + 7-level bloom, ACES, vignette, grain.
- [x] Blackbody colour LUT (Planck × CIE 1931), temperature gradient across the disk.
- [x] Detailed disk texture (GPU noise, mipmaps, anisotropic filtering, explicit gradients), dust lanes, wisps.
- [x] Real Milky Way (NASA SVS / Gaia), lensed, credited.
- [x] Hardware-acceleration check → tip + light mode.
- Frame rate on the owner's PC (AMD Radeon integrated, 1600×900, full resolution): ~35–55 fps sharp, ~33–45 while
  moving.
- [ ] Owner review against ref1.jpg / ref2.jpg.
