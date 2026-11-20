# Interstellar simulation: Gargantua and the wormhole

The black hole and the wormhole from *Interstellar*, rendered live in the browser from the equations of general
relativity, using the same physics as the film's visual effects team. Every pixel is a ray of light traced through
curved spacetime on the GPU (WebGL2). No libraries, no build step.

**Live:** https://dablewcodes.site/tools/gargantua/ · the wormhole: https://dablewcodes.site/tools/gargantua/wormhole/ ·
how it works: https://dablewcodes.site/tools/gargantua/how-it-works/

- **Gargantua:** a spinning (Kerr) black hole with its thin accretion disk. Fly to it, or take a guided trip (~2 min),
  then switch the engines off and fall through the event horizon down to the inner horizon, where known physics stops.
- **The wormhole** near Saturn: fly through its 1 km throat into Gargantua's galaxy.

## Run it

Any static file server works (ES modules need http, not `file://`). With Node 20+ (no install):

```
npm start          # http://localhost:8080
npm test           # the physics tests
```

Or `python -m http.server 8080`. Needs a browser with WebGL2 and float render targets (any recent Chrome, Edge,
Firefox or Safari). Weak GPUs and phones switch to a lighter mode automatically.

Controls: W / S fly, A / D strafe, Q / E down / up, Shift faster, drag to look, wheel to fly, C grid sky, M sound,
R reset, Esc back to the start screen. On touch screens: drag to look, on-screen buttons to fly.

## How it works

| | |
| --- | --- |
| Spacetime | Kerr metric in Cartesian Kerr–Schild form, g = η + f l⊗l: smooth through both horizons, exact cheap inverse |
| Light | Hamilton's equations H = ½ g^μν p_μ p_ν = 0, RK4 backwards from the camera; metric gradients by forward-mode automatic differentiation (dual numbers) in GLSL |
| Capture | rays that can only fall in are cut early with an exact test from Carter's constant (R(r) > 0 below the ray) |
| Camera | the local non-rotating observer (ZAMO), Lorentz-boosted by your velocity (aberration); free fall is a timelike geodesic in proper time |
| Disk | thin, between the ISCO and its edge, Keplerian rotation, blackbody colour (Planck × CIE 1931); `?mode=reality` adds the true Doppler + gravitational shift g = 1/(u^t(E − ΩL)), brightness g⁴ |
| Real time | a cache of ray results (only re-traced when the camera moves), low resolution + TAA while moving, full resolution in bands when still, edges and the photon ring re-traced at screen resolution with ray bundles (stencil-packed), GPU-timer frame budget |
| Wormhole | the "Dneg" wormhole of James et al. 2015: by spherical symmetry each ray depends on one angle, so a table of 2048 rays per frame serves every pixel |

The full engineering log (every problem, the options, the measured numbers) is in
[docs/engineering-log.md](docs/engineering-log.md). File paths there refer to the portfolio site this lives in
(`site/tools/gargantua/` = this repo's root).

| File | |
| --- | --- |
| `kerr.js` | the black hole physics in 64-bit JavaScript (metric, geodesics, observers, orbits, capture test), shared with the tests |
| `shader.js` | the GPU passes: geodesic tracer, refinement masks, disk/sky shading, bloom, TAA, composite |
| `app.js` | WebGL setup, the ray cache and frame budget, camera, flight, the fall, the guided trip, sound |
| `colour.js` | blackbody colour lookup (Wyman, Sloan & Shirley 2013 fit of CIE 1931) |
| `wormhole/` | `worm.js` physics, `worm-shader.js` GPU passes, `wormhole.js` page |
| `tests/` | 26 tests: exact results of general relativity (shadow √27 M, Kerr shadow edges, ISCO, photon orbits, horizons, the free-fall cycloid, aberration), conserved quantities (E, L, Carter's Q, H = 0) along rays, the GPU step policy against a precise integrator, the wormhole's exact edge |

## Checked against the physics

- Shadow of a non-spinning hole: traced 5.19615 M, exact √27 M = 5.19615.
- Shadow edges at spin 0.6: 3.83849 / 6.31565 M, exact 3.83850 / 6.31568 (Bardeen 1973).
- Shadow edge on screen: under 0.1 px (spin ≤ 0.6), under 1.5 px (spin 0.999).
- Capture test: 1,877 of 1,877 rays right against a precise integrator. Free fall: within 0.002 M of the exact cycloid.

## Sources

- O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, "Gravitational lensing by spinning black holes in
  astrophysics, and in the movie Interstellar", Classical and Quantum Gravity 32, 065001 (2015).
- O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, "Visualizing Interstellar's Wormhole", American Journal of
  Physics 83, 486 (2015).
- K. S. Thorne, *The Science of Interstellar* (2014); R. P. Kerr (1963); B. Carter (1968); J. M. Bardeen, W. H. Press,
  S. A. Teukolsky (1972); J. M. Bardeen (1973); E. Poisson & W. Israel (1990); A. J. S. Hamilton & P. P. Avelino (2010).

## License and credits

The code is MIT licensed (see [LICENSE](LICENSE)). The media are not part of that license and keep their own terms:

- `sky/milkyway.webp`: NASA/Goddard Space Flight Center Scientific Visualization Studio, *Deep Star Maps 2020*
  (https://svs.gsfc.nasa.gov/4851), from Gaia DR2: ESA/Gaia/DPAC. Credit required. `scripts/make-sky.py` rebuilds it
  from the original EXR.
- `sound/blackhole.mp3`: a black hole sonification released by NASA. Credit: NASA.
- Fonts: Rubik (Google Fonts, SIL OFL), Bootstrap Icons (MIT), both loaded from their CDNs.

Built by Wajahat Ali ([Dablew Codes](https://dablewcodes.site)).
