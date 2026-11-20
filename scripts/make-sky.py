"""Convert NASA SVS "Deep Star Maps 2020" Milky Way background (HDR EXR, galactic coordinates) into the 8-bit
equirectangular WebP the Gargantua tool loads (site/tools/gargantua/sky/milkyway.webp).

Source (free to use, credit "NASA/Goddard Space Flight Center Scientific Visualization Studio", Gaia DR2:
ESA/Gaia/DPAC): https://svs.gsfc.nasa.gov/4851 -> milkyway_2020_4k_gal.exr (4096x2048, ~35 MB). Not committed;
download it to a scratch folder and run:

    pip install --target <scratch>/pylib OpenEXR      (numpy + Pillow must be available)
    python scripts/gargantua/make-sky.py <scratch>/milkyway_2020_4k_gal.exr site/tools/gargantua/sky/milkyway.webp <scratch>/pylib

Encoding: linear radiance divided by its 99.95th percentile, clipped to 1, then gamma 1/2.2. The shader decodes it
with pow(texel, 2.2) * brightness. The scale factor is printed and must match MILKY_WAY_SCALE in shader.js only
if the absolute level matters (it doesn't: the brightness is an art knob).
"""
import sys

src, out = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    sys.path.insert(0, sys.argv[3])

import numpy as np
import OpenEXR
from PIL import Image

with OpenEXR.File(src) as f:
    part = f.parts[0]
    ch = part.channels
    if "RGB" in ch:
        rgb = np.asarray(ch["RGB"].pixels, dtype=np.float32)
    else:
        rgb = np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in ("R", "G", "B")], axis=-1)

rgb = np.nan_to_num(np.maximum(rgb, 0.0))
lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
p = float(np.percentile(lum, 99.95))
print(f"size {rgb.shape[1]}x{rgb.shape[0]}, luminance median {np.median(lum):.4g}, 99.95% {p:.4g}, max {lum.max():.4g}")
enc = np.clip(rgb / p, 0.0, 1.0) ** (1 / 2.2)
img = Image.fromarray((enc * 255 + 0.5).astype(np.uint8), "RGB")
img.save(out, "WEBP", quality=88, method=6)
print(f"wrote {out}")
