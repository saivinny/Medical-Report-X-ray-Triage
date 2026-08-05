"""
Generates placeholder, procedurally-drawn grayscale images shaped like the
sample/clinical_notes.csv rows, purely so the training/evaluation pipeline
can be run and tested end to end without waiting on real (credentialed)
X-ray data.

These are NOT real medical images and carry no diagnostic meaning -- they
are simple synthetic textures (a rib-cage-like grid plus randomised
"opacity" blobs whose density loosely scales with the assigned label) used
only to exercise the code. Swap in a real dataset before drawing any
conclusions (see README section 4, "Datasets").
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

OUT_ROOT = Path(__file__).resolve().parent / "xrays"
SIZE = 256

# Rough "opacity density" per label -- purely for making the synthetic
# images visually distinguishable in the demo, not clinically meaningful.
DENSITY = {"normal": 1, "attention": 3, "urgent": 6}


def draw_synthetic_xray(density: int) -> Image.Image:
    base = np.random.normal(loc=60, scale=12, size=(SIZE, SIZE)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # faint rib-like arcs so it reads as "chest-shaped" rather than pure noise
    for i in range(6):
        y = 40 + i * 30
        draw.arc([30, y, SIZE - 30, y + 200], start=200, end=340, fill=(90, 90, 90), width=3)

    # randomised "opacity" blobs -- density scales loosely with label severity
    for _ in range(density):
        cx, cy = random.randint(60, SIZE - 60), random.randint(60, SIZE - 60)
        r = random.randint(10, 26)
        shade = random.randint(140, 200)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(shade, shade, shade))

    return img.convert("L")


def main():
    for label_dir, density in DENSITY.items():
        out_dir = OUT_ROOT / label_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # generate a few extra images per folder so counts comfortably cover the CSV
        n_images = 10
        for i in range(1, n_images + 1):
            img = draw_synthetic_xray(density)
            img.save(out_dir / f"sample_{i:03d}.png")
        print(f"Wrote {n_images} synthetic images to {out_dir}")


if __name__ == "__main__":
    main()
