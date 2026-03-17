"""
fog_generator.py — Synthetic Fog Generation using Atmospheric Scattering Model
Physical model: I(x) = J(x)·t(x) + A·(1 - t(x))
  - J(x) = clear image
  - t(x) = transmission map = exp(-beta * d(x))
  - A    = atmospheric light (global)
  - beta = scattering coefficient (fog density)
  - d(x) = scene depth
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2


def estimate_depth(image: np.ndarray) -> np.ndarray:
    """
    Simple depth estimation heuristic for outdoor scenes.
    Uses vertical position + brightness as depth proxy
    (objects higher in frame & brighter tend to be farther).
    Range: [0, 1] — 0 = near, 1 = far
    """
    h, w = image.shape[:2]
    # Vertical gradient (top = far, bottom = near)
    y_grad = np.linspace(1.0, 0.0, h)[:, None]          # (H, 1)
    y_grad = np.tile(y_grad, (1, w))                      # (H, W)

    # Brightness channel (brighter = farther in haze)
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    depth  = 0.6 * y_grad + 0.4 * gray
    depth  = np.clip(depth, 0.0, 1.0)
    return depth


def add_fog(
    image: np.ndarray,
    beta: float = 1.0,
    A: float = 0.95,
    depth_map: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fog to a clear image using the Atmospheric Scattering Model.

    Args:
        image     : (H, W, 3) uint8 RGB image [0–255]
        beta      : scattering coefficient — higher = denser fog (default 1.0)
        A         : atmospheric light intensity [0–1] (default 0.95)
        depth_map : optional (H, W) float32 depth [0–1]; estimated if None

    Returns:
        foggy_img  : (H, W, 3) uint8 foggy image
        trans_map  : (H, W)   float32 transmission map
    """
    img_float = image.astype(np.float32) / 255.0          # normalize

    if depth_map is None:
        depth_map = estimate_depth(image)

    # Transmission map: t(x) = exp(-beta * d(x))
    trans_map = np.exp(-beta * depth_map).astype(np.float32)  # (H, W) in [0,1]

    # Atmospheric light (scalar → 3-channel)
    atm = np.array([A, A, A], dtype=np.float32)

    # Apply model: I(x) = J(x)·t(x) + A·(1 - t(x))
    t3        = trans_map[:, :, np.newaxis]               # (H, W, 1)
    foggy     = img_float * t3 + atm * (1 - t3)
    foggy     = np.clip(foggy, 0.0, 1.0)
    foggy_u8  = (foggy * 255).astype(np.uint8)

    return foggy_u8, trans_map


def generate_foggy_dataset(
    input_dir: str,
    output_dir: str,
    beta_values: list[float] = [0.5, 1.0, 2.0],
    img_size: int = 256,
):
    """
    Generate multiple foggy versions of every image in input_dir.
    Saves foggy images + transmission maps.
    """
    os.makedirs(output_dir, exist_ok=True)
    fog_dir   = os.path.join(output_dir, "foggy")
    trans_dir = os.path.join(output_dir, "transmission")
    clear_dir = os.path.join(output_dir, "clear")
    for d in [fog_dir, trans_dir, clear_dir]:
        os.makedirs(d, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    files      = [f for f in os.listdir(input_dir)
                  if os.path.splitext(f)[1].lower() in extensions]

    print(f"Found {len(files)} images. Generating foggy versions with β={beta_values}")

    for fname in files:
        stem = os.path.splitext(fname)[0]
        path = os.path.join(input_dir, fname)
        img  = Image.open(path).convert("RGB").resize((img_size, img_size))
        img_np = np.array(img)

        # Save clear image (resized)
        img.save(os.path.join(clear_dir, f"{stem}.png"))

        for beta in beta_values:
            foggy_np, trans_np = add_fog(img_np, beta=beta)

            # Save foggy image
            fog_name = f"{stem}_beta{beta:.1f}.png"
            Image.fromarray(foggy_np).save(os.path.join(fog_dir, fog_name))

            # Save transmission map (grayscale)
            trans_u8 = (trans_np * 255).astype(np.uint8)
            Image.fromarray(trans_u8).save(os.path.join(trans_dir, fog_name))

    print(f"Done. Saved to {output_dir}/")


# ─────────────────────────────────────────────
# Demo: add fog to a single image
# ─────────────────────────────────────────────
def demo_single(image_path: str, output_path: str = "foggy_demo.png", beta: float = 1.2):
    img      = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np   = np.array(img)
    foggy_np, trans = add_fog(img_np, beta=beta)

    # Side-by-side comparison
    combined = np.hstack([img_np, foggy_np])
    Image.fromarray(combined).save(output_path)
    print(f"Saved side-by-side to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Fog Generator")
    parser.add_argument("--input_dir",  required=True,   help="Directory of clear images")
    parser.add_argument("--output_dir", required=True,   help="Output directory")
    parser.add_argument("--beta",       nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="Fog density values (e.g. 0.5 1.0 2.0)")
    parser.add_argument("--img_size",   type=int, default=256)
    args = parser.parse_args()
    generate_foggy_dataset(args.input_dir, args.output_dir, args.beta, args.img_size)
