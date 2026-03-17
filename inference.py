"""
inference.py — Run the trained model on new hazy images
Produces: dehazed image, transmission map, segmentation overlay
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FoggyEnhancementModel

# Segmentation class colors (RGBA)
SEG_COLORS = {
    0: (100, 100, 100),   # Background — grey
    1: (128, 64,  128),   # Road       — purple
    2: (  0,  0, 255),    # Car        — blue
    3: ( 70, 130, 180),   # Sky        — steel blue
    4: (220,  20,  60),   # Pedestrian — crimson
}
SEG_NAMES  = {0: "Background", 1: "Road", 2: "Car", 3: "Sky", 4: "Pedestrian"}


def colorize_seg(seg_map: np.ndarray) -> np.ndarray:
    """Convert (H,W) class index map → (H,W,3) RGB color image."""
    h, w   = seg_map.shape
    color  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in SEG_COLORS.items():
        color[seg_map == cls] = rgb
    return color


def load_model(checkpoint_path: str, device: torch.device, img_size=256) -> FoggyEnhancementModel:
    model = FoggyEnhancementModel(img_size=img_size, embed_dim=512, depth=6, num_heads=8)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)       # support both full checkpoint and plain state_dict
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(
    model:      FoggyEnhancementModel,
    image_path: str,
    img_size:   int,
    device:     torch.device,
):
    """Run model on a single image. Returns dict of result arrays."""
    img   = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    img_t = torch.from_numpy(np.array(img)).float() / 255.0   # (H,W,3)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)    # (1,3,H,W)

    pred_clear, trans_map, atm_light, seg_logits = model(img_t)

    clear_np = pred_clear[0].cpu().permute(1, 2, 0).numpy()
    clear_np = (clear_np * 255).clip(0, 255).astype(np.uint8)

    trans_np = trans_map[0, 0].cpu().numpy()
    trans_np = (trans_np * 255).clip(0, 255).astype(np.uint8)

    seg_cls  = seg_logits[0].argmax(dim=0).cpu().numpy()
    seg_rgb  = colorize_seg(seg_cls)

    return {
        "hazy":      np.array(img),
        "clear":     clear_np,
        "trans":     trans_np,
        "seg_cls":   seg_cls,
        "seg_color": seg_rgb,
        "atm_light": atm_light[0].cpu().numpy().flatten(),
    }


def save_result_figure(results: dict, output_path: str):
    """Save a 4-panel result figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Foggy Image Enhancement — Results", fontsize=14, fontweight="bold")

    axes[0].imshow(results["hazy"]);       axes[0].set_title("Input (Hazy)")
    axes[1].imshow(results["clear"]);      axes[1].set_title("Dehazed Output")
    axes[2].imshow(results["trans"], cmap="gray"); axes[2].set_title("Transmission Map")
    axes[3].imshow(results["seg_color"]);  axes[3].set_title("Segmentation")

    # Add legend to seg panel
    patches = [mpatches.Patch(color=np.array(c)/255, label=SEG_NAMES[cls])
               for cls, c in SEG_COLORS.items()]
    axes[3].legend(handles=patches, loc="lower right", fontsize=7)

    A = results["atm_light"]
    axes[1].set_xlabel(f"Atmospheric Light A = [{A[0]:.2f}, {A[1]:.2f}, {A[2]:.2f}]", fontsize=8)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved result to {output_path}")


def process_directory(model, input_dir, output_dir, img_size, device):
    os.makedirs(output_dir, exist_ok=True)
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in exts]
    print(f"Processing {len(files)} images...")
    for fname in files:
        stem    = os.path.splitext(fname)[0]
        results = run_inference(model, os.path.join(input_dir, fname), img_size, device)
        # Save individual outputs
        Image.fromarray(results["clear"]).save(os.path.join(output_dir, f"{stem}_clear.png"))
        Image.fromarray(results["seg_color"]).save(os.path.join(output_dir, f"{stem}_seg.png"))
        save_result_figure(results, os.path.join(output_dir, f"{stem}_result.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on hazy images")
    parser.add_argument("--checkpoint", required=True,          help="Path to model checkpoint")
    parser.add_argument("--image",      default=None,           help="Single image path")
    parser.add_argument("--input_dir",  default=None,           help="Directory of images")
    parser.add_argument("--output",     default="outputs/result.png", help="Output path (single image)")
    parser.add_argument("--output_dir", default="outputs/",     help="Output dir (batch)")
    parser.add_argument("--img_size",   type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device, args.img_size)

    if args.image:
        results = run_inference(model, args.image, args.img_size, device)
        save_result_figure(results, args.output)

    elif args.input_dir:
        process_directory(model, args.input_dir, args.output_dir, args.img_size, device)

    else:
        print("Provide --image or --input_dir")
