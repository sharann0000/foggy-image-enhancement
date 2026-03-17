"""
visualize.py — Visualization utilities
  - Side-by-side comparison plots
  - Training curve plotter
  - Segmentation overlay
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SEG_COLORS = {
    0: (100, 100, 100),
    1: (128,  64, 128),
    2: (  0,   0, 255),
    3: ( 70, 130, 180),
    4: (220,  20,  60),
}
SEG_NAMES = {0: "Background", 1: "Road", 2: "Car", 3: "Sky", 4: "Pedestrian"}


def plot_training_curves(history_path: str, output_path: str = "training_curves.png"):
    """Plot train/val loss, PSNR, SSIM curves from history.json."""
    with open(history_path) as f:
        history = json.load(f)

    epochs    = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss   = [h["val"]["loss"]   for h in history]
    val_psnr   = [h["val"]["psnr"]   for h in history]
    val_ssim   = [h["val"]["ssim"]   for h in history]
    val_miou   = [h["val"]["miou"]   for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Curves — Foggy Image Enhancement", fontsize=13)

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="#F44336")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_psnr, color="#4CAF50")
    axes[1].set_title("Val PSNR (dB)"); axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylabel("PSNR (dB)")

    axes[2].plot(epochs, val_ssim, label="SSIM",  color="#9C27B0")
    axes[2].plot(epochs, val_miou, label="mIoU",  color="#FF9800")
    axes[2].set_title("Val SSIM & mIoU"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def colorize_seg(seg_map: np.ndarray) -> np.ndarray:
    color = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cls, rgb in SEG_COLORS.items():
        color[seg_map == cls] = rgb
    return color


def overlay_seg(image: np.ndarray, seg_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend segmentation colors over the image."""
    seg_rgb = colorize_seg(seg_map)
    blended = (alpha * seg_rgb + (1 - alpha) * image).astype(np.uint8)
    return blended


def comparison_grid(images: list, titles: list, output_path: str, figsize=(20, 5)):
    """Generic side-by-side image grid."""
    n   = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {output_path}")
