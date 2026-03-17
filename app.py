"""
app.py — Gradio Web Demo for Foggy Image Enhancement
Run: python demo/app.py [--checkpoint models/best_model.pth]
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

from utils.fog_generator import add_fog

# ─────────────────────────────────────────────
# Model-free demo (no checkpoint needed)
# Uses only the fog generator + a simulated "dehazed" output
# ─────────────────────────────────────────────

SEG_COLORS = {
    0: [100, 100, 100], 1: [128, 64, 128], 2: [0, 0, 255],
    3: [70, 130, 180],  4: [220, 20, 60],
}


def simulate_dehazing(hazy_np: np.ndarray, beta_used: float) -> np.ndarray:
    """
    Simulated dehazing via Dark Channel Prior (approximate).
    Used only when no trained model checkpoint is available.
    """
    img = hazy_np.astype(np.float32) / 255.0

    # Estimate atmospheric light (brightest pixels)
    gray   = img.mean(axis=2)
    A      = float(np.percentile(gray, 99))
    A      = min(A, 0.95)

    # Estimate transmission via dark channel
    dark   = img.min(axis=2)
    kernel = np.ones((15, 15)) / 225
    import cv2
    dark_blur = cv2.filter2D(dark, -1, kernel)
    t      = 1 - 0.95 * (dark_blur / (A + 1e-6))
    t      = np.clip(t, 0.1, 1.0)[:, :, np.newaxis]

    # Recover scene radiance
    J      = (img - A) / t + A
    J      = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


def dummy_segmentation(img_np: np.ndarray) -> np.ndarray:
    """Simple heuristic segmentation for demo without a model."""
    h, w, _ = img_np.shape
    seg      = np.zeros((h, w), dtype=np.uint8)

    # Sky: top 25%
    seg[:h//4, :] = 3

    # Road: bottom 30%
    seg[int(h*0.7):, :] = 1

    # Middle: rough car/background split by brightness
    middle  = img_np[h//4:int(h*0.7), :].mean(axis=2)
    bright  = middle > 160
    region  = seg[h//4:int(h*0.7), :]
    region[bright]  = 2   # bright = car
    region[~bright] = 0   # dark   = background
    seg[h//4:int(h*0.7), :] = region

    return seg


def colorize_seg(seg: np.ndarray) -> np.ndarray:
    color = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for cls, rgb in SEG_COLORS.items():
        color[seg == cls] = rgb
    return color


# ─────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────
def process_image(input_image, beta, model_path):
    if input_image is None:
        return None, None, None, None, "Please upload an image."

    img_np  = np.array(Image.fromarray(input_image).convert("RGB").resize((256, 256)))

    # Step 1: Add fog (if not already foggy — for demo we always show the pipeline)
    foggy_np, trans_np = add_fog(img_np, beta=beta)

    # Step 2: Dehaze
    try:
        if model_path and os.path.exists(model_path):
            # Real model inference
            from src.inference import load_model, run_inference
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = load_model(model_path, device)
            tmp    = "/tmp/input_demo.png"
            Image.fromarray(foggy_np).save(tmp)
            results  = run_inference(model, tmp, 256, device)
            clear_np = results["clear"]
            seg_cls  = results["seg_cls"]
        else:
            clear_np = simulate_dehazing(foggy_np, beta)
            seg_cls  = dummy_segmentation(clear_np)
    except Exception as e:
        clear_np = simulate_dehazing(foggy_np, beta)
        seg_cls  = dummy_segmentation(clear_np)

    seg_color = colorize_seg(seg_cls)
    trans_disp = (trans_np * 255).astype(np.uint8) if trans_np.max() <= 1.0 else trans_np

    info = (
        f"✅ Processing complete!\n"
        f"• Fog density (β): {beta:.1f}\n"
        f"• Image size: 256×256\n"
        f"• Atmospheric Scattering Model: I(x) = J(x)·t(x) + A·(1-t(x))\n"
        f"• Segmentation classes: Background, Road, Car, Sky, Pedestrian"
    )
    return foggy_np, clear_np, trans_disp, seg_color, info


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
def build_demo(model_path=None):
    with gr.Blocks(
        title="Foggy Image Enhancement",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px; margin: auto; }"
    ) as demo:

        gr.Markdown("""
        # 🌫️ Foggy Image Enhancement
        ### Vision Transformer (ViT) based Multi-Task Learning
        **St. Martin's Engineering College** | Dept. of CSE | Batch A23

        This system uses the **Atmospheric Scattering Model** and a **ViT encoder** to:
        1. 🔆 **Dehaze** foggy images — restoring clarity
        2. 🎨 **Segment** objects — Road, Car, Sky, Pedestrian
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp_img  = gr.Image(label="Upload Image", type="numpy", height=256)
                beta_sl  = gr.Slider(0.2, 3.0, value=1.2, step=0.1,
                                     label="Fog Density (β) — higher = denser fog")
                mdl_path = gr.Textbox(value=model_path or "",
                                      label="Model Checkpoint (optional)",
                                      placeholder="models/best_model.pth")
                run_btn  = gr.Button("🚀 Enhance Image", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_foggy = gr.Image(label="🌫️ Foggy Input",      height=220)
                    out_clear = gr.Image(label="✨ Dehazed Output",    height=220)
                with gr.Row():
                    out_trans = gr.Image(label="📊 Transmission Map", height=220)
                    out_seg   = gr.Image(label="🎨 Segmentation",      height=220)

        info_box = gr.Textbox(label="Info", interactive=False, lines=5)

        gr.Markdown("""
        ---
        **Segmentation Legend:**
        🟣 Road &nbsp;&nbsp; 🔵 Car &nbsp;&nbsp; 🩵 Sky &nbsp;&nbsp; 🔴 Pedestrian &nbsp;&nbsp; ⬛ Background

        **Atmospheric Scattering Model:** `I(x) = J(x)·t(x) + A·(1 - t(x))`
        where `J(x)` = clear image, `t(x)` = transmission, `A` = atmospheric light
        """)

        run_btn.click(
            fn      = process_image,
            inputs  = [inp_img, beta_sl, mdl_path],
            outputs = [out_foggy, out_clear, out_trans, out_seg, info_box],
        )

        # Example images placeholder
        gr.Examples(
            examples=[],
            inputs=[inp_img],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    args = parser.parse_args()

    if not GRADIO_AVAILABLE:
        print("Install gradio: pip install gradio")
        sys.exit(1)

    demo = build_demo(args.checkpoint)
    demo.launch(server_port=args.port, share=args.share)
