# Foggy Image Enhancement
### Vision Transformer (ViT) based Multi-Task Learning for Dehazing & Segmentation
**St. Martin's Engineering College — Dept. of Computer Science & Engineering**
Batch A23 | Under guidance of Mrs. A. Rajeswari

---

## Overview
This project implements a **Multi-Task Learning (MTL)** framework using a **Vision Transformer (ViT)** backbone to simultaneously:
1. **Dehaze** foggy/hazy images (restore clarity)
2. **Segment** objects in the scene (road, car, sky, pedestrian)

The "One Body, Two Heads" architecture shares a ViT encoder and splits into two decoder branches.

---

## Architecture
```
Hazy Image (3×H×W)
       │
  Patch Embed (16×16 patches)
       │
  Positional Encoding
       │
  ┌────────────────┐
  │  ViT Encoder   │  ← Self-Attention layers (global context)
  └────────────────┘
       │
  ┌────┴────┐
  │         │
Restoration  Segmentation
  Head        Head
  │            │
Clear Image  Seg Mask
(Dehazing)  (Road/Car/Sky)
```

**Atmospheric Scattering Model:**
```
I(x) = J(x)·t(x) + A·(1 - t(x))
```
Where:
- `I(x)` = observed hazy image
- `J(x)` = scene radiance (clear image)
- `t(x)` = transmission map
- `A`     = global atmospheric light

---

## Project Structure
```
foggy-enhancement/
├── src/
│   ├── model.py          # ViT backbone + dual-head architecture
│   ├── dataset.py        # Dataset loader with fog augmentation
│   ├── train.py          # Training loop with AdamW optimizer
│   ├── inference.py      # Run on new hazy images
│   └── losses.py         # Combined dehazing + segmentation loss
├── utils/
│   ├── fog_generator.py  # Synthetic fog generator (Atmospheric Scattering)
│   ├── metrics.py        # PSNR, SSIM, mIoU metrics
│   └── visualize.py      # Side-by-side result visualization
├── demo/
│   └── app.py            # Gradio web demo
├── models/               # Saved checkpoints
├── outputs/              # Result images
├── requirements.txt
└── README.md
```

---

## Setup & Installation
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate synthetic foggy data
```bash
python utils/fog_generator.py --input_dir data/clear --output_dir data/foggy
```

### 2. Train the model
```bash
python src/train.py --epochs 50 --batch_size 8 --lr 1e-4
```

### 3. Run inference on an image
```bash
python src/inference.py --image path/to/hazy.jpg --output outputs/result.jpg
```

### 4. Launch web demo
```bash
python demo/app.py
```

---

## Datasets
- **RESIDE Dataset** — Li et al. (2018) — Benchmark for single image dehazing
- **Cityscapes Dataset** — Cordts et al. (2016) — Urban scene segmentation

---

## References
1. He, K., Sun, J., & Tang, X. (2010). *Single Image Haze Removal Using Dark Channel Prior.* IEEE TPAMI.
2. Dosovitskiy, A., et al. (2020). *An Image is Worth 16×16 Words.* arXiv.
3. McCartney, E.J. (1976). *Optics of the Atmosphere.* John Wiley & Sons.
4. Liang, J., et al. (2021). *SwinIR: Image Restoration Using Swin Transformer.* ICCV Workshops.
