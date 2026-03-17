"""
metrics.py — Evaluation Metrics
  - PSNR  (Peak Signal-to-Noise Ratio) for dehazing quality
  - SSIM  (Structural Similarity Index) for perceptual quality
  - mIoU  (Mean Intersection over Union) for segmentation
"""

import torch
import numpy as np


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between predicted and target images.
    Higher is better (dB). > 30 dB is generally good for dehazing.
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(torch.tensor(max_val ** 2) / mse)).item()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute mean SSIM over a batch.
    Range [0, 1] — higher is better.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def _ssim_single(p, t):
        mu_p  = p.mean();  mu_t = t.mean()
        sig_p = p.var();   sig_t = t.var()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        num   = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den   = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
        return (num / den).item()

    scores = [_ssim_single(pred[i], target[i]) for i in range(pred.shape[0])]
    return float(np.mean(scores))


def miou(pred_logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5, ignore_index: int = 255) -> float:
    """
    Compute mean Intersection over Union for segmentation.
    pred_logits: (B, C, H, W)
    targets    : (B, H, W) with class indices
    """
    preds = pred_logits.argmax(dim=1)             # (B, H, W)
    ious  = []
    for cls in range(num_classes):
        pred_mask   = (preds   == cls)
        target_mask = (targets == cls) & (targets != ignore_index)
        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue                               # class not present
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


class AverageMeter:
    """Running average tracker for training metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.avg:.4f}"
