"""
losses.py — Combined Loss for Multi-Task Learning
  - Dehazing Loss: L1 + Perceptual (SSIM) + Transmission regularization
  - Segmentation Loss: Cross-Entropy
  - Total: weighted sum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SSIM Loss (Structural Similarity)
# ─────────────────────────────────────────────
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        self.window      = self._create_window(window_size, channel)

    def _gaussian(self, window_size, sigma=1.5):
        import math
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        window = self.window.to(img1.device).type_as(img1)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.window_size // 2

        mu1    = F.conv2d(img1, window, padding=pad, groups=self.channel)
        mu2    = F.conv2d(img2, window, padding=pad, groups=self.channel)
        mu1_sq = mu1 ** 2;  mu2_sq = mu2 ** 2;  mu1_mu2 = mu1 * mu2

        s1    = F.conv2d(img1 * img1, window, padding=pad, groups=self.channel) - mu1_sq
        s2    = F.conv2d(img2 * img2, window, padding=pad, groups=self.channel) - mu2_sq
        s12   = F.conv2d(img1 * img2, window, padding=pad, groups=self.channel) - mu1_mu2

        num   = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
        den   = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        return (num / den).mean()

    def forward(self, img1, img2):
        return 1 - self._ssim(img1, img2)


# ─────────────────────────────────────────────
# Dehazing Loss
# ─────────────────────────────────────────────
class DehazingLoss(nn.Module):
    """
    Combines:
      - L1 pixel loss
      - SSIM loss (structural similarity)
      - Transmission smoothness regularization
    """

    def __init__(self, alpha=0.8, beta=0.2, gamma=0.01):
        super().__init__()
        self.alpha   = alpha   # weight for L1
        self.beta    = beta    # weight for SSIM
        self.gamma   = gamma   # weight for transmission smoothness
        self.l1      = nn.L1Loss()
        self.ssim    = SSIMLoss()

    def _transmission_smoothness(self, t):
        """Encourage spatially smooth transmission maps."""
        dx = torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])
        dy = torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])
        return dx.mean() + dy.mean()

    def forward(self, pred_clear, target_clear, trans_map):
        l1_loss   = self.l1(pred_clear, target_clear)
        ssim_loss = self.ssim(pred_clear, target_clear)
        smooth    = self._transmission_smoothness(trans_map)
        total     = self.alpha * l1_loss + self.beta * ssim_loss + self.gamma * smooth
        return total, {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "smooth": smooth.item(),
        }


# ─────────────────────────────────────────────
# Segmentation Loss
# ─────────────────────────────────────────────
class SegmentationLoss(nn.Module):
    """
    Weighted Cross-Entropy loss for semantic segmentation.
    Class weights handle class imbalance (road/sky much larger than pedestrians).
    """

    def __init__(self, num_classes=5, ignore_index=255):
        super().__init__()
        # Weights: Background, Road, Car, Sky, Pedestrian
        weights = torch.tensor([0.5, 1.0, 1.5, 0.8, 2.0])
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        logits : (B, C, H, W)
        targets: (B, H, W) with class indices
        """
        return self.ce(logits, targets)


# ─────────────────────────────────────────────
# Combined Multi-Task Loss
# ─────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Total loss = lambda_d * L_dehaze + lambda_s * L_segment
    Default: 0.7 dehazing + 0.3 segmentation
    """

    def __init__(self, lambda_d=0.7, lambda_s=0.3):
        super().__init__()
        self.lambda_d   = lambda_d
        self.lambda_s   = lambda_s
        self.dehaze_loss = DehazingLoss()
        self.seg_loss    = SegmentationLoss()

    def forward(self, pred_clear, target_clear, trans_map, seg_logits, seg_targets):
        d_loss, d_components = self.dehaze_loss(pred_clear, target_clear, trans_map)
        s_loss               = self.seg_loss(seg_logits, seg_targets)
        total                = self.lambda_d * d_loss + self.lambda_s * s_loss

        return total, {
            "total":         total.item(),
            "dehazing":      d_loss.item(),
            "segmentation":  s_loss.item(),
            **{f"dehaze_{k}": v for k, v in d_components.items()},
        }
