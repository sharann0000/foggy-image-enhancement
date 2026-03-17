"""
model.py — Multi-Task Vision Transformer for Foggy Image Enhancement
"One Body, Two Heads" Architecture:
  - Shared ViT Encoder (backbone)
  - Branch A: Restoration Head (Dehazing)
  - Branch B: Segmentation Head (Semantic Segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────
# 1.  Patch Embedding
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Split image into 16×16 patches and project to embedding dim."""

    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)  →  (B, num_patches, embed_dim)
        x = self.projection(x)                         # (B, E, H/P, W/P)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


# ─────────────────────────────────────────────
# 2.  Multi-Head Self-Attention
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))


# ─────────────────────────────────────────────
# 3.  Feed-Forward Block
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# ─────────────────────────────────────────────
# 4.  Single Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


# ─────────────────────────────────────────────
# 5.  ViT Encoder (Shared Backbone)
# ─────────────────────────────────────────────
class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed  = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches  = self.patch_embed.num_patches
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed    = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout  = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                           # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, E)
        x   = torch.cat([cls, x], dim=1)                  # (B, N+1, E)
        x   = self.pos_dropout(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x)
        # Return patch tokens only (drop CLS)
        return x[:, 1:, :]                                # (B, N, E)


# ─────────────────────────────────────────────
# 6.  Restoration Head (Dehazing Decoder)
# ─────────────────────────────────────────────
class RestorationHead(nn.Module):
    """
    Decodes patch tokens → clear image.
    Atmospheric Scattering Model: I(x) = J(x)·t(x) + A·(1-t(x))
    The head predicts both the transmission map t(x) and atmospheric light A,
    then reconstructs J(x) = (I(x) - A) / max(t(x), 0.1) + A
    """

    def __init__(self, embed_dim=512, img_size=256, patch_size=16):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        h = w = img_size // patch_size             # feature map size (16×16 for 256 input)

        # Upsample patch features back to full resolution
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),           # 16→32
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),           # 32→64
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),           # 64→128
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),           # 128→256
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Transmission map head (per-pixel, 0–1)
        self.trans_head = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # Atmospheric light head (global, 3-channel, 0–1)
        self.atm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

        # Final clean image refinement
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, tokens, hazy_img):
        B, N, E = tokens.shape
        h = w = self.img_size // self.patch_size
        # Reshape tokens to spatial feature map
        feat = rearrange(tokens, 'b (h w) e -> b e h w', h=h, w=w)
        feat = self.decoder(feat)                  # (B, 16, H, W)

        t = self.trans_head(feat)                  # (B, 1, H, W)  transmission
        A = self.atm_head(feat)                    # (B, 3)        atmospheric light
        A = A.view(B, 3, 1, 1)

        # Recover clear image via atmospheric scattering model
        t  = t.clamp(0.1, 1.0)
        J  = (hazy_img - A) / t + A               # (B, 3, H, W)
        J  = J.clamp(0.0, 1.0)

        # Refine
        J  = self.refine(J)
        return J, t, A


# ─────────────────────────────────────────────
# 7.  Segmentation Head
# ─────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """
    Decodes patch tokens → per-pixel class labels.
    Classes: 0=Background, 1=Road, 2=Car, 3=Sky, 4=Pedestrian
    """

    NUM_CLASSES = 5

    def __init__(self, embed_dim=512, img_size=256, patch_size=16):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, self.NUM_CLASSES, 1),   # class logits
        )

    def forward(self, tokens):
        B, N, E = tokens.shape
        h = w = self.img_size // self.patch_size
        feat = rearrange(tokens, 'b (h w) e -> b e h w', h=h, w=w)
        logits = self.decoder(feat)               # (B, C, H, W)
        return logits


# ─────────────────────────────────────────────
# 8.  Full Multi-Task Model
# ─────────────────────────────────────────────
class FoggyEnhancementModel(nn.Module):
    """
    One Body, Two Heads:
      - Shared ViT Encoder
      - Restoration Head (dehazing)
      - Segmentation Head
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        self.encoder     = ViTEncoder(img_size, patch_size, 3, embed_dim, depth, num_heads)
        self.restore     = RestorationHead(embed_dim, img_size, patch_size)
        self.segment     = SegmentationHead(embed_dim, img_size, patch_size)

    def forward(self, hazy_img):
        """
        Args:
            hazy_img: (B, 3, H, W) normalized to [0, 1]
        Returns:
            clear_img:  (B, 3, H, W)  — dehazed output
            trans_map:  (B, 1, H, W)  — transmission map
            atm_light:  (B, 3, 1, 1)  — atmospheric light
            seg_logits: (B, C, H, W)  — segmentation logits
        """
        tokens     = self.encoder(hazy_img)
        clear_img, trans_map, atm_light = self.restore(tokens, hazy_img)
        seg_logits = self.segment(tokens)
        return clear_img, trans_map, atm_light, seg_logits


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = FoggyEnhancementModel(img_size=256, embed_dim=512, depth=6, num_heads=8)
    dummy = torch.randn(2, 3, 256, 256)
    clear, t, A, seg = model(dummy)
    print(f"Input shape     : {dummy.shape}")
    print(f"Clear image     : {clear.shape}")
    print(f"Transmission map: {t.shape}")
    print(f"Atmospheric light: {A.shape}")
    print(f"Segmentation    : {seg.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
