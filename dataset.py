"""
dataset.py — Dataset and DataLoader for Foggy Image Enhancement
Supports:
  - Paired hazy/clear images (RESIDE-style)
  - On-the-fly synthetic fog generation
  - Segmentation mask loading (Cityscapes-style)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Import fog generator ──────────────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fog_generator import add_fog

# ── Cityscapes class mapping (simplified to 5 classes) ───────────────────────
#  Original Cityscapes → our 5 classes:
#    road/sidewalk → 1 (Road)
#    car/truck/bus → 2 (Car)
#    sky           → 3 (Sky)
#    person/rider  → 4 (Pedestrian)
#    everything else→0 (Background)
CITYSCAPES_TO_5 = {
    0:  0, 1:  0, 2:  0, 3:  0, 4:  0, 5:  0, 6:  0,  # unlabeled
    7:  1, 8:  1,                                        # road, sidewalk
    9:  0, 10: 0, 11: 0, 12: 0, 13: 0,                  # buildings etc.
    14: 0, 15: 0, 16: 0, 17: 0, 18: 0,                  # poles, signs
    19: 0, 20: 0, 21: 0, 22: 0,                          # vegetation
    23: 3,                                               # sky
    24: 4, 25: 4,                                        # person, rider
    26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2,  # vehicles
}


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """Map Cityscapes labels → 5-class labels."""
    out = np.zeros_like(mask)
    for src, dst in CITYSCAPES_TO_5.items():
        out[mask == src] = dst
    return out


class FoggyDataset(Dataset):
    """
    Expects directory structure:
        data/
          clear/   *.png  — ground truth clear images
          foggy/   *.png  — corresponding hazy images  (or auto-generated)
          masks/   *.png  — segmentation masks (optional)

    If foggy/ is absent or a pair is missing, fog is synthesized on-the-fly.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 256,
        augment: bool = True,
        beta_range: tuple = (0.5, 2.5),
        use_masks: bool = True,
    ):
        self.img_size   = img_size
        self.augment    = augment and (split == "train")
        self.beta_range = beta_range
        self.use_masks  = use_masks

        self.clear_dir = os.path.join(data_dir, "clear")
        self.foggy_dir = os.path.join(data_dir, "foggy")
        self.mask_dir  = os.path.join(data_dir, "masks")

        # Collect filenames
        assert os.path.exists(self.clear_dir), f"clear/ dir not found in {data_dir}"
        all_files = sorted([
            f for f in os.listdir(self.clear_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # Train/val/test split (80/10/10)
        n = len(all_files)
        if   split == "train": self.files = all_files[:int(0.8*n)]
        elif split == "val":   self.files = all_files[int(0.8*n):int(0.9*n)]
        else:                  self.files = all_files[int(0.9*n):]

        # Transforms
        self.to_tensor = transforms.ToTensor()          # [0,1], (C,H,W)
        self.resize    = transforms.Resize((img_size, img_size), antialias=True)

    def __len__(self):
        return len(self.files)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _augment(self, clear: Image.Image, foggy: Image.Image, mask):
        """Apply matching random augmentations to all modalities."""
        if random.random() > 0.5:
            clear = transforms.functional.hflip(clear)
            foggy = transforms.functional.hflip(foggy)
            if mask is not None:
                mask  = transforms.functional.hflip(mask)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            clear = transforms.functional.rotate(clear, angle)
            foggy = transforms.functional.rotate(foggy, angle)
            if mask is not None:
                mask  = transforms.functional.rotate(mask, angle)
        return clear, foggy, mask

    def __getitem__(self, idx):
        fname      = self.files[idx]
        stem       = os.path.splitext(fname)[0]

        # ── Load clear image ──────────────────────────────────────
        clear_path = os.path.join(self.clear_dir, fname)
        clear_img  = self._load_image(clear_path).resize(
            (self.img_size, self.img_size), Image.BICUBIC
        )

        # ── Load or generate foggy image ─────────────────────────
        foggy_path = os.path.join(self.foggy_dir, fname)
        if os.path.exists(foggy_path):
            foggy_img = self._load_image(foggy_path).resize(
                (self.img_size, self.img_size), Image.BICUBIC
            )
        else:
            beta      = random.uniform(*self.beta_range)
            foggy_np, _ = add_fog(np.array(clear_img), beta=beta)
            foggy_img = Image.fromarray(foggy_np)

        # ── Load segmentation mask ────────────────────────────────
        mask = None
        if self.use_masks:
            mask_path = os.path.join(self.mask_dir, f"{stem}.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )

        # ── Augment ───────────────────────────────────────────────
        if self.augment:
            clear_img, foggy_img, mask = self._augment(clear_img, foggy_img, mask)

        # ── To tensor ─────────────────────────────────────────────
        clear_t = self.to_tensor(clear_img)    # (3, H, W) float [0,1]
        foggy_t = self.to_tensor(foggy_img)    # (3, H, W) float [0,1]

        if mask is not None:
            mask_np = remap_mask(np.array(mask).astype(np.int64))
            mask_t  = torch.from_numpy(mask_np).long()   # (H, W)
        else:
            # Dummy mask — ignored in loss with ignore_index
            mask_t  = torch.full((self.img_size, self.img_size), 255, dtype=torch.long)

        return {
            "foggy": foggy_t,
            "clear": clear_t,
            "mask":  mask_t,
            "name":  fname,
        }


def get_dataloaders(
    data_dir: str,
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FoggyDataset(data_dir, split=split, img_size=img_size,
                          augment=(split == "train"))
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == "train"),
        )
    return loaders


if __name__ == "__main__":
    # Quick test with synthetic fog on a single image
    import tempfile, shutil
    from utils.fog_generator import add_fog

    # Create a tiny dummy dataset
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "clear"))
        dummy = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        dummy.save(os.path.join(tmp, "clear", "test.png"))

        ds = FoggyDataset(tmp, split="train", img_size=256)
        sample = ds[0]
        print("foggy:", sample["foggy"].shape, sample["foggy"].min().item(), sample["foggy"].max().item())
        print("clear:", sample["clear"].shape)
        print("mask:", sample["mask"].shape, sample["mask"].unique())
        print("Dataset test passed!")
