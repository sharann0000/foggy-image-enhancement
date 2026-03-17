"""
train.py — Training Script
  - AdamW optimizer (as specified in the project)
  - Cosine LR schedule with warmup
  - Checkpoint saving
  - TensorBoard-compatible logging
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model   import FoggyEnhancementModel
from src.losses  import MultiTaskLoss
from src.dataset import get_dataloaders
from utils.metrics import psnr, ssim_metric, miou, AverageMeter


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    meters = {k: AverageMeter() for k in
              ["loss", "dehazing", "segmentation", "psnr", "ssim"]}

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for batch in pbar:
        foggy  = batch["foggy"].to(device)
        clear  = batch["clear"].to(device)
        masks  = batch["mask"].to(device)

        # Forward pass
        pred_clear, trans_map, atm_light, seg_logits = model(foggy)

        # Compute loss
        loss, components = criterion(pred_clear, clear, trans_map, seg_logits, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            b_psnr = psnr(pred_clear, clear)
            b_ssim = ssim_metric(pred_clear, clear)

        n = foggy.size(0)
        meters["loss"].update(components["total"], n)
        meters["dehazing"].update(components["dehazing"], n)
        meters["segmentation"].update(components["segmentation"], n)
        meters["psnr"].update(b_psnr, n)
        meters["ssim"].update(b_ssim, n)

        pbar.set_postfix({
            "loss": f"{meters['loss'].avg:.4f}",
            "psnr": f"{meters['psnr'].avg:.2f}",
        })

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    meters = {k: AverageMeter() for k in
              ["loss", "psnr", "ssim", "miou"]}

    for batch in tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False):
        foggy  = batch["foggy"].to(device)
        clear  = batch["clear"].to(device)
        masks  = batch["mask"].to(device)

        pred_clear, trans_map, _, seg_logits = model(foggy)
        loss, components = criterion(pred_clear, clear, trans_map, seg_logits, masks)

        n = foggy.size(0)
        meters["loss"].update(components["total"], n)
        meters["psnr"].update(psnr(pred_clear, clear), n)
        meters["ssim"].update(ssim_metric(pred_clear, clear), n)
        meters["miou"].update(miou(seg_logits, masks), n)

    return {k: v.avg for k, v in meters.items()}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # Dataloaders
    loaders = get_dataloaders(
        data_dir    = args.data_dir,
        img_size    = args.img_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    print(f"Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Model
    model = FoggyEnhancementModel(
        img_size  = args.img_size,
        patch_size= 16,
        embed_dim = args.embed_dim,
        depth     = args.depth,
        num_heads = args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Loss & Optimizer (AdamW as specified in project)
    criterion = MultiTaskLoss(lambda_d=0.7, lambda_s=0.3)
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
        betas        = (0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint dir
    os.makedirs(args.save_dir, exist_ok=True)
    best_psnr  = 0.0
    history    = []

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device, epoch)
        val_metrics   = validate(model, loaders["val"],   criterion, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"time {elapsed:.1f}s | lr {lr_now:.2e} | "
            f"train_loss {train_metrics['loss']:.4f} | "
            f"val_psnr {val_metrics['psnr']:.2f} dB | "
            f"val_ssim {val_metrics['ssim']:.4f} | "
            f"val_miou {val_metrics['miou']:.4f}"
        )

        # Log history
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
            "lr":    lr_now,
        })

        # Save best checkpoint
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "val_psnr":   best_psnr,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ New best PSNR {best_psnr:.2f} dB — saved to {ckpt_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth"))

    # Save training history
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in {args.save_dir}/")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Foggy Image Enhancement Model")
    parser.add_argument("--data_dir",    default="data",      help="Dataset root dir")
    parser.add_argument("--save_dir",    default="models",    help="Checkpoint save dir")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--img_size",    type=int, default=256)
    parser.add_argument("--embed_dim",   type=int, default=512)
    parser.add_argument("--depth",       type=int, default=6,  help="Number of ViT encoder layers")
    parser.add_argument("--num_heads",   type=int, default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
