"""
train.py — Full Training Loop for DeepMeow Detector
=====================================================
This script is the main entry point for training. It wires together:
  - DataLoader: loads and augments COCO cat images
  - DeepMeowDetector: the full model (Backbone -> FPN -> Head)
  - AdamW optimizer with weight decay
  - Cosine Annealing LR schedule with linear warmup
  - Gradient clipping (prevents exploding gradients)
  - Validation loop with mAP computation
  - Checkpoint saving (best model + periodic saves)

Training strategy:
  We use AdamW instead of plain SGD because:
    - It handles sparse gradients well (common in detection)
    - Weight decay is applied correctly (decoupled from the gradient update)
    - It converges faster with less LR tuning

  Learning rate schedule:
    - Warmup: linearly ramp LR from 0 -> base_lr over the first N epochs
      (prevents large gradient updates at the start when weights are random)
    - Cosine Annealing: slowly decay LR following a cosine curve
      (encourages the model to settle into a good local minimum)

  Gradient clipping:
    - If gradient norm exceeds max_norm, all gradients are scaled down
    - Prevents the "gradient explosion" that can destabilize training

Usage (in Colab):
  !python src/train.py --epochs 50 --batch_size 8 --lr 1e-4
"""

import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# ── Add repo root to sys.path so we can import src.* ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import DeepMeowDetector
from src.data.dataset import CatDataset
from src.data.augmentations import build_train_transform, build_val_transform
from src.utils.metrics import MeanAveragePrecision


# ─── Collate function ─────────────────────────────────────────────
def collate_fn(batch):
    """
    Custom DataLoader collate function.

    Standard PyTorch collate stacks tensors into one big tensor,
    which doesn't work here because each image can have a different
    number of bounding boxes. We instead keep them as separate lists.

    Args:
        batch: List of (image_tensor, target_dict) pairs from the Dataset.

    Returns:
        images  (Tensor): [B, 3, H, W] — stacked image batch
        targets (list):   List of B target dicts, kept separate
    """
    images  = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


# ─── Learning Rate Schedule ───────────────────────────────────────
def build_lr_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    """
    Build a combined Warmup + Cosine Annealing LR scheduler.

    During warmup (first `warmup_epochs`):
        lr = base_lr * (epoch / warmup_epochs)   <- linear ramp

    After warmup:
        lr = base_lr * 0.5 * (1 + cos(pi * (epoch - warmup) / (total - warmup)))
        <- cosine decay from base_lr down to near 0

    Args:
        optimizer     (Optimizer): The AdamW optimizer
        total_epochs  (int):       Total number of training epochs
        warmup_epochs (int):       Number of warmup epochs

    Returns:
        scheduler (LambdaLR): The combined scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: ramp from 0 to 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing: decay from 1 to ~0
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Training Epoch ───────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, max_grad_norm=10.0):
    """
    Run one full epoch of training.

    For each batch:
      1. Move data to GPU
      2. Forward pass -> loss
      3. Backward pass -> gradients
      4. Clip gradients (prevent explosion)
      5. Optimizer step -> update weights
      6. Log progress

    Args:
        model         (nn.Module):  DeepMeowDetector in train mode
        optimizer     (Optimizer):  AdamW
        loader        (DataLoader): Training data loader
        device        (torch.device)
        epoch         (int):        Current epoch number (for logging)
        max_grad_norm (float):      Maximum gradient L2 norm before clipping

    Returns:
        avg_loss (float): Average total loss over this epoch
        loss_breakdown (dict): Average of each individual loss component
    """
    model.train()

    total_loss = 0.0
    total_box  = 0.0
    total_obj  = 0.0
    total_cls  = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        # Move targets to device — skip non-tensor fields like image_id (int)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        # ── Forward pass ─────────────────────────────────────────
        loss, loss_dict = model(images, targets)

        # ── Backward pass ────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # ── Gradient clipping ────────────────────────────────────
        # Computes the global L2 norm of all gradients and scales them
        # down if it exceeds max_grad_norm.
        # This is important early in training when gradients can be very large.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # ── Weight update ────────────────────────────────────────
        optimizer.step()

        # ── Accumulate loss for reporting ────────────────────────
        total_loss += loss_dict["total"]
        total_box  += loss_dict["box"]
        total_obj  += loss_dict["obj"]
        total_cls  += loss_dict["cls"]

        # ── Progress logging (every 10 batches) ─────────────────
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            pct = 100.0 * (batch_idx + 1) / n_batches
            print(f"  Epoch {epoch:3d} [{batch_idx+1:4d}/{n_batches:4d}] ({pct:5.1f}%)"
                  f"  loss={loss_dict['total']:.3f}"
                  f"  box={loss_dict['box']:.3f}"
                  f"  obj={loss_dict['obj']:.3f}"
                  f"  cls={loss_dict['cls']:.3f}")

    return {
        "total": total_loss / n_batches,
        "box":   total_box  / n_batches,
        "obj":   total_obj  / n_batches,
        "cls":   total_cls  / n_batches,
    }


# ─── Validation Epoch ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run validation: compute mAP on the val set.

    Args:
        model          (nn.Module):  DeepMeowDetector
        loader         (DataLoader): Val data loader
        device         (torch.device)
        conf_threshold (float):      Confidence score cutoff for predictions
        iou_threshold  (float):      NMS IoU threshold

    Returns:
        metrics (dict): {'mAP_50': ..., 'mAP_50_95': ..., 'AP_per_class': [...]}
    """
    model.eval()

    evaluator = MeanAveragePrecision(num_classes=1)

    for images, targets in loader:
        images = images.to(device)

        # Get detections for this batch
        results = model.predict(images,
                                conf_threshold=conf_threshold,
                                iou_threshold=iou_threshold)

        # Accumulate predictions and targets in the evaluator
        evaluator.update(results, targets)

    return evaluator.compute()


# ─── Checkpoint Utilities ─────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path: Path):
    """Save full training state to disk for resuming later."""
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict(),
        "metrics":      metrics,
    }, path)
    print(f"  Checkpoint saved to {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler):
    """Load training state from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    prev_map = ckpt.get("metrics", {}).get("mAP_50", 0.0)
    if prev_map is None:
        prev_map = 0.0
    print(f"  Resumed from epoch {ckpt['epoch']} (previous best mAP@50: {prev_map:.4f})")
    return ckpt["epoch"], float(prev_map)


# ─── Main Training Function ───────────────────────────────────────
def train(
    data_root: str,
    save_dir: str  = "checkpoints",
    epochs: int    = 50,
    batch_size: int = 8,
    lr: float      = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int  = 3,
    num_workers: int    = 2,
    resume: str    = None,
):
    """
    Full training pipeline.

    Args:
        data_root     (str): Path to dataset root (e.g. 'data' or Google Drive path)
        save_dir      (str): Directory to save checkpoints
        epochs        (int): Total training epochs
        batch_size    (int): Images per batch
        lr            (float): Initial learning rate for AdamW
        weight_decay  (float): L2 regularization strength
        warmup_epochs (int): Number of LR warmup epochs
        num_workers   (int): DataLoader worker processes (use 2 in Colab)
        resume        (str): Path to checkpoint to resume training from (or None)
    """
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training on: {device}")
    print(f"Data root  : {data_root}")
    print(f"Epochs     : {epochs}   Batch size: {batch_size}   LR: {lr}")

    # CatDataset expects:
    #   ann_file  = path to the COCO-format JSON annotation file
    #   image_dir = path to the folder containing raw/ train/ val/ images
    #   transform = an AlbumentationsWrapper callable
    data_path = Path(data_root)
    train_dataset = CatDataset(
        ann_file  = str(data_path / "annotations" / "train.json"),
        image_dir = str(data_path / "raw"),
        transform = build_train_transform(input_size=416),
    )
    val_dataset = CatDataset(
        ann_file  = str(data_path / "annotations" / "val.json"),
        image_dir = str(data_path / "raw"),
        transform = build_val_transform(input_size=416),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}   |  Val batches: {len(val_loader)}")

    # ── Build Model ───────────────────────────────────────────────
    model = DeepMeowDetector(num_classes=1, input_size=416).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Build Optimizer ───────────────────────────────────────────
    # Separate weight decay: apply to weights but NOT to biases or BatchNorm params.
    # BatchNorm and bias terms are scale/shift parameters — decaying them is harmful.
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and "bias" not in n and "bn" not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ("bias" in n or "bn" in n)]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)

    # ── Build LR Scheduler ────────────────────────────────────────
    scheduler = build_lr_scheduler(optimizer, epochs, warmup_epochs)

    # ── Resume from checkpoint (optional) ────────────────────────
    start_epoch = 0
    best_map    = 0.0

    if resume:
        resume_path = None
        if resume == "auto":
            # Look for latest.pt or best.pt in save_dir
            if (save_dir / "latest.pt").exists():
                resume_path = save_dir / "latest.pt"
            elif (save_dir / "best.pt").exists():
                resume_path = save_dir / "best.pt"
        elif Path(resume).exists():
            resume_path = Path(resume)

        if resume_path and resume_path.exists():
            start_epoch, best_map = load_checkpoint(resume_path, model, optimizer, scheduler)
        else:
            print(f"  Checkpoint not found at '{resume}', starting from epoch 1.")

    # ── Training Loop ─────────────────────────────────────────────
    history = {"train_loss": [], "val_map50": [], "lr": []}

    print("\n" + "=" * 60)
    print(f"Starting training (Epoch {start_epoch + 1} to {epochs})...")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{epochs}  (LR = {current_lr:.2e})")

        # ── Train ─────────────────────────────────────────────────
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        scheduler.step()

        # ── Validate (every 5 epochs and last epoch) ──────────────
        do_val = ((epoch + 1) % 5 == 0) or (epoch + 1 == epochs)
        if do_val:
            print("  Running validation...")
            val_metrics = validate(model, val_loader, device)
            print(f"  mAP@50     : {val_metrics['mAP_50']:.4f}")
            print(f"  mAP@50:95  : {val_metrics['mAP_50_95']:.4f}")
        else:
            val_metrics = {"mAP_50": None, "mAP_50_95": None}

        # ── Record history ────────────────────────────────────────
        history["train_loss"].append(train_losses["total"])
        history["val_map50"].append(val_metrics["mAP_50"])
        history["lr"].append(current_lr)

        # ── Save latest checkpoint every epoch ───────────────────
        save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics,
                        save_dir / "latest.pt")

        # ── Save periodic checkpoint every 10 epochs ─────────────
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics,
                            save_dir / f"epoch_{epoch+1:03d}.pt")

        # ── Save best model (by mAP@50) ───────────────────────────
        if val_metrics["mAP_50"] is not None and val_metrics["mAP_50"] > best_map:
            best_map = val_metrics["mAP_50"]
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics,
                            save_dir / "best.pt")
            print(f"  New best mAP@50: {best_map:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s"
              f"  | Avg loss: {train_losses['total']:.4f}")

    print("\n" + "=" * 60)
    print(f"Training complete! Best mAP@50: {best_map:.4f}")
    print(f"Best checkpoint saved to: {save_dir / 'best.pt'}")
    print("=" * 60)

    return history


# ─── Command-Line Entry Point ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepMeow cat detector")

    parser.add_argument("--data_root",    type=str,   default="data",
                        help="Dataset root directory (local or Drive path)")
    parser.add_argument("--save_dir",     type=str,   default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",type=int,   default=3)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    train(
        data_root     = args.data_root,
        save_dir      = args.save_dir,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        num_workers   = args.num_workers,
        resume        = args.resume,
    )
