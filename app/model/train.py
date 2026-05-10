"""
app/model/train.py
-------------------
Training script for the AgeGenderNet model.

Usage:
    python -m app.model.train \
        --data_dir data/UTKFace \
        --save_dir models \
        --epochs 20 \
        --batch_size 64 \
        --input_size 96 \
        --lr 1e-3

The script will:
1. Load UTKFace images
2. Split 80/20 train/val
3. Train with multi-task loss (gender + age)
4. Save best model by validation accuracy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .architecture import GENDER_LABELS, build_model, get_age_bins, get_age_group_labels
from .dataset import UTKFaceDataset, get_train_transforms, get_val_transforms


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _class_weights(
    samples: list[tuple[Path, int, int]],
    indices: list[int],
    label_pos: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.tensor([samples[i][label_pos] for i in indices], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * max(num_classes, 1))
    weights[counts == 0] = 0.0
    return weights.to(device)


# ────────────────────────────────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    save_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    input_size: int = 96,
    val_split: float = 0.2,
    device_str: str = "cpu",
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    logger.info(f"Training on device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────── #
    train_base = UTKFaceDataset(data_dir, transform=get_train_transforms(input_size))
    val_base = UTKFaceDataset(data_dir, transform=get_val_transforms(input_size))
    if len(train_base) == 0:
        raise RuntimeError(
            f"No valid images found in {data_dir}. "
            "Download UTKFace and place .jpg files there."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    n_val = int(len(train_base) * val_split)
    if len(train_base) > 1:
        n_val = max(1, min(n_val, len(train_base) - 1))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)} samples")
    logger.info(f"Age bins: {get_age_bins()}")
    logger.info(f"Age labels: {get_age_group_labels()}")

    # ── Model ─────────────────────────────────────────────────────────────── #
    model = build_model(pretrained=True).to(device)
    gender_weights = _class_weights(train_base.samples, train_indices, 1, len(GENDER_LABELS), device)
    age_weights = _class_weights(train_base.samples, train_indices, 2, len(get_age_group_labels()), device)
    gender_criterion = nn.CrossEntropyLoss(weight=gender_weights)
    age_criterion = nn.CrossEntropyLoss(weight=age_weights)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    save_path = Path(save_dir) / "age_gender_model.pth"

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────── #
        model.train()
        train_loss = 0.0
        g_correct = a_correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for imgs, gender_lbl, age_lbl in pbar:
            imgs = imgs.to(device)
            gender_lbl = gender_lbl.to(device)
            age_lbl = age_lbl.to(device)

            optimizer.zero_grad()
            gender_logits, age_logits = model(imgs)

            loss = gender_criterion(gender_logits, gender_lbl) + age_criterion(age_logits, age_lbl)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            g_correct += (gender_logits.argmax(1) == gender_lbl).sum().item()
            a_correct += (age_logits.argmax(1) == age_lbl).sum().item()
            total += imgs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss /= total
        g_train_acc = g_correct / total
        a_train_acc = a_correct / total

        # ── Validation ───────────────────────────────────────────────────── #
        model.eval()
        val_loss = 0.0
        g_correct = a_correct = total = 0

        with torch.no_grad():
            for imgs, gender_lbl, age_lbl in val_loader:
                imgs = imgs.to(device)
                gender_lbl = gender_lbl.to(device)
                age_lbl = age_lbl.to(device)

                gender_logits, age_logits = model(imgs)
                loss = gender_criterion(gender_logits, gender_lbl) + age_criterion(age_logits, age_lbl)

                val_loss += loss.item() * imgs.size(0)
                g_correct += (gender_logits.argmax(1) == gender_lbl).sum().item()
                a_correct += (age_logits.argmax(1) == age_lbl).sum().item()
                total += imgs.size(0)

        val_loss /= total
        g_val_acc = g_correct / total
        a_val_acc = a_correct / total
        combined_acc = (g_val_acc + a_val_acc) / 2

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss {train_loss:.3f}/{val_loss:.3f} | "
            f"Gender {g_train_acc:.3f}/{g_val_acc:.3f} | "
            f"Age {a_train_acc:.3f}/{a_val_acc:.3f}"
        )

        # ── Save best ─────────────────────────────────────────────────────── #
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_combined_acc": best_val_acc,
                    "gender_val_acc": g_val_acc,
                    "age_val_acc": a_val_acc,
                    "input_size": int(input_size),
                    "num_gender_classes": int(len(GENDER_LABELS)),
                    "num_age_classes": int(len(get_age_group_labels())),
                    "gender_labels": list(GENDER_LABELS),
                    "age_bins": list(get_age_bins()),
                    "age_group_labels": list(get_age_group_labels()),
                    "class_weighting": "balanced",
                    "seed": int(seed),
                },
                save_path,
            )
            logger.info(f"  ✓ Best model saved → {save_path} (combined acc: {best_val_acc:.3f})")

    logger.info(f"Training complete. Best combined accuracy: {best_val_acc:.3f}")


# ────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AgeGenderNet on UTKFace")
    p.add_argument("--data_dir", default="data/UTKFace", help="UTKFace image directory")
    p.add_argument("--save_dir", default="models", help="Output directory for weights")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--input_size", type=int, default=96)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--device", default="cpu", help="cuda or cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_size=args.input_size,
        val_split=args.val_split,
        device_str=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
    )
