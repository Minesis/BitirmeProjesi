"""
app/model/dataset.py
---------------------
PyTorch Dataset for the UTKFace dataset.

UTKFace filename format:
    <age>_<gender>_<race>_<timestamp>.jpg
    (some distributions use *.jpg.chip.jpg)
    gender: 0 = Male, 1 = Female
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from .architecture import age_to_group


# ────────────────────────────────────────────────────────────────────────────
#  Transforms
# ────────────────────────────────────────────────────────────────────────────

def get_train_transforms(input_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(input_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ────────────────────────────────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────────────────────────────────

def _parse_utkface_filename(filename: str, max_age: int) -> tuple[int, int] | None:
    """
    Parse labels from a UTKFace filename.

    Supports both:
      - <age>_<gender>_<race>_<timestamp>.jpg
      - <age>_<gender>_<race>_<timestamp>.jpg.chip.jpg
    """
    parts = filename.split("_")
    if len(parts) < 4:
        return None

    age_s, gender_s, race_s = parts[0], parts[1], parts[2]
    if not age_s.isdigit() or gender_s not in ("0", "1") or not race_s.isdigit():
        return None

    age = int(age_s)
    if age < 0 or age > max_age:
        return None

    gender = int(gender_s)
    age_group = age_to_group(age)
    return gender, age_group


class UTKFaceDataset(Dataset):
    """
    Loads UTKFace images and parses age + gender labels from filenames.

    Args:
        root_dir   : Path to the directory containing UTKFace images (can include subfolders).
        transform  : torchvision transform pipeline.
        max_age    : Skip images with age > max_age (default 116).
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
        max_age: int = 116,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or get_val_transforms()
        self.samples: list[tuple[Path, int, int]] = []  # (path, gender_label, age_group)

        skipped = 0
        for fpath in sorted(self.root_dir.rglob("*.jpg")):
            parsed = _parse_utkface_filename(fpath.name, max_age=max_age)
            if parsed is None:
                skipped += 1
                continue
            gender, age_group = parsed
            self.samples.append((fpath, gender, age_group))

        logger.info(
            f"UTKFaceDataset: {len(self.samples)} valid samples "
            f"({skipped} skipped) in {root_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, gender_label, age_group = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Return a black image if the file is corrupted
            img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img = self.transform(img)
        return (
            img,
            torch.tensor(gender_label, dtype=torch.long),
            torch.tensor(age_group, dtype=torch.long),
        )
