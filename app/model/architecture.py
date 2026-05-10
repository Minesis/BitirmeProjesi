"""
app/model/architecture.py
--------------------------
CNN architecture for simultaneous age-group and gender prediction.

Input  : (B, 3, 64, 64) RGB face crop
Outputs:
    - gender_logits : (B, 2)   → Male / Female
    - age_logits    : (B, N)   → configurable age bins
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


# ────────────────────────────────────────────────────────────────────────────
#  Label helpers
# ────────────────────────────────────────────────────────────────────────────

GENDER_LABELS = ["Male", "Female"]

# Default (recommended) bins used by the UI + training unless overridden in config.
# If you change these, you must retrain the age head.
DEFAULT_AGE_BINS = [13, 18, 25, 35, 45, 55, 65, 75, 85]
DEFAULT_AGE_GROUP_LABELS = [
    "0-12",
    "13-17",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65-74",
    "75-84",
    "85+",
]

# Backward-compatibility fallback for older checkpoints (4-class age head).
LEGACY_AGE_GROUP_LABELS_4 = ["0-18", "18-30", "30-50", "50+"]


def get_age_bins() -> list[int]:
    raw = config.get("age_gender.age_bins", default=None)
    if not raw:
        return list(DEFAULT_AGE_BINS)
    try:
        bins = [int(x) for x in list(raw)]
    except Exception:
        return list(DEFAULT_AGE_BINS)
    bins = sorted(set(bins))
    bins = [b for b in bins if b > 0]
    return bins or list(DEFAULT_AGE_BINS)


def get_age_group_labels() -> list[str]:
    raw = config.get("age_gender.age_labels", default=None)
    bins = get_age_bins()
    if not raw:
        return _labels_from_bins(bins)
    labels = [str(x) for x in list(raw)]
    if len(labels) != len(bins) + 1:
        return _labels_from_bins(bins)
    return labels or _labels_from_bins(bins)


def _labels_from_bins(bins: list[int]) -> list[str]:
    labels: list[str] = []
    start = 0
    for threshold in bins:
        labels.append(f"{start}-{threshold - 1}")
        start = threshold
    labels.append(f"{start}+")
    return labels


def age_to_group(age: int, age_bins: list[int] | None = None) -> int:
    """Convert raw integer age to group index based on configurable thresholds."""
    bins = age_bins or get_age_bins()
    for idx, thr in enumerate(bins):
        if age < thr:
            return idx
    return len(bins)


# ────────────────────────────────────────────────────────────────────────────
#  Model
# ────────────────────────────────────────────────────────────────────────────

class AgeGenderNet(nn.Module):
    """
    MobileNetV2 backbone with two classification heads.
    Lightweight enough for real-time inference on CPU.
    """

    def __init__(
        self,
        num_age_classes: int = len(DEFAULT_AGE_GROUP_LABELS),
        num_gender_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = tv_models.mobilenet_v2(weights=weights)

        # Remove the original classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier[1].in_features  # 1280

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Task-specific heads
        self.gender_head = nn.Linear(512, num_gender_classes)
        self.age_head = nn.Linear(512, num_age_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        return self.gender_head(x), self.age_head(x)


# ────────────────────────────────────────────────────────────────────────────
#  Factory
# ────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> AgeGenderNet:
    age_labels = get_age_group_labels()
    return AgeGenderNet(
        num_age_classes=len(age_labels),
        num_gender_classes=len(GENDER_LABELS),
        pretrained=pretrained,
    )


def load_model(path: str, device: str = "cpu") -> AgeGenderNet:
    """Load a saved model checkpoint."""
    state: Any = torch.load(path, map_location=device)
    state_dict = state.get("model_state_dict") if isinstance(state, dict) and "model_state_dict" in state else state
    if not isinstance(state_dict, dict):
        raise RuntimeError("Invalid checkpoint format: expected a state_dict or checkpoint dict.")

    # Infer head sizes from checkpoint so different age-binnings are supported.
    try:
        if isinstance(state, dict):
            n_gender = int(state.get("num_gender_classes", state_dict["gender_head.weight"].shape[0]))
            n_age = int(state.get("num_age_classes", state_dict["age_head.weight"].shape[0]))
        else:
            n_gender = int(state_dict["gender_head.weight"].shape[0])
            n_age = int(state_dict["age_head.weight"].shape[0])
    except Exception as exc:
        raise RuntimeError(f"Could not infer head sizes from checkpoint: {exc}")

    model = AgeGenderNet(num_age_classes=n_age, num_gender_classes=n_gender, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
