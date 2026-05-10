"""
app/model/inference.py
-----------------------
Real-time inference wrapper for AgeGenderNet.

Usage:
    predictor = AgeGenderPredictor("models/age_gender_model.pth")
    result = predictor.predict(face_bgr_crop)
    # result: {"gender": "Female", "age_group": "25-34", "gender_conf": 0.91, "age_conf": 0.78}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torchvision import transforms

from .architecture import (
    DEFAULT_AGE_GROUP_LABELS,
    GENDER_LABELS,
    LEGACY_AGE_GROUP_LABELS_4,
    AgeGenderNet,
    get_age_group_labels,
)


# ────────────────────────────────────────────────────────────────────────────
#  Transform (inference only – no augmentation)
# ────────────────────────────────────────────────────────────────────────────

def _build_infer_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ────────────────────────────────────────────────────────────────────────────
#  Predictor
# ────────────────────────────────────────────────────────────────────────────

class AgeGenderPredictor:
    """
    Wraps AgeGenderNet for single-frame face crop prediction.

    Falls back to a rule-based "Unknown" result if the model file is absent
    so the rest of the pipeline can continue without crashing.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self._model: AgeGenderNet | None = None
        self._gender_labels: list[str] = list(GENDER_LABELS)
        self._age_labels: list[str] = list(get_age_group_labels())
        self._transform = _build_infer_transform(64)

        path = Path(model_path)
        if path.exists():
            try:
                state = torch.load(str(path), map_location=device)
                state_dict = state.get("model_state_dict") if isinstance(state, dict) and "model_state_dict" in state else state
                if not isinstance(state_dict, dict):
                    raise RuntimeError("Invalid checkpoint format")

                # Infer head sizes from checkpoint (supports different age bin counts).
                n_gender = int(state_dict["gender_head.weight"].shape[0])
                n_age = int(state_dict["age_head.weight"].shape[0])
                configured_age_labels = get_age_group_labels()
                if n_age != len(configured_age_labels):
                    logger.warning(
                        "Age/gender checkpoint has "
                        f"{n_age} age classes, but config defines {len(configured_age_labels)}. "
                        "Retrain the model to use the configured age_labels."
                    )

                # Labels from checkpoint (preferred), else fall back based on class count.
                if isinstance(state, dict) and isinstance(state.get("gender_labels"), list):
                    self._gender_labels = [str(x) for x in state["gender_labels"]]
                else:
                    self._gender_labels = list(GENDER_LABELS)

                if isinstance(state, dict) and isinstance(state.get("age_group_labels"), list):
                    self._age_labels = [str(x) for x in state["age_group_labels"]]
                else:
                    if n_age == 4:
                        self._age_labels = list(LEGACY_AGE_GROUP_LABELS_4)
                    elif n_age == len(configured_age_labels):
                        self._age_labels = list(configured_age_labels)
                    elif n_age == len(DEFAULT_AGE_GROUP_LABELS):
                        self._age_labels = list(DEFAULT_AGE_GROUP_LABELS)
                    else:
                        self._age_labels = [f"bin_{i}" for i in range(n_age)]

                # Input size can be stored in checkpoint; else use config default (64).
                input_size = 64
                if isinstance(state, dict) and state.get("input_size"):
                    try:
                        input_size = int(state["input_size"])
                    except Exception:
                        input_size = 64
                self._transform = _build_infer_transform(input_size)

                model = AgeGenderNet(num_age_classes=n_age, num_gender_classes=n_gender, pretrained=False)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                self._model = model

                logger.info(f"AgeGenderPredictor loaded from {path} (age_classes={n_age})")
            except Exception as exc:
                logger.warning(f"Failed to load age/gender model: {exc}. Using fallback.")
        else:
            logger.warning(
                f"Model file not found at {path}. "
                "Demographics will be 'Unknown' until model is trained."
            )

    def predict(self, face_bgr: np.ndarray) -> dict:
        """
        Predict gender and age group from a BGR face crop (OpenCV format).

        Args:
            face_bgr: H×W×3 numpy array in BGR colour space.

        Returns:
            {
                "gender"       : "Male" | "Female" | "Unknown",
                "age_group"    : configured age label such as "25-34" or "85+",
                "gender_conf"  : float (0.0 – 1.0),
                "age_conf"     : float (0.0 – 1.0),
            }
        """
        if self._model is None or face_bgr is None or face_bgr.size == 0:
            return self._unknown()

        try:
            # BGR → RGB → PIL
            rgb = face_bgr[:, :, ::-1].copy()
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                gender_logits, age_logits = self._model(tensor)
                gender_probs = F.softmax(gender_logits, dim=1)[0]
                age_probs = F.softmax(age_logits, dim=1)[0]

            gender_idx = gender_probs.argmax().item()
            age_idx = age_probs.argmax().item()

            return {
                "gender": self._gender_labels[gender_idx] if gender_idx < len(self._gender_labels) else "Unknown",
                "age_group": self._age_labels[age_idx] if age_idx < len(self._age_labels) else "Unknown",
                "gender_conf": round(gender_probs[gender_idx].item(), 3),
                "age_conf": round(age_probs[age_idx].item(), 3),
            }
        except Exception as exc:
            logger.debug(f"Inference error: {exc}")
            return self._unknown()

    @staticmethod
    def _unknown() -> dict:
        return {
            "gender": "Unknown",
            "age_group": "Unknown",
            "gender_conf": 0.0,
            "age_conf": 0.0,
        }
