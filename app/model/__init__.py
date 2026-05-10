from .architecture import AgeGenderNet, build_model, get_age_group_labels, load_model, GENDER_LABELS
from .inference import AgeGenderPredictor
from .dataset import UTKFaceDataset

AGE_GROUP_LABELS = get_age_group_labels()

__all__ = [
    "AgeGenderNet",
    "build_model",
    "load_model",
    "AgeGenderPredictor",
    "UTKFaceDataset",
    "GENDER_LABELS",
    "AGE_GROUP_LABELS",
    "get_age_group_labels",
]
