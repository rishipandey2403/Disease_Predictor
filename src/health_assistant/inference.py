from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MODEL_DIR


@dataclass
class ModelBundle:
    diabetes: Any
    heart: Any
    parkinsons: Any
    liver: Any
    liver_scaler: Any


def _load_pickle(model_path: Path) -> Any:
    with model_path.open("rb") as f:
        return pickle.load(f)


def load_models(model_dir: Path = MODEL_DIR) -> ModelBundle:
    diabetes_model = _load_pickle(model_dir / "diabetes_model.sav")
    heart_model = _load_pickle(model_dir / "heart_disease_model.sav")
    parkinsons_model = _load_pickle(model_dir / "parkinsons_model.sav")
    liver_model, liver_scaler = _load_pickle(model_dir / "liver_model.sav")

    return ModelBundle(
        diabetes=diabetes_model,
        heart=heart_model,
        parkinsons=parkinsons_model,
        liver=liver_model,
        liver_scaler=liver_scaler,
    )


def predict_binary(model: Any, values: list[float]) -> int:
    prediction = model.predict([values])
    return int(prediction[0])
