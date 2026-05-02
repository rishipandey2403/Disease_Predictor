from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Health Assistant"
    page_title: str = "AI Health Risk Screening"
    page_icon: str = "🩺"
    layout: str = "wide"


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "saved_models"
DATASET_DIR = ROOT_DIR / "dataset"
