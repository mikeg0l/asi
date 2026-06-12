import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTOGLUON_PATH = PROJECT_ROOT / "data/06_models/autogluon"
BASELINE_PATH = PROJECT_ROOT / "data/06_models/baseline_model.pkl"

ModelType = Literal["autogluon", "baseline"]


@dataclass
class LoadedModel:
    predictor: Any
    model_type: ModelType
    model_name: str


def _load_autogluon() -> LoadedModel:
    """Ładuje model AutoGluon TabularPredictor z dysku."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(AUTOGLUON_PATH))
    return LoadedModel(
        predictor=predictor,
        model_type="autogluon",
        model_name=predictor.model_best,
    )


def _load_baseline() -> LoadedModel:
    """Ładuje baseline'owy pipeline sklearn z pliku pickle."""
    with BASELINE_PATH.open("rb") as file:
        predictor = pickle.load(file)
    return LoadedModel(
        predictor=predictor,
        model_type="baseline",
        model_name="baseline_RandomForest",
    )


def load_model() -> dict[str, Any]:
    """Ładuje model raz przy starcie. Najpierw AutoGluon, potem fallback do baseline pickle."""
    loaded: LoadedModel | None = None

    if AUTOGLUON_PATH.exists():
        try:
            loaded = _load_autogluon()
            logger.info("Loaded AutoGluon model: %s", loaded.model_name)
        except Exception:
            logger.exception("Failed to load AutoGluon model from %s", AUTOGLUON_PATH)

    if loaded is None and BASELINE_PATH.exists():
        try:
            loaded = _load_baseline()
            logger.info("Loaded baseline model: %s", loaded.model_name)
        except Exception:
            logger.exception("Failed to load baseline model from %s", BASELINE_PATH)

    if loaded is None:
        logger.error("No model could be loaded")
        return {
            "predictor": None,
            "model_type": None,
            "model_name": None,
            "model_loaded": False,
        }

    return {
        "predictor": loaded.predictor,
        "model_type": loaded.model_type,
        "model_name": loaded.model_name,
        "model_loaded": True,
    }


def run_prediction(predictor: Any, features: dict[str, Any]) -> float:
    """Konwertuje cechy do DataFrame i zwraca znormalizowaną wartość predykcji."""
    input_df = pd.DataFrame([features])
    prediction = predictor.predict(input_df)
    return float(
        prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0]
    )
