import logging
import os
from typing import Any, Dict

import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

logger = logging.getLogger(__name__)

load_dotenv()


def preprocess(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Balansuje zbiór danych poprzez oversampling mniejszościowej klasy.

    Args:
        data: Surowy DataFrame z danymi.
        parameters: Słownik zawierający:
            - target_column: nazwa kolumny docelowej
            - split.random_state: ziarno losowości

    Returns:
        DataFrame z zbalansowanym rozkładem klas.
    """
    target = parameters["target_column"]

    logger.info("Dane surowe: %d wierszy, rozkład klas:\n%s",
                len(data), data[target].value_counts().to_string())

    majority = data[data[target] == 0]
    minority = data[data[target] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=parameters["split"]["random_state"],
    )

    data_balanced = pd.concat([majority, minority_upsampled])
    data_balanced = shuffle(
        data_balanced, random_state=parameters["split"]["random_state"])

    logger.info("Po balansowaniu: %d wierszy, rozkład klas:\n%s",
                len(data_balanced), data_balanced[target].value_counts().to_string())

    return data_balanced


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]):
    """Dzieli dane na zbiory treningowy, walidacyjny i testowy.

    Args:
        data: DataFrame z danymi (po balansowaniu).
        parameters: Słownik zawierający:
            - target_column: nazwa kolumny docelowej
            - split.test_size: frakcja danych na test (np. 0.2)
            - split.val_ratio: frakcja danych testowych na walidację
            - split.random_state: ziarno losowości

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    target = parameters["target_column"]
    split_params = parameters["split"]

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"],
        stratify=y_temp,
    )

    logger.info("Train: %d, Val: %d, Test: %d",
                len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]) -> Pipeline:
    """Trenuje pipeline klasyfikatora RandomForest z imputacją i skalowaniem.

    Args:
        X_train: Cecha zbioru treningowego.
        y_train: Etykiety zbioru treningowego.
        parameters: Słownik zawierający:
            - model.n_estimators: liczba drzew
            - model.max_depth: maks. głębokość drzew (opcjonalnie, domyślnie None)
            - model.random_state: ziarno losowości
            - model.class_weight: wagi klas
            - model.n_jobs: liczba rdzeni
            - imputer.strategy: strategia imputacji (np. 'mean')

    Returns:
        Wytrenowany Pipeline z imputerem, skalierem i klasyfikatorem.
    """
    model_params = parameters["model"]
    imputer_params = parameters["imputer"]

    model_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_params["strategy"])),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=model_params["n_estimators"],
                    max_depth=model_params.get("max_depth"),
                    random_state=model_params["random_state"],
                    class_weight=model_params["class_weight"],
                    n_jobs=model_params["n_jobs"],
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    logger.info(
        "Model wytrenowany: %d drzew, max_depth=%s, class_weight=%s",
        model_params["n_estimators"],
        model_params.get("max_depth"),
        model_params["class_weight"],
    )
    return model_pipeline


def evaluate_and_log(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Ewaluuje model na zbiorze walidacyjnym i loguje wyniki do W&B.

    Args:
        model: Wytrenowany pipeline scikit-learn.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Etykiety zbioru walidacyjnego.
        parameters: Cały słownik parameters.yml — logowany jako config.

    Returns:
        Słownik z metrykami ewaluacji.
    """
    mp = parameters["model"]
    depth = mp.get("max_depth")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "asi-housing"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"rf-n{parameters['model']['n_estimators']}-d{parameters['model']['max_depth']}",
        config={
            "model_type": "RandomForestClassifier",
            "n_estimators": mp["n_estimators"],
            "max_depth": depth,
            "random_state": mp["random_state"],
            "class_weight": mp["class_weight"],
            "n_jobs": mp["n_jobs"],
            "test_size": parameters["split"]["test_size"],
            "val_ratio": parameters["split"]["val_ratio"],
            "imputer_strategy": parameters["imputer"]["strategy"],
        },
        tags=["baseline", "sklearn", "classification"],
    )

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "rmse": round(float(root_mean_squared_error(y_val, y_pred_proba)), 4),
        "mae": round(float(mean_absolute_error(y_val, y_pred_proba)), 4),
        "r2": round(float(r2_score(y_val, y_pred_proba)), 4),
    }

    wandb.log(metrics)

    wandb.sklearn.plot_confusion_matrix(
        y_val, y_pred, labels=["klasa_0", "klasa_1"]
    )

    artifact = wandb.Artifact(
        name="baseline-model",
        type="model",
        description=f"RandomForest n={mp['n_estimators']}, max_depth={depth}",
    )
    artifact.add_file("data/06_models/baseline_model.pkl")
    wandb.log_artifact(artifact)

    wandb.finish()

    logger.info(
        "W&B: run zalogowany. accuracy=%.4f, f1=%.4f",
        metrics["accuracy"],
        metrics["f1"],
    )
    return metrics
