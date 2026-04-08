import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

logger = logging.getLogger(__name__)


def preprocess(data: pd.DataFrame, parameters: Dict[str, Any],) -> pd.DataFrame:

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
    data_balanced = shuffle(data_balanced, random_state=parameters["split"]["random_state"])

    logger.info("Po balansowaniu: %d wierszy, rozkład klas:\n%s",
                len(data_balanced), data_balanced[target].value_counts().to_string())

    return data_balanced


def split_data(data: pd.DataFrame, parameters: Dict[str, Any]):

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

    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]) -> Pipeline:

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
                    random_state=model_params["random_state"],
                    class_weight=model_params["class_weight"],
                    n_jobs=model_params["n_jobs"],
                ),
            ),
        ]
    )

    model_pipeline.fit(X_train, y_train)

    logger.info(
        "Model wytrenowany: %d drzew, class_weight=%s",
        model_params["n_estimators"],
        model_params["class_weight"],
    )
    return model_pipeline


def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_val, y_pred_proba)), 4),
        "mae": round(float(mean_absolute_error(y_val, y_pred_proba)), 4),
        "r2": round(float(r2_score(y_val, y_pred_proba)), 4),
    }

    logger.info(
        "Metryki: accuracy=%.4f, precision=%.4f, RMSE=%.4f, MAE=%.4f, R2=%.4f",
        metrics["accuracy"], metrics["precision"],
        metrics["rmse"], metrics["mae"], metrics["r2"],
    )
    return metrics
