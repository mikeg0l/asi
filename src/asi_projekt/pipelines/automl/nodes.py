import logging
import os
from typing import Any, Dict

import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any],
) -> TabularPredictor:
    """Trenuje modele AutoML za pomocą AutoGluon TabularPredictor.

    Args:
        X_train: Cechy zbioru treningowego.
        y_train: Wartości docelowe zbioru treningowego.
        parameters: Słownik parametrów z parameters.yml.

    Returns:
        Wytrenowany TabularPredictor.
    """
    target_column = parameters["target_column"]

    train_data = pd.concat([X_train, y_train], axis=1)

    logger.info(
        "AutoGluon: trening z presetem '%s', time_limit=%ds",
        parameters["automl"]["presets"],
        parameters["automl"]["time_limit"],
    )

    predictor = TabularPredictor(
        label=target_column,
        eval_metric=parameters["automl"]["eval_metric"],
        path="data/06_models/autogluon",
    ).fit(
        train_data,
        time_limit=parameters["automl"]["time_limit"],
        presets=parameters["automl"]["presets"],
        verbosity=1,
    )

    logger.info("AutoGluon: trening zakończony.")
    return predictor


def evaluate_automl(
    predictor: TabularPredictor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Ewaluuje AutoGluon na zbiorze walidacyjnym i loguje wyniki do W&B.

    Args:
        predictor: Wytrenowany TabularPredictor.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Wartości docelowe zbioru walidacyjnego.
        parameters: Słownik parametrów z parameters.yml.

    Returns:
        Słownik z metrykami najlepszego modelu.
    """
    eval_metric = parameters["automl"]["eval_metric"]

    val_data = pd.concat([X_val, y_val], axis=1)
    leaderboard = predictor.leaderboard(data=val_data, silent=True)

    best_model_name = leaderboard.iloc[0]["model"]
    best_score_raw = leaderboard.iloc[0]["score_val"]

    metrics_negated = [
        "root_mean_squared_error",
        "mean_absolute_error",
        "mean_squared_error",
        "median_absolute_error",
    ]
    best_metric_value = (
        -best_score_raw if eval_metric in metrics_negated else best_score_raw
    )

    logger.info(
        "AutoGluon: najlepszy model = %s, %s = %.4f",
        best_model_name,
        eval_metric,
        best_metric_value,
    )

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "asi-housing"),
        entity=os.getenv("WANDB_ENTITY"),
        name=(
            f"automl-{parameters['automl']['presets']}-"
            f"{parameters['automl']['time_limit']}s"
        ),
        config={
            "model_type": "AutoGluon",
            "presets": parameters["automl"]["presets"],
            "time_limit": parameters["automl"]["time_limit"],
            "eval_metric": eval_metric,
            "best_model": best_model_name,
        },
        tags=["automl", "autogluon"],
    )

    wandb.log({eval_metric: best_metric_value})

    leaderboard_table = wandb.Table(
        dataframe=leaderboard[
            ["model", "score_val", "pred_time_val", "fit_time"]
        ].reset_index(drop=True)
    )
    wandb.log({"leaderboard": leaderboard_table})

    wandb.finish()

    return {
        "best_model": best_model_name,
        eval_metric: float(best_metric_value),
        "n_models_trained": len(leaderboard),
        "presets": parameters["automl"]["presets"],
        "time_limit": parameters["automl"]["time_limit"],
    }
