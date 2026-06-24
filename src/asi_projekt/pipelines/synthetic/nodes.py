import logging
import os
from typing import Any, Dict

import pandas as pd
import wandb
from dotenv import load_dotenv
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

load_dotenv()

logger = logging.getLogger(__name__)


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje nazwy kolumn do ``str``.

    Dane z ``pandas.SQLTableDataset`` mają nazwy kolumn typu ``quoted_name``
    (SQLAlchemy), na których SDV i zapis do CSV potrafią się wyłożyć. Ta sama
    normalizacja jest stosowana w pipeline ``data_processing`` (``split_data``).
    """
    data = data.copy()
    data.columns = [str(c) for c in data.columns]
    return data


def generate_synthetic_data(
    real_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Generuje dane syntetyczne synthesizerem GaussianCopula.

    Args:
        real_data: Dane rzeczywiste wczytane z katalogu (``raw_data`` z SQLite).
        parameters: Sekcja ``params:synthetic`` z ``parameters.yml``.

    Returns:
        Wygenerowany zbiór syntetyczny o ``n_samples`` wierszach.
    """
    real_data = _normalize_columns(real_data)

    metadata = Metadata.detect_from_dataframe(real_data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)

    synthetic = synthesizer.sample(num_rows=parameters["n_samples"])
    logger.info("SDV: wygenerowano %d rekordów syntetycznych", len(synthetic))
    return synthetic


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Ewaluuje jakość danych syntetycznych i loguje wyniki do W&B.

    Args:
        real_data: Dane rzeczywiste.
        synthetic_data: Dane wygenerowane przez SDV.
        parameters: Sekcja ``params:synthetic``.

    Returns:
        Słownik: ``diagnostic_score`` (~1.0 — poprawność struktury) oraz
        ``quality_score`` (<1.0 — podobieństwo statystyczne).
    """
    real_data = _normalize_columns(real_data)

    metadata = Metadata.detect_from_dataframe(real_data)

    diagnostic = run_diagnostic(real_data, synthetic_data, metadata)
    quality = evaluate_quality(real_data, synthetic_data, metadata)

    scores = {
        "diagnostic_score": float(diagnostic.get_score()),
        "quality_score": float(quality.get_score()),
    }

    logger.info(
        "SDV: diagnostic_score=%.4f, quality_score=%.4f",
        scores["diagnostic_score"],
        scores["quality_score"],
    )

    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "asi-projekt"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"sdv-n{parameters['n_samples']}",
        job_type="sdv_evaluation",
        config={"n_samples": parameters["n_samples"]},
        tags=["sdv", "synthetic"],
    ):
        wandb.log(
            {
                "sdv/diagnostic_score": scores["diagnostic_score"],
                "sdv/quality_score": scores["quality_score"],
            }
        )

    return scores
