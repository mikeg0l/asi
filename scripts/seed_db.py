"""Jednorazowy seed bazy SQLite z pliku water_potability.csv.

Tworzy ``data/01_raw/dataset.db`` i ładuje CSV do tabeli ``water_potability``
(zgodnie z ``conf/base/catalog.yml`` — dataset ``raw_data``). Uruchom raz po
sklonowaniu repozytorium, zanim odpalisz Kedro / API / Streamlit:

    python scripts/seed_db.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "water_potability.csv"
DB_PATH = PROJECT_ROOT / "data" / "01_raw" / "dataset.db"
TABLE_NAME = "water_potability"


def main() -> None:
    """Wczytuje CSV i zapisuje go do tabeli SQLite (nadpisując istniejącą)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {CSV_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(
        f"Zaseedowano {len(df)} wierszy do {DB_PATH} (tabela '{TABLE_NAME}')."
    )


if __name__ == "__main__":
    main()
