"""Dashboard Streamlit dla projektu ASI — predykcja jakości wody + dane syntetyczne (SDV).

Trzy zakładki:
- Predykcja: formularz cech → POST /predict (FastAPI) → wynik 0/1.
- Dane: podgląd zbioru z SQLite + statystyki + histogram.
- Dane syntetyczne: generowanie SDV (GaussianCopula) i porównanie z oryginałem.

Uruchomienie (z katalogu projektu):
    python -m streamlit run app/streamlit_app.py

Wymaga działającego API (osobny terminal):
    uvicorn api.main:app --reload
"""

# --- 1. IMPORTY ---
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# --- 2. KONFIGURACJA ---
# API_URL z konfiguracji (zmienna środowiskowa), NIE hardcoded — zob. wymagania sprintu.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
# Ścieżka kotwiczona do pliku (jak api/model_loader.py), a nie do CWD — działa niezależnie
# od katalogu, z którego uruchomiono `streamlit run`.
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "01_raw" / "dataset.db"
TABLE_NAME = "water_potability"

# Cechy wody: (nazwa, min, max, wartość_domyślna, krok).
# Zakresy zgodne z ograniczeniami Field w api/schemas.py (WaterPotabilityFeatures).
FEATURES = [
    ("ph", 0.0, 14.0, 7.08, 0.01),
    ("Hardness", 0.0, 500.0, 196.37, 0.1),
    ("Solids", 0.0, 70000.0, 22014.09, 1.0),
    ("Chloramines", 0.0, 20.0, 7.12, 0.01),
    ("Sulfate", 0.0, 500.0, 333.78, 0.1),
    ("Conductivity", 0.0, 800.0, 426.21, 0.1),
    ("Organic_carbon", 0.0, 30.0, 14.28, 0.01),
    ("Trihalomethanes", 0.0, 130.0, 66.40, 0.1),
    ("Turbidity", 0.0, 10.0, 3.97, 0.01),
]


# --- 3. FUNKCJE CACHE'OWANE ---
@st.cache_data
def load_data() -> pd.DataFrame:
    """Wczytuje dane z SQLite. Cache'owane — zapytanie wykonuje się raz."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)


@st.cache_resource
def fit_synthesizer(real_data: pd.DataFrame) -> GaussianCopulaSynthesizer:
    """Trenuje synthesizer raz i cache'uje wytrenowany obiekt (fit jest kosztowny)."""
    metadata = Metadata.detect_from_dataframe(real_data)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(real_data)
    return synth


# --- 4. LAYOUT ---
st.set_page_config(page_title="ASI - Jakość wody", layout="wide")
st.title("ASI - Dashboard MLOps (jakość wody)")
tab_pred, tab_data, tab_synth = st.tabs(["Predykcja", "Dane", "Dane syntetyczne"])


# --- 5a. ZAKŁADKA: PREDYKCJA ---
with tab_pred:
    st.header("Predykcja zdatności wody do picia")
    st.caption(f"API: {API_URL}")

    values: dict[str, float] = {}
    col1, col2 = st.columns(2)
    for i, (name, lo, hi, default, step) in enumerate(FEATURES):
        target_col = col1 if i % 2 == 0 else col2
        values[name] = target_col.slider(name, lo, hi, default, step)

    if st.button("Przewiduj", type="primary"):
        try:
            response = requests.post(f"{API_URL}/predict", json=values, timeout=10)
            if response.status_code == 200:
                result = response.json()
                label = int(round(result["prediction"]))
                if label == 1:
                    st.success("Woda ZDATNA do picia (Potability = 1) ✅")
                else:
                    st.warning("Woda NIEZDATNA do picia (Potability = 0) ❌")
                st.caption(f"Model: {result['model']}")
            elif response.status_code == 422:
                st.error("Błędne dane wejściowe (walidacja Pydantic).")
                st.json(response.json())
            elif response.status_code == 503:
                st.error(
                    "API działa, ale model nie jest załadowany. "
                    "Uruchom `kedro run` (lub `--pipeline=automl`) i zrestartuj API."
                )
            else:
                st.error(f"Błąd API ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Nie można połączyć się z API pod {API_URL}. "
                "Czy `uvicorn api.main:app` jest uruchomione?"
            )
        except requests.exceptions.Timeout:
            st.error("API nie odpowiedziało w wyznaczonym czasie (timeout). Spróbuj ponownie.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Błąd komunikacji z API: {exc}")


# --- 5b. ZAKŁADKA: DANE ---
with tab_data:
    st.header("Podgląd danych")
    try:
        df = load_data()
    except Exception as exc:  # noqa: BLE001 — czytelny komunikat zamiast tracebacku
        st.error(
            f"Nie udało się wczytać danych z `{DB_PATH}`. "
            "Uruchom `python scripts/seed_db.py`, aby zaseedować bazę.\n\n"
            f"Szczegóły: {exc}"
        )
    else:
        st.write(f"Liczba rekordów: {len(df)}")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Statystyki opisowe")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Histogram wybranej kolumny")
        column = st.selectbox("Kolumna", df.select_dtypes("number").columns)
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=30)
        ax.set_xlabel(column)
        ax.set_ylabel("Liczność")
        st.pyplot(fig)
        plt.close(fig)  # bez tego figury akumulują się w globalnym stanie pyplot


# --- 5c. ZAKŁADKA: DANE SYNTETYCZNE ---
with tab_synth:
    st.header("Dane syntetyczne (SDV)")
    try:
        df = load_data()
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Brak danych w `{DB_PATH}` - zaseeduj bazę: `python scripts/seed_db.py`.\n\n"
            f"Szczegóły: {exc}"
        )
    else:
        n_samples = st.number_input(
            "Liczba rekordów do wygenerowania",
            min_value=100,
            max_value=10_000,
            value=1000,
            step=100,
        )

        if st.button("Generuj dane syntetyczne"):
            with st.spinner("Trenowanie synthesizera i generowanie..."):
                synth = fit_synthesizer(df)  # cache'owane — fit tylko raz
                st.session_state["synthetic"] = synth.sample(num_rows=int(n_samples))

        if "synthetic" in st.session_state:  # przetrwało rerun dzięki session_state
            synthetic = st.session_state["synthetic"]
            st.success(f"Wygenerowano {len(synthetic)} rekordów.")

            col_real, col_synth = st.columns(2)
            with col_real:
                st.subheader("Oryginał (statystyki)")
                st.dataframe(df.describe(), use_container_width=True)
            with col_synth:
                st.subheader("Syntetyczne (statystyki)")
                st.dataframe(synthetic.describe(), use_container_width=True)

            st.subheader("Podgląd danych syntetycznych")
            st.dataframe(synthetic.head(20), use_container_width=True)
