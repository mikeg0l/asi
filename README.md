```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
cp .env.example .env
conda create -n asi python=3.10 -y
conda activate asi
pip install -r requirements.txt
```

dodać credentials do conf/local/credentials.yml

## Uruchomienie (Sprint 6 — Streamlit + dane syntetyczne)

### 0. Przygotowanie środowiska (raz)

```bash
conda activate asi
pip install -r requirements.txt
# Seed bazy SQLite z water_potability.csv (tworzy data/01_raw/dataset.db, tabela water_potability):
python scripts/seed_db.py
# Wytrenuj model dla API (baseline -> data/06_models/baseline_model.pkl):
kedro run
# (opcjonalnie lepszy model: kedro run --pipeline=automl)
```

`conf/local/credentials.yml` musi zawierać połączenie do bazy:

```yaml
db_credentials:
  con: sqlite:///data/01_raw/dataset.db
```

> **W&B:** pipeline'y logują do Weights & Biases (`wandb.init`), więc przed `kedro run`
> ustaw `WANDB_API_KEY` w `.env` albo wykonaj `wandb login`. Bez tego run zatrzyma się na
> prompcie o klucz. Tryb offline: bash `WANDB_MODE=offline kedro run ...`,
> PowerShell `$env:WANDB_MODE="offline"; kedro run ...`.

### 1. Pipeline danych syntetycznych (SDV)

```bash
kedro run --pipeline=synthetic
```

Produkuje `data/03_primary/synthetic_data.csv` oraz `data/08_reporting/synthetic_scores.json`
i loguje `sdv/diagnostic_score` (~1.0) i `sdv/quality_score` (<1.0) do W&B.

### 2. API (FastAPI) — osobny terminal

```bash
uvicorn api.main:app --reload      # http://127.0.0.1:8000/docs
```

### 3. Dashboard (Streamlit) — osobny terminal

```bash
python -m streamlit run app/streamlit_app.py   # http://localhost:8501
```

Adres API konfiguruje zmienna środowiskowa `API_URL` (domyślnie `http://127.0.0.1:8000`):

```powershell
# PowerShell (Windows)
$env:API_URL = "http://127.0.0.1:8000"; python -m streamlit run app/streamlit_app.py
```

```bash
# bash / zsh (Linux/macOS)
API_URL=http://127.0.0.1:8000 python -m streamlit run app/streamlit_app.py
```

Dashboard ma trzy zakładki: **Predykcja** (formularz → `/predict`), **Dane**
(podgląd z SQLite + statystyki + histogram), **Dane syntetyczne** (generowanie SDV
i porównanie z oryginałem).
