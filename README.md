# QM 640 Capstone: U.S. Energy Transition & Decarbonization Analysis

A multi-method empirical analysis of U.S. state-level renewable portfolio standards (RPS) and their causal effect on electricity sector CO₂ intensity, 2001–2022.

## Research Questions

| BH | Question | Method |
|----|----------|--------|
| BH1 | Does RPS adoption causally reduce CO₂ intensity? | Two-Way Fixed Effects + IV |
| BH2 | What is the ATT/ATE of RPS? Who benefits most? | Staggered DiD + Causal Forest |
| BH3 | Can LSTM reliably forecast CO₂ intensity multi-horizon? | LSTM (1/3/6-month horizons) |
| BH4 | What features predict top-decarbonizer state status? | XGBoost + SHAP + Stacking Ensemble |
| BH5 | Does RPS effect vary across Census regions? | Regional Subgroup TWFE |

## Data Sources (Public Government Only)

| Source | Data | URL |
|--------|------|-----|
| EIA | Monthly generation & consumption by state/fuel | https://www.eia.gov/electricity/data/browser/ |
| EPA | eGRID annual CO₂ intensity by state | https://www.epa.gov/egrid |
| NOAA | Monthly temperature by climate division | https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/ |
| BEA | State annual real GDP (SAGDP1) | https://apps.bea.gov/regional/ |
| DSIRE | Renewable Portfolio Standard adoption dates | https://www.dsireusa.org/ |

## Setup

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Place raw data files in `data/raw/` (see `src/data_loader.py` for expected filenames).

## Running the Analysis

```python
# 1. Build the master panel dataset
from src.data_loader import run_pipeline
run_pipeline()

# 2. Run each business hypothesis
from src.panel_models import run_bh1
from src.did_causal_forest import run_bh2
from src.lstm_forecaster import run_bh3
from src.xgboost_classifier import run_bh4
from src.regional_analysis import run_bh5

run_bh1()   # outputs/BH1/
run_bh2()   # outputs/BH2/
run_bh3()   # outputs/BH3/
run_bh4()   # outputs/BH4/
run_bh5()   # outputs/BH5/
```

## Project Structure

```
qm640_energy_analysis/
├── data/
│   ├── raw/          # Source files (not committed)
│   └── processed/    # Pipeline outputs
├── src/
│   ├── data_loader.py         # Data pipeline (14 functions)
│   ├── panel_models.py        # BH1: TWFE + IV
│   ├── did_causal_forest.py   # BH2: DiD + Causal Forest
│   ├── lstm_forecaster.py     # BH3: LSTM forecasting
│   ├── xgboost_classifier.py  # BH4: XGBoost + SHAP
│   └── regional_analysis.py   # BH5: Regional TWFE
├── notebooks/         # Original Google Colab notebooks
├── outputs/           # Figures and result CSVs by BH
├── synopsis/          # APA 7th edition synopsis content
├── docs/              # Compliance checklist
└── requirements.txt
```

## Key Results Summary

| BH | Primary Finding | Effect Size |
|----|-----------------|-------------|
| BH1 | RPS reduces CO₂ intensity (preferred Spec 3) | β = −0.00362, p = 0.017 |
| BH2 | ATT of RPS adoption | −0.134 SD, p < 0.0001 |
| BH3 | LSTM skill score negative at all horizons (near-unit-root CO₂ dynamics) | SS₁ = −0.043, R² = 0.971 |
| BH4 | Top predictor of decarbonizer status: Fossil_Intensity | Stacking AUC = 0.961 |
| BH5 | Largest RPS effect in West; positive (paradox) in Northeast | West β = −0.00859*** |

## Dependencies

- `linearmodels` — PanelOLS, IV-2SLS, IV-LIML
- `econml` — Causal Forest (GRF)
- `tensorflow` / `keras` — LSTM
- `xgboost`, `lightgbm` — gradient boosted trees
- `shap` — model interpretability
- `statsmodels`, `scikit-learn` — supporting statistics
