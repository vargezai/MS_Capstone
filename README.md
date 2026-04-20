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
| BEA | State quarterly real GDP (SQGDP1) | https://apps.bea.gov/regional/ |
| DSIRE | Renewable Portfolio Standard adoption dates | https://www.dsireusa.org/ |

## Setup

```bash
# Create conda environment (recommended — avoids NumPy version conflicts)
conda create -n qm640 python=3.11 -y
conda activate qm640
pip install -r requirements.txt
brew install libomp   # macOS only — required for XGBoost

# Place raw data files in data/raw/
```

## Running the Analysis

```python
# Step 1 — Build master panel dataset
from src.data_loader import main
main()                  # → data/processed/FINAL_MASTER_DATASET_2001_2026.csv

# Step 2 — Outlier detection & rectification
from src.outlier_treatment import run_outlier_treatment
run_outlier_treatment() # → data/processed/FINAL_MASTER_DATASET_CLEAN.csv

# Step 3 — Exploratory Data Analysis
from src.eda import run_eda
run_eda()               # → outputs/EDA/ (8 figures + 2 CSVs)

# Step 4 — Run business hypotheses (all use clean dataset)
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
│   ├── raw/                           # Source files (gitignored)
│   └── processed/
│       ├── FINAL_MASTER_DATASET_2001_2026.csv   # Raw pipeline output
│       ├── FINAL_MASTER_DATASET_CLEAN.csv        # After outlier treatment
│       ├── FINAL_COMPACT_DATASET_2001_2026.csv
│       └── SUMMARY_STATISTICS.csv
├── src/
│   ├── data_loader.py         # Data pipeline (14 functions)
│   ├── outlier_treatment.py   # Outlier detection & rectification (7 issues)
│   ├── eda.py                 # Exploratory Data Analysis (8 plots)
│   ├── panel_models.py        # BH1: TWFE + IV
│   ├── did_causal_forest.py   # BH2: DiD + Causal Forest
│   ├── lstm_forecaster.py     # BH3: LSTM forecasting
│   ├── xgboost_classifier.py  # BH4: XGBoost + SHAP
│   └── regional_analysis.py   # BH5: Regional TWFE
├── outputs/
│   ├── EDA/   # 8 EDA figures + outlier log
│   ├── BH1/   # Coefficient plots, results table
│   ├── BH2/   # Event-study, CATE plots
│   ├── BH3/   # LSTM models (.keras), metrics, predictions
│   ├── BH4/   # SHAP plots, confusion matrix, feature importance
│   └── BH5/   # Regional figures, interaction results
├── synopsis/
│   └── synopsis_content.md    # APA 7th edition synopsis (H₀/H₁ for all BHs)
├── docs/
│   └── qm640_compliance_checklist.md
├── requirements.txt
└── README.md
```

## Outlier Treatment Summary

7 data quality issues identified and resolved before analysis:

| Issue | Rows | Treatment |
|-------|------|-----------|
| Negative `Total_Generation_MWh` | 22 | Set to NaN — DC pumped storage artefact |
| Negative `Fossil_Intensity` | 3 | Clamped to 0 — DC 2018 calculation artefact |
| Negative `Nuclear_Share_Pct` | 38 | Clamped to 0 — EIA net-metering rounding (7 states) |
| `CO2_Intensity = 0` | 1 | Set to NaN — DC Oct 2017 implausible zero |
| `RPS_Target_Pct = 10,000` (TX) | 301 | Replaced with 3.0% — TX uses MW mandate, not % |
| `CO2 > 1.1` (DC 2009, KY, OH, WV) | 50 | Flagged only — legitimate 100% fossil grids |
| High renewable share (WA, ID, OR…) | 1,652 | No change — legitimate hydro-heavy states |

## EDA Key Findings

- **Fossil intensity** is the dominant correlate with CO₂ intensity (r = +0.830)
- Raw CO₂ gap: RPS states average **0.173 tons/MWh lower** than no-RPS states (unadjusted)
- Renewable share: r = −0.587 with CO₂; Nuclear: r = −0.346
- Temperature has near-zero raw correlation (r = +0.049)

## Key Results Summary

| BH | H₀ | Primary Finding | Decision |
|----|-----|-----------------|----------|
| BH1 | β_RPS = 0 | RPS reduces CO₂ intensity (Spec 3) | Reject H₀ (β = −0.00373, p = 0.016) |
| BH2 | ATT = 0 | Causal effect confirmed, parallel trends hold | Reject H₀ (ATT = −0.139, p < 0.0001) |
| BH3 | SS ≤ 0 | LSTM cannot beat persistence (near-unit-root) | Fail to reject H₀ (SS₁ = −2.52) |
| BH4 | AUC = 0.50 | Fossil_Intensity is top predictor (38.9% SHAP) | Reject H₀ (CV AUC = 0.956) |
| BH5 | Homogeneous effect | West β = −0.008 vs Northeast β = +0.001 | Reject H₀ (RPS×West p < 0.001) |

## Dependencies

- `linearmodels` — PanelOLS, IV-2SLS, IV-LIML
- `econml` — Causal Forest (GRF)
- `tensorflow` / `keras` — LSTM
- `xgboost==3.0.0`, `lightgbm` — gradient boosted trees
- `shap==0.51.0` — model interpretability
- `statsmodels`, `scikit-learn` — supporting statistics
