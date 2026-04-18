# QM 640 Compliance Checklist

**Project:** U.S. Energy Transition & Decarbonization Analysis  
**Course:** QM 640 — Advanced Business Analytics  
**Date:** April 18, 2026  

---

## Checklist Item 1 — Business Problem First

**Requirement:** The analysis must be motivated by a clear business or policy problem, not a technical exercise.

**Status:** PASS

**Evidence:**

The project opens with a concrete dual-mandate business problem: the U.S. electricity sector must simultaneously decarbonize and maintain grid reliability underpinning $25 trillion in annual economic activity. Three stakeholder audiences are explicitly named with specific decision contexts:

1. **State regulators** — setting optimal RPS stringency and design
2. **Electric utilities** — capital allocation across renewable generation types
3. **Corporate sustainability officers** — Scope 2 emissions management and REC procurement

All five business hypotheses (BH1–BH5) are framed as answers to business decisions, not statistical exercises. The synopsis Section 5 (Recommendations) ties each empirical finding to an actionable decision with quantified business impact (e.g., $102,000/year avoided carbon cost per 50,000 MWh site; $39.4M NPV for a 50-site portfolio in RPS-adopting states).

---

## Checklist Item 2 — Public Data Sources Only

**Requirement:** All data must come from public, verifiable, non-proprietary sources. No Kaggle datasets.

**Status:** PASS

| Source | Type | URL | Data Used |
|--------|------|-----|-----------|
| U.S. Energy Information Administration (EIA) | Federal agency | https://www.eia.gov/ | Monthly electricity generation by state and fuel type; monthly consumption |
| U.S. EPA eGRID | Federal agency | https://www.epa.gov/egrid | Annual CO₂ intensity (lb/MWh) by state, 2004–2022 |
| NOAA Climate at a Glance | Federal agency | https://www.ncei.noaa.gov/ | Monthly average temperature by climate division, 1950–2022 |
| U.S. Bureau of Economic Analysis (BEA) | Federal agency | https://apps.bea.gov/ | State annual real GDP (SAGDP1), 1997–2024 |
| DSIRE (NC Clean Energy Technology Center) | University/DOE-funded | https://www.dsireusa.org/ | RPS adoption dates and mandated percentages by state |

**No Kaggle, no proprietary commercial databases, no paywalled sources.**

All raw files are stored locally at `data/raw/` and reproducible from the above URLs. File inventory:
- `generation_monthly.xlsx` (EIA)
- `consumption_monthly.xlsx` (EIA)
- `GDP_Table.xlsx` (BEA SAGDP1)
- `RPS_list.xlsx` (DSIRE)
- `average_monthly_temperature_by_state_1950-2022.csv` (NOAA)
- `eGRID2009_data.xls` through `egrid2023_data_rev2.xlsx` (EPA, 14 files)

---

## Checklist Item 3 — Sample Size Calculation

**Requirement:** Explicit power analysis for each business hypothesis.

**Status:** PASS

| BH | Method | Rule Applied | Required n | Actual n | Power Ratio |
|----|--------|-------------|-----------|---------|-------------|
| BH1 | Two-Way Fixed Effects | Green's Rule (50 + 8k, k=7) | 106 | 1,122 | **10.6×** |
| BH2 | Staggered DiD + Causal Forest | DiD panel power formula | 75 (medium f²) | 1,122 | **14.96×** |
| BH3 | LSTM | Sequence count ≥ 1,000 | 1,000 sequences | 7,701 (1-mo) | **7.7×** |
| BH4 | XGBoost Classification | EPV ≥ 10, k=8 features | 80 events | 374 events | **EPV = 46.8** |
| BH5 | Regional Subgroup TWFE | Green's Rule per region (smallest: Northeast, ~11 states × 22 years = 242) | 106 | 242–528 | **2.3–5×** |

Full derivations are in `synopsis/synopsis_content.md` Appendix A.

---

## Checklist Item 4 — Statistics + ML/AI Balance

**Requirement:** The project must balance traditional statistical methods with modern ML/AI techniques.

**Status:** PASS

**Statistical Methods (3):**

| Method | BH | File | Description |
|--------|-----|------|-------------|
| Two-Way Fixed Effects Panel Regression | BH1 | `src/panel_models.py` | Causal panel identification with state/year FE and state-specific trends |
| Instrumental Variables (IV-2SLS, IV-LIML) | BH1 | `src/panel_models.py` | Robustness to endogenous RPS adoption via spatial lag instrument |
| Staggered Difference-in-Differences | BH2 | `src/did_causal_forest.py` | Event-study with parallel trends validation, k = −3 to +10 |

**ML/AI Methods (4):**

| Method | BH | File | Description |
|--------|-----|------|-------------|
| Causal Forest (GRF) | BH2 | `src/did_causal_forest.py` | Heterogeneous treatment effect estimation via honest causal trees |
| LSTM Neural Network | BH3 | `src/lstm_forecaster.py` | Multi-horizon time series forecasting (1/3/6 months), 2-layer LSTM |
| XGBoost + SHAP | BH4 | `src/xgboost_classifier.py` | Binary classification with feature importance interpretability |
| Stacking Ensemble | BH4 | `src/xgboost_classifier.py` | XGBoost + LightGBM + RF → LogisticRegression meta-learner |

**Regional heterogeneity (BH5)** applies TWFE across Census regions, bridging statistical identification and strategic business segmentation.

---

## Checklist Item 5 — Actionable Insights

**Requirement:** Findings must translate to specific, quantified business recommendations.

**Status:** PASS

**Recommendation 1 (Regulators — West/South):** A 10 percentage-point RPS increase in a Western state reduces CO₂ intensity by ~0.086 short tons/MWh/year. Prioritize RPS stringency where resource endowments are richest.

**Recommendation 2 (Regulators — Northeast):** RPS alone shows near-zero marginal effect in the Northeast (β = +0.00134). Complement with carbon pricing and grid investment. This prevents misallocation of policy capital.

**Recommendation 3 (Regulators — Timeline):** Near-unit-root CO₂ dynamics (BH3 SS < 0) imply 5–10 year RPS compliance schedules outperform annual ratchets.

**Recommendation 4 (Utilities):** States with Fossil_Intensity > 60% face steep barriers to decarbonizer status (top SHAP predictor at 43.5% importance). Fuel-switching is the highest-leverage capital decision. Estimated 0.18 probability-unit gain per 10-point fossil intensity reduction.

**Recommendation 5 (Utilities — Risk Scoring):** Stacking Ensemble (AUC = 0.961) enables state-level REC procurement risk scoring. Non-decarbonizer states will face higher future compliance costs; use model scores to time REC purchases forward.

**Recommendation 6 (Corporate Sustainability):** ROI of locating in RPS-adopting vs. non-adopting states: $102,000/year avoided carbon cost per 50,000 MWh site; $39.4M NPV for 50-site portfolios over 10 years (@ $51/ton Social Cost of Carbon, 5% discount rate).

---

## Checklist Item 6 — GitHub Repository Structure

**Requirement:** Well-organized, reproducible repository with clear structure and documentation.

**Status:** PASS

```
qm640_energy_analysis/
├── README.md                          # Project overview, setup, usage, results
├── requirements.txt                   # Pinned dependencies
├── .gitignore                         # Excludes raw data, model artifacts, venv
├── data/
│   ├── raw/                           # Source files (gitignored — too large)
│   └── processed/                     # Pipeline outputs (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data pipeline — 14 documented functions
│   ├── panel_models.py                # BH1: 5 panel specifications + placebo
│   ├── did_causal_forest.py           # BH2: Event-study DiD + Causal Forest
│   ├── lstm_forecaster.py             # BH3: 2-layer LSTM, 3 forecast horizons
│   ├── xgboost_classifier.py          # BH4: XGBoost + SHAP + ensemble
│   └── regional_analysis.py           # BH5: Regional subgroup TWFE
├── notebooks/
│   ├── Final_data_loading.ipynb       # Original working Colab notebook
│   ├── BH1.ipynb                      # BH1 development notebook
│   ├── BH2.ipynb                      # BH2 development notebook
│   ├── BH3.ipynb                      # BH3 development notebook
│   ├── BH4.ipynb                      # BH4 development notebook
│   └── BH5.ipynb                      # BH5 development notebook
├── outputs/
│   ├── BH1/                           # BH1 figures and result tables
│   ├── BH2/                           # BH2 figures and result tables
│   ├── BH3/                           # BH3 model files, metrics, predictions
│   ├── BH4/                           # BH4 figures, SHAP plots, confusion matrix
│   └── BH5/                           # BH5 regional figures and tables
├── synopsis/
│   └── synopsis_content.md            # Full 8-10 page APA 7th edition content
└── docs/
    └── qm640_compliance_checklist.md  # This document
```

**Reproducibility:** Any analyst can clone the repository, install requirements, copy raw data files to `data/raw/`, and run `run_pipeline()` + `run_bh1()` through `run_bh5()` to reproduce all results from scratch.

**Code quality:** All `src/` modules use explicit function definitions, typed inputs, and save outputs deterministically to named subdirectories. No hardcoded local paths — `PROJECT_ROOT` is computed relative to the module location.

---

## Summary Status

| Item | Requirement | Status |
|------|-------------|--------|
| 1 | Business problem first | ✓ PASS |
| 2 | Public data sources only | ✓ PASS |
| 3 | Sample size calculation | ✓ PASS |
| 4 | Stats + ML/AI balance | ✓ PASS |
| 5 | Actionable insights | ✓ PASS |
| 6 | GitHub repository structure | ✓ PASS |

**All 6 compliance items satisfied.**
