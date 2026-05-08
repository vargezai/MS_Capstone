# Executive Summary
## U.S. Energy Transition & Decarbonization Analysis
**QM 640 — Advanced Business Analytics Capstone | Walsh College**  
**Author:** Manoj Varghese | **Date:** May 2026

---

## The Business Problem

The U.S. electricity sector must cut carbon emissions while keeping the lights on for an economy worth $25 trillion a year. The main policy tool most states have used is a **Renewable Portfolio Standard (RPS)** — a law requiring utilities to generate a minimum share of electricity from renewable sources such as wind and solar. As of 2022, 30 states plus Washington D.C. have these rules in place.

But do they work? And if so, where, and for whom? This project answers those questions using 22 years of government data across all 50 states and five analytical methods.

---

## What We Studied

Using publicly available data from the EIA, EPA, NOAA, BEA, and DSIRE (2001–2022), we ran five complementary analyses, each designed to answer a different aspect of the decarbonization question:

| Question | Approach | Plain-Language Purpose |
|----------|----------|----------------------|
| Do RPS laws actually *cause* lower emissions? | Statistical panel model | Rule out coincidence — isolate the policy effect |
| Which states benefit most? | Causal comparison + machine learning | Find where the policy works best |
| Can we forecast CO₂ trends? | Deep learning (LSTM neural network) | Understand short-run vs. long-run dynamics |
| What predicts a "top decarbonizer" state? | XGBoost classification | Build a state-level risk scorecard |
| Does region matter? | Regional sub-analysis | Tailor policy to geography |

---

## Five Key Findings

### 1. Renewable Portfolio Standards Work — But the Effect Is Gradual
States that adopt RPS laws reduce their electricity sector CO₂ intensity by roughly **0.4% per year** compared to states without them. Over a decade, that compounds to an 8–12% lower carbon footprint than a comparable non-adopting state. The finding is statistically robust: it holds across six different model specifications, including tests that rule out pre-existing differences between states.

> **Bottom line for regulators:** RPS adoption is a proven lever. The question is not whether to use it, but how aggressively to set the target.

---

### 2. The Average Hides Large Regional Differences
The national average obscures a wide spread. Western and Southern states see the biggest reductions when they adopt or strengthen RPS; Northeastern states see near-zero benefit from RPS alone.

| Region | Effect of RPS on CO₂ |
|--------|---------------------|
| West | Strong reduction ✓ |
| South | Strong reduction ✓ |
| Midwest | Moderate reduction ✓ |
| Northeast | Near-zero / slightly positive |

The Northeast paradox is not because RPS is harmful there — it reflects that the region already relies heavily on nuclear power and natural gas, leaving little room for incremental carbon reduction through renewable mandates alone. These states need complementary policies: carbon pricing, grid investment, and demand flexibility programs.

> **Bottom line for regulators:** A one-size-fits-all national RPS mandate will underdeliver. Design targets by region.

---

### 3. Short-Term CO₂ Forecasting Is Hard — Structural Change Is What Matters
A state's CO₂ intensity this month is almost entirely explained by last month's value (correlation ≈ 0.85). Even a sophisticated deep learning model cannot reliably outperform that simple rule for 1–6 month forecasts. This tells us something important: **short-run fluctuations in CO₂ are dominated by noise and weather, not policy.** Policy effects show up over years, not months.

> **Bottom line for sustainability managers:** Do not set quarterly carbon targets expecting policy changes to show up in the numbers within the year. Plan on 3–5 year horizons for measurable impact.

---

### 4. A Machine Learning Model Can Score States for Decarbonization Risk with 97% Accuracy
Our XGBoost classifier predicts whether a state will become a "top decarbonizer" (bottom third for CO₂ intensity) with a cross-validated accuracy of 97% (AUC = 0.97). The top predictors, in order of importance:

1. **RPS target stringency** — how ambitious the renewable mandate is (17% of the model's explanatory power)
2. **Fossil fuel share of the grid** — states with >60% fossil generation face a structural barrier
3. **Combined clean energy share** (renewables + nuclear) — both matter, not just renewables
4. **Climate/demand patterns** — cooling and heating degree days shape grid mix at peak times

> **Bottom line for utilities:** Use this model as a state-level risk scorecard. States predicted as non-decarbonizers will face rising compliance costs. Procure renewable energy certificates (RECs) early in those markets.

---

### 5. RPS Adoption Has a Quantifiable Dollar Value for Corporate Facilities
The treatment effect from analysis #2 implies that locating a commercial facility in an RPS-adopting state versus a non-adopting state produces measurable avoided carbon costs:

| Scale | Annual Saving | 10-Year NPV (5% discount) |
|-------|--------------|--------------------------|
| Single 50,000 MWh/yr site | **$102,000/year** | $787,000 |
| Portfolio of 50 sites | **$5.1 million/year** | $39.4 million |

*Assumes $51/ton Social Cost of Carbon (EPA 2023 rate) and 2,000 short tons/year avoided.*

> **Bottom line for corporate sustainability:** The financial case for siting operations in RPS-adopting states is real and quantifiable. Include state RPS trajectory in site selection criteria alongside energy price and labor cost.

---

## Summary of Results

| Analysis | Question | Answer |
|----------|----------|--------|
| BH1 — Panel Model | Does RPS causally reduce CO₂? | **Yes** — β = −0.0036/year, p = 0.017 |
| BH2 — Causal Comparison | What is the ATT of RPS adoption? | **−13.4% of a standard deviation**, p < 0.001 |
| BH3 — Deep Learning Forecast | Can LSTM beat naive persistence? | **No** — CO₂ is near-random-walk short-term |
| BH4 — Classification Model | What predicts top decarbonizers? | **97% AUC** — RPS stringency is #1 driver |
| BH5 — Regional Analysis | Does region change the answer? | **Yes** — 6× variation West vs. Northeast |

---

## Three Actions Decision-Makers Should Take Now

**1. Prioritize RPS stringency in the West and South.**  
These regions have the resource endowments and the regulatory runway to deliver the largest CO₂ reductions per percentage-point increase in the RPS target. A 10-point increase in a Western state's RPS target is estimated to reduce CO₂ intensity by 0.086 short tons/MWh annually.

**2. Pair Northeastern RPS with carbon pricing.**  
The data shows that RPS alone is insufficient in the Northeast. Carbon pricing, grid interconnection investment, and demand-side flexibility programs are needed to unlock additional reductions in a region already at the technological frontier.

**3. Use the state risk scorecard for capital and procurement decisions.**  
The XGBoost decarbonizer model turns 14 observable variables into a probability score for each state. Utilities should use this to time renewable investments; corporate sustainability officers should use it to prioritize REC procurement ahead of compliance cost increases in non-decarbonizer states.

---

## Data and Reproducibility

All data used in this project are freely available from U.S. government sources (EIA, EPA, NOAA, BEA, DSIRE). No proprietary or commercial datasets were used. The full analysis code and processed datasets are available in the project repository.

---

*For technical details, methodology, and full statistical tables, see `synopsis/synopsis_content.md`.*  
*For compliance documentation, see `docs/qm640_compliance_checklist.md`.*
