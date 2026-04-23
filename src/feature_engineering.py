"""
Feature Engineering Module
Adds derived features to FINAL_MASTER_DATASET_CLEAN.csv
Output: FINAL_MASTER_DATASET_FEATURES.csv

Usage:
    from src.feature_engineering import run_feature_engineering
    run_feature_engineering()
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_CLEAN.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_FEATURES.csv"

# ── Constants ───────────────────────────────────────────────────────────────
HDD_BASE = 65.0   # standard heating degree day base (°F)
CDD_BASE = 65.0   # standard cooling degree day base (°F)


def run_feature_engineering():
    print("=" * 70)
    print("  FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["STATE", "YEAR", "MONTH"]).reset_index(drop=True)
    n_orig = df.shape[1]
    log = []

    # ── 1. Clean_Share — combined low-carbon generation ─────────────────────
    print("\n[1] Clean_Share = Renewable_Share_Pct + Nuclear_Share_Pct")
    df["Clean_Share"] = df["Renewable_Share_Pct"].fillna(0) + df["Nuclear_Share_Pct"].fillna(0)
    df["Clean_Share"] = df["Clean_Share"].clip(0, 100)
    log.append({"Feature": "Clean_Share", "Formula": "Renewable + Nuclear share (%)",
                "Used in": "BH3, BH4, BH1/2/5 (control)"})
    print(f"   mean={df['Clean_Share'].mean():.1f}%  "
          f"min={df['Clean_Share'].min():.1f}%  max={df['Clean_Share'].max():.1f}%")

    # ── 2. RPS_Maturity — log-transformed years since adoption ──────────────
    print("\n[2] RPS_Maturity = log(1 + Years_Since_RPS)")
    df["RPS_Maturity"] = np.log1p(df["Years_Since_RPS"].fillna(0))
    log.append({"Feature": "RPS_Maturity", "Formula": "log(1 + Years_Since_RPS)",
                "Used in": "BH3, BH4"})
    print(f"   mean={df['RPS_Maturity'].mean():.3f}  "
          f"max={df['RPS_Maturity'].max():.3f}  "
          f"(years_max={df['Years_Since_RPS'].max():.0f})")

    # ── 3. Fossil_to_Renewable_Ratio ─────────────────────────────────────────
    print("\n[3] Fossil_to_Renewable_Ratio = Fossil_Intensity / (Renewable_Share_Pct + 1)")
    df["Fossil_to_Renewable_Ratio"] = (
        df["Fossil_Intensity"].fillna(0) / (df["Renewable_Share_Pct"].fillna(0) + 1)
    )
    log.append({"Feature": "Fossil_to_Renewable_Ratio",
                "Formula": "Fossil / (Renewable + 1)",
                "Used in": "BH3, BH4"})
    print(f"   mean={df['Fossil_to_Renewable_Ratio'].mean():.2f}  "
          f"min={df['Fossil_to_Renewable_Ratio'].min():.2f}  "
          f"max={df['Fossil_to_Renewable_Ratio'].max():.2f}")

    # ── 4. HDD / CDD — Heating and Cooling Degree Days ──────────────────────
    print("\n[4] HDD = max(0, 65 - Avg_Temp_F)  |  CDD = max(0, Avg_Temp_F - 65)")
    df["HDD"] = np.maximum(0, HDD_BASE - df["Avg_Temp_F"])
    df["CDD"] = np.maximum(0, df["Avg_Temp_F"] - CDD_BASE)
    log.append({"Feature": "HDD / CDD", "Formula": "max(0, 65−T) / max(0, T−65)",
                "Used in": "BH3, BH4, BH1/2/5 (control)"})
    print(f"   HDD: mean={df['HDD'].mean():.1f}  max={df['HDD'].max():.1f}")
    print(f"   CDD: mean={df['CDD'].mean():.1f}  max={df['CDD'].max():.1f}")

    # ── 5. CO2_YoY_Change — annual CO₂ % change per state ───────────────────
    print("\n[5] CO2_YoY_Change = annual % change in CO2_Intensity_Combined per state")
    annual_co2 = (df[df["MONTH"] == 1]
                  .groupby(["STATE", "YEAR"])["CO2_Intensity_Combined"]
                  .mean()
                  .reset_index())
    annual_co2 = annual_co2.sort_values(["STATE", "YEAR"])
    annual_co2["CO2_YoY_Change"] = (
        annual_co2.groupby("STATE")["CO2_Intensity_Combined"]
        .pct_change() * 100
    )
    df = df.merge(annual_co2[["STATE", "YEAR", "CO2_YoY_Change"]],
                  on=["STATE", "YEAR"], how="left")
    log.append({"Feature": "CO2_YoY_Change",
                "Formula": "Annual % change in CO2 per state",
                "Used in": "BH4 only"})
    print(f"   mean={df['CO2_YoY_Change'].mean():.2f}%  "
          f"std={df['CO2_YoY_Change'].std():.2f}%  "
          f"missing={df['CO2_YoY_Change'].isna().sum()}")

    # ── 6. Renewable_Momentum — 3-month rolling change in renewable share ────
    print("\n[6] Renewable_Momentum = 3-month rolling mean of Renewable_Share_Pct change")
    df["_Renew_diff"] = df.groupby("STATE")["Renewable_Share_Pct"].diff()
    df["Renewable_Momentum"] = (
        df.groupby("STATE")["_Renew_diff"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df = df.drop(columns=["_Renew_diff"])
    log.append({"Feature": "Renewable_Momentum",
                "Formula": "3-month rolling mean of ΔRenewable_Share",
                "Used in": "BH3 only"})
    print(f"   mean={df['Renewable_Momentum'].mean():.3f}  "
          f"std={df['Renewable_Momentum'].std():.3f}")

    # ── 7. Seasonal encoding — sin/cos of month ──────────────────────────────
    print("\n[7] Seasonal_Sin / Seasonal_Cos = sin/cos encoding of month")
    df["Seasonal_Sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["Seasonal_Cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    log.append({"Feature": "Seasonal_Sin / Seasonal_Cos",
                "Formula": "sin/cos(2π × month / 12)",
                "Used in": "BH3 only"})
    print(f"   Seasonal_Sin range: [{df['Seasonal_Sin'].min():.3f}, {df['Seasonal_Sin'].max():.3f}]")
    print(f"   Seasonal_Cos range: [{df['Seasonal_Cos'].min():.3f}, {df['Seasonal_Cos'].max():.3f}]")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_new = df.shape[1] - n_orig
    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"\n  New features added: {n_new}")
    print(f"  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    df_log = pd.DataFrame(log)
    print(df_log.to_string(index=False))

    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Saved: {OUT_PATH.name}")
    print("\n  FEATURE ENGINEERING COMPLETE ✅")
    return df


# ── Convenience accessors for each BH ───────────────────────────────────────

BH3_FEATURES = [
    "CO2_Intensity_Combined",   # target at lag
    "Fossil_Intensity",
    "Renewable_Share_Pct",
    "Has_RPS",
    "Avg_Temp_F",
    "Temp_Extreme",
    "Real_GDP_Millions",
    "Total_Generation_MWh",
    "RPS_Target_Pct",
    # engineered
    "Clean_Share",
    "RPS_Maturity",
    "Fossil_to_Renewable_Ratio",
    "HDD",
    "CDD",
    "Renewable_Momentum",
    "Seasonal_Sin",
    "Seasonal_Cos",
]

BH4_FEATURES = [
    "Fossil_Intensity",
    "Renewable_Share_Pct",
    "Has_RPS",
    "RPS_Target_Pct",
    "Avg_Temp_F",
    "Real_GDP_Millions",
    "Temp_Extreme",
    "Total_Generation_MWh",
    # engineered
    "Clean_Share",
    "RPS_Maturity",
    "Fossil_to_Renewable_Ratio",
    "HDD",
    "CDD",
    "CO2_YoY_Change",
]

BH_PANEL_EXTRA_CONTROLS = ["Clean_Share", "HDD", "CDD"]


if __name__ == "__main__":
    run_feature_engineering()
