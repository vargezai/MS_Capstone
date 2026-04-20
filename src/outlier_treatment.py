"""
Outlier Detection & Rectification
U.S. Energy Transition Panel Dataset
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_2001_2026.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_CLEAN.csv"
OUT_DIR      = PROJECT_ROOT / "outputs" / "EDA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


def run_outlier_treatment():
    print("=" * 70)
    print("  OUTLIER DETECTION & RECTIFICATION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    original_shape = df.shape
    log = []

    # ── ISSUE 1: Negative Total_Generation_MWh (DC pumped storage) ─────────
    print("\n[1] Negative Total_Generation_MWh")
    mask = df["Total_Generation_MWh"] < 0
    n = mask.sum()
    print(f"    Found {n} rows — DC pumped storage artefact")
    df.loc[mask, ["Total_Generation_MWh","Fossil_Generation_MWh",
                  "Renewable_Generation_MWh","Fossil_Share_Pct",
                  "Renewable_Share_Pct","Fossil_Intensity"]] = np.nan
    log.append({"Issue": "Negative Total_Generation_MWh", "N": n,
                "Treatment": "Set to NaN (DC pumped storage artefact)"})
    print(f"    → Set to NaN")

    # ── ISSUE 2: Negative Fossil_Intensity (DC 2018) ────────────────────────
    print("\n[2] Negative Fossil_Intensity")
    mask = df["Fossil_Intensity"] < 0
    n = mask.sum()
    print(f"    Found {n} rows")
    print(df.loc[mask, ["STATE","YEAR","MONTH","Fossil_Intensity"]].to_string(index=False))
    df.loc[mask, "Fossil_Intensity"] = 0.0
    log.append({"Issue": "Negative Fossil_Intensity", "N": n,
                "Treatment": "Clamped to 0 (pumped storage calculation artefact)"})
    print(f"    → Clamped to 0")

    # ── ISSUE 3: Negative Nuclear_Share_Pct ─────────────────────────────────
    print("\n[3] Negative Nuclear_Share_Pct")
    mask = df["Nuclear_Share_Pct"] < 0
    n = mask.sum()
    print(f"    Found {n} rows across {df.loc[mask,'STATE'].nunique()} states: "
          f"{list(df.loc[mask,'STATE'].unique())}")
    df.loc[mask, "Nuclear_Share_Pct"] = 0.0
    df.loc[mask, "Nuclear"] = 0.0
    df.loc[mask, "Nuclear_Generation_MWh"] = 0.0
    log.append({"Issue": "Negative Nuclear_Share_Pct", "N": n,
                "Treatment": "Clamped to 0 (EIA net-metering reporting artefact)"})
    print(f"    → Clamped to 0")

    # ── ISSUE 4: CO2_Intensity_Combined == 0 ───────────────────────────────
    print("\n[4] CO2_Intensity_Combined = 0")
    mask = df["CO2_Intensity_Combined"] == 0
    n = mask.sum()
    print(f"    Found {n} row(s)")
    print(df.loc[mask, ["STATE","YEAR","MONTH","CO2_Intensity_Combined"]].to_string(index=False))
    df.loc[mask, "CO2_Intensity_Combined"] = np.nan
    log.append({"Issue": "CO2_Intensity = 0", "N": n,
                "Treatment": "Set to NaN (implausible zero — missing eGRID entry)"})
    print(f"    → Set to NaN")

    # ── ISSUE 5: RPS_Target_Pct = 10,000 (TX MW encoding) ──────────────────
    print("\n[5] RPS_Target_Pct = 10,000 (TX all years)")
    mask = df["RPS_Target_Pct"] > 100
    n = mask.sum()
    print(f"    Found {n} rows — TX MW mandate encoded as 10000")
    print(f"    Corrected to 3.0% (5,880 MW ≈ 3% of TX generation)")
    df.loc[mask, "RPS_Target_Pct"] = 3.0
    log.append({"Issue": "RPS_Target_Pct=10000 (TX)", "N": n,
                "Treatment": "Replaced with 3.0 (TX MW mandate ≈ 3% generation equivalent)"})
    print(f"    → Set to 3.0%")

    # ── ISSUE 6: High CO2 > 1.1 — flag only (legitimate) ──────────────────
    print("\n[6] High CO2_Intensity_Combined > 1.1")
    mask_hi = df["CO2_Intensity_Combined"] > 1.1
    n = mask_hi.sum()
    print(f"    Found {n} rows — DC 2009, KY/OH/WV high-fossil states")
    print("    → LEGITIMATE — flagged only, no values changed")
    df["CO2_Outlier_Flag"] = mask_hi.astype(int)
    log.append({"Issue": "High CO2 > 1.1", "N": n,
                "Treatment": "Flagged only — legitimate 100% fossil small grid"})

    # ── ISSUE 7: High Renewable (hydro states) — no change ─────────────────
    print("\n[7] High Renewable_Share_Pct (IQR outliers)")
    Q3 = df["Renewable_Share_Pct"].quantile(0.75)
    IQR = Q3 - df["Renewable_Share_Pct"].quantile(0.25)
    mask_r = df["Renewable_Share_Pct"] > Q3 + 1.5 * IQR
    n = mask_r.sum()
    print(f"    Found {n} rows — WA, ID, OR, SD, ME, VT (hydro-heavy states)")
    print("    → LEGITIMATE — no change")
    log.append({"Issue": "High Renewable IQR outliers", "N": n,
                "Treatment": "No change — hydro-heavy states are legitimate"})

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OUTLIER RECTIFICATION SUMMARY")
    print("=" * 70)
    df_log = pd.DataFrame(log)
    print(df_log.to_string(index=False))

    df.to_csv(OUT_PATH, index=False)
    print(f"\n  Clean dataset saved: {OUT_PATH.name}")
    print(f"  Shape: {original_shape} → {df.shape}")

    # ── Before/after plot ───────────────────────────────────────────────────
    df_orig = pd.read_csv(DATA_PATH)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    checks = [
        ("Fossil_Intensity",       "Fossil Intensity (%)",      axes[0,0]),
        ("Nuclear_Share_Pct",      "Nuclear Share (%)",         axes[0,1]),
        ("CO2_Intensity_Combined", "CO₂ Intensity (tons/MWh)",  axes[0,2]),
        ("Total_Generation_MWh",   "Total Generation (MWh)",    axes[1,0]),
        ("RPS_Target_Pct",         "RPS Target (%)",            axes[1,1]),
        ("Renewable_Share_Pct",    "Renewable Share (%)",       axes[1,2]),
    ]
    for col, label, ax in checks:
        ax.hist(df_orig[col].dropna(), bins=40, alpha=0.55, color="#FF5722",
                label="Before", density=True)
        ax.hist(df[col].dropna(), bins=40, alpha=0.55, color="#2196F3",
                label="After", density=True)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("Outlier Treatment — Before vs After", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_08_outlier_treatment.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_log.to_csv(OUT_DIR / "EDA_outlier_log.csv", index=False)
    print(f"  Plot saved: EDA_08_outlier_treatment.png")
    print(f"  Log saved:  EDA_outlier_log.csv")
    print("\n  OUTLIER TREATMENT COMPLETE ✅")
    return df


if __name__ == "__main__":
    run_outlier_treatment()
