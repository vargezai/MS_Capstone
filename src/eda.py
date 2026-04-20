"""
EDA — Exploratory Data Analysis
U.S. Energy Transition Panel Dataset (2001–2026)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_CLEAN.csv"
OUT_DIR      = PROJECT_ROOT / "outputs" / "EDA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

CENSUS_REGION = {
    "CT":"Northeast","ME":"Northeast","MA":"Northeast","NH":"Northeast",
    "NJ":"Northeast","NY":"Northeast","PA":"Northeast","RI":"Northeast","VT":"Northeast",
    "AL":"South","AR":"South","DC":"South","DE":"South","FL":"South","GA":"South",
    "KY":"South","LA":"South","MD":"South","MS":"South","NC":"South","OK":"South",
    "SC":"South","TN":"South","TX":"South","VA":"South","WV":"South",
    "IL":"Midwest","IN":"Midwest","IA":"Midwest","KS":"Midwest","MI":"Midwest",
    "MN":"Midwest","MO":"Midwest","NE":"Midwest","ND":"Midwest","OH":"Midwest",
    "SD":"Midwest","WI":"Midwest",
    "AK":"West","AZ":"West","CA":"West","CO":"West","HI":"West","ID":"West",
    "MT":"West","NV":"West","NM":"West","OR":"West","UT":"West","WA":"West","WY":"West",
}
REG_COLORS = {"Northeast":"#4CAF50","South":"#FF9800","Midwest":"#2196F3","West":"#9C27B0"}


def run_eda():
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS — U.S. ENERGY TRANSITION PANEL")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["Census_Region"] = df["STATE"].map(CENSUS_REGION)
    annual = (df.groupby(["STATE", "YEAR"])
                .agg(CO2=("CO2_Intensity_Combined", "mean"),
                     Fossil=("Fossil_Intensity", "mean"),
                     Renewable=("Renewable_Share_Pct", "mean"),
                     Nuclear=("Nuclear_Share_Pct", "mean"),
                     Has_RPS=("Has_RPS", "max"),
                     RPS_Pct=("RPS_Target_Pct", "max"),
                     Temp=("Avg_Temp_F", "mean"),
                     GDP=("Real_GDP_Millions", "mean"),
                     GDP_g=("GDP_Growth_Rate_Annual", "mean"),
                     Region=("Census_Region", "first"))
                .reset_index())

    print(f"\n  Master dataset : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Annual panel   : {annual.shape[0]:,} state-years | "
          f"{annual['STATE'].nunique()} states | {annual['YEAR'].nunique()} years")
    print(f"  Date range     : {df['YEAR'].min()}–{df['YEAR'].max()}")
    print(f"  RPS states     : {(annual.groupby('STATE')['Has_RPS'].max()==1).sum()}")

    # ── 1. Summary statistics ───────────────────────────────────────────────
    print("\n[1/8] Summary statistics...")
    key_vars = ["CO2_Intensity_Combined", "Fossil_Intensity", "Renewable_Share_Pct",
                "Nuclear_Share_Pct", "Avg_Temp_F", "Real_GDP_Millions",
                "Has_RPS", "RPS_Target_Pct", "CO2_Intensity_Proxy"]
    stats = df[key_vars].describe().T
    stats["missing"] = df[key_vars].isnull().sum()
    stats["missing%"] = (stats["missing"] / len(df) * 100).round(1)
    stats.to_csv(OUT_DIR / "EDA_summary_statistics.csv")
    print(stats[["count","mean","std","min","50%","max","missing%"]].to_string())

    # ── 2. Target variable distribution ────────────────────────────────────
    print("\n[2/8] Target variable distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.hist(annual["CO2"].dropna(), bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(annual["CO2"].mean(), color="red", linestyle="--", label=f"Mean={annual['CO2'].mean():.3f}")
    ax.axvline(annual["CO2"].median(), color="orange", linestyle="--", label=f"Median={annual['CO2'].median():.3f}")
    ax.set_xlabel("CO₂ Intensity (tons/MWh)")
    ax.set_title("Distribution of CO₂ Intensity")
    ax.legend(fontsize=9)

    ax = axes[1]
    rps_label = annual["Has_RPS"].map({1: "RPS States", 0: "No-RPS States"})
    for label, color in [("RPS States","#2196F3"), ("No-RPS States","#FF5722")]:
        vals = annual.loc[rps_label == label, "CO2"].dropna()
        ax.hist(vals, bins=30, alpha=0.6, label=f"{label} (n={len(vals)})", color=color)
    ax.set_xlabel("CO₂ Intensity (tons/MWh)")
    ax.set_title("CO₂ Intensity by RPS Status")
    ax.legend(fontsize=9)

    ax = axes[2]
    trend = annual.groupby("YEAR")["CO2"].mean()
    ax.plot(trend.index, trend.values, "o-", color="#2196F3", linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Mean CO₂ Intensity")
    ax.set_title("National Mean CO₂ Intensity Over Time")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    fig.suptitle("CO₂ Intensity — Target Variable Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_01_target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 3. Missing data ─────────────────────────────────────────────────────
    print("[3/8] Missing data analysis...")
    miss_pct = df[key_vars].isnull().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_bar = ["#e74c3c" if v > 20 else "#f39c12" if v > 5 else "#2ecc71"
                  for v in miss_pct.values]
    miss_pct.plot(kind="barh", ax=ax, color=colors_bar)
    ax.axvline(5, color="orange", linestyle="--", alpha=0.7, label="5% threshold")
    ax.axvline(20, color="red", linestyle="--", alpha=0.7, label="20% threshold")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Data by Variable")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_02_missing_data.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 4. Correlation matrix ───────────────────────────────────────────────
    print("[4/8] Correlation matrix...")
    corr_vars = ["CO2", "Fossil", "Renewable", "Nuclear", "Temp", "GDP", "Has_RPS", "RPS_Pct"]
    corr = annual[corr_vars].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Pearson Correlation Matrix — Key Variables", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_03_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    corr.to_csv(OUT_DIR / "EDA_correlation_matrix.csv")

    print("  Key correlations with CO₂ Intensity:")
    print(corr["CO2"].drop("CO2").sort_values().to_string())

    # ── 5. Generation mix ───────────────────────────────────────────────────
    print("[5/8] Generation mix trends...")
    mix = annual.groupby("YEAR")[["Fossil","Renewable","Nuclear"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.stackplot(mix.index, mix["Fossil"], mix["Renewable"], mix["Nuclear"],
                 labels=["Fossil","Renewable","Nuclear"],
                 colors=["#c0392b","#27ae60","#f39c12"], alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.set_title("National Generation Mix Over Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax = axes[1]
    for label, col, color in [("Fossil","Fossil","#c0392b"),
                                ("Renewable","Renewable","#27ae60"),
                                ("Nuclear","Nuclear","#f39c12")]:
        ax.plot(mix.index, mix[col], "o-", label=label, color=color, linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.set_title("Generation Mix Trends")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("U.S. Electricity Generation Mix (2001–2026)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_04_generation_mix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 6. RPS adoption ─────────────────────────────────────────────────────
    print("[6/8] RPS adoption analysis...")
    rps_counts = df.groupby("YEAR")["Has_RPS"].apply(
        lambda x: (x.groupby(df.loc[x.index, "STATE"]).max() == 1).sum()
    ).reset_index()
    rps_counts.columns = ["YEAR", "N_RPS_States"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(rps_counts["YEAR"], rps_counts["N_RPS_States"],
           color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Year"); ax.set_ylabel("States with Active RPS")
    ax.set_title("RPS Adoption Over Time")

    ax = axes[1]
    rps_co2 = annual.groupby(["YEAR","Has_RPS"])["CO2"].mean().unstack()
    if 0 in rps_co2.columns and 1 in rps_co2.columns:
        ax.plot(rps_co2.index, rps_co2[0], "o-", color="#FF5722", label="No RPS", linewidth=2)
        ax.plot(rps_co2.index, rps_co2[1], "s-", color="#2196F3", label="RPS", linewidth=2)
        ax.fill_between(rps_co2.index, rps_co2[0], rps_co2[1], alpha=0.12, color="grey")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean CO₂ Intensity (tons/MWh)")
    ax.set_title("CO₂ Intensity: RPS vs No-RPS States")
    ax.legend(fontsize=9)

    fig.suptitle("Renewable Portfolio Standards — Adoption & Effect", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_05_rps_adoption.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 7. Regional analysis ────────────────────────────────────────────────
    print("[7/8] Regional breakdown...")
    reg_annual = annual.dropna(subset=["Region"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for reg, color in REG_COLORS.items():
        sub = reg_annual[reg_annual["Region"] == reg].groupby("YEAR")["CO2"].mean()
        ax.plot(sub.index, sub.values, "o-", label=reg, color=color, linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Mean CO₂ Intensity")
    ax.set_title("CO₂ Intensity by Census Region")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    order = reg_annual.groupby("Region")["Renewable"].median().sort_values(ascending=False).index
    sns.boxplot(data=reg_annual[["Region","Renewable"]].dropna(),
                x="Region", y="Renewable", order=order, palette=REG_COLORS, ax=ax)
    ax.set_title("Renewable Share by Region")
    ax.set_ylabel("Renewable Share (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax = axes[1, 0]
    sns.violinplot(data=reg_annual[["Region","Fossil"]].dropna(),
                   x="Region", y="Fossil", order=order,
                   palette=REG_COLORS, ax=ax, inner="box")
    ax.set_title("Fossil Intensity by Region")
    ax.set_ylabel("Fossil Share (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax = axes[1, 1]
    rps_region = reg_annual.groupby(["Region","YEAR"])["Has_RPS"].mean() * 100
    for reg, color in REG_COLORS.items():
        if reg in rps_region.index.get_level_values(0):
            sub = rps_region[reg]
            ax.plot(sub.index, sub.values, "o-", label=reg, color=color, linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("% States with RPS")
    ax.set_title("RPS Adoption Rate by Region")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    fig.suptitle("Regional Heterogeneity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "EDA_06_regional_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── 8. Pairplot ─────────────────────────────────────────────────────────
    print("[8/8] Pairplot...")
    pair_vars = ["CO2", "Fossil", "Renewable", "Nuclear", "Temp", "GDP_g"]
    pair_df = annual[pair_vars + ["Has_RPS"]].dropna()
    pair_df["RPS"] = pair_df["Has_RPS"].map({1:"RPS", 0:"No RPS"})
    g = sns.pairplot(pair_df[pair_vars + ["RPS"]], hue="RPS",
                     palette={"RPS":"#2196F3","No RPS":"#FF5722"},
                     plot_kws={"alpha":0.4, "s":15},
                     diag_kind="kde", corner=True)
    g.fig.suptitle("Pairplot — Key Variables by RPS Status", y=1.01,
                   fontsize=13, fontweight="bold")
    g.fig.savefig(OUT_DIR / "EDA_07_pairplot.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 70)
    print("  EDA SUMMARY")
    print("=" * 70)
    print(f"\n  CO₂ Intensity (tons/MWh):")
    print(f"    National mean  : {annual['CO2'].mean():.4f}")
    print(f"    RPS states     : {annual.loc[annual['Has_RPS']==1,'CO2'].mean():.4f}")
    print(f"    No-RPS states  : {annual.loc[annual['Has_RPS']==0,'CO2'].mean():.4f}")
    print(f"    Raw gap        : {annual.loc[annual['Has_RPS']==0,'CO2'].mean() - annual.loc[annual['Has_RPS']==1,'CO2'].mean():.4f}")
    print(f"\n  Strongest correlates with CO₂:")
    top = corr["CO2"].drop("CO2").abs().sort_values(ascending=False)
    for var, val in top.items():
        direction = "+" if corr["CO2"][var] > 0 else "-"
        print(f"    {var:<15}: r = {direction}{val:.3f}")
    print(f"\n  Outputs saved to: {OUT_DIR}")
    print("\n  EDA COMPLETE ✅")


if __name__ == "__main__":
    run_eda()
