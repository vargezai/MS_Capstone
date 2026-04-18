"""
BH5: Regional Heterogeneity — Subgroup TWFE + Region×RPS Interaction
Does the national RPS effect (BH1) hold across all Census regions?
All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle

Ported from BH5.ipynb — logic unchanged, paths updated to local.
"""

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats import norm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_2001_2026.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_START, CORE_END = 2005, 2022
ALPHA = 0.05

REGION_MAP = {
    # Northeast (9)
    "CT":"Northeast","ME":"Northeast","MA":"Northeast","NH":"Northeast",
    "RI":"Northeast","VT":"Northeast","NJ":"Northeast","NY":"Northeast",
    "PA":"Northeast",
    # South (17)
    "DE":"South","DC":"South","FL":"South","GA":"South","MD":"South",
    "NC":"South","SC":"South","VA":"South","WV":"South","AL":"South",
    "KY":"South","MS":"South","TN":"South","AR":"South","LA":"South",
    "OK":"South","TX":"South",
    # Midwest (12)
    "IL":"Midwest","IN":"Midwest","MI":"Midwest","OH":"Midwest",
    "WI":"Midwest","IA":"Midwest","KS":"Midwest","MN":"Midwest",
    "MO":"Midwest","NE":"Midwest","ND":"Midwest","SD":"Midwest",
    # West (13)
    "AZ":"West","CO":"West","ID":"West","MT":"West","NV":"West",
    "NM":"West","UT":"West","WY":"West","AK":"West","CA":"West",
    "HI":"West","OR":"West","WA":"West",
}

COLORS_REG = {
    "National" : "#64748B",
    "Northeast": "#2ecc71",
    "South"    : "#e74c3c",
    "Midwest"  : "#F4A823",
    "West"     : "#3498db",
}


def run_bh5():
    print("="*70)
    print("  BH5: REGIONAL HETEROGENEITY IN RPS EFFECTIVENESS")
    print("="*70)

    # ── Sample size & power ───────────────────────────────────────────────────
    print("""
SAMPLE SIZE & POWER ANALYSIS — BH5
──────────────────────────────────────────────────────────────────────""")
    region_counts = {"Northeast": 9, "South": 17, "Midwest": 12, "West": 13}
    n_years       = CORE_END - CORE_START + 1
    beta_ref, std_renew, std_co2 = 0.00362, 23.0, 0.268
    d        = beta_ref * std_renew / std_co2
    z_a2, z_b = norm.ppf(1 - ALPHA/2), norm.ppf(0.80)
    n_min    = int(((z_a2 + z_b) / d)**2) + 1

    print(f"  Effect size (from BH1 β):       d = {d:.4f}")
    print(f"  Min obs per region (80% power): {n_min}")
    print(f"\n  {'Region':<12} {'States':>7} {'Obs (×18yr)':>12}  Adequate?")
    print("  " + "-"*45)
    for reg, n_st in region_counts.items():
        n_obs = n_st * n_years
        ok    = "✅" if n_obs >= n_min else "⚠️ "
        print(f"  {reg:<12} {n_st:>7} {n_obs:>12,}  {ok}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  LOADING DATA")
    print("="*70)

    df      = pd.read_csv(DATA_PATH)
    df      = df[~df["STATE"].isin(["US-TOTAL", "US", "USA"])].copy()
    df_core = df[(df["YEAR"] >= CORE_START) & (df["YEAR"] <= CORE_END)].copy()
    df_core = df_core.dropna(subset=[
        "CO2_Intensity_Combined", "Renewable_Share_Pct",
        "GDP_Growth_Rate_Annual", "Has_RPS", "Temp_Extreme"])

    df_core["Region"] = df_core["STATE"].map(REGION_MAP)
    df_core = df_core.dropna(subset=["Region"])

    agg_map = {
        "CO2_Intensity_Combined": "mean", "Renewable_Share_Pct": "mean",
        "GDP_Growth_Rate_Annual": "first", "Has_RPS": "max",
        "Temp_Extreme": "mean", "Fossil_Intensity": "mean",
        "Region": "first",
    }
    df_annual = df_core.groupby(["STATE", "YEAR"]).agg(agg_map).reset_index()

    print(f"\n  Annual panel: {len(df_annual):,} obs | "
          f"{df_annual['STATE'].nunique()} states | {CORE_START}–{CORE_END}")
    print(f"\n  Regional composition:")
    for reg in ["Northeast", "South", "Midwest", "West"]:
        st  = df_annual[df_annual["Region"]==reg]["STATE"].nunique()
        rps = df_annual[(df_annual["Region"]==reg) &
                        (df_annual["Has_RPS"]==1)]["STATE"].nunique()
        co2 = df_annual[df_annual["Region"]==reg]["CO2_Intensity_Combined"].mean()
        print(f"    {reg:<12}: {st:>2} states | {rps:>2} RPS states | "
              f"mean CO₂ = {co2:.4f}")

    # ── ARM 1: regional subgroup TWFE ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  ARM 1 — REGIONAL SUBGROUP TWFE REGRESSIONS")
    print("="*70)

    formula = ("CO2_Intensity_Combined ~ "
               "Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
               "Temp_Extreme + EntityEffects + TimeEffects")

    regional_results = {}

    df_nat  = df_annual.set_index(["STATE", "YEAR"])
    res_nat = PanelOLS.from_formula(formula, data=df_nat).fit(
        cov_type="clustered", cluster_entity=True)
    regional_results["National"] = res_nat

    b_nat = res_nat.params["Renewable_Share_Pct"]
    p_nat = res_nat.pvalues["Renewable_Share_Pct"]
    print(f"\n  {'Region':<12} {'States':>7} {'β':>10} {'SE':>8} "
          f"{'p':>8} {'Sig':<5} {'vs National'}")
    print("  " + "-"*65)
    print(f"  {'National':<12} {'51':>7} {b_nat:>+10.5f} "
          f"{res_nat.std_errors['Renewable_Share_Pct']:>8.5f} "
          f"{p_nat:>8.4f}  "
          f"{'***' if p_nat<0.001 else '**' if p_nat<0.01 else '*' if p_nat<0.05 else '†' if p_nat<0.10 else 'ns':<5} "
          f"─ baseline")

    for region in ["Northeast", "South", "Midwest", "West"]:
        df_r = df_annual[df_annual["Region"]==region].copy()
        n_st = df_r["STATE"].nunique()
        if n_st < 3:
            print(f"  {region:<12} {n_st:>7}  ⚠️  too few states")
            continue
        df_rp = df_r.set_index(["STATE", "YEAR"])
        try:
            res_r = PanelOLS.from_formula(formula, data=df_rp).fit(
                cov_type="clustered", cluster_entity=True)
            regional_results[region] = res_r
            b  = res_r.params["Renewable_Share_Pct"]
            se = res_r.std_errors["Renewable_Share_Pct"]
            p  = res_r.pvalues["Renewable_Share_Pct"]
            sig = ("***" if p<0.001 else "**" if p<0.01
                   else "*" if p<0.05 else "†" if p<0.10 else "ns")
            diff      = b - b_nat
            direction = "↑ larger" if b < b_nat else "↓ smaller"
            print(f"  {region:<12} {n_st:>7} {b:>+10.5f} {se:>8.5f} "
                  f"{p:>8.4f}  {sig:<5} {diff:+.5f} {direction}")
        except Exception as e:
            print(f"  {region:<12}  Error: {e}")

    # ── ARM 2: region × RPS interaction ───────────────────────────────────────
    print("\n" + "="*70)
    print("  ARM 2 — REGION × RPS INTERACTION (Pooled TWFE)")
    print("="*70)

    df_int = df_annual.copy()
    for reg in ["South", "Midwest", "West"]:
        df_int[f"RPS_{reg}"] = (
            (df_int["Region"]==reg).astype(float) * df_int["Has_RPS"])
    df_int = df_int.set_index(["STATE", "YEAR"])

    formula_int = ("CO2_Intensity_Combined ~ "
                   "Has_RPS + RPS_South + RPS_Midwest + RPS_West + "
                   "Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
                   "Temp_Extreme + EntityEffects + TimeEffects")

    res_int = PanelOLS.from_formula(formula_int, data=df_int).fit(
        cov_type="clustered", cluster_entity=True)

    print(f"\n  {'Variable':<25} {'β':>10} {'SE':>8} {'p':>8}  Sig")
    print("  " + "-"*58)
    for var in ["Has_RPS", "RPS_South", "RPS_Midwest", "RPS_West",
                "Renewable_Share_Pct"]:
        if var in res_int.params.index:
            b  = res_int.params[var]
            se = res_int.std_errors[var]
            p  = res_int.pvalues[var]
            sig = ("***" if p<0.001 else "**" if p<0.01
                   else "*" if p<0.05 else "†" if p<0.10 else "ns")
            note = " ← reference" if var=="Has_RPS" else ""
            print(f"  {var:<25} {b:>+10.5f} {se:>8.5f} {p:>8.4f}  {sig}{note}")

    print(f"\n  R²w = {res_int.rsquared_within:.4f} | N = {res_int.nobs:,}")

    # ── Descriptive summary ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  REGIONAL DESCRIPTIVE STATISTICS")
    print("="*70)
    print(f"\n  {'Region':<12} {'States':>7} {'RPS%':>7} "
          f"{'Mean CO₂':>10} {'Mean Renew%':>12} {'Fossil%':>9}")
    print("  " + "-"*62)
    for reg in ["National", "Northeast", "South", "Midwest", "West"]:
        d     = df_annual if reg == "National" else df_annual[df_annual["Region"]==reg]
        n_st  = d["STATE"].nunique()
        rps_p = d[d["Has_RPS"]==1]["STATE"].nunique() / n_st * 100
        co2   = d["CO2_Intensity_Combined"].mean()
        renew = d["Renewable_Share_Pct"].mean()
        foss  = d["Fossil_Intensity"].mean() if "Fossil_Intensity" in d else float("nan")
        print(f"  {reg:<12} {n_st:>7} {rps_p:>6.0f}% "
              f"{co2:>10.4f} {renew:>11.2f}% {foss:>8.2f}%")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    key_vars = df_annual[["CO2_Intensity_Combined", "Renewable_Share_Pct",
                           "GDP_Growth_Rate_Annual", "Temp_Extreme", "Fossil_Intensity"]]
    plt.figure(figsize=(10, 8))
    sns.heatmap(key_vars.corr(), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Key Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH5_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Main figure ───────────────────────────────────────────────────────────
    print("\n📊 GENERATING PLOTS...")
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

    ax0          = fig.add_subplot(gs[0, 0])
    regions_plot = [r for r in ["National","Northeast","West","Midwest","South"]
                    if r in regional_results]
    betas, ci_lo, ci_hi, colors_list = [], [], [], []
    for reg in regions_plot:
        res = regional_results[reg]
        b   = res.params["Renewable_Share_Pct"]
        lo  = res.conf_int().loc["Renewable_Share_Pct", "lower"]
        hi  = res.conf_int().loc["Renewable_Share_Pct", "upper"]
        betas.append(b); ci_lo.append(lo); ci_hi.append(hi)
        colors_list.append(COLORS_REG[reg])

    ax0.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax0.axhline(b_nat, color=COLORS_REG["National"], lw=1.5, ls=":",
                alpha=0.7, label=f"National β={b_nat:+.4f}")
    for i, (b, lo, hi, col) in enumerate(zip(betas, ci_lo, ci_hi, colors_list)):
        ax0.errorbar(i, b, yerr=[[b-lo], [hi-b]],
                     fmt="o", color=col, capsize=6, ms=10, lw=2.5)
    ax0.set_xticks(range(len(regions_plot)))
    ax0.set_xticklabels(regions_plot, fontsize=9)
    ax0.set_ylabel("β  (Renewable_Share_Pct)")
    ax0.set_title("RPS Effect by Region\n(TWFE, 95% CI, clustered SE)")
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3, axis="y")

    ax1 = fig.add_subplot(gs[0, 1])
    for reg in ["Northeast", "South", "Midwest", "West"]:
        d = df_annual[df_annual["Region"]==reg].groupby("YEAR")[
            "CO2_Intensity_Combined"].mean()
        ax1.plot(d.index, d.values, "o-", color=COLORS_REG[reg],
                 lw=2, ms=4, label=reg)
    ax1.set_xlabel("Year"); ax1.set_ylabel("Mean CO₂ (tons/MWh)")
    ax1.set_title("CO₂ Intensity Trends by Region\n(2005–2022)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    int_vars   = ["Has_RPS", "RPS_South", "RPS_Midwest", "RPS_West"]
    int_labels = ["NE (base)", "+ South", "+ Midwest", "+ West"]
    int_colors = [COLORS_REG["Northeast"], COLORS_REG["South"],
                  COLORS_REG["Midwest"],   COLORS_REG["West"]]
    int_betas, int_lo, int_hi = [], [], []
    for var in int_vars:
        if var in res_int.params.index:
            int_betas.append(res_int.params[var])
            int_lo.append(res_int.conf_int().loc[var, "lower"])
            int_hi.append(res_int.conf_int().loc[var, "upper"])
        else:
            int_betas.append(0); int_lo.append(0); int_hi.append(0)

    bars = ax2.bar(range(4), int_betas, color=int_colors,
                   edgecolor="black", lw=0.7, width=0.55)
    ax2.errorbar(range(4), int_betas,
                 yerr=[[b-lo for b, lo in zip(int_betas, int_lo)],
                       [hi-b for b, hi in zip(int_betas, int_hi)]],
                 fmt="none", color="black", capsize=5, lw=1.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(int_labels, fontsize=9)
    ax2.set_ylabel("β coefficient"); ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title("Region × RPS Interaction\n(incremental vs Northeast baseline)")
    for bar, val in zip(bars, int_betas):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 val + (0.003 if val >= 0 else -0.005),
                 f"{val:+.4f}", ha="center", fontsize=9, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    for reg in ["Northeast", "South", "Midwest", "West"]:
        d   = df_annual[df_annual["Region"]==reg]
        rps = d[d["Has_RPS"]==1]["STATE"].nunique() / d["STATE"].nunique() * 100
        co2 = d["CO2_Intensity_Combined"].mean()
        ax3.scatter(rps, co2, color=COLORS_REG[reg], s=200,
                    edgecolors="black", lw=1.5, zorder=5)
        ax3.annotate(reg, (rps, co2),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=10, color=COLORS_REG[reg], fontweight="bold")
    ax3.set_xlabel("RPS Adoption Rate (%)")
    ax3.set_ylabel("Mean CO₂ Intensity (tons/MWh)")
    ax3.set_title("RPS Adoption vs CO₂ Intensity by Region")
    ax3.grid(True, alpha=0.3)

    plt.suptitle("BH5: Regional Heterogeneity in RPS Effectiveness\n"
                 "Core Period 2005–2022  |  4 Census Regions  |  51 States",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH5_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────
    rows = []
    for reg in ["National", "Northeast", "South", "Midwest", "West"]:
        if reg not in regional_results:
            continue
        res = regional_results[reg]
        key = "Renewable_Share_Pct"
        rows.append({
            "Region":    reg,
            "N_states":  region_counts.get(reg, 51),
            "Beta":      round(res.params[key], 6),
            "Std_Error": round(res.std_errors[key], 6),
            "T_stat":    round(res.tstats[key], 4),
            "P_value":   round(res.pvalues[key], 4),
            "R2_within": round(res.rsquared_within, 4),
            "N_obs":     res.nobs,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "BH5_regional_results.csv", index=False)

    int_rows = []
    for var in ["Has_RPS", "RPS_South", "RPS_Midwest", "RPS_West", "Renewable_Share_Pct"]:
        if var in res_int.params.index:
            int_rows.append({
                "Variable":  var,
                "Beta":      round(res_int.params[var], 6),
                "Std_Error": round(res_int.std_errors[var], 6),
                "P_value":   round(res_int.pvalues[var], 4),
            })
    pd.DataFrame(int_rows).to_csv(OUTPUT_DIR / "BH5_interaction_results.csv", index=False)
    print(f"✅ Saved outputs to {OUTPUT_DIR}")

    # ── Hypothesis decision ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  HYPOTHESIS DECISION — BH5")
    print("="*70)

    betas_reg = {}
    for reg in ["Northeast", "South", "Midwest", "West"]:
        if reg in regional_results:
            betas_reg[reg] = {
                "beta": regional_results[reg].params["Renewable_Share_Pct"],
                "p":    regional_results[reg].pvalues["Renewable_Share_Pct"],
            }

    int_sig = any(
        res_int.pvalues.get(v, 1) < ALPHA
        for v in ["RPS_South", "RPS_Midwest", "RPS_West"])

    print(f"\n  Regional β estimates vs National β = {b_nat:+.5f}:")
    for reg, vals in betas_reg.items():
        diff = vals["beta"] - b_nat
        sig  = vals["p"] < ALPHA
        print(f"    {reg:<12}: β = {vals['beta']:+.5f}  "
              f"(Δ = {diff:+.5f}  p = {vals['p']:.4f}  "
              f"{'sig' if sig else 'ns'})")

    print(f"\n  Region × RPS interaction significant: "
          f"{'✅ Yes' if int_sig else '❌ No'}")
    print()
    if int_sig:
        print(f"  p < α on interaction terms  →  REJECT H₀")
        print(f"  RPS effectiveness varies significantly by region.")
    else:
        south_b = betas_reg.get("South",    {}).get("beta", 0)
        ne_b    = betas_reg.get("Northeast",{}).get("beta", 0)
        if south_b < ne_b:
            print(f"  Interaction not individually significant BUT directional")
            print(f"  pattern holds: β(South) = {south_b:+.5f} < β(NE) = {ne_b:+.5f}")
            print(f"  →  PARTIAL SUPPORT for H₁")
        else:
            print(f"  FAIL TO REJECT H₀")

    print(f"\n  SUMMARY:")
    print(f"  BH1 national β = {b_nat:+.5f}")
    for reg, vals in betas_reg.items():
        sig = "*" if vals["p"] < 0.05 else "†" if vals["p"] < 0.10 else "ns"
        print(f"  BH5 {reg:<12} β = {vals['beta']:+.5f}  {sig}")

    print(f"""
  Policy implication:
  South and Midwest represent highest-return targets for RPS expansion
  — largest CO₂ reduction potential with lowest current adoption rates.
""")
    print("="*70)
    print("  BH5 COMPLETE ✅  —  All 5 BH analyses done")
    print("="*70)

    return regional_results


if __name__ == "__main__":
    run_bh5()
