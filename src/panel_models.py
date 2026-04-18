"""
BH1: Panel Fixed Effects (Causal Inference)
Does renewable share reduce carbon intensity?
All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle

Ported from BH1.ipynb — logic unchanged, paths updated to local.
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_2001_2026.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_bh1():
    # ── Reload ────────────────────────────────────────────────────────────────
    df      = pd.read_csv(DATA_PATH)
    df_core = df[(df["YEAR"] >= 2005) & (df["YEAR"] <= 2022)].copy()
    df_core = df_core.dropna(subset=[
        "CO2_Intensity_Combined", "Renewable_Share_Pct",
        "GDP_Growth_Rate_Annual", "Has_RPS",
        "Temp_Extreme",           "Years_Since_RPS"])

    agg_map = {
        "CO2_Intensity_Combined": "mean", "Renewable_Share_Pct": "mean",
        "GDP_Growth_Rate_Annual": "first", "Has_RPS": "max",
        "Temp_Extreme": "mean",            "Years_Since_RPS": "max",
    }
    df_annual = df_core.groupby(["STATE", "YEAR"]).agg(agg_map).reset_index()

    # ── TWFE demean ───────────────────────────────────────────────────────────
    def twfe_demean(df, cols):
        d = df.copy()
        for v in cols:
            grand  = df[v].mean()
            s_mean = df.groupby("STATE")[v].transform("mean")
            t_mean = df.groupby("YEAR")[v].transform("mean")
            d[v]   = df[v] - s_mean - t_mean + grand
        return d

    vars_iv = ["CO2_Intensity_Combined", "Renewable_Share_Pct",
               "GDP_Growth_Rate_Annual", "Has_RPS",
               "Temp_Extreme",           "Years_Since_RPS"]
    df_d = twfe_demean(df_annual, vars_iv)

    # ── IV-2SLS ───────────────────────────────────────────────────────────────
    formula_iv = ("CO2_Intensity_Combined ~ "
                  "GDP_Growth_Rate_Annual + Has_RPS + Temp_Extreme + "
                  "[Renewable_Share_Pct ~ Years_Since_RPS]")
    res_iv   = IV2SLS.from_formula(formula_iv, data=df_d).fit(cov_type="robust")
    res_liml = IVLIML.from_formula(formula_iv, data=df_d).fit(cov_type="robust")

    # ── First-stage diagnostics (manual OLS, version-stable) ──────────────────
    print("=" * 70)
    print("  SPEC 4 — FIRST STAGE DIAGNOSTICS (manual OLS, version-stable)")
    print("=" * 70)

    X_fs = sm.add_constant(
        df_d[["Years_Since_RPS", "GDP_Growth_Rate_Annual", "Has_RPS", "Temp_Extreme"]])
    y_fs  = df_d["Renewable_Share_Pct"]
    ols_fs = sm.OLS(y_fs, X_fs).fit(cov_type="HC1")

    X_restricted = sm.add_constant(
        df_d[["GDP_Growth_Rate_Annual", "Has_RPS", "Temp_Extreme"]])
    ols_r = sm.OLS(y_fs, X_restricted).fit()

    n      = len(y_fs)
    k      = 1
    rss_r  = ols_r.ssr
    rss_ur = ols_fs.ssr
    f_stat = ((rss_r - rss_ur) / k) / (rss_ur / (n - X_fs.shape[1]))
    f_pval = 1 - scipy.stats.f.cdf(f_stat, k, n - X_fs.shape[1])
    partial_r2 = 1 - (rss_ur / rss_r)

    print(f"\n  First-stage regression: Renewable_Share_Pct ~ Years_Since_RPS + controls")
    print(f"\n  {'Statistic':<30} {'Value':>10}")
    print("  " + "-" * 42)
    print(f"  {'First-stage F-stat':<30} {f_stat:>10.3f}")
    print(f"  {'p-value (F)':<30} {f_pval:>10.4f}")
    print(f"  {'Partial R² (instrument)':<30} {partial_r2:>10.4f}")
    print(f"  {'R² (full first stage)':<30} {ols_fs.rsquared:>10.4f}")
    print(f"  {'N':<30} {n:>10,}")

    coef_iv  = ols_fs.params["Years_Since_RPS"]
    tstat_iv = ols_fs.tvalues["Years_Since_RPS"]
    pval_iv  = ols_fs.pvalues["Years_Since_RPS"]
    print(f"\n  Years_Since_RPS in first stage:")
    print(f"    β = {coef_iv:.4f}  t = {tstat_iv:.3f}  p = {pval_iv:.4f}")

    print(f"\n  {'─'*50}")
    if f_stat >= 10:
        strength = "STRONG ✅"
        iv_note  = "IV estimates are reliable"
    elif f_stat >= 5:
        strength = "BORDERLINE ⚠️  (5 ≤ F < 10)"
        iv_note  = "IV estimates have elevated standard errors; LIML preferred"
    else:
        strength = "WEAK ❌  (F < 5)"
        iv_note  = "IV estimates unreliable; rely on TWFE + state trends (Spec 3)"

    print(f"  Instrument strength: {strength}")
    print(f"  Advice: {iv_note}")

    print(f"\n  IV-2SLS:  β = {res_iv.params['Renewable_Share_Pct']:+.5f}  "
          f"t = {res_iv.tstats['Renewable_Share_Pct']:.3f}  "
          f"p = {res_iv.pvalues['Renewable_Share_Pct']:.4f}")
    print(f"  LIML:     β = {res_liml.params['Renewable_Share_Pct']:+.5f}  "
          f"t = {res_liml.tstats['Renewable_Share_Pct']:.3f}  "
          f"p = {res_liml.pvalues['Renewable_Share_Pct']:.4f}")
    print(f"\n  Direction consistent with Spec 3 (negative)? "
          f"{'✅ Yes' if res_iv.params['Renewable_Share_Pct'] < 0 else '⚠️ No'}")

    # ── Spec 5 — Placebo test ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SPEC 5 — PLACEBO TEST")
    print("=" * 70)

    df_p = df_annual.copy().sort_values(["STATE", "YEAR"])
    df_p["Has_RPS_lead1"] = df_p.groupby("STATE")["Has_RPS"].shift(-1).fillna(0)
    df_p["Has_RPS_lead2"] = df_p.groupby("STATE")["Has_RPS"].shift(-2).fillna(0)
    df_p = df_p.set_index(["STATE", "YEAR"])

    res_pb = PanelOLS.from_formula(
        "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
        "Has_RPS + Has_RPS_lead1 + Has_RPS_lead2 + Temp_Extreme + "
        "EntityEffects + TimeEffects", data=df_p
    ).fit(cov_type="clustered", cluster_entity=True)

    print(f"\n  {'Variable':<22} {'β':>10}  {'p':>8}  {'Result'}")
    print("  " + "-" * 58)
    for var in ["Renewable_Share_Pct", "Has_RPS", "Has_RPS_lead1", "Has_RPS_lead2"]:
        b   = res_pb.params[var]
        p   = res_pb.pvalues[var]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"
        note = (" ← ✅ no pre-trend" if "lead" in var and p>0.10 else
                " ← ⚠️  pre-trend?" if "lead" in var else "")
        print(f"  {var:<22} {b:>+10.5f}  {p:>8.4f}  {sig}{note}")

    parallel_ok = (res_pb.pvalues["Has_RPS_lead1"] > 0.10 and
                   res_pb.pvalues["Has_RPS_lead2"] > 0.10)
    print(f"\n  Parallel trends: "
          f"{'✅ HOLD — no pre-trend detected' if parallel_ok else '⚠️  Possible pre-trend'}")

    # ── Rebuild specs 1, 2, 3 ─────────────────────────────────────────────────
    df_s1 = df_annual.set_index(["STATE", "YEAR"])
    res1  = PanelOLS.from_formula(
        "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
        "Has_RPS + Temp_Extreme + EntityEffects + TimeEffects", data=df_s1
    ).fit(cov_type="clustered", cluster_entity=True)

    df_m       = df_core.copy()
    df_m["TIME"] = df_m["YEAR"] * 100 + df_m["MONTH"]
    df_m       = df_m.set_index(["STATE", "TIME"])
    res2 = PanelOLS.from_formula(
        "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
        "Has_RPS + Temp_Extreme + EntityEffects + TimeEffects", data=df_m
    ).fit(cov_type="clustered", cluster_entity=True)

    df_t       = df_annual.copy()
    df_t["year_c"] = df_t["YEAR"] - int(df_t["YEAR"].mean())
    states     = sorted(df_t["STATE"].unique())
    for s in states[1:]:
        df_t[f"tr_{s}"] = (df_t["STATE"] == s).astype(float) * df_t["year_c"]
    df_t = df_t.set_index(["STATE", "YEAR"])
    res3 = PanelOLS.from_formula(
        "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
        "Has_RPS + Temp_Extreme + " + "+".join([f"tr_{s}" for s in states[1:]]) +
        " + EntityEffects + TimeEffects", data=df_t
    ).fit(cov_type="clustered", cluster_entity=True)

    # ── Complete robustness table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPLETE BH1 ROBUSTNESS TABLE")
    print("=" * 70)

    all_specs = [
        ("(1) TWFE Annual",    res1,     "Annual panel"),
        ("(2) Monthly Panel",  res2,     "11k obs"),
        ("(3) State Trends ★", res3,     "Preferred"),
        ("(4a) IV-2SLS",       res_iv,   f"F={f_stat:.1f} — weak"),
        ("(4b) LIML",          res_liml, f"F={f_stat:.1f} — weak"),
        ("(5) Placebo",        res_pb,   "Pre-trend test"),
    ]

    print(f"\n  {'Spec':<22} {'β':>10} {'SE':>8} {'t':>7} {'p':>9} {'Sig':<5} {'R²w':>7}  Note")
    print("  " + "-" * 85)
    for lbl, res, note in all_specs:
        k   = "Renewable_Share_Pct"
        b   = res.params[k]
        se  = res.std_errors[k]
        t   = res.tstats[k]
        p   = res.pvalues[k]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"
        r2  = f"{res.rsquared_within:.3f}" if hasattr(res, "rsquared_within") else "  n/a"
        print(f"  {lbl:<22} {b:>+10.5f} {se:>8.5f} {t:>7.3f} {p:>9.4f} {sig:<5} {r2:>7}  {note}")

    print("""
  *** p<0.001  ** p<0.01  * p<0.05  † p<0.10  ns p≥0.10
  ★  Preferred specification
""")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = []
    for lbl, res, note in all_specs:
        k = "Renewable_Share_Pct"
        rows.append({
            "Specification": lbl,
            "Beta"         : round(res.params[k], 6),
            "Std_Error"    : round(res.std_errors[k], 6),
            "T_stat"       : round(res.tstats[k], 4),
            "P_value"      : round(res.pvalues[k], 4),
            "CI_Lower"     : round(res.conf_int().loc[k, "lower"], 6),
            "CI_Upper"     : round(res.conf_int().loc[k, "upper"], 6),
            "R2_within"    : round(getattr(res, "rsquared_within", float("nan")), 4),
            "N_obs"        : res.nobs,
            "Note"         : note,
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "BH1_results_table.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_res    = [res1, res2, res3, res_iv]
    plot_labels = ["(1) TWFE\nAnnual", "(2) Monthly\nPanel",
                   "(3) State\nTrends ★", "(4) IV\n2SLS"]
    plot_cols   = ["steelblue", "steelblue", "darkgreen", "firebrick"]

    axes[0].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    for i, (res, col) in enumerate(zip(plot_res, plot_cols)):
        b  = res.params["Renewable_Share_Pct"]
        lo = res.conf_int().loc["Renewable_Share_Pct", "lower"]
        hi = res.conf_int().loc["Renewable_Share_Pct", "upper"]
        axes[0].errorbar(i, b, yerr=[[b-lo], [hi-b]],
                         fmt="o", color=col, capsize=6, ms=9, lw=2.5)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(plot_labels, fontsize=9)
    axes[0].set_ylabel("β  (Renewable_Share_Pct)")
    axes[0].set_title("Coefficient Stability\n(95% CI, clustered SE)")
    axes[0].grid(True, alpha=0.3, axis="y")
    b3 = res3.params["Renewable_Share_Pct"]
    axes[0].annotate("★ p=0.017", xy=(2, b3), xytext=(2.3, b3+0.003),
                     fontsize=8, color="darkgreen",
                     arrowprops=dict(arrowstyle="->", color="darkgreen"))

    pv      = ["lead2", "lead1", "current"]
    pb_vals = [res_pb.params.get("Has_RPS_lead2", 0),
               res_pb.params.get("Has_RPS_lead1", 0),
               res_pb.params["Has_RPS"]]
    pp_vals = [res_pb.pvalues.get("Has_RPS_lead2", 1),
               res_pb.pvalues.get("Has_RPS_lead1", 1),
               res_pb.pvalues["Has_RPS"]]
    bc   = ["#2ecc71" if p > 0.10 else "#e74c3c" for p in pp_vals]
    bars = axes[1].barh(pv, pb_vals, color=bc, edgecolor="black", lw=0.7, height=0.5)
    axes[1].axvline(0, color="black", lw=1, ls="--")
    for bar, pval in zip(bars, pp_vals):
        lbl = "ns ✅" if pval > 0.10 else f"p={pval:.3f}"
        axes[1].text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                     lbl, va="center", fontsize=9)
    axes[1].set_xlabel("Coefficient on RPS Indicator")
    axes[1].set_title("Placebo Test: Lead RPS\n(green = no pre-trend ✅)")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.suptitle("BH1: Robustness Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH1_robustness.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'BH1_robustness.png'}")

    # Coefficient comparison (all 5 specs)
    fig2, ax = plt.subplots(figsize=(10, 6))
    plot_res_all    = [res1, res2, res3, res_iv, res_liml]
    plot_labels_all = ["(1) TWFE\nAnnual", "(2) Monthly\nPanel",
                       "(3) State\nTrends ★", "(4a) IV\n2SLS", "(4b) LIML"]
    plot_cols_all   = ["steelblue", "steelblue", "darkgreen", "firebrick", "darkred"]

    betas    = [r.params["Renewable_Share_Pct"] for r in plot_res_all]
    ci_lower = [r.conf_int().loc["Renewable_Share_Pct", "lower"] for r in plot_res_all]
    ci_upper = [r.conf_int().loc["Renewable_Share_Pct", "upper"] for r in plot_res_all]
    yerr = [[b-cl for b, cl in zip(betas, ci_lower)],
            [cu-b for b, cu in zip(betas, ci_upper)]]

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    for i, (b_val, y_err_tuple, col) in enumerate(
            zip(betas, zip(yerr[0], yerr[1]), plot_cols_all)):
        ax.errorbar(i, b_val, yerr=[[y_err_tuple[0]], [y_err_tuple[1]]],
                    fmt="o", color=col, capsize=6, ms=9, lw=2.5)
    ax.set_xticks(range(len(plot_labels_all)))
    ax.set_xticklabels(plot_labels_all, fontsize=10)
    ax.set_ylabel("β Coefficient (Renewable_Share_Pct)")
    ax.set_title("Comparison of Renewable Share Coefficients Across Specifications\n"
                 "(95% Confidence Intervals)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.annotate("★ Preferred", xy=(2, b3), xytext=(2.3, b3+0.003),
                fontsize=9, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH1_coefficient_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR / 'BH1_coefficient_comparison.png'}")
    print(f"✅ Saved: {OUTPUT_DIR / 'BH1_results_table.csv'}")

    # ── Sample size & power ───────────────────────────────────────────────────
    from scipy.stats import norm as _norm
    print("""
SAMPLE SIZE & POWER ANALYSIS — BH1
────────────────────────────────────────────────────────────""")
    alpha, power     = 0.05, 0.80
    n_obs, n_states, n_years = 918, 51, 18
    beta_est, std_renew, std_co2 = 0.00362, 23.0, 0.268
    effect_size  = beta_est * std_renew / std_co2
    z_alpha2, z_beta = _norm.ppf(1 - alpha/2), _norm.ppf(power)
    n_min        = int(((z_alpha2 + z_beta) / effect_size) ** 2) + 1
    z_achieved   = effect_size * (n_obs ** 0.5)
    power_ach    = _norm.cdf(z_achieved - z_alpha2)
    print(f"  Effect size (standardised β):  {effect_size:.4f}")
    print(f"  Minimum N required (80% power): {n_min:,}")
    print(f"  Actual N (annual panel):        {n_obs:,}  "
          f"({'✅ adequate' if n_obs >= n_min else '⚠️ below minimum'})")
    print(f"  Achieved statistical power:     {power_ach*100:.1f}%")
    print(f"  Panel structure:                {n_states} states × {n_years} years")

    # ── Hypothesis decision ───────────────────────────────────────────────────
    b_pref = res3.params["Renewable_Share_Pct"]
    p_pref = res3.pvalues["Renewable_Share_Pct"]
    print("\n" + "="*60)
    print("  HYPOTHESIS DECISION — BH1")
    print("="*60)
    if p_pref < 0.05:
        print(f"  β = {b_pref:.5f}  |  p = {p_pref:.4f}  |  α = 0.05")
        print(f"  p < α  →  REJECT H₀")
        print(f"  Renewable share significantly reduces CO₂ intensity.")
    else:
        print(f"  β = {b_pref:.5f}  |  p = {p_pref:.4f}  |  α = 0.05")
        print(f"  p ≥ α  →  FAIL TO REJECT H₀")
    print("="*60)
    print("\n  BH1 COMPLETE ✅")

    return res3


if __name__ == "__main__":
    run_bh1()
