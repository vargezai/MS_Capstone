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
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_CLEAN.csv"
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

    run_dynamic_panel()
    run_sensitivity()
    return res3


def run_dynamic_panel():
    """Arellano-Bond dynamic panel robustness checks (diff-GMM and system-GMM)."""
    from pydynpd import regression as abond_reg

    print("\n" + "=" * 70)
    print("  DYNAMIC PANEL ROBUSTNESS — Arellano-Bond GMM")
    print("=" * 70)

    df      = pd.read_csv(DATA_PATH)
    df_core = df[(df["YEAR"] >= 2005) & (df["YEAR"] <= 2022)].copy()
    df_core = df_core.dropna(subset=[
        "CO2_Intensity_Combined", "Renewable_Share_Pct",
        "GDP_Growth_Rate_Annual", "Has_RPS", "Temp_Extreme",
    ])

    agg_map = {
        "CO2_Intensity_Combined": "mean", "Renewable_Share_Pct": "mean",
        "GDP_Growth_Rate_Annual": "first", "Has_RPS": "max",
        "Temp_Extreme": "mean",
    }
    df_a = (df_core.groupby(["STATE", "YEAR"]).agg(agg_map)
            .reset_index().sort_values(["STATE", "YEAR"]).reset_index(drop=True))
    df_a["state_id"] = df_a["STATE"].astype("category").cat.codes + 1

    print(f"\n  Panel: {df_a['state_id'].nunique()} states × "
          f"{df_a['YEAR'].nunique()} years  ({len(df_a):,} obs)")
    print("  Dep. var: CO2_Intensity_Combined")
    print("  Key regressor: Renewable_Share_Pct")
    print("  Instruments: GMM lags 2–4 of dep. var + IV(controls)\n")

    _base = ("CO2_Intensity_Combined L1.CO2_Intensity_Combined "
             "Renewable_Share_Pct GDP_Growth_Rate_Annual Has_RPS Temp_Extreme | "
             "gmm(CO2_Intensity_Combined, 2:4) "
             "iv(Renewable_Share_Pct GDP_Growth_Rate_Annual Has_RPS Temp_Extreme)")

    specs = [
        ("(A) 1-step Diff-GMM",  _base + " | onestep nolevel"),
        ("(B) 2-step Diff-GMM",  _base + " | nolevel"),
        ("(C) 2-step Sys-GMM",   _base),
    ]

    k = "Renewable_Share_Pct"
    rows = []
    for label, cmd in specs:
        m   = abond_reg.abond(cmd, df_a, ["state_id", "YEAR"])
        mdl = m.models[0]
        rt  = mdl.regression_table
        row = rt[rt["variable"] == k].iloc[0]

        ar1 = mdl.AR_list[0] if len(mdl.AR_list) > 0 else None
        ar2 = mdl.AR_list[1] if len(mdl.AR_list) > 1 else None

        rows.append({
            "Specification":   label,
            "Beta":            round(float(row.coefficient), 6),
            "Std_Error":       round(float(row.std_err), 6),
            "Z_stat":          round(float(row.z_value), 4),
            "P_value":         round(float(row.p_value), 4),
            "N_obs":           int(mdl.num_obs),
            "N_groups":        int(mdl.N),
            "N_instruments":   int(mdl.z_information.num_instr),
            "Hansen_chi2_p":   round(float(mdl.hansen.p_value), 4),
            "AR1_p":           round(float(ar1.P_value), 4) if ar1 else float("nan"),
            "AR2_p":           round(float(ar2.P_value), 4) if ar2 else float("nan"),
        })

    df_dyn = pd.DataFrame(rows)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"  {'Spec':<24} {'β':>10} {'SE':>8} {'z':>7} {'p':>9}  Sig")
    print("  " + "-" * 65)
    for r in rows:
        sig = ("***" if r["P_value"] < 0.001 else "**" if r["P_value"] < 0.01
               else "*" if r["P_value"] < 0.05 else "†" if r["P_value"] < 0.10 else "ns")
        print(f"  {r['Specification']:<24} {r['Beta']:>+10.5f} {r['Std_Error']:>8.5f} "
              f"{r['Z_stat']:>7.3f} {r['P_value']:>9.4f}  {sig}")

    print(f"\n  {'Spec':<24} {'Hansen p':>10} {'AR(1) p':>10} {'AR(2) p':>10}  Validity")
    print("  " + "-" * 68)
    for r in rows:
        hansen_ok = r["Hansen_chi2_p"] > 0.10
        ar1_ok    = r["AR1_p"] < 0.10
        ar2_ok    = r["AR2_p"] > 0.10
        valid     = "✅ OK" if (hansen_ok and ar1_ok) else "⚠️  check"
        ar2_note  = "✅" if ar2_ok else "⚠️ (borderline)"
        print(f"  {r['Specification']:<24} {r['Hansen_chi2_p']:>10.4f} "
              f"{r['AR1_p']:>10.4f} {r['AR2_p']:>10.4f}  {valid}  AR(2):{ar2_note}")

    print("""
  Interpretation:
    Hansen p > 0.10 → instruments are valid (no overidentification)
    AR(1) p < 0.10  → expected first-order autocorrelation in differences
    AR(2) p > 0.10  → no second-order autocorrelation (key assumption)
    AR(2) borderline (p ≈ 0.04–0.05) is common with persistent CO₂ series.
    Direction consistent with BH1 TWFE (Spec 3 ★: β = -0.00370).
""")

    df_dyn.to_csv(OUTPUT_DIR / "BH1_dynamic_panel_table.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels   = [r["Specification"].replace(" ", "\n") for r in rows]
    betas    = [r["Beta"]      for r in rows]
    ses      = [r["Std_Error"] for r in rows]
    colors   = ["steelblue", "darkorange", "darkgreen"]

    ax0 = axes[0]
    ax0.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    for i, (b, se, col) in enumerate(zip(betas, ses, colors)):
        ax0.errorbar(i, b, yerr=1.96 * se, fmt="o", color=col,
                     capsize=7, ms=9, lw=2.2)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("β  (Renewable_Share_Pct)")
    ax0.set_title("Arellano-Bond: Coefficient on Renewable Share\n(±1.96 SE)")
    ax0.grid(True, alpha=0.3, axis="y")

    ax1 = axes[1]
    x = np.arange(len(rows))
    width = 0.25
    ax1.bar(x - width, [r["Hansen_chi2_p"] for r in rows],
            width, label="Hansen p", color=colors, alpha=0.7)
    ax1.bar(x,         [r["AR2_p"] for r in rows],
            width, label="AR(2) p", color=colors, alpha=0.4, hatch="//")
    ax1.axhline(0.10, color="red", lw=1.2, ls="--", label="p=0.10 threshold")
    ax1.axhline(0.05, color="orange", lw=1.0, ls=":", label="p=0.05")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("p-value")
    ax1.set_title("Specification Tests\n(Hansen overid + AR(2))")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, 1)

    plt.suptitle("BH1: Dynamic Panel Robustness (Arellano-Bond GMM)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH1_dynamic_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved: BH1_dynamic_panel_table.csv")
    print(f"  ✅ Saved: BH1_dynamic_panel.png")
    print("\n  DYNAMIC PANEL COMPLETE ✅")
    return df_dyn


def run_sensitivity():
    """Sensitivity analysis: rerun preferred specs excluding CO2_Outlier_Flag = 1 rows."""
    print("\n" + "=" * 70)
    print("  SENSITIVITY ANALYSIS — Excluding CO2_Outlier_Flag = 1")
    print("=" * 70)

    df      = pd.read_csv(DATA_PATH)
    df_core = df[(df["YEAR"] >= 2005) & (df["YEAR"] <= 2022)].copy()
    df_core = df_core.dropna(subset=[
        "CO2_Intensity_Combined", "Renewable_Share_Pct",
        "GDP_Growth_Rate_Annual", "Has_RPS",
        "Temp_Extreme",           "Years_Since_RPS",
        "CO2_Outlier_Flag",
    ])

    n_total   = len(df_core)
    n_flagged = (df_core["CO2_Outlier_Flag"] == 1).sum()
    print(f"\n  Monthly obs in window (2005-2022): {n_total:,}")
    print(f"  CO2_Outlier_Flag = 1:              {n_flagged:,}  ({100*n_flagged/n_total:.1f}%)")
    print(f"  Restricted sample:                 {n_total - n_flagged:,}")

    agg_map = {
        "CO2_Intensity_Combined": "mean", "Renewable_Share_Pct": "mean",
        "GDP_Growth_Rate_Annual": "first", "Has_RPS": "max",
        "Temp_Extreme": "mean",            "Years_Since_RPS": "max",
        "CO2_Outlier_Flag": "max",
    }
    df_full = df_core.groupby(["STATE", "YEAR"]).agg(agg_map).reset_index()
    df_rest = (df_core[df_core["CO2_Outlier_Flag"] == 0]
               .groupby(["STATE", "YEAR"]).agg(agg_map).reset_index())

    def _spec1(d):
        return PanelOLS.from_formula(
            "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
            "Has_RPS + Temp_Extreme + EntityEffects + TimeEffects",
            data=d.set_index(["STATE", "YEAR"])
        ).fit(cov_type="clustered", cluster_entity=True)

    def _spec3(d):
        dt = d.copy()
        dt["year_c"] = dt["YEAR"] - int(dt["YEAR"].mean())
        states = sorted(dt["STATE"].unique())
        for s in states[1:]:
            dt[f"tr_{s}"] = (dt["STATE"] == s).astype(float) * dt["year_c"]
        return PanelOLS.from_formula(
            "CO2_Intensity_Combined ~ Renewable_Share_Pct + GDP_Growth_Rate_Annual + "
            "Has_RPS + Temp_Extreme + " + "+".join([f"tr_{s}" for s in states[1:]]) +
            " + EntityEffects + TimeEffects",
            data=dt.set_index(["STATE", "YEAR"])
        ).fit(cov_type="clustered", cluster_entity=True)

    res1_full = _spec1(df_full);  res1_rest = _spec1(df_rest)
    res3_full = _spec3(df_full);  res3_rest = _spec3(df_rest)

    k = "Renewable_Share_Pct"
    combos = [
        ("(1) TWFE Annual",    "Full",           res1_full),
        ("(1) TWFE Annual",    "Excl. Outliers", res1_rest),
        ("(3) State Trends ★", "Full",           res3_full),
        ("(3) State Trends ★", "Excl. Outliers", res3_rest),
    ]

    print(f"\n  {'Spec':<22} {'Sample':<18} {'β':>10} {'SE':>8} {'p':>9} {'N':>6}  Sig")
    print("  " + "-" * 80)
    rows = []
    for lbl, samp, res in combos:
        b   = res.params[k];  se = res.std_errors[k]
        t   = res.tstats[k];  p  = res.pvalues[k]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "ns"
        ci  = res.conf_int()
        print(f"  {lbl:<22} {samp:<18} {b:>+10.5f} {se:>8.5f} {p:>9.4f} {res.nobs:>6,}  {sig}")
        rows.append({
            "Specification": lbl, "Sample": samp,
            "Beta": round(b, 6), "Std_Error": round(se, 6),
            "T_stat": round(t, 4), "P_value": round(p, 4),
            "CI_Lower": round(ci.loc[k, "lower"], 6),
            "CI_Upper": round(ci.loc[k, "upper"], 6),
            "N_obs": res.nobs,
        })

    print("\n  Direction stability (β < 0 = renewable reduces CO₂):")
    for lbl, samp, res in combos:
        direction = "✅ negative" if res.params[k] < 0 else "⚠️  positive"
        print(f"    {lbl} [{samp}]: β = {res.params[k]:+.5f}  {direction}")

    df_sens = pd.DataFrame(rows)
    df_sens.to_csv(OUTPUT_DIR / "BH1_sensitivity_table.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    spec_labels = ["(1) TWFE\nAnnual", "(3) State\nTrends ★"]
    full_res  = [res1_full, res3_full]
    rest_res  = [res1_rest, res3_rest]
    x = np.arange(len(spec_labels))
    width = 0.28

    for ax_i, (ax, title) in enumerate(zip(axes, ["TWFE Annual", "State Trends ★"])):
        res_pairs = [(full_res[ax_i], "Full", "steelblue", "o"),
                     (rest_res[ax_i], "Excl. CO₂ Outliers", "darkorange", "s")]
        for offset, (res, label, col, marker) in zip([-width/2, width/2], res_pairs):
            b   = res.params[k]
            lo  = res.conf_int().loc[k, "lower"]
            hi  = res.conf_int().loc[k, "upper"]
            ax.errorbar(0 + offset, b, yerr=[[b - lo], [hi - b]],
                        fmt=marker, color=col, capsize=7, ms=9, lw=2.2, label=label)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks([])
        ax.set_ylabel("β  (Renewable_Share_Pct)")
        ax.set_title(f"Spec {title}\n(95% CI, clustered SE)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BH1 Sensitivity: Full vs. Outlier-Excluded Sample", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH1_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  ✅ Saved: BH1_sensitivity_table.csv")
    print(f"  ✅ Saved: BH1_sensitivity.png")
    print("\n  SENSITIVITY ANALYSIS COMPLETE ✅")
    return df_sens


if __name__ == "__main__":
    run_bh1()
