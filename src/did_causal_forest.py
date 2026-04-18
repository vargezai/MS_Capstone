"""
BH2: Staggered Difference-in-Differences + Causal Forest (Policy Evaluation)
What is the causal impact of RPS policies on CO2 intensity?
All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle

Ported from BH2.ipynb — logic unchanged, paths updated to local.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.panel import PanelOLS
from scipy import stats as scipy_stats
from scipy.stats import norm

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    print("📦 Installing econml ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "econml", "-q"])
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_2001_2026.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_START, CORE_END = 2005, 2022


def run_bh2():
    print("\n" + "="*70)
    print("  BH2: RPS POLICY IMPACT — CAUSAL FOREST + EVENT-STUDY DiD")
    print("="*70)

    df_full = pd.read_csv(DATA_PATH)
    df_full = df_full[~df_full["STATE"].isin(["US-TOTAL", "US", "USA"])].copy()
    print(f"\n✅ States after removing aggregates: {df_full['STATE'].nunique()}")

    df_core = df_full[
        (df_full["YEAR"] >= CORE_START) & (df_full["YEAR"] <= CORE_END)].copy()
    df_core = df_core.dropna(subset=[
        "CO2_Intensity_Combined", "Has_RPS", "Years_Since_RPS",
        "RPS_Implementation_Year", "Renewable_Share_Pct",
        "GDP_Growth_Rate_Annual", "Temp_Extreme"])

    agg_map = {
        "CO2_Intensity_Combined": "mean", "Has_RPS":                "max",
        "Years_Since_RPS":        "max",  "RPS_Implementation_Year": "first",
        "Renewable_Share_Pct":    "mean", "GDP_Growth_Rate_Annual":  "first",
        "Temp_Extreme":           "mean", "Fossil_Intensity":        "mean",
    }
    df_annual = df_core.groupby(["STATE", "YEAR"]).agg(agg_map).reset_index()

    # Baseline covariates from full dataset
    df_base = (df_full[df_full["YEAR"] == CORE_START]
               [["STATE", "CO2_Intensity_Combined", "Renewable_Share_Pct",
                 "Fossil_Intensity", "Avg_Temp_F", "Real_GDP_Millions"]]
               .groupby("STATE").mean().reset_index()
               .rename(columns={
                   "CO2_Intensity_Combined": "CO2_base",
                   "Renewable_Share_Pct":    "Renew_base",
                   "Fossil_Intensity":       "Fossil_base",
                   "Avg_Temp_F":             "Temp_base",
                   "Real_GDP_Millions":      "GDP_base"}))

    df_state = (df_full[
        (df_full["YEAR"] >= CORE_START) & (df_full["YEAR"] <= CORE_END)]
        .groupby("STATE")
        .agg(CO2_mean=("CO2_Intensity_Combined", "mean"),
             Has_RPS =("Has_RPS", "max"))
        .reset_index())

    df_cf = df_state.merge(df_base, on="STATE")
    for col in ["CO2_base", "Renew_base", "Fossil_base", "Temp_base", "GDP_base"]:
        df_cf[col] = df_cf[col].fillna(df_cf[col].median())

    rps_states  = sorted(df_cf[df_cf["Has_RPS"]==1]["STATE"].unique())
    ctrl_states = sorted(df_cf[df_cf["Has_RPS"]==0]["STATE"].unique())
    print(f"\n📊 CF sample: {len(df_cf)} states | "
          f"treated={len(rps_states)} | control={len(ctrl_states)}")
    print(f"   Control: {ctrl_states}")

    # ── Causal Forest ─────────────────────────────────────────────────────────
    ATT = None; ATE = None; ATT_p = 1.0
    feat_names = ["CO2_base", "Renew_base", "Fossil_base", "Temp_base", "GDP_base"]
    importance = np.zeros(len(feat_names))
    df_cf_out  = pd.DataFrame()

    if ECONML_AVAILABLE and len(ctrl_states) >= 3:
        print("\n" + "="*70)
        print("  CAUSAL FOREST")
        print("="*70)

        Y = df_cf["CO2_mean"].values.reshape(-1, 1)
        T = df_cf["Has_RPS"].values.reshape(-1, 1)
        X = df_cf[feat_names].values

        cf = CausalForest(n_estimators=2000, min_samples_leaf=3,
                          max_depth=10, honest=True, random_state=42, n_jobs=-1)
        cf.fit(X, T, Y)

        raw = cf.predict(X, interval=True, alpha=0.05)
        if len(raw) == 3:
            tau_hat = raw[0].flatten()
            tau_lo  = raw[1].flatten()
            tau_hi  = raw[2].flatten()
        else:
            tau_hat = raw[0].flatten()
            tau_lo  = raw[1][:, 0].flatten()
            tau_hi  = raw[1][:, 1].flatten()

        df_cf_out = df_cf.copy()
        df_cf_out["CATE"]    = tau_hat
        df_cf_out["CATE_lo"] = tau_lo
        df_cf_out["CATE_hi"] = tau_hi

        treated = df_cf["Has_RPS"].values == 1
        ATE     = float(tau_hat.mean())
        ATT     = float(tau_hat[treated].mean())
        ATT_se  = float(tau_hat[treated].std() / np.sqrt(treated.sum()))
        ATT_t   = ATT / ATT_se if ATT_se > 1e-10 else 0.0
        ATT_p   = float(2*(1 - scipy_stats.norm.cdf(abs(ATT_t))))
        importance = cf.feature_importances_

        print(f"\n   ATE = {ATE:+.5f} tons/MWh")
        print(f"   ATT = {ATT:+.5f} tons/MWh  SE={ATT_se:.5f}  t={ATT_t:.3f}  p={ATT_p:.4f}")
        print(f"\n   Feature importance:")
        for f, i in sorted(zip(feat_names, importance), key=lambda x: -x[1]):
            print(f"   {f:<15} {i:.4f}  {'█'*int(i*60)}")

        df_rps_s = df_cf_out[df_cf_out["Has_RPS"]==1].sort_values("CATE")
        print(f"\n   Largest CO₂ reductions (top 5 RPS states):")
        for _, r in df_rps_s.head(5).iterrows():
            print(f"     {r['STATE']}: {r['CATE']:+.4f}  (fossil_base={r['Fossil_base']:.1f}%)")
        print(f"   Smallest CO₂ reductions (bottom 5 RPS states):")
        for _, r in df_rps_s.tail(5).iterrows():
            print(f"     {r['STATE']}: {r['CATE']:+.4f}  (fossil_base={r['Fossil_base']:.1f}%)")

    # ── Event-study DiD ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  EVENT-STUDY DiD — Dynamic Treatment Effects")
    print("="*70)

    rps_yr_map = (df_annual[df_annual["Has_RPS"]==1]
                  .groupby("STATE")["RPS_Implementation_Year"]
                  .first().to_dict())

    df_es = df_annual.copy()
    df_es["RPS_Year"]  = df_es["STATE"].map(rps_yr_map)
    df_es["EventTime"] = (df_es["YEAR"] - df_es["RPS_Year"]).where(
        df_es["Has_RPS"]==1, other=np.nan)

    df_w    = df_es.copy()
    df_w["ET"] = df_w["EventTime"].clip(-3, 10)

    event_ks = [-3, -2] + list(range(0, 11))

    for k in event_ks:
        col = f"Et_m{abs(k)}" if k < 0 else f"Et_{k}"
        df_w[col] = np.where(
            (df_w["Has_RPS"]==1) & (df_w["ET"]==k), 1.0, 0.0)

    et_cols = [f"Et_m{abs(k)}" if k<0 else f"Et_{k}" for k in event_ks]

    print(f"\n   Event-time dummies created: {et_cols}")
    zero_check = [c for c in et_cols if df_w[c].sum() == 0]
    if zero_check:
        print(f"   ⚠️  Dropping zero columns: {zero_check}")
        et_cols  = [c for c in et_cols if c not in zero_check]
        event_ks = [k for k in event_ks
                    if (f"Et_m{abs(k)}" if k<0 else f"Et_{k}") in et_cols]
    else:
        print(f"   ✅ All dummy columns have non-zero values")

    df_p    = df_w.set_index(["STATE", "YEAR"])
    formula = ("CO2_Intensity_Combined ~ "
               + " + ".join(et_cols)
               + " + GDP_Growth_Rate_Annual + Temp_Extreme"
               + " + EntityEffects + TimeEffects")

    res_es = PanelOLS.from_formula(formula, data=df_p).fit(
        cov_type="clustered", cluster_entity=True)

    es = {}
    for k in event_ks:
        col = f"Et_m{abs(k)}" if k<0 else f"Et_{k}"
        if col in res_es.params.index:
            es[k] = dict(
                beta=float(res_es.params[col]),
                se  =float(res_es.std_errors[col]),
                lo  =float(res_es.conf_int().loc[col, "lower"]),
                hi  =float(res_es.conf_int().loc[col, "upper"]),
                p   =float(res_es.pvalues[col]))
    es[-1] = dict(beta=0.0, se=0.0, lo=0.0, hi=0.0, p=1.0)

    print(f"\n   {'k':>4}  {'β':>9}  {'SE':>8}  {'p':>8}  Sig  Period")
    print("   " + "-"*55)
    for k in sorted(es.keys()):
        d   = es[k]
        sig = ("***" if d["p"]<0.001 else "**" if d["p"]<0.01
               else "*" if d["p"]<0.05 else "†" if d["p"]<0.10 else "ns")
        per = "ref" if k==-1 else "pre" if k<0 else "post"
        print(f"   {k:>4}  {d['beta']:>+9.4f}  {d['se']:>8.4f}  "
              f"{d['p']:>8.4f}  {sig:<4} [{per}]")

    pre_ks         = [k for k in sorted(es.keys()) if k < -1]
    pre_sig        = sum(1 for k in pre_ks if es[k]["p"] < 0.10)
    parallel_holds = pre_sig == 0

    print(f"\n   Pre-periods tested: {len(pre_ks)}  |  Significant (p<0.10): {pre_sig}")
    print(f"   {'✅ PARALLEL TRENDS HOLD' if parallel_holds else '⚠️  Pre-trend detected'}")

    post_ks    = [k for k in sorted(es.keys()) if k >= 0]
    post_betas = [es[k]["beta"] for k in post_ks]
    post_avg   = float(np.mean(post_betas)) if post_betas else float("nan")
    post_sig_n = sum(1 for k in post_ks if es[k]["p"] < 0.10)
    print(f"\n   Post-treatment average effect : {post_avg:+.5f} tons/MWh")
    print(f"   Post-periods p<0.10           : {post_sig_n}/{len(post_betas)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 GENERATING PLOTS...")
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    ax0   = fig.add_subplot(gs[0, 0])
    ks_s  = sorted(es.keys())
    betas = [es[k]["beta"] for k in ks_s]
    lo    = [es[k]["lo"]   for k in ks_s]
    hi    = [es[k]["hi"]   for k in ks_s]
    ax0.fill_between(ks_s, lo, hi, alpha=0.18, color="steelblue", label="95% CI")
    ax0.plot(ks_s, betas, "o-", color="steelblue", lw=2, ms=5, label="β estimate")
    ax0.axhline(0, color="black", lw=0.8, ls="--")
    ax0.axvline(-0.5, color="red", lw=1.2, ls=":", alpha=0.8, label="RPS adoption")
    ax0.axvspan(-0.5, 10.5, alpha=0.05, color="green")
    ax0.set_xlabel("Years Relative to RPS Adoption  (k = −1 is reference)")
    ax0.set_ylabel("β (CO₂ Intensity, short tons/MWh)")
    ax0.set_title("Event-Study: Dynamic Effect of RPS on CO₂\n(TWFE, clustered SE by state)")
    ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    pre_ks_plot  = [k for k in ks_s if k < 0]
    post_ks_plot = [k for k in ks_s if k >= 0]
    ax1.bar(pre_ks_plot,
            [es[k]["beta"] for k in pre_ks_plot],
            color=["#e74c3c" if es[k]["p"]<0.10 else "#bdc3c7" for k in pre_ks_plot],
            edgecolor="black", lw=0.5, width=0.8, label="Pre-adoption")
    ax1.bar(post_ks_plot,
            [es[k]["beta"] for k in post_ks_plot],
            color=["#2ecc71" if es[k]["beta"]<0 else "#e67e22" for k in post_ks_plot],
            edgecolor="black", lw=0.5, width=0.8, label="Post-adoption")
    if not np.isnan(post_avg):
        ax1.axhline(post_avg, color="blue", lw=1.5, ls="--",
                    label=f"Post avg = {post_avg:+.4f}")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Event Time (k)")
    ax1.set_ylabel("β coefficient")
    ax1.set_title("Event-Time Coefficients\n(grey/red = pre  |  green/orange = post)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

    ax2 = fig.add_subplot(gs[1, 0])
    if ATT is not None and len(df_cf_out) > 0:
        df_s = df_cf_out.sort_values("CATE")
        cols = ["#e74c3c" if h==1 else "#95a5a6" for h in df_s["Has_RPS"]]
        ax2.barh(range(len(df_s)), df_s["CATE"], color=cols,
                 edgecolor="none", height=0.8)
        ax2.errorbar(df_s["CATE"], range(len(df_s)),
                     xerr=[df_s["CATE"]-df_s["CATE_lo"],
                            df_s["CATE_hi"]-df_s["CATE"]],
                     fmt="none", color="black", alpha=0.2, lw=0.7)
        ax2.axvline(0,   color="black", lw=0.8, ls="--")
        ax2.axvline(ATT, color="blue",  lw=1.5, ls="-.", label=f"ATT={ATT:+.4f}")
        ax2.set_xlabel("CATE (short tons/MWh)")
        ax2.set_title("Causal Forest: State Treatment Effects\n"
                      "(red = RPS states, grey = control)")
        ax2.set_yticks([]); ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x")

    ax3 = fig.add_subplot(gs[1, 1])
    if ATT is not None and len(df_cf_out) > 0:
        sc = ax3.scatter(df_cf_out["Fossil_base"], df_cf_out["CATE"],
                         c=df_cf_out["Has_RPS"], cmap="RdYlGn_r",
                         alpha=0.8, s=70, edgecolors="black", lw=0.5)
        m, b_l = np.polyfit(df_cf_out["Fossil_base"], df_cf_out["CATE"], 1)
        xr     = np.linspace(df_cf_out["Fossil_base"].min(),
                              df_cf_out["Fossil_base"].max(), 100)
        ax3.plot(xr, m*xr+b_l, "r--", lw=1.5, label=f"slope = {m:+.4f}")
        ax3.axhline(0, color="black", lw=0.8, ls="--")
        ax3.set_xlabel("Baseline Fossil Intensity (%, 2005)")
        ax3.set_ylabel("CATE (short tons/MWh)")
        ax3.set_title("Heterogeneity: RPS Effect vs Fossil Dependence")
        plt.colorbar(sc, ax=ax3, label="Has RPS")
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.suptitle("BH2: RPS Policy Impact — Causal Forest + Event-Study DiD",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH2_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Trend plot
    df_grouped = df_annual.groupby(["YEAR", "Has_RPS"])["CO2_Intensity_Combined"].mean().reset_index()
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_grouped, x="YEAR", y="CO2_Intensity_Combined",
                 hue="Has_RPS", marker="o", palette={0: "red", 1: "green"})
    plt.title("Average CO2 Intensity Trends: RPS vs. Non-RPS States (2005-2022)")
    plt.xlabel("Year"); plt.ylabel("Average CO2 Intensity (tons/MWh)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Has RPS", labels=["No RPS (Control)", "Has RPS (Treated)"],
               loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH2_trends.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────
    pd.DataFrame([
        {"k": k, "Beta": round(v["beta"],5), "SE": round(v["se"],5),
         "P": round(v["p"],4), "CI_lo": round(v["lo"],5), "CI_hi": round(v["hi"],5)}
        for k, v in sorted(es.items())
    ]).to_csv(OUTPUT_DIR / "BH2_eventstudy_table.csv", index=False)

    if ATT is not None and len(df_cf_out) > 0:
        df_cf_out[["STATE", "Has_RPS", "CATE", "CATE_lo", "CATE_hi",
                   "Fossil_base", "Renew_base", "CO2_base"]
                 ].to_csv(OUTPUT_DIR / "BH2_cate_by_state.csv", index=False)

    # ── Sample size & power ───────────────────────────────────────────────────
    print("""
SAMPLE SIZE & POWER ANALYSIS — BH2
────────────────────────────────────────────────────────────""")
    n_treated, n_control = 27, 24
    mean_diff, pooled_sd = 0.163, 0.268
    cohens_d = mean_diff / pooled_sd
    z_a2, z_b = norm.ppf(0.975), norm.ppf(0.80)
    n_per_group_min  = int(2 * ((z_a2 + z_b) / cohens_d) ** 2) + 1
    power_achieved   = norm.cdf(
        cohens_d * (min(n_treated, n_control) ** 0.5) / (2**0.5) - z_a2)
    print(f"  Cohen's d (effect size):        {cohens_d:.4f}")
    print(f"  Min N per group (80% power):    {n_per_group_min}")
    print(f"  Actual: {n_treated} treated / {n_control} control  "
          f"({'✅ adequate' if min(n_treated, n_control) >= n_per_group_min else '⚠️ below minimum'})")
    print(f"  Achieved power:                 {power_achieved*100:.1f}%")

    # ── Hypothesis decision ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HYPOTHESIS DECISION — BH2")
    print("="*60)
    if ATT is not None:
        if ATT_p < 0.05 and ATT < 0:
            print(f"  ATT = {ATT:.5f}  |  p = {ATT_p:.4f}  |  α = 0.05")
            print(f"  p < α and ATT < 0  →  REJECT H₀")
            print(f"  RPS adoption causally reduces CO₂ intensity.")
        else:
            print(f"  ATT = {ATT:.5f}  |  p = {ATT_p:.4f}  |  α = 0.05")
            print(f"  FAIL TO REJECT H₀")
    print("="*60)

    cf_str  = (f"ATT = {ATT:+.5f} tons/MWh  ATE = {ATE:+.5f}"
               if ATT is not None else "not run")
    es_str  = (f"{post_avg:+.5f} tons/MWh  ({post_sig_n}/{len(post_betas)} post-periods p<0.10)"
               if not np.isnan(post_avg) else "not available")
    pt_str  = (f"✅ Hold ({pre_sig}/{len(pre_ks)} sig)"
               if parallel_holds else f"⚠️  {pre_sig} significant pre-period(s)")

    print(f"\n  Causal Forest:        {cf_str}")
    print(f"  Event-Study post avg: {es_str}")
    print(f"  Parallel trends:      {pt_str}")
    print("\n  BH2 COMPLETE ✅")
    print("="*70)

    return es, ATT


if __name__ == "__main__":
    run_bh2()
