"""
BH4: XGBoost + SHAP — Interpretable ML for Decarbonization Predictors
What drives high vs low decarbonization at the state level?
All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle

Ported from BH4.ipynb — logic unchanged, paths updated to local.
"""

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_CLEAN.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_START, CORE_END = 2005, 2022
RANDOM_STATE = 42

FEATURES = [
    "Renewable_Share_Pct",
    "Fossil_Intensity",
    "Has_RPS",
    "Years_Since_RPS",
    "GDP_Growth_Rate_Annual",
    "Temp_Extreme",
    "Nuclear_Share_Pct",
    "Avg_Temp_F",
]


def run_bh4():
    print("=" * 70)
    print("  BH4: XGBOOST + SHAP — DECARBONIZATION PREDICTOR ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df = df[~df["STATE"].isin(["US-TOTAL", "US", "USA"])].copy()
    df_core = df[(df["YEAR"] >= CORE_START) & (df["YEAR"] <= CORE_END)].copy()

    agg_map = {
        "CO2_Intensity_Combined" : "mean",
        "Renewable_Share_Pct"    : "mean",
        "Fossil_Intensity"       : "mean",
        "Has_RPS"                : "max",
        "Years_Since_RPS"        : "max",
        "GDP_Growth_Rate_Annual" : "mean",
        "Temp_Extreme"           : "mean",
        "Total_Generation_MWh"   : "sum",
        "Nuclear_Share_Pct"      : "mean",
        "Avg_Temp_F"             : "mean",
    }
    df_annual = df_core.groupby(["STATE", "YEAR"]).agg(agg_map).reset_index()

    df_annual["CO2_tercile"] = df_annual.groupby("YEAR")[
        "CO2_Intensity_Combined"].transform(
        lambda x: pd.qcut(x, q=3, labels=[0, 1, 2], duplicates="drop"))
    df_annual["High_Decarbonizer"] = (
        df_annual["CO2_tercile"].astype(float) == 0).astype(int)

    print(f"\n📊 SAMPLE:")
    print(f"   Annual obs        : {len(df_annual):,}")
    print(f"   States            : {df_annual['STATE'].nunique()}")
    print(f"   Years             : {CORE_START}–{CORE_END}")
    print(f"   High decarbonizers: {df_annual['High_Decarbonizer'].sum():,} "
          f"({df_annual['High_Decarbonizer'].mean()*100:.1f}%)")

    df_model = df_annual.dropna(subset=FEATURES + ["High_Decarbonizer"])
    X = df_model[FEATURES].values
    y = df_model["High_Decarbonizer"].values

    print(f"\n   Features         : {len(FEATURES)}")
    print(f"   Modelling sample : {len(df_model):,}")
    print(f"   Class balance    : {y.mean()*100:.1f}% positive")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1,
    )
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    # ── Cross-validated performance ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE (5-fold stratified CV)")
    print("=" * 70)

    models = {
        "XGBoost (primary)"    : xgb_model,
        "LightGBM (secondary)" : lgb_model,
        "Random Forest (base)" : rf_model,
    }
    cv_results = {}
    for name, model in models.items():
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_results[name] = {
            "AUC_mean": auc_scores.mean(), "AUC_std": auc_scores.std(),
            "Acc_mean": acc_scores.mean(), "Acc_std": acc_scores.std(),
        }
        print(f"\n   {name}")
        print(f"   AUC  : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        print(f"   Acc  : {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    # ── Fit final models ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FITTING FINAL MODELS ON FULL DATASET")
    print("=" * 70)

    xgb_model.fit(X, y)
    lgb_model.fit(X, y)
    rf_model.fit(X, y)

    y_pred_xgb = xgb_model.predict(X);  y_prob_xgb = xgb_model.predict_proba(X)[:, 1]
    y_pred_lgb = lgb_model.predict(X);  y_prob_lgb = lgb_model.predict_proba(X)[:, 1]
    y_pred_rf  = rf_model.predict(X);   y_prob_rf  = rf_model.predict_proba(X)[:, 1]

    print(f"\n   XGBoost  — Full AUC: {roc_auc_score(y, y_prob_xgb):.4f}  "
          f"Acc: {accuracy_score(y, y_pred_xgb):.4f}")
    print(f"   LightGBM — Full AUC: {roc_auc_score(y, y_prob_lgb):.4f}  "
          f"Acc: {accuracy_score(y, y_pred_lgb):.4f}")
    print(f"   RF       — Full AUC: {roc_auc_score(y, y_prob_rf):.4f}  "
          f"Acc: {accuracy_score(y, y_pred_rf):.4f}")
    print(f"\n   XGBoost Classification Report:")
    print(classification_report(y, y_pred_xgb, target_names=["Low", "High"]))

    # ── GridSearch tuning ─────────────────────────────────────────────────────
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth":    [3, 4, 5],
        "learning_rate":[0.05, 0.1],
    }
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)
    best_xgb_model = grid_search.best_estimator_
    y_pred_best = best_xgb_model.predict(X)
    y_prob_best = best_xgb_model.predict_proba(X)[:, 1]
    print(f"\n   Best params:  {grid_search.best_params_}")
    print(f"   Tuned CV AUC: {grid_search.best_score_:.4f}")
    print(f"   Tuned Full AUC: {roc_auc_score(y, y_prob_best):.4f}  "
          f"Acc: {accuracy_score(y, y_pred_best):.4f}")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — Tuned XGBoost Feature Importance")
    print("=" * 70)

    explainer        = shap.TreeExplainer(best_xgb_model)
    shap_values      = explainer.shap_values(X)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_shap = np.abs(sv).mean(axis=0)
    shap_df   = pd.DataFrame({
        "Feature"   : FEATURES,
        "SHAP_mean" : mean_shap,
        "SHAP_pct"  : mean_shap / mean_shap.sum() * 100,
    }).sort_values("SHAP_mean", ascending=False).reset_index(drop=True)

    print(f"\n   {'Rank':<5} {'Feature':<25} {'SHAP':>8}  {'%':>6}")
    print("   " + "-"*48)
    for i, row in shap_df.iterrows():
        print(f"   {i+1:<5} {row['Feature']:<25} {row['SHAP_mean']:>8.4f}  "
              f"{row['SHAP_pct']:>5.1f}%")

    # ── Cross-model importance ─────────────────────────────────────────────────
    xgb_imp_raw = mean_shap / mean_shap.sum()
    lgb_imp_raw = lgb_model.booster_.feature_importance(importance_type="gain")
    lgb_imp     = lgb_imp_raw / lgb_imp_raw.sum()
    rf_imp      = rf_model.feature_importances_

    imp_rows = []
    for feat, x_v, l_v, r_v in zip(FEATURES, xgb_imp_raw, lgb_imp, rf_imp):
        imp_rows.append({"Feature": feat, "XGB": x_v, "LGB": l_v,
                         "RF": r_v, "Avg": (x_v+l_v+r_v)/3})
    imp_df = pd.DataFrame(imp_rows).sort_values("Avg", ascending=False)

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    ensemble_model = StackingClassifier(
        estimators=[("xgb", best_xgb_model), ("lgbm", lgb_model), ("rf", rf_model)],
        final_estimator=LogisticRegression(random_state=RANDOM_STATE, solver="liblinear"),
        cv=cv, n_jobs=-1, passthrough=True,
    )
    ens_auc = cross_val_score(ensemble_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    ens_acc = cross_val_score(ensemble_model, X, y, cv=cv, scoring="accuracy",  n_jobs=-1)
    cv_results["Stacking Ensemble"] = {
        "AUC_mean": ens_auc.mean(), "AUC_std": ens_auc.std(),
        "Acc_mean": ens_acc.mean(), "Acc_std": ens_acc.std(),
    }
    print(f"\n   Stacking Ensemble  AUC: {ens_auc.mean():.4f} ± {ens_auc.std():.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 GENERATING PLOTS...")

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, :])
    shap.summary_plot(sv, X, feature_names=FEATURES,
                      plot_type="dot", show=False, max_display=8, color_bar=True)
    ax0.set_title("SHAP Summary Plot — Tuned XGBoost\n"
                  "(each dot = one state-year; colour = feature value)", fontsize=11)

    ax1 = fig.add_subplot(gs[1, 0])
    colors_shap = ["#2ecc71" if i==0 else "#3498db" if i==1 else "#95a5a6"
                   for i in range(len(shap_df))]
    bars = ax1.barh(shap_df["Feature"][::-1], shap_df["SHAP_mean"][::-1],
                    color=colors_shap[::-1], edgecolor="black", lw=0.6)
    ax1.set_xlabel("Mean |SHAP value|")
    ax1.set_title("XGBoost Feature Importance\n(SHAP — mean absolute impact)")
    ax1.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, shap_df["SHAP_mean"][::-1]):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(FEATURES))
    w = 0.25
    ax2.bar(x_pos - w, [dict(zip(FEATURES, xgb_imp_raw))[f] for f in imp_df["Feature"]],
            w, label="XGBoost", color="#2ecc71", edgecolor="black", lw=0.5)
    ax2.bar(x_pos,     [dict(zip(FEATURES, lgb_imp))[f]     for f in imp_df["Feature"]],
            w, label="LightGBM", color="#3498db", edgecolor="black", lw=0.5)
    ax2.bar(x_pos + w, [dict(zip(FEATURES, rf_imp))[f]      for f in imp_df["Feature"]],
            w, label="Random Forest", color="#e74c3c", edgecolor="black", lw=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(imp_df["Feature"], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Normalised Importance")
    ax2.set_title("Feature Importance: All Three Models")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[2, 0])
    for name_r, probs, col in [
        ("XGBoost",       y_prob_best, "#2ecc71"),
        ("LightGBM",      y_prob_lgb,  "#3498db"),
        ("Random Forest", y_prob_rf,   "#e74c3c"),
    ]:
        fpr, tpr, _ = roc_curve(y, probs)
        ax3.plot(fpr, tpr, lw=2, color=col,
                 label=f"{name_r} AUC={roc_auc_score(y, probs):.3f}")
    ax3.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random")
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves — All Models")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    model_names = ["XGBoost", "LightGBM", "Random\nForest"]
    auc_means   = [cv_results["XGBoost (primary)"]["AUC_mean"],
                   cv_results["LightGBM (secondary)"]["AUC_mean"],
                   cv_results["Random Forest (base)"]["AUC_mean"]]
    auc_stds    = [cv_results["XGBoost (primary)"]["AUC_std"],
                   cv_results["LightGBM (secondary)"]["AUC_std"],
                   cv_results["Random Forest (base)"]["AUC_std"]]
    bars4 = ax4.bar(range(3), auc_means, color=["#2ecc71","#3498db","#e74c3c"],
                    edgecolor="black", lw=0.7, width=0.5)
    ax4.errorbar(range(3), auc_means, yerr=auc_stds,
                 fmt="none", color="black", capsize=6, lw=2)
    ax4.set_xticks(range(3)); ax4.set_xticklabels(model_names, fontsize=9)
    ax4.set_ylabel("Cross-Validated AUC")
    ax4.set_title("Model Comparison: 5-Fold CV AUC\n(mean ± std)")
    ax4.set_ylim(0.5, 1.05)
    ax4.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    for bar, mean, std in zip(bars4, auc_means, auc_stds):
        ax4.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BH4: XGBoost + SHAP — Decarbonization Predictor Analysis",
                 fontsize=14, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "BH4_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP dependence plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for i, feat in enumerate(shap_df["Feature"].iloc[:2].tolist()):
        shap.dependence_plot(FEATURES.index(feat), sv, X,
                             feature_names=FEATURES, ax=axes2[i], show=False)
        axes2[i].set_title(f"SHAP Dependence: {feat}", fontsize=11)
    plt.suptitle("BH4: SHAP Dependence Plots — Top 2 Features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH4_shap_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Fossil_Intensity dependence
    fig3, ax5 = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(FEATURES.index("Fossil_Intensity"), sv, X,
                         feature_names=FEATURES, interaction_index=None, ax=ax5, show=False)
    ax5.set_title("SHAP Dependence: Fossil_Intensity vs. Decarbonization Probability")
    ax5.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH4_shap_dependence_fossil_intensity.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Low", "Predicted High"],
                yticklabels=["Actual Low", "Actual High"])
    plt.title("Confusion Matrix for Tuned XGBoost Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH4_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────
    shap_df.to_csv(OUTPUT_DIR / "BH4_shap_importance.csv", index=False)
    imp_df.to_csv(OUTPUT_DIR / "BH4_feature_importance_all_models.csv", index=False)
    pd.DataFrame([
        {"Model": m, "CV_AUC_mean": round(v["AUC_mean"],4),
         "CV_AUC_std": round(v["AUC_std"],4),
         "CV_Acc_mean": round(v["Acc_mean"],4),
         "CV_Acc_std": round(v["Acc_std"],4)}
        for m, v in cv_results.items()
    ]).to_csv(OUTPUT_DIR / "BH4_cv_performance.csv", index=False)

    df_model_out = df_model[["STATE", "YEAR", "High_Decarbonizer"]].copy()
    df_model_out["XGB_prob"] = y_prob_best
    df_model_out["XGB_pred"] = y_pred_best
    df_model_out.to_csv(OUTPUT_DIR / "BH4_predictions.csv", index=False)

    print(f"✅ Saved outputs to {OUTPUT_DIR}")

    # ── Sample size & power ───────────────────────────────────────────────────
    print("""
SAMPLE SIZE & POWER ANALYSIS — BH4
────────────────────────────────────────────────────────────""")
    n_total    = len(df_model)
    n_positive = int(y.sum())
    n_negative = n_total - n_positive
    n_features = len(FEATURES)
    epv        = n_positive / n_features
    print(f"  Total modelling sample:         {n_total:,}")
    print(f"  Positive class (high decarb):   {n_positive:,}  ({n_positive/n_total*100:.1f}%)")
    print(f"  Negative class (low decarb):    {n_negative:,}")
    print(f"  Features:                       {n_features}")
    print(f"  Events per variable (EPV):      {epv:.1f}  "
          f"({'✅ EPV > 10 — adequate' if epv >= 10 else '⚠️ EPV < 10'})")
    print(f"  CV strategy:                    5-fold stratified")

    # ── Hypothesis decision ───────────────────────────────────────────────────
    xgb_cv_auc = cv_results["XGBoost (primary)"]["AUC_mean"]
    print("\n" + "="*60)
    print("  HYPOTHESIS DECISION — BH4")
    print("="*60)
    print(f"  CV AUC = {xgb_cv_auc:.4f}  |  Threshold = 0.70")
    if xgb_cv_auc > 0.70:
        print(f"  AUC >> 0.5  →  REJECT H₀")
        print(f"  Structural features reliably predict decarbonizer status.")
        print(f"  Top predictor: {shap_df['Feature'].iloc[0]} "
              f"({shap_df['SHAP_pct'].iloc[0]:.1f}% SHAP importance)")
    else:
        print(f"  AUC not meaningfully above 0.5  →  FAIL TO REJECT H₀")
    print("="*60)
    print("\n  BH4 COMPLETE ✅")

    return best_xgb_model, shap_df


if __name__ == "__main__":
    run_bh4()
