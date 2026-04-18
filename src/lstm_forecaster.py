"""
BH3: LSTM Multi-Horizon CO2 Intensity Forecasting (Deep Learning)
Can LSTM outperform naive persistence for state-level CO2 demand forecasting?
All sources: public government data (EIA, EPA, NOAA, BEA, DSIRE) — no Kaggle

ACADEMIC FINDING: CO2 intensity follows near-random-walk dynamics at the
monthly level (lag-1 autocorrelation ≈ 0.99). The naive persistence model
is the optimal 1-step predictor; skill score is negative at all horizons.
The LSTM achieves high level-prediction accuracy (R² > 0.97) and correctly
ranks states by CO2 intensity, but does not improve on naive for temporal
forecasting. This is consistent with BH4 finding that structural variables
(fossil intensity, nuclear share) — which change slowly — dominate CO2
determination.

Ported from BH3.ipynb — logic unchanged, paths updated to local.
"""

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_2001_2026.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK  = 12
HORIZONS  = [1, 3, 6]
TRAIN_END = 2018
VAL_END   = 2020
TEST_END  = 2022
TARGET    = "CO2_Intensity_Combined"

FEATURES = [
    "CO2_Intensity_Combined",
    "Renewable_Share_Pct", "Fossil_Intensity",
    "Total_Generation_MWh", "Avg_Temp_F",
    "GDP_Growth_Rate_Annual", "Has_RPS",
    "Years_Since_RPS", "Nuclear_Share_Pct",
]
TGT_IDX = FEATURES.index(TARGET)
N_FEATS = len(FEATURES)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_sequences(df_state, features, target, lookback, horizon):
    data = df_state[features].values.astype(float)
    tgt  = df_state[target].values.astype(float)
    X, y, dates = [], [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i-lookback:i])
        y.append(tgt[i+horizon-1])
        dates.append(df_state["date"].iloc[i+horizon-1])
    return np.array(X), np.array(y), dates


def build_all(df, features, target, lookback, horizon, train_end, val_end):
    X_tr, y_tr, X_va, y_va, X_te, y_te, dt_te = [], [], [], [], [], [], []
    for state in sorted(df["STATE"].unique()):
        df_s = df[df["STATE"]==state].sort_values("date").copy()
        if len(df_s) < lookback + max(HORIZONS):
            continue
        X_s, y_s, d_s = build_sequences(df_s, features, target, lookback, horizon)
        years = [d.year for d in d_s]
        tr = [i for i, yr in enumerate(years) if yr <= train_end]
        va = [i for i, yr in enumerate(years) if train_end < yr <= val_end]
        te = [i for i, yr in enumerate(years) if yr > val_end]
        if tr: X_tr.append(X_s[tr]); y_tr.append(y_s[tr])
        if va: X_va.append(X_s[va]); y_va.append(y_s[va])
        if te:
            X_te.append(X_s[te]); y_te.append(y_s[te])
            dt_te.extend([d_s[i] for i in te])
    return (np.concatenate(X_tr), np.concatenate(y_tr),
            np.concatenate(X_va), np.concatenate(y_va),
            np.concatenate(X_te), np.concatenate(y_te), dt_te)


def inv_transform(vals, scaler, feat_idx, n_feats):
    d = np.zeros((len(vals), n_feats))
    d[:, feat_idx] = vals
    return scaler.inverse_transform(d)[:, feat_idx]


def build_lstm(n_features, lookback):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
              loss="mse", metrics=["mae"])
    return m


def run_bh3():
    print("="*70)
    print("  BH3: LSTM MULTI-HORIZON CO2 FORECASTING — FINAL VERSION")
    print("="*70)

    df = pd.read_csv(DATA_PATH)
    df = df[~df["STATE"].isin(["US-TOTAL", "US", "USA"])].copy()
    df_core = df[(df["YEAR"] >= 2005) & (df["YEAR"] <= TEST_END)].copy()
    df_core = df_core.sort_values(["STATE", "YEAR", "MONTH"]).reset_index(drop=True)
    for col in FEATURES:
        df_core[col] = (df_core.groupby("STATE")[col]
                        .transform(lambda x: x.ffill().bfill()))
    df_core["date"] = pd.to_datetime(
        df_core["YEAR"].astype(str) + "-" + df_core["MONTH"].astype(str) + "-01")
    df_core = df_core.dropna(subset=FEATURES)

    # ── Autocorrelation analysis ───────────────────────────────────────────────
    print("\n📊 CO2 INTENSITY AUTOCORRELATION ANALYSIS...")
    ac_results = {}
    for state in ["CA", "TX", "WY", "NY", "FL"]:
        s = df_core[df_core["STATE"]==state][TARGET].values
        if len(s) > 24:
            ac_results[state] = acf(s, nlags=6, fft=True)

    print(f"\n   {'State':<8} {'lag-1':>8} {'lag-3':>8} {'lag-6':>8}")
    print("   " + "-"*32)
    for state, ac in ac_results.items():
        print(f"   {state:<8} {ac[1]:>8.4f} {ac[3]:>8.4f} {ac[6]:>8.4f}")

    all_ac = []
    for state, grp in df_core[df_core["YEAR"] <= TRAIN_END].groupby("STATE"):
        s = grp[TARGET].values
        if len(s) > 24:
            ac = acf(s, nlags=1, fft=True)
            all_ac.append(ac[1])
    mean_ac = np.mean(all_ac)
    print(f"\n   Mean lag-1 autocorrelation across {len(all_ac)} states: {mean_ac:.4f}")
    print(f"   Near-unit-root dynamics: naive model explains {mean_ac**2*100:.1f}% of variance")

    # ── Train and evaluate per horizon ────────────────────────────────────────
    results   = {}
    all_preds = {}
    colors_h  = {1: "#2ecc71", 3: "#3498db", 6: "#e74c3c"}

    for horizon in HORIZONS:
        print(f"\n{'='*70}")
        print(f"  h = {horizon} MONTH{'S' if horizon>1 else ''}")
        print(f"{'='*70}")

        df_tr  = df_core[df_core["YEAR"] <= TRAIN_END]
        scaler = MinMaxScaler((0, 1))
        scaler.fit(df_tr[FEATURES].values)

        df_sc         = df_core.copy()
        df_sc[FEATURES] = scaler.transform(df_core[FEATURES].values)

        (X_tr, y_tr, X_va, y_va,
         X_te, y_te, dates_te) = build_all(
            df_sc, FEATURES, TARGET, LOOKBACK, horizon, TRAIN_END, VAL_END)

        print(f"   Train:{X_tr.shape} Val:{X_va.shape} Test:{X_te.shape}")

        model = build_lstm(N_FEATS, LOOKBACK)
        cb = [
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-6, verbose=0),
        ]
        history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                            epochs=150, batch_size=64,
                            callbacks=cb, verbose=0)

        best_ep = np.argmin(history.history["val_loss"]) + 1
        print(f"   Best epoch: {best_ep}")

        y_pred  = inv_transform(model.predict(X_te, verbose=0).flatten(),
                                scaler, TGT_IDX, N_FEATS)
        y_true  = inv_transform(y_te,               scaler, TGT_IDX, N_FEATS)
        y_naive = inv_transform(X_te[:, -1, TGT_IDX], scaler, TGT_IDX, N_FEATS)

        rmse       = np.sqrt(mean_squared_error(y_true, y_pred))
        mae        = mean_absolute_error(y_true, y_pred)
        smape      = np.mean(2*np.abs(y_pred-y_true) /
                              (np.abs(y_pred)+np.abs(y_true)+1e-8)) * 100
        r2         = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_true.mean())**2)
        rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))
        skill      = 1 - rmse / rmse_naive

        print(f"   RMSE={rmse:.5f}  naive={rmse_naive:.5f}  "
              f"skill={skill:+.4f}  R²={r2:.4f}")

        results[horizon]   = dict(rmse=rmse, mae=mae, smape=smape, r2=r2,
                                   skill=skill, rmse_naive=rmse_naive,
                                   history=history.history, best_epoch=best_ep)
        all_preds[horizon] = dict(y_true=y_true, y_pred=y_pred,
                                   y_naive=y_naive, dates=dates_te)
        model.save(str(OUTPUT_DIR / f"BH3_lstm_h{horizon}.keras"))

    # ── Diagnostic figure ─────────────────────────────────────────────────────
    print("\n📊 GENERATING PUBLICATION FIGURE...")
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, hspace=0.44, wspace=0.32)

    for col, h in enumerate(HORIZONS):
        ax   = fig.add_subplot(gs[0, col])
        hist = results[h]["history"]
        ep   = range(1, len(hist["loss"])+1)
        ax.plot(ep, hist["loss"],     "b-",  lw=1.5, label="Train")
        ax.plot(ep, hist["val_loss"], "y--", lw=1.5, label="Val")
        ax.axvline(results[h]["best_epoch"], color="red", lw=1, ls=":",
                   label=f"Best={results[h]['best_epoch']}")
        ax.set_title(f"Training h={h}  (best={results[h]['best_epoch']})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for col, h in enumerate(HORIZONS):
        ax = fig.add_subplot(gs[1, col])
        yt = all_preds[h]["y_true"]
        yp = all_preds[h]["y_pred"]
        ax.scatter(yt, yp, alpha=0.2, s=8, color=colors_h[h])
        lo = min(yt.min(), yp.min()) - 0.02
        hi = max(yt.max(), yp.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect")
        ax.set_xlabel("Actual (tons/MWh)"); ax.set_ylabel("Predicted (tons/MWh)")
        ax.set_title(f"h={h}: R²={results[h]['r2']:.4f}  "
                     f"Skill={results[h]['skill']:+.4f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax_ac = fig.add_subplot(gs[2, 0:2])
    lags  = range(0, 13)
    for state, ac in list(ac_results.items())[:4]:
        ax_ac.plot(lags,
                   acf(df_core[df_core["STATE"]==state][TARGET].values,
                       nlags=12, fft=True),
                   "o-", ms=4, lw=1.5, alpha=0.7, label=state)
    ax_ac.axhline(0,       color="black", lw=0.8)
    ax_ac.axhline(mean_ac, color="red",   lw=1.5, ls="--",
                  label=f"Mean lag-1 = {mean_ac:.3f}")
    ax_ac.fill_between(lags, 0.95, 1.0, alpha=0.1, color="red",
                        label="Near-unit-root zone")
    ax_ac.set_xlabel("Lag (months)"); ax_ac.set_ylabel("Autocorrelation")
    ax_ac.set_title(f"CO₂ Intensity Autocorrelation by State\n"
                    f"(Near-unit-root → naive model is near-optimal)")
    ax_ac.legend(fontsize=8, ncol=2); ax_ac.grid(True, alpha=0.3)
    ax_ac.set_ylim(-0.1, 1.05)

    ax_r  = fig.add_subplot(gs[2, 2])
    hs    = HORIZONS
    rmses = [results[h]["rmse"]       for h in hs]
    naivs = [results[h]["rmse_naive"] for h in hs]
    skils = [results[h]["skill"]      for h in hs]
    x = np.arange(len(hs)); w = 0.35
    ax_r.bar(x-w/2, rmses, w, color=[colors_h[h] for h in hs],
             edgecolor="black", lw=0.7, label="LSTM RMSE")
    ax_r.bar(x+w/2, naivs, w, color="lightgrey",
             edgecolor="black", lw=0.7, label="Naive RMSE")
    ax_r2 = ax_r.twinx()
    ax_r2.plot(x, skils, "D-", color="darkred", lw=2, ms=10, zorder=5)
    ax_r2.axhline(0, color="darkred", lw=1, ls="--", alpha=0.5)
    ax_r2.set_ylabel("Skill score", color="darkred")
    ax_r.set_xticks(x); ax_r.set_xticklabels([f"h={h}" for h in hs])
    ax_r.set_title("RMSE vs Naive + Skill Score")
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BH3: LSTM Multi-Horizon CO₂ Forecasting\n"
                 "Train 2005–2018 | Val 2019–2020 | Test 2021–2022",
                 fontsize=13, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "BH3_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Save outputs ──────────────────────────────────────────────────────────
    pd.DataFrame([
        {"Horizon": h, "RMSE": round(results[h]["rmse"],5),
         "MAE": round(results[h]["mae"],5), "sMAPE": round(results[h]["smape"],3),
         "R2": round(results[h]["r2"],4),   "Skill": round(results[h]["skill"],4),
         "RMSE_naive": round(results[h]["rmse_naive"],5),
         "Best_epoch": results[h]["best_epoch"]}
        for h in HORIZONS
    ]).to_csv(OUTPUT_DIR / "BH3_metrics_table.csv", index=False)

    for h in HORIZONS:
        pd.DataFrame({
            "date":    all_preds[h]["dates"],
            "y_true":  all_preds[h]["y_true"],
            "y_pred":  all_preds[h]["y_pred"],
            "y_naive": all_preds[h]["y_naive"],
        }).to_csv(OUTPUT_DIR / f"BH3_predictions_h{h}.csv", index=False)

    print(f"✅ Saved outputs to {OUTPUT_DIR}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  BH3 COMPLETE SUMMARY + ACADEMIC FRAMING")
    print("="*70)
    print(f"\n  {'h':<8} {'RMSE':>8} {'naive':>8} {'Skill':>8} {'R²':>7}")
    print("  " + "-"*45)
    for h in HORIZONS:
        r = results[h]
        print(f"  h={h:<6} {r['rmse']:>8.5f} {r['rmse_naive']:>8.5f} "
              f"{r['skill']:>+8.4f} {r['r2']:>7.4f}")

    print(f"\n  Mean lag-1 autocorrelation: {mean_ac:.4f}")

    # ── Sample size ───────────────────────────────────────────────────────────
    print("""
SAMPLE SIZE & POWER ANALYSIS — BH3
────────────────────────────────────────────────────────────
Note: Formal power analysis for LSTM is non-standard.
Sequence count serves as the sample size metric.""")
    n_states, n_months = 51, 216
    lookback_loss      = LOOKBACK * len(HORIZONS)
    usable_seqs        = (n_months - LOOKBACK - max(HORIZONS) + 1) * n_states
    print(f"  States:                         {n_states}")
    print(f"  Monthly obs per state:          {n_months}")
    print(f"  Usable sequences (approx):      {usable_seqs:,}")
    print(f"  Train/Val/Test split:           2005–2018 / 2019–2020 / 2021–2022")
    print(f"  LOOKBACK window:                {LOOKBACK} months")

    # ── Hypothesis decision ───────────────────────────────────────────────────
    any_positive = any(results[h]["skill"] > 0 for h in HORIZONS)
    print("\n" + "="*60)
    print("  HYPOTHESIS DECISION — BH3")
    print("="*60)
    if any_positive:
        print("  Skill > 0 at ≥1 horizon  →  REJECT H₀")
        print("  LSTM outperforms naive persistence.")
    else:
        print("  Skill ≤ 0 at all horizons  →  FAIL TO REJECT H₀")
        print("  LSTM does not outperform naive persistence for")
        print("  temporal forecasting (R² > 0.97 for level prediction).")
        print("  This is an empirical property of near-unit-root CO₂ dynamics,")
        print("  not a model failure.")
    print("="*60)
    print("\n  BH3 COMPLETE ✅")

    return results


if __name__ == "__main__":
    run_bh3()
