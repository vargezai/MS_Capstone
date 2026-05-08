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
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "FINAL_MASTER_DATASET_FEATURES.csv"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "BH3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK  = 12
HORIZONS  = [1, 3, 6]
TRAIN_END = 2018
VAL_END   = 2020
TEST_END  = 2022
TARGET    = "CO2_Intensity_Combined"

FEATURES = [
    # original
    "CO2_Intensity_Combined",
    "Renewable_Share_Pct", "Fossil_Intensity",
    "Total_Generation_MWh", "Avg_Temp_F",
    "GDP_Growth_Rate_Annual", "Has_RPS",
    "Years_Since_RPS", "Nuclear_Share_Pct",
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
TGT_IDX = FEATURES.index(TARGET)   # auto-computed from list
N_FEATS = len(FEATURES)             # auto-computed from list


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

    run_lstm_tuning()
    return results


def run_lstm_tuning():
    """
    Hyperparameter optimisation for the BH3 LSTM using Keras Tuner (Hyperband).
    Tunes on h=1 train/val split, then applies best config to h=1, h=3, h=6.
    Compares tuned vs baseline metrics and saves:
      - BH3_hparam_results.csv        (all Hyperband trial results for h=1)
      - BH3_hparam_comparison.csv     (baseline vs tuned across all horizons)
      - BH3_hparam_tuning.png         (comparison figure)
    """
    import keras_tuner as kt
    from tensorflow.keras.callbacks import EarlyStopping

    print("=" * 70)
    print("  BH3: LSTM HYPERPARAMETER OPTIMISATION (Keras Tuner — Hyperband)")
    print("=" * 70)

    # ── Load & prepare data (identical pipeline to run_bh3) ──────────────────
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

    df_tr_raw = df_core[df_core["YEAR"] <= TRAIN_END]
    scaler    = MinMaxScaler((0, 1))
    scaler.fit(df_tr_raw[FEATURES].values)

    df_sc           = df_core.copy()
    df_sc[FEATURES] = scaler.transform(df_core[FEATURES].values)

    # Build train/val for h=1 (used for tuning)
    (X_tr1, y_tr1, X_va1, y_va1,
     X_te1, y_te1, _) = build_all(
        df_sc, FEATURES, TARGET, LOOKBACK, 1, TRAIN_END, VAL_END)
    print(f"\n  Tuning data (h=1): Train {X_tr1.shape}  Val {X_va1.shape}")

    # ── Define model builder ─────────────────────────────────────────────────
    def _build(hp):
        u1 = hp.Choice("units_1",  [32, 64, 128])
        u2 = hp.Choice("units_2",  [16, 32, 64])
        dr = hp.Choice("dropout",  [0.1, 0.2, 0.3])
        lr = hp.Choice("lr",       [1e-3, 5e-4, 1e-4])
        m  = Sequential([
            LSTM(u1, return_sequences=True,
                 input_shape=(LOOKBACK, N_FEATS)),
            Dropout(dr),
            LSTM(u2, return_sequences=False),
            Dropout(dr),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
        return m

    TUNER_DIR = OUTPUT_DIR / "tuner"
    tuner = kt.Hyperband(
        _build,
        objective="val_loss",
        max_epochs=30,
        factor=3,
        directory=str(TUNER_DIR),
        project_name="bh3_lstm",
        overwrite=True,
    )

    print(f"\n  Search space summary:")
    tuner.search_space_summary()

    print(f"\n  Running Hyperband search (max_epochs=30, factor=3)...")
    tuner.search(
        X_tr1, y_tr1,
        validation_data=(X_va1, y_va1),
        epochs=30,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True, verbose=0)],
        verbose=0,
    )

    # ── Extract all trial results ─────────────────────────────────────────────
    trial_rows = []
    for trial in tuner.oracle.trials.values():
        hp_vals = trial.hyperparameters.values
        score   = trial.score
        if score is not None:
            trial_rows.append({
                "trial_id":  trial.trial_id,
                "units_1":   hp_vals.get("units_1"),
                "units_2":   hp_vals.get("units_2"),
                "dropout":   hp_vals.get("dropout"),
                "lr":        hp_vals.get("lr"),
                "val_loss":  round(float(score), 8),
            })
    df_trials = (pd.DataFrame(trial_rows)
                 .sort_values("val_loss")
                 .reset_index(drop=True))
    df_trials["rank"] = df_trials.index + 1

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_u1 = best_hp.get("units_1")
    best_u2 = best_hp.get("units_2")
    best_dr = best_hp.get("dropout")
    best_lr = best_hp.get("lr")

    print(f"\n  Best hyperparameters found:")
    print(f"    units_1 = {best_u1}  |  units_2 = {best_u2}  |  "
          f"dropout = {best_dr}  |  lr = {best_lr:.0e}")
    print(f"    (baseline: units_1=64, units_2=32, dropout=0.2, lr=1e-3)")

    print(f"\n  Top 5 trials (h=1 val_loss):")
    print(f"  {'Rank':<5} {'units_1':>8} {'units_2':>8} {'dropout':>8} {'lr':>8} {'val_loss':>12}")
    print("  " + "-" * 55)
    for _, r in df_trials.head(5).iterrows():
        print(f"  {int(r['rank']):<5} {int(r['units_1']):>8} {int(r['units_2']):>8} "
              f"{r['dropout']:>8.1f} {r['lr']:>8.0e} {r['val_loss']:>12.8f}")

    df_trials.to_csv(OUTPUT_DIR / "BH3_hparam_results.csv", index=False)

    # ── Retrain with best config across all 3 horizons ───────────────────────
    print(f"\n{'='*70}")
    print(f"  RETRAINING WITH BEST CONFIG ACROSS ALL 3 HORIZONS")
    print(f"{'='*70}")

    # Load baseline metrics
    baseline_path = OUTPUT_DIR / "BH3_metrics_table.csv"
    if baseline_path.exists():
        df_baseline = pd.read_csv(baseline_path)
        baseline_metrics = {
            int(r["Horizon"]): r for _, r in df_baseline.iterrows()
        }
    else:
        baseline_metrics = {}

    def _build_tuned(n_features, lookback):
        m = Sequential([
            LSTM(best_u1, return_sequences=True,
                 input_shape=(lookback, n_features)),
            Dropout(best_dr),
            LSTM(best_u2, return_sequences=False),
            Dropout(best_dr),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        m.compile(optimizer=Adam(learning_rate=best_lr, clipnorm=1.0),
                  loss="mse", metrics=["mae"])
        return m

    comparison_rows = []
    for horizon in HORIZONS:
        print(f"\n  h = {horizon} month{'s' if horizon > 1 else ''}")

        (X_tr, y_tr, X_va, y_va,
         X_te, y_te, _) = build_all(
            df_sc, FEATURES, TARGET, LOOKBACK, horizon, TRAIN_END, VAL_END)

        model = _build_tuned(N_FEATS, LOOKBACK)
        cb = [
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-6, verbose=0),
        ]
        model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=150, batch_size=64, callbacks=cb, verbose=0)

        y_pred  = inv_transform(model.predict(X_te, verbose=0).flatten(),
                                scaler, TGT_IDX, N_FEATS)
        y_true  = inv_transform(y_te, scaler, TGT_IDX, N_FEATS)
        y_naive = inv_transform(X_te[:, -1, TGT_IDX], scaler, TGT_IDX, N_FEATS)

        rmse_t  = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_t    = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        skill_t = 1 - rmse_t / np.sqrt(mean_squared_error(y_true, y_naive))

        row = {
            "Horizon":       horizon,
            "RMSE_baseline": round(float(baseline_metrics[horizon]["RMSE"]), 5)
                             if horizon in baseline_metrics else float("nan"),
            "RMSE_tuned":    round(rmse_t, 5),
            "R2_baseline":   round(float(baseline_metrics[horizon]["R2"]), 4)
                             if horizon in baseline_metrics else float("nan"),
            "R2_tuned":      round(r2_t, 4),
            "Skill_baseline":round(float(baseline_metrics[horizon]["Skill"]), 4)
                             if horizon in baseline_metrics else float("nan"),
            "Skill_tuned":   round(skill_t, 4),
            "Best_units_1":  best_u1,
            "Best_units_2":  best_u2,
            "Best_dropout":  best_dr,
            "Best_lr":       best_lr,
        }
        comparison_rows.append(row)

        if horizon in baseline_metrics:
            rmse_b  = float(baseline_metrics[horizon]["RMSE"])
            delta   = (rmse_t - rmse_b) / rmse_b * 100
            print(f"    Baseline RMSE={rmse_b:.5f}  Tuned RMSE={rmse_t:.5f}  "
                  f"Δ={delta:+.1f}%  Skill={skill_t:+.4f}")
        else:
            print(f"    Tuned RMSE={rmse_t:.5f}  R²={r2_t:.4f}  Skill={skill_t:+.4f}")

        model.save(str(OUTPUT_DIR / f"BH3_lstm_tuned_h{horizon}.keras"))

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(OUTPUT_DIR / "BH3_hparam_comparison.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: val_loss across trials
    ax0 = axes[0]
    ax0.scatter(range(len(df_trials)), df_trials["val_loss"],
                c=range(len(df_trials)), cmap="YlOrRd_r", s=60, zorder=3)
    ax0.axhline(df_trials["val_loss"].iloc[0], color="green", lw=1.5,
                ls="--", label=f"Best={df_trials['val_loss'].iloc[0]:.6f}")
    ax0.set_xlabel("Trial (sorted by val_loss)")
    ax0.set_ylabel("Val loss (MSE, scaled)")
    ax0.set_title("Hyperband Trials — h=1\n(all configurations)")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # Panel 2: RMSE baseline vs tuned per horizon
    ax1 = axes[1]
    x   = np.arange(len(HORIZONS))
    w   = 0.32
    has_baseline = all(~np.isnan(r["RMSE_baseline"]) for r in comparison_rows)
    if has_baseline:
        ax1.bar(x - w/2, [r["RMSE_baseline"] for r in comparison_rows],
                w, label="Baseline", color="steelblue", edgecolor="black", lw=0.7)
    ax1.bar(x + w/2 if has_baseline else x,
            [r["RMSE_tuned"] for r in comparison_rows],
            w, label="Tuned", color="darkorange", edgecolor="black", lw=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax1.set_ylabel("RMSE (tons/MWh)")
    ax1.set_title("RMSE: Baseline vs Tuned\n(by forecast horizon)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y")

    # Panel 3: Skill score baseline vs tuned
    ax2 = axes[2]
    if has_baseline:
        ax2.plot(HORIZONS, [r["Skill_baseline"] for r in comparison_rows],
                 "o--", color="steelblue", lw=2, ms=8, label="Baseline")
    ax2.plot(HORIZONS, [r["Skill_tuned"] for r in comparison_rows],
             "s-", color="darkorange", lw=2, ms=8, label="Tuned")
    ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.6, label="Naive baseline")
    ax2.set_xlabel("Forecast horizon (months)")
    ax2.set_ylabel("Skill score vs naive")
    ax2.set_title("Skill Score: Baseline vs Tuned")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"BH3 Hyperparameter Optimisation (Hyperband)\n"
        f"Best config: units=({best_u1},{best_u2}), dropout={best_dr}, lr={best_lr:.0e}",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "BH3_hparam_tuning.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  ✅ Saved: BH3_hparam_results.csv")
    print(f"  ✅ Saved: BH3_hparam_comparison.csv")
    print(f"  ✅ Saved: BH3_hparam_tuning.png")
    print(f"  ✅ Saved: BH3_lstm_tuned_h1/3/6.keras")
    print("\n  LSTM HYPERPARAMETER OPTIMISATION COMPLETE ✅")
    return df_comp


if __name__ == "__main__":
    run_bh3()
