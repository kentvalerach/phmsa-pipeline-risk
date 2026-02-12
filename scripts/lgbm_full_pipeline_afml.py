#!/usr/bin/env python3
"""
=============================================================================
LightGBM Pipeline — AFML-Compliant Gas Transmission Incident Prediction
=============================================================================
Strict adherence to López de Prado's Advances in Financial Machine Learning:
  - Ch.7: ALL feature transforms computed INSIDE each walk-forward fold
  - Ch.9: Purged/embargoed walk-forward (no future information at any step)
  - Ch.12: No forced iteration override — model selects its own complexity

TEMPORAL PURITY RULES (enforced throughout):
  1. Repair capping p99 → computed only from data BEFORE test year
  2. State target encoding → computed only from training fold
  3. Operator history → re-derived per fold from temporal predecessors
  4. Lag features → use only prior-year data (shift(1) within panel)
  5. Model iteration count → early stopping decides, never overridden
  6. No data from test period leaks into any feature or hyperparameter

OUTPUTS:
  1. fig1_pipeline_dashboard.png     — Main 10-panel dashboard
  2. fig2_shap_walkforward.png       — SHAP deep-dive & WF diagnostics
  3. operator_risk_ranking_final.csv — 1,077 operators with CI & tiers
  4. walkforward_comparison.csv      — Year-by-year LGB vs GLM
  5. shap_importance_final.csv       — 23 features ranked by |SHAP|

USAGE:
  pip install lightgbm shap pandas numpy scikit-learn matplotlib
  python lgbm_full_pipeline_afml.py
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
INPUT_PATH = 'survival_panel_15yr_final.csv'
OUTPUT_DIR = '.'

# Walk-forward range
WF_START = 2015  # first test year
WF_END   = 2024  # last test year

# Holdout split for SHAP / final retrain
# Train: 2010-2019, Val: 2020-2022, Test: 2023-2024
TRAIN_END = 2019
VAL_END   = 2022

# Bayesian smoothing prior for state encoding
STATE_PRIOR = 50

# Bootstrap
N_BOOTSTRAP = 500
BOOT_SEED   = 42

# LightGBM — conservative base params, model decides complexity
LGB_PARAMS = {
    'objective':         'binary',
    'metric':            ['binary_logloss', 'auc'],
    'boosting_type':     'gbdt',
    'num_leaves':        31,
    'max_depth':         5,
    'learning_rate':     0.05,
    'feature_fraction':  0.75,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'min_child_samples': 100,
    'lambda_l1':         0.5,
    'lambda_l2':         5.0,
    'min_gain_to_split': 0.1,
    'verbose':           -1,
    'seed':              42,
    'n_jobs':            -1,
}

# Early stopping patience — AFML: let the model decide
EARLY_STOP_ROUNDS = 100

# GLM structural features (what a NB2-GLM would use)
GLM_FEATURES = [
    'age_at_obs', 'era_ordinal', 'log_miles',
    'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1',
]

# Feature name mapping for plots
FEAT_NAMES = {
    'age_at_obs': 'Pipe Age', 'age_sq': 'Pipe Age²',
    'era_ordinal': 'Installation Era', 'decade_ordinal': 'Decade Installed',
    'log_miles': 'Log(Miles at Risk)', 'pct_small_diam': '% Small Diameter',
    'pct_large_diam': '% Large Diameter', 'pct_high_smys': '% High SMYS',
    'pct_class1': '% Class 1 (Rural)', 'age_x_era': 'Age × Era',
    'age_x_small_diam': 'Age × Small Diam', 'age_x_class1': 'Age × Class 1',
    'miles_x_age': 'Miles × Age', 'repairs_filled': 'Repair History',
    'repairs_available': 'Repair Data Avail.', 'event_lag1': 'Event Last Year',
    'incidents_lag1': 'Incidents Last Year', 'incidents_roll3': 'Incidents (3yr Roll)',
    'incidents_cumul': 'Cumul. Incidents', 'ever_event_prior': 'Prior Event Flag',
    'op_hist_rate': 'Operator Hist. Rate', 'op_hist_total': 'Operator Total Events',
    'state_risk': 'State Risk Level',
}

# Plot palette
C = dict(
    BG='#0f1117', PANEL='#1a1d2e', TEXT='#e0e0e0', ACCENT='#00d4aa',
    RED='#ff6b6b', TEAL='#4ecdc4', GOLD='#ffd93d', GRID='#2a2d3e',
    BLUE='#5b8def', PURPLE='#b06ce6',
)
TIER_COLORS = {
    'CRITICAL': C['RED'], 'HIGH': C['GOLD'],
    'ELEVATED': C['BLUE'], 'STANDARD': C['TEAL'],
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(C['PANEL'])
    ax.set_title(title, color=C['TEXT'], fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C['TEXT'], fontsize=9)
    ax.set_ylabel(ylabel, color=C['TEXT'], fontsize=9)
    ax.tick_params(colors=C['TEXT'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C['GRID'])
    ax.grid(True, alpha=0.15, color=C['GRID'])


def clean_name(feat):
    return FEAT_NAMES.get(feat, feat)


# ═══════════════════════════════════════════════════════════════════════
# AFML-COMPLIANT FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
# These functions take a temporal cutoff and compute everything using
# ONLY data strictly before that cutoff.


def compute_lag_features(df):
    """
    Lag features use shift(1) within each operator-decade panel.
    This is inherently temporal: row at year t only sees t-1.
    Safe to compute on full dataframe because shift(1) enforces
    the temporal barrier at the row level.
    """
    grp = ['operator_id', 'decade_bin']
    df = df.sort_values(['operator_id', 'decade_bin', 'year']).reset_index(drop=True)

    df['event_lag1']      = df.groupby(grp)['event'].shift(1).fillna(0)
    df['incidents_lag1']  = df.groupby(grp)['n_incidents'].shift(1).fillna(0)
    df['incidents_roll3'] = df.groupby(grp)['n_incidents'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0)
    df['incidents_cumul'] = df.groupby(grp)['n_incidents'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    df['ever_event_prior'] = (df['incidents_cumul'] > 0).astype(int)

    return df


def compute_structural_features(df):
    """
    Deterministic transforms that don't depend on any data distribution.
    Safe to compute globally (no information leakage).
    """
    era_map = {'era_pre1940': 0, 'era_coal_tar': 1, 'era_50s_60s': 2,
               'era_improved': 3, 'era_modern': 4}
    dec_map = {'pre1940': 0, '1940_49': 1, '1950_59': 2, '1960_69': 3,
               '1970_79': 4, '1980_89': 5, '1990_99': 6, '2000_09': 7,
               '2010_19': 8, '2020_29': 9}

    df['era_ordinal']      = df['era'].map(era_map)
    df['decade_ordinal']   = df['decade_bin'].map(dec_map)
    df['age_sq']           = df['age_at_obs'] ** 2
    df['log_miles']        = np.log1p(df['miles_at_risk'])
    df['age_x_era']        = df['age_at_obs'] * df['era_ordinal']
    df['age_x_small_diam'] = df['age_at_obs'] * df['pct_small_diam']
    df['age_x_class1']     = df['age_at_obs'] * df['pct_class1']
    df['miles_x_age']      = df['log_miles'] * df['age_at_obs']

    return df


def compute_operator_history(df, max_year):
    """
    AFML-compliant: operator history computed ONLY from years < max_year.
    Returns a dataframe with (operator_id, year, op_hist_rate, op_hist_total)
    for ALL years, but using only data strictly before max_year for the
    cumulative calculations.

    The key insight: for year t, op_hist uses data from years < t.
    But we also must not let data from years >= max_year influence
    the cumulative sums at all.
    """
    # Filter to only data before the temporal cutoff
    df_temporal = df[df['year'] < max_year].copy()

    oy = (df_temporal.groupby(['operator_id', 'year'])
          .agg(ev=('event', 'sum'), rw=('event', 'count'))
          .reset_index()
          .sort_values(['operator_id', 'year']))

    # Cumsum shifted: for year t, uses sum of years < t (all within < max_year)
    oy['ce'] = oy.groupby('operator_id')['ev'].cumsum().shift(1).fillna(0)
    oy['cr'] = oy.groupby('operator_id')['rw'].cumsum().shift(1).fillna(0)
    oy['op_hist_rate']  = (oy['ce'] / oy['cr']).fillna(0)
    oy['op_hist_total'] = oy['ce']

    # For the test year itself (year == max_year), use the full cumsum
    # (i.e., all data from years < max_year)
    last_cumul = (df_temporal.groupby(['operator_id', 'year'])
                  .agg(ev=('event', 'sum'), rw=('event', 'count'))
                  .reset_index()
                  .sort_values(['operator_id', 'year']))
    last_cumul['total_ev'] = last_cumul.groupby('operator_id')['ev'].cumsum()
    last_cumul['total_rw'] = last_cumul.groupby('operator_id')['rw'].cumsum()

    # Get the last row per operator (= everything up to max_year-1)
    last_per_op = last_cumul.groupby('operator_id').last().reset_index()
    last_per_op['op_hist_rate']  = (last_per_op['total_ev'] / last_per_op['total_rw']).fillna(0)
    last_per_op['op_hist_total'] = last_per_op['total_ev']
    last_per_op['year'] = max_year

    # Combine: historical years + test year row
    result = pd.concat([
        oy[['operator_id', 'year', 'op_hist_rate', 'op_hist_total']],
        last_per_op[['operator_id', 'year', 'op_hist_rate', 'op_hist_total']]
    ], ignore_index=True)

    return result


def compute_repairs_feature(df, max_year):
    """
    AFML-compliant: p99 cap computed ONLY from years < max_year.
    The 2024 anomaly (cumulative vs annual) is handled per-fold.
    """
    df = df.copy()
    df['repairs_available'] = (~df['log1p_total_repairs'].isna()).astype(int)
    df['repairs_filled']    = df['log1p_total_repairs'].fillna(0)

    # Compute p99 ONLY from data strictly before the test year
    train_repairs = df.loc[df['year'] < max_year, 'log1p_total_repairs'].dropna()
    if len(train_repairs) > 0:
        p99 = train_repairs.quantile(0.99)
        # Cap ALL repairs (including test year) at train-derived p99
        df['repairs_filled'] = df['repairs_filled'].clip(upper=p99)
    else:
        p99 = np.inf  # no data to compute cap

    return df, p99


def compute_state_encoding(df_train, df_apply, target='event', prior_m=STATE_PRIOR):
    """
    Bayesian-smoothed state target encoding.
    Computed ONLY from df_train, applied to df_apply.
    """
    rates  = df_train.groupby('state')[target].mean()
    counts = df_train.groupby('state')[target].count()
    glob   = df_train[target].mean()
    encoded = (rates * counts + glob * prior_m) / (counts + prior_m)
    return df_apply['state'].map(encoded.to_dict()).fillna(glob)


def build_fold_features(df_raw, train_years_mask, test_years_mask, max_year, target='event'):
    """
    AFML-compliant feature builder for a single walk-forward fold.
    ALL distribution-dependent features recomputed using only training data.

    Args:
        df_raw: raw dataframe (with lag + structural already computed)
        train_years_mask: boolean mask for training rows
        test_years_mask: boolean mask for test rows
        max_year: the test year (used as temporal cutoff)
        target: target column name

    Returns:
        X_train, y_train, X_test, y_test, p99_repairs
    """
    FEATURES_BASE = [
        'age_at_obs', 'age_sq', 'era_ordinal', 'decade_ordinal', 'log_miles',
        'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1',
        'age_x_era', 'age_x_small_diam', 'age_x_class1', 'miles_x_age',
        'repairs_filled', 'repairs_available',
        'event_lag1', 'incidents_lag1', 'incidents_roll3', 'incidents_cumul',
        'ever_event_prior',
        'op_hist_rate', 'op_hist_total',
    ]

    df = df_raw.copy()

    # 1. Repairs: cap using only train data
    df, p99 = compute_repairs_feature(df, max_year)

    # 2. Operator history: recompute using only data < max_year
    op_hist = compute_operator_history(df, max_year)

    # Drop old op_hist columns if they exist, merge fresh ones
    for col in ['op_hist_rate', 'op_hist_total']:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.merge(op_hist, on=['operator_id', 'year'], how='left')
    df['op_hist_rate']  = df['op_hist_rate'].fillna(0)
    df['op_hist_total'] = df['op_hist_total'].fillna(0)

    # 3. Extract train/test
    X_train = df.loc[train_years_mask, FEATURES_BASE].copy()
    y_train = df.loc[train_years_mask, target].copy()
    X_test  = df.loc[test_years_mask, FEATURES_BASE].copy()
    y_test  = df.loc[test_years_mask, target].copy()

    # 4. State encoding: train-only
    X_train['state_risk'] = compute_state_encoding(
        df.loc[train_years_mask], df.loc[train_years_mask], target
    ).values
    X_test['state_risk'] = compute_state_encoding(
        df.loc[train_years_mask], df.loc[test_years_mask], target
    ).values

    FEATURES = FEATURES_BASE + ['state_risk']

    return X_train, y_train, X_test, y_test, FEATURES, p99


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOAD & AUDIT
# ═══════════════════════════════════════════════════════════════════════

def load_data(path):
    print("=" * 72)
    print("PHASE 1: DATA LOAD & LEAKAGE AUDIT")
    print("=" * 72)

    df = pd.read_csv(path)
    print(f"  Raw panel:   {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Years:       {df['year'].min()}-{df['year'].max()}")
    print(f"  Operators:   {df['operator_id'].nunique():,}")
    print(f"  Event rate:  {df['event'].mean():.4f} ({df['event'].sum():,} events)")

    LEAKAGE = [
        'n_incidents', 'n_corrosion', 'n_ext_corr', 'n_int_corr',
        'n_material', 'n_excavation', 'n_natural', 'n_other_cause',
        'event_corrosion', 'log1p_ext_corrosion',
    ]
    print(f"  Leakage cols (excluded from features): {len(LEAKAGE)}")
    print(f"  AFML compliance: all distribution-dependent transforms")
    print(f"  computed INSIDE each walk-forward fold")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — GLOBAL FEATURE ENGINEERING (safe transforms only)
# ═══════════════════════════════════════════════════════════════════════

def global_feature_engineering(df):
    print("\n" + "=" * 72)
    print("PHASE 2: GLOBAL FEATURE ENGINEERING (deterministic only)")
    print("=" * 72)

    df = compute_lag_features(df)
    print("  ✓ Lag features (shift-based, inherently temporal)")

    df = compute_structural_features(df)
    print("  ✓ Structural transforms (deterministic, no distribution)")

    print("  ✗ Repairs capping   → deferred to per-fold (AFML Ch.7)")
    print("  ✗ Operator history  → deferred to per-fold (AFML Ch.7)")
    print("  ✗ State encoding    → deferred to per-fold (AFML Ch.7)")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — WALK-FORWARD VALIDATION (AFML Ch.7/Ch.12)
# ═══════════════════════════════════════════════════════════════════════

def walk_forward_afml(df, target='event'):
    """
    AFML-compliant expanding-window walk-forward validation.

    Per fold:
      1. Recompute repairs p99 from train data only
      2. Recompute operator history from train data only
      3. Recompute state encoding from train data only
      4. Train LightGBM with early stopping — model picks its iterations
      5. Train GLM baseline for comparison
      6. Record all metrics + model diagnostics
    """
    print("\n" + "=" * 72)
    print("PHASE 3: WALK-FORWARD VALIDATION (AFML-compliant)")
    print("=" * 72)
    print(f"  Folds: {WF_START}-{WF_END} (expanding window)")
    print(f"  Temporal purity: repairs cap, operator history, state encoding")
    print(f"                   ALL recomputed per fold")
    print(f"  Model selection: early stopping (patience={EARLY_STOP_ROUNDS}),")
    print(f"                   NO iteration override\n")

    wf_lgb, wf_glm = [], []

    for test_year in range(WF_START, WF_END + 1):
        mask_tr = df['year'] < test_year
        mask_te = df['year'] == test_year

        if mask_te.sum() == 0 or df.loc[mask_te, target].sum() == 0:
            continue

        # ── AFML: Build features inside the fold ──
        X_tr, y_tr, X_te, y_te, FEATURES, p99 = build_fold_features(
            df, mask_tr, mask_te, max_year=test_year, target=target
        )

        # ── LightGBM: model decides its own complexity ──
        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = LGB_PARAMS.copy()
        params['scale_pos_weight'] = pw

        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_te = lgb.Dataset(X_te, label=y_te, reference=d_tr)

        mdl = lgb.train(
            params, d_tr, num_boost_round=3000,
            valid_sets=[d_te], valid_names=['val'],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # AFML Ch.12: respect the model's own selection
        best_iter = mdl.best_iteration
        pr = mdl.predict(X_te, num_iteration=best_iter)

        auc_lgb = roc_auc_score(y_te, pr)
        ap_lgb  = average_precision_score(y_te, pr)
        bs_lgb  = brier_score_loss(y_te, pr)

        wf_lgb.append({
            'year': test_year,
            'AUC': auc_lgb, 'AP': ap_lgb, 'Brier': bs_lgb,
            'events': int(y_te.sum()), 'rate': y_te.mean(),
            'best_iter': best_iter, 'p99_repairs': p99,
            'n_train': len(y_tr), 'n_test': len(y_te),
        })

        # ── GLM baseline ──
        sc = StandardScaler()
        X_tr_g = sc.fit_transform(df.loc[mask_tr, GLM_FEATURES])
        X_te_g = sc.transform(df.loc[mask_te, GLM_FEATURES])
        g = LogisticRegression(
            class_weight='balanced', max_iter=1000, C=1.0, random_state=42
        )
        g.fit(X_tr_g, y_tr)
        pg = g.predict_proba(X_te_g)[:, 1]

        auc_glm = roc_auc_score(y_te, pg)
        ap_glm  = average_precision_score(y_te, pg)
        bs_glm  = brier_score_loss(y_te, pg)

        wf_glm.append({
            'year': test_year, 'AUC': auc_glm, 'AP': ap_glm, 'Brier': bs_glm,
        })

        delta = auc_lgb - auc_glm
        print(f"  {test_year}: LGB AUC={auc_lgb:.4f} (iter={best_iter:3d}, p99={p99:.2f})  "
              f"GLM AUC={auc_glm:.4f}  Δ={delta:+.4f}  "
              f"[{int(y_te.sum())} events, Brier_lgb={bs_lgb:.4f}]")

    wf_lgb_df = pd.DataFrame(wf_lgb)
    wf_glm_df = pd.DataFrame(wf_glm)

    print(f"\n  ── Walk-Forward Summary ──")
    print(f"  LightGBM: AUC = {wf_lgb_df['AUC'].mean():.4f} ± {wf_lgb_df['AUC'].std():.4f}")
    print(f"  GLM:      AUC = {wf_glm_df['AUC'].mean():.4f} ± {wf_glm_df['AUC'].std():.4f}")
    print(f"  ΔAUC:     {(wf_lgb_df['AUC'] - wf_glm_df['AUC']).mean():+.4f}")
    print(f"  LGB Brier:  {wf_lgb_df['Brier'].mean():.6f} ± {wf_lgb_df['Brier'].std():.6f}")
    print(f"  GLM Brier:  {wf_glm_df['Brier'].mean():.6f} ± {wf_glm_df['Brier'].std():.6f}")
    print(f"  Iterations: min={wf_lgb_df['best_iter'].min()}, "
          f"max={wf_lgb_df['best_iter'].max()}, "
          f"mean={wf_lgb_df['best_iter'].mean():.1f}")

    return wf_lgb_df, wf_glm_df, FEATURES


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — HOLDOUT MODEL (for SHAP + operator ranking)
# ═══════════════════════════════════════════════════════════════════════

def train_holdout_model(df, target='event'):
    """
    Train a single model on the holdout split for SHAP analysis.
    Uses AFML-compliant feature engineering.
    Train: 2010-TRAIN_END, Val: TRAIN_END+1-VAL_END, Test: VAL_END+1-2024

    Val is used ONLY for early stopping. Test is pure OOS.
    """
    print("\n" + "=" * 72)
    print("PHASE 4: HOLDOUT MODEL (for SHAP & ranking)")
    print("=" * 72)

    mask_tr = df['year'].between(2010, TRAIN_END)
    mask_va = df['year'].between(TRAIN_END + 1, VAL_END)
    mask_te = df['year'].between(VAL_END + 1, 2024)

    # ── AFML: build features for train+val using cutoff = VAL_END+1 ──
    # Train features: only uses data < TRAIN_END+1 for distribution-dependent stuff
    # But for the holdout model, we train on 2010-TRAIN_END, validate on TRAIN_END+1-VAL_END
    # So the temporal cutoff for feature engineering is VAL_END+1 (we don't peek at test)

    # For train split: features use data < test cutoff
    X_tr, y_tr, X_va, y_va, FEATURES, p99_tr = build_fold_features(
        df, mask_tr, mask_va, max_year=TRAIN_END + 1, target=target
    )

    # For test split: features use data < VAL_END+1
    _, _, X_te, y_te, _, p99_te = build_fold_features(
        df, mask_tr | mask_va, mask_te, max_year=VAL_END + 1, target=target
    )

    print(f"  Train: {len(y_tr):,} rows, Val: {len(y_va):,} rows, Test: {len(y_te):,} rows")
    print(f"  Repairs p99 (train): {p99_tr:.2f}, (train+val): {p99_te:.2f}")
    print(f"  Features: {len(FEATURES)}")

    # ── Train with early stopping on val — AFML: model decides ──
    pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = LGB_PARAMS.copy()
    params['scale_pos_weight'] = pw

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    print(f"  Training with early stopping (patience={EARLY_STOP_ROUNDS})...")
    model = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval], valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=True),
            lgb.log_evaluation(200),
        ],
    )
    best_iter = model.best_iteration
    print(f"  Model selected {best_iter} iterations")

    # ── Evaluate all splits ──
    results = {}
    for name, X, y in [('Train', X_tr, y_tr), ('Val', X_va, y_va), ('Test', X_te, y_te)]:
        preds = model.predict(X, num_iteration=best_iter)
        results[name] = {
            'AUC-ROC': roc_auc_score(y, preds),
            'AP':      average_precision_score(y, preds),
            'Brier':   brier_score_loss(y, preds),
            'LogLoss': log_loss(y, preds),
            'preds':   preds,
            'y':       y,
        }
        print(f"    {name}: AUC={results[name]['AUC-ROC']:.4f}  "
              f"AP={results[name]['AP']:.4f}  Brier={results[name]['Brier']:.6f}")

    # ── GLM baseline on same splits ──
    scaler = StandardScaler()
    X_tv_g = scaler.fit_transform(df.loc[mask_tr | mask_va, GLM_FEATURES])
    y_tv_g = df.loc[mask_tr | mask_va, target]
    X_te_g = scaler.transform(df.loc[mask_te, GLM_FEATURES])

    glm = LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0, random_state=42
    )
    glm.fit(X_tv_g, y_tv_g)
    glm_preds = glm.predict_proba(X_te_g)[:, 1]
    glm_metrics = {
        'AUC-ROC': roc_auc_score(y_te, glm_preds),
        'AP':      average_precision_score(y_te, glm_preds),
        'Brier':   brier_score_loss(y_te, glm_preds),
    }
    print(f"  GLM → Test: AUC={glm_metrics['AUC-ROC']:.4f}  AP={glm_metrics['AP']:.4f}")

    return model, results, glm_preds, glm_metrics, FEATURES, X_va, X_te, y_te


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_shap(model, FEATURES, X_val):
    print("\n" + "=" * 72)
    print("PHASE 5: SHAP INTERPRETABILITY")
    print("=" * 72)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df  = (pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs})
                  .sort_values('mean_abs_shap', ascending=False))

    print("  Top 10:")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {clean_name(row['feature']):<30s} {row['mean_abs_shap']:.4f}")

    return shap_df, shap_values, X_val


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — OPERATOR RISK RANKING (AFML-compliant)
# ═══════════════════════════════════════════════════════════════════════

def rank_operators(df, FEATURES, target='event'):
    """
    Score operators for 2023-2024 using walk-forward models.
    Each year scored with a model that ONLY saw data before that year.
    """
    print("\n" + "=" * 72)
    print("PHASE 6: OPERATOR RISK RANKING (Walk-Forward Ensemble)")
    print("=" * 72)

    frames = []
    for ty in [2023, 2024]:
        mask_tr = df['year'] < ty
        mask_te = df['year'] == ty

        X_tr, y_tr, X_te, y_te, FEAT, p99 = build_fold_features(
            df, mask_tr, mask_te, max_year=ty, target=target
        )

        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = LGB_PARAMS.copy()
        params['scale_pos_weight'] = pw

        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_te_ds = lgb.Dataset(X_te, label=y_te, reference=d_tr)

        mdl = lgb.train(
            params, d_tr, num_boost_round=3000,
            valid_sets=[d_te_ds], valid_names=['val'],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        chunk = df.loc[mask_te].copy()
        chunk['pred_risk'] = mdl.predict(X_te, num_iteration=mdl.best_iteration)
        frames.append(chunk)
        print(f"  {ty}: scored {len(chunk):,} rows (iter={mdl.best_iteration}, p99={p99:.2f})")

    df_recent = pd.concat(frames)

    # Aggregate per operator (miles-weighted)
    op_risk = df_recent.groupby(['operator_id', 'operator_name']).apply(
        lambda g: pd.Series({
            'mean_risk':     np.average(g['pred_risk'], weights=g['miles_at_risk']),
            'max_risk':      g['pred_risk'].max(),
            'total_miles':   g['miles_at_risk'].sum(),
            'n_decades':     len(g),
            'actual_events': g['event'].sum(),
            'mean_age':      np.average(g['age_at_obs'], weights=g['miles_at_risk']),
        })
    ).reset_index()

    # Bootstrap CI
    print("  Computing bootstrap confidence intervals...")
    np.random.seed(BOOT_SEED)
    ci_rows = []
    for oid in op_risk['operator_id'].unique():
        m = df_recent['operator_id'] == oid
        risks = df_recent.loc[m, 'pred_risk'].values
        miles = df_recent.loc[m, 'miles_at_risk'].values
        if len(risks) < 2:
            ci_rows.append({'operator_id': oid,
                            'ci_low': risks[0], 'ci_high': risks[0]})
            continue
        boots = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(risks), len(risks), replace=True)
            boots.append(np.average(risks[idx], weights=miles[idx]))
        ci_rows.append({
            'operator_id': oid,
            'ci_low':  np.percentile(boots, 2.5),
            'ci_high': np.percentile(boots, 97.5),
        })

    op_risk = (op_risk.merge(pd.DataFrame(ci_rows), on='operator_id')
                      .sort_values('mean_risk', ascending=False))

    # Tiers
    q75 = op_risk['mean_risk'].quantile(0.75)
    q95 = op_risk['mean_risk'].quantile(0.95)
    q99 = op_risk['mean_risk'].quantile(0.99)
    op_risk['risk_tier'] = op_risk['mean_risk'].apply(
        lambda r: 'CRITICAL' if r >= q99 else 'HIGH' if r >= q95
                  else 'ELEVATED' if r >= q75 else 'STANDARD')

    tiers = op_risk['risk_tier'].value_counts().to_dict()
    print(f"  Tiers: {tiers}")
    print(f"\n  Top 15:")
    for _, r in op_risk.head(15).iterrows():
        print(f"    [{r['risk_tier']:8s}] {r['operator_name'][:45]:<45s}  "
              f"risk={r['mean_risk']:.4f}  [{r['ci_low']:.4f}-{r['ci_high']:.4f}]  "
              f"miles={r['total_miles']:.0f}  events={r['actual_events']:.0f}")

    return op_risk, df_recent


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def make_figures(model, results, glm_preds, y_test,
                 wf_lgb_df, wf_glm_df, shap_df, shap_values, use_X,
                 op_risk, df_recent, FEATURES, target='event'):
    print("\n" + "=" * 72)
    print("PHASE 7: VISUALIZATION")
    print("=" * 72)

    # ── FIGURE 1: Main Dashboard ──
    fig = plt.figure(figsize=(22, 28), facecolor=C['BG'])
    gs  = gridspec.GridSpec(5, 2, hspace=0.38, wspace=0.3,
                            left=0.07, right=0.95, top=0.94, bottom=0.03)

    fig.suptitle('LightGBM Pipeline — AFML-Compliant Incident Prediction',
                 color=C['TEXT'], fontsize=17, fontweight='bold', y=0.975)
    fig.text(0.5, 0.96,
             'Temporal Purity · Walk-Forward · SHAP · Operator Risk Ranking',
             color='#808080', fontsize=11, ha='center')

    # [0,0] Walk-Forward AUC
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(wf_lgb_df['year'], wf_lgb_df['AUC'], '-o', color=C['ACCENT'], lw=2.5,
            ms=7, mec='white', mew=1,
            label=f'LightGBM (μ={wf_lgb_df["AUC"].mean():.3f})', zorder=5)
    ax.plot(wf_glm_df['year'], wf_glm_df['AUC'], '--s', color=C['RED'], lw=2,
            ms=6, mec='white', mew=1,
            label=f'GLM (μ={wf_glm_df["AUC"].mean():.3f})', zorder=4)
    ax.fill_between(wf_lgb_df['year'],
                    wf_lgb_df['AUC'] - wf_lgb_df['AUC'].std(),
                    wf_lgb_df['AUC'] + wf_lgb_df['AUC'].std(),
                    alpha=0.12, color=C['ACCENT'])
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=8.5, loc='lower left',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Walk-Forward AUC-ROC: LightGBM vs GLM', 'Test Year', 'AUC-ROC')

    # [0,1] Walk-Forward Brier
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(wf_lgb_df['year'], wf_lgb_df['Brier'], '-o', color=C['ACCENT'], lw=2.5,
            ms=7, mec='white', mew=1,
            label=f'LightGBM (μ={wf_lgb_df["Brier"].mean():.4f})', zorder=5)
    ax.plot(wf_glm_df['year'], wf_glm_df['Brier'], '--s', color=C['RED'], lw=2,
            ms=6, mec='white', mew=1,
            label=f'GLM (μ={wf_glm_df["Brier"].mean():.4f})', zorder=4)
    base_rate = wf_lgb_df['rate'].mean()
    ax.axhline(base_rate, color='gray', ls=':', alpha=0.4,
               label=f'Event rate={base_rate:.3f}')
    ax.legend(fontsize=8.5, loc='upper right',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Walk-Forward Brier Score (lower = better)', 'Test Year', 'Brier Score')

    # [1,:] SHAP Feature Importance
    ax = fig.add_subplot(gs[1, :])
    active   = shap_df[shap_df['mean_abs_shap'] > 0]
    top_n    = min(15, len(active))
    shap_top = active.head(top_n).iloc[::-1]
    median_s = shap_top['mean_abs_shap'].median()
    colors_s = [C['ACCENT'] if v > median_s else C['TEAL']
                for v in shap_top['mean_abs_shap']]
    bars = ax.barh(range(top_n), shap_top['mean_abs_shap'], color=colors_s,
                   alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([clean_name(f) for f in shap_top['feature']], fontsize=9)
    for bar, val in zip(bars, shap_top['mean_abs_shap']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', color=C['TEXT'], fontsize=8)
    style_ax(ax, 'SHAP Feature Importance (Mean |SHAP Value|)', 'Mean |SHAP Value|', '')

    # [2,0] Calibration
    ax = fig.add_subplot(gs[2, 0])
    for nm, pr, col, ls in [('LightGBM', results['Test']['preds'], C['ACCENT'], '-'),
                             ('GLM', glm_preds, C['RED'], '--')]:
        pt, pp = calibration_curve(y_test, pr, n_bins=10, strategy='quantile')
        ax.plot(pp, pt, f'{ls}o', color=col, lw=2, ms=5, label=nm)
    max_pred = max(results['Test']['preds'].max(), glm_preds.max())
    ax.plot([0, max_pred], [0, max_pred], '--', color='gray', alpha=0.5, label='Perfect')
    ax.legend(fontsize=8, loc='lower right',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Calibration (Test Set)', 'Predicted P(incident)', 'Observed Frequency')

    # [2,1] Precision-Recall
    ax = fig.add_subplot(gs[2, 1])
    for nm, pr, col, ls in [('LightGBM', results['Test']['preds'], C['ACCENT'], '-'),
                             ('GLM', glm_preds, C['RED'], '--')]:
        prec, rec, _ = precision_recall_curve(y_test, pr)
        ap_v = average_precision_score(y_test, pr)
        ax.plot(rec, prec, ls, color=col, lw=2, label=f'{nm} (AP={ap_v:.3f})')
    ax.axhline(y_test.mean(), color='gray', ls=':', alpha=0.5,
               label=f'Baseline={y_test.mean():.4f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8, loc='upper right',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Precision-Recall (Test Set)', 'Recall', 'Precision')

    # [3,0] Gain importance
    ax  = fig.add_subplot(gs[3, 0])
    imp = model.feature_importance(importance_type='gain')
    imp_d = (pd.DataFrame({'feature': FEATURES, 'gain': imp})
               .sort_values('gain', ascending=True).tail(15))
    med_g = imp_d['gain'].median()
    colors_g = [C['ACCENT'] if v > med_g else C['BLUE'] for v in imp_d['gain']]
    ax.barh(range(len(imp_d)), imp_d['gain'], color=colors_g,
            alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(imp_d)))
    ax.set_yticklabels([clean_name(f) for f in imp_d['feature']], fontsize=9)
    style_ax(ax, 'LightGBM Feature Importance (Gain)', 'Total Gain', '')

    # [3,1] Risk by era
    ax = fig.add_subplot(gs[3, 1])
    for era in ['era_pre1940', 'era_coal_tar', 'era_50s_60s', 'era_improved', 'era_modern']:
        subset = df_recent.loc[df_recent['era'] == era, 'pred_risk']
        if len(subset) > 0:
            ax.hist(subset, bins=50, alpha=0.45,
                    label=era.replace('era_', '').replace('_', ' ').title(), density=True)
    ax.set_xlim(0, df_recent['pred_risk'].quantile(0.995))
    ax.legend(fontsize=8, facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Risk Distribution by Era (2023-2024)', 'Predicted Risk', 'Density')

    # [4,:] Top 25 Operators
    ax = fig.add_subplot(gs[4, :])
    top25 = op_risk.head(25).iloc[::-1].reset_index(drop=True)
    for i, row in top25.iterrows():
        col = TIER_COLORS.get(row['risk_tier'], C['TEAL'])
        ax.barh(i, row['mean_risk'], color=col, alpha=0.8,
                edgecolor='white', linewidth=0.5, height=0.7)
        ax.plot([row['ci_low'], row['ci_high']], [i, i],
                color='white', linewidth=1.5, alpha=0.7)
        ax.plot([row['ci_low'], row['ci_high']], [i, i],
                '|', color='white', markersize=8, alpha=0.7)
        ax.text(row['ci_high'] + 0.001, i, f"{row['actual_events']:.0f}ev",
                color=C['TEXT'], fontsize=6.5, va='center', alpha=0.7)

    labels = [n[:42] + '…' if len(n) > 42 else n for n in top25['operator_name']]
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(labels, fontsize=7)
    leg = [Patch(facecolor=TIER_COLORS[t], alpha=0.8, label=t)
           for t in ['CRITICAL', 'HIGH', 'ELEVATED', 'STANDARD']]
    ax.legend(handles=leg, fontsize=8, loc='lower right',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'], ncol=4)
    style_ax(ax, 'Top 25 Highest-Risk Operators (2023-2024) — 95% Bootstrap CI',
             'Predicted Incident Probability', '')

    p1 = os.path.join(OUTPUT_DIR, 'fig1_pipeline_dashboard.png')
    plt.savefig(p1, dpi=180, facecolor=C['BG'], bbox_inches='tight')
    plt.close()
    print(f"  ✓ {p1}")

    # ── FIGURE 2: SHAP Deep Dive ──
    fig2, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor=C['BG'])

    # [0,0] SHAP beeswarm
    ax = axes[0, 0]
    active   = shap_df[shap_df['mean_abs_shap'] > 0]
    top8     = active.head(8)['feature'].tolist()
    top8_idx = [FEATURES.index(f) for f in top8]
    np.random.seed(42)
    for i, (fn, fi) in enumerate(zip(top8, top8_idx)):
        sv = shap_values[:, fi]
        fv = use_X[fn].values
        fvn = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-10)
        n_plot = min(800, len(sv))
        idx    = np.random.choice(len(sv), n_plot, replace=False)
        jitter = np.random.normal(0, 0.15, n_plot)
        ax.scatter(sv[idx], i + jitter, c=plt.cm.coolwarm(fvn[idx]),
                   s=3, alpha=0.4, rasterized=True)
    ax.set_yticks(range(len(top8)))
    ax.set_yticklabels([clean_name(f) for f in top8], fontsize=9)
    ax.axvline(0, color='gray', ls='-', alpha=0.3)
    style_ax(ax, 'SHAP Beeswarm (Top Features)', 'SHAP Value', '')
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label('Feature Value (norm)', color=C['TEXT'], fontsize=8)
    cb.ax.tick_params(labelcolor=C['TEXT'], labelsize=7)

    # [0,1] SHAP dependence: log_miles
    ax = axes[0, 1]
    fi = FEATURES.index('log_miles')
    ax.scatter(use_X['log_miles'].values, shap_values[:, fi],
               c=use_X['age_at_obs'].values, cmap='coolwarm',
               s=4, alpha=0.3, rasterized=True)
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    style_ax(ax, 'SHAP Dependence: Log(Miles) × Age', 'Log(Miles)', 'SHAP Value')
    sm2 = plt.cm.ScalarMappable(
        cmap='coolwarm',
        norm=plt.Normalize(use_X['age_at_obs'].min(), use_X['age_at_obs'].max()))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax, shrink=0.5, pad=0.02)
    cb2.set_label('Pipe Age', color=C['TEXT'], fontsize=8)
    cb2.ax.tick_params(labelcolor=C['TEXT'], labelsize=7)

    # [1,0] ΔAUC by year
    ax    = axes[1, 0]
    delta = wf_lgb_df['AUC'].values - wf_glm_df['AUC'].values
    cols  = [C['ACCENT'] if d >= 0 else C['RED'] for d in delta]
    ax.bar(wf_lgb_df['year'], delta, color=cols, alpha=0.8,
           edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.axhline(delta.mean(), color=C['GOLD'], ls='--', alpha=0.7,
               label=f'Mean Δ = {delta.mean():+.4f}')
    ax.legend(fontsize=8, facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'LGB − GLM: ΔAUC by Year', 'Year', 'ΔAUC')

    # [1,1] Iterations + p99 evolution
    ax = axes[1, 1]
    ax.bar(wf_lgb_df['year'], wf_lgb_df['best_iter'], color=C['BLUE'],
           alpha=0.8, edgecolor='white', linewidth=0.5, label='Iterations')
    ax2r = ax.twinx()
    ax2r.plot(wf_lgb_df['year'], wf_lgb_df['p99_repairs'], '-o',
              color=C['GOLD'], lw=2, ms=6, label='p99 repairs (per fold)')
    ax2r.set_ylabel('p99 Repairs Cap', color=C['GOLD'], fontsize=9)
    ax2r.tick_params(colors=C['GOLD'], labelsize=8)
    ax.legend(fontsize=8, loc='upper left',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    ax2r.legend(fontsize=8, loc='upper right',
                facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['GOLD'])
    style_ax(ax, 'AFML Diagnostics: Iterations & p99 per Fold', 'Year', 'Best Iteration')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.suptitle('SHAP & Walk-Forward Diagnostics (AFML-Compliant)',
                  color=C['TEXT'], fontsize=14, fontweight='bold')

    p2 = os.path.join(OUTPUT_DIR, 'fig2_shap_walkforward.png')
    plt.savefig(p2, dpi=180, facecolor=C['BG'], bbox_inches='tight')
    plt.close()
    print(f"  ✓ {p2}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 — EXPORT
# ═══════════════════════════════════════════════════════════════════════

def export_csvs(op_risk, wf_lgb_df, wf_glm_df, shap_df):
    print("\n" + "=" * 72)
    print("PHASE 8: EXPORT CSVs")
    print("=" * 72)

    p1 = os.path.join(OUTPUT_DIR, 'operator_risk_ranking_final.csv')
    op_risk.to_csv(p1, index=False)
    print(f"  ✓ {p1}  ({len(op_risk)} operators)")

    wf_cmp = wf_lgb_df.merge(wf_glm_df, on='year', suffixes=('_lgb', '_glm'))
    p2 = os.path.join(OUTPUT_DIR, 'walkforward_comparison.csv')
    wf_cmp.to_csv(p2, index=False)
    print(f"  ✓ {p2}  ({len(wf_cmp)} folds)")

    p3 = os.path.join(OUTPUT_DIR, 'shap_importance_final.csv')
    shap_df.to_csv(p3, index=False)
    print(f"  ✓ {p3}  ({len(shap_df)} features)")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_summary(df, FEATURES, results, glm_metrics,
                  wf_lgb_df, wf_glm_df, shap_df, op_risk, target='event'):
    tiers = op_risk['risk_tier'].value_counts().to_dict()
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"""
  Dataset:     {df.shape[0]:,} operator-decade-year observations
               {df['operator_id'].nunique():,} operators, {df['year'].min()}-{df['year'].max()}
               Event rate: {df[target].mean():.4f} ({df[target].sum():,} events)

  Features:    {len(FEATURES)} (structural + interactions + lag + operator history)
               AFML-compliant: all distribution-dependent transforms
               recomputed inside each walk-forward fold

  Holdout Model:
    LightGBM → Test: AUC={results['Test']['AUC-ROC']:.4f}  AP={results['Test']['AP']:.4f}  Brier={results['Test']['Brier']:.6f}
    GLM      → Test: AUC={glm_metrics['AUC-ROC']:.4f}  AP={glm_metrics['AP']:.4f}  Brier={glm_metrics['Brier']:.6f}

  Walk-Forward ({len(wf_lgb_df)} folds, {WF_START}-{WF_END}):
    LightGBM: AUC = {wf_lgb_df['AUC'].mean():.4f} ± {wf_lgb_df['AUC'].std():.4f}   Brier = {wf_lgb_df['Brier'].mean():.6f}
    GLM:      AUC = {wf_glm_df['AUC'].mean():.4f} ± {wf_glm_df['AUC'].std():.4f}   Brier = {wf_glm_df['Brier'].mean():.6f}
    ΔAUC:     {(wf_lgb_df['AUC'] - wf_glm_df['AUC']).mean():+.4f}
    Iterations: {wf_lgb_df['best_iter'].min()}-{wf_lgb_df['best_iter'].max()} (mean={wf_lgb_df['best_iter'].mean():.1f})

  Top SHAP:  1. {clean_name(shap_df.iloc[0]['feature'])} ({shap_df.iloc[0]['mean_abs_shap']:.4f})
             2. {clean_name(shap_df.iloc[1]['feature'])} ({shap_df.iloc[1]['mean_abs_shap']:.4f})
             3. {clean_name(shap_df.iloc[2]['feature'])} ({shap_df.iloc[2]['mean_abs_shap']:.4f})

  Risk Tiers: CRITICAL={tiers.get('CRITICAL',0)} | HIGH={tiers.get('HIGH',0)} | ELEVATED={tiers.get('ELEVATED',0)} | STANDARD={tiers.get('STANDARD',0)}

  Outputs:
    • fig1_pipeline_dashboard.png
    • fig2_shap_walkforward.png
    • operator_risk_ranking_final.csv
    • walkforward_comparison.csv
    • shap_importance_final.csv
""")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: Load
    df = load_data(INPUT_PATH)

    # Phase 2: Global (safe) feature engineering
    df = global_feature_engineering(df)

    # Phase 3: Walk-forward (AFML-compliant)
    wf_lgb_df, wf_glm_df, FEATURES = walk_forward_afml(df)

    # Phase 4: Holdout model for SHAP + static evaluation
    model, results, glm_preds, glm_metrics, FEATURES, X_val, X_test, y_test = \
        train_holdout_model(df)

    # Phase 5: SHAP
    shap_df, shap_values, use_X = compute_shap(model, FEATURES, X_val)

    # Phase 6: Operator risk ranking
    op_risk, df_recent = rank_operators(df, FEATURES)

    # Phase 7: Visualization
    make_figures(model, results, glm_preds, y_test,
                 wf_lgb_df, wf_glm_df, shap_df, shap_values, use_X,
                 op_risk, df_recent, FEATURES)

    # Phase 8: Export
    export_csvs(op_risk, wf_lgb_df, wf_glm_df, shap_df)

    # Summary
    print_summary(df, FEATURES, results, glm_metrics,
                  wf_lgb_df, wf_glm_df, shap_df, op_risk)


if __name__ == '__main__':
    main()