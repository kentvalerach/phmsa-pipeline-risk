"""
=============================================================================
LightGBM Pipeline — Gas Transmission Incident Prediction
=============================================================================
Standalone script: loads survival_panel_15yr_final.csv and generates:

  OUTPUTS:
    1. fig1_pipeline_dashboard.png  — Main 10-panel model dashboard
    2. fig2_shap_walkforward.png    — SHAP deep-dive & walk-forward diagnostics
    3. operator_risk_ranking_final.csv
    4. walkforward_comparison.csv   — Year-by-year LGB vs GLM
    5. shap_importance_final.csv

  PIPELINE:
    Phase 1 — Data load & leakage audit
    Phase 2 — Feature engineering (lag, operator history, interactions)
    Phase 3 — Feature matrix & temporal splits
    Phase 4 — LightGBM training (2-phase: CV + retrain)
    Phase 5 — Evaluation on all splits
    Phase 6 — GLM baseline comparison
    Phase 7 — Walk-forward expanding-window validation (LGB vs GLM)
    Phase 8 — SHAP interpretability
    Phase 9 — Operator risk ranking with bootstrap CI
    Phase 10 — Visualization (2 figures)

  USAGE:
    pip install lightgbm shap pandas numpy scikit-learn matplotlib
    python lgbm_full_pipeline.py

  NOTES:
    - Input CSV expected in same directory (or edit INPUT_PATH)
    - All outputs saved to OUTPUT_DIR
    - DATA LEAKAGE AUDIT: n_incidents and sub-cause columns are same-year
      outcomes → excluded from features. Only lagged/prior-year data used.
=============================================================================
"""

import os
import sys
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
OUTPUT_DIR = '.'           # change to a folder path if desired

# Temporal split boundaries
TRAIN_END    = 2019        # Train: 2010-2019
VAL_END      = 2022        # Val:   2020-2022
                           # Test:  2023-2024

# Walk-forward start year
WF_START     = 2015

# Bayesian smoothing prior weight for state encoding
STATE_PRIOR  = 50

# Bootstrap config for operator CI
N_BOOTSTRAP  = 500
BOOT_SEED    = 42

# LightGBM hyperparameters
LGB_PARAMS = {
    'objective':        'binary',
    'metric':           ['binary_logloss', 'auc'],
    'boosting_type':    'gbdt',
    'num_leaves':       31,
    'max_depth':        5,
    'learning_rate':    0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'min_child_samples': 100,
    'lambda_l1':        0.5,
    'lambda_l2':        5.0,
    'min_gain_to_split': 0.1,
    'verbose':          -1,
    'seed':             42,
    'n_jobs':           -1,
}

# GLM features (structural only — what NB2-GLM would use)
GLM_FEATURES = [
    'age_at_obs', 'era_ordinal', 'log_miles',
    'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1',
]

# Human-readable feature names for plots
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
    """Apply dark-theme styling to a matplotlib axes."""
    ax.set_facecolor(C['PANEL'])
    ax.set_title(title, color=C['TEXT'], fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, color=C['TEXT'], fontsize=9)
    ax.set_ylabel(ylabel, color=C['TEXT'], fontsize=9)
    ax.tick_params(colors=C['TEXT'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C['GRID'])
    ax.grid(True, alpha=0.15, color=C['GRID'])


def clean_name(feat):
    """Map raw feature name to human-readable label."""
    return FEAT_NAMES.get(feat, feat)


def state_target_encode(df_train, df_apply, target, prior_m=STATE_PRIOR):
    """Bayesian-smoothed state target encoding (train → apply)."""
    rates  = df_train.groupby('state')[target].mean()
    counts = df_train.groupby('state')[target].count()
    glob   = df_train[target].mean()
    encoded = (rates * counts + glob * prior_m) / (counts + prior_m)
    return df_apply['state'].map(encoded.to_dict()).fillna(glob)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOAD & LEAKAGE AUDIT
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
    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    print("\n" + "=" * 72)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 72)

    df = df.sort_values(['operator_id', 'decade_bin', 'year']).reset_index(drop=True)
    grp = ['operator_id', 'decade_bin']

    # -- Lag features (strict temporal: only prior-year data) --
    df['event_lag1']      = df.groupby(grp)['event'].shift(1).fillna(0)
    df['incidents_lag1']  = df.groupby(grp)['n_incidents'].shift(1).fillna(0)
    df['incidents_roll3'] = df.groupby(grp)['n_incidents'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0)
    df['incidents_cumul'] = df.groupby(grp)['n_incidents'].transform(
        lambda x: x.shift(1).cumsum()
    ).fillna(0)
    df['ever_event_prior'] = (df['incidents_cumul'] > 0).astype(int)
    print("  ✓ Lag features (event_lag1, incidents_lag1/roll3/cumul, ever_event_prior)")

    # -- Operator-level historical features (only prior years) --
    oy = (df.groupby(['operator_id', 'year'])
            .agg(ev=('event', 'sum'), rw=('event', 'count'))
            .reset_index()
            .sort_values(['operator_id', 'year']))
    oy['ce'] = oy.groupby('operator_id')['ev'].cumsum().shift(1).fillna(0)
    oy['cr'] = oy.groupby('operator_id')['rw'].cumsum().shift(1).fillna(0)
    oy['op_hist_rate']  = (oy['ce'] / oy['cr']).fillna(0)
    oy['op_hist_total'] = oy['ce']
    df = df.merge(oy[['operator_id', 'year', 'op_hist_rate', 'op_hist_total']],
                  on=['operator_id', 'year'], how='left')
    print("  ✓ Operator history (op_hist_rate, op_hist_total)")

    # -- Structural / engineering features --
    era_map = {'era_pre1940': 0, 'era_coal_tar': 1, 'era_50s_60s': 2,
               'era_improved': 3, 'era_modern': 4}
    dec_map = {'pre1940': 0, '1940_49': 1, '1950_59': 2, '1960_69': 3,
               '1970_79': 4, '1980_89': 5, '1990_99': 6, '2000_09': 7,
               '2010_19': 8, '2020_29': 9}

    df['era_ordinal']     = df['era'].map(era_map)
    df['decade_ordinal']  = df['decade_bin'].map(dec_map)
    df['age_sq']          = df['age_at_obs'] ** 2
    df['log_miles']       = np.log1p(df['miles_at_risk'])
    df['age_x_era']       = df['age_at_obs'] * df['era_ordinal']
    df['age_x_small_diam'] = df['age_at_obs'] * df['pct_small_diam']
    df['age_x_class1']    = df['age_at_obs'] * df['pct_class1']
    df['miles_x_age']     = df['log_miles'] * df['age_at_obs']
    print("  ✓ Interactions & transforms (age², log_miles, age×era, age×diam, ...)")

    # -- Repairs: handle missing + 2024 anomaly --
    df['repairs_available'] = (~df['log1p_total_repairs'].isna()).astype(int)
    df['repairs_filled']    = df['log1p_total_repairs'].fillna(0)
    p99 = df.loc[df['year'] < 2024, 'log1p_total_repairs'].dropna().quantile(0.99)
    df.loc[df['year'] == 2024, 'repairs_filled'] = (
        df.loc[df['year'] == 2024, 'repairs_filled'].clip(upper=p99)
    )
    print(f"  ✓ Repairs cleaned (NaN→0, 2024 capped at p99={p99:.2f})")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — FEATURE MATRIX & TEMPORAL SPLITS
# ═══════════════════════════════════════════════════════════════════════

def build_splits(df):
    print("\n" + "=" * 72)
    print("PHASE 3: FEATURE MATRIX & TEMPORAL SPLITS")
    print("=" * 72)

    FEATURES = [
        'age_at_obs', 'age_sq', 'era_ordinal', 'decade_ordinal', 'log_miles',
        'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1',
        'age_x_era', 'age_x_small_diam', 'age_x_class1', 'miles_x_age',
        'repairs_filled', 'repairs_available',
        'event_lag1', 'incidents_lag1', 'incidents_roll3', 'incidents_cumul',
        'ever_event_prior',
        'op_hist_rate', 'op_hist_total',
    ]
    TARGET = 'event'

    tr = df['year'].between(2010, TRAIN_END)
    va = df['year'].between(TRAIN_END + 1, VAL_END)
    te = df['year'].between(VAL_END + 1, 2024)

    X_train = df.loc[tr, FEATURES].copy()
    y_train = df.loc[tr, TARGET].copy()
    X_val   = df.loc[va, FEATURES].copy()
    y_val   = df.loc[va, TARGET].copy()
    X_test  = df.loc[te, FEATURES].copy()
    y_test  = df.loc[te, TARGET].copy()

    # State target encoding (train-only information)
    X_train['state_risk'] = state_target_encode(df.loc[tr], df.loc[tr], TARGET).values
    X_val['state_risk']   = state_target_encode(df.loc[tr], df.loc[va], TARGET).values
    X_test['state_risk']  = state_target_encode(df.loc[tr], df.loc[te], TARGET).values
    FEATURES.append('state_risk')

    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"  {name:5s}: {len(y):>6,} rows  |  event rate = {y.mean():.4f}")
    print(f"  Features: {len(FEATURES)}")

    return FEATURES, TARGET, X_train, y_train, X_val, y_val, X_test, y_test, tr, va, te


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 & 5 — LIGHTGBM TRAINING + EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def train_lightgbm(FEATURES, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 72)
    print("PHASE 4-5: LIGHTGBM TRAINING & EVALUATION")
    print("=" * 72)

    pw = (y_train == 0).sum() / (y_train == 1).sum()
    params = LGB_PARAMS.copy()
    params['scale_pos_weight'] = pw

    # Phase A: find optimal rounds via early stopping on val
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    print("  Phase A — finding optimal rounds (train→val early stop)...")
    model_cv = lgb.train(
        params, dtrain, num_boost_round=3000,
        valid_sets=[dtrain, dval], valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(200)],
    )
    best_rounds = max(model_cv.best_iteration, 50)
    print(f"    Best iteration: {model_cv.best_iteration} → using {best_rounds}")

    # Phase B: retrain on train+val with fixed rounds
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    pw_tv = (y_tv == 0).sum() / (y_tv == 1).sum()
    params['scale_pos_weight'] = pw_tv

    print(f"  Phase B — retrain on train+val ({len(y_tv):,} rows), {best_rounds} rounds...")
    dtv   = lgb.Dataset(X_tv, label=y_tv)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtv)
    model = lgb.train(
        params, dtv, num_boost_round=best_rounds,
        valid_sets=[dtv, dtest], valid_names=['trainval', 'test'],
        callbacks=[lgb.log_evaluation(50)],
    )

    # Evaluate
    results = {}
    for name, X, y in [('TrainVal', X_tv, y_tv), ('Test', X_test, y_test)]:
        preds = model.predict(X)
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

    return model, model_cv, results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — GLM BASELINE
# ═══════════════════════════════════════════════════════════════════════

def train_glm(df, tr, va, te, y_test, TARGET):
    print("\n" + "=" * 72)
    print("PHASE 6: GLM BASELINE")
    print("=" * 72)

    scaler = StandardScaler()
    X_tv_g = scaler.fit_transform(df.loc[tr | va, GLM_FEATURES])
    y_tv_g = df.loc[tr | va, TARGET]
    X_te_g = scaler.transform(df.loc[te, GLM_FEATURES])

    glm = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42)
    glm.fit(X_tv_g, y_tv_g)

    glm_preds = glm.predict_proba(X_te_g)[:, 1]
    glm_auc   = roc_auc_score(y_test, glm_preds)
    glm_ap    = average_precision_score(y_test, glm_preds)
    glm_brier = brier_score_loss(y_test, glm_preds)
    print(f"  GLM → Test: AUC={glm_auc:.4f}  AP={glm_ap:.4f}  Brier={glm_brier:.6f}")

    return glm_preds, {'AUC-ROC': glm_auc, 'AP': glm_ap, 'Brier': glm_brier}


# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def walk_forward(df, FEATURES, TARGET):
    print("\n" + "=" * 72)
    print("PHASE 7: WALK-FORWARD EXPANDING-WINDOW VALIDATION")
    print("=" * 72)

    wf_lgb, wf_glm = [], []

    for test_year in range(WF_START, 2025):
        mask_tr = df['year'] < test_year
        mask_te = df['year'] == test_year
        if mask_te.sum() == 0 or df.loc[mask_te, TARGET].sum() == 0:
            continue

        y_tr = df.loc[mask_tr, TARGET]
        y_te = df.loc[mask_te, TARGET]

        # --- LightGBM ---
        X_tr = df.loc[mask_tr, FEATURES[:-1]].copy()   # without state_risk
        X_te = df.loc[mask_te, FEATURES[:-1]].copy()
        X_tr['state_risk'] = state_target_encode(df.loc[mask_tr], df.loc[mask_tr], TARGET).values
        X_te['state_risk'] = state_target_encode(df.loc[mask_tr], df.loc[mask_te], TARGET).values

        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        p = LGB_PARAMS.copy()
        p['scale_pos_weight'] = pw

        d_tr = lgb.Dataset(X_tr, label=y_tr)
        d_te = lgb.Dataset(X_te, label=y_te, reference=d_tr)
        mdl = lgb.train(
            p, d_tr, num_boost_round=500,
            valid_sets=[d_te], valid_names=['val'],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        pr = mdl.predict(X_te, num_iteration=mdl.best_iteration)

        wf_lgb.append({
            'year': test_year,
            'AUC': roc_auc_score(y_te, pr),
            'AP': average_precision_score(y_te, pr),
            'Brier': brier_score_loss(y_te, pr),
            'events': int(y_te.sum()),
            'rate': y_te.mean(),
            'best_iter': mdl.best_iteration,
        })

        # --- GLM ---
        sc = StandardScaler()
        X_tr_g = sc.fit_transform(df.loc[mask_tr, GLM_FEATURES])
        X_te_g = sc.transform(df.loc[mask_te, GLM_FEATURES])
        g = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42)
        g.fit(X_tr_g, y_tr)
        pg = g.predict_proba(X_te_g)[:, 1]

        wf_glm.append({
            'year': test_year,
            'AUC': roc_auc_score(y_te, pg),
            'AP': average_precision_score(y_te, pg),
            'Brier': brier_score_loss(y_te, pg),
        })

        delta = wf_lgb[-1]['AUC'] - wf_glm[-1]['AUC']
        print(f"  {test_year}: LGB AUC={wf_lgb[-1]['AUC']:.4f} (iter={mdl.best_iteration:3d})  "
              f"GLM AUC={wf_glm[-1]['AUC']:.4f}  Δ={delta:+.4f}")

    wf_lgb_df = pd.DataFrame(wf_lgb)
    wf_glm_df = pd.DataFrame(wf_glm)

    print(f"\n  Summary:")
    print(f"    LightGBM WF: AUC = {wf_lgb_df['AUC'].mean():.4f} ± {wf_lgb_df['AUC'].std():.4f}")
    print(f"    GLM WF:      AUC = {wf_glm_df['AUC'].mean():.4f} ± {wf_glm_df['AUC'].std():.4f}")
    print(f"    Mean ΔAUC:   {(wf_lgb_df['AUC'] - wf_glm_df['AUC']).mean():+.4f}")

    return wf_lgb_df, wf_glm_df


# ═══════════════════════════════════════════════════════════════════════
# PHASE 8 — SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def compute_shap(model, model_cv, FEATURES, X_val, X_test):
    print("\n" + "=" * 72)
    print("PHASE 8: SHAP INTERPRETABILITY")
    print("=" * 72)

    # Use the model with more iterations for richer SHAP
    use_model = model_cv if model_cv.best_iteration > 1 else model
    use_X     = X_val    if model_cv.best_iteration > 1 else X_test

    explainer   = shap.TreeExplainer(use_model)
    shap_values = explainer.shap_values(use_X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df  = (pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs})
                  .sort_values('mean_abs_shap', ascending=False))

    print("  Top 10 SHAP features:")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {clean_name(row['feature']):<30s} {row['mean_abs_shap']:.4f}")

    return shap_df, shap_values, use_X


# ═══════════════════════════════════════════════════════════════════════
# PHASE 9 — OPERATOR RISK RANKING
# ═══════════════════════════════════════════════════════════════════════

def rank_operators(df, FEATURES, TARGET):
    print("\n" + "=" * 72)
    print("PHASE 9: OPERATOR RISK RANKING (Walk-Forward Ensemble)")
    print("=" * 72)

    params = LGB_PARAMS.copy()
    frames = []

    for ty in [2023, 2024]:
        mask_tr = df['year'] < ty
        mask_te = df['year'] == ty
        y_tr = df.loc[mask_tr, TARGET]

        X_tr = df.loc[mask_tr, FEATURES[:-1]].copy()
        X_te = df.loc[mask_te, FEATURES[:-1]].copy()
        X_tr['state_risk'] = state_target_encode(df.loc[mask_tr], df.loc[mask_tr], TARGET).values
        X_te['state_risk'] = state_target_encode(df.loc[mask_tr], df.loc[mask_te], TARGET).values

        pw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        p = params.copy()
        p['scale_pos_weight'] = pw

        d_tr = lgb.Dataset(X_tr, label=y_tr)
        mdl  = lgb.train(p, d_tr, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])

        chunk = df.loc[mask_te].copy()
        chunk['pred_risk'] = mdl.predict(X_te)
        frames.append(chunk)

    df_recent = pd.concat(frames)

    # Aggregate per operator (miles-weighted mean)
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

    # Bootstrap 95% CI
    print("  Computing bootstrap confidence intervals...")
    np.random.seed(BOOT_SEED)
    ci_rows = []
    for oid in op_risk['operator_id'].unique():
        m = df_recent['operator_id'] == oid
        risks = df_recent.loc[m, 'pred_risk'].values
        miles = df_recent.loc[m, 'miles_at_risk'].values
        if len(risks) < 2:
            ci_rows.append({'operator_id': oid, 'ci_low': risks[0], 'ci_high': risks[0]})
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

    # Risk tiers
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
# PHASE 10 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def make_figures(model, results, glm_preds, y_test,
                 wf_lgb_df, wf_glm_df, shap_df, shap_values, use_X,
                 op_risk, df_recent, FEATURES, TARGET):
    print("\n" + "=" * 72)
    print("PHASE 10: VISUALIZATION")
    print("=" * 72)

    # ── FIGURE 1: Main Dashboard (5 rows × 2 cols = 10 panels) ──
    fig = plt.figure(figsize=(22, 28), facecolor=C['BG'])
    gs  = gridspec.GridSpec(5, 2, hspace=0.38, wspace=0.3,
                            left=0.07, right=0.95, top=0.94, bottom=0.03)

    fig.suptitle('LightGBM Pipeline — Gas Transmission Incident Prediction',
                 color=C['TEXT'], fontsize=17, fontweight='bold', y=0.975)
    fig.text(0.5, 0.96,
             'Walk-Forward Temporal Validation · SHAP Interpretability · Operator Risk Ranking',
             color='#808080', fontsize=11, ha='center')

    # [0,0] Walk-Forward AUC: LGB vs GLM
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(wf_lgb_df['year'], wf_lgb_df['AUC'], '-o', color=C['ACCENT'], lw=2.5,
            ms=7, mec='white', mew=1, label=f'LightGBM (μ={wf_lgb_df["AUC"].mean():.3f})', zorder=5)
    ax.plot(wf_glm_df['year'], wf_glm_df['AUC'], '--s', color=C['RED'], lw=2,
            ms=6, mec='white', mew=1, label=f'GLM (μ={wf_glm_df["AUC"].mean():.3f})', zorder=4)
    ax.fill_between(wf_lgb_df['year'],
                    wf_lgb_df['AUC'] - wf_lgb_df['AUC'].std(),
                    wf_lgb_df['AUC'] + wf_lgb_df['AUC'].std(), alpha=0.12, color=C['ACCENT'])
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
    ax.set_ylim(0.55, 0.95)
    ax.legend(fontsize=8.5, loc='lower left', facecolor=C['PANEL'],
              edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Walk-Forward AUC-ROC: LightGBM vs GLM', 'Test Year', 'AUC-ROC')

    # [0,1] Walk-Forward AP
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(wf_lgb_df['year'], wf_lgb_df['AP'], '-o', color=C['GOLD'], lw=2.5,
            ms=7, mec='white', mew=1, label=f'LightGBM (μ={wf_lgb_df["AP"].mean():.3f})', zorder=5)
    ax.plot(wf_glm_df['year'], wf_glm_df['AP'], '--s', color=C['RED'], lw=2,
            ms=6, label=f'GLM (μ={wf_glm_df["AP"].mean():.3f})', zorder=4)
    base_rate = wf_lgb_df['rate'].mean()
    ax.axhline(base_rate, color='gray', ls=':', alpha=0.4, label=f'Baseline={base_rate:.3f}')
    ax.set_ylim(0, 0.2)
    ax.legend(fontsize=8.5, loc='upper left', facecolor=C['PANEL'],
              edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Walk-Forward Average Precision', 'Test Year', 'AP (PR-AUC)')

    # [1,:] SHAP Feature Importance
    ax = fig.add_subplot(gs[1, :])
    top_n    = min(15, len(shap_df[shap_df['mean_abs_shap'] > 0]))
    shap_top = shap_df.head(top_n).iloc[::-1]
    median_s = shap_top['mean_abs_shap'].median()
    colors_s = [C['ACCENT'] if v > median_s else C['TEAL'] for v in shap_top['mean_abs_shap']]
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
    ax.plot([0, 0.15], [0, 0.15], '--', color='gray', alpha=0.5, label='Perfect')
    ax.legend(fontsize=8, loc='lower right', facecolor=C['PANEL'],
              edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Calibration (Test Set)', 'Predicted P(incident)', 'Observed Frequency')

    # [2,1] Precision-Recall
    ax = fig.add_subplot(gs[2, 1])
    for nm, pr, col, ls in [('LightGBM', results['Test']['preds'], C['ACCENT'], '-'),
                             ('GLM', glm_preds, C['RED'], '--')]:
        prec, rec, _ = precision_recall_curve(y_test, pr)
        ap_v = average_precision_score(y_test, pr)
        ax.plot(rec, prec, ls, color=col, lw=2, label=f'{nm} (AP={ap_v:.3f})')
    ax.axhline(y_test.mean(), color='gray', ls=':', alpha=0.5, label=f'Baseline={y_test.mean():.4f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8, loc='upper right', facecolor=C['PANEL'],
              edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Precision-Recall (Test Set)', 'Recall', 'Precision')

    # [3,0] LightGBM Gain importance
    ax  = fig.add_subplot(gs[3, 0])
    imp = model.feature_importance(importance_type='gain')
    imp_d = (pd.DataFrame({'feature': FEATURES, 'gain': imp})
               .sort_values('gain', ascending=True).tail(15))
    med_g = imp_d['gain'].median()
    colors_g = [C['ACCENT'] if v > med_g else C['BLUE'] for v in imp_d['gain']]
    ax.barh(range(len(imp_d)), imp_d['gain'], color=colors_g, alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(imp_d)))
    ax.set_yticklabels([clean_name(f) for f in imp_d['feature']], fontsize=9)
    style_ax(ax, 'LightGBM Feature Importance (Gain)', 'Total Gain', '')

    # [3,1] Risk distribution by era
    ax = fig.add_subplot(gs[3, 1])
    for era_name in ['era_pre1940', 'era_coal_tar', 'era_50s_60s', 'era_improved', 'era_modern']:
        subset = df_recent.loc[df_recent['era'] == era_name, 'pred_risk']
        if len(subset) > 0:
            label = era_name.replace('era_', '').replace('_', ' ').title()
            ax.hist(subset, bins=50, alpha=0.45, label=label, density=True)
    ax.set_xlim(0, df_recent['pred_risk'].quantile(0.995))
    ax.legend(fontsize=8, facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'Risk Distribution by Installation Era (2023-2024)', 'Predicted Risk', 'Density')

    # [4,:] Top 25 Operator Risk with CI
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
    legend_els = [Patch(facecolor=TIER_COLORS[t], alpha=0.8, label=t)
                  for t in ['CRITICAL', 'HIGH', 'ELEVATED', 'STANDARD']]
    ax.legend(handles=legend_els, fontsize=8, loc='lower right',
              facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'], ncol=4)
    style_ax(ax, 'Top 25 Highest-Risk Operators (2023-2024) — 95% Bootstrap CI',
             'Predicted Incident Probability', '')

    path1 = os.path.join(OUTPUT_DIR, 'fig1_pipeline_dashboard.png')
    plt.savefig(path1, dpi=180, facecolor=C['BG'], bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path1}")

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
    cb.set_label('Feature Value (normalized)', color=C['TEXT'], fontsize=8)
    cb.ax.tick_params(labelcolor=C['TEXT'], labelsize=7)

    # [0,1] SHAP dependence: log_miles
    ax = axes[0, 1]
    fi = FEATURES.index('log_miles')
    sc = ax.scatter(use_X['log_miles'].values, shap_values[:, fi],
                    c=use_X['age_at_obs'].values, cmap='coolwarm',
                    s=4, alpha=0.3, rasterized=True)
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    style_ax(ax, 'SHAP Dependence: Log(Miles) colored by Age',
             'Log(Miles at Risk)', 'SHAP Value')
    sm2 = plt.cm.ScalarMappable(
        cmap='coolwarm',
        norm=plt.Normalize(use_X['age_at_obs'].min(), use_X['age_at_obs'].max()))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax, shrink=0.5, pad=0.02)
    cb2.set_label('Pipe Age', color=C['TEXT'], fontsize=8)
    cb2.ax.tick_params(labelcolor=C['TEXT'], labelsize=7)

    # [1,0] Walk-forward ΔAUC
    ax    = axes[1, 0]
    delta = wf_lgb_df['AUC'].values - wf_glm_df['AUC'].values
    cols  = [C['ACCENT'] if d >= 0 else C['RED'] for d in delta]
    ax.bar(wf_lgb_df['year'], delta, color=cols, alpha=0.8,
           edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='gray', ls='-', alpha=0.5)
    ax.axhline(delta.mean(), color=C['GOLD'], ls='--', alpha=0.7,
               label=f'Mean Δ = {delta.mean():+.4f}')
    ax.legend(fontsize=8, facecolor=C['PANEL'], edgecolor=C['GRID'], labelcolor=C['TEXT'])
    style_ax(ax, 'LightGBM vs GLM: AUC Difference by Year', 'Year', 'ΔAUC (LGB − GLM)')

    # [1,1] Walk-forward iterations + events
    ax  = axes[1, 1]
    ax.bar(wf_lgb_df['year'], wf_lgb_df['best_iter'], color=C['BLUE'],
           alpha=0.8, edgecolor='white', linewidth=0.5)
    ax2r = ax.twinx()
    ax2r.plot(wf_lgb_df['year'], wf_lgb_df['events'], '-o', color=C['GOLD'], lw=2, ms=6)
    ax2r.set_ylabel('Events in Test Year', color=C['GOLD'], fontsize=9)
    ax2r.tick_params(colors=C['GOLD'], labelsize=8)
    style_ax(ax, 'Walk-Forward: Optimal Iterations & Events', 'Year', 'Best Iteration')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.suptitle('SHAP & Walk-Forward Diagnostics',
                  color=C['TEXT'], fontsize=14, fontweight='bold')

    path2 = os.path.join(OUTPUT_DIR, 'fig2_shap_walkforward.png')
    plt.savefig(path2, dpi=180, facecolor=C['BG'], bbox_inches='tight')
    plt.close()
    print(f"  ✓ {path2}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 11 — EXPORT CSVs
# ═══════════════════════════════════════════════════════════════════════

def export_csvs(op_risk, wf_lgb_df, wf_glm_df, shap_df):
    print("\n" + "=" * 72)
    print("PHASE 11: EXPORT CSVs")
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

def print_summary(df, FEATURES, model, results, wf_lgb_df, wf_glm_df,
                  shap_df, op_risk, glm_metrics, TARGET):
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    tiers = op_risk['risk_tier'].value_counts().to_dict()
    print(f"""
  Dataset:     {df.shape[0]:,} operator-decade-year observations
               {df['operator_id'].nunique():,} operators, {df['year'].min()}-{df['year'].max()}
               Event rate: {df[TARGET].mean():.4f} ({df[TARGET].sum():,} events)

  Features:    {len(FEATURES)} (structural + interactions + lag + operator history)
               Zero leakage: all same-year outcome vars excluded

  LightGBM:   Test AUC-ROC = {results['Test']['AUC-ROC']:.4f}
               Test AP      = {results['Test']['AP']:.4f}
               Test Brier   = {results['Test']['Brier']:.6f}

  GLM Baseline: Test AUC-ROC = {glm_metrics['AUC-ROC']:.4f}
                Test AP      = {glm_metrics['AP']:.4f}

  Walk-Forward (LightGBM): AUC = {wf_lgb_df['AUC'].mean():.4f} ± {wf_lgb_df['AUC'].std():.4f}
  Walk-Forward (GLM):      AUC = {wf_glm_df['AUC'].mean():.4f} ± {wf_glm_df['AUC'].std():.4f}
  Mean ΔAUC:               {(wf_lgb_df['AUC'] - wf_glm_df['AUC']).mean():+.4f}

  Top SHAP:  1. {clean_name(shap_df.iloc[0]['feature'])} ({shap_df.iloc[0]['mean_abs_shap']:.4f})
             2. {clean_name(shap_df.iloc[1]['feature'])} ({shap_df.iloc[1]['mean_abs_shap']:.4f})
             3. {clean_name(shap_df.iloc[2]['feature'])} ({shap_df.iloc[2]['mean_abs_shap']:.4f})

  Risk Tiers: CRITICAL={tiers.get('CRITICAL',0)} | HIGH={tiers.get('HIGH',0)} | ELEVATED={tiers.get('ELEVATED',0)} | STANDARD={tiers.get('STANDARD',0)}

  Output files:
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

    # Phase 1
    df = load_data(INPUT_PATH)

    # Phase 2
    df = engineer_features(df)

    # Phase 3
    FEATURES, TARGET, X_train, y_train, X_val, y_val, X_test, y_test, tr, va, te = build_splits(df)

    # Phase 4-5
    model, model_cv, results = train_lightgbm(FEATURES, X_train, y_train, X_val, y_val, X_test, y_test)

    # Phase 6
    glm_preds, glm_metrics = train_glm(df, tr, va, te, y_test, TARGET)

    # Phase 7
    wf_lgb_df, wf_glm_df = walk_forward(df, FEATURES, TARGET)

    # Phase 8
    shap_df, shap_values, use_X = compute_shap(model, model_cv, FEATURES, X_val, X_test)

    # Phase 9
    op_risk, df_recent = rank_operators(df, FEATURES, TARGET)

    # Phase 10
    make_figures(model, results, glm_preds, y_test,
                 wf_lgb_df, wf_glm_df, shap_df, shap_values, use_X,
                 op_risk, df_recent, FEATURES, TARGET)

    # Phase 11
    export_csvs(op_risk, wf_lgb_df, wf_glm_df, shap_df)

    # Summary
    print_summary(df, FEATURES, model, results, wf_lgb_df, wf_glm_df,
                  shap_df, op_risk, glm_metrics, TARGET)


if __name__ == '__main__':

    main()
