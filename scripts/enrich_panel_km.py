"""
ROSEN P4 — PANEL ENRICHMENT: Part K + M Feature Engineering
=============================================================
Reads all Part K (SMYS x Class) and Part M (Integrity Management)
files for 2010-2024, engineers new features, joins to the existing
survival panel, and runs the diagnostic to measure AUC improvement.

AFML TEMPORAL SAFETY:
  - Part K features: same-year join (static pipe characteristics)
  - Part M repairs: LAGGED by 1 year (repairs_t-1 -> predict events_t)
  - Part M tickets: same-year join (independent mechanism)
  - Cumulative features: sum through year t-1 only

NEW FEATURES:
  From Part K (per operator-state-year):
    k1. pct_low_smys         — miles(<30ksi) / total
    k2. pct_unknown_smys     — miles(unknown SMYS) / total
    k3. pct_non_steel        — miles(non-steel) / total
    k4. pct_high_class       — miles(Class 3+4) / total
    k5. pct_low_smys_hi_cls  — miles(<30ksi AND Class 3-4) / total

  From Part M (per operator-state-year):
    m1. log_tickets_per_mile — log1p(excavation tickets / miles)   [same year]
    m2. lag_repairs_cl12     — repairs found CL1-2 (year t-1)      [lagged]
    m3. lag_repairs_hca      — repairs found ON HCA (year t-1)     [lagged]
    m4. lag_damages          — excavation damages (year t-1)       [lagged]
    m5. lag_ext_corr_total   — ext corrosion repairs all loc (t-1) [lagged]
    m6. cum_repairs          — cumulative repairs through t-1      [rolling]
    m7. cum_ext_corrosion    — cumulative ext corr through t-1     [rolling]
    m8. cum_damages          — cumulative damages through t-1      [rolling]

Usage:
    cd C:\Phmsa\annual_gt
    python enrich_panel_km.py

Input:
    survival_panel_15yr_final.csv          (same directory)
    extracted_csvs\GT_AR_YYYY_Part_K.csv   (2010-2024)
    extracted_csvs\GT_AR_YYYY_Part_M.csv   (2010-2024)

Output:
    survival_panel_15yr_enriched.csv       — enriched panel
    enrichment_diagnostic.png              — AUC comparison dashboard
    enrichment_report.txt                  — text summary

Author: Kent (ROSEN Project 4)
Date:   2026-02-04
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PANEL_FILE   = os.path.join(BASE_DIR, 'survival_panel_15yr_final.csv')
EXTRACT_DIR  = os.path.join(BASE_DIR, 'extracted_csvs')
OUTPUT_DIR   = BASE_DIR

YEARS        = list(range(2010, 2025))
TRAIN_END    = 2019
TEST_START   = 2020
ENCODINGS    = ['latin-1', 'cp1252', 'utf-8', 'iso-8859-1']


# ============================================================
# UTILITIES
# ============================================================

def safe_numeric(series):
    """Convert to numeric, coerce errors to 0."""
    return pd.to_numeric(series, errors='coerce').fillna(0.0)


def safe_read_csv(path):
    """Read CSV trying multiple encodings."""
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot read {path} with any encoding: {ENCODINGS}")


def safe_divide(a, b, fill=0.0):
    """Safe division, fill where denominator is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(b > 0, a / b, fill)
    return result


# ============================================================
# PHASE 1: LOAD PART K FILES → SMYS/CLASS FEATURES
# ============================================================

def load_part_k_all(extract_dir, years):
    """Load all Part K files and compute SMYS x Class features."""
    print("=" * 70)
    print("PHASE 1: LOADING PART K (SMYS x CLASS)")
    print("=" * 70)

    all_k = []
    loaded_years = []

    for yr in years:
        path = os.path.join(extract_dir, f'GT_AR_{yr}_Part_K.csv')
        if not os.path.exists(path):
            print(f"  {yr}: NOT FOUND — skipping")
            continue

        df = safe_read_csv(path)
        loaded_years.append(yr)

        # --- Extract key columns ---
        records = []
        for _, row in df.iterrows():
            oid = row.get('OPERATOR_ID')
            state = str(row.get('STATE_NAME', '')).strip().upper()
            if pd.isna(oid) or not state:
                continue

            total = safe_numeric(pd.Series([row.get('PARTKONTOTAL', 0)])).iloc[0]
            if total <= 0:
                # Try alternative total column
                total = safe_numeric(pd.Series([row.get('PARTKTOTAL', 0)])).iloc[0]
            if total <= 0:
                continue

            # Low SMYS: <20 ksi + 20-29 ksi
            low_smys = (safe_numeric(pd.Series([row.get('PARTK20LESSTOT', 0)])).iloc[0] +
                        safe_numeric(pd.Series([row.get('PARTK2029TOT', 0)])).iloc[0])

            # Unknown SMYS
            unk_smys = safe_numeric(pd.Series([row.get('PARTKUNKNOWNTOT', 0)])).iloc[0]

            # Non-steel
            non_steel = safe_numeric(pd.Series([row.get('PARTKNONSTEELTOT', 0)])).iloc[0]

            # High class (3 + 4)
            cl3 = safe_numeric(pd.Series([row.get('PARTKONC3TOT', 0)])).iloc[0]
            cl4 = safe_numeric(pd.Series([row.get('PARTKONC4TOT', 0)])).iloc[0]

            # Low SMYS x High Class
            low_hi = 0.0
            for prefix in ['PARTK20LESS', 'PARTK2029']:
                for cl in ['C3', 'C4']:
                    col = f'{prefix}{cl}'
                    low_hi += safe_numeric(pd.Series([row.get(col, 0)])).iloc[0]

            records.append({
                'operator_id': int(oid),
                'state': state,
                'year': yr,
                'k_total_miles': total,
                'k_low_smys': low_smys,
                'k_unknown_smys': unk_smys,
                'k_non_steel': non_steel,
                'k_high_class': cl3 + cl4,
                'k_low_smys_hi_cls': low_hi,
            })

        k_year = pd.DataFrame(records)
        all_k.append(k_year)
        print(f"  {yr}: {len(k_year)} rows loaded")

    if not all_k:
        print("  *** NO Part K files found!")
        return pd.DataFrame()

    k_all = pd.concat(all_k, ignore_index=True)

    # AGGREGATE duplicates: same operator-state-year can appear multiple times
    # (Interstate vs Intrastate, supplemental reports). Sum miles.
    agg_cols = ['k_total_miles', 'k_low_smys', 'k_unknown_smys',
                'k_non_steel', 'k_high_class', 'k_low_smys_hi_cls']
    k_all = k_all.groupby(['operator_id', 'state', 'year'], as_index=False)[agg_cols].sum()
    print(f"  After dedup aggregation: {len(k_all)} rows")

    # Compute ratio features
    miles = k_all['k_total_miles'].values
    k_all['pct_low_smys']        = safe_divide(k_all['k_low_smys'].values, miles)
    k_all['pct_unknown_smys']    = safe_divide(k_all['k_unknown_smys'].values, miles)
    k_all['pct_non_steel']       = safe_divide(k_all['k_non_steel'].values, miles)
    k_all['pct_high_class']      = safe_divide(k_all['k_high_class'].values, miles)
    k_all['pct_low_smys_hi_cls'] = safe_divide(k_all['k_low_smys_hi_cls'].values, miles)

    print(f"\n  Total Part K: {len(k_all)} rows, years: {sorted(loaded_years)}")
    feat_cols = ['pct_low_smys', 'pct_unknown_smys', 'pct_non_steel',
                 'pct_high_class', 'pct_low_smys_hi_cls']
    for c in feat_cols:
        nz = (k_all[c] > 0).sum()
        print(f"    {c:25s}: nonzero={nz:5d}/{len(k_all)} ({nz/len(k_all)*100:5.1f}%)")

    return k_all[['operator_id', 'state', 'year'] + feat_cols]


# ============================================================
# PHASE 2: LOAD PART M FILES → INTEGRITY MANAGEMENT FEATURES
# ============================================================

# Column mappings for cause-specific repair totals
# For each cause, sum all location types (T_ON + T_LF + T_OFF + G_ON + G_OFF)
CAUSE_COLUMNS = {
    'ext_corrosion': {
        'trans': ['PARTMTCECONHCA', 'PARTMTLFMCAEC', 'PARTMTLFCL34EC', 'PARTMTLFCL12EC',
                  'PARTMTCECOFFHCA', 'PARTMTCECOFFNHCA', 'PARTMTCECFHCA'],
        'gather': ['PARTMGCECONA', 'PARTMGCECONB', 'PARTMGCECONC', 'PARTMGCECOFF'],
    },
    'int_corrosion': {
        'trans': ['PARTMTCICONHCA', 'PARTMTLFMCAIC', 'PARTMTLFCL34IC', 'PARTMTLFCL12IC',
                  'PARTMTCICOFFHCA', 'PARTMTCICOFFNHCA', 'PARTMTCICFHCA'],
        'gather': ['PARTMGCICONA', 'PARTMGCICONB', 'PARTMGCICONC', 'PARTMGCICOFF'],
    },
    'scc': {
        'trans': ['PARTMTCSCONHCA', 'PARTMTLFMCASC', 'PARTMTLFCL34SC', 'PARTMTLFCL12SC',
                  'PARTMTCSCOFFHCA', 'PARTMTCSCOFFNHCA', 'PARTMTCSCFHCA'],
        'gather': ['PARTMGCSCONA', 'PARTMGCSCONB', 'PARTMGCSCONC', 'PARTMGCSCOFF'],
    },
    'equipment': {
        'trans': ['PARTMTCEONHCA', 'PARTMTLFMCAEQ', 'PARTMTLFCL34EQ', 'PARTMTLFCL12EQ',
                  'PARTMTCEOFFHCA', 'PARTMTCEOFFNHCA', 'PARTMTCEFHCA'],
        'gather': ['PARTMGCEONA', 'PARTMGCEONB', 'PARTMGCEONC', 'PARTMGCEOFF'],
    },
    'earth_movement': {
        'trans': ['PARTMTCEDONHCA', 'PARTMTPDMCAED', 'PARTMTPDCL34ED', 'PARTMTPDCL12ED',
                  'PARTMTCEDOFFHCA', 'PARTMTCEDOFFNHCA', 'PARTMTCEDFHCA'],
        'gather': ['PARTMGCEDONA', 'PARTMGCEDONB', 'PARTMGCEDONC', 'PARTMGCEDOFF'],
    },
}

TOTAL_COLUMNS = {
    'repairs_hca':   'PARTMTONHCATOT',
    'repairs_cl34':  'PARTMTFCLASS34TOT',
    'repairs_cl12':  'PARTMTFCLASS12TOT',
    'repairs_fhca':  'PARTMTFHCATOT',
    'gather_a':      'PARTMGONATOT',
    'gather_b':      'PARTMGONBTOT',
    'gather_c':      'PARTMGONCTOT',
}

DAMAGE_COLUMNS = {
    'damages_t':     'PARTM4TOTDAMAGES',
    'tickets_t':     'PARTM4TOTTICKETS',
    'damages_g':     'PARTM5TOTDAMAGES',
    'tickets_g':     'PARTM5TOTTICKETS',
}

SURVEY_COLUMNS = {
    'leak_survey_t': 'PARTMTLSFR',
    'leak_survey_total': 'PARTMTLSRTOTAL',
    'cp_miles':      'PARTMM3TOTAL',
}


def sum_cause_columns(row, col_list, available_cols):
    """Sum numeric values for available columns."""
    total = 0.0
    for col in col_list:
        if col in available_cols:
            total += safe_numeric(pd.Series([row.get(col, 0)])).iloc[0]
    return total


def load_part_m_all(extract_dir, years):
    """Load all Part M files and extract raw metrics per operator-state-year."""
    print(f"\n{'='*70}")
    print("PHASE 2: LOADING PART M (INTEGRITY MANAGEMENT)")
    print("=" * 70)

    all_m = []
    loaded_years = []

    for yr in years:
        path = os.path.join(extract_dir, f'GT_AR_{yr}_Part_M.csv')
        if not os.path.exists(path):
            print(f"  {yr}: NOT FOUND — skipping")
            continue

        df = safe_read_csv(path)
        available = set(df.columns)
        loaded_years.append(yr)

        records = []
        for _, row in df.iterrows():
            oid = row.get('OPERATOR_ID')
            state = str(row.get('STATE_NAME', '')).strip().upper()
            if pd.isna(oid) or not state:
                continue

            rec = {
                'operator_id': int(oid),
                'state': state,
                'year': yr,
            }

            # Cause-specific totals (sum across all location types)
            for cause, col_groups in CAUSE_COLUMNS.items():
                all_cols = col_groups.get('trans', []) + col_groups.get('gather', [])
                rec[f'n_{cause}'] = sum_cause_columns(row, all_cols, available)

            # Pre-computed totals
            for name, col in TOTAL_COLUMNS.items():
                rec[name] = safe_numeric(pd.Series([row.get(col, 0)])).iloc[0]

            # Damage prevention
            for name, col in DAMAGE_COLUMNS.items():
                rec[name] = safe_numeric(pd.Series([row.get(col, 0)])).iloc[0]

            # Surveys
            for name, col in SURVEY_COLUMNS.items():
                rec[name] = safe_numeric(pd.Series([row.get(col, 0)])).iloc[0]

            # Computed totals
            rec['total_repairs'] = (rec['repairs_hca'] + rec['repairs_cl34'] +
                                    rec['repairs_cl12'] + rec['repairs_fhca'] +
                                    rec['gather_a'] + rec['gather_b'] +
                                    rec['gather_c'])
            rec['total_damages'] = rec['damages_t'] + rec['damages_g']
            rec['total_tickets'] = rec['tickets_t'] + rec['tickets_g']
            rec['total_corrosion'] = (rec['n_ext_corrosion'] +
                                      rec['n_int_corrosion'] + rec['n_scc'])

            records.append(rec)

        m_year = pd.DataFrame(records)
        all_m.append(m_year)
        print(f"  {yr}: {len(m_year)} rows, "
              f"tot_repairs={m_year['total_repairs'].sum():.0f}, "
              f"ext_corr={m_year['n_ext_corrosion'].sum():.0f}, "
              f"tickets={m_year['total_tickets'].sum():.0f}")

    if not all_m:
        print("  *** NO Part M files found!")
        return pd.DataFrame()

    m_all = pd.concat(all_m, ignore_index=True)

    # AGGREGATE duplicates: same operator-state-year from Interstate/Intrastate
    # and supplemental reports. Sum all numeric columns.
    num_cols = [c for c in m_all.columns if c not in ['operator_id', 'state', 'year']]
    m_all = m_all.groupby(['operator_id', 'state', 'year'], as_index=False)[num_cols].sum()
    print(f"  After dedup aggregation: {len(m_all)} rows")

    print(f"\n  Total Part M: {len(m_all)} rows, years: {sorted(loaded_years)}")

    return m_all


# ============================================================
# PHASE 3: ENGINEER FEATURES WITH TEMPORAL SAFETY
# ============================================================

def engineer_m_features(m_all):
    """Build lagged and cumulative features from Part M data.

    AFML temporal safety rules:
      - Same-year: only tickets (independent mechanism, not an outcome)
      - Lagged (t-1): repairs, damages, cause-specific counts
      - Cumulative (through t-1): rolling sums for rare events
    """
    print(f"\n{'='*70}")
    print("PHASE 3: FEATURE ENGINEERING (AFML TEMPORAL SAFETY)")
    print("=" * 70)

    if m_all.empty:
        return pd.DataFrame()

    # Sort for proper lag computation
    m_all = m_all.sort_values(['operator_id', 'state', 'year']).reset_index(drop=True)

    # Group key
    gk = ['operator_id', 'state']

    # --- SAME-YEAR features (safe: tickets are independent of incidents) ---
    m_all['log_tickets'] = np.log1p(m_all['total_tickets'])

    # --- LAGGED features (t-1) ---
    lag_cols = ['total_repairs', 'repairs_cl12', 'repairs_hca', 'repairs_cl34',
                'total_damages', 'n_ext_corrosion', 'n_int_corrosion', 'n_scc',
                'n_equipment', 'n_earth_movement', 'total_corrosion',
                'leak_survey_t', 'cp_miles']

    grouped = m_all.groupby(gk)
    for col in lag_cols:
        m_all[f'lag_{col}'] = grouped[col].shift(1)

    # --- CUMULATIVE features (sum through t-1) ---
    cum_cols = ['total_repairs', 'n_ext_corrosion', 'total_corrosion',
                'total_damages', 'n_scc']

    for col in cum_cols:
        # cumsum then shift = sum of all values BEFORE current year
        m_all[f'cum_{col}'] = grouped[col].cumsum().shift(1)

    # Fill NaN from lag/cumsum (first year for each operator has no history)
    fill_cols = [c for c in m_all.columns if c.startswith('lag_') or c.startswith('cum_')]
    m_all[fill_cols] = m_all[fill_cols].fillna(0.0)

    # Report
    print("  Same-year features: log_tickets")
    print(f"  Lagged features (t-1): {len(lag_cols)} columns")
    print(f"  Cumulative features (through t-1): {len(cum_cols)} columns")
    print(f"  Total rows with features: {len(m_all)}")

    # Select output columns
    out_cols = (gk + ['year', 'log_tickets'] +
                [f'lag_{c}' for c in lag_cols] +
                [f'cum_{c}' for c in cum_cols])

    return m_all[out_cols]


# ============================================================
# PHASE 4: JOIN TO PANEL
# ============================================================

def join_to_panel(panel, k_features, m_features):
    """Join Part K and M features to the survival panel."""
    print(f"\n{'='*70}")
    print("PHASE 4: JOINING TO SURVIVAL PANEL")
    print("=" * 70)

    n_orig = len(panel)
    print(f"  Panel: {n_orig:,} rows")

    # Standardize state names in panel
    panel['state_upper'] = panel['state'].str.strip().str.upper()

    # --- Join Part K ---
    if not k_features.empty:
        k_features = k_features.rename(columns={'state': 'state_upper'})
        k_features['state_upper'] = k_features['state_upper'].str.strip().str.upper()
        panel = panel.merge(
            k_features, on=['operator_id', 'state_upper', 'year'], how='left'
        )
        k_cols = [c for c in k_features.columns if c.startswith('pct_')]
        matched_k = panel[k_cols[0]].notna().sum() if k_cols else 0
        print(f"  Part K matched: {matched_k:,}/{n_orig:,} ({matched_k/n_orig*100:.1f}%)")
    else:
        k_cols = []
        print("  Part K: NO DATA")

    # --- Join Part M ---
    if not m_features.empty:
        m_features = m_features.rename(columns={'state': 'state_upper'})
        m_features['state_upper'] = m_features['state_upper'].str.strip().str.upper()
        panel = panel.merge(
            m_features, on=['operator_id', 'state_upper', 'year'], how='left'
        )
        m_cols = [c for c in m_features.columns
                  if c.startswith('lag_') or c.startswith('cum_') or c == 'log_tickets']
        matched_m = panel['log_tickets'].notna().sum() if 'log_tickets' in panel.columns else 0
        print(f"  Part M matched: {matched_m:,}/{n_orig:,} ({matched_m/n_orig*100:.1f}%)")
    else:
        m_cols = []
        print("  Part M: NO DATA")

    # Fill remaining NaN with 0 (unmatched operators = no data = no repairs)
    all_new = k_cols + m_cols
    for c in all_new:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)

    # Compute derived features requiring both M and panel data
    if 'log_tickets' in panel.columns:
        panel['log_tickets_per_mile'] = np.log1p(
            np.expm1(panel['log_tickets']) /
            panel['miles_at_risk'].replace(0, np.nan)
        ).fillna(0.0)
        all_new.append('log_tickets_per_mile')

    # Log-transform cumulative features for better model behavior
    for c in [col for col in panel.columns if col.startswith('cum_')]:
        log_col = f'log1p_{c}'
        panel[log_col] = np.log1p(panel[c])
        all_new.append(log_col)

    panel.drop(columns=['state_upper'], inplace=True)

    print(f"\n  Final panel: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"  New features added: {len(all_new)}")

    return panel, all_new


# ============================================================
# PHASE 5: DIAGNOSTIC — AUC COMPARISON
# ============================================================

def run_diagnostic(panel, new_features):
    """Compare AUC: baseline vs enriched models."""
    print(f"\n{'='*70}")
    print("PHASE 5: DIAGNOSTIC — AUC COMPARISON")
    print("=" * 70)

    # Clean data
    panel['log1p_total_repairs'] = panel['log1p_total_repairs'].fillna(0)
    if 'log1p_ext_corrosion' in panel.columns:
        panel.drop(columns=['log1p_ext_corrosion'], inplace=True, errors='ignore')
    panel['log_miles'] = np.log1p(panel['miles_at_risk'])

    y = panel['event'].values
    train = (panel['year'] <= TRAIN_END).values
    test = (panel['year'] >= TEST_START).values
    y_test = y[test]

    # Baseline features (original panel)
    era_dum = pd.get_dummies(panel['era'], prefix='era', drop_first=True)
    base_cols = ['log_miles', 'age_at_obs', 'log1p_total_repairs',
                 'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1']
    X_base = np.column_stack([panel[base_cols].values, era_dum.values])

    results = []

    # --- Model 0: log(miles) only ---
    X0 = panel[['log_miles']].values
    lr0 = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr0.fit(X0[train], y[train])
    auc0 = roc_auc_score(y_test, lr0.predict_proba(X0[test])[:, 1])
    results.append(('log(miles) only', auc0, 1))

    # --- Model 1: original baseline (11 features) ---
    lr1 = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr1.fit(X_base[train], y[train])
    auc1 = roc_auc_score(y_test, lr1.predict_proba(X_base[test])[:, 1])
    results.append(('Original baseline (11)', auc1, X_base.shape[1]))

    # --- Model 2: + Part K features only ---
    k_cols = [c for c in new_features if c.startswith('pct_') and c not in base_cols]
    k_cols_available = [c for c in k_cols if c in panel.columns]
    if k_cols_available:
        X2 = np.column_stack([X_base, panel[k_cols_available].values])
        lr2 = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr2.fit(X2[train], y[train])
        auc2 = roc_auc_score(y_test, lr2.predict_proba(X2[test])[:, 1])
        results.append((f'+ Part K ({len(k_cols_available)})', auc2, X2.shape[1]))
    else:
        auc2 = auc1
        results.append(('+ Part K (none)', auc2, X_base.shape[1]))

    # --- Model 3: + Part M lagged features ---
    m_lag_cols = [c for c in new_features
                  if c.startswith('lag_') or c.startswith('log1p_cum')]
    m_lag_available = [c for c in m_lag_cols if c in panel.columns]
    if m_lag_available:
        X3_cols = np.column_stack([X_base,
                                    panel[k_cols_available].values if k_cols_available else np.empty((len(panel), 0)),
                                    panel[m_lag_available].values])
        lr3 = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.5)
        lr3.fit(X3_cols[train], y[train])
        auc3 = roc_auc_score(y_test, lr3.predict_proba(X3_cols[test])[:, 1])
        results.append((f'+ Part M lagged ({len(m_lag_available)})', auc3, X3_cols.shape[1]))
    else:
        auc3 = auc2
        results.append(('+ Part M lagged (none)', auc3, X_base.shape[1]))

    # --- Model 4: + tickets (same-year, independent) ---
    ticket_cols = [c for c in new_features
                   if 'ticket' in c.lower() and c in panel.columns]
    all_new_available = k_cols_available + m_lag_available + ticket_cols
    all_new_available = list(dict.fromkeys(all_new_available))  # deduplicate

    if all_new_available:
        X4 = np.column_stack([X_base, panel[all_new_available].values])
        lr4 = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.5)
        lr4.fit(X4[train], y[train])
        auc4 = roc_auc_score(y_test, lr4.predict_proba(X4[test])[:, 1])
        results.append((f'ALL enriched ({len(all_new_available)} new)', auc4, X4.shape[1]))
    else:
        auc4 = auc3
        results.append(('ALL enriched (none)', auc4, X_base.shape[1]))

    # --- Print results ---
    ceiling = max(r[1] for r in results) - 0.5
    if ceiling <= 0:
        ceiling = 0.001

    print(f"\n  {'Model':<40s} {'Nf':>4s}  {'AUC':>7s}  {'dAUC':>7s}  {'%ceil':>6s}")
    print(f"  {'-'*40} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}")
    for name, auc, nf in results:
        d = auc - 0.5
        p = d / ceiling * 100
        print(f"  {name:<40s} {nf:>4d}  {auc:.4f}  {d:+.4f}  {p:5.1f}%")

    # Delta from original baseline
    delta_k = (auc2 - auc1) * 100
    delta_m = (auc3 - auc2) * 100
    delta_total = (auc4 - auc1) * 100

    print(f"\n  MARGINAL GAINS:")
    print(f"    Part K (SMYS/Class):    {delta_k:+.2f}pp")
    print(f"    Part M (lagged repairs): {delta_m:+.2f}pp")
    print(f"    Total enrichment:        {delta_total:+.2f}pp")
    print(f"    Original ceiling delta:  {(auc1-auc0)*100:+.2f}pp was old 10-feature gain")

    if delta_total > 0.5:
        print(f"\n  ** ENRICHMENT BREAKS THE CEILING (+{delta_total:.2f}pp)")
    elif delta_total > 0.1:
        print(f"\n  ** Modest improvement (+{delta_total:.2f}pp) — worth including")
    else:
        print(f"\n  ** Enrichment adds minimal signal (+{delta_total:.2f}pp)")

    # --- Feature-level signal (new features only) ---
    print(f"\n  NEW FEATURE CORRELATIONS WITH TARGET:")
    for c in all_new_available:
        r = panel[c].corr(panel['event'])
        # Partial correlation (controlled for log_miles)
        X_ctrl = panel[['log_miles']].values
        from numpy.linalg import lstsq
        res_y = y - X_ctrl @ lstsq(X_ctrl, y, rcond=None)[0]
        feat_v = panel[c].values
        res_f = feat_v - X_ctrl @ lstsq(X_ctrl, feat_v, rcond=None)[0]
        pr = np.corrcoef(res_y, res_f)[0, 1] if np.std(res_f) > 0 else 0
        nz = (panel[c] > 0).sum()
        print(f"    {c:35s}: r={r:+.4f}  partial_r={pr:+.4f}  nz={nz/len(panel)*100:5.1f}%")

    return results, all_new_available


# ============================================================
# PHASE 6: VISUALIZATION
# ============================================================

def plot_enrichment_diagnostic(results, panel, new_features, output_dir):
    """Create comparison dashboard."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('ROSEN P4 — PANEL ENRICHMENT DIAGNOSTIC: Part K + M Features',
                 fontsize=15, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.07, right=0.97)
    C = {'main': '#2563EB', 'red': '#DC2626', 'gray': '#6B7280',
         'green': '#059669', 'orange': '#D97706', 'purple': '#7C3AED'}

    # Panel 1: AUC comparison
    ax = fig.add_subplot(gs[0, 0])
    names = [r[0] for r in results]
    aucs = [r[1] for r in results]
    colors_bar = [C['gray'], C['gray'], C['main'], C['green'], C['purple']]
    colors_bar = colors_bar[:len(names)]
    bars = ax.barh(range(len(names)), aucs, color=colors_bar, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i, a in enumerate(aucs):
        ax.text(a + 0.001, i, f'{a:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0.75, max(aucs) + 0.02)
    ax.set_xlabel('AUC')
    ax.set_title('(1) AUC: Baseline vs Enriched', fontweight='bold')
    ax.axvline(aucs[1], color=C['red'], linestyle='--', alpha=0.5, label='Original baseline')
    ax.legend(fontsize=8)

    # Panel 2: Marginal gains
    ax = fig.add_subplot(gs[0, 1])
    marginals = [aucs[0] - 0.5]
    for i in range(1, len(aucs)):
        marginals.append(aucs[i] - aucs[i - 1])
    mpp = [m * 100 for m in marginals]
    bclr = [C['gray']] + [C['green'] if m > 0 else C['orange'] for m in marginals[1:]]
    ax.bar(range(len(names)), mpp, color=bclr)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    for i, m in enumerate(mpp):
        label = f'+{m:.1f}' if i == 0 else f'{m:+.2f}'
        ax.text(i, m + (0.1 if m >= 0 else -0.3), label, ha='center', fontsize=8)
    ax.set_ylabel('Marginal AUC (pp)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('(2) Marginal Value per Feature Group', fontweight='bold')

    # Panel 3: New feature correlations
    ax = fig.add_subplot(gs[0, 2])
    if new_features:
        y_vals = panel['event'].values
        corrs = [(c, panel[c].corr(panel['event'])) for c in new_features if c in panel.columns]
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        corrs = corrs[:12]  # top 12
        cnames = [c[0].replace('log1p_cum_', 'cum_').replace('lag_', 'L1_')[:25] for c in corrs]
        cvals = [c[1] for c in corrs]
        fc = [C['green'] if abs(v) > 0.05 else C['gray'] for v in cvals]
        ax.barh(range(len(cnames)), cvals, color=fc, height=0.6)
        ax.set_yticks(range(len(cnames)))
        ax.set_yticklabels(cnames, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.5)
        for i, v in enumerate(cvals):
            ax.text(v + 0.002 * np.sign(v), i, f'{v:+.4f}', va='center', fontsize=8)
    ax.set_xlabel('Correlation with event')
    ax.set_title('(3) New Feature Correlations', fontweight='bold')

    # Panel 4: Feature nonzero rates
    ax = fig.add_subplot(gs[1, 0])
    if new_features:
        nz_rates = [(c, (panel[c] > 0).sum() / len(panel) * 100)
                     for c in new_features if c in panel.columns]
        nz_rates.sort(key=lambda x: x[1], reverse=True)
        nz_rates = nz_rates[:12]
        nz_names = [c[0].replace('log1p_cum_', 'cum_').replace('lag_', 'L1_')[:25] for c in nz_rates]
        nz_vals = [c[1] for c in nz_rates]
        fc = [C['main'] if v > 20 else C['gray'] for v in nz_vals]
        ax.barh(range(len(nz_names)), nz_vals, color=fc, height=0.6)
        ax.set_yticks(range(len(nz_names)))
        ax.set_yticklabels(nz_names, fontsize=8)
        for i, v in enumerate(nz_vals):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    ax.set_xlabel('Nonzero rate (%)')
    ax.set_title('(4) Feature Coverage', fontweight='bold')

    # Panel 5: Year-by-year AUC (if enough features)
    ax = fig.add_subplot(gs[1, 1])
    base_cols_list = ['log_miles', 'age_at_obs', 'log1p_total_repairs',
                      'pct_small_diam', 'pct_large_diam', 'pct_high_smys', 'pct_class1']
    era_dum = pd.get_dummies(panel['era'], prefix='era', drop_first=True)
    X_base_full = np.column_stack([panel[base_cols_list].values, era_dum.values])

    avail_new = [c for c in new_features if c in panel.columns]
    if avail_new:
        X_enriched = np.column_stack([X_base_full, panel[avail_new].values])
    else:
        X_enriched = X_base_full

    y_all = panel['event'].values
    test_years = sorted(panel[panel['year'] >= TEST_START]['year'].unique())
    auc_base_yr, auc_enrich_yr = [], []
    for yr in test_years:
        tr = (panel['year'] < yr).values
        te = (panel['year'] == yr).values
        if y_all[te].sum() < 5:
            continue
        lr_b = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr_b.fit(X_base_full[tr], y_all[tr])
        ab = roc_auc_score(y_all[te], lr_b.predict_proba(X_base_full[te])[:, 1])
        lr_e = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.5)
        lr_e.fit(X_enriched[tr], y_all[tr])
        ae = roc_auc_score(y_all[te], lr_e.predict_proba(X_enriched[te])[:, 1])
        auc_base_yr.append(ab)
        auc_enrich_yr.append(ae)

    if auc_base_yr:
        x = range(len(test_years[:len(auc_base_yr)]))
        ax.plot(x, auc_base_yr, 'o-', color=C['gray'], label='Baseline', linewidth=2)
        ax.plot(x, auc_enrich_yr, 's-', color=C['green'], label='Enriched', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(test_years[:len(auc_base_yr)])
        ax.legend(fontsize=9)
    ax.set_xlabel('Test Year')
    ax.set_ylabel('AUC')
    ax.set_title('(5) Year-by-Year AUC Comparison', fontweight='bold')

    # Panel 6: Summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    total_delta = (aucs[-1] - aucs[1]) * 100 if len(aucs) > 1 else 0
    n_new = len(avail_new)
    txt = (
        f"(6) ENRICHMENT SUMMARY\n\n"
        f"New features: {n_new}\n"
        f"  Part K (SMYS/Class): {len([c for c in avail_new if c.startswith('pct_') and c not in base_cols_list])}\n"
        f"  Part M (lagged):     {len([c for c in avail_new if c.startswith('lag_') or c.startswith('log1p_cum')])}\n"
        f"  Part M (tickets):    {len([c for c in avail_new if 'ticket' in c])}\n\n"
        f"AUC improvement: {total_delta:+.2f}pp\n"
        f"  Baseline: {aucs[1]:.4f}\n"
        f"  Enriched: {aucs[-1]:.4f}\n\n"
    )
    if total_delta > 0.5:
        txt += "VERDICT: Enrichment adds\nmeaningful signal."
    elif total_delta > 0.1:
        txt += "VERDICT: Modest improvement.\nMay help at operator level."
    else:
        txt += "VERDICT: Minimal gain.\nSignal ceiling unchanged."

    ax.text(0.5, 0.5, txt, transform=ax.transAxes, fontsize=11,
            va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0FDF4',
                      edgecolor=C['green'], alpha=0.9))

    path = os.path.join(output_dir, 'enrichment_diagnostic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {path}")


# ============================================================
# PHASE 7: SAVE OUTPUTS
# ============================================================

def save_outputs(panel, results, new_features, output_dir):
    """Save enriched panel and report."""
    # --- Enriched panel ---
    panel_path = os.path.join(output_dir, 'survival_panel_15yr_enriched.csv')
    panel.to_csv(panel_path, index=False)
    print(f"  Saved: {panel_path} ({len(panel):,} rows x {len(panel.columns)} cols)")

    # --- Text report ---
    report_path = os.path.join(output_dir, 'enrichment_report.txt')
    lines = [
        "=" * 70,
        "ROSEN P4 — PANEL ENRICHMENT REPORT",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "INCREMENTAL AUC TABLE",
        "-" * 50,
    ]
    for name, auc, nf in results:
        lines.append(f"  {name:<40s}  AUC={auc:.4f}  ({nf} features)")

    lines += [
        "",
        "NEW FEATURES ADDED",
        "-" * 50,
    ]
    for c in sorted(new_features):
        if c in panel.columns:
            nz = (panel[c] > 0).sum()
            r = panel[c].corr(panel['event'])
            lines.append(f"  {c:35s}: r={r:+.4f}  nonzero={nz/len(panel)*100:.1f}%")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ROSEN P4 — PANEL ENRICHMENT: Part K + M Features")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Check inputs
    if not os.path.exists(PANEL_FILE):
        print(f"ERROR: Panel not found: {PANEL_FILE}")
        sys.exit(1)
    if not os.path.isdir(EXTRACT_DIR):
        print(f"ERROR: Extract dir not found: {EXTRACT_DIR}")
        print(f"  Expected: {EXTRACT_DIR}")
        print(f"  Make sure extracted_csvs/ is in the same directory as this script")
        sys.exit(1)

    # List available files
    k_files = sorted([f for f in os.listdir(EXTRACT_DIR) if 'Part_K' in f])
    m_files = sorted([f for f in os.listdir(EXTRACT_DIR) if 'Part_M' in f])
    print(f"\n  Part K files found: {len(k_files)}")
    print(f"  Part M files found: {len(m_files)}")

    # Load panel
    print(f"\n  Loading panel: {PANEL_FILE}")
    panel = pd.read_csv(PANEL_FILE)
    panel['log1p_total_repairs'] = panel['log1p_total_repairs'].fillna(0)
    if 'log1p_ext_corrosion' in panel.columns:
        nonnan = panel['log1p_ext_corrosion'].dropna()
        if nonnan.nunique() <= 1:
            panel.drop(columns=['log1p_ext_corrosion'], inplace=True)
    print(f"  Panel: {len(panel):,} rows, {panel['event'].sum():.0f} events")

    # Phase 1: Part K
    k_features = load_part_k_all(EXTRACT_DIR, YEARS)

    # Phase 2: Part M
    m_raw = load_part_m_all(EXTRACT_DIR, YEARS)

    # Phase 3: Engineer features
    m_features = engineer_m_features(m_raw)

    # Phase 4: Join
    panel, new_features = join_to_panel(panel, k_features, m_features)

    # Phase 5: Diagnostic
    results, used_features = run_diagnostic(panel, new_features)

    # Phase 6: Plot
    plot_enrichment_diagnostic(results, panel, used_features, OUTPUT_DIR)

    # Phase 7: Save
    print(f"\n{'='*70}")
    print("PHASE 7: SAVING OUTPUTS")
    print("=" * 70)
    save_outputs(panel, results, used_features, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':

    main()
