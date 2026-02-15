"""
ROSEN P4 — CP COVERAGE EXTRACTION FIX
======================================
Extracts cathodic protection coverage from Part D using correct column names.

Part D Column Structure:
  CPB = Cathodic Protection Bare steel
  CPC = Cathodic Protection Coated steel  
  CUB = Cathodically Unprotected Bare steel
  CUC = Cathodically Unprotected Coated steel

Usage:
    cd C:\Phmsa\annual_gt
    python extract_cp_and_test.py

Author: Kent (ROSEN Project 4)
Date:   2026-02-05
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_FILE = os.path.join(BASE_DIR, 'survival_panel_15yr_enriched.csv')
if not os.path.exists(PANEL_FILE):
    PANEL_FILE = os.path.join(BASE_DIR, 'survival_panel_15yr_final.csv')

YEARS = list(range(2010, 2025))
TRAIN_END = 2019
TEST_START = 2020

STATE_ABBREV = {
    'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS',
    'CA': 'CALIFORNIA', 'CO': 'COLORADO', 'CT': 'CONNECTICUT', 'DE': 'DELAWARE',
    'FL': 'FLORIDA', 'GA': 'GEORGIA', 'HI': 'HAWAII', 'ID': 'IDAHO',
    'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA', 'KS': 'KANSAS',
    'KY': 'KENTUCKY', 'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND',
    'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI',
    'MO': 'MISSOURI', 'MT': 'MONTANA', 'NE': 'NEBRASKA', 'NV': 'NEVADA',
    'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY', 'NM': 'NEW MEXICO', 'NY': 'NEW YORK',
    'NC': 'NORTH CAROLINA', 'ND': 'NORTH DAKOTA', 'OH': 'OHIO', 'OK': 'OKLAHOMA',
    'OR': 'OREGON', 'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA',
    'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE', 'TX': 'TEXAS', 'UT': 'UTAH',
    'VT': 'VERMONT', 'VA': 'VIRGINIA', 'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA',
    'WI': 'WISCONSIN', 'WY': 'WYOMING', 'DC': 'DISTRICT OF COLUMBIA',
}
STATE_TO_ABBREV = {v: k for k, v in STATE_ABBREV.items()}


def safe_float(x):
    try:
        return float(x) if pd.notna(x) else 0.0
    except:
        return 0.0


def load_cp_data(base_dir, years):
    """Load Part D data and extract CP coverage."""
    print("=" * 70)
    print("EXTRACTING CATHODIC PROTECTION FROM PART D")
    print("=" * 70)
    
    # Column mappings for CP extraction
    # Transmission totals
    CP_COLS_T = ['PARTDTCPBTOTAL', 'PARTDTCPCTOTAL']  # CP protected
    CU_COLS_T = ['PARTDTCUBTOTAL', 'PARTDTCUCTOTAL']  # Unprotected
    
    # Gathering totals
    CP_COLS_G = ['PARTDGCPBTOTAL', 'PARTDGCPCTOTAL']
    CU_COLS_G = ['PARTDGCUBTOTAL', 'PARTDGCUCTOTAL']
    
    # Total miles
    TOTAL_COL = 'PARTDTOTALMILES'
    
    # Alternative: use transmission-only totals
    TOTAL_T = 'PARTDTTOTAL'
    TOTAL_G = 'PARTDGTOTAL'
    
    all_records = []
    
    for year in years:
        # Try different file patterns
        patterns = [
            os.path.join(base_dir, f'GT AR {year} Part A to D.csv'),
            os.path.join(base_dir, 'extracted_csvs', f'GT_AR_{year}_Part_A_to_D.csv'),
        ]
        
        filepath = None
        for p in patterns:
            if os.path.exists(p):
                filepath = p
                break
        
        if not filepath:
            print(f"  {year}: File not found")
            continue
        
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            
            # Check which columns exist
            available = set(df.columns)
            
            for _, row in df.iterrows():
                oid = row.get('OPERATOR_ID')
                state = str(row.get('STATE_NAME', row.get('PARTA4STATE', ''))).strip().upper()
                
                # Try to get state from different columns
                if not state or len(state) < 2:
                    continue
                if len(state) == 2:
                    state = STATE_ABBREV.get(state, state)
                
                if pd.isna(oid):
                    continue
                
                # Sum CP protected miles
                cp_miles = 0.0
                for col in CP_COLS_T + CP_COLS_G:
                    if col in available:
                        cp_miles += safe_float(row.get(col, 0))
                
                # Sum unprotected miles
                cu_miles = 0.0
                for col in CU_COLS_T + CU_COLS_G:
                    if col in available:
                        cu_miles += safe_float(row.get(col, 0))
                
                # Get total miles
                total_miles = safe_float(row.get(TOTAL_COL, 0))
                if total_miles <= 0:
                    # Try sum of T + G totals
                    total_miles = safe_float(row.get(TOTAL_T, 0)) + safe_float(row.get(TOTAL_G, 0))
                
                # Only include if we have meaningful data
                steel_total = cp_miles + cu_miles
                if steel_total > 0:
                    all_records.append({
                        'operator_id': int(oid),
                        'state': state,
                        'year': year,
                        'miles_cp_protected': cp_miles,
                        'miles_unprotected': cu_miles,
                        'miles_steel_total': steel_total,
                        'miles_total': total_miles,
                        'pct_cp_protected': cp_miles / steel_total,
                        'pct_unprotected': cu_miles / steel_total,
                    })
            
            n_year = len([r for r in all_records if r['year'] == year])
            
            # Sample statistics for this year
            year_data = [r for r in all_records if r['year'] == year]
            if year_data:
                avg_cp = np.mean([r['pct_cp_protected'] for r in year_data])
                avg_cu = np.mean([r['pct_unprotected'] for r in year_data])
                print(f"  {year}: {n_year} records, avg CP={avg_cp:.1%}, avg unprotected={avg_cu:.1%}")
            else:
                print(f"  {year}: 0 records")
            
        except Exception as e:
            print(f"  {year}: ERROR - {e}")
            continue
    
    if not all_records:
        print("\n  ERROR: No CP data extracted")
        return None
    
    df = pd.DataFrame(all_records)
    
    # Aggregate duplicates (same operator-state-year from interstate/intrastate)
    df = df.groupby(['operator_id', 'state', 'year'], as_index=False).agg({
        'miles_cp_protected': 'sum',
        'miles_unprotected': 'sum',
        'miles_steel_total': 'sum',
        'miles_total': 'sum',
    })
    
    # Recalculate percentages after aggregation
    df['pct_cp_protected'] = df['miles_cp_protected'] / df['miles_steel_total'].replace(0, np.nan)
    df['pct_unprotected'] = df['miles_unprotected'] / df['miles_steel_total'].replace(0, np.nan)
    df = df.dropna(subset=['pct_cp_protected'])
    
    print(f"\n  Total: {len(df)} operator-state-year records")
    print(f"  Mean CP protection: {df['pct_cp_protected'].mean():.1%}")
    print(f"  Mean unprotected: {df['pct_unprotected'].mean():.1%}")
    print(f"  Operators with >10% unprotected: {(df['pct_unprotected'] > 0.1).sum()}")
    
    # Save
    df.to_csv(os.path.join(base_dir, 'phmsa_cp_coverage_fixed.csv'), index=False)
    print(f"\n  Saved: phmsa_cp_coverage_fixed.csv")
    
    return df


def test_cp_signal(panel, cp_df):
    """Join CP data to panel and test AUC."""
    print(f"\n{'='*70}")
    print("TESTING CP COVERAGE SIGNAL")
    print("=" * 70)
    
    n_orig = len(panel)
    
    # Standardize state
    panel['state_upper'] = panel['state'].str.strip().str.upper()
    
    # Join CP data
    cp_df = cp_df.rename(columns={'state': 'state_cp'})
    panel = panel.merge(
        cp_df[['operator_id', 'state_cp', 'year', 'pct_cp_protected', 'pct_unprotected']],
        left_on=['operator_id', 'state_upper', 'year'],
        right_on=['operator_id', 'state_cp', 'year'],
        how='left'
    )
    
    matched = panel['pct_cp_protected'].notna().sum()
    print(f"  CP data matched: {matched:,}/{n_orig:,} ({matched/n_orig*100:.1f}%)")
    
    # Fill missing with median (assume average protection)
    median_cp = panel['pct_cp_protected'].median()
    panel['pct_cp_protected'] = panel['pct_cp_protected'].fillna(median_cp)
    panel['pct_unprotected'] = panel['pct_unprotected'].fillna(1 - median_cp)
    
    print(f"  Median CP protection (fill value): {median_cp:.1%}")
    
    # Prepare for modeling
    panel['log_miles'] = np.log1p(panel['miles_at_risk'])
    
    y = panel['event'].values
    train = (panel['year'] <= TRAIN_END).values
    test = (panel['year'] >= TEST_START).values
    y_test = y[test]
    
    # Baseline features
    era_dum = pd.get_dummies(panel['era'], prefix='era', drop_first=True)
    base_cols = ['log_miles', 'age_at_obs', 'pct_small_diam', 'pct_large_diam', 
                 'pct_high_smys', 'pct_class1']
    for c in ['log1p_total_repairs', 'pct_low_smys', 'pct_high_class']:
        if c in panel.columns:
            base_cols.append(c)
    
    X_base = np.column_stack([panel[base_cols].fillna(0).values, era_dum.values])
    
    results = []
    
    # Model 0: log(miles) only
    X0 = panel[['log_miles']].values
    lr0 = LogisticRegression(max_iter=1000)
    lr0.fit(X0[train], y[train])
    auc0 = roc_auc_score(y_test, lr0.predict_proba(X0[test])[:, 1])
    results.append(('log(miles) only', auc0, 1))
    
    # Model 1: baseline
    lr1 = LogisticRegression(max_iter=1000)
    lr1.fit(X_base[train], y[train])
    auc1 = roc_auc_score(y_test, lr1.predict_proba(X_base[test])[:, 1])
    results.append((f'Baseline ({X_base.shape[1]} feat)', auc1, X_base.shape[1]))
    
    # Model 2: + CP features
    cp_features = ['pct_cp_protected', 'pct_unprotected']
    X_cp = np.column_stack([X_base, panel[cp_features].values])
    lr2 = LogisticRegression(max_iter=1000, C=0.5)
    lr2.fit(X_cp[train], y[train])
    auc2 = roc_auc_score(y_test, lr2.predict_proba(X_cp[test])[:, 1])
    results.append((f'+ CP coverage (2 feat)', auc2, X_cp.shape[1]))
    
    # Model 3: + unprotected only (the risk indicator)
    X_cu = np.column_stack([X_base, panel[['pct_unprotected']].values])
    lr3 = LogisticRegression(max_iter=1000, C=0.5)
    lr3.fit(X_cu[train], y[train])
    auc3 = roc_auc_score(y_test, lr3.predict_proba(X_cu[test])[:, 1])
    results.append(('+ pct_unprotected only', auc3, X_cu.shape[1]))
    
    # Print results
    print(f"\n  {'Model':<30s} {'Nf':>4s}  {'AUC':>7s}  {'Δ Base':>8s}")
    print(f"  {'-'*30} {'-'*4}  {'-'*7}  {'-'*8}")
    for name, auc, nf in results:
        delta = (auc - auc1) * 100
        print(f"  {name:<30s} {nf:>4d}  {auc:.4f}  {delta:+.2f}pp")
    
    delta = (auc2 - auc1) * 100
    print(f"\n  {'='*50}")
    print(f"  CP COVERAGE GAIN: {delta:+.2f}pp")
    print(f"  {'='*50}")
    
    # Correlations
    print(f"\n  FEATURE CORRELATIONS:")
    for c in cp_features:
        r = panel[c].corr(panel['event'])
        # Partial
        X_ctrl = panel[['log_miles']].values
        res_y = y - X_ctrl @ np.linalg.lstsq(X_ctrl, y, rcond=None)[0]
        feat_v = panel[c].values
        res_f = feat_v - X_ctrl @ np.linalg.lstsq(X_ctrl, feat_v, rcond=None)[0]
        pr = np.corrcoef(res_y, res_f)[0, 1]
        print(f"    {c:<20s}: r={r:+.4f}  partial_r={pr:+.4f}")
    
    # Check if unprotected correlates positively with events (as expected)
    r_unp = panel['pct_unprotected'].corr(panel['event'])
    if r_unp > 0:
        print(f"\n  ✓ pct_unprotected positively correlated with events (expected)")
    else:
        print(f"\n  ✗ pct_unprotected NEGATIVELY correlated — confounding?")
    
    return panel, results


def main():
    print("=" * 70)
    print("ROSEN P4 — CP COVERAGE EXTRACTION AND TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Load panel
    print(f"\n  Loading panel: {PANEL_FILE}")
    panel = pd.read_csv(PANEL_FILE)
    print(f"  Panel: {len(panel):,} rows, {panel['event'].sum():.0f} events")
    
    # Extract CP data
    cp_df = load_cp_data(BASE_DIR, YEARS)
    
    if cp_df is None:
        print("\n  FAILED: Could not extract CP data")
        sys.exit(1)
    
    # Test signal
    panel_enriched, results = test_cp_signal(panel, cp_df)
    
    # Save
    panel_enriched.to_csv(os.path.join(BASE_DIR, 'survival_panel_cp_enriched.csv'), index=False)
    print(f"\n  Saved: survival_panel_cp_enriched.csv")
    
    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':

    main()
