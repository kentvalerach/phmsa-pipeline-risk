"""
ROSEN P4 — EXTERNAL DATA ENRICHMENT SPRINT (API VERSION)
=========================================================
Queries real external APIs and processes local PHMSA data:
  1. USDA SSURGO — Soil corrosivity for steel (REST API)
  2. USGS Earthquakes — Seismic events by state-year (REST API)
  3. PHMSA Parts A-D — Cathodic protection coverage (local CSV)

Joins to existing panel and tests AUC improvement.

Usage:
    cd C:\Phmsa\annual_gt
    python external_enrichment_api.py

Requirements:
    pip install requests pandas numpy scikit-learn matplotlib

Author: Kent (ROSEN Project 4)
Date:   2026-02-05
"""

import os
import sys
import json
import warnings
from datetime import datetime
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PANEL_FILE = os.path.join(BASE_DIR, 'survival_panel_15yr_enriched.csv')
if not os.path.exists(PANEL_FILE):
    PANEL_FILE = os.path.join(BASE_DIR, 'survival_panel_15yr_final.csv')
EXTRACT_DIR = os.path.join(BASE_DIR, 'extracted_csvs')
OUTPUT_DIR = BASE_DIR

YEARS = list(range(2010, 2025))
TRAIN_END = 2019
TEST_START = 2020

# State mappings
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
    'PR': 'PUERTO RICO', 'VI': 'VIRGIN ISLANDS', 'GU': 'GUAM',
}
STATE_TO_ABBREV = {v: k for k, v in STATE_ABBREV.items()}

# State names that appear in USGS place strings
STATE_NAMES_USGS = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY',
}


# ============================================================
# PHASE 1: SSURGO SOIL CORROSIVITY (API)
# ============================================================

def query_ssurgo_soil_corrosivity():
    """Query USDA Soil Data Access for steel corrosivity by state."""
    print("=" * 70)
    print("PHASE 1: SSURGO SOIL CORROSIVITY (API)")
    print("=" * 70)
    
    url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
    
    # Query aggregates corrosion of steel ratings by state
    # corsteel values: 'High', 'Moderate', 'Low', NULL
    query = """
    SELECT 
        LEFT(l.areasymbol, 2) as state_abbrev,
        SUM(CASE WHEN c.corsteel = 'High' THEN c.comppct_r * mu.muacres / 100.0 ELSE 0 END) as high_acres,
        SUM(CASE WHEN c.corsteel = 'Moderate' THEN c.comppct_r * mu.muacres / 100.0 ELSE 0 END) as mod_acres,
        SUM(CASE WHEN c.corsteel = 'Low' THEN c.comppct_r * mu.muacres / 100.0 ELSE 0 END) as low_acres,
        SUM(c.comppct_r * mu.muacres / 100.0) as total_acres
    FROM legend l
    INNER JOIN mapunit mu ON mu.lkey = l.lkey
    INNER JOIN component c ON c.mukey = mu.mukey
    WHERE c.corsteel IS NOT NULL
      AND c.comppct_r > 0
      AND mu.muacres > 0
      AND LEN(l.areasymbol) >= 2
    GROUP BY LEFT(l.areasymbol, 2)
    HAVING SUM(c.comppct_r * mu.muacres / 100.0) > 0
    ORDER BY LEFT(l.areasymbol, 2)
    """
    
    payload = {"query": query, "format": "JSON"}
    
    try:
        print("  Connecting to USDA Soil Data Access API...")
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code != 200:
            print(f"  ERROR: API returned status {response.status_code}")
            return None
        
        data = response.json()
        
        if 'Table' not in data:
            print(f"  ERROR: Unexpected response format: {list(data.keys())}")
            return None
        
        rows = data['Table']
        print(f"  Retrieved {len(rows)} state records from SSURGO")
        
        records = []
        for row in rows:
            state = str(row[0]).strip().upper() if row[0] else None
            if not state or len(state) != 2:
                continue
            
            high = float(row[1] or 0)
            mod = float(row[2] or 0)
            low = float(row[3] or 0)
            total = float(row[4] or 0)
            
            if total > 0:
                records.append({
                    'state_abbrev': state,
                    'pct_high_corr': high / total,
                    'pct_mod_corr': mod / total,
                    'pct_low_corr': low / total,
                    'soil_corr_index': (high * 1.0 + mod * 0.5 + low * 0.0) / total,
                })
        
        df = pd.DataFrame(records)
        print(f"  Processed {len(df)} states with soil corrosivity")
        
        # Show results
        df_sorted = df.sort_values('soil_corr_index', ascending=False)
        print(f"\n  TOP 10 HIGH CORROSIVITY STATES:")
        for _, r in df_sorted.head(10).iterrows():
            print(f"    {r['state_abbrev']}: index={r['soil_corr_index']:.3f} "
                  f"(H={r['pct_high_corr']:.0%} M={r['pct_mod_corr']:.0%} L={r['pct_low_corr']:.0%})")
        
        # Save to CSV for future use
        df.to_csv(os.path.join(OUTPUT_DIR, 'ssurgo_soil_corrosivity.csv'), index=False)
        print(f"\n  Saved: ssurgo_soil_corrosivity.csv")
        
        return df
        
    except requests.exceptions.Timeout:
        print("  ERROR: API request timed out (180s)")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ============================================================
# PHASE 2: USGS EARTHQUAKE DATA (API)
# ============================================================

def query_usgs_earthquakes():
    """Query USGS earthquake catalog for seismic events by state-year."""
    print(f"\n{'='*70}")
    print("PHASE 2: USGS EARTHQUAKE SEISMICITY (API)")
    print("=" * 70)
    
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    all_records = []
    
    for year in YEARS:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        
        params = {
            'format': 'geojson',
            'starttime': start,
            'endtime': end,
            'minmagnitude': 3.0,
            'maxlatitude': 50,
            'minlatitude': 24,
            'maxlongitude': -65,
            'minlongitude': -125,
            'limit': 20000,
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=60)
            if response.status_code != 200:
                print(f"  {year}: API error {response.status_code}")
                continue
            
            data = response.json()
            features = data.get('features', [])
            
            # Aggregate by state
            state_data = defaultdict(lambda: {'count': 0, 'max_mag': 0, 'energy': 0})
            
            for feat in features:
                props = feat.get('properties', {})
                place = props.get('place', '') or ''
                mag = props.get('mag', 0) or 0
                
                # Parse state from place
                state_found = None
                for state_name, abbrev in STATE_NAMES_USGS.items():
                    if state_name in place:
                        state_found = abbrev
                        break
                
                if state_found and mag > 0:
                    state_data[state_found]['count'] += 1
                    state_data[state_found]['max_mag'] = max(state_data[state_found]['max_mag'], mag)
                    state_data[state_found]['energy'] += 10 ** (1.5 * mag)
            
            n_states = len(state_data)
            print(f"  {year}: {len(features):,} earthquakes M≥3.0 in {n_states} states")
            
            for state, vals in state_data.items():
                all_records.append({
                    'state_abbrev': state,
                    'year': year,
                    'earthquake_count': vals['count'],
                    'max_magnitude': vals['max_mag'],
                    'log_seismic_energy': np.log1p(vals['energy']),
                })
            
            time.sleep(0.2)  # Be nice to USGS
            
        except Exception as e:
            print(f"  {year}: ERROR - {e}")
            continue
    
    if not all_records:
        print("  No earthquake data retrieved")
        return None
    
    df = pd.DataFrame(all_records)
    print(f"\n  Total: {len(df)} state-year records")
    
    # Summary
    state_summary = df.groupby('state_abbrev').agg({
        'earthquake_count': 'sum',
        'max_magnitude': 'max',
    }).sort_values('earthquake_count', ascending=False)
    
    print(f"\n  TOP 10 SEISMICALLY ACTIVE STATES (2010-2024):")
    for state, row in state_summary.head(10).iterrows():
        print(f"    {state}: {row['earthquake_count']:,} events, max M{row['max_magnitude']:.1f}")
    
    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'usgs_earthquakes.csv'), index=False)
    print(f"\n  Saved: usgs_earthquakes.csv")
    
    return df


# ============================================================
# PHASE 3: PHMSA PARTS A-D (CATHODIC PROTECTION)
# ============================================================

def load_parts_ad_cp(base_dir, years):
    """Load Parts A-D from annual reports for cathodic protection coverage."""
    print(f"\n{'='*70}")
    print("PHASE 3: PHMSA PARTS A-D (CATHODIC PROTECTION)")
    print("=" * 70)
    
    all_records = []
    
    # Try different file naming patterns
    patterns = [
        ('GT AR {year} Part A to D.csv', base_dir),
        ('GT_AR_{year}_Part_A_to_D.csv', os.path.join(base_dir, 'extracted_csvs')),
    ]
    
    for year in years:
        filepath = None
        for pattern, directory in patterns:
            candidate = os.path.join(directory, pattern.format(year=year))
            if os.path.exists(candidate):
                filepath = candidate
                break
        
        if not filepath:
            continue
        
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            
            # Print column structure for first year
            if year == years[0]:
                cp_cols = [c for c in df.columns if 'CP' in c.upper() or 
                           'PROTECT' in c.upper() or 'BARE' in c.upper() or
                           'COAT' in c.upper() or 'PARTD' in c.upper() or
                           'PARTC' in c.upper()]
                print(f"\n  Part A-D CP-related columns found:")
                for c in sorted(cp_cols)[:20]:
                    print(f"    {c}")
                if len(cp_cols) > 20:
                    print(f"    ... and {len(cp_cols)-20} more")
            
            for _, row in df.iterrows():
                oid = row.get('OPERATOR_ID')
                state = str(row.get('STATE_NAME', '')).strip().upper()
                if pd.isna(oid) or not state:
                    continue
                
                # Try to find CP columns - different naming across years
                # Pattern 1: PARTD columns
                coated_cp = 0
                bare_cp = 0
                unprotected = 0
                
                for col in df.columns:
                    col_up = col.upper()
                    val = float(row.get(col, 0) or 0)
                    
                    # Steel with coating and CP
                    if 'STLCPCOAT' in col_up or 'STEELCPCOAT' in col_up or 'CPCOATED' in col_up:
                        coated_cp += val
                    # Steel bare with CP
                    elif 'STLCPBARE' in col_up or 'STEELCPBARE' in col_up or 'BARECP' in col_up:
                        bare_cp += val
                    # Steel unprotected (no CP)
                    elif 'STLNOCP' in col_up or 'STEELNOCP' in col_up or 'NOCP' in col_up or 'UNPROT' in col_up:
                        unprotected += val
                
                # Alternative: look for PARTC total miles by material
                # PARTCTOTALONSHORE might have total, then we need CP breakdown
                
                total = coated_cp + bare_cp + unprotected
                
                if total > 0:
                    all_records.append({
                        'operator_id': int(oid),
                        'state': state,
                        'year': year,
                        'miles_coated_cp': coated_cp,
                        'miles_bare_cp': bare_cp,
                        'miles_unprotected': unprotected,
                        'pct_cp_protected': (coated_cp + bare_cp) / total,
                        'pct_unprotected': unprotected / total,
                    })
            
            n_year = len([r for r in all_records if r['year'] == year])
            print(f"  {year}: {n_year} operator-state records with CP data")
            
        except Exception as e:
            print(f"  {year}: ERROR - {e}")
            continue
    
    if not all_records:
        print("\n  No Part A-D CP data extracted")
        print("  Trying alternative column detection...")
        return try_alternative_cp_extraction(base_dir, years)
    
    df = pd.DataFrame(all_records)
    
    # Aggregate duplicates
    df = df.groupby(['operator_id', 'state', 'year'], as_index=False).agg({
        'miles_coated_cp': 'sum',
        'miles_bare_cp': 'sum',
        'miles_unprotected': 'sum',
        'pct_cp_protected': 'mean',
        'pct_unprotected': 'mean',
    })
    
    print(f"\n  Total CP records: {len(df)}")
    print(f"  Mean CP protection: {df['pct_cp_protected'].mean():.1%}")
    print(f"  Mean unprotected: {df['pct_unprotected'].mean():.1%}")
    
    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'phmsa_cp_coverage.csv'), index=False)
    print(f"  Saved: phmsa_cp_coverage.csv")
    
    return df


def try_alternative_cp_extraction(base_dir, years):
    """Try alternative column patterns for CP extraction."""
    print("\n  Attempting alternative CP column detection...")
    
    # Read one file and print ALL columns for debugging
    for year in [2024, 2023, 2022]:
        for pattern, directory in [
            ('GT AR {year} Part A to D.csv', base_dir),
            ('GT_AR_{year}_Part_A_to_D.csv', os.path.join(base_dir, 'extracted_csvs')),
        ]:
            filepath = os.path.join(directory, pattern.format(year=year))
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, encoding='latin-1', nrows=5)
                print(f"\n  All columns in {os.path.basename(filepath)}:")
                for i, c in enumerate(df.columns):
                    print(f"    {i:3d}: {c}")
                return None
    
    return None


# ============================================================
# PHASE 4: JOIN TO PANEL AND TEST
# ============================================================

def join_and_test(panel, ssurgo_df, seismic_df, cp_df):
    """Join external features to panel and test AUC improvement."""
    print(f"\n{'='*70}")
    print("PHASE 4: JOIN AND TEST AUC")
    print("=" * 70)
    
    n_orig = len(panel)
    
    # Standardize state column
    if 'state' in panel.columns:
        panel['state_upper'] = panel['state'].str.strip().str.upper()
    else:
        print("  ERROR: No state column in panel")
        return None
    
    panel['state_abbrev'] = panel['state_upper'].map(STATE_TO_ABBREV)
    
    new_features = []
    
    # --- Join SSURGO ---
    if ssurgo_df is not None and len(ssurgo_df) > 0:
        panel = panel.merge(
            ssurgo_df[['state_abbrev', 'soil_corr_index', 'pct_high_corr']],
            on='state_abbrev', how='left'
        )
        panel['soil_corr_index'] = panel['soil_corr_index'].fillna(
            panel['soil_corr_index'].median() if panel['soil_corr_index'].notna().any() else 0.5
        )
        panel['pct_high_corr'] = panel['pct_high_corr'].fillna(0)
        new_features.extend(['soil_corr_index', 'pct_high_corr'])
        matched = panel['soil_corr_index'].notna().sum()
        print(f"  SSURGO joined: {matched:,}/{n_orig:,} ({matched/n_orig*100:.1f}%)")
    else:
        print("  SSURGO: No data to join")
    
    # --- Join Seismic ---
    if seismic_df is not None and len(seismic_df) > 0:
        panel = panel.merge(
            seismic_df[['state_abbrev', 'year', 'earthquake_count', 'log_seismic_energy']],
            on=['state_abbrev', 'year'], how='left'
        )
        panel['earthquake_count'] = panel['earthquake_count'].fillna(0)
        panel['log_seismic_energy'] = panel['log_seismic_energy'].fillna(0)
        new_features.extend(['earthquake_count', 'log_seismic_energy'])
        matched = (panel['earthquake_count'] > 0).sum()
        print(f"  Seismic joined: {matched:,}/{n_orig:,} ({matched/n_orig*100:.1f}%) with events")
    else:
        print("  Seismic: No data to join")
    
    # --- Join CP Coverage ---
    if cp_df is not None and len(cp_df) > 0:
        panel = panel.merge(
            cp_df[['operator_id', 'state', 'year', 'pct_cp_protected', 'pct_unprotected']],
            left_on=['operator_id', 'state_upper', 'year'],
            right_on=['operator_id', 'state', 'year'],
            how='left'
        )
        panel['pct_cp_protected'] = panel['pct_cp_protected'].fillna(
            panel['pct_cp_protected'].median() if panel['pct_cp_protected'].notna().any() else 0.95
        )
        panel['pct_unprotected'] = panel['pct_unprotected'].fillna(0)
        new_features.extend(['pct_cp_protected', 'pct_unprotected'])
        matched = panel['pct_cp_protected'].notna().sum()
        print(f"  CP Coverage joined: {matched:,}/{n_orig:,} ({matched/n_orig*100:.1f}%)")
    else:
        print("  CP Coverage: No data to join")
    
    if not new_features:
        print("\n  ERROR: No external features available")
        return None
    
    print(f"\n  External features: {new_features}")
    
    # --- Run AUC comparison ---
    panel['log_miles'] = np.log1p(panel['miles_at_risk'])
    
    y = panel['event'].values
    train = (panel['year'] <= TRAIN_END).values
    test = (panel['year'] >= TEST_START).values
    y_test = y[test]
    
    if y_test.sum() < 10:
        print("  ERROR: Too few events in test set")
        return None
    
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
    
    # Model 2: + external
    ext_available = [c for c in new_features if c in panel.columns]
    X_ext = np.column_stack([X_base, panel[ext_available].fillna(0).values])
    lr2 = LogisticRegression(max_iter=1000, C=0.5)
    lr2.fit(X_ext[train], y[train])
    auc2 = roc_auc_score(y_test, lr2.predict_proba(X_ext[test])[:, 1])
    results.append((f'+ External ({len(ext_available)} feat)', auc2, X_ext.shape[1]))
    
    # Print
    print(f"\n  {'Model':<35s} {'Nf':>4s}  {'AUC':>7s}  {'Δ vs Base':>9s}")
    print(f"  {'-'*35} {'-'*4}  {'-'*7}  {'-'*9}")
    for name, auc, nf in results:
        delta = (auc - auc1) * 100
        print(f"  {name:<35s} {nf:>4d}  {auc:.4f}  {delta:+.2f}pp")
    
    delta = (auc2 - auc1) * 100
    print(f"\n  {'='*50}")
    print(f"  EXTERNAL DATA GAIN: {delta:+.2f}pp")
    print(f"  {'='*50}")
    
    if delta > 0.3:
        print(f"  ✓ POSITIVE SIGNAL — worth including")
    elif delta > 0.1:
        print(f"  ~ MARGINAL SIGNAL — consider including")
    elif delta < -0.1:
        print(f"  ✗ NEGATIVE IMPACT — features add noise")
    else:
        print(f"  ○ NEGLIGIBLE — no meaningful change")
    
    # Feature correlations
    print(f"\n  EXTERNAL FEATURE CORRELATIONS:")
    print(f"  {'Feature':<25s} {'r':>8s} {'partial_r':>10s} {'nz%':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*7}")
    
    for c in ext_available:
        r = panel[c].corr(panel['event'])
        # Partial correlation
        X_ctrl = panel[['log_miles']].values
        res_y = y - X_ctrl @ np.linalg.lstsq(X_ctrl, y, rcond=None)[0]
        feat_v = panel[c].fillna(0).values
        res_f = feat_v - X_ctrl @ np.linalg.lstsq(X_ctrl, feat_v, rcond=None)[0]
        pr = np.corrcoef(res_y, res_f)[0, 1] if np.std(res_f) > 0 else 0
        nz = (panel[c] > 0).sum() / len(panel) * 100
        print(f"  {c:<25s} {r:+.4f}   {pr:+.4f}    {nz:5.1f}%")
    
    return panel, results, ext_available


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ROSEN P4 — EXTERNAL DATA ENRICHMENT SPRINT (API)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Load panel
    print(f"\n  Loading panel: {PANEL_FILE}")
    if not os.path.exists(PANEL_FILE):
        print(f"  ERROR: Panel not found at {PANEL_FILE}")
        sys.exit(1)
    
    panel = pd.read_csv(PANEL_FILE)
    print(f"  Panel: {len(panel):,} rows, {panel['event'].sum():.0f} events")
    
    # Phase 1: SSURGO
    ssurgo_df = query_ssurgo_soil_corrosivity()
    
    # Phase 2: USGS
    seismic_df = query_usgs_earthquakes()
    
    # Phase 3: CP Coverage
    cp_df = load_parts_ad_cp(BASE_DIR, YEARS)
    
    # Phase 4: Join and test
    result = join_and_test(panel, ssurgo_df, seismic_df, cp_df)
    
    if result:
        panel_enriched, results, features = result
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, 'survival_panel_external_enriched.csv')
        panel_enriched.to_csv(out_path, index=False)
        print(f"\n  Saved enriched panel: {out_path}")
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print("=" * 70)
        print(f"  SSURGO soil corrosivity: {'✓' if ssurgo_df is not None else '✗'}")
        print(f"  USGS earthquakes: {'✓' if seismic_df is not None else '✗'}")
        print(f"  PHMSA CP coverage: {'✓' if cp_df is not None else '✗'}")
        print(f"  External features added: {len(features)}")
        print(f"  AUC improvement: {(results[-1][1] - results[1][1])*100:+.2f}pp")
    
    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':

    main()
