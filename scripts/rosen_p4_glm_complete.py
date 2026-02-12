"""
ROSEN Proyecto 4 -- Poisson GLM with PHMSA Annual Report Covariables
====================================================================
Integrates survival_panel_7yr.csv with Parts H (diameter), K (%SMYS x class), 
M (leak/repair history) from PHMSA GT Annual Reports.

Inputs:
  - survival_panel_7yr.csv         (34,226 rows: operator x state x year x decade)
  - GT_AR_YYYY_Part_H.csv          (diameter distribution by operator x state)
  - GT_AR_YYYY_Part_K.csv          (%SMYS x location class by operator x state)
  - GT_AR_YYYY_Part_M.csv          (leak/repair counts by operator x state)

Outputs:
  - glm_allcause_results.html      (NegBin model for all-cause incidents)
  - glm_corrosion_results.html     (Poisson/robust model for corrosion)
  - model_comparison.csv           (AIC/deviance comparison across all specs)
  - operator_risk_scores.csv       (predicted rates per operator x vintage)
  - diagnostics.txt                (overdispersion, residual analysis)

Usage:
  python rosen_p4_glm_complete.py [--data-dir PATH] [--part-dir PATH]

Author: Kent / Claude -- ROSEN P4 Pipeline Survival Analysis
Date: February 2026
"""

import os
import sys
import glob
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# ============================================================================
#  CONFIGURATION
# ============================================================================

# Default paths (override with --data-dir, --part-dir)
DEFAULT_DATA_DIR = '.'           # Where survival_panel_7yr.csv lives
DEFAULT_PART_DIR = '.'           # Where GT_AR_YYYY_Part_X.csv files live
SURVIVAL_PANEL = 'survival_panel_7yr.csv'

# Technology era definitions (from Phase 1 analysis)
ERA_MAP = {
    'pre1940':  'bare',
    'pre_1940': 'bare',
    '1940_49':  'bare',
    '1950_59':  'early_coat',   # REFERENCE category
    '1960_69':  'early_coat',
    '1970_79':  'coal_tar',
    '1980_89':  'improved',
    '1990_99':  'improved',
    '2000_09':  'modern',
    '2010_19':  'modern',
    '2020_29':  'modern',
}

# Feature engineering parameters
AGE_CENTER = 32.2  # Mean age from Phase 1 analysis


def find_part_dir(base_dir):
    """
    Auto-detect where Part CSV files are located.
    Searches: base_dir itself, then common subdirectories.
    """
    candidates = [
        base_dir,
        os.path.join(base_dir, 'extracted_csvs'),
        os.path.join(base_dir, 'annual_gt', 'extracted_csvs'),
        os.path.join(base_dir, 'annual_gt'),
    ]
    
    for cand in candidates:
        if os.path.isdir(cand):
            pattern = os.path.join(cand, 'GT_AR_*_Part_*.csv')
            files = glob.glob(pattern)
            if files:
                print(f"  Auto-detected Part files in: {cand} ({len(files)} files)")
                return cand
    
    return base_dir  # fallback


# ============================================================================
#  1. SURVIVAL PANEL PARSER
# ============================================================================

def parse_survival_panel(filepath):
    """
    Parse survival_panel_7yr.csv which has peculiar quoting:
    entire rows wrapped in quotes, operator names with commas use escaped quotes ("").
    Pattern: 288,""ALGONQUIN GAS TRANSMISSION, L.L.C."",STATE,...
    """
    print(f"[1/6] Loading survival panel: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse header
    header_line = lines[0].strip().strip('"')
    columns = header_line.split(',')
    n_cols = len(columns)
    
    def parse_row(raw):
        """Parse row handling ""-quoted fields with embedded commas."""
        result = []
        i = 0
        n = len(raw)
        while i < n:
            if raw[i:i+2] == '""':
                # Start of ""-quoted field: find closing ""
                end = raw.find('""', i + 2)
                if end == -1:
                    result.append(raw[i+2:])
                    break
                else:
                    result.append(raw[i+2:end])
                    i = end + 2
                    if i < n and raw[i] == ',':
                        i += 1
            elif raw[i] == ',':
                result.append('')
                i += 1
            else:
                end = raw.find(',', i)
                if end == -1:
                    result.append(raw[i:])
                    break
                else:
                    result.append(raw[i:end])
                    i = end + 1
        return result
    
    # Parse data rows
    records = []
    parse_errors = 0
    
    for i, line in enumerate(lines[1:], start=2):
        raw = line.strip().strip('"')
        if not raw:
            continue
        
        fields = parse_row(raw)
        
        if len(fields) >= n_cols:
            records.append(fields[:n_cols])
        else:
            parse_errors += 1
    
    df = pd.DataFrame(records, columns=columns)
    
    # Type conversions
    numeric_cols = ['operator_id', 'year', 'install_midpoint', 'age_at_obs',
                    'miles_at_risk', 'n_incidents', 'n_corrosion', 'n_material',
                    'event', 'event_corrosion']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['operator_id'] = df['operator_id'].astype(int)
    
    print(f"  Loaded: {len(df):,} rows, {df['operator_id'].nunique()} operators")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Incidents: {df['n_incidents'].sum():.0f} total, {df['n_corrosion'].sum():.0f} corrosion")
    print(f"  Exposure: {df['miles_at_risk'].sum():,.0f} mile-years")
    if parse_errors:
        print(f"  [!] Parse errors: {parse_errors} rows skipped")
    
    return df


# ============================================================================
#  2. PART H LOADER -- Diameter Distribution
# ============================================================================

def load_part_h(part_dir, years=None):
    """
    Load and aggregate Part H files (pipe diameter by miles).
    
    Part H reports onshore/offshore miles by nominal diameter (4" to 58"+).
    Aggregates to OPERATOR_ID level per year, computing:
      - pct_small_diam:  % miles <= 8" (higher = more distribution-style)
      - pct_large_diam:  % miles >= 24" (higher = more transmission-style)
      - avg_diameter_weighted: weighted average diameter (risk proxy)
    """
    print(f"\n[2/6] Loading Part H (diameter) from: {part_dir}")
    
    pattern = os.path.join(part_dir, 'GT_AR_*_Part_H.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("  [!] No Part H files found! Returning empty DataFrame.")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'pct_small_diam', 'pct_large_diam',
                                     'avg_diameter_weighted', 'total_miles_h'])
    
    # Diameter midpoints for weighted average (inches)
    diam_map = {
        'PARTHON4LESS': 3, 'PARTHON6': 6, 'PARTHON8': 8, 'PARTHON10': 10,
        'PARTHON12': 12, 'PARTHON14': 14, 'PARTHON16': 16, 'PARTHON18': 18,
        'PARTHON20': 20, 'PARTHON22': 22, 'PARTHON24': 24, 'PARTHON26': 26,
        'PARTHON28': 28, 'PARTHON30': 30, 'PARTHON32': 32, 'PARTHON34': 34,
        'PARTHON36': 36, 'PARTHON38': 38, 'PARTHON40': 40, 'PARTHON42': 42,
        'PARTHON44': 44, 'PARTHON46': 46, 'PARTHON48': 48, 'PARTHON52': 52,
        'PARTHON56': 56, 'PARTHON58OVER': 60,
    }
    small_cols = ['PARTHON4LESS', 'PARTHON6', 'PARTHON8']
    large_cols = [c for c in diam_map if int(''.join(filter(str.isdigit, c.replace('PARTHON','') or '0')) or 0) >= 24
                  or c == 'PARTHON58OVER']
    # Explicit large diameter columns
    large_cols = ['PARTHON24', 'PARTHON26', 'PARTHON28', 'PARTHON30', 'PARTHON32',
                  'PARTHON34', 'PARTHON36', 'PARTHON38', 'PARTHON40', 'PARTHON42',
                  'PARTHON44', 'PARTHON46', 'PARTHON48', 'PARTHON52', 'PARTHON56',
                  'PARTHON58OVER']
    
    all_dfs = []
    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract year from filename: GT_AR_YYYY_Part_H.csv
        try:
            yr = int(fname.split('_')[2])
        except (IndexError, ValueError):
            continue
        
        if years and yr not in years:
            continue
        
        df = pd.read_csv(fpath)
        
        # Convert diameter columns to numeric
        for col in list(diam_map.keys()) + ['PARTHONTOTAL']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Aggregate to operator level (sum across states)
        agg_dict = {}
        for col in diam_map:
            if col in df.columns:
                agg_dict[col] = (col, 'sum')
        agg_dict['total_miles_h'] = ('PARTHONTOTAL', 'sum')
        
        op_agg = df.groupby('OPERATOR_ID').agg(**agg_dict).reset_index()
        op_agg['report_year'] = yr
        all_dfs.append(op_agg)
    
    if not all_dfs:
        print("  [!] No valid Part H data loaded!")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'pct_small_diam', 'pct_large_diam',
                                     'avg_diameter_weighted', 'total_miles_h'])
    
    result = pd.concat(all_dfs, ignore_index=True)
    
    # Compute features
    existing_small = [c for c in small_cols if c in result.columns]
    existing_large = [c for c in large_cols if c in result.columns]
    
    result['miles_small'] = result[existing_small].sum(axis=1)
    result['miles_large'] = result[existing_large].sum(axis=1)
    total_safe = result['total_miles_h'].replace(0, np.nan)
    
    result['pct_small_diam'] = result['miles_small'] / total_safe
    result['pct_large_diam'] = result['miles_large'] / total_safe
    
    # Weighted average diameter
    existing_diam = {c: m for c, m in diam_map.items() if c in result.columns}
    weighted_sum = sum(result[c] * m for c, m in existing_diam.items())
    total_diam_miles = sum(result[c] for c in existing_diam)
    result['avg_diameter_weighted'] = weighted_sum / total_diam_miles.replace(0, np.nan)
    
    # Keep only features
    keep_cols = ['OPERATOR_ID', 'report_year', 'pct_small_diam', 'pct_large_diam',
                 'avg_diameter_weighted', 'total_miles_h']
    result = result[keep_cols].copy()
    
    years_loaded = sorted(result['report_year'].unique())
    print(f"  Loaded {len(files)} files, years: {years_loaded}")
    print(f"  {len(result)} operator-year records")
    print(f"  Avg diameter features: small={result['pct_small_diam'].mean():.1%}, "
          f"large={result['pct_large_diam'].mean():.1%}, "
          f"avg_diam={result['avg_diameter_weighted'].mean():.1f}\"")
    
    return result


# ============================================================================
#  3. PART K LOADER -- %SMYS x Location Class
# ============================================================================

def load_part_k(part_dir, years=None):
    """
    Load and aggregate Part K files (%SMYS x location class).
    
    Part K reports onshore miles cross-tabulated by:
      - %SMYS bands: <20, 20-29, 30-40, 41-50, 51-60, 61-72, 73-80, >80, unknown, non-steel
      - Location class: 1 (rural), 2 (suburban), 3 (urban), 4 (near buildings)
    
    Computes:
      - pct_class1:     % miles in Class 1 (rural, lower consequence)
      - pct_class3_4:   % miles in Class 3+4 (urban, higher consequence)
      - pct_high_smys:  % miles at >60% SMYS (higher stress, higher risk)
      - pct_low_smys:   % miles at <30% SMYS (lower stress, lower risk)
    """
    print(f"\n[3/6] Loading Part K (%SMYS x class) from: {part_dir}")
    
    pattern = os.path.join(part_dir, 'GT_AR_*_Part_K.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("  [!] No Part K files found! Returning empty DataFrame.")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'pct_class1', 'pct_class3_4',
                                     'pct_high_smys', 'pct_low_smys', 'total_miles_k'])
    
    # SMYS band columns (onshore totals by band)
    high_smys_tot = ['PARTK6172TOT', 'PARTK7380TOT', 'PARTK80MORETOT']
    low_smys_tot = ['PARTK20LESSTOT', 'PARTK2029TOT']
    
    all_dfs = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            yr = int(fname.split('_')[2])
        except (IndexError, ValueError):
            continue
        
        if years and yr not in years:
            continue
        
        df = pd.read_csv(fpath)
        
        # Convert all numeric columns
        num_cols = [c for c in df.columns if c.startswith('PARTK')]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Build aggregation dict
        agg_dict = {
            'total_miles_k': ('PARTKONTOTAL', 'sum'),
            'class1_miles': ('PARTKONC1TOT', 'sum'),
            'class2_miles': ('PARTKONC2TOT', 'sum'),
            'class3_miles': ('PARTKONC3TOT', 'sum'),
            'class4_miles': ('PARTKONC4TOT', 'sum'),
        }
        for col in high_smys_tot + low_smys_tot:
            if col in df.columns:
                agg_dict[col] = (col, 'sum')
        
        op_agg = df.groupby('OPERATOR_ID').agg(**agg_dict).reset_index()
        op_agg['report_year'] = yr
        all_dfs.append(op_agg)
    
    if not all_dfs:
        print("  [!] No valid Part K data loaded!")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'pct_class1', 'pct_class3_4',
                                     'pct_high_smys', 'pct_low_smys', 'total_miles_k'])
    
    result = pd.concat(all_dfs, ignore_index=True)
    
    total_safe = result['total_miles_k'].replace(0, np.nan)
    result['pct_class1'] = result['class1_miles'] / total_safe
    result['pct_class3_4'] = (result['class3_miles'] + result['class4_miles']) / total_safe
    
    existing_high = [c for c in high_smys_tot if c in result.columns]
    existing_low = [c for c in low_smys_tot if c in result.columns]
    result['pct_high_smys'] = result[existing_high].sum(axis=1) / total_safe
    result['pct_low_smys'] = result[existing_low].sum(axis=1) / total_safe
    
    keep_cols = ['OPERATOR_ID', 'report_year', 'pct_class1', 'pct_class3_4',
                 'pct_high_smys', 'pct_low_smys', 'total_miles_k']
    result = result[keep_cols].copy()
    
    years_loaded = sorted(result['report_year'].unique())
    print(f"  Loaded {len(files)} files, years: {years_loaded}")
    print(f"  {len(result)} operator-year records")
    print(f"  Class features: class1={result['pct_class1'].mean():.1%}, "
          f"class3+4={result['pct_class3_4'].mean():.1%}")
    print(f"  SMYS features: high(>60%)={result['pct_high_smys'].mean():.1%}, "
          f"low(<30%)={result['pct_low_smys'].mean():.1%}")
    
    return result


# ============================================================================
#  4. PART M LOADER -- Leak/Repair History
# ============================================================================

def load_part_m(part_dir, years=None):
    """
    Load and aggregate Part M files (leak and repair counts).
    
    Part M reports repairs by cause:
      - EC = External Corrosion
      - IC = Internal Corrosion  
      - SC = Stress Corrosion Cracking
      - CM/CC = Construction/Material/Coupling defects
      - CE/CIO/CED/CPD/CV/CNF/COOFD/CO = Other causes
    
    Plus: Section 2 = Leaks & Service Ruptures, Section 3 = Repairs to L&SR
    
    Computes:
      - n_ext_corrosion:   total external corrosion repairs
      - n_int_corrosion:   total internal corrosion repairs
      - n_all_corrosion:   EC + IC + SCC combined
      - n_total_repairs:   all repair types combined
      - n_leaks:           leaks and service ruptures
      - has_corrosion_history: binary flag (any EC/IC/SCC > 0)
      - repair_intensity:  total repairs per mile (needs miles from Part H or K)
    """
    print(f"\n[4/6] Loading Part M (leaks/repairs) from: {part_dir}")
    
    pattern = os.path.join(part_dir, 'GT_AR_*_Part_M.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("  [!] No Part M files found! Returning empty DataFrame.")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'n_ext_corrosion', 'n_int_corrosion',
                                     'n_all_corrosion', 'n_total_repairs',
                                     'n_leaks', 'has_corrosion_history'])
    
    # Column groups for each repair type
    ec_cols = ['PARTMTCECONHCA', 'PARTMTCECONNHCA', 'PARTMTCECOFFHCA', 'PARTMTCECOFFNHCA']
    ic_cols = ['PARTMTCICONHCA', 'PARTMTCICONNHCA', 'PARTMTCICOFFHCA', 'PARTMTCICOFFNHCA']
    scc_cols = ['PARTMTCSCONHCA', 'PARTMTCSCONNHCA', 'PARTMTCSCOFFHCA', 'PARTMTCSCOFFNHCA']
    total_cols = ['PARTM1TONHCATOTAL', 'PARTM1TONNHCATOTAL', 
                  'PARTM1TOFFHCATOTAL', 'PARTM1TOFFNHCATOTAL']
    
    all_dfs = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            yr = int(fname.split('_')[2])
        except (IndexError, ValueError):
            continue
        
        if years and yr not in years:
            continue
        
        df = pd.read_csv(fpath)
        
        # Convert all PARTM columns to numeric
        partm_cols = [c for c in df.columns if c.startswith('PARTM')]
        for col in partm_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Compute per-row sums
        existing_ec = [c for c in ec_cols if c in df.columns]
        existing_ic = [c for c in ic_cols if c in df.columns]
        existing_scc = [c for c in scc_cols if c in df.columns]
        existing_total = [c for c in total_cols if c in df.columns]
        
        df['_n_ec'] = df[existing_ec].sum(axis=1) if existing_ec else 0
        df['_n_ic'] = df[existing_ic].sum(axis=1) if existing_ic else 0
        df['_n_scc'] = df[existing_scc].sum(axis=1) if existing_scc else 0
        df['_n_total'] = df[existing_total].sum(axis=1) if existing_total else 0
        df['_n_leaks'] = df['PARTM2TLSR'] if 'PARTM2TLSR' in df.columns else 0
        
        # Aggregate to operator level
        op_agg = df.groupby('OPERATOR_ID').agg(
            n_ext_corrosion=('_n_ec', 'sum'),
            n_int_corrosion=('_n_ic', 'sum'),
            n_scc=('_n_scc', 'sum'),
            n_total_repairs=('_n_total', 'sum'),
            n_leaks=('_n_leaks', 'sum'),
        ).reset_index()
        
        op_agg['report_year'] = yr
        op_agg['n_all_corrosion'] = (op_agg['n_ext_corrosion'] + 
                                      op_agg['n_int_corrosion'] + 
                                      op_agg['n_scc'])
        op_agg['has_corrosion_history'] = (op_agg['n_all_corrosion'] > 0).astype(int)
        
        all_dfs.append(op_agg)
    
    if not all_dfs:
        print("  [!] No valid Part M data loaded!")
        return pd.DataFrame(columns=['OPERATOR_ID', 'report_year',
                                     'n_ext_corrosion', 'n_int_corrosion',
                                     'n_all_corrosion', 'n_total_repairs',
                                     'n_leaks', 'has_corrosion_history'])
    
    result = pd.concat(all_dfs, ignore_index=True)
    
    keep_cols = ['OPERATOR_ID', 'report_year', 'n_ext_corrosion', 'n_int_corrosion',
                 'n_all_corrosion', 'n_total_repairs', 'n_leaks', 'has_corrosion_history']
    result = result[keep_cols].copy()
    
    years_loaded = sorted(result['report_year'].unique())
    print(f"  Loaded {len(files)} files, years: {years_loaded}")
    print(f"  {len(result)} operator-year records")
    corr_ops = result.groupby('OPERATOR_ID')['has_corrosion_history'].max().sum()
    print(f"  Operators with any corrosion history: {corr_ops}")
    print(f"  Total ext corrosion repairs: {result['n_ext_corrosion'].sum():.0f}")
    print(f"  Total leaks/ruptures: {result['n_leaks'].sum():.0f}")
    
    return result


# ============================================================================
#  5. FEATURE ENGINEERING & JOIN
# ============================================================================

def build_analysis_panel(panel, part_h, part_k, part_m):
    """
    Join survival panel with Part H/K/M features.
    
    Strategy for temporal alignment:
      - If Part files have matching year to panel observation --> use same-year
      - If Part files have year t-1 (lagged) --> use lagged (preferred for prediction)
      - If only limited years available --> use time-invariant operator average
    
    Also engineers the technology era + age features.
    """
    print(f"\n[5/6] Building analysis panel...")
    
    df = panel.copy()
    
    # ---- Technology era features ----
    df['tech_era'] = df['decade_bin'].map(ERA_MAP).fillna('unknown')
    
    # Centered age and quadratic
    df['age_centered'] = df['age_at_obs'] - AGE_CENTER
    df['age_c2'] = df['age_centered'] ** 2
    
    # Calendar year centered
    df['year_centered'] = df['year'] - df['year'].mean()
    
    # Offset for Poisson (log miles at risk)
    df['log_exposure'] = np.log(df['miles_at_risk'].clip(lower=1e-6))
    
    # ---- Merge Part H (diameter) ----
    n_before = len(df)
    if len(part_h) > 0:
        # Try year-matched join first
        panel_years = set(df['year'].unique())
        part_h_years = set(part_h['report_year'].unique())
        
        if panel_years & part_h_years:
            # Direct year match available
            h_merge = part_h.rename(columns={'OPERATOR_ID': 'operator_id',
                                             'report_year': 'year'})
            df = df.merge(h_merge[['operator_id', 'year', 'pct_small_diam', 
                                   'pct_large_diam', 'avg_diameter_weighted', 'total_miles_h']],
                         on=['operator_id', 'year'], how='left')
            print(f"  Part H: year-matched join, matched {df['pct_small_diam'].notna().sum():,}/{len(df):,} rows")
        else:
            # Use time-invariant operator average (Parts from different years)
            h_avg = part_h.groupby('OPERATOR_ID').agg(
                pct_small_diam=('pct_small_diam', 'mean'),
                pct_large_diam=('pct_large_diam', 'mean'),
                avg_diameter_weighted=('avg_diameter_weighted', 'mean'),
                total_miles_h=('total_miles_h', 'mean'),
            ).reset_index().rename(columns={'OPERATOR_ID': 'operator_id'})
            
            df = df.merge(h_avg, on='operator_id', how='left')
            print(f"  Part H: time-invariant join, matched {df['pct_small_diam'].notna().sum():,}/{len(df):,} rows")
    else:
        print("  Part H: no data available")
    
    # ---- Merge Part K (%SMYS x class) ----
    if len(part_k) > 0:
        panel_years = set(df['year'].unique())
        part_k_years = set(part_k['report_year'].unique())
        
        if panel_years & part_k_years:
            k_merge = part_k.rename(columns={'OPERATOR_ID': 'operator_id',
                                             'report_year': 'year'})
            df = df.merge(k_merge[['operator_id', 'year', 'pct_class1', 'pct_class3_4',
                                   'pct_high_smys', 'pct_low_smys', 'total_miles_k']],
                         on=['operator_id', 'year'], how='left')
            print(f"  Part K: year-matched join, matched {df['pct_class1'].notna().sum():,}/{len(df):,} rows")
        else:
            k_avg = part_k.groupby('OPERATOR_ID').agg(
                pct_class1=('pct_class1', 'mean'),
                pct_class3_4=('pct_class3_4', 'mean'),
                pct_high_smys=('pct_high_smys', 'mean'),
                pct_low_smys=('pct_low_smys', 'mean'),
                total_miles_k=('total_miles_k', 'mean'),
            ).reset_index().rename(columns={'OPERATOR_ID': 'operator_id'})
            
            df = df.merge(k_avg, on='operator_id', how='left')
            print(f"  Part K: time-invariant join, matched {df['pct_class1'].notna().sum():,}/{len(df):,} rows")
    else:
        print("  Part K: no data available")
    
    # ---- Merge Part M (leak history) ----
    if len(part_m) > 0:
        panel_years = set(df['year'].unique())
        part_m_years = set(part_m['report_year'].unique())
        
        if panel_years & part_m_years:
            m_merge = part_m.rename(columns={'OPERATOR_ID': 'operator_id',
                                             'report_year': 'year'})
            df = df.merge(m_merge[['operator_id', 'year', 'n_ext_corrosion',
                                   'n_int_corrosion', 'n_all_corrosion',
                                   'n_total_repairs', 'n_leaks', 'has_corrosion_history']],
                         on=['operator_id', 'year'], how='left')
            print(f"  Part M: year-matched join, matched {df['has_corrosion_history'].notna().sum():,}/{len(df):,} rows")
        else:
            # For Part M, use CUMULATIVE history (sum across all available years)
            m_cum = part_m.groupby('OPERATOR_ID').agg(
                n_ext_corrosion=('n_ext_corrosion', 'sum'),
                n_int_corrosion=('n_int_corrosion', 'sum'),
                n_all_corrosion=('n_all_corrosion', 'sum'),
                n_total_repairs=('n_total_repairs', 'sum'),
                n_leaks=('n_leaks', 'sum'),
            ).reset_index().rename(columns={'OPERATOR_ID': 'operator_id'})
            m_cum['has_corrosion_history'] = (m_cum['n_all_corrosion'] > 0).astype(int)
            
            df = df.merge(m_cum, on='operator_id', how='left')
            print(f"  Part M: cumulative join, matched {df['has_corrosion_history'].notna().sum():,}/{len(df):,} rows")
    else:
        print("  Part M: no data available")
    
    # ---- Fill NaN with sensible defaults ----
    fill_zero = ['n_ext_corrosion', 'n_int_corrosion', 'n_all_corrosion',
                 'n_total_repairs', 'n_leaks', 'has_corrosion_history']
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # For Part H/K features, fill with median (conservative) if available
    fill_median = ['pct_small_diam', 'pct_large_diam', 'avg_diameter_weighted',
                   'pct_class1', 'pct_class3_4', 'pct_high_smys', 'pct_low_smys']
    for col in fill_median:
        if col in df.columns:
            median_val = df[col].median()
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_fill > 0:
                print(f"  Filled {n_fill:,} NaN in {col} with median={median_val:.3f}")
    
    # ---- Log-transform repair counts (right-skewed) ----
    if 'n_ext_corrosion' in df.columns:
        df['log1p_ext_corr'] = np.log1p(df['n_ext_corrosion'])
        df['log1p_total_repairs'] = np.log1p(df['n_total_repairs'])
        df['log1p_leaks'] = np.log1p(df['n_leaks'])
    
    # Remove rows with 0 exposure
    n_zero = (df['miles_at_risk'] <= 0).sum()
    if n_zero > 0:
        df = df[df['miles_at_risk'] > 0].copy()
        print(f"  Removed {n_zero} rows with zero exposure")
    
    # Remove unknown eras
    n_unk = (df['tech_era'] == 'unknown').sum()
    if n_unk > 0:
        df = df[df['tech_era'] != 'unknown'].copy()
        print(f"  Removed {n_unk} rows with unknown tech_era")
    
    print(f"\n  Final panel: {len(df):,} rows, {df['operator_id'].nunique()} operators")
    print(f"  Features available: {[c for c in df.columns if c not in panel.columns]}")
    
    return df


# ============================================================================
#  6. GLM MODEL SPECIFICATIONS
# ============================================================================

def fit_models(df, output_dir='.'):
    """
    Fit multiple GLM specifications and compare.
    
    Models:
      1. Baseline: age + tech_era (Poisson) -- no covariables
      2. Baseline: age + tech_era (NegBin) -- no covariables  
      3. Full: age + tech_era + diameter + SMYS/class + leak history (NegBin)
      4. Corrosion: age + tech_era + covariables (Poisson/robust)
    """
    print(f"\n[6/6] Fitting GLM models...")
    
    results = {}
    comparison = []
    
    # --- Check which features are available ---
    has_h = 'pct_small_diam' in df.columns and df['pct_small_diam'].notna().any()
    has_k = 'pct_class1' in df.columns and df['pct_class1'].notna().any()
    has_m = 'has_corrosion_history' in df.columns and df['has_corrosion_history'].notna().any()
    
    print(f"  Features available: Part H={has_h}, Part K={has_k}, Part M={has_m}")
    
    # --- Create dummies for tech_era ---
    era_dummies = pd.get_dummies(df['tech_era'], prefix='era', drop_first=False, dtype=float)
    # Reference: early_coat
    era_cols = [c for c in era_dummies.columns if c != 'era_early_coat']
    
    # Ensure all response and predictor variables are float64
    for col in ['n_incidents', 'n_corrosion', 'age_centered', 'age_c2', 'log_exposure']:
        df[col] = df[col].astype(np.float64)
    
    # ============================================
    # MODEL 1: Baseline -- All-cause, Poisson
    # ============================================
    print("\n  --- Model 1: All-cause baseline (Poisson) ---")
    X1 = pd.DataFrame({
        'const': 1,
        'age_centered': df['age_centered'],
        'age_c2': df['age_c2'],
    })
    for col in era_cols:
        X1[col] = era_dummies[col].values
    
    try:
        m1 = GLM(df['n_incidents'], X1, 
                  family=families.Poisson(),
                  offset=df['log_exposure']).fit()
        results['m1_baseline_poisson'] = m1
        pearson_chi2 = m1.pearson_chi2
        phi = pearson_chi2 / m1.df_resid
        comparison.append({
            'model': '1. All-cause baseline (Poisson)',
            'family': 'Poisson',
            'n_params': m1.df_model + 1,
            'AIC': m1.aic,
            'deviance': m1.deviance,
            'pearson_chi2': pearson_chi2,
            'phi': phi,
            'n_obs': m1.nobs,
        })
        print(f"    AIC={m1.aic:.0f}, Deviance={m1.deviance:.0f}, phi={phi:.1f}")
    except Exception as e:
        print(f"    [!] Failed: {e}")
    
    # ============================================
    # MODEL 2: Baseline -- All-cause, NegBin
    # ============================================
    print("\n  --- Model 2: All-cause baseline (NegBin) ---")
    try:
        m2 = GLM(df['n_incidents'], X1,
                  family=families.NegativeBinomial(alpha=1.0),
                  offset=df['log_exposure']).fit()
        results['m2_baseline_negbin'] = m2
        pearson_chi2 = m2.pearson_chi2
        phi = pearson_chi2 / m2.df_resid
        comparison.append({
            'model': '2. All-cause baseline (NegBin)',
            'family': 'NegBin',
            'n_params': m2.df_model + 1,
            'AIC': m2.aic,
            'deviance': m2.deviance,
            'pearson_chi2': pearson_chi2,
            'phi': phi,
            'n_obs': m2.nobs,
        })
        print(f"    AIC={m2.aic:.0f}, Deviance={m2.deviance:.0f}, phi={phi:.1f}")
    except Exception as e:
        print(f"    [!] Failed: {e}")
    
    # ============================================
    # MODEL 3: Full -- All-cause, NegBin + covariables
    # ============================================
    if has_h or has_k or has_m:
        print("\n  --- Model 3: All-cause FULL (NegBin + covariables) ---")
        X3 = X1.copy()
        
        covar_names = []
        if has_h:
            X3['pct_small_diam'] = df['pct_small_diam'].astype(np.float64).values
            X3['pct_large_diam'] = df['pct_large_diam'].astype(np.float64).values
            covar_names += ['pct_small_diam', 'pct_large_diam']
        if has_k:
            X3['pct_class3_4'] = df['pct_class3_4'].astype(np.float64).values
            X3['pct_high_smys'] = df['pct_high_smys'].astype(np.float64).values
            covar_names += ['pct_class3_4', 'pct_high_smys']
        if has_m:
            X3['has_corrosion_history'] = df['has_corrosion_history'].astype(np.float64).values
            X3['log1p_total_repairs'] = df['log1p_total_repairs'].astype(np.float64).values
            covar_names += ['has_corrosion_history', 'log1p_total_repairs']
        
        try:
            m3 = GLM(df['n_incidents'], X3,
                      family=families.NegativeBinomial(alpha=1.0),
                      offset=df['log_exposure']).fit()
            results['m3_full_negbin'] = m3
            pearson_chi2 = m3.pearson_chi2
            phi = pearson_chi2 / m3.df_resid
            comparison.append({
                'model': f'3. All-cause FULL (NegBin + {"+".join(covar_names)})',
                'family': 'NegBin',
                'n_params': m3.df_model + 1,
                'AIC': m3.aic,
                'deviance': m3.deviance,
                'pearson_chi2': pearson_chi2,
                'phi': phi,
                'n_obs': m3.nobs,
            })
            print(f"    Covariables: {covar_names}")
            print(f"    AIC={m3.aic:.0f}, Deviance={m3.deviance:.0f}, phi={phi:.1f}")
            
            # Improvement vs baseline
            if 'm2_baseline_negbin' in results:
                aic_delta = m3.aic - m2.aic
                print(f"    Delta-AIC vs baseline: {aic_delta:+.0f} ({'BETTER' if aic_delta < 0 else 'worse'})")
        except Exception as e:
            print(f"    [!] Failed: {e}")
    
    # ============================================
    # MODEL 4: Corrosion -- Poisson/robust baseline
    # ============================================
    print("\n  --- Model 4: Corrosion baseline (Poisson/robust) ---")
    try:
        m4 = GLM(df['n_corrosion'], X1,
                  family=families.Poisson(),
                  offset=df['log_exposure']).fit(cov_type='HC1')
        results['m4_corrosion_baseline'] = m4
        pearson_chi2 = m4.pearson_chi2
        phi = pearson_chi2 / m4.df_resid
        comparison.append({
            'model': '4. Corrosion baseline (Poisson/robust)',
            'family': 'Poisson/HC1',
            'n_params': m4.df_model + 1,
            'AIC': m4.aic,
            'deviance': m4.deviance,
            'pearson_chi2': pearson_chi2,
            'phi': phi,
            'n_obs': m4.nobs,
        })
        print(f"    AIC={m4.aic:.0f}, Deviance={m4.deviance:.0f}, phi={phi:.1f}")
    except Exception as e:
        print(f"    [!] Failed: {e}")
    
    # ============================================
    # MODEL 5: Corrosion -- Full with covariables
    # ============================================
    if has_h or has_k or has_m:
        print("\n  --- Model 5: Corrosion FULL (Poisson/robust + covariables) ---")
        # For corrosion, only 106 events --> be parsimonious (10 EPV rule)
        # Max ~10 parameters
        X5 = pd.DataFrame({
            'const': 1,
            'age_centered': df['age_centered'],
        })
        for col in era_cols:
            X5[col] = era_dummies[col].values
        
        corr_covars = []
        if has_h:
            X5['pct_small_diam'] = df['pct_small_diam'].astype(np.float64).values
            corr_covars.append('pct_small_diam')
        if has_k:
            X5['pct_high_smys'] = df['pct_high_smys'].astype(np.float64).values
            corr_covars.append('pct_high_smys')
        if has_m:
            X5['has_corrosion_history'] = df['has_corrosion_history'].astype(np.float64).values
            corr_covars.append('has_corrosion_history')
        
        try:
            m5 = GLM(df['n_corrosion'], X5,
                      family=families.Poisson(),
                      offset=df['log_exposure']).fit(cov_type='HC1')
            results['m5_corrosion_full'] = m5
            pearson_chi2 = m5.pearson_chi2
            phi = pearson_chi2 / m5.df_resid
            comparison.append({
                'model': f'5. Corrosion FULL (Poisson/robust + {"+".join(corr_covars)})',
                'family': 'Poisson/HC1',
                'n_params': m5.df_model + 1,
                'AIC': m5.aic,
                'deviance': m5.deviance,
                'pearson_chi2': pearson_chi2,
                'phi': phi,
                'n_obs': m5.nobs,
            })
            print(f"    Covariables: {corr_covars}")
            print(f"    AIC={m5.aic:.0f}, Deviance={m5.deviance:.0f}, phi={phi:.1f}")
            
            if 'm4_corrosion_baseline' in results:
                aic_delta = m5.aic - m4.aic
                print(f"    Delta-AIC vs baseline: {aic_delta:+.0f} ({'BETTER' if aic_delta < 0 else 'worse'})")
        except Exception as e:
            print(f"    [!] Failed: {e}")
    
    # ---- Save comparison table ----
    comp_df = pd.DataFrame(comparison)
    comp_path = os.path.join(output_dir, 'model_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Model comparison saved: {comp_path}")
    
    return results, comp_df


# ============================================================================
#  7. DIAGNOSTICS & OUTPUT
# ============================================================================

def generate_outputs(df, results, comp_df, output_dir='.'):
    """Generate HTML summaries, diagnostics, and risk scores."""
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # ---- Print comparison table ----
    print(f"\n{'Model':<55} {'AIC':>8} {'Deviance':>10} {'phi':>8} {'Params':>7}")
    print("-" * 95)
    for _, row in comp_df.iterrows():
        print(f"{row['model']:<55} {row['AIC']:>8.0f} {row['deviance']:>10.0f} "
              f"{row['phi']:>8.1f} {row['n_params']:>7.0f}")
    
    # ---- Print coefficient tables for key models ----
    for name, model in results.items():
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        # Coefficient table with rate ratios
        coefs = model.params
        ci = model.conf_int()
        pvals = model.pvalues
        
        print(f"\n{'Variable':<25} {'Coef':>8} {'SE':>8} {'p-val':>8} {'RR':>8} {'RR_lo':>8} {'RR_hi':>8}")
        print("-" * 85)
        for var in coefs.index:
            se = model.bse[var] if var in model.bse.index else np.nan
            p = pvals[var] if var in pvals.index else np.nan
            rr = np.exp(coefs[var])
            rr_lo = np.exp(ci.loc[var, 0]) if var in ci.index else np.nan
            rr_hi = np.exp(ci.loc[var, 1]) if var in ci.index else np.nan
            
            sig = ''
            if p < 0.001: sig = '***'
            elif p < 0.01: sig = '**'
            elif p < 0.05: sig = '*'
            elif p < 0.10: sig = '+'
            
            print(f"{var:<25} {coefs[var]:>8.4f} {se:>8.4f} {p:>8.4f} "
                  f"{rr:>8.3f} {rr_lo:>8.3f} {rr_hi:>8.3f} {sig}")
    
    # ---- Save HTML summaries ----
    for name, model in results.items():
        html_path = os.path.join(output_dir, f'{name}_summary.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"<html><head><meta charset='utf-8'><title>{name}</title></head><body>")
            f.write(f"<h1>{name}</h1>")
            f.write(f"<pre>{model.summary().as_html()}</pre>")
            f.write("</body></html>")
    
    # ---- Generate operator risk scores ----
    # Use the best all-cause model
    best_ac = None
    for key in ['m3_full_negbin', 'm2_baseline_negbin', 'm1_baseline_poisson']:
        if key in results:
            best_ac = results[key]
            best_ac_name = key
            break
    
    if best_ac is not None:
        # Predict rates
        df['predicted_rate_allcause'] = best_ac.predict(
            exog=best_ac.model.exog,
            offset=np.zeros(len(df))  # Rate per mile-year (no offset)
        )
        
        # Aggregate to operator level
        op_risk = df.groupby(['operator_id', 'operator_name']).agg(
            total_miles=('miles_at_risk', 'sum'),
            total_incidents=('n_incidents', 'sum'),
            total_corrosion=('n_corrosion', 'sum'),
            mean_predicted_rate=('predicted_rate_allcause', 'mean'),
            max_predicted_rate=('predicted_rate_allcause', 'max'),
        ).reset_index()
        
        op_risk['observed_rate'] = op_risk['total_incidents'] / op_risk['total_miles'].replace(0, np.nan)
        op_risk = op_risk.sort_values('mean_predicted_rate', ascending=False)
        
        risk_path = os.path.join(output_dir, 'operator_risk_scores.csv')
        op_risk.to_csv(risk_path, index=False, float_format='%.6f')
        print(f"\n  Operator risk scores saved: {risk_path} ({len(op_risk)} operators)")
    
    # ---- Diagnostics file ----
    diag_path = os.path.join(output_dir, 'diagnostics.txt')
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write("ROSEN P4 -- GLM Diagnostics\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write(comp_df.to_string(index=False))
        f.write("\n\n")
        
        for name, model in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"Model: {name}\n")
            f.write(f"{'='*60}\n")
            f.write(str(model.summary()))
            f.write(f"\n\nPearson chi2: {model.pearson_chi2:.2f}")
            f.write(f"\nDispersion (phi): {model.pearson_chi2 / model.df_resid:.2f}")
            f.write(f"\nDeviance/df: {model.deviance / model.df_resid:.4f}")
            f.write("\n\n")
    
    print(f"  Diagnostics saved: {diag_path}")
    
    return df


# ============================================================================
#  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ROSEN P4 Poisson GLM')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Directory containing survival_panel_7yr.csv')
    parser.add_argument('--part-dir', default=DEFAULT_PART_DIR,
                        help='Directory containing GT_AR_YYYY_Part_X.csv files')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as data-dir)')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    part_dir = args.part_dir
    output_dir = args.output_dir or data_dir
    
    # Auto-detect Part file location if not explicitly set or if no files found
    if not glob.glob(os.path.join(part_dir, 'GT_AR_*_Part_*.csv')):
        part_dir = find_part_dir(data_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ROSEN Proyecto 4 -- Poisson GLM with PHMSA Covariables")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Data dir: {data_dir}")
    print(f"Part dir: {part_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    
    # 1. Load survival panel
    panel_path = os.path.join(data_dir, SURVIVAL_PANEL)
    panel = parse_survival_panel(panel_path)
    
    # 2-4. Load Part files
    part_h = load_part_h(part_dir)
    part_k = load_part_k(part_dir)
    part_m = load_part_m(part_dir)
    
    # 5. Build analysis panel
    df = build_analysis_panel(panel, part_h, part_k, part_m)
    
    # 6. Fit models
    model_results, comp_df = fit_models(df, output_dir)
    
    # 7. Generate outputs
    df = generate_outputs(df, model_results, comp_df, output_dir)
    
    print(f"\n{'='*60}")
    print("DONE! All outputs saved.")
    print(f"{'='*60}")
    
    return df, model_results, comp_df


if __name__ == '__main__':
    df, results, comp = main()