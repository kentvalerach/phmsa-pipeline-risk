"""
ROSEN Project 4 -- 15-Year Survival Panel Builder + NegBin GLM
================================================================
Expands the 7-year panel (2017-2024) to full 15-year (2010-2024).
Reads all Part J CSVs, matches incidents, integrates Parts H/K/M,
and estimates Negative Binomial GLM.

Usage:
    cd C:\Phmsa\annual_gt
    python rosen_p4_15yr_panel.py --part-dir extracted_csvs

Or with explicit paths:
    python rosen_p4_15yr_panel.py \
        --data-dir C:\Phmsa\annual_gt \
        --part-dir C:\Phmsa\annual_gt\extracted_csvs \
        --incident-dir C:\Phmsa \
        --output-dir C:\Phmsa\results_15yr

Author: Kent (ROSEN Project 4)
Date: 2026-02-04
"""

import argparse
import csv
import glob
import os
import sys
import math
import re
from collections import defaultdict
from datetime import datetime

# ============================================================
# 0. CONFIGURATION
# ============================================================

DECADE_COLS_TRANS = {
    'pre1940':  ('PARTJTONPRE1940', 1925),
    '1940_49':  ('PARTJTON194049',  1945),
    '1950_59':  ('PARTJTON195059',  1955),
    '1960_69':  ('PARTJTON196069',  1965),
    '1970_79':  ('PARTJTON197079',  1975),
    '1980_89':  ('PARTJTON198089',  1985),
    '1990_99':  ('PARTJTON199099',  1995),
    '2000_09':  ('PARTJTON200009',  2005),
    '2010_19':  ('PARTJTON201019',  2015),
    '2020_29':  ('PARTJTON202029',  2022),
}

ERA_MAP = {
    'pre1940':  'era_pre1940',
    '1940_49':  'era_50s_60s',
    '1950_59':  'era_50s_60s',
    '1960_69':  'era_50s_60s',
    '1970_79':  'era_coal_tar',
    '1980_89':  'era_improved',
    '1990_99':  'era_improved',
    '2000_09':  'era_modern',
    '2010_19':  'era_modern',
    '2020_29':  'era_modern',
}

STATE_ABBREV_TO_NAME = {
    'AL':'ALABAMA','AK':'ALASKA','AZ':'ARIZONA','AR':'ARKANSAS',
    'CA':'CALIFORNIA','CO':'COLORADO','CT':'CONNECTICUT','DE':'DELAWARE',
    'FL':'FLORIDA','GA':'GEORGIA','HI':'HAWAII','ID':'IDAHO',
    'IL':'ILLINOIS','IN':'INDIANA','IA':'IOWA','KS':'KANSAS',
    'KY':'KENTUCKY','LA':'LOUISIANA','ME':'MAINE','MD':'MARYLAND',
    'MA':'MASSACHUSETTS','MI':'MICHIGAN','MN':'MINNESOTA','MS':'MISSISSIPPI',
    'MO':'MISSOURI','MT':'MONTANA','NE':'NEBRASKA','NV':'NEVADA',
    'NH':'NEW HAMPSHIRE','NJ':'NEW JERSEY','NM':'NEW MEXICO','NY':'NEW YORK',
    'NC':'NORTH CAROLINA','ND':'NORTH DAKOTA','OH':'OHIO','OK':'OKLAHOMA',
    'OR':'OREGON','PA':'PENNSYLVANIA','RI':'RHODE ISLAND','SC':'SOUTH CAROLINA',
    'SD':'SOUTH DAKOTA','TN':'TENNESSEE','TX':'TEXAS','UT':'UTAH',
    'VT':'VERMONT','VA':'VIRGINIA','WA':'WASHINGTON','WV':'WEST VIRGINIA',
    'WI':'WISCONSIN','WY':'WYOMING','DC':'DISTRICT OF COLUMBIA',
    'GU':'GUAM','PR':'PUERTO RICO','VI':'VIRGIN ISLANDS',
}


def safe_float(v, default=0.0):
    """Convert value to float safely."""
    if v is None or v == '':
        return default
    try:
        s = str(v).strip().replace(',', '')
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_int(v, default=None):
    """Convert value to int safely."""
    if v is None or v == '':
        return default
    try:
        return int(float(str(v).strip().replace(',', '')))
    except (ValueError, TypeError):
        return default


def get_decade_bin(install_year):
    """Map installation year to decade bin."""
    if install_year is None:
        return None
    if install_year < 1940:
        return 'pre1940'
    elif install_year < 1950:
        return '1940_49'
    elif install_year < 1960:
        return '1950_59'
    elif install_year < 1970:
        return '1960_69'
    elif install_year < 1980:
        return '1970_79'
    elif install_year < 1990:
        return '1980_89'
    elif install_year < 2000:
        return '1990_99'
    elif install_year < 2010:
        return '2000_09'
    elif install_year < 2020:
        return '2010_19'
    else:
        return '2020_29'


# ============================================================
# 1. INCIDENT LOADING
# ============================================================

def find_incident_files(incident_dir):
    """Find PHMSA incident CSV/Excel files."""
    patterns = [
        os.path.join(incident_dir, '**', '*gt_flagged*.csv'),
        os.path.join(incident_dir, '**', '*GT*Flagged*.csv'),
        os.path.join(incident_dir, '**', '*incident*.csv'),
        os.path.join(incident_dir, '**', '*Incident*.csv'),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return list(set(files))


def load_incidents_from_csv(filepath):
    """Load incidents from a single CSV file."""
    incidents = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                incidents.append(row)
    except Exception as e:
        print(f"    Warning: Could not read {filepath}: {e}")
    return incidents


def load_incidents_from_panel(panel_path):
    """Extract incident counts from existing survival panel CSV."""
    incident_counts = defaultdict(lambda: {
        'total': 0, 'corrosion': 0, 'ext_corr': 0, 'int_corr': 0,
        'material': 0, 'excavation': 0, 'natural': 0, 'other': 0
    })
    
    with open(panel_path, 'r', encoding='utf-8', errors='replace') as f:
        # Handle the special quoting format
        content = f.read()
    
    lines = content.strip().split('\n')
    if not lines:
        return incident_counts
    
    # Parse header
    header_line = lines[0].strip()
    if header_line.startswith('"') and header_line.endswith('"'):
        header_line = header_line[1:-1]
    headers = []
    in_quote = False
    current = ''
    for ch in header_line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == ',' and not in_quote:
            headers.append(current.strip())
            current = ''
        else:
            current += ch
    headers.append(current.strip())
    
    # Parse rows
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        
        fields = []
        in_quote = False
        current = ''
        for ch in line:
            if ch == '"':
                if in_quote and len(current) > 0 and current[-1] == '"':
                    current = current[:-1] + '"'
                else:
                    in_quote = not in_quote
            elif ch == ',' and not in_quote:
                fields.append(current.strip())
                current = ''
            else:
                current += ch
        fields.append(current.strip())
        
        if len(fields) != len(headers):
            continue
        
        row = dict(zip(headers, fields))
        op_id = row.get('operator_id', '')
        state = row.get('state', '')
        year = safe_int(row.get('year'))
        decade = row.get('decade_bin', '')
        n_inc = safe_int(row.get('n_incidents'), 0)
        n_corr = safe_int(row.get('n_corrosion'), 0)
        n_ext = safe_int(row.get('n_ext_corr'), 0)
        n_int = safe_int(row.get('n_int_corr'), 0)
        n_mat = safe_int(row.get('n_material'), 0)
        n_exc = safe_int(row.get('n_excavation'), 0)
        n_nat = safe_int(row.get('n_natural'), 0)
        n_oth = safe_int(row.get('n_other_cause'), 0)
        
        if n_inc and n_inc > 0 and year:
            key = (op_id, state, year, decade)
            incident_counts[key]['total'] += n_inc
            incident_counts[key]['corrosion'] += (n_corr or 0)
            incident_counts[key]['ext_corr'] += (n_ext or 0)
            incident_counts[key]['int_corr'] += (n_int or 0)
            incident_counts[key]['material'] += (n_mat or 0)
            incident_counts[key]['excavation'] += (n_exc or 0)
            incident_counts[key]['natural'] += (n_nat or 0)
            incident_counts[key]['other'] += (n_oth or 0)
    
    return incident_counts


def index_incidents(incident_rows):
    """Index raw incident rows by (operator, state, year, decade)."""
    incident_counts = defaultdict(lambda: {
        'total': 0, 'corrosion': 0, 'ext_corr': 0, 'int_corr': 0,
        'material': 0, 'excavation': 0, 'natural': 0, 'other': 0
    })
    
    for inc in incident_rows:
        # Try various column name patterns
        op_id = str(inc.get('OPERATOR_ID', inc.get('operator_id', '')) or '').strip()
        
        # State: try abbreviation first, then full name
        state_abbrev = str(inc.get('ONSHORE_STATE_ABBREVIATION',
                          inc.get('STATE', '')) or '').strip().upper()
        state_name = STATE_ABBREV_TO_NAME.get(state_abbrev, state_abbrev)
        
        iyear = safe_int(inc.get('IYEAR', inc.get('iyear', inc.get('REPORT_YEAR', ''))))
        install_year = safe_int(inc.get('INSTALLATION_YEAR', inc.get('install_year', '')))
        
        cause = str(inc.get('CAUSE', inc.get('cause', '')) or '').upper()
        cause_detail = str(inc.get('CAUSE_DETAILS', inc.get('cause_details', '')) or '').upper()
        
        decade = get_decade_bin(install_year)
        if not op_id or not state_name or not iyear:
            continue
        
        # If no decade from install year, skip (cannot match to panel cell)
        if not decade:
            continue
        
        key = (op_id, state_name, iyear, decade)
        incident_counts[key]['total'] += 1
        
        if 'CORROSION' in cause:
            incident_counts[key]['corrosion'] += 1
            if 'INTERNAL' in cause_detail:
                incident_counts[key]['int_corr'] += 1
            else:
                incident_counts[key]['ext_corr'] += 1
        elif any(x in cause for x in ['MATERIAL', 'WELD', 'EQUIP']):
            incident_counts[key]['material'] += 1
        elif 'EXCAVATION' in cause:
            incident_counts[key]['excavation'] += 1
        elif 'NATURAL' in cause:
            incident_counts[key]['natural'] += 1
        else:
            incident_counts[key]['other'] += 1
    
    return incident_counts


# ============================================================
# 2. PART J PANEL BUILDER
# ============================================================

def find_part_files(part_dir, part_letter='J'):
    """Find all Part J (or H, K, M) CSV files and extract years."""
    pattern = os.path.join(part_dir, f'GT_AR_*_Part_{part_letter}.csv')
    files = {}
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        match = re.search(r'GT_AR_(\d{4})_Part_', basename)
        if match:
            year = int(match.group(1))
            files[year] = path
    return files


def build_panel(part_j_files, incident_counts):
    """Build survival panel from Part J files + incidents."""
    panel_rows = []
    
    for report_year in sorted(part_j_files.keys()):
        path = part_j_files[report_year]
        year_count = 0
        year_events = 0
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    op_id = str(row.get('OPERATOR_ID', '') or '').strip()
                    state = str(row.get('STATE_NAME', '') or '').strip().upper()
                    op_name = str(row.get('PARTA2NAMEOFCOMP', '') or '').strip()
                    
                    if not op_id or not state:
                        continue
                    
                    for decade_label, (col, midpoint_year) in DECADE_COLS_TRANS.items():
                        miles = safe_float(row.get(col, 0))
                        if miles <= 0:
                            continue
                        
                        age = report_year - midpoint_year
                        if age < 0:
                            continue
                        
                        era = ERA_MAP.get(decade_label, 'era_50s_60s')
                        
                        # Match incidents
                        key = (op_id, state, report_year, decade_label)
                        inc = incident_counts.get(key, {
                            'total': 0, 'corrosion': 0, 'ext_corr': 0,
                            'int_corr': 0, 'material': 0, 'excavation': 0,
                            'natural': 0, 'other': 0
                        })
                        
                        panel_rows.append({
                            'operator_id': op_id,
                            'operator_name': op_name,
                            'state': state,
                            'year': report_year,
                            'decade_bin': decade_label,
                            'install_midpoint': midpoint_year,
                            'age_at_obs': age,
                            'era': era,
                            'miles_at_risk': round(miles, 3),
                            'n_incidents': inc['total'],
                            'n_corrosion': inc['corrosion'],
                            'n_ext_corr': inc.get('ext_corr', 0),
                            'n_int_corr': inc.get('int_corr', 0),
                            'n_material': inc['material'],
                            'n_excavation': inc.get('excavation', 0),
                            'n_natural': inc.get('natural', 0),
                            'n_other_cause': inc.get('other', 0),
                            'event': 1 if inc['total'] > 0 else 0,
                            'event_corrosion': 1 if inc['corrosion'] > 0 else 0,
                        })
                        year_count += 1
                        if inc['total'] > 0:
                            year_events += 1
        
        except Exception as e:
            print(f"    ERROR reading {path}: {e}")
            continue
        
        print(f"    {report_year}: {year_count:,} obs, {year_events} with events")
    
    return panel_rows


# ============================================================
# 3. COVARIABLE INTEGRATION (Parts H, K, M)
# ============================================================

def load_part_h(filepath):
    """Extract diameter distribution from Part H."""
    operators = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_id = str(row.get('OPERATOR_ID', '') or '').strip()
                state = str(row.get('STATE_NAME', '') or '').strip().upper()
                if not op_id:
                    continue
                
                # Sum diameter categories
                small_cols = [c for c in row.keys() if 
                    any(x in c.upper() for x in ['LESS4', '4LESS6', '6LESS8', '4TO6', '6TO8',
                        'PARTHN1', 'PARTHN2', 'PARTHN3'])]
                large_cols = [c for c in row.keys() if
                    any(x in c.upper() for x in ['20TO22', '22TO24', '24TO26', '26TO28', '28TO30',
                        '20LESS22', '22LESS24', '24LESS26', '26LESS28', '28LESS30',
                        'GT30', 'GREATER30', 'MORE30'])]
                total_cols = [c for c in row.keys() if 'TOTAL' in c.upper() and 'PART' in c.upper()]
                
                small_miles = sum(safe_float(row.get(c, 0)) for c in small_cols)
                large_miles = sum(safe_float(row.get(c, 0)) for c in large_cols)
                
                # Try to get total from explicit total column
                total_miles = 0
                for c in total_cols:
                    v = safe_float(row.get(c, 0))
                    if v > total_miles:
                        total_miles = v
                
                # Fallback: sum all numeric diameter columns
                if total_miles <= 0:
                    all_miles = sum(safe_float(row.get(c, 0)) for c in row.keys() 
                                   if 'PARTH' in c.upper() and 'TOTAL' not in c.upper())
                    total_miles = all_miles if all_miles > 0 else 1
                
                key = (op_id, state)
                operators[key] = {
                    'pct_small_diam': small_miles / max(total_miles, 0.001),
                    'pct_large_diam': large_miles / max(total_miles, 0.001),
                }
    except Exception as e:
        print(f"    Warning: Part H read error: {e}")
    return operators


def load_part_k(filepath):
    """Extract SMYS and class location from Part K."""
    operators = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_id = str(row.get('OPERATOR_ID', '') or '').strip()
                state = str(row.get('STATE_NAME', '') or '').strip().upper()
                if not op_id:
                    continue
                
                # High SMYS: >60% columns
                high_smys_cols = [c for c in row.keys() if
                    any(x in c.upper() for x in ['6172', '61TO72', '61_72', 'GT72', 'GREATER72', 'MORE72'])]
                class1_cols = [c for c in row.keys() if 'CLASS1' in c.upper() or 'CL1' in c.upper()]
                total_cols = [c for c in row.keys() if 'TOTAL' in c.upper() and 'PART' in c.upper()]
                
                high_smys = sum(safe_float(row.get(c, 0)) for c in high_smys_cols)
                class1 = sum(safe_float(row.get(c, 0)) for c in class1_cols)
                
                total_miles = 0
                for c in total_cols:
                    v = safe_float(row.get(c, 0))
                    if v > total_miles:
                        total_miles = v
                
                if total_miles <= 0:
                    total_miles = 1
                
                key = (op_id, state)
                operators[key] = {
                    'pct_high_smys': high_smys / total_miles,
                    'pct_class1': class1 / total_miles,
                }
    except Exception as e:
        print(f"    Warning: Part K read error: {e}")
    return operators


def load_part_m(filepath):
    """Extract leak/repair history from Part M."""
    operators = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_id = str(row.get('OPERATOR_ID', '') or '').strip()
                state = str(row.get('STATE_NAME', '') or '').strip().upper()
                if not op_id:
                    continue
                
                # Sum all repair/leak columns
                repair_cols = [c for c in row.keys() if 'PARTM' in c.upper()]
                corr_cols = [c for c in row.keys() if 'PARTM' in c.upper() and 
                            any(x in c.upper() for x in ['CORR', 'EXTCOR', 'INTCOR'])]
                
                total_repairs = sum(abs(safe_float(row.get(c, 0))) for c in repair_cols)
                ext_corrosion = sum(abs(safe_float(row.get(c, 0))) for c in corr_cols)
                
                key = (op_id, state)
                operators[key] = {
                    'log1p_total_repairs': math.log1p(total_repairs),
                    'log1p_ext_corrosion': math.log1p(ext_corrosion),
                }
    except Exception as e:
        print(f"    Warning: Part M read error: {e}")
    return operators


def integrate_covariables(panel_rows, part_dir):
    """Join Parts H, K, M covariables to panel rows using year-matched joins."""
    
    # Find all available Part files
    part_h_files = find_part_files(part_dir, 'H')
    part_k_files = find_part_files(part_dir, 'K')
    part_m_files = find_part_files(part_dir, 'M')
    
    print(f"\n  Covariable files found:")
    print(f"    Part H: {len(part_h_files)} years {sorted(part_h_files.keys()) if part_h_files else '(none)'}")
    print(f"    Part K: {len(part_k_files)} years {sorted(part_k_files.keys()) if part_k_files else '(none)'}")
    print(f"    Part M: {len(part_m_files)} years {sorted(part_m_files.keys()) if part_m_files else '(none)'}")
    
    # Load all years
    h_by_year = {}
    k_by_year = {}
    m_by_year = {}
    
    for yr, path in part_h_files.items():
        h_by_year[yr] = load_part_h(path)
    for yr, path in part_k_files.items():
        k_by_year[yr] = load_part_k(path)
    for yr, path in part_m_files.items():
        m_by_year[yr] = load_part_m(path)
    
    available_h_years = sorted(h_by_year.keys())
    available_k_years = sorted(k_by_year.keys())
    available_m_years = sorted(m_by_year.keys())
    
    def find_nearest_year(target, available, max_gap=2):
        """Find nearest available year within max_gap."""
        if target in available:
            return target
        best = None
        best_dist = float('inf')
        for yr in available:
            d = abs(yr - target)
            if d < best_dist and d <= max_gap:
                best_dist = d
                best = yr
        return best
    
    # Attach covariables
    matched_h = matched_k = matched_m = 0
    
    for row in panel_rows:
        op_key = (row['operator_id'], row['state'])
        panel_year = row['year']
        
        # Part H
        h_year = find_nearest_year(panel_year, available_h_years)
        if h_year and op_key in h_by_year.get(h_year, {}):
            h_data = h_by_year[h_year][op_key]
            row['pct_small_diam'] = round(h_data['pct_small_diam'], 4)
            row['pct_large_diam'] = round(h_data['pct_large_diam'], 4)
            matched_h += 1
        else:
            row['pct_small_diam'] = None
            row['pct_large_diam'] = None
        
        # Part K
        k_year = find_nearest_year(panel_year, available_k_years)
        if k_year and op_key in k_by_year.get(k_year, {}):
            k_data = k_by_year[k_year][op_key]
            row['pct_high_smys'] = round(k_data['pct_high_smys'], 4)
            row['pct_class1'] = round(k_data['pct_class1'], 4)
            matched_k += 1
        else:
            row['pct_high_smys'] = None
            row['pct_class1'] = None
        
        # Part M
        m_year = find_nearest_year(panel_year, available_m_years)
        if m_year and op_key in m_by_year.get(m_year, {}):
            m_data = m_by_year[m_year][op_key]
            row['log1p_total_repairs'] = round(m_data['log1p_total_repairs'], 4)
            row['log1p_ext_corrosion'] = round(m_data['log1p_ext_corrosion'], 4)
            matched_m += 1
        else:
            row['log1p_total_repairs'] = None
            row['log1p_ext_corrosion'] = None
    
    n = len(panel_rows)
    print(f"\n  Covariable match rates:")
    print(f"    Part H (diameter):   {matched_h:,}/{n:,} ({100*matched_h/n:.1f}%)")
    print(f"    Part K (SMYS/class): {matched_k:,}/{n:,} ({100*matched_k/n:.1f}%)")
    print(f"    Part M (repairs):    {matched_m:,}/{n:,} ({100*matched_m/n:.1f}%)")
    
    return panel_rows


# ============================================================
# 4. NEGATIVE BINOMIAL GLM
# ============================================================

def fit_negbin_glm(panel_rows, response='n_incidents', label='All-Cause'):
    """Fit NegBin GLM using statsmodels."""
    try:
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.genmod.families import NegativeBinomial as NB_family
    except ImportError:
        print("  ERROR: statsmodels/numpy not installed. Run: pip install numpy statsmodels")
        return None
    
    # Prepare arrays
    y_list = []
    offset_list = []
    era_values = []
    age_values = []
    cov_small = []
    cov_large = []
    cov_smys = []
    cov_class1 = []
    cov_repairs = []
    cov_ext_corr = []
    
    has_covariables = False
    
    for row in panel_rows:
        y_val = safe_int(row.get(response), 0) or 0
        miles = safe_float(row.get('miles_at_risk', 0))
        if miles <= 0:
            continue
        
        y_list.append(y_val)
        offset_list.append(math.log(miles))
        era_values.append(row.get('era', 'era_50s_60s'))
        age_values.append(safe_float(row.get('age_at_obs', 0)))
        
        # Covariables (may be None)
        sd = row.get('pct_small_diam')
        ld = row.get('pct_large_diam')
        hs = row.get('pct_high_smys')
        c1 = row.get('pct_class1')
        tr = row.get('log1p_total_repairs')
        ec = row.get('log1p_ext_corrosion')
        
        if sd is not None:
            has_covariables = True
        
        cov_small.append(safe_float(sd, 0))
        cov_large.append(safe_float(ld, 0))
        cov_smys.append(safe_float(hs, 0))
        cov_class1.append(safe_float(c1, 0))
        cov_repairs.append(safe_float(tr, 0))
        cov_ext_corr.append(safe_float(ec, 0))
    
    y = np.array(y_list, dtype=np.float64)
    offset = np.array(offset_list, dtype=np.float64)
    age = np.array(age_values, dtype=np.float64)
    
    n = len(y)
    print(f"\n  {label} GLM: {n:,} obs, {int(y.sum())} events, mean={y.mean():.4f}")
    
    # Era dummies (reference = era_50s_60s)
    era_set = ['era_pre1940', 'era_coal_tar', 'era_improved', 'era_modern']
    era_dummies = {}
    for era_name in era_set:
        era_dummies[era_name] = np.array([1.0 if e == era_name else 0.0 for e in era_values])
    
    # Build design matrix
    col_names = ['const', 'age_at_obs'] + era_set
    X_cols = [np.ones(n), age]
    for era_name in era_set:
        X_cols.append(era_dummies[era_name])
    
    # Add covariables if available
    if has_covariables:
        cov_arrays = {
            'pct_small_diam': np.array(cov_small, dtype=np.float64),
            'pct_high_smys': np.array(cov_smys, dtype=np.float64),
            'log1p_total_repairs': np.array(cov_repairs, dtype=np.float64),
        }
        for cname, carr in cov_arrays.items():
            if carr.std() > 0.001:
                col_names.append(cname)
                X_cols.append(carr)
    
    X = np.column_stack(X_cols).astype(np.float64)
    
    # Fit NegBin
    print(f"  Fitting NegBin ({X.shape[1]} features)...")
    try:
        model = sm.NegativeBinomialP(y, X, offset=offset, p=2)
        result = model.fit(method='newton', maxiter=100, disp=0)
        
        print(f"  Converged: {result.mle_retvals.get('converged', 'unknown')}")
        print(f"  AIC: {result.aic:.1f}")
        print(f"  alpha (dispersion): {result.params[-1]:.4f}")
        
        # Print coefficients
        print(f"\n  {'Variable':<25} {'Coeff':>10} {'RR':>10} {'p-value':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        
        params = result.params[:-1]  # Exclude alpha
        pvalues = result.pvalues[:-1]
        
        results_data = []
        for i, name in enumerate(col_names):
            if i < len(params):
                coef = params[i]
                rr = math.exp(coef) if abs(coef) < 10 else float('inf')
                pval = pvalues[i] if i < len(pvalues) else 1.0
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  {name:<25} {coef:>10.4f} {rr:>10.4f} {pval:>10.4f} {sig}")
                results_data.append({
                    'variable': name, 'coefficient': coef,
                    'rate_ratio': rr, 'p_value': pval, 'significance': sig
                })
        
        alpha = result.params[-1]
        print(f"  {'alpha (dispersion)':<25} {alpha:>10.4f}")
        
        return {
            'result': result,
            'col_names': col_names,
            'coefficients': results_data,
            'aic': result.aic,
            'alpha': alpha,
            'n_obs': n,
            'n_events': int(y.sum()),
            'label': label,
        }
    
    except Exception as e:
        print(f"  ERROR fitting model: {e}")
        
        # Fallback: Poisson
        print(f"  Trying Poisson fallback...")
        try:
            model_p = sm.GLM(y, X, offset=offset,
                            family=sm.families.Poisson())
            result_p = model_p.fit(maxiter=100)
            
            pearson_chi2 = result_p.pearson_chi2
            phi = pearson_chi2 / (n - X.shape[1])
            
            print(f"  Poisson AIC: {result_p.aic:.1f}")
            print(f"  Overdispersion phi: {phi:.1f}")
            
            print(f"\n  {'Variable':<25} {'Coeff':>10} {'RR':>10} {'p-value':>10}")
            print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
            
            for i, name in enumerate(col_names):
                if i < len(result_p.params):
                    coef = result_p.params[i]
                    rr = math.exp(coef) if abs(coef) < 10 else float('inf')
                    pval = result_p.pvalues[i]
                    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                    print(f"  {name:<25} {coef:>10.4f} {rr:>10.4f} {pval:>10.4f} {sig}")
            
            return {
                'result': result_p,
                'col_names': col_names,
                'aic': result_p.aic,
                'phi': phi,
                'n_obs': n,
                'n_events': int(y.sum()),
                'label': label + ' (Poisson)',
            }
        except Exception as e2:
            print(f"  Poisson also failed: {e2}")
            return None


# ============================================================
# 5. RISK SCORING
# ============================================================

def compute_risk_scores(panel_rows, model_result, col_names):
    """Compute operator-level risk scores."""
    try:
        import numpy as np
    except ImportError:
        return []
    
    if model_result is None:
        return []
    
    result = model_result['result']
    params = result.params[:-1] if hasattr(result, 'params') else []
    
    # Aggregate by operator-state
    op_data = defaultdict(lambda: {
        'total_incidents': 0, 'total_corrosion': 0,
        'total_miles': 0.0, 'n_obs': 0,
        'operator_name': '', 'years': set()
    })
    
    for row in panel_rows:
        key = (row['operator_id'], row['state'])
        op_data[key]['total_incidents'] += safe_int(row.get('n_incidents'), 0) or 0
        op_data[key]['total_corrosion'] += safe_int(row.get('n_corrosion'), 0) or 0
        op_data[key]['total_miles'] += safe_float(row.get('miles_at_risk', 0))
        op_data[key]['n_obs'] += 1
        op_data[key]['operator_name'] = row.get('operator_name', '')
        op_data[key]['years'].add(row.get('year', 0))
    
    # Compute overall average rate
    total_inc = sum(d['total_incidents'] for d in op_data.values())
    total_mi = sum(d['total_miles'] for d in op_data.values())
    avg_rate = total_inc / max(total_mi, 1) * 1000  # per 1K miles
    
    risk_scores = []
    for (op_id, state), data in op_data.items():
        if data['total_miles'] < 1:
            continue
        
        obs_rate = data['total_incidents'] / data['total_miles'] * 1000
        risk_score = obs_rate / max(avg_rate, 0.0001)
        
        risk_scores.append({
            'operator_id': op_id,
            'operator_name': data['operator_name'],
            'state': state,
            'total_miles': round(data['total_miles'], 1),
            'total_incidents': data['total_incidents'],
            'total_corrosion': data['total_corrosion'],
            'obs_rate_per_1k': round(obs_rate, 4),
            'risk_score': round(risk_score, 3),
            'n_years': len(data['years']),
        })
    
    risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)
    return risk_scores


# ============================================================
# 6. OUTPUT GENERATION
# ============================================================

def save_panel_csv(panel_rows, output_path):
    """Save panel to CSV."""
    if not panel_rows:
        return
    fieldnames = list(panel_rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(panel_rows)
    print(f"  Panel saved: {output_path} ({len(panel_rows):,} rows)")


def save_risk_scores(risk_scores, output_path):
    """Save risk scores to CSV."""
    if not risk_scores:
        return
    fieldnames = list(risk_scores[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(risk_scores)
    print(f"  Risk scores saved: {output_path} ({len(risk_scores):,} operators)")


def generate_report(panel_rows, model_allcause, model_corrosion, risk_scores, output_path):
    """Generate results report in plain text."""
    
    n = len(panel_rows)
    years = sorted(set(r['year'] for r in panel_rows))
    total_events = sum(safe_int(r.get('n_incidents'), 0) or 0 for r in panel_rows)
    total_corrosion = sum(safe_int(r.get('n_corrosion'), 0) or 0 for r in panel_rows)
    total_miles = sum(safe_float(r.get('miles_at_risk', 0)) for r in panel_rows)
    
    lines = []
    lines.append("=" * 70)
    lines.append("ROSEN Project 4 -- 15-Year Panel GLM Results")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)
    
    lines.append(f"\n1. PANEL SUMMARY")
    lines.append(f"   Total observations:    {n:,}")
    lines.append(f"   Panel years:           {min(years)}-{max(years)} ({len(years)} years)")
    lines.append(f"   Years present:         {years}")
    lines.append(f"   All-cause incidents:   {total_events:,}")
    lines.append(f"   Corrosion incidents:   {total_corrosion:,}")
    lines.append(f"   Total pipe-miles:      {total_miles:,.0f}")
    lines.append(f"   Overall rate:          {total_events/max(total_miles,1)*1000:.4f} per 1K mi-yr")
    lines.append(f"   Corrosion rate:        {total_corrosion/max(total_miles,1)*1000:.5f} per 1K mi-yr")
    
    # Era breakdown
    lines.append(f"\n2. INCIDENTS BY ERA")
    era_stats = defaultdict(lambda: {'events': 0, 'corr': 0, 'miles': 0.0})
    for r in panel_rows:
        era = r.get('era', 'unknown')
        era_stats[era]['events'] += safe_int(r.get('n_incidents'), 0) or 0
        era_stats[era]['corr'] += safe_int(r.get('n_corrosion'), 0) or 0
        era_stats[era]['miles'] += safe_float(r.get('miles_at_risk', 0))
    
    lines.append(f"   {'Era':<20} {'Events':>8} {'Corr':>8} {'Miles':>12} {'Rate/1K':>10}")
    lines.append(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    for era in ['era_pre1940', 'era_50s_60s', 'era_coal_tar', 'era_improved', 'era_modern']:
        s = era_stats[era]
        rate = s['events'] / max(s['miles'], 1) * 1000
        lines.append(f"   {era:<20} {s['events']:>8,} {s['corr']:>8,} {s['miles']:>12,.0f} {rate:>10.4f}")
    
    # Age breakdown
    lines.append(f"\n3. INCIDENTS BY AGE GROUP")
    age_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120)]
    lines.append(f"   {'Age Range':<15} {'Events':>8} {'Corr':>8} {'Miles':>12} {'Rate/1K':>10}")
    lines.append(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
    for lo, hi in age_bins:
        subset = [r for r in panel_rows if lo <= safe_float(r.get('age_at_obs', 0)) < hi]
        ev = sum(safe_int(r.get('n_incidents'), 0) or 0 for r in subset)
        co = sum(safe_int(r.get('n_corrosion'), 0) or 0 for r in subset)
        mi = sum(safe_float(r.get('miles_at_risk', 0)) for r in subset)
        rate = ev / max(mi, 1) * 1000
        lines.append(f"   {f'{lo}-{hi}':<15} {ev:>8,} {co:>8,} {mi:>12,.0f} {rate:>10.4f}")
    
    # Model results
    for model_data, model_label in [(model_allcause, 'ALL-CAUSE'), (model_corrosion, 'CORROSION')]:
        if model_data is None:
            continue
        lines.append(f"\n4. {model_label} GLM RESULTS")
        lines.append(f"   N obs:     {model_data.get('n_obs', 'N/A'):,}")
        lines.append(f"   N events:  {model_data.get('n_events', 'N/A'):,}")
        lines.append(f"   AIC:       {model_data.get('aic', 'N/A'):.1f}" if model_data.get('aic') else "   AIC: N/A")
        if 'alpha' in model_data:
            lines.append(f"   Alpha:     {model_data['alpha']:.4f}")
        if 'phi' in model_data:
            lines.append(f"   Phi (OD):  {model_data['phi']:.1f}")
        
        if 'coefficients' in model_data:
            lines.append(f"\n   {'Variable':<25} {'Coeff':>10} {'RR':>10} {'p-value':>10} {'Sig':>5}")
            lines.append(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
            for c in model_data['coefficients']:
                lines.append(f"   {c['variable']:<25} {c['coefficient']:>10.4f} {c['rate_ratio']:>10.4f} {c['p_value']:>10.4f} {c['significance']:>5}")
    
    # Top risk operators
    if risk_scores:
        lines.append(f"\n5. TOP 20 HIGHEST-RISK OPERATORS")
        lines.append(f"   {'Rank':>4} {'Operator':<30} {'State':<6} {'Miles':>10} {'Inc':>5} {'Score':>8}")
        lines.append(f"   {'-'*4} {'-'*30} {'-'*6} {'-'*10} {'-'*5} {'-'*8}")
        for i, rs in enumerate(risk_scores[:20]):
            name = rs['operator_name'][:28]
            lines.append(f"   {i+1:>4} {name:<30} {rs['state']:<6} {rs['total_miles']:>10,.0f} {rs['total_incidents']:>5} {rs['risk_score']:>8.3f}")
    
    lines.append(f"\n{'=' * 70}")
    lines.append(f"End of Report")
    
    report_text = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Report saved: {output_path}")
    
    return report_text


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ROSEN P4 -- 15-Year Panel Builder + NegBin GLM')
    parser.add_argument('--data-dir', default='.', help='Directory containing survival_panel_7yr.csv')
    parser.add_argument('--part-dir', default=None, help='Directory containing Part J/H/K/M CSVs')
    parser.add_argument('--incident-dir', default=None, help='Directory containing incident CSVs')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--panel-only', action='store_true', help='Only build panel, skip GLM')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect part-dir
    part_dir = args.part_dir
    if part_dir is None:
        candidates = ['extracted_csvs', 'csvs', '.']
        for cand in candidates:
            test = os.path.join(data_dir, cand) if cand != '.' else data_dir
            if os.path.isdir(test):
                test_files = glob.glob(os.path.join(test, 'GT_AR_*_Part_J.csv'))
                if test_files:
                    part_dir = test
                    break
        if part_dir is None:
            part_dir = data_dir
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    print("=" * 65)
    print("ROSEN Project 4 -- 15-Year Panel Builder + NegBin GLM")
    print(f"Date: {timestamp}")
    print(f"Data dir:    {os.path.abspath(data_dir)}")
    print(f"Part dir:    {os.path.abspath(part_dir)}")
    print(f"Output dir:  {os.path.abspath(output_dir)}")
    print("=" * 65)
    
    # -- Step 1: Find Part J files --
    print("\n[1/6] Scanning for Part J files...")
    part_j_files = find_part_files(part_dir, 'J')
    
    if not part_j_files:
        # Try in data_dir itself
        part_j_files = find_part_files(data_dir, 'J')
    
    if not part_j_files:
        print("  ERROR: No Part J CSV files found!")
        print(f"  Searched: {part_dir}")
        print(f"  Expected: GT_AR_YYYY_Part_J.csv")
        sys.exit(1)
    
    print(f"  Found {len(part_j_files)} Part J files:")
    for yr in sorted(part_j_files.keys()):
        print(f"    {yr}: {os.path.basename(part_j_files[yr])}")
    
    # -- Step 2: Load incidents --
    print("\n[2/6] Loading incident data...")
    incident_counts = defaultdict(lambda: {
        'total': 0, 'corrosion': 0, 'ext_corr': 0, 'int_corr': 0,
        'material': 0, 'excavation': 0, 'natural': 0, 'other': 0
    })
    
    # Try loading from existing 7yr panel first
    panel_7yr_path = os.path.join(data_dir, 'survival_panel_7yr.csv')
    if os.path.exists(panel_7yr_path):
        print(f"  Loading incidents from existing panel: {panel_7yr_path}")
        incident_counts = load_incidents_from_panel(panel_7yr_path)
        print(f"  Loaded {len(incident_counts)} incident keys from panel")
    
    # Also try raw incident files
    incident_dir = args.incident_dir or data_dir
    for search_dir in [incident_dir, os.path.dirname(data_dir), os.path.join(os.path.dirname(data_dir), '..')]:
        inc_files = find_incident_files(search_dir)
        if inc_files:
            for inc_file in inc_files:
                print(f"  Also loading: {os.path.basename(inc_file)}")
                raw_incidents = load_incidents_from_csv(inc_file)
                raw_indexed = index_incidents(raw_incidents)
                for k, v in raw_indexed.items():
                    for cause_key in v:
                        incident_counts[k][cause_key] = max(incident_counts[k][cause_key], v[cause_key])
            break
    
    total_inc_events = sum(v['total'] for v in incident_counts.values())
    total_corr_events = sum(v['corrosion'] for v in incident_counts.values())
    print(f"  Total incident keys: {len(incident_counts)}")
    print(f"  Total events: {total_inc_events}, Corrosion: {total_corr_events}")
    
    # -- Step 3: Build 15-year panel --
    print("\n[3/6] Building 15-year survival panel...")
    panel_rows = build_panel(part_j_files, incident_counts)
    
    total_events = sum(1 for r in panel_rows if r['event'] == 1)
    total_corrosion = sum(r['n_corrosion'] for r in panel_rows)
    print(f"\n  PANEL TOTALS:")
    print(f"    Observations:     {len(panel_rows):,}")
    print(f"    Years:            {min(r['year'] for r in panel_rows)}-{max(r['year'] for r in panel_rows)}")
    print(f"    All-cause events: {total_events:,}")
    print(f"    Corrosion events: {total_corrosion:,}")
    
    # -- Step 4: Integrate covariables --
    print("\n[4/6] Integrating covariables from Parts H, K, M...")
    panel_rows = integrate_covariables(panel_rows, part_dir)
    
    # -- Step 5: Save panel --
    print("\n[5/6] Saving 15-year panel...")
    panel_path = os.path.join(output_dir, 'survival_panel_15yr.csv')
    save_panel_csv(panel_rows, panel_path)
    
    if args.panel_only:
        print("\n  --panel-only flag set, skipping GLM.")
        print("  Done!")
        return
    
    # -- Step 6: Fit GLM --
    print("\n[6/6] Fitting Negative Binomial GLMs...")
    
    # All-cause model
    model_allcause = fit_negbin_glm(panel_rows, response='n_incidents', label='All-Cause')
    
    # Corrosion model
    model_corrosion = fit_negbin_glm(panel_rows, response='n_corrosion', label='Corrosion')
    
    # Risk scores
    print("\n  Computing operator risk scores...")
    risk_scores = compute_risk_scores(panel_rows, model_allcause, 
                                       model_allcause.get('col_names', []) if model_allcause else [])
    
    # Save risk scores
    risk_path = os.path.join(output_dir, 'operator_risk_scores_15yr.csv')
    save_risk_scores(risk_scores, risk_path)
    
    # Generate report
    report_path = os.path.join(output_dir, 'glm_results_15yr.txt')
    report_text = generate_report(panel_rows, model_allcause, model_corrosion, risk_scores, report_path)
    
    # Print summary
    print("\n" + "=" * 65)
    print("COMPLETE")
    print("=" * 65)
    print(f"  Panel:       {panel_path}")
    print(f"  Risk scores: {risk_path}")
    print(f"  Report:      {report_path}")
    
    if model_allcause:
        print(f"\n  All-cause AIC: {model_allcause.get('aic', 'N/A'):.1f}")
    if model_corrosion:
        print(f"  Corrosion AIC: {model_corrosion.get('aic', 'N/A'):.1f}")
    
    print(f"\n  15-year panel: {len(panel_rows):,} obs vs 7-year: 34,226 obs")
    print(f"  Expected improvement: ~2x sample size, ~2x corrosion events")
    print(f"  This should reduce coefficient CIs by ~30% (1/sqrt(2))")


if __name__ == '__main__':

    main()
