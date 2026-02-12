# Data Dictionary

## Survival Panel (`survival_panel_15yr_final.csv`)

The core analysis dataset with one row per operator-state-year observation.

### Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `operator_id` | int | PHMSA unique operator identifier |
| `state` | str | U.S. state name (uppercase) |
| `year` | int | Observation year (2010-2024) |

### Target Variable

| Column | Type | Description |
|--------|------|-------------|
| `event` | int | Binary indicator: 1 if any reportable incident occurred, 0 otherwise |
| `n_incidents` | int | Count of reportable incidents in this operator-state-year |

### Exposure Features

| Column | Type | Description |
|--------|------|-------------|
| `miles_at_risk` | float | Total pipeline miles operated |
| `log_miles` | float | `log1p(miles_at_risk)` - primary exposure measure |

### Infrastructure Features

| Column | Type | Description |
|--------|------|-------------|
| `age_at_obs` | float | Weighted average pipeline age in years |
| `pct_small_diam` | float | Fraction of miles with diameter < 12 inches |
| `pct_large_diam` | float | Fraction of miles with diameter > 24 inches |
| `pct_high_smys` | float | Fraction of miles with SMYS ≥ 52,000 psi |
| `pct_low_smys` | float | Fraction of miles with SMYS < 35,000 psi |
| `pct_class1` | float | Fraction of miles in Class 1 location |
| `pct_high_class` | float | Fraction of miles in Class 3 or 4 location |
| `era` | str | Installation era category: 'pre1970', '1970s', '1980s', '1990s', '2000s', '2010s' |

### Condition Features (Lagged)

| Column | Type | Description |
|--------|------|-------------|
| `lag_repairs_cl12` | int | Class 1&2 repairs in year t-1 |
| `lag_repairs_hca` | int | HCA repairs in year t-1 |
| `lag_ext_corrosion` | int | External corrosion findings in year t-1 |
| `cum_total_repairs` | int | Cumulative repairs through year t-1 |
| `log1p_cum_corrosion` | float | Log of cumulative corrosion findings |

### External Enrichment Features

| Column | Type | Description |
|--------|------|-------------|
| `soil_corr_index` | float | State-level soil corrosivity index (0-1) from SSURGO |
| `pct_high_corr` | float | Fraction of state with high corrosivity soils |
| `earthquake_count` | int | Number of M≥3.0 earthquakes in state-year |
| `log_seismic_energy` | float | Log of cumulative seismic energy in state-year |

---

## Incident Database (`flagged_gas_transmission_incidents.csv`)

PHMSA reportable incidents joined with flags.

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `REPORT_NUMBER` | str | Unique incident report identifier |
| `OPERATOR_ID` | int | PHMSA operator identifier |
| `ACCIDENT_STATE` | str | State where incident occurred |
| `LOCAL_DATETIME` | datetime | Date and time of incident |
| `CAUSE` | str | Primary cause category |
| `FATALITIES` | int | Number of fatalities |
| `INJURIES` | int | Number of injuries |
| `PROPERTY_DAMAGE` | float | Property damage in dollars |

### Cause Categories

| Cause Code | Description |
|------------|-------------|
| `CORROSION` | External or internal corrosion |
| `EXCAVATION DAMAGE` | Third-party dig-in |
| `MATERIAL/WELD/EQUIP FAILURE` | Manufacturing or construction defect |
| `NATURAL FORCE DAMAGE` | Earth movement, weather, etc. |
| `OTHER OUTSIDE FORCE DAMAGE` | Vehicle strikes, vandalism, etc. |
| `INCORRECT OPERATION` | Human error |
| `ALL OTHER CAUSES` | Miscellaneous |

---

## Annual Report Parts

### Part A-D: Operator Information & Mileage

| Column | Description |
|--------|-------------|
| `OPERATOR_ID` | Operator identifier |
| `PARTA2NAMEOFCOMP` | Company name |
| `STATE_NAME` | State |
| `PARTDTOTALMILES` | Total miles by material |
| `PARTDTCPBTOTAL` | Miles with cathodic protection (bare) |
| `PARTDTCPCTOTAL` | Miles with cathodic protection (coated) |
| `PARTDTCUBTOTAL` | Miles unprotected (bare) |
| `PARTDTCUCTOTAL` | Miles unprotected (coated) |

### Part H: Diameter Distribution

| Column | Description |
|--------|-------------|
| `PARTHON4LESS` | Miles with OD ≤ 4 inches |
| `PARTHON6` | Miles with OD 4-6 inches |
| ... | (continues by diameter) |
| `PARTHON42PLUS` | Miles with OD > 42 inches |

### Part J: Installation Decade

| Column | Description |
|--------|-------------|
| `PARTJON1940PRIOR` | Miles installed before 1940 |
| `PARTJON1940` | Miles installed 1940-1949 |
| ... | (continues by decade) |
| `PARTJON2020` | Miles installed 2020-2029 |
| `PARTJONUNK` | Miles with unknown install date |

### Part K: SMYS and Location Class

| Column | Description |
|--------|-------------|
| `PARTKCL1LT30` | Class 1, SMYS < 30 ksi |
| `PARTKCL1GE30LT42` | Class 1, SMYS 30-42 ksi |
| ... | (continues by class/SMYS) |
| `PARTKCL4GE70` | Class 4, SMYS ≥ 70 ksi |

### Part M: Integrity Management

| Column | Description |
|--------|-------------|
| `PARTMREPAIRSCL12` | Class 1&2 repairs |
| `PARTMREPAIRSHCA` | HCA repairs |
| `PARTMEXTCORR` | External corrosion findings |
| `PARTMINTCORR` | Internal corrosion findings |
| `PARTMSCC` | Stress corrosion cracking findings |
| `PARTMDAMAGES` | Third-party damage findings |

---

## Model Outputs

### Risk Rankings (`operator_risk_ranking_final.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `operator_id` | int | Operator identifier |
| `operator_name` | str | Company name |
| `state` | str | State |
| `year` | int | Observation year |
| `risk_score` | float | Predicted incident probability (calibrated) |
| `risk_rank` | int | Rank within year (1 = highest risk) |
| `risk_tier` | str | Risk category: 'Very High', 'High', 'Medium', 'Low', 'Very Low' |
| `actual_event` | int | Whether incident actually occurred |

### Feature Importance (`shap_importance_final.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `feature` | str | Feature name |
| `importance` | float | Mean absolute SHAP value |
| `importance_pct` | float | Percentage of total importance |
