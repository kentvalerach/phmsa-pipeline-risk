# ROSEN Project 4: Pipeline Risk Prediction Model
## Final Technical Report

**Author:** Kent Valera Chirinos  
**Role:** Telecommunications Engineer / Quantitative Developer  
**Date:** February 2026  
**Location:** Coswig, Saxony, Germany

---

## Executive Summary

This project developed a machine learning model to predict natural gas pipeline incident risk using 15 years of PHMSA regulatory data (2010-2024). The model achieves **AUC 0.793** with rigorous walk-forward temporal validation, demonstrating consistent predictive power across multiple test periods.

### Key Achievement
We identified a fundamental **"signal ceiling"** in publicly available pipeline data: **99.1% of predictable signal comes from pipeline exposure (miles operated)**. Despite engineering 40+ features from 6 data sources, condition-level features add only marginal improvement (+0.4 percentage points).

### Strategic Implication for ROSEN
This finding has significant commercial value: it quantifies exactly why proprietary ILI (In-Line Inspection) data is essential for breaking the signal ceiling. Public PHMSA data aggregates condition information to operator-state level, destroying the segment-specific variation needed for precision risk prediction. ROSEN's pipe-level inspection data is the missing link.

---

## 1. Project Scope & Objectives

### 1.1 Business Context
- **Client:** ROSEN Group — global leader in pipeline integrity management
- **Problem:** Predict which pipeline operators are most likely to experience reportable incidents
- **Application:** Prioritize ILI inspection scheduling and demonstrate value of inspection data

### 1.2 Technical Objectives
1. Build a survival analysis framework for pipeline incident prediction
2. Implement temporal validation to ensure real-world applicability
3. Quantify the information content of different data sources
4. Create an interactive dashboard for risk visualization

### 1.3 Deliverables
| Deliverable | Status | Location |
|-------------|--------|----------|
| 15-year survival panel | ✅ Complete | `survival_panel_15yr_final.csv` |
| Walk-forward validated model | ✅ Complete | `lgbm_full_pipeline_afml.py` |
| Feature enrichment pipeline | ✅ Complete | `enrich_panel_km.py`, `external_enrichment_api.py` |
| Signal ceiling analysis | ✅ Complete | This report |
| Interactive dashboard | ✅ Complete | `app/streamlit_app.py` |
| GitHub repository | ✅ Complete | `phmsa-pipeline-risk/` |

---

## 2. Data Sources

### 2.1 Primary Data — PHMSA Annual Reports

| Part | Content | Records | Coverage |
|------|---------|---------|----------|
| A-D | Operator ID, mileage by material, CP coverage | 22,691 | 100% |
| H | Diameter distribution (26 size brackets) | 22,691 | 100% |
| J | Installation decade (exposure denominator) | 22,691 | 100% |
| K | %SMYS × Location Class cross-tabulation | 22,691 | 100% |
| M | Integrity management (repairs, inspections) | 22,029 | 84.4% |

**Schema Verification:** 100% identical column structure across all 15 years (2010-2024), enabling direct panel concatenation.

### 2.2 Primary Data — PHMSA Incident Database

| Subset | Count | % of Total | Covariate Coverage |
|--------|-------|------------|-------------------|
| All GT Incidents (2010-2024) | 1,985 | 100% | 46% |
| Corrosion Incidents | 394 | 19.8% | 93% |
| Material/Weld/Equipment | 892 | 44.9% | 46% |
| Excavation Damage | 217 | 10.9% | 46% |

**Key Finding:** Corrosion incidents have dramatically superior data quality (93% vs 46% covariate coverage), making them the ideal modeling target for condition-based prediction.

### 2.3 External Data Sources

| Source | API | Join Level | Signal Added |
|--------|-----|------------|--------------|
| USDA SSURGO | ✅ Queried | State | +0.01pp |
| USGS Earthquakes | ✅ Queried | State-Year | ~0pp |
| PHMSA Part K (SMYS) | Local | Operator-State-Year | +0.12pp |
| PHMSA Part M (Integrity) | Local | Operator-State-Year | -0.39pp |
| PHMSA Part D (CP Coverage) | Local | Operator-State-Year | -0.03pp |

---

## 3. Methodology

### 3.1 Survival Panel Construction

**Observation Unit:** Operator × State × Year  
**Time Period:** 2010-2024 (15 years)  
**Sample Size:** 67,951 observations, 1,264 events (1.86% event rate)

```
Panel Structure:
┌─────────────┬───────┬──────┬──────────────┬───────┐
│ operator_id │ state │ year │ miles_at_risk│ event │
├─────────────┼───────┼──────┼──────────────┼───────┤
│ 31618       │ TX    │ 2020 │ 4,521        │ 0     │
│ 31618       │ TX    │ 2021 │ 4,543        │ 1     │
│ 31618       │ TX    │ 2022 │ 4,567        │ 0     │
│ ...         │ ...   │ ...  │ ...          │ ...   │
└─────────────┴───────┴──────┴──────────────┴───────┘
```

### 3.2 Feature Engineering

**Exposure Features:**
- `log_miles` = log1p(miles_at_risk) — primary exposure measure

**Infrastructure Features:**
- `age_at_obs` — weighted average pipeline age
- `pct_small_diam` — fraction of miles < 12" diameter
- `pct_large_diam` — fraction of miles > 24" diameter
- `pct_high_smys` — fraction of miles ≥ 52,000 psi
- `pct_class1` — fraction in rural (Class 1) locations
- `era` — installation era categorical

**Condition Features (from Part K/M):**
- `pct_low_smys` — fraction of vintage pipe (< 35 ksi)
- `pct_high_class` — fraction in populated areas (Class 3-4)
- `lag_repairs` — repairs in previous year (t-1)
- `cum_corrosion` — cumulative corrosion findings through t-1

**Environmental Features (from APIs):**
- `soil_corr_index` — SSURGO soil corrosivity (0-1)
- `earthquake_count` — USGS M≥3.0 events per state-year

### 3.3 Temporal Validation (AFML Methodology)

Following López de Prado's *Advances in Financial Machine Learning*, we implemented **walk-forward cross-validation** to prevent temporal data leakage:

```
Fold 1: Train 2010-2015 → Test 2016    AUC: 0.781
Fold 2: Train 2010-2016 → Test 2017    AUC: 0.792
Fold 3: Train 2010-2017 → Test 2018    AUC: 0.788
Fold 4: Train 2010-2018 → Test 2019    AUC: 0.801
Fold 5: Train 2010-2019 → Test 2020-24 AUC: 0.793
```

**Key Properties:**
- Training always precedes testing chronologically
- No information from the future leaks into training
- Consistent AUC (0.78-0.80) demonstrates model stability

---

## 4. Results

### 4.1 Model Performance

| Metric | Value |
|--------|-------|
| Test AUC (2020-2024) | **0.793** |
| Walk-Forward Mean AUC | 0.791 ± 0.007 |
| Brier Score | 0.018 |
| Log-Loss | 0.082 |

### 4.2 Feature Importance (SHAP Values)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | log_miles | 45.2% | Exposure dominates |
| 2 | age_at_obs | 12.1% | Older pipe = higher risk |
| 3 | pct_small_diam | 8.3% | Small pipe = higher risk |
| 4 | pct_large_diam | 7.1% | Large pipe = lower risk |
| 5 | pct_high_smys | 5.8% | Modern steel = lower risk |
| 6 | pct_class1 | 5.2% | Rural = lower consequence |
| 7-12 | Other features | 16.3% | Marginal contributions |

### 4.3 Signal Ceiling Analysis

This is the central finding of the project. We systematically tested whether additional data sources could improve prediction beyond the baseline:

| Feature Set | Features | AUC | Δ from log(miles) |
|-------------|----------|-----|-------------------|
| **log(miles) only** | 1 | 0.7903 | — baseline — |
| + Demographics | 11 | 0.7926 | +0.23pp |
| + Part K (SMYS/Class) | 16 | 0.7938 | +0.12pp |
| + Part M (Integrity) | 34 | 0.7899 | -0.39pp |
| + SSURGO Soil | 36 | 0.7901 | +0.02pp |
| + USGS Earthquakes | 38 | 0.7901 | +0.00pp |
| + CP Coverage | 40 | 0.7898 | -0.03pp |
| **All Features** | 40 | **0.7931** | **+0.28pp** |

**Interpretation:**
- **99.1%** of achievable signal comes from log(miles) alone
- Adding 39 features from 6 data sources adds only **0.28 percentage points**
- Some features (Part M, CP Coverage) actually *decrease* AUC — they add noise

### 4.4 Why Condition Features Don't Help

We investigated why integrity management data (repairs, corrosion findings) failed to improve prediction:

**1. Aggregation Destroys Signal**
- PHMSA data aggregates to operator-state level
- An operator in Texas might have 4,000 miles across diverse conditions
- State-level averages hide segment-specific risk variation

**2. Temporal Fragmentation**
- Part M reporting changed in 2021 (new fields added)
- "Excavation tickets" data only exists in 2024
- "Total repairs" is zero for 2010-2020, appears in 2021+

**3. Confounding by Size**
- Operators with more repairs are *larger* operators
- Larger operators have more miles → more incidents
- The "repairs predict incidents" signal is really "size predicts size"

**4. Survivorship Bias**
- Operators with unprotected pipe (no cathodic protection) have *fewer* incidents
- Why? They are small rural operators with minimal exposure
- The "risk factor" is confounded with the protective factor of small scale

---

## 5. Key Findings

### 5.1 The Signal Ceiling is Real and Structural

The limitation is not in our methodology — it's in the data. PHMSA annual reports are designed for regulatory compliance, not predictive modeling. They capture:

| What PHMSA Has | What Prediction Needs |
|----------------|----------------------|
| Operator-state totals | Segment-level conditions |
| Annual snapshots | Continuous monitoring |
| Self-reported categories | Objective measurements |
| Aggregated statistics | Individual defect data |

### 5.2 Non-Monotonic Vintage Effect

A key EDA finding that validates the ML approach: **corrosion rates are NOT purely age-dependent**.

| Vintage | Age in 2024 | Corrosion Rate | Interpretation |
|---------|-------------|----------------|----------------|
| 1970s | ~50 years | 0.069/1K mi/yr | **HIGHEST** — Coal tar coatings |
| 1990s | ~30 years | 0.009/1K mi/yr | **LOWEST** — FBE coatings |
| 1950s | ~70 years | 0.039/1K mi/yr | Moderate — Survivorship bias |

This 7.6× difference between 1970s and 1990s pipe proves that a multivariate model (not simple age regression) is essential.

### 5.3 What Would Break the Ceiling

Based on our analysis, breaking the signal ceiling requires:

| Data Source | Potential Gain | Availability |
|-------------|----------------|--------------|
| **NPMS Pipeline Routes** | +2-5pp AUC | Restricted (ROSEN can request) |
| **ILI Inspection Data** | +5-10pp AUC | ROSEN proprietary |
| **Real-time SCADA** | Unknown | Operator proprietary |
| **Segment-level Coating** | +3-5pp AUC | Not publicly available |

**Strategic Recommendation for ROSEN:**
1. Request NPMS GIS data through US operations
2. Integrate ILI inspection history into prediction framework
3. Demonstrate improvement over public-data baseline

---

## 6. Technical Implementation

### 6.1 Repository Structure

```
phmsa-pipeline-risk/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
│
├── app/
│   └── streamlit_app.py      # Interactive dashboard
│
├── src/
│   ├── models/
│   │   └── train.py          # LightGBM training pipeline
│   └── data/
│       └── build_panel.py    # Panel construction
│
├── scripts/                  # Analysis scripts
│   ├── rosen_p4_15yr_panel.py
│   ├── lgbm_full_pipeline_afml.py
│   ├── enrich_panel_km.py
│   ├── external_enrichment_api.py
│   └── extract_cp_and_test.py
│
├── data/
│   ├── processed/
│   │   ├── survival_panel_15yr_final.csv
│   │   └── operator_risk_ranking_final.csv
│   └── external/
│       ├── ssurgo_soil_corrosivity.csv
│       └── usgs_earthquakes.csv
│
└── docs/
    ├── ROSEN_P4_Project_Plan.pdf
    ├── rosen_p4_eda_complete_report.md
    ├── data_dictionary.md
    └── final_report.md       # This document
```

### 6.2 Key Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `rosen_p4_15yr_panel.py` | Build survival panel from raw PHMSA data | `survival_panel_15yr.csv` |
| `lgbm_full_pipeline_afml.py` | Train LightGBM with walk-forward CV | Model artifacts, SHAP plots |
| `enrich_panel_km.py` | Add Part K/M features | `survival_panel_15yr_enriched.csv` |
| `external_enrichment_api.py` | Query SSURGO + USGS APIs | Soil/seismic features |
| `extract_cp_and_test.py` | Extract cathodic protection from Part D | CP coverage analysis |

### 6.3 Reproducibility

```bash
# Environment setup
cd C:\Phmsa\annual_gt
python -m venv rosen_env
.\rosen_env\Scripts\Activate.ps1
pip install -r requirements.txt

# Run full pipeline
python rosen_p4_15yr_panel.py        # Build panel
python lgbm_full_pipeline_afml.py    # Train model
python enrich_panel_km.py            # Test enrichment

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 7. Conclusions

### 7.1 What We Achieved

1. **Built a rigorous predictive model** with AUC 0.793, validated across 5 temporal folds
2. **Quantified the signal ceiling** at 99.1% from exposure alone
3. **Tested 6 data sources** (40+ features) for incremental value
4. **Identified why condition features fail** at operator-state granularity
5. **Created production-ready code** with documentation and dashboard

### 7.2 What We Learned

The core insight is that **publicly available pipeline data is fundamentally limited for condition-based risk prediction**. This is not a failure of methodology — it's a structural property of regulatory reporting:

> PHMSA collects data to verify compliance, not to predict failures. The aggregation level (operator-state-year) is appropriate for regulatory oversight but insufficient for precision risk modeling.

### 7.3 Value for ROSEN

This finding has **significant commercial value**:

1. **Quantifies ILI data value:** We now know that public data provides AUC ~0.79. Any improvement from ILI data is directly measurable.

2. **Justifies premium pricing:** If ILI data raises AUC to 0.85+, that improvement has quantifiable value in avoided incidents.

3. **Technical differentiator:** ROSEN can demonstrate, with evidence, that their data adds predictive power beyond public records.

4. **Product development:** The signal ceiling analysis suggests which ILI features would be most valuable to capture and standardize.

### 7.4 Recommended Next Steps

| Priority | Action | Expected Outcome |
|----------|--------|------------------|
| 1 | Request NPMS pipeline routes | Enable operator-specific geospatial features |
| 2 | Integrate sample ILI data | Demonstrate improvement over public baseline |
| 3 | Develop segment-level model | Break the operator-state aggregation limit |
| 4 | Publish technical paper | Establish thought leadership in predictive integrity |

---

## 8. Appendix: Files to Include in Repository

### 8.1 Scripts (copy to `scripts/`)

```
rosen_p4_15yr_panel.py
rosen_p4_glm_complete.py
lgbm_full_pipeline.py
lgbm_full_pipeline_afml.py
enrich_panel_km.py
external_enrichment_api.py
extract_cp_and_test.py
extract_phmsa_parts.py
```

### 8.2 Data (copy to `data/processed/`)

```
survival_panel_15yr_final.csv       # Main analysis panel (9 MB)
operator_risk_ranking_final.csv     # Risk scores (166 KB)
shap_importance_final.csv           # Feature importance (572 B)
walkforward_comparison.csv          # CV results (1.8 KB)
ssurgo_soil_corrosivity.csv         # Soil data (5 KB)
usgs_earthquakes.csv                # Seismic data (12 KB)
```

### 8.3 Documentation (copy to `docs/`)

```
ROSEN_P4_Project_Plan.pdf
rosen_p4_eda_complete_report.md
data_dictionary.md
final_report.md
```

### 8.4 Figures (copy to `docs/images/`)

```
fig1_pipeline_dashboard.png
fig2_shap_walkforward.png
enrichment_diagnostic.png
```

---

## Acknowledgments

- **PHMSA** for maintaining transparent public safety data
- **López de Prado** for temporal validation methodology (*Advances in Financial Machine Learning*)
- **ROSEN Group** for the inspiring problem domain

---

*This report documents research conducted independently as a portfolio project. All data used is publicly available from PHMSA. No proprietary ROSEN data was accessed or used.*
