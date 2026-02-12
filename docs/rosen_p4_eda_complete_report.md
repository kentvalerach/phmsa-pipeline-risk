# ROSEN Project 4: Predictive Maintenance with Survival Models
## Complete Exploratory Data Analysis & Model Feasibility Report

**Autor:** Kent — Telecommunications Engineer / ML Practitioner  
**Fecha:** 3 febrero 2026  
**Datasets:** PHMSA Flagged Incidents + Annual Reports (Gas Transmission & Gathering)

---

## Executive Summary

This report documents the complete exploratory data analysis of PHMSA pipeline safety data for building a survival model that predicts pipeline failure risk as a function of pipe age, material properties, and operational characteristics. The analysis confirms strong feasibility with a key insight: **corrosion failure rates are NOT purely age-dependent** — 1970s-vintage pipe fails at 0.069/1K miles/year while 1990s pipe fails at only 0.009/1K miles/year. This non-monotonic pattern proves that a multivariate ML model (not simple age regression) is essential, validating ROSEN's value proposition of combining ILI inspection data with public incident records.

---

## 1. Dataset Inventory

### 1.1 Flagged Incidents (Events)
- **Source:** PHMSA_Pipeline_Safety_Flagged_Incidents.zip
- **Scope:** 21,134 incidents from 1986 to January 2026
- **Primary dataset:** Gas Transmission 2010–Present (1,985 incidents × 637 columns)
- **Corrosion subset:** 394 incidents with 93% covariate coverage

### 1.2 Annual Reports (Exposure / Denominator)
- **Source:** annual_gas_transmission_gathering_2010_present.zip
- **Scope:** 15 annual files (2010–2024), ~1,400 operators, 54 states
- **Schema compatibility:** ✅ 100% identical across 2017–2024 (verified)
- **Parts analyzed:** A-D (material), H (diameter), J (installation decade), K (%SMYS × class), M (leaks by cause)

### 1.3 Scale

| Metric | Value |
|--------|-------|
| Transmission pipeline miles | 300,863 (2024) |
| Gathering pipeline miles | 591,222 (2024) |
| Unique operators | 1,404 |
| Annual incidents (GT) | ~95/year |
| Annual corrosion incidents | ~12/year |
| Panel observations (4 years) | 19,282 |
| Projected full panel (15 years) | ~72,000 |

---

## 2. Event Analysis (Flagged Incidents)

### 2.1 Cause Distribution

| Cause | Count | % | ILI Detectable? |
|-------|-------|---|-----------------|
| Material/Weld/Equipment | 892 | 44.9% | ✅ Partially |
| **Corrosion** | **394** | **19.8%** | **✅ Yes — core ILI** |
| Excavation Damage | 217 | 10.9% | ❌ |
| Natural Force | 146 | 7.4% | ❌ |
| Incorrect Operation | 133 | 6.7% | ❌ |
| Other Outside Force | 114 | 5.7% | ❌ |
| All Other | 89 | 4.5% | — |

Corrosion + Material/Weld = 64.7% of all incidents — precisely what ROSEN's ILI technology detects.

### 2.2 Corrosion Failures — Superior Data Quality

The corrosion subset has dramatically better covariate coverage than the general dataset:

| Variable | General | Corrosion | Δ |
|----------|---------|-----------|---|
| Pipe diameter | 46% | 93% | +47pp |
| Wall thickness | 46% | 93% | +47pp |
| SMYS | 45% | 93% | +48pp |
| Coating type | 46% | 93% | +47pp |
| Seam type | 46% | 93% | +47pp |
| Installation year | 97% | 99% | +2pp |
| Location (lat/lon) | 60% | 100% | +40pp |

### 2.3 Age at Failure — Bathtub Curve Confirmed

From individual incident records (installation year known for 87% general, 95% corrosion):

| Group | Mean Age | Median | Range |
|-------|----------|--------|-------|
| All causes | 38.6 years | 42 | 0–112 |
| Corrosion only | 46.4 years | 46 | 6–95 |

The distribution is bimodal: infant mortality peak at 0–9 years (18.4%) plus degradation peak at 40–69 years (46.2%).

---

## 3. Exposure Analysis (Annual Reports)

### 3.1 National Pipeline Age Profile (Transmission, 2024)

| Installation Decade | Miles | % Network | Approx. Age |
|--------------------|-------|-----------|-------------|
| Pre-1940 | 7,545 | 2.5% | 84+ |
| 1940–49 | 19,495 | 6.5% | 75–84 |
| **1950–59** | **62,943** | **20.9%** | **65–74** |
| **1960–69** | **67,046** | **22.3%** | **55–64** |
| 1970–79 | 29,035 | 9.7% | 45–54 |
| 1980–89 | 24,400 | 8.1% | 35–44 |
| 1990–99 | 29,363 | 9.8% | 25–34 |
| 2000–09 | 27,852 | 9.3% | 15–24 |
| 2010–19 | 22,902 | 7.6% | 5–14 |
| 2020–29 | 8,461 | 2.8% | 0–4 |

**Critical finding:** 43.2% of the transmission network was installed in the 1950s–60s. These 130,000 miles of pipe are now 55–74 years old — entering the peak corrosion risk zone.

### 3.2 Network Trend 2017→2024

Pre-1940 pipe is declining by -16.7% (retirement/replacement), while 2010–19 vintage grew +37.9% (new construction). The network is slowly modernizing but the 1950s–60s cohort remains dominant.

### 3.3 %SMYS × Class Location Cross-Tabulation (Part K)

| %SMYS | Class 1 | Class 2 | Class 3 | Class 4 | Total |
|-------|---------|---------|---------|---------|-------|
| <20% | 13,485 | 2,859 | 3,771 | 24 | 20,139 |
| 20–29% | 16,141 | 3,814 | 6,182 | 301 | 26,438 |
| 30–40% | 19,403 | 3,845 | 6,446 | 388 | 30,083 |
| 41–50% | 31,324 | 6,358 | 11,559 | 28 | 49,268 |
| 51–60% | 36,837 | 6,173 | 4,122 | 1 | 47,133 |
| **61–72%** | **103,381** | **6,425** | **420** | **0** | **110,226** |
| 73–80% | 9,800 | 127 | 16 | 0 | 9,943 |

The 61–72% SMYS bracket dominates Class 1 (rural) locations with 103K miles — these are the large-diameter, high-pressure transmission lines that are ROSEN's primary ILI market.

### 3.4 Leak/Repair History (Part M, 2024)

| Cause | Transmission | Gathering |
|-------|-------------|-----------|
| Equipment | 747 | 464 |
| External Corrosion | 155 | 337 |
| Construction | 105 | 15 |
| Other | 52 | 13 |
| Internal Corrosion | 40 | 209 |
| Incorrect Operation | 38 | 44 |
| Natural Force | 37 | 10 |
| Material/Weld | 36 | 9 |

---

## 4. Survival Model — Proof of Concept

### 4.1 Panel Structure

The survival table joins Annual Reports (exposure) with Flagged Incidents (events) by the key `(OPERATOR_ID, STATE_NAME, REPORT_YEAR, DECADE_BIN)`:

```
Unit of observation: (operator × state × installation_decade × year)
Time variable: age = report_year - decade_midpoint
Exposure: miles_at_risk (Poisson offset)
Event: incident count (0, 1, 2, ...)
```

### 4.2 Empirical Hazard Rates (Pooled 2017+2018+2021+2024)

**All-cause failure rate by age bracket:**

| Age Bracket | Miles (×4yr) | Events | Rate /1K mi/yr |
|------------|-------------|--------|---------------|
| 0–5 | 44,673 | 47 | **1.052** |
| 6–15 | 102,831 | 60 | 0.583 |
| 16–25 | 114,923 | 32 | 0.278 |
| 26–35 | 108,358 | 32 | 0.295 |
| 36–45 | 107,921 | 26 | 0.241 |
| 46–55 | 194,331 | 45 | 0.232 |
| 56–65 | 265,654 | 66 | 0.248 |
| 66–75 | 168,238 | 36 | 0.214 |
| 76–85 | 39,474 | 6 | 0.152 |
| 86+ | 33,396 | 12 | 0.359 |

The all-cause bathtub curve is clear: infant mortality (1.05/1K) → stable phase (0.24/1K) → slight increase in oldest pipe.

### 4.3 ⭐ KEY FINDING: Corrosion Rate by Installation Vintage

**Corrosion failure rate by decade of installation (the central result):**

| Vintage | Miles (×4yr) | Corr Events | Rate /1K mi/yr |
|---------|-------------|-------------|----------------|
| 2020–29 | 12,013 | 0 | 0.000 |
| 2010–19 | 81,902 | 5 | 0.061 * |
| 2000–09 | 113,060 | 3 | 0.027 |
| **1990–99** | **117,517** | **1** | **0.009** |
| 1980–89 | 98,215 | 3 | 0.031 |
| **1970–79** | **116,378** | **8** | **0.069** |
| 1960–69 | 271,655 | 12 | 0.044 |
| 1950–59 | 258,494 | 10 | 0.039 |
| 1940–49 | 80,800 | 4 | 0.050 |
| Pre-1940 | 33,396 | 1 | 0.030 |

*\* 2010-19 rate of 0.061 is likely noise from only 5 events — needs more years to stabilize.*

**This pattern is NOT monotonic.** The 1970s vintage has the highest corrosion rate (0.069/1K), while 1990s has the lowest (0.009/1K) — a 7.6× difference. This reflects:
- **1970s:** Coal tar enamel coatings, early cathodic protection systems, known coating degradation issues
- **1990s:** Modern fusion-bonded epoxy (FBE) coatings, effective cathodic protection, better installation practices
- **Pre-1940:** Surprisingly LOW rate — survivorship bias (the weakest pipes already failed and were replaced)

**Implication:** A simple age-based model would be WRONG. The model requires covariables (coating type, material, CP status, %SMYS) to capture the vintage × technology interactions. This is exactly what justifies ML over simple regression, and what makes ROSEN's ILI data uniquely valuable.

---

## 5. Data Quality Assessment

### 5.1 Strengths
- ✅ Schema 100% identical across 2017–2024 (Part J verified)
- ✅ 93% covariate coverage for corrosion incidents
- ✅ 300K miles denominator with <1% unknown installation decade
- ✅ 1,400 operators × 54 states = rich cross-sectional variation
- ✅ Join key (OPERATOR_ID × STATE) matches 93% of incident records

### 5.2 Limitations
- ⚠️ No cross-tabulation of material × diameter × age in Annual Reports (marginals only)
- ⚠️ Gathering data: 85% unknown installation decade (focus on Transmission)
- ⚠️ Only 47 corrosion events in 4-year panel (need full 15-year panel for power)
- ⚠️ Decade bins are coarse (10-year granularity for age variable)
- ⚠️ Possible duplicate reports (INITIAL + SUPPLEMENTAL per operator)

### 5.3 Mitigations
- Full 15-year panel will provide ~175 corrosion events (adequate for 10–15 model features)
- Ecological inference techniques can recover joint distributions from marginals
- ROSEN's ILI data provides pipe-level detail that PHMSA data lacks

---

## 6. Model Design Specification

### 6.1 Recommended Model: Poisson GLM → Random Survival Forest

**Phase 1 (MVP): Poisson GLM with offset**
```
log(E[incidents]) = log(miles) + β₁·age + β₂·vintage + β₃·material_mix
                    + β₄·diameter_mix + β₅·smys_class + β₆·state + ε
```

**Phase 2: Random Survival Forest**
- Captures nonlinear interactions (coating × age, diameter × %SMYS)
- Variable importance identifies key risk drivers
- Partial dependence plots for interpretability

**Phase 3: DeepSurv Neural Network**
- State of the art for competing risks
- Cause-specific hazard models (corrosion vs material vs excavation)

### 6.2 Feature Matrix

| Feature | Source | Type | Coverage |
|---------|--------|------|----------|
| Pipe age (decade midpoint) | Part J | Continuous | 99.4% |
| Material mix (9 types) | Part D | Compositional | 100% |
| Diameter distribution (26 sizes) | Part H | Compositional | 100% |
| %SMYS × Class location | Part K | Cross-tab | 100% |
| HCA ratio | Part B | Continuous [0,1] | 100% |
| Inspection coverage | Part B | Continuous [0,1] | 100% |
| Leak history by cause | Part M | Count | 100% |
| Operator size (total miles) | Part D | Continuous | 100% |
| State/Region | All parts | Categorical | 100% |
| Miles at risk (offset) | Part J | Continuous | 100% |

---

## 7. Next Steps

### Immediate (Data Assembly)
1. Upload remaining Part J CSVs (2010–2016, 2019–2020, 2022–2023) — 11 files
2. Upload Part D and Part K for same years (covariables)
3. Assemble complete 15-year panel (~72K observations)

### Phase 1 (MVP Model, 1–2 weeks)
4. Fit Poisson GLM with age + vintage + material covariables
5. Compute cause-specific hazard rates
6. Validate against held-out year (2024)
7. Generate operator-level risk scores

### Phase 2 (Advanced Model, 2–4 weeks)
8. Random Survival Forest with full feature matrix
9. Integrate Mendeley ILI dataset (pipe-level metal loss data)
10. ERA5 environmental features (temperature, precipitation from lat/lon)

### Phase 3 (ROSEN Deliverable)
11. Interactive risk dashboard (React/Plotly)
12. Technical paper draft for submission
13. ROSEN pitch deck with model outputs