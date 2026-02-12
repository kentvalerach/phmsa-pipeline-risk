# 🛢️ PHMSA Pipeline Risk Model

<p align="center">
  <img src="docs/images/dashboard_preview.png" alt="Dashboard Preview" width="800"/>
</p>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://phmsa-pipeline-risk-tp7enmy8jud43u9l8y9rct.streamlit.app)

**A machine learning system for predicting pipeline incident risk using 15 years of PHMSA regulatory data.**

This project demonstrates advanced techniques in survival analysis, temporal validation, and regulatory data engineering applied to critical infrastructure safety.

---

## 🎯 Project Overview

### The Challenge
The U.S. has over 300,000 miles of natural gas transmission pipelines. Predicting which operator-state combinations are most likely to experience reportable incidents enables proactive safety interventions.

### The Approach
- **Survival Analysis Framework**: Each operator-state-year is a discrete observation period
- **Walk-Forward Validation**: Train on 2010-2019, test on 2020-2024 (no data leakage)
- **AFML Methodology**: Following López de Prado's *Advances in Financial Machine Learning* for temporal cross-validation

### Key Results

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.793 |
| **Walk-Forward Consistency** | 0.78 - 0.81 across 5 folds |
| **Events Predicted** | 1,264 incidents over 67,951 operator-state-years |
| **Signal Ceiling Identified** | 99.1% of signal from exposure (miles) |

---

## 📊 Interactive Dashboard

**[Launch the Live Demo →](https://phmsa-pipeline-risk-tp7enmy8jud43u9l8y9rct.streamlit.app)**

The Streamlit dashboard allows you to:
- 🔍 Explore operator risk rankings by state and year
- 📈 Visualize model performance metrics and calibration
- 🗺️ View geographic risk distribution
- 📋 Generate risk reports for specific operators

---

## 🔬 Technical Highlights

### 1. Temporal Data Engineering
```python
# Triple barrier labeling for survival outcomes
panel['event'] = (incidents > 0).astype(int)
panel['miles_at_risk'] = log1p(total_miles)  # Exposure measure
```

### 2. Walk-Forward Cross-Validation
```
Fold 1: Train 2010-2015 → Test 2016
Fold 2: Train 2010-2016 → Test 2017
Fold 3: Train 2010-2017 → Test 2018
Fold 4: Train 2010-2018 → Test 2019
Fold 5: Train 2010-2019 → Test 2020-2024
```

### 3. Signal Ceiling Analysis
Extensive feature engineering revealed a fundamental limitation:

| Feature Set | Features | AUC | Δ from log(miles) |
|------------|----------|-----|-------------------|
| Exposure only | 1 | 0.790 | baseline |
| + Demographics | 11 | 0.793 | +0.3pp |
| + Part K (SMYS/Class) | 16 | 0.794 | +0.1pp |
| + Part M (Integrity) | 34 | 0.790 | -0.3pp |
| + External (SSURGO/USGS) | 38 | 0.793 | +0.0pp |

**Finding**: Pipeline miles (exposure) explains 99.1% of achievable signal. Condition-level features at operator-state granularity add minimal predictive power.

### 4. Model Architecture
```
LightGBM Classifier
├── Walk-forward temporal CV
├── Isotonic calibration for probability estimates
├── SHAP values for interpretability
└── Automated hyperparameter tuning
```

---

## 📁 Repository Structure

```
phmsa-pipeline-risk/
├── README.md
├── requirements.txt
├── setup.py
│
├── app/                      # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── pages/
│   └── components/
│
├── src/                      # Core modules
│   ├── data/
│   │   ├── extract_phmsa.py      # PHMSA data extraction
│   │   ├── build_panel.py        # Survival panel construction
│   │   └── enrich_features.py    # Feature engineering
│   │
│   ├── models/
│   │   ├── train.py              # Model training pipeline
│   │   ├── evaluate.py           # Walk-forward evaluation
│   │   └── predict.py            # Inference utilities
│   │
│   └── utils/
│       ├── temporal_cv.py        # AFML cross-validation
│       └── calibration.py        # Probability calibration
│
├── data/
│   ├── raw/                  # Original PHMSA files (not tracked)
│   ├── processed/            # Cleaned panels
│   └── external/             # SSURGO, USGS data
│
├── models/                   # Trained model artifacts
│   ├── lgbm_final.pkl
│   └── calibrator.pkl
│
├── notebooks/                # Analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_signal_ceiling_analysis.ipynb
│
├── docs/                     # Documentation
│   ├── methodology.md
│   ├── data_dictionary.md
│   └── images/
│
└── tests/                    # Unit tests
    ├── test_data.py
    └── test_models.py
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/phmsa-pipeline-risk.git
cd phmsa-pipeline-risk
pip install -r requirements.txt
```

### Run the Dashboard Locally
```bash
streamlit run app/streamlit_app.py
```

### Train the Model
```bash
python -m src.models.train --config configs/default.yaml
```

---

## 📚 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [PHMSA Annual Reports](https://www.phmsa.dot.gov/data-and-statistics/pipeline/gas-distribution-gas-gathering-gas-transmission-hazardous-liquids) | Operator infrastructure data (Parts A-R) | Public |
| [PHMSA Incident Database](https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files) | Reportable pipeline incidents | Public |
| [USDA SSURGO](https://sdmdataaccess.nrcs.usda.gov/) | Soil corrosivity ratings | Public API |
| [USGS Earthquakes](https://earthquake.usgs.gov/fdsnws/event/1/) | Seismic event catalog | Public API |

---

## 📖 Key Findings

### 1. Exposure Dominates Risk
Operators with more pipeline miles have proportionally more incidents. This "exposure effect" accounts for 99% of predictable variation.

### 2. Condition Data Limitations
PHMSA annual reports aggregate condition indicators (repairs, inspections, corrosion findings) at the operator-state level. This granularity is too coarse to differentiate risk beyond exposure.

### 3. The Path Forward
Breaking the signal ceiling requires:
- **NPMS pipeline routes** for operator-specific geospatial features
- **ILI (inline inspection) data** for segment-level condition assessment
- **Real-time SCADA data** for operational anomaly detection

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Kent**  
Telecommunications Engineer | Data Scientist  
[LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com)

*Currently seeking opportunities in pipeline integrity analytics and risk modeling.*

---

## 🙏 Acknowledgments

- PHMSA for maintaining transparent public safety data
- López de Prado's *Advances in Financial Machine Learning* for temporal validation methodology
- The open-source Python data science ecosystem
