"""
PHMSA Pipeline Risk Dashboard
=============================
Interactive Streamlit application for exploring pipeline incident risk predictions.

Author: Kent
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PHMSA Pipeline Risk Model",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #FFA000;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING (with caching)
# ============================================================

@st.cache_data
def load_sample_data():
    """Load or generate sample data for demonstration."""
    # In production, this would load from data/processed/
    # For demo, we generate realistic sample data
    
    np.random.seed(42)
    
    states = ['TX', 'LA', 'OK', 'CA', 'PA', 'OH', 'KS', 'WY', 'NM', 'CO',
              'MI', 'NY', 'IL', 'WV', 'MS', 'AL', 'AR', 'KY', 'MT', 'ND']
    
    years = list(range(2010, 2025))
    
    # Generate operator data
    operators = []
    for i in range(200):
        op_id = 30000 + i
        op_name = f"Pipeline Operator {i+1}"
        state = np.random.choice(states)
        
        # Generate yearly data
        base_miles = np.random.lognormal(5, 1.5)
        base_age = np.random.uniform(20, 60)
        
        for year in years:
            miles = base_miles * (1 + np.random.uniform(-0.1, 0.1))
            age = base_age + (year - 2010)
            
            # Risk score based on miles and age
            log_miles = np.log1p(miles)
            risk_score = 0.1 + 0.15 * (log_miles / 10) + 0.05 * (age / 50) + np.random.normal(0, 0.05)
            risk_score = np.clip(risk_score, 0.01, 0.95)
            
            # Event probability
            event = 1 if np.random.random() < risk_score * 0.15 else 0
            
            operators.append({
                'operator_id': op_id,
                'operator_name': op_name,
                'state': state,
                'year': year,
                'miles': miles,
                'age': age,
                'risk_score': risk_score,
                'event': event,
                'predicted_prob': risk_score * 0.15 + np.random.normal(0, 0.02)
            })
    
    df = pd.DataFrame(operators)
    df['predicted_prob'] = np.clip(df['predicted_prob'], 0.001, 0.999)
    
    return df


@st.cache_data
def load_model_metrics():
    """Load model performance metrics."""
    return {
        'test_auc': 0.793,
        'train_auc': 0.812,
        'walkforward_aucs': [0.781, 0.792, 0.788, 0.801, 0.793],
        'walkforward_years': [2016, 2017, 2018, 2019, '2020-24'],
        'n_events': 1264,
        'n_observations': 67951,
        'event_rate': 0.0186,
        'feature_importance': {
            'log_miles': 0.45,
            'age_at_obs': 0.12,
            'pct_small_diam': 0.08,
            'pct_large_diam': 0.07,
            'pct_high_smys': 0.06,
            'pct_class1': 0.05,
            'era_2000s': 0.04,
            'era_2010s': 0.04,
            'pct_low_smys': 0.03,
            'pct_high_class': 0.03,
            'log1p_cum_corrosion': 0.02,
            'lag_repairs': 0.01,
        }
    }


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Header
    st.markdown('<p class="main-header">🛢️ PHMSA Pipeline Risk Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine learning system for predicting natural gas pipeline incident risk using 15 years of regulatory data</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_sample_data()
    metrics = load_model_metrics()
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/logo.svg", width=50)
        st.markdown("### Navigation")
        page = st.radio(
            "Select View",
            ["📊 Overview", "🎯 Risk Rankings", "📈 Model Performance", "🔍 Operator Lookup", "📚 Methodology"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Filters")
        
        selected_years = st.slider(
            "Year Range",
            min_value=2010,
            max_value=2024,
            value=(2020, 2024)
        )
        
        selected_states = st.multiselect(
            "States",
            options=sorted(df['state'].unique()),
            default=None,
            placeholder="All states"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard demonstrates a machine learning approach 
        to pipeline safety risk assessment.
        
        **Data Sources:**
        - PHMSA Annual Reports
        - PHMSA Incident Database
        - USDA SSURGO (soil)
        - USGS Earthquakes
        
        **Author:** Kent  
        [GitHub](https://github.com) | [LinkedIn](https://linkedin.com)
        """)
    
    # Filter data
    filtered_df = df[
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1])
    ]
    if selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    
    # Page routing
    if page == "📊 Overview":
        render_overview(filtered_df, metrics)
    elif page == "🎯 Risk Rankings":
        render_risk_rankings(filtered_df)
    elif page == "📈 Model Performance":
        render_model_performance(metrics)
    elif page == "🔍 Operator Lookup":
        render_operator_lookup(df)
    elif page == "📚 Methodology":
        render_methodology()


def render_overview(df, metrics):
    """Render the overview dashboard page."""
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Test AUC",
            value=f"{metrics['test_auc']:.3f}",
            delta="+0.003 vs baseline"
        )
    
    with col2:
        st.metric(
            label="Total Observations",
            value=f"{metrics['n_observations']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Incidents Predicted",
            value=f"{metrics['n_events']:,}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Event Rate",
            value=f"{metrics['event_rate']:.2%}",
            delta=None
        )
    
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Incidents Over Time")
        
        yearly = df.groupby('year').agg({
            'event': 'sum',
            'miles': 'sum',
            'operator_id': 'nunique'
        }).reset_index()
        yearly['incident_rate'] = yearly['event'] / yearly['miles'] * 1000
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=yearly['year'], y=yearly['event'], name="Incidents", 
                   marker_color='#E53935'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=yearly['year'], y=yearly['incident_rate'], 
                      name="Rate per 1000 mi", line=dict(color='#1E88E5', width=3)),
            secondary_y=True
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Incident Count", secondary_y=False)
        fig.update_yaxes(title_text="Rate per 1000 miles", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Risk by State")
        
        state_risk = df.groupby('state').agg({
            'risk_score': 'mean',
            'event': 'sum',
            'miles': 'sum'
        }).reset_index()
        state_risk['incident_rate'] = state_risk['event'] / state_risk['miles'] * 1000
        
        fig = px.choropleth(
            state_risk,
            locations='state',
            locationmode='USA-states',
            color='incident_rate',
            color_continuous_scale='Reds',
            scope='usa',
            labels={'incident_rate': 'Incidents/1000mi'}
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            geo=dict(bgcolor='rgba(0,0,0,0)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance (SHAP Values)")
    
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in metrics['feature_importance'].items()
    ]).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    st.markdown("""
    <div class="insight-box">
    <strong>💡 Key Finding: The Signal Ceiling</strong><br>
    Pipeline miles (exposure) accounts for <strong>99.1%</strong> of predictable signal. 
    Condition-level features from PHMSA annual reports add only marginal improvement 
    because they aggregate to operator-state level, losing the segment-specific 
    variation needed to differentiate risk.
    </div>
    """, unsafe_allow_html=True)


def render_risk_rankings(df):
    """Render the risk rankings page."""
    
    st.markdown("### 🎯 Operator Risk Rankings")
    
    # Aggregation options
    col1, col2 = st.columns([1, 3])
    with col1:
        agg_by = st.selectbox(
            "Aggregate by",
            ["Operator", "State", "Operator-State"]
        )
    
    # Calculate rankings
    if agg_by == "Operator":
        rankings = df.groupby(['operator_id', 'operator_name']).agg({
            'risk_score': 'mean',
            'predicted_prob': 'mean',
            'event': 'sum',
            'miles': 'mean',
            'state': 'first'
        }).reset_index()
        rankings = rankings.rename(columns={'miles': 'avg_miles'})
        display_cols = ['operator_name', 'state', 'avg_miles', 'risk_score', 'event']
        
    elif agg_by == "State":
        rankings = df.groupby('state').agg({
            'risk_score': 'mean',
            'predicted_prob': 'mean',
            'event': 'sum',
            'miles': 'sum',
            'operator_id': 'nunique'
        }).reset_index()
        rankings = rankings.rename(columns={
            'operator_id': 'n_operators',
            'miles': 'total_miles'
        })
        display_cols = ['state', 'n_operators', 'total_miles', 'risk_score', 'event']
        
    else:  # Operator-State
        rankings = df.groupby(['operator_id', 'operator_name', 'state']).agg({
            'risk_score': 'mean',
            'predicted_prob': 'mean',
            'event': 'sum',
            'miles': 'mean'
        }).reset_index()
        rankings = rankings.rename(columns={'miles': 'avg_miles'})
        display_cols = ['operator_name', 'state', 'avg_miles', 'risk_score', 'event']
    
    rankings = rankings.sort_values('risk_score', ascending=False)
    rankings['rank'] = range(1, len(rankings) + 1)
    
    # Risk tier assignment
    rankings['risk_tier'] = pd.cut(
        rankings['risk_score'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        high_risk = (rankings['risk_tier'].isin(['High', 'Very High'])).sum()
        st.metric("High/Very High Risk", f"{high_risk} ({high_risk/len(rankings)*100:.1f}%)")
    with col2:
        st.metric("Total Entities", f"{len(rankings):,}")
    with col3:
        st.metric("Total Incidents", f"{rankings['event'].sum():,.0f}")
    
    st.markdown("---")
    
    # Distribution chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Risk Score Distribution")
        fig = px.histogram(
            rankings, x='risk_score', nbins=30,
            color_discrete_sequence=['#1E88E5']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Risk Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk vs Incidents")
        fig = px.scatter(
            rankings, x='risk_score', y='event',
            color='risk_tier',
            color_discrete_map={
                'Very Low': '#4CAF50',
                'Low': '#8BC34A', 
                'Medium': '#FFC107',
                'High': '#FF9800',
                'Very High': '#F44336'
            },
            hover_data=['operator_name'] if 'operator_name' in rankings.columns else None
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rankings table
    st.markdown("#### Top 20 Highest Risk")
    
    display_df = rankings.head(20)[['rank'] + display_cols + ['risk_tier']].copy()
    display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = rankings.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Rankings",
        data=csv,
        file_name="risk_rankings.csv",
        mime="text/csv"
    )


def render_model_performance(metrics):
    """Render the model performance page."""
    
    st.markdown("### 📈 Model Performance")
    
    # Walk-forward validation
    st.markdown("#### Walk-Forward Validation Results")
    st.markdown("""
    The model was validated using temporal cross-validation to prevent data leakage.
    Training always precedes testing chronologically.
    """)
    
    wf_df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(5)],
        'Test Period': metrics['walkforward_years'],
        'AUC': metrics['walkforward_aucs']
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=wf_df['Test Period'].astype(str),
            y=wf_df['AUC'],
            marker_color=['#1E88E5'] * 4 + ['#43A047'],
            text=[f"{x:.3f}" for x in wf_df['AUC']],
            textposition='outside'
        ))
        fig.add_hline(y=0.793, line_dash="dash", line_color="red", 
                     annotation_text="Final Test AUC")
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Test Period",
            yaxis_title="AUC",
            yaxis_range=[0.7, 0.85]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Validation Strategy")
        st.code("""
Fold 1: Train 2010-2015 → Test 2016
Fold 2: Train 2010-2016 → Test 2017
Fold 3: Train 2010-2017 → Test 2018
Fold 4: Train 2010-2018 → Test 2019
Fold 5: Train 2010-2019 → Test 2020-2024
        """)
        
        st.markdown("""
        **Key Properties:**
        - ✅ No future information leakage
        - ✅ Expanding training window
        - ✅ Consistent performance across folds
        - ✅ Final test on most recent 5 years
        """)
    
    st.markdown("---")
    
    # Signal ceiling analysis
    st.markdown("#### Signal Ceiling Analysis")
    
    signal_data = pd.DataFrame({
        'Feature Set': ['log(miles) only', '+ Demographics', '+ Part K (SMYS)', 
                       '+ Part M (Integrity)', '+ External APIs'],
        'Features': [1, 11, 16, 34, 38],
        'AUC': [0.790, 0.793, 0.794, 0.790, 0.793],
        'Delta': [0, 0.3, 0.1, -0.3, 0.0]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=signal_data['Feature Set'],
        y=signal_data['AUC'],
        marker_color=['#90CAF9', '#42A5F5', '#1E88E5', '#EF5350', '#9E9E9E'],
        text=[f"{x:.3f}" for x in signal_data['AUC']],
        textposition='outside'
    ))
    
    fig.add_hline(y=0.790, line_dash="dash", line_color="#666",
                 annotation_text="Exposure-only baseline")
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=80),
        xaxis_title="",
        yaxis_title="AUC",
        yaxis_range=[0.78, 0.82]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Signal Ceiling Identified</strong><br>
    Despite adding 37 features from multiple data sources (PHMSA Parts K, M, SSURGO soil data, 
    USGS earthquakes), the model shows no meaningful improvement over using pipeline miles alone.
    <br><br>
    <strong>Implication:</strong> Breaking this ceiling requires data at finer granularity 
    (pipeline segment level) rather than operator-state aggregates.
    </div>
    """, unsafe_allow_html=True)


def render_operator_lookup(df):
    """Render the operator lookup page."""
    
    st.markdown("### 🔍 Operator Lookup")
    
    # Operator selection
    operators = df.groupby(['operator_id', 'operator_name']).size().reset_index()
    operator_options = {f"{row['operator_name']} ({row['operator_id']})": row['operator_id'] 
                       for _, row in operators.iterrows()}
    
    selected = st.selectbox(
        "Select Operator",
        options=list(operator_options.keys()),
        placeholder="Type to search..."
    )
    
    if selected:
        op_id = operator_options[selected]
        op_df = df[df['operator_id'] == op_id].sort_values('year')
        
        # Operator summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Primary State", op_df['state'].mode().iloc[0])
        with col2:
            st.metric("Avg Miles", f"{op_df['miles'].mean():,.0f}")
        with col3:
            st.metric("Total Incidents", f"{op_df['event'].sum():.0f}")
        with col4:
            st.metric("Current Risk Score", f"{op_df.iloc[-1]['risk_score']:.3f}")
        
        st.markdown("---")
        
        # Historical trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Risk Score History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=op_df['year'],
                y=op_df['risk_score'],
                mode='lines+markers',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Year",
                yaxis_title="Risk Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Miles Operated")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=op_df['year'],
                y=op_df['miles'],
                marker_color='#43A047'
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Year",
                yaxis_title="Miles"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed data table
        st.markdown("#### Historical Data")
        st.dataframe(
            op_df[['year', 'state', 'miles', 'age', 'risk_score', 'event']].round(3),
            use_container_width=True,
            hide_index=True
        )


def render_methodology():
    """Render the methodology page."""
    
    st.markdown("### 📚 Methodology")
    
    st.markdown("""
    ## Problem Formulation
    
    This project frames pipeline safety as a **survival analysis** problem:
    
    - **Observation unit**: Operator × State × Year
    - **Event**: Reportable incident occurrence (binary)
    - **Exposure**: Pipeline miles operated
    - **Covariates**: Infrastructure characteristics, maintenance history, environmental factors
    
    ## Data Sources
    
    ### PHMSA Annual Reports (2010-2024)
    | Part | Content | Features Extracted |
    |------|---------|-------------------|
    | A-D | Operator ID, location, mileage | Total miles, commodity type |
    | H | Diameter distribution | % small (<12"), % large (>24") |
    | J | Installation decade | Age at observation, era dummies |
    | K | SMYS and location class | % low SMYS, % high class |
    | M | Integrity management | Repairs, inspections, damages |
    
    ### External Data
    - **USDA SSURGO**: Soil corrosivity index by state
    - **USGS Earthquakes**: Seismic event counts by state-year
    
    ## Feature Engineering
    
    ```python
    # Core exposure feature (explains 99% of signal)
    log_miles = log1p(total_miles)
    
    # Infrastructure age
    age_at_obs = observation_year - weighted_avg_install_year
    
    # Material risk indicators
    pct_low_smys = miles_smys_below_42ksi / total_miles
    pct_high_class = miles_class3_or_4 / total_miles
    
    # Lagged condition indicators (AFML temporal safety)
    lag_repairs = repairs_in_year_t_minus_1
    cum_corrosion = cumulative_corrosion_findings_through_t_minus_1
    ```
    
    ## Model Architecture
    
    **Algorithm**: LightGBM Classifier
    
    **Hyperparameters** (tuned via walk-forward CV):
    - `n_estimators`: 200
    - `max_depth`: 6
    - `learning_rate`: 0.05
    - `reg_lambda`: 1.0
    - `min_child_samples`: 20
    
    **Calibration**: Isotonic regression for probability estimates
    
    ## Validation Strategy
    
    Temporal walk-forward cross-validation following López de Prado's AFML methodology:
    
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │ Fold 1: ████████████░░░░ Train (2010-15) │ Test (2016)     │
    │ Fold 2: █████████████░░░ Train (2010-16) │ Test (2017)     │
    │ Fold 3: ██████████████░░ Train (2010-17) │ Test (2018)     │
    │ Fold 4: ███████████████░ Train (2010-18) │ Test (2019)     │
    │ Fold 5: ████████████████ Train (2010-19) │ Test (2020-24)  │
    └─────────────────────────────────────────────────────────────┘
    ```
    
    This ensures:
    - No information from the future leaks into training
    - Model is tested on genuinely out-of-sample data
    - Performance is evaluated on recent, relevant time periods
    
    ## Key Finding: The Signal Ceiling
    
    Despite extensive feature engineering from multiple data sources, 
    the model's predictive power is fundamentally limited:
    
    | Data Added | AUC Change |
    |------------|------------|
    | Exposure (miles) | 0.790 (baseline) |
    | Demographics | +0.3 pp |
    | SMYS/Class (Part K) | +0.1 pp |
    | Integrity data (Part M) | -0.3 pp |
    | External APIs | +0.0 pp |
    
    **Root cause**: PHMSA annual reports aggregate condition data at the 
    operator-state level, losing the segment-specific variation needed 
    to differentiate risk beyond exposure.
    
    ## Recommendations for Breaking the Ceiling
    
    1. **NPMS Pipeline Routes**: Geographic routes would enable operator-specific 
       environmental exposure calculations
    
    2. **ILI Data**: Inline inspection results provide segment-level condition 
       information
    
    3. **Real-time SCADA**: Operational anomalies could provide leading indicators
    
    ---
    
    ## References
    
    - López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
    - PHMSA (2024). *Gas Transmission and Gathering Pipeline Annual Report Form*.
    - Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting 
      model predictions*. NeurIPS.
    """)


if __name__ == "__main__":
    main()
