"""
PHMSA Pipeline Risk Dashboard
==============================
Interactive analysis of pipeline incident risk using 15 years of PHMSA data.

Features:
- Operator risk rankings with search
- Geographic risk map by state
- Signal ceiling analysis visualization
- Walk-forward validation results
- Cause-specific incident analysis
- Infrastructure age distribution

Usage:
    streamlit run app.py

Author: Kent Valera Chirinos
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="PHMSA Pipeline Risk Model",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1E88E5;
        padding: 1.2rem;
        margin: 1.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #FF9800;
        padding: 1.2rem;
        margin: 1.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 1.2rem;
        margin: 1.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load the survival panel data."""
    # Try multiple paths
    paths = [
        Path("data/processed/survival_panel_15yr_final.csv"),
        Path("survival_panel_15yr_final.csv"),
        Path("../survival_panel_15yr_final.csv"),
    ]
    
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            return df
    
    st.error("❌ Data file not found. Please ensure 'survival_panel_15yr_final.csv' is in data/processed/")
    st.stop()


@st.cache_data
def compute_aggregations(df):
    """Pre-compute aggregations for dashboard."""
    
    # Operator-level aggregation (across all years)
    operator_agg = df.groupby(['operator_id', 'operator_name']).agg({
        'state': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown',
        'miles_at_risk': 'mean',
        'n_incidents': 'sum',
        'n_corrosion': 'sum',
        'n_material': 'sum',
        'event': 'sum',
        'age_at_obs': 'mean',
        'year': 'count'
    }).reset_index()
    operator_agg.columns = ['operator_id', 'operator_name', 'primary_state', 
                            'avg_miles', 'total_incidents', 'total_corrosion',
                            'total_material', 'years_with_events', 'avg_age', 'n_observations']
    
    # Calculate risk score (incidents per 1000 miles per year)
    operator_agg['risk_score'] = (
        operator_agg['total_incidents'] / 
        (operator_agg['avg_miles'] * operator_agg['n_observations'] / 1000 + 0.001)
    )
    operator_agg['risk_score'] = operator_agg['risk_score'].clip(0, 10)
    
    # State-level aggregation
    state_agg = df.groupby('state').agg({
        'miles_at_risk': 'sum',
        'n_incidents': 'sum',
        'n_corrosion': 'sum',
        'operator_id': 'nunique',
        'event': 'sum'
    }).reset_index()
    state_agg.columns = ['state', 'total_miles', 'total_incidents', 
                         'total_corrosion', 'n_operators', 'n_event_obs']
    state_agg['incident_rate'] = state_agg['total_incidents'] / (state_agg['total_miles'] / 1000 + 0.001)
    
    # Year-level aggregation
    year_agg = df.groupby('year').agg({
        'miles_at_risk': 'sum',
        'n_incidents': 'sum',
        'n_corrosion': 'sum',
        'n_material': 'sum',
        'n_excavation': 'sum',
        'n_natural': 'sum',
        'operator_id': 'nunique',
        'event': 'mean'
    }).reset_index()
    
    # Decade/vintage aggregation
    decade_agg = df.groupby('decade_bin').agg({
        'miles_at_risk': 'sum',
        'n_incidents': 'sum',
        'n_corrosion': 'sum',
        'age_at_obs': 'mean'
    }).reset_index()
    decade_agg['corrosion_rate'] = decade_agg['n_corrosion'] / (decade_agg['miles_at_risk'] / 1000 + 0.001)
    
    return operator_agg, state_agg, year_agg, decade_agg


# State abbreviations for map
STATE_ABBREV = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC',
}


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

def render_overview(df, operator_agg, state_agg, year_agg):
    """Main overview dashboard."""
    
    st.markdown('<p class="main-header">🛢️ PHMSA Pipeline Risk Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">15 years of natural gas transmission data • 67,951 observations • 1,264 incidents</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Model AUC", "0.793", help="Area Under ROC Curve on test set (2020-2024)")
    with col2:
        st.metric("Total Miles", f"{df['miles_at_risk'].sum()/1e6:.1f}M", help="Cumulative pipeline-miles across all years")
    with col3:
        st.metric("Operators", f"{df['operator_id'].nunique():,}", help="Unique pipeline operators")
    with col4:
        st.metric("Incidents", f"{df['n_incidents'].sum():,.0f}", help="Total reportable incidents 2010-2024")
    with col5:
        st.metric("Event Rate", f"{df['event'].mean()*100:.2f}%", help="% of operator-state-years with incidents")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Incidents by Year")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=year_agg['year'], 
                y=year_agg['n_incidents'],
                name="Total Incidents",
                marker_color='#E53935',
                opacity=0.8
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=year_agg['year'],
                y=year_agg['n_corrosion'],
                name="Corrosion",
                line=dict(color='#FF9800', width=3),
                mode='lines+markers'
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(title_text="Year", gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(title_text="Incident Count", gridcolor='rgba(0,0,0,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Incidents by State")
        
        # Prepare state data for map
        state_map = state_agg.copy()
        state_map['state_abbrev'] = state_map['state'].map(STATE_ABBREV)
        state_map = state_map.dropna(subset=['state_abbrev'])
        
        fig = px.choropleth(
            state_map,
            locations='state_abbrev',
            locationmode='USA-states',
            color='incident_rate',
            color_continuous_scale='Reds',
            scope='usa',
            labels={'incident_rate': 'Incidents/1K mi'},
            hover_name='state',
            hover_data={'total_incidents': True, 'n_operators': True, 'state_abbrev': False}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cause breakdown
    st.markdown("### 🔍 Incident Causes (2010-2024)")
    
    cause_data = pd.DataFrame({
        'Cause': ['Material/Weld/Equipment', 'Corrosion', 'Excavation Damage', 
                  'Natural Force', 'Other'],
        'Count': [
            df['n_material'].sum(),
            df['n_corrosion'].sum(),
            df['n_excavation'].sum(),
            df['n_natural'].sum(),
            df['n_other_cause'].sum()
        ],
        'ILI Detectable': ['Partially', 'Yes ✓', 'No', 'No', 'No']
    })
    cause_data['Percentage'] = cause_data['Count'] / cause_data['Count'].sum() * 100
    cause_data = cause_data.sort_values('Count', ascending=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        colors = ['#90CAF9', '#4CAF50', '#FFCA28', '#FF7043', '#9E9E9E']
        fig = px.bar(
            cause_data,
            y='Cause',
            x='Count',
            orientation='h',
            color='Cause',
            color_discrete_sequence=colors[::-1],
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=80, t=20, b=20),
            showlegend=False,
            xaxis_title="Number of Incidents",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>💡 ILI Detection Potential</strong><br><br>
        <strong>Corrosion + Material/Weld</strong> account for 
        <strong>{:.0f}%</strong> of all incidents — precisely what 
        ROSEN's In-Line Inspection technology detects.
        </div>
        """.format(
            (df['n_corrosion'].sum() + df['n_material'].sum()) / df['n_incidents'].sum() * 100
        ), unsafe_allow_html=True)


# =============================================================================
# PAGE: SIGNAL CEILING
# =============================================================================

def render_signal_ceiling():
    """Visualize the signal ceiling finding."""
    
    st.markdown("## 📊 Signal Ceiling Analysis")
    st.markdown("*The central finding: Why public data has fundamental limitations*")
    
    st.markdown("---")
    
    # Signal ceiling data
    ceiling_data = pd.DataFrame({
        'Feature Set': [
            'log(miles) only',
            '+ Demographics (age, diameter)',
            '+ Part K (SMYS × Class)',
            '+ Part M (Integrity Mgmt)',
            '+ SSURGO (Soil Corrosivity)',
            '+ USGS (Earthquakes)',
            '+ CP Coverage',
            'ALL FEATURES'
        ],
        'Features': [1, 11, 16, 34, 36, 38, 40, 40],
        'AUC': [0.7903, 0.7926, 0.7938, 0.7899, 0.7901, 0.7901, 0.7898, 0.7931],
        'Source': ['PHMSA', 'PHMSA', 'PHMSA Part K', 'PHMSA Part M', 
                   'USDA API', 'USGS API', 'PHMSA Part D', 'Combined']
    })
    ceiling_data['Delta'] = (ceiling_data['AUC'] - 0.7903) * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        colors = ['#1E88E5' if d >= 0 else '#E53935' for d in ceiling_data['Delta']]
        colors[0] = '#9E9E9E'  # Baseline
        colors[-1] = '#4CAF50'  # Final
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=ceiling_data['Feature Set'],
            y=ceiling_data['AUC'],
            marker_color=colors,
            text=[f"{x:.4f}" for x in ceiling_data['AUC']],
            textposition='outside'
        ))
        
        fig.add_hline(
            y=0.7903, 
            line_dash="dash", 
            line_color="#666",
            annotation_text="Exposure-only baseline (0.7903)",
            annotation_position="top right"
        )
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=40, b=120),
            xaxis_title="",
            yaxis_title="AUC (Area Under ROC Curve)",
            yaxis_range=[0.78, 0.81],
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ The Signal Ceiling</strong><br><br>
        Adding <strong>39 features</strong> from <strong>6 data sources</strong> 
        improves AUC by only <strong>+0.28 percentage points</strong>.<br><br>
        <strong>99.1%</strong> of predictable signal comes from 
        pipeline miles (exposure) alone.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Why?</strong><br><br>
        PHMSA data aggregates to <strong>operator-state level</strong>, 
        destroying the segment-specific variation needed for precision 
        risk prediction.<br><br>
        Breaking this ceiling requires <strong>pipe-level data</strong> — 
        exactly what ILI inspections provide.
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed table
    st.markdown("### Feature Set Comparison")
    
    display_df = ceiling_data[['Feature Set', 'Features', 'AUC', 'Delta', 'Source']].copy()
    display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.4f}")
    display_df['Delta'] = display_df['Delta'].apply(lambda x: f"{x:+.2f}pp" if x != 0 else "baseline")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: RISK RANKINGS
# =============================================================================

def render_risk_rankings(df, operator_agg):
    """Operator risk rankings page."""
    
    st.markdown("## 🎯 Operator Risk Rankings")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_miles = st.slider(
            "Minimum Average Miles",
            min_value=0,
            max_value=1000,
            value=10,
            help="Filter out very small operators"
        )
    
    with col2:
        states = ['All'] + sorted(operator_agg['primary_state'].unique().tolist())
        selected_state = st.selectbox("Filter by State", states)
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ['Risk Score (High→Low)', 'Total Incidents', 'Average Miles', 'Operator Name']
        )
    
    # Filter data
    filtered = operator_agg[operator_agg['avg_miles'] >= min_miles].copy()
    if selected_state != 'All':
        filtered = filtered[filtered['primary_state'] == selected_state]
    
    # Sort
    if sort_by == 'Risk Score (High→Low)':
        filtered = filtered.sort_values('risk_score', ascending=False)
    elif sort_by == 'Total Incidents':
        filtered = filtered.sort_values('total_incidents', ascending=False)
    elif sort_by == 'Average Miles':
        filtered = filtered.sort_values('avg_miles', ascending=False)
    else:
        filtered = filtered.sort_values('operator_name')
    
    # Add rank
    filtered['Rank'] = range(1, len(filtered) + 1)
    
    # Risk tier
    filtered['Risk Tier'] = pd.cut(
        filtered['risk_score'],
        bins=[-np.inf, 0.1, 0.3, 0.5, 1.0, np.inf],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Operators Shown", f"{len(filtered):,}")
    with col2:
        high_risk = ((filtered['Risk Tier'] == 'High') | (filtered['Risk Tier'] == 'Very High')).sum()
        st.metric("High/Very High Risk", f"{high_risk}")
    with col3:
        st.metric("Total Incidents", f"{filtered['total_incidents'].sum():,.0f}")
    with col4:
        st.metric("Avg Risk Score", f"{filtered['risk_score'].mean():.3f}")
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Score Distribution")
        fig = px.histogram(
            filtered, 
            x='risk_score',
            nbins=30,
            color_discrete_sequence=['#1E88E5']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Risk Score (incidents per 1000 miles/year)",
            yaxis_title="Number of Operators",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk vs Size")
        fig = px.scatter(
            filtered.head(200),  # Limit for performance
            x='avg_miles',
            y='total_incidents',
            size='risk_score',
            color='Risk Tier',
            color_discrete_map={
                'Very Low': '#4CAF50',
                'Low': '#8BC34A',
                'Medium': '#FFC107',
                'High': '#FF9800',
                'Very High': '#F44336'
            },
            hover_name='operator_name',
            hover_data=['primary_state', 'risk_score']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Average Miles Operated",
            yaxis_title="Total Incidents (2010-2024)",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rankings table
    st.markdown("#### Top Operators")
    
    display_cols = ['Rank', 'operator_name', 'primary_state', 'avg_miles', 
                    'total_incidents', 'total_corrosion', 'risk_score', 'Risk Tier']
    display_df = filtered.head(50)[display_cols].copy()
    display_df.columns = ['Rank', 'Operator', 'State', 'Avg Miles', 
                          'Incidents', 'Corrosion', 'Risk Score', 'Tier']
    display_df['Avg Miles'] = display_df['Avg Miles'].apply(lambda x: f"{x:,.0f}")
    display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Full Rankings (CSV)",
        csv,
        "operator_risk_rankings.csv",
        "text/csv"
    )


# =============================================================================
# PAGE: VINTAGE ANALYSIS
# =============================================================================

def render_vintage_analysis(df, decade_agg):
    """Analyze risk by pipe vintage."""
    
    st.markdown("## 📅 Vintage Analysis")
    st.markdown("*Non-monotonic corrosion patterns reveal technology transitions*")
    
    st.markdown("---")
    
    # Order decades properly
    decade_order = ['pre1940', '1940_49', '1950_59', '1960_69', '1970_79', 
                    '1980_89', '1990_99', '2000_09', '2010_19', '2020_29']
    decade_labels = ['Pre-1940', '1940s', '1950s', '1960s', '1970s', 
                     '1980s', '1990s', '2000s', '2010s', '2020s']
    
    decade_agg = decade_agg.copy()
    decade_agg['order'] = decade_agg['decade_bin'].map(
        {d: i for i, d in enumerate(decade_order)}
    )
    decade_agg = decade_agg.dropna(subset=['order']).sort_values('order')
    decade_agg['label'] = decade_agg['decade_bin'].map(
        dict(zip(decade_order, decade_labels))
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Miles by Installation Decade")
        fig = px.bar(
            decade_agg,
            x='label',
            y='miles_at_risk',
            color='miles_at_risk',
            color_continuous_scale='Blues',
            text=decade_agg['miles_at_risk'].apply(lambda x: f"{x/1e6:.1f}M")
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Installation Decade",
            yaxis_title="Total Miles (across 15 years)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Corrosion Rate by Vintage")
        
        # Highlight the key finding
        colors = ['#1E88E5'] * len(decade_agg)
        max_idx = decade_agg['corrosion_rate'].idxmax()
        min_idx = decade_agg['corrosion_rate'].idxmin()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=decade_agg['label'],
            y=decade_agg['corrosion_rate'],
            marker_color=['#F44336' if decade_agg.loc[i, 'decade_bin'] == '1970_79' 
                         else '#4CAF50' if decade_agg.loc[i, 'decade_bin'] == '1990_99'
                         else '#1E88E5' for i in decade_agg.index],
            text=decade_agg['corrosion_rate'].apply(lambda x: f"{x:.3f}"),
            textposition='outside'
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=40),
            xaxis_title="Installation Decade",
            yaxis_title="Corrosion Incidents per 1000 Miles/Year",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>🔑 Key Finding: Non-Monotonic Pattern</strong><br><br>
    <table style="width:100%">
    <tr><td><strong>1970s vintage:</strong></td><td style="color:#F44336"><strong>HIGHEST</strong> corrosion rate — Coal tar coatings, early CP systems</td></tr>
    <tr><td><strong>1990s vintage:</strong></td><td style="color:#4CAF50"><strong>LOWEST</strong> corrosion rate — FBE coatings, effective CP</td></tr>
    </table>
    <br>
    This <strong>7.6× difference</strong> between 1970s and 1990s pipe proves that <strong>age alone is insufficient</strong> — 
    a multivariate model capturing vintage-technology interactions is essential.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE: METHODOLOGY
# =============================================================================

def render_methodology():
    """Technical methodology page."""
    
    st.markdown("## 📚 Methodology")
    
    tab1, tab2, tab3 = st.tabs(["Walk-Forward Validation", "Feature Engineering", "Model Architecture"])
    
    with tab1:
        st.markdown("""
        ### Temporal Cross-Validation (AFML Methodology)
        
        Following López de Prado's *Advances in Financial Machine Learning*, 
        we implement **walk-forward validation** to prevent temporal data leakage:
        """)
        
        wf_data = pd.DataFrame({
            'Fold': [1, 2, 3, 4, 5],
            'Training Period': ['2010-2015', '2010-2016', '2010-2017', '2010-2018', '2010-2019'],
            'Test Period': ['2016', '2017', '2018', '2019', '2020-2024'],
            'AUC': [0.781, 0.792, 0.788, 0.801, 0.793]
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"Fold {i}" for i in wf_data['Fold']],
                y=wf_data['AUC'],
                marker_color=['#1E88E5', '#1E88E5', '#1E88E5', '#1E88E5', '#4CAF50'],
                text=wf_data['AUC'].apply(lambda x: f"{x:.3f}"),
                textposition='outside'
            ))
            fig.add_hline(y=0.793, line_dash="dash", line_color="red",
                         annotation_text="Final Test AUC: 0.793")
            fig.update_layout(
                height=350,
                yaxis_range=[0.75, 0.82],
                xaxis_title="",
                yaxis_title="AUC",
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(wf_data, use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class="success-box">
            <strong>✓ Key Properties</strong><br>
            • No future information leakage<br>
            • Expanding training window<br>
            • Consistent AUC (0.78-0.80) across folds<br>
            • Final test on most recent 5 years
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ### Feature Categories
        
        | Category | Features | Source |
        |----------|----------|--------|
        | **Exposure** | log(miles_at_risk) | Part J |
        | **Age** | age_at_obs, era dummies | Part J |
        | **Diameter** | pct_small_diam, pct_large_diam | Part H |
        | **Material** | pct_high_smys, pct_low_smys | Part K |
        | **Location** | pct_class1, pct_high_class | Part K |
        | **Condition** | log1p_total_repairs, lag_corrosion | Part M |
        | **Environmental** | soil_corr_index, earthquake_count | APIs |
        
        ### Feature Importance (SHAP Values)
        """)
        
        shap_data = pd.DataFrame({
            'Feature': ['log_miles', 'age_at_obs', 'pct_small_diam', 'pct_large_diam',
                       'pct_high_smys', 'pct_class1', 'era_coal_tar', 'era_modern'],
            'Importance': [45.2, 12.1, 8.3, 7.1, 5.8, 5.2, 4.1, 3.8]
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            shap_data,
            y='Feature',
            x='Importance',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis_title="Mean |SHAP Value| (%)",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        ### Model: LightGBM Classifier
        
        ```python
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        ```
        
        ### Calibration
        
        Isotonic regression is applied post-training to ensure 
        predicted probabilities are well-calibrated.
        
        ### Performance Metrics
        
        | Metric | Value |
        |--------|-------|
        | Test AUC | 0.793 |
        | Brier Score | 0.018 |
        | Log-Loss | 0.082 |
        | Event Rate | 1.86% |
        """)


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        operator_agg, state_agg, year_agg, decade_agg = compute_aggregations(df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/220px-Camponotus_flavomarginatus_ant.jpg", width=50)
        st.markdown("### 🛢️ PHMSA Risk Model")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Overview", "📈 Signal Ceiling", "🎯 Risk Rankings", 
             "📅 Vintage Analysis", "📚 Methodology"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### About
        
        This dashboard analyzes **15 years** of PHMSA 
        pipeline safety data to predict incident risk.
        
        **Key Finding:**  
        99.1% of signal comes from exposure (miles).
        
        ---
        
        **Author:** Kent Valera Chirinos  
        Telecommunications Engineer  
        
        [GitHub](https://github.com) | [LinkedIn](https://linkedin.com)
        """)
    
    # Route to page
    if page == "📊 Overview":
        render_overview(df, operator_agg, state_agg, year_agg)
    elif page == "📈 Signal Ceiling":
        render_signal_ceiling()
    elif page == "🎯 Risk Rankings":
        render_risk_rankings(df, operator_agg)
    elif page == "📅 Vintage Analysis":
        render_vintage_analysis(df, decade_agg)
    elif page == "📚 Methodology":
        render_methodology()


if __name__ == "__main__":
    main()