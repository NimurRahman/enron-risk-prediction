# streamlit_sna_dashboard.py
"""
Enhanced Modular SNA Risk Dashboard
NOW WITH 9 TABS INCLUDING CLUSTERING ANALYSIS
"""
import streamlit as st
from pathlib import Path

# Import dashboard modules
from dashboard.config import setup_page, DEFAULT_DATA_DIR, OUTPUTS_DIR, SHAP_DIR
from dashboard.data_loader import load_all_data, validate_critical_data, display_load_summary
from dashboard.data_processor import process_data
from dashboard.components.sidebar import render_sidebar
from dashboard.components.kpis import render_kpis

# Import all tabs
from dashboard.tabs import (
    tab_activity,
    tab_risk_over_time,
    tab_top_risky,
    tab_feature_importance,
    tab_predictions,
    tab_insights,
    tab_ai_agent,
    tab_simulator,
    tab_clustering  # ← CLUSTERING TAB
)

# Setup page configuration and CSS
setup_page()

# Main header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
            SNA Risk Dashboard
        </h1>
        <p style='color: #666; font-size: 1.1rem; margin-top: 0;'>
            Enhanced Analytics powered by Machine Learning, SHAP & Clustering
        </p>
    </div>
""", unsafe_allow_html=True)

# Render sidebar and get filters
filters = render_sidebar(DEFAULT_DATA_DIR)

# Load data with caching
with st.sidebar:
    with st.expander("Data Load Status", expanded=False):
        with st.spinner("Loading data..."):
            raw_data = load_all_data(
                data_dir=Path(filters['data_dir']),
                outputs_dir=OUTPUTS_DIR,
                shap_dir=SHAP_DIR,
                filenames=filters['filenames']
            )
        display_load_summary(raw_data)

# Validate critical data
if not validate_critical_data(raw_data):
    st.error("Cannot start dashboard - critical data missing!")
    st.stop()

# Process data with filters
processed_data = process_data(raw_data, filters)

# Render KPI cards
st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
render_kpis(processed_data)
st.markdown("</div>", unsafe_allow_html=True)

# Visual separator
st.markdown("""
    <hr style='margin: 2rem 0; border: none; border-top: 2px solid #e0e0e0;'>
""", unsafe_allow_html=True)

# Create 9 tabs
tabs = st.tabs([
    "Activity Trends",
    "Risk Over Time",
    "Top Risky Entities",
    "Feature Importance",
    "Predictions",
    "Risk Simulator",
    "Clustering",  # ← NEW TAB #7
    "Insights & Report",  
    "AI Agent"
])

# Tab 1: Activity Trends
with tabs[0]:
    tab_activity.render(processed_data, filters)

# Tab 2: Risk Over Time
with tabs[1]:
    tab_risk_over_time.render(processed_data, filters)

# Tab 3: Top Risky Entities
with tabs[2]:
    tab_top_risky.render(processed_data, filters)

# Tab 4: Feature Importance
with tabs[3]:
    tab_feature_importance.render(processed_data, filters)

# Tab 5: Predictions
with tabs[4]:
    tab_predictions.render(processed_data, filters)

# Tab 6: Risk Simulator
with tabs[5]:
    tab_simulator.render(processed_data, filters)

# Tab 7: Clustering Analysis ← NEW!
with tabs[6]:
    tab_clustering.render(processed_data, filters)

# Tab 8: Insights & Report
with tabs[7]:
    tab_insights.render(processed_data, filters)

# Tab 9: AI Agent
with tabs[8]:
    tab_ai_agent.render(processed_data, filters)

# Footer
st.markdown("""
    <hr style='margin: 3rem 0 1rem 0; border: none; border-top: 1px solid #e0e0e0;'>
    <div style='text-align: center; color: #888; font-size: 0.9rem; padding-bottom: 1rem;'>
        <p>Enhanced SNA Risk Dashboard v2.2 | 9 Interactive Tabs | ML + SHAP + Clustering Analysis</p>
    </div>
""", unsafe_allow_html=True)