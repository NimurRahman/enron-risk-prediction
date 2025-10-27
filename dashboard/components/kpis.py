# dashboard/components/kpis.py
"""
KPI metric cards for dashboard overview
Shows statistics for FILTERED data (respects week selection)
"""
import streamlit as st
import pandas as pd
from dashboard.utils import format_number

def render_kpis(data: dict):
    """Render KPI cards at top of dashboard"""
    
    # Use FILTERED predictions as main source (respects week filter)
    preds = data.get('preds_filtered')
    
    # Create 4 columns for KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    # KPI 1: Total Individuals (in filtered view)
    with col1:
        if preds is not None and not preds.empty:
            total_nodes = preds['node_id'].nunique() if 'node_id' in preds.columns else 0
        else:
            total_nodes = 0
        
        st.metric(
            label="Total Individuals",
            value=format_number(total_nodes, decimals=0),
            help="Unique individuals in selected time period"
        )
    
    # KPI 2: High/Critical Risk Count
    with col2:
        if preds is not None and not preds.empty:
            if 'y_pred' in preds.columns:
                high_risk_preds = preds[preds['y_pred'] == 'High/Critical']
                high_risk_count = high_risk_preds['node_id'].nunique()
            elif 'risk_score' in preds.columns:
                high_risk_preds = preds[preds['risk_score'] > 0.5]
                high_risk_count = high_risk_preds['node_id'].nunique()
            else:
                high_risk_count = 0
        else:
            high_risk_count = 0
        
        st.metric(
            label="High/Critical Risk",
            value=format_number(high_risk_count, decimals=0),
            help="Individuals predicted as High/Critical risk"
        )
    
    # KPI 3: Average Risk Score
    with col3:
        if preds is not None and not preds.empty and 'risk_score' in preds.columns:
            avg_risk = preds['risk_score'].mean()
        else:
            avg_risk = None
        
        st.metric(
            label="Avg Risk Score",
            value=format_number(avg_risk, decimals=3),
            help="Mean risk score across all individuals in selected period"
        )
    
    # KPI 4: Weeks of Data (in filtered view)
    with col4:
        if preds is not None and not preds.empty and 'week_start' in preds.columns:
            weeks = preds['week_start'].nunique()
        else:
            weeks = 0
        
        st.metric(
            label="Weeks of Data",
            value=format_number(weeks, decimals=0),
            help="Number of unique weeks in current filtered view"
        )