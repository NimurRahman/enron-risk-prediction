# dashboard/tabs/tab_activity.py
"""
Activity Trends tab - shows network activity over time
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.config import show_plot

def render(data: dict, filters: dict):
    """Render activity trends tab"""
    
    st.subheader("Network Activity Trends")
    
    # Use filtered edges data
    edges = data.get('edges_filtered')
    features = data.get('features_filtered')
    
    if edges is None or edges.empty:
        st.warning("No edge data available for selected filters")
        return
    
    try:
        edges_copy = edges.copy()
        
        # Find date column
        date_col = None
        for col in ['week_start', 'week', 'date']:
            if col in edges_copy.columns:
                date_col = col
                break
        
        if date_col is None:
            st.error("No date column found in edges data")
            return
        
        # Convert to datetime
        edges_copy[date_col] = pd.to_datetime(edges_copy[date_col], errors='coerce')
        edges_copy = edges_copy.dropna(subset=[date_col])
        
        if edges_copy.empty:
            st.warning("No valid dates in edges data")
            return
        
        # Show date range
        min_date = edges_copy[date_col].min()
        max_date = edges_copy[date_col].max()
        st.caption(f"Displaying data from {min_date.date()} to {max_date.date()}")
        
        # Aggregate by week
        if 'weight' in edges_copy.columns:
            weekly = edges_copy.groupby(date_col).agg({
                'weight': 'sum',
                'source_id': 'count' if 'source_id' in edges_copy.columns else 'size'
            }).reset_index()
            weekly.columns = [date_col, 'total_weight', 'num_edges']
        else:
            weekly = edges_copy.groupby(date_col).size().reset_index(name='num_edges')
            weekly['total_weight'] = 0
        
        # Sort by date
        weekly = weekly.sort_values(date_col)
        
        # Plot 1: Connection Volume Over Time
        st.markdown("**Email Connections Over Time**")
        
        fig1 = px.line(
            weekly,
            x=date_col,
            y='num_edges',
            markers=True,
            title="Number of Email Connections per Week"
        )
        fig1.update_traces(line_color='#3498db', marker=dict(size=8))
        fig1.update_layout(
            xaxis_title="Week",
            yaxis_title="Number of Connections",
            height=350
        )
        show_plot(fig1)
        
        # Plot 2: Total Weight (if available)
        if 'total_weight' in weekly.columns and weekly['total_weight'].sum() > 0:
            st.markdown("**Total Communication Weight Over Time**")
            
            fig2 = px.bar(
                weekly,
                x=date_col,
                y='total_weight',
                title="Total Communication Weight per Week"
            )
            fig2.update_traces(marker_color='#2ecc71')
            fig2.update_layout(
                xaxis_title="Week",
                yaxis_title="Total Weight",
                height=350
            )
            show_plot(fig2)
        
        # Summary Stats
        st.markdown("**Summary Statistics**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Weeks",
                len(weekly)
            )
        
        with col2:
            st.metric(
                "Avg Connections/Week",
                f"{weekly['num_edges'].mean():,.0f}"
            )
        
        with col3:
            if 'total_weight' in weekly.columns:
                st.metric(
                    "Total Weight",
                    f"{weekly['total_weight'].sum():,.0f}"
                )
        
    except Exception as e:
        st.error(f"Error rendering activity trends: {e}")
        import traceback
        st.code(traceback.format_exc())