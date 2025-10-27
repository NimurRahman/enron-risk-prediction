# dashboard/components/plots.py
"""
Reusable plotting functions and utilities
"""
import streamlit as st
import plotly.graph_objects as go

def show_plot(fig, use_container_width: bool = True):
    """
    Display plotly figure with consistent settings
    
    Args:
        fig: Plotly figure object
        use_container_width: Whether to use full container width
    """
    # Apply consistent theme
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Display
    st.plotly_chart(fig, use_container_width=use_container_width)

def create_empty_plot(message: str = "No data available"):
    """Create empty plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=400
    )
    return fig

def add_risk_threshold_lines(fig, thresholds: dict = None):
    """
    Add horizontal lines for risk thresholds
    
    Args:
        fig: Plotly figure
        thresholds: Dict of {band_name: threshold_value}
    """
    if thresholds is None:
        thresholds = {
            "Critical": 0.75,
            "High": 0.5,
            "Elevated": 0.25
        }
    
    colors = {
        "Critical": "red",
        "High": "orange", 
        "Elevated": "yellow"
    }
    
    for band, threshold in thresholds.items():
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=colors.get(band, "gray"),
            annotation_text=band,
            annotation_position="right"
        )
    
    return fig