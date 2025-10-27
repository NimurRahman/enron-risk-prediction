# dashboard/tabs/tab_risk_over_time.py
"""
Risk Over Time tab - shows risk trends and distribution
NOW WITH 4-BAND SYSTEM AND AUTOMATED VISUALIZATIONS
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.config import RISK_COLORS, RISK_ORDER_4BAND, RISK_ORDER_2BAND, show_plot

def render(data: dict, filters: dict):
    """Render risk over time tab"""
    
    st.subheader("Risk Trends Over Time")
    
    # Use preds_filtered (now has both 2-band and 4-band systems)
    preds = data.get('preds_filtered')
    
    if preds is None or preds.empty:
        st.warning("No risk data available for selected filters")
        return
    
    try:
        preds_copy = preds.copy()
        
        # Find date column
        date_col = None
        for col in ['week_start', 'week', 'date']:
            if col in preds_copy.columns:
                date_col = col
                break
        
        if date_col is None:
            st.error("No date column found in predictions data")
            return
        
        # Determine which risk system to use
        # Prefer 4-band if available
        if 'risk_band_4' in preds_copy.columns:
            risk_col = 'risk_band_4'
            risk_order = RISK_ORDER_4BAND
            system_name = "4-Band"
        elif 'y_pred' in preds_copy.columns:
            risk_col = 'y_pred'
            risk_order = RISK_ORDER_2BAND
            system_name = "2-Band"
        else:
            st.error("No risk prediction column found")
            return
        
        # Convert date
        preds_copy[date_col] = pd.to_datetime(preds_copy[date_col], errors='coerce')
        preds_copy = preds_copy.dropna(subset=[date_col])
        
        if preds_copy.empty:
            st.warning("No valid dates in prediction data")
            return
        
        # Show date range and system info
        min_date = preds_copy[date_col].min()
        max_date = preds_copy[date_col].max()
        st.caption(f"Displaying data from {min_date.date()} to {max_date.date()} | Using {system_name} Risk System")
        
        # Plot 1: Risk Distribution Over Time (Stacked Area)
        st.markdown(f"**Risk Band Distribution Over Time ({system_name} System)**")
        
        # Count by week and risk band
        dist = preds_copy.groupby([date_col, risk_col], observed=False).size().reset_index(name='count')
        dist = dist.sort_values(date_col)
        
        # Create stacked area chart
        fig1 = px.area(
            dist,
            x=date_col,
            y='count',
            color=risk_col,
            color_discrete_map=RISK_COLORS,
            category_orders={risk_col: risk_order},
            title=f"Risk Band Distribution by Week ({system_name})"
        )
        fig1.update_layout(
            xaxis_title="Week",
            yaxis_title="Number of Individuals",
            height=400,
            legend_title="Risk Level"
        )
        show_plot(fig1)
        
        # NEW: Risk Score Distribution by Band (Automated)
        st.markdown("---")
        st.markdown(f"**Risk Score Distribution by Risk Band ({system_name} System)**")
        st.caption("Shows how risk scores are distributed within each risk category (updates with filters)")
        
        render_risk_score_by_band(preds_copy, risk_col)
        
        # Summary Table
        st.markdown("---")
        st.markdown("**Risk Summary Statistics**")
        
        summary = preds_copy.groupby(risk_col, observed=False).agg({
            'node_id': 'nunique'
        }).reset_index()
        summary.columns = ['Risk Band', 'Unique Individuals']
        
        # Calculate percentage
        total = summary['Unique Individuals'].sum()
        if total > 0:
            summary['Percentage'] = (summary['Unique Individuals'] / total * 100).round(1)
            summary['Percentage'] = summary['Percentage'].astype(str) + '%'
        else:
            summary['Percentage'] = '0.0%'
        
        # Sort by risk order
        summary['Risk Band'] = pd.Categorical(
            summary['Risk Band'],
            categories=risk_order,
            ordered=True
        )
        summary = summary.sort_values('Risk Band')
        
        st.dataframe(summary, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error rendering risk trends: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_risk_score_by_band(preds_df: pd.DataFrame, risk_col: str):
    """
    NEW: Automated risk score distribution by band
    Creates interactive box plot showing risk score distributions
    Supports both 2-band and 4-band systems
    """
    
    if 'risk_score' not in preds_df.columns or risk_col not in preds_df.columns:
        st.info("Risk score or risk band columns not available")
        return
    
    try:
        # Filter to valid data
        plot_data = preds_df[['risk_score', risk_col]].dropna()
        
        if plot_data.empty:
            st.info("No valid risk score data available")
            return
        
        # Determine bands and thresholds based on system
        if risk_col == 'risk_band_4':
            bands = RISK_ORDER_4BAND
            thresholds = [
                (0.25, "Elevated (0.25)", "orange"),
                (0.50, "High (0.50)", "darkorange"),
                (0.75, "Critical (0.75)", "red")
            ]
        else:  # 2-band system
            bands = RISK_ORDER_2BAND
            thresholds = [
                (0.50, "High (0.50)", "orange"),
                (0.75, "Critical (0.75)", "red")
            ]
        
        # Create box plot
        fig = go.Figure()
        
        # Add box for each risk band
        for band in bands:
            if band in plot_data[risk_col].values:
                band_data = plot_data[plot_data[risk_col] == band]['risk_score']
                
                if not band_data.empty:
                    fig.add_trace(go.Box(
                        y=band_data,
                        name=band,
                        marker_color=RISK_COLORS.get(band, '#95a5a6'),
                        boxmean='sd'  # Show mean and std dev
                    ))
        
        # Add threshold lines
        for threshold, label, color in thresholds:
            fig.add_hline(
                y=threshold, 
                line_dash="dash", 
                line_color=color, 
                annotation_text=label, 
                annotation_position="right"
            )
        
        system_name = "4-Band" if risk_col == 'risk_band_4' else "2-Band"
        fig.update_layout(
            title=f"Risk Score Distribution by Risk Band ({system_name} System)",
            yaxis_title="Risk Score",
            xaxis_title="Risk Band",
            height=400,
            showlegend=False
        )
        
        show_plot(fig)
        
        # Summary statistics
        if risk_col == 'risk_band_4':
            # 4-band system: 4 columns
            cols = st.columns(4)
            for idx, band in enumerate(bands):
                band_data = plot_data[plot_data[risk_col] == band]['risk_score']
                if not band_data.empty:
                    with cols[idx]:
                        st.metric(
                            f"{band}",
                            f"{band_data.mean():.3f}",
                            help=f"Mean score | Range: {band_data.min():.3f} - {band_data.max():.3f}"
                        )
        else:
            # 2-band system: 2 columns
            col1, col2 = st.columns(2)
            
            with col1:
                low_med = plot_data[plot_data[risk_col] == 'Low/Medium']['risk_score']
                if not low_med.empty:
                    st.metric(
                        "Low/Medium - Mean",
                        f"{low_med.mean():.3f}",
                        help=f"Range: {low_med.min():.3f} - {low_med.max():.3f}"
                    )
            
            with col2:
                high_crit = plot_data[plot_data[risk_col] == 'High/Critical']['risk_score']
                if not high_crit.empty:
                    st.metric(
                        "High/Critical - Mean",
                        f"{high_crit.mean():.3f}",
                        help=f"Range: {high_crit.min():.3f} - {high_crit.max():.3f}"
                    )
        
        st.caption("Box plot shows median (line), quartiles (box), and outliers (points)")
        
    except Exception as e:
        st.warning(f"Could not generate risk score distribution: {e}")
        import traceback
        st.code(traceback.format_exc())