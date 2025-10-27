# dashboard/tabs/tab_top_risky.py
"""
Top Risky Entities tab - ranked list and visualization
CRITICAL: Must update when filters change!
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from dashboard.config import RISK_ORDER_2BAND, RISK_COLORS, show_plot

def render(data: dict, filters: dict):
    """Render top risky entities tab - REACTIVE to all filters"""
    
    st.subheader("Top Risky Individuals/Teams")
    
    # CRITICAL: Use predictions data (preds_filtered) - already filtered by data_processor
    preds = data.get('preds_filtered')
    nodes = data.get('nodes')
    
    if preds is None or preds.empty:
        st.warning("No prediction data available for selected filters")
        st.info("Try adjusting the week range or search filters in the sidebar")
        return
    
    try:
        # Make a copy to avoid modifying original
        risk_data = preds.copy()
        
        # Show current filter status
        week_lookback = filters.get('week_lookback', 8)
        st.caption(f"Displaying data from last {week_lookback} weeks (adjust in sidebar)")
        
        # Convert week_start to datetime for date display
        if 'week_start' in risk_data.columns:
            risk_data['week_start'] = pd.to_datetime(risk_data['week_start'], errors='coerce')
            
            # Show date range
            if not risk_data.empty:
                min_date = risk_data['week_start'].min()
                max_date = risk_data['week_start'].max()
                num_weeks = risk_data['week_start'].nunique()
                st.info(f"Data range: {min_date.date()} to {max_date.date()} ({num_weeks} weeks)")
        
        # Merge with nodes to get email/domain
        if nodes is not None and 'node_id' in risk_data.columns:
            risk_data = risk_data.merge(
                nodes[['node_id', 'email', 'domain']],
                on='node_id',
                how='left'
            )
        
        # Check if risk_score column exists
        if 'risk_score' not in risk_data.columns:
            st.error("risk_score column not found in predictions data")
            st.write("Available columns:", list(risk_data.columns))
            return
        
        # Determine grouping from filters
        group_by = filters.get('group_by', 'individual')
        
        # Set group column
        if group_by == 'team' and 'domain' in risk_data.columns:
            group_col = 'domain'
            group_label = "Team (Domain)"
        else:
            group_col = 'email' if 'email' in risk_data.columns else 'node_id'
            group_label = "Individual"
        
        st.markdown(f"**Grouped by: {group_label}**")
        
        # Calculate average risk per group across ALL weeks in filtered period
        if group_by == 'team' and 'domain' in risk_data.columns:
            # Team grouping - aggregate by domain
            ranked = risk_data.groupby(group_col).agg({
                'risk_score': 'mean',
                'node_id': 'nunique'  # Count unique individuals per team
            }).reset_index()
            ranked.columns = [group_col, 'avg_risk_score', 'num_individuals']
            ranked = ranked.sort_values('avg_risk_score', ascending=False)
        else:
            # Individual grouping - aggregate by email/node_id
            ranked = risk_data.groupby(group_col).agg({
                'risk_score': 'mean'
            }).reset_index()
            ranked.columns = [group_col, 'avg_risk_score']
            ranked = ranked.sort_values('avg_risk_score', ascending=False)
        
        # Add risk band counts if y_pred column exists
        if 'y_pred' in risk_data.columns:
            band_counts = risk_data.groupby(group_col)['y_pred'].value_counts().unstack(fill_value=0)
            
            # Only include bands that exist in data
            available_bands = [band for band in RISK_ORDER_2BAND if band in band_counts.columns]
            if available_bands:
                band_counts = band_counts[available_bands].reset_index()
                ranked = ranked.merge(band_counts, on=group_col, how='left')
        
        # Display ranked table
        st.markdown("### Top Entities by Average Risk Score")
        
        # Slider for number of entities to show
        max_show = min(50, len(ranked))
        top_n = st.slider("Number of entities to display", 5, max_show, min(20, max_show))
        
        # Show top N
        st.dataframe(
            ranked.head(top_n),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization
        st.markdown("### Risk Score Visualization")
        
        top_for_chart = ranked.head(min(15, top_n))
        
        fig = px.bar(
            top_for_chart,
            x='avg_risk_score',
            y=group_col,
            orientation='h',
            color='avg_risk_score',
            color_continuous_scale='Reds',
            title=f"Top {len(top_for_chart)} Highest Risk {group_label}s"
        )
        fig.update_layout(
            height=max(400, len(top_for_chart) * 30),
            xaxis_title="Average Risk Score",
            yaxis_title=group_label,
            yaxis={'categoryorder': 'total ascending'}
        )
        show_plot(fig)
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Entities",
                f"{len(ranked):,}"
            )
        
        with col2:
            st.metric(
                "Avg Risk Score",
                f"{ranked['avg_risk_score'].mean():.3f}"
            )
        
        with col3:
            high_risk = ranked[ranked['avg_risk_score'] > 0.5]
            st.metric(
                "High Risk (>0.5)",
                f"{len(high_risk):,}"
            )
        
        with col4:
            critical_risk = ranked[ranked['avg_risk_score'] > 0.75]
            st.metric(
                "Critical (>0.75)",
                f"{len(critical_risk):,}"
            )
        
        # Distribution plot
        st.markdown("### Risk Score Distribution")
        
        fig_dist = px.histogram(
            ranked,
            x='avg_risk_score',
            nbins=30,
            title="Distribution of Average Risk Scores",
            labels={'avg_risk_score': 'Average Risk Score', 'count': 'Number of Entities'}
        )
        fig_dist.update_layout(height=300)
        show_plot(fig_dist)
        
        # Download button
        st.markdown("---")
        buf = io.BytesIO()
        ranked.to_csv(buf, index=False)
        buf.seek(0)
        
        st.download_button(
            label=f"Download Full Ranked List (CSV)",
            data=buf.getvalue(),
            file_name=f"top_risky_{group_by}s.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error rendering top risky entities: {e}")
        import traceback
        st.code(traceback.format_exc())