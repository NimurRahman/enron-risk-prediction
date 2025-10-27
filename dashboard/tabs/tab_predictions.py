# dashboard/tabs/tab_predictions.py
"""
ENHANCED Predictions tab with 7 enhancements:
1. Top Contributing Features (SHAP) - personalized
2. Peer Comparison - percentile rankings  
3. Actionable Recommendations
4. Historical Context - rolling stats
5. Confidence Intervals
6. Multi-Model Ensemble
7. Risk Change Indicators

NOW WITH AUTOMATIC RISK OVERVIEW SECTION
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from collections import Counter
from dashboard.config import FEATURE_GLOSSARY, RISK_COLORS, RISK_THRESHOLDS, show_plot
from dashboard.utils import parse_top_features, format_number, calculate_percentile

def render(data: dict, filters: dict):
    """Render enhanced predictions tab"""
    
    st.subheader("Enhanced Predictions with SHAP Explanations")
    
    preds = data.get('preds_filtered')
    nodes = data.get('nodes')
    
    if preds is None or preds.empty:
        st.warning("No prediction data available for selected filters")
        return
    
    try:
        preds_copy = preds.copy()
        
        # Merge with nodes to get email/domain
        if nodes is not None and 'node_id' in preds_copy.columns:
            preds_copy = preds_copy.merge(
                nodes[['node_id', 'email', 'domain']],
                on='node_id',
                how='left'
            )
        
        # Convert week_start to datetime
        if 'week_start' in preds_copy.columns:
            preds_copy['week_start'] = pd.to_datetime(preds_copy['week_start'], errors='coerce')
        
        # RENDER RISK OVERVIEW SECTION FIRST
        render_risk_overview(preds_copy)
        
        st.markdown("---")
        
        # Search/Filter Interface
        st.markdown("### Search Individual Predictions")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_term = st.text_input(
                "Search by email or node ID",
                placeholder="e.g., john.doe@example.com or 12345",
                help="Filter predictions by email address or node ID"
            )
        
        with col2:
            show_all = st.checkbox("Show All Records", value=False)
        
        # Apply search filter with smart matching
        if search_term and not show_all:
            # Check if search term is purely numeric (likely node_id)
            if search_term.isdigit():
                # Exact match for numeric node_id
                mask = preds_copy['node_id'].astype(str) == search_term
            else:
                # Partial match for email addresses
                if 'email' in preds_copy.columns:
                    mask = preds_copy['email'].astype(str).str.contains(search_term, case=False, na=False)
                else:
                    mask = preds_copy['node_id'].astype(str).str.contains(search_term, case=False, na=False)
            
            preds_display = preds_copy[mask]
            
            # If no results, show helpful message
            if preds_display.empty:
                week_lookback = filters.get('week_lookback', 8)
                st.info(f"No results found for '{search_term}' in the selected period. Try increasing the week range or check 'Show All Records'.")
                return
                
        elif show_all:
            preds_display = preds_copy
        else:
            preds_display = preds_copy.head(100)
        
        st.caption(f"Showing {len(preds_display):,} records")
        
        # Main Predictions Table
        st.markdown("### Prediction Results")
        
        # Select columns to display
        display_cols = ['node_id']
        if 'email' in preds_display.columns:
            display_cols.append('email')
        if 'week_start' in preds_display.columns:
            display_cols.append('week_start')
        if 'risk_score' in preds_display.columns:
            display_cols.append('risk_score')
        if 'y_pred' in preds_display.columns:
            display_cols.append('y_pred')
        if 'top_features' in preds_display.columns:
            display_cols.append('top_features')
        if 'trend' in preds_display.columns:
            display_cols.append('trend')
        if 'percentile' in preds_display.columns:
            display_cols.append('percentile')
        if 'recommendation' in preds_display.columns:
            display_cols.append('recommendation')
        
        # Display available columns
        available_cols = [col for col in display_cols if col in preds_display.columns]
        
        st.dataframe(
            preds_display[available_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Individual Deep Dive
        st.markdown("---")
        st.markdown("### Individual Risk Analysis")
        
        # Use ALL data from preds_copy for dropdown, sorted alphabetically
        if 'email' in preds_copy.columns:
            all_unique_emails = sorted(preds_copy['email'].dropna().unique())
            
            if len(all_unique_emails) > 0:
                selected_email = st.selectbox(
                    "Select individual for detailed analysis",
                    options=all_unique_emails,
                    help="Choose an email to see personalized risk breakdown"
                )
                
                # Get individual's data from FULL dataset (preds_copy), not filtered display
                individual_data = preds_copy[preds_copy['email'] == selected_email]
                
                if not individual_data.empty:
                    render_individual_analysis(individual_data, preds_copy)
                else:
                    st.info(f"No data found for {selected_email}")
            else:
                st.info("No email addresses available")
        
        # Download Button
        st.markdown("---")
        buf = io.BytesIO()
        preds_display.to_csv(buf, index=False)
        buf.seek(0)
        
        st.download_button(
            label="Download Predictions (CSV)",
            data=buf.getvalue(),
            file_name="predictions_filtered.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error rendering predictions: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_risk_overview(preds_df: pd.DataFrame):
    """
    NEW: Render automatic risk overview section
    Shows top 20 risky, distributions, model agreement, aggregate SHAP
    """
    
    st.markdown("### Risk Overview")
    st.caption("Automatic analysis of current risk landscape")
    
    
    latest_data = preds_df.copy()
    
    if latest_data.empty:
        st.info("No data available for risk overview")
        return
    
    # Section 1: Top 20 Highest Risk
    st.markdown("#### Top 20 Highest Risk Individuals")
    
    if 'risk_score' in latest_data.columns:
        top_20 = latest_data.nlargest(20, 'risk_score').copy()
        
        # Select display columns
        display_cols = []
        if 'email' in top_20.columns:
            display_cols.append('email')
        display_cols.extend(['risk_score', 'y_pred'])
        if 'trend' in top_20.columns:
            display_cols.append('trend')
        if 'percentile' in top_20.columns:
            display_cols.append('percentile')
        
        # Add rank
        top_20.insert(0, 'Rank', range(1, len(top_20) + 1))
        display_cols = ['Rank'] + display_cols
        
        st.dataframe(
            top_20[display_cols],
            use_container_width=True,
            hide_index=True
        )
    
    # Section 2: Risk Distribution Visualizations
    st.markdown("#### Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of risk scores
        if 'risk_score' in latest_data.columns:
            fig_hist = px.histogram(
                latest_data,
                x='risk_score',
                nbins=30,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Individuals'}
            )
            fig_hist.update_layout(height=300)
            show_plot(fig_hist)
    
    with col2:
        # Pie chart of risk categories
        if 'y_pred' in latest_data.columns:
            risk_counts = latest_data['y_pred'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Category Distribution",
                color=risk_counts.index,
                color_discrete_map={'Low/Medium': '#27ae60', 'High/Critical': '#c0392b'}
            )
            fig_pie.update_layout(height=300)
            show_plot(fig_pie)
    
    # Section 3: Model Agreement Statistics
    if 'model_agreement' in latest_data.columns:
        st.markdown("#### Model Agreement Analysis")
        
        agreement_counts = latest_data['model_agreement'].value_counts()
        total = len(latest_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_pct = (agreement_counts.get('high', 0) / total * 100) if total > 0 else 0
            st.metric(
                "High Agreement",
                f"{high_pct:.1f}%",
                help="All models agree on prediction"
            )
        
        with col2:
            medium_pct = (agreement_counts.get('medium', 0) / total * 100) if total > 0 else 0
            st.metric(
                "Medium Agreement",
                f"{medium_pct:.1f}%",
                help="2 out of 3 models agree"
            )
        
        with col3:
            low_pct = (agreement_counts.get('low', 0) / total * 100) if total > 0 else 0
            st.metric(
                "Low Agreement",
                f"{low_pct:.1f}%",
                help="Models disagree - needs review"
            )
        
        # Agreement distribution bar chart
        fig_agree = px.bar(
            x=list(agreement_counts.index),
            y=list(agreement_counts.values),
            title="Model Agreement Distribution",
            labels={'x': 'Agreement Level', 'y': 'Count'},
            color=list(agreement_counts.index),
            color_discrete_map={'high': '#27ae60', 'medium': '#f39c12', 'low': '#c0392b'}
        )
        fig_agree.update_layout(height=250, showlegend=False)
        show_plot(fig_agree)
    
    # Section 4: Aggregate SHAP Analysis
    if 'top_features' in latest_data.columns and 'y_pred' in latest_data.columns:
        st.markdown("#### Most Common Risk Factors (High/Critical Risk Group)")
        
        # Filter to high risk only
        high_risk = latest_data[latest_data['y_pred'] == 'High/Critical']
        
        if not high_risk.empty:
            # Parse all top_features and count frequency
            all_features = []
            for features_str in high_risk['top_features'].dropna():
                parsed = parse_top_features(features_str)
                for feat_name, feat_val in parsed:
                    all_features.append(feat_name)
            
            if all_features:
                # Count feature frequency
                feature_counts = Counter(all_features)
                top_10_features = feature_counts.most_common(10)
                
                # Create dataframe
                feat_df = pd.DataFrame(top_10_features, columns=['Feature', 'Frequency'])
                feat_df['Description'] = feat_df['Feature'].map(FEATURE_GLOSSARY)
                feat_df['Description'] = feat_df['Description'].fillna('Network or communication feature')
                
                # Horizontal bar chart
                fig_shap = px.bar(
                    feat_df,
                    x='Frequency',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Features Contributing to High Risk",
                    hover_data=['Description']
                )
                fig_shap.update_layout(
                    height=350,
                    yaxis={'categoryorder': 'total ascending'}
                )
                show_plot(fig_shap)
                
                st.caption(f"Analysis based on {len(high_risk)} high-risk individuals")
            else:
                st.info("No feature data available for aggregate analysis")


def render_individual_analysis(individual_data: pd.DataFrame, all_data: pd.DataFrame):
    """
    Render detailed analysis for a selected individual
    Shows all 7 enhancements
    """
    
    # Get latest record
    if 'week_start' in individual_data.columns:
        latest = individual_data.sort_values('week_start', ascending=False).iloc[0]
    else:
        latest = individual_data.iloc[0]
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Overview",
        "SHAP Explanations", 
        "Peer Comparison",
        "Recommendations"
    ])
    
    # TAB 1: Risk Overview
    with tab1:
        render_risk_overview_individual(latest, individual_data)
    
    # TAB 2: SHAP Explanations (Enhancement 1)
    with tab2:
        render_shap_explanations(latest)
    
    # TAB 3: Peer Comparison (Enhancement 2)
    with tab3:
        render_peer_comparison(latest, all_data)
    
    # TAB 4: Recommendations (Enhancement 3)
    with tab4:
        render_recommendations(latest)


def render_risk_overview_individual(latest: pd.Series, individual_data: pd.DataFrame):
    """Enhancement 5 & 7: Risk score with confidence + trend indicators"""
    
    st.markdown("#### Current Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'risk_score' in latest.index:
            st.metric(
                "Risk Score",
                f"{latest['risk_score']:.3f}",
                help="Current risk probability (0-1)"
            )
    
    with col2:
        if 'y_pred' in latest.index:
            risk_band = latest['y_pred']
            color = RISK_COLORS.get(risk_band, "#95a5a6")
            st.markdown(f"**Risk Band**")
            st.markdown(f"<h3 style='color: {color};'>{risk_band}</h3>", unsafe_allow_html=True)
    
    with col3:
        if 'trend' in latest.index:
            trend = latest['trend']
            st.metric(
                "Trend",
                f"{trend}",
                help="Risk trend over recent weeks"
            )
    
    # Confidence Interval (Enhancement 5)
    if 'confidence_low' in latest.index and 'confidence_high' in latest.index:
        st.markdown("#### Model Confidence")
        
        conf_lower = latest.get('confidence_low', 0)
        conf_upper = latest.get('confidence_high', 1)
        risk_score = latest.get('risk_score', 0.5)
        
        st.markdown(f"95% Confidence Interval: **{conf_lower:.3f} - {conf_upper:.3f}**")
        
        # Visualize confidence
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[risk_score],
            y=['Risk Score'],
            orientation='h',
            marker_color='red',
            name='Point Estimate'
        ))
        fig.add_trace(go.Scatter(
            x=[conf_lower, conf_upper],
            y=['Risk Score', 'Risk Score'],
            mode='markers',
            marker=dict(size=12, symbol='line-ns', color='blue'),
            name='Confidence Interval'
        ))
        fig.update_layout(
            height=150,
            showlegend=False,
            xaxis_range=[0, 1],
            margin=dict(l=0, r=0, t=0, b=0)
        )
        show_plot(fig)
    
    # Historical Trend (Enhancement 4)
    if len(individual_data) > 1 and 'week_start' in individual_data.columns:
        st.markdown("#### Historical Risk Trend")
        
        hist = individual_data.copy()
        hist = hist.sort_values('week_start')
        
        if 'risk_score' in hist.columns:
            fig = px.line(
                hist,
                x='week_start',
                y='risk_score',
                markers=True,
                title="Risk Score Over Time"
            )
            fig.add_hline(y=RISK_THRESHOLDS['Critical'], line_dash="dash", line_color="red", annotation_text="Critical")
            fig.add_hline(y=RISK_THRESHOLDS['High'], line_dash="dash", line_color="orange", annotation_text="High")
            fig.update_layout(height=300, yaxis_range=[0, 1])
            show_plot(fig)


def render_shap_explanations(latest: pd.Series):
    """Enhancement 1: Top contributing features (SHAP)"""
    
    st.markdown("#### Top Contributing Features (SHAP)")
    
    if 'top_features' not in latest.index or pd.isna(latest['top_features']):
        st.info("No SHAP feature data available for this individual")
        return
    
    # Parse top features
    features = parse_top_features(latest['top_features'])
    
    if not features:
        st.info("Could not parse feature importance data")
        return
    
    # Create dataframe
    feat_df = pd.DataFrame(features, columns=['Feature', 'SHAP Value'])
    feat_df['Description'] = feat_df['Feature'].map(FEATURE_GLOSSARY)
    feat_df['Description'] = feat_df['Description'].fillna('Unknown')
    feat_df['Impact'] = feat_df['SHAP Value'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
    
    # Horizontal bar chart
    fig = px.bar(
        feat_df,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        color='SHAP Value',
        color_continuous_scale='RdYlGn_r',
        title="Feature Impact on Risk Score",
        hover_data=['Description']
    )
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    show_plot(fig)
    
    # Feature table
    st.dataframe(
        feat_df[['Feature', 'SHAP Value', 'Description', 'Impact']],
        use_container_width=True,
        hide_index=True
    )
    
    st.caption("SHAP values show how much each feature contributes to this individual's risk score")


def render_peer_comparison(latest: pd.Series, all_data: pd.DataFrame):
    """Enhancement 2: Peer comparison with percentile rankings"""
    
    st.markdown("#### Peer Comparison")
    
    if 'risk_score' not in latest.index or all_data.empty:
        st.info("Insufficient data for peer comparison")
        return
    
    risk_score = latest['risk_score']
    
    # Calculate percentile
    if 'risk_score' in all_data.columns:
        percentile = calculate_percentile(risk_score, all_data['risk_score'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Risk Percentile",
                f"{percentile:.1f}%",
                help=f"This individual's risk is higher than {percentile:.1f}% of all individuals"
            )
        
        with col2:
            if percentile >= 90:
                interpretation = "Top 10% - Very High Risk"
            elif percentile >= 75:
                interpretation = "Top 25% - High Risk"
            elif percentile >= 50:
                interpretation = "Above Average"
            else:
                interpretation = "Below Average"
            
            st.markdown(f"**Interpretation:** {interpretation}")
        
        # Distribution plot with marker
        st.markdown("**Position in Overall Distribution**")
        
        fig = go.Figure()
        
        # Histogram of all scores
        fig.add_trace(go.Histogram(
            x=all_data['risk_score'],
            nbinsx=50,
            name='All Individuals',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add vertical line for this individual
        fig.add_vline(
            x=risk_score,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text="This Individual",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Count",
            height=350,
            showlegend=False
        )
        show_plot(fig)


def render_recommendations(latest: pd.Series):
    """Enhancement 3: Actionable recommendations"""
    
    st.markdown("#### Actionable Recommendations")
    
    if 'recommendation' in latest.index and not pd.isna(latest['recommendation']):
        # Parse recommendations (assuming semicolon-separated)
        recs = str(latest['recommendation']).split(';')
        
        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec.strip()}")
    else:
        # Generate generic recommendations based on risk band
        risk_band = latest.get('y_pred', 'Unknown')
        
        if 'High/Critical' in str(risk_band):
            st.error("**High/Critical Risk - Immediate Action Required:**")
            st.markdown("""
            1. Schedule immediate review with compliance team
            2. Investigate recent communication patterns
            3. Review network connections for anomalies
            4. Consider enhanced monitoring
            """)
        elif 'Low/Medium' in str(risk_band):
            st.success("**Low/Medium Risk - Standard Monitoring:**")
            st.markdown("""
            1. Continue standard monitoring
            2. Periodic review (monthly)
            3. Track for any pattern changes
            """)
        else:
            st.info("**Risk Level Unknown:**")
            st.markdown("Monitor and assess as data becomes available")
    
    st.caption("Recommendations are generated based on risk level and network behavior patterns")