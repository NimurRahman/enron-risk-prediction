# dashboard/tabs/tab_feature_importance.py
"""
Feature Importance tab - uses SHAP importance data
NOW WITH ALL SHAP VISUALIZATIONS + FEATURE DISTRIBUTIONS BY RISK
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dashboard.config import FEATURE_GLOSSARY, RISK_COLORS, show_plot

def render(data: dict, filters: dict):
    """Render feature importance tab"""
    
    st.subheader("Model Feature Importance (SHAP)")
    
    feat_imp = data.get('feature_importance')
    shap_pngs = data.get('shap_pngs', {})
    
    if feat_imp is None or feat_imp.empty:
        st.warning("No feature importance data available")
        st.info("Expected file: shap_importance_xgboost.csv in outputs folder")
        return
    
    try:
        feat_imp_copy = feat_imp.copy()
        
        # Handle different column names for importance
        importance_col = None
        if 'importance' in feat_imp_copy.columns:
            importance_col = 'importance'
        elif 'shap_importance' in feat_imp_copy.columns:
            importance_col = 'shap_importance'
            feat_imp_copy['importance'] = feat_imp_copy['shap_importance']
        
        # Ensure required columns exist
        if 'feature' not in feat_imp_copy.columns or importance_col is None:
            st.error("Feature importance data missing required columns")
            st.write("Available columns:", list(feat_imp_copy.columns))
            return
        
        # Sort by importance
        feat_imp_copy = feat_imp_copy.sort_values('importance', ascending=False)
        
        # Add descriptions from glossary
        feat_imp_copy['description'] = feat_imp_copy['feature'].map(FEATURE_GLOSSARY)
        feat_imp_copy['description'] = feat_imp_copy['description'].fillna('Network or communication feature')
        
        # Interactive Bar Chart
        st.markdown("**Top Features by SHAP Importance**")
        
        top_n = st.slider("Number of features to display", 5, 20, 15)
        top_features = feat_imp_copy.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Blues',
            title=f"Top {top_n} Most Important Features for Risk Prediction",
            hover_data=['description']
        )
        fig.update_layout(
            height=max(400, top_n * 30),
            xaxis_title="SHAP Importance Score",
            yaxis_title="Feature",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        show_plot(fig)
        
        # Feature Table
        st.markdown("**All Features with Descriptions**")
        
        # Format importance as rounded values
        feat_imp_display = feat_imp_copy.copy()
        feat_imp_display['importance'] = feat_imp_display['importance'].round(4)
        
        st.dataframe(
            feat_imp_display[['feature', 'importance', 'description']],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # NEW SECTION: Display All SHAP Visualizations
        st.markdown("### SHAP Visualizations")
        st.caption("Comprehensive SHAP analysis from model training")
        
        if shap_pngs:
            # Define expected SHAP visualizations with descriptions
            shap_viz_info = {
                'shap_summary': {
                    'title': 'SHAP Summary Plot (Beeswarm)',
                    'description': 'Shows how each feature impacts predictions across all individuals. Each dot is one person, color shows feature value (red=high, blue=low).'
                },
                'shap_importance_bar': {
                    'title': 'SHAP Feature Importance (Bar Chart)',
                    'description': 'Horizontal bar chart showing average absolute SHAP values for each feature.'
                },
                'shap_waterfall': {
                    'title': 'SHAP Waterfall Example',
                    'description': 'Shows how features contribute to one individual\'s prediction, starting from base value.'
                },
                'shap_dependence': {
                    'title': 'SHAP Dependence Plots',
                    'description': 'Shows how feature values relate to SHAP values (feature impact), revealing non-linear relationships.'
                },
                'xgboost_importance': {
                    'title': 'XGBoost Native Feature Importance',
                    'description': 'Feature importance from XGBoost model itself (gain-based), complementing SHAP analysis.'
                }
            }
            
            # Check which visualizations are available
            available_viz = []
            for key in shap_viz_info.keys():
                if key in shap_pngs:
                    available_viz.append(key)
            
            if available_viz:
                st.info(f"Found {len(available_viz)} SHAP visualizations")
                
                # Display each visualization
                for viz_key in available_viz:
                    viz_info = shap_viz_info[viz_key]
                    viz_path = shap_pngs[viz_key]
                    
                    st.markdown(f"#### {viz_info['title']}")
                    st.caption(viz_info['description'])
                    
                    # Display image
                    st.image(str(viz_path), use_container_width=True)
                    
                    st.markdown("")  # Spacing
            else:
                st.info("No SHAP visualization PNGs found. Check shap_analysis/ folder.")
        else:
            st.info("No SHAP PNGs available. They should be in the shap_analysis/ folder.")
        
        st.markdown("---")
        
        # NEW SECTION: Feature Statistics (if features data available)
        features_df = data.get('features_filtered')
        if features_df is not None and not features_df.empty:
            render_feature_statistics(features_df, feat_imp_copy)
        
        st.markdown("---")
        
        # NEW SECTION: Feature Distributions by Risk Level
        st.markdown("### Feature Distributions by Risk Level")
        st.caption("Compare how feature values differ between risk groups (updates with filters)")
        
        # Need both predictions and features data
        preds_df = data.get('preds_filtered')
        features_df = data.get('features_filtered')
        
        if preds_df is not None and features_df is not None:
            render_feature_distributions_by_risk(preds_df, features_df, feat_imp_copy)
        else:
            st.info("Predictions or features data not available for distribution analysis")
        
        # Explanation
        with st.expander("What is SHAP Importance?", expanded=False):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values measure how much each feature 
            contributes to the model's predictions.
            
            - **Higher values** = feature has more impact on risk predictions
            - **Network features** (degree, betweenness) often show high importance
            - **Communication patterns** (sent/received emails) also contribute significantly
            
            **How to interpret:**
            - **Beeswarm plot**: Each dot is one person. Red dots = high feature value, blue = low. 
              Position shows SHAP value (impact on risk).
            - **Bar chart**: Shows average absolute impact across all predictions.
            - **Waterfall**: Shows step-by-step how features add up to final prediction for one person.
            - **Dependence plots**: Shows relationship between feature value and its impact.
            
            This helps us understand what behaviors and network positions correlate with higher risk.
            """)
        
    except Exception as e:
        st.error(f"Error rendering feature importance: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_feature_statistics(features_df: pd.DataFrame, feat_imp_df: pd.DataFrame):
    """
    Render feature statistics section
    Shows distribution of top features across risk levels
    """
    
    st.markdown("### Feature Statistics")
    st.caption("Distribution of top features in the dataset")
    
    # Get top 5 features
    top_5_features = feat_imp_df.head(5)['feature'].tolist()
    
    # Check which features exist in features_df
    available_features = [f for f in top_5_features if f in features_df.columns]
    
    if not available_features:
        st.info("Feature data not available for detailed statistics")
        return
    
    # Select feature to analyze
    selected_feature = st.selectbox(
        "Select feature to analyze",
        options=available_features,
        help="Choose a feature to see its distribution",
        key="feat_stats_selector"
    )
    
    if selected_feature and selected_feature in features_df.columns:
        # Calculate statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean",
                f"{features_df[selected_feature].mean():.3f}"
            )
        
        with col2:
            st.metric(
                "Median",
                f"{features_df[selected_feature].median():.3f}"
            )
        
        with col3:
            st.metric(
                "Std Dev",
                f"{features_df[selected_feature].std():.3f}"
            )
        
        with col4:
            st.metric(
                "Max",
                f"{features_df[selected_feature].max():.3f}"
            )
        
        # Distribution histogram
        fig_dist = px.histogram(
            features_df,
            x=selected_feature,
            nbins=50,
            title=f"Distribution of {selected_feature}",
            labels={selected_feature: selected_feature, 'count': 'Frequency'}
        )
        fig_dist.update_layout(height=300)
        show_plot(fig_dist)
        
        # Description
        description = FEATURE_GLOSSARY.get(selected_feature, "Network or communication feature")
        st.caption(f"**What this means:** {description}")


def render_feature_distributions_by_risk(preds_df: pd.DataFrame, features_df: pd.DataFrame, feat_imp_df: pd.DataFrame):
    """
    NEW: Automated feature distributions by risk level
    Shows how top features differ between Low/Medium vs High/Critical risk
    """
    
    try:
        # Merge predictions with features
        merged = preds_df.merge(
            features_df,
            on=['node_id', 'week_start'],
            how='inner'
        )
        
        if merged.empty:
            st.info("No matching data between predictions and features")
            return
        
        # Get top 5 features from importance
        top_features = feat_imp_df.head(5)['feature'].tolist()
        
        # Filter to features that exist in merged data
        available_features = [f for f in top_features if f in merged.columns]
        
        if not available_features:
            st.info("Top features not found in features dataset")
            return
        
        # Select feature to visualize
        selected_feature = st.selectbox(
            "Select feature to compare across risk levels",
            options=available_features,
            help="Shows distribution of this feature for Low/Medium vs High/Critical risk",
            key="feat_dist_selector"
        )
        
        if selected_feature and 'y_pred' in merged.columns:
            # Create violin plot
            fig = go.Figure()
            
            for band in ['Low/Medium', 'High/Critical']:
                if band in merged['y_pred'].values:
                    band_data = merged[merged['y_pred'] == band][selected_feature].dropna()
                    
                    if not band_data.empty:
                        fig.add_trace(go.Violin(
                            y=band_data,
                            name=band,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=RISK_COLORS.get(band, '#95a5a6'),
                            opacity=0.6
                        ))
            
            fig.update_layout(
                title=f"Distribution of {selected_feature} by Risk Level",
                yaxis_title=selected_feature,
                xaxis_title="Risk Band",
                height=400,
                showlegend=True
            )
            
            show_plot(fig)
            
            # Statistics comparison
            st.markdown("**Statistical Comparison**")
            
            col1, col2, col3 = st.columns(3)
            
            low_med = merged[merged['y_pred'] == 'Low/Medium'][selected_feature].dropna()
            high_crit = merged[merged['y_pred'] == 'High/Critical'][selected_feature].dropna()
            
            with col1:
                if not low_med.empty:
                    st.metric(
                        "Low/Medium - Mean",
                        f"{low_med.mean():.3f}"
                    )
            
            with col2:
                if not high_crit.empty:
                    st.metric(
                        "High/Critical - Mean",
                        f"{high_crit.mean():.3f}"
                    )
            
            with col3:
                if not low_med.empty and not high_crit.empty and low_med.mean() != 0:
                    diff_pct = ((high_crit.mean() - low_med.mean()) / low_med.mean() * 100)
                    st.metric(
                        "Difference",
                        f"{diff_pct:+.1f}%",
                        help="Percentage difference: High/Critical vs Low/Medium"
                    )
            
            # Interpretation
            description = FEATURE_GLOSSARY.get(selected_feature, "Network or communication feature")
            st.caption(f"**What this means:** {description}")
            
            if not low_med.empty and not high_crit.empty:
                if high_crit.mean() > low_med.mean():
                    st.info(f"High/Critical risk individuals have higher {selected_feature} values on average, suggesting this is a risk factor.")
                else:
                    st.info(f"High/Critical risk individuals have lower {selected_feature} values on average, which may indicate protective behavior or model complexity.")
        
    except Exception as e:
        st.warning(f"Could not generate feature distributions: {e}")
        import traceback
        st.code(traceback.format_exc())