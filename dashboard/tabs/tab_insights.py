# dashboard/tabs/tab_insights.py
"""
Insights & Report tab - summary and export
NOW WITH MODEL PERFORMANCE SECTION
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

def render(data: dict, filters: dict):
    """Render insights and report tab"""
    
    st.subheader("Insights & Report Generation")
    
    # Use predictions as main data source (most complete)
    preds = data.get('preds_filtered')
    nodes = data.get('nodes')
    outputs_dir = data.get('_outputs_dir')
    
    # NEW SECTION: Model Performance Analysis
    st.markdown("### Model Performance Analysis")
    st.caption("Comprehensive model validation and comparison")
    
    render_model_performance(outputs_dir)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### Key Insights")
    
    if preds is not None and not preds.empty:
        
        # Insight 1: Total individuals analyzed
        total_individuals = preds['node_id'].nunique() if 'node_id' in preds.columns else 0
        st.markdown(f"- Analyzed **{total_individuals:,}** unique individuals")
        
        # Insight 2: Risk distribution (using 2-band system from predictions)
        if 'y_pred' in preds.columns:
            risk_dist = preds.groupby('y_pred')['node_id'].nunique()
            for band in ['Low/Medium', 'High/Critical']:
                if band in risk_dist.index:
                    count = risk_dist[band]
                    pct = (count / total_individuals * 100) if total_individuals > 0 else 0
                    st.markdown(f"- **{band}** risk: {count:,} individuals ({pct:.1f}%)")
        
        # Insight 3: Time range
        if 'week_start' in preds.columns:
            preds_copy = preds.copy()
            preds_copy['week_start'] = pd.to_datetime(preds_copy['week_start'], errors='coerce')
            min_date = preds_copy['week_start'].min()
            max_date = preds_copy['week_start'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                st.markdown(f"- Data period: **{min_date.date()}** to **{max_date.date()}**")
        
        # Insight 4: Average risk score
        if 'risk_score' in preds.columns:
            avg_risk = preds['risk_score'].mean()
            st.markdown(f"- Average risk score: **{avg_risk:.3f}**")
        
        # Insight 5: Trend analysis
        if 'trend' in preds.columns:
            trend_counts = preds['trend'].value_counts()
            rising_trends = [t for t in trend_counts.index if 'UP' in str(t) or 'RISING' in str(t)]
            rising = sum(trend_counts.get(t, 0) for t in rising_trends)
            if rising > 0:
                st.markdown(f"- **{rising:,}** individuals with rising risk trends")
        
        # Insight 6: High confidence predictions
        if 'model_agreement' in preds.columns:
            high_conf = (preds['model_agreement'] == 'high').sum()
            high_conf_pct = (high_conf / len(preds) * 100) if len(preds) > 0 else 0
            st.markdown(f"- **{high_conf_pct:.1f}%** of predictions have high model agreement")
    
    else:
        st.info("No data available for insights")
    
    st.markdown("---")
    
    # Report Export
    st.markdown("### Export Report")
    
    st.markdown("""
    Generate a comprehensive report including:
    - Executive summary with key metrics
    - Risk distribution analysis
    - Top risky individuals/teams
    - Trend analysis
    - Actionable recommendations
    """)
    
    if st.button("Generate CSV Report", type="primary"):
        if preds is not None and not preds.empty:
            import io
            
            # Create summary report
            summary_data = []
            
            # Add metrics
            summary_data.append({
                'Metric': 'Total Individuals',
                'Value': preds['node_id'].nunique() if 'node_id' in preds.columns else 0
            })
            
            if 'risk_score' in preds.columns:
                summary_data.append({
                    'Metric': 'Average Risk Score',
                    'Value': f"{preds['risk_score'].mean():.3f}"
                })
            
            if 'y_pred' in preds.columns:
                high_risk = (preds['y_pred'] == 'High/Critical').sum()
                summary_data.append({
                    'Metric': 'High/Critical Risk Count',
                    'Value': high_risk
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Convert to CSV
            buf = io.BytesIO()
            summary_df.to_csv(buf, index=False)
            buf.seek(0)
            
            st.download_button(
                "Download Summary Report (CSV)",
                buf.getvalue(),
                file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data available for report generation")
    
    # Data Quality Metrics
    st.markdown("---")
    st.markdown("### Data Quality Metrics")
    
    if preds is not None and not preds.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Records Loaded",
                f"{len(preds):,}"
            )
        
        with col2:
            if 'node_id' in preds.columns:
                missing = preds['node_id'].isna().sum()
                st.metric(
                    "Missing Node IDs",
                    f"{missing:,}"
                )
        
        with col3:
            if 'risk_score' in preds.columns:
                complete = preds['risk_score'].notna().sum()
                pct = (complete / len(preds) * 100) if len(preds) > 0 else 0
                st.metric(
                    "Prediction Coverage",
                    f"{pct:.1f}%"
                )
    
    # Filters Applied
    st.markdown("---")
    st.markdown("### Current Filters Applied")
    
    filter_info = {
        "Week Lookback": filters.get('week_lookback', 8),
        "Risk Bands Filter": ", ".join(filters.get('risk_bands', [])) if filters.get('risk_bands') else "All",
        "Group By": filters.get('group_by', 'individual'),
        "Report Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.json(filter_info)


def render_model_performance(outputs_dir: Path):
    """
    NEW: Render model performance section with all validation charts
    """
    
    if outputs_dir is None or not outputs_dir.exists():
        st.warning("Outputs directory not found - cannot display model performance")
        return
    
    # Section 1: Model Comparison Table
    st.markdown("#### Model Comparison Summary")
    
    comparison_csv = outputs_dir / "model_comparison_table.csv"
    if comparison_csv.exists():
        try:
            comp_df = pd.read_csv(comparison_csv)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.caption("Comparison of Random Forest, XGBoost, and Baseline models across key metrics")
        except Exception as e:
            st.warning(f"Could not load model comparison table: {e}")
    else:
        st.info("Model comparison table not available")
    
    # Section 2: Model Comparison Chart
    comp_chart = outputs_dir / "model_comparison_chart.png"
    if comp_chart.exists():
        st.markdown("**Visual Model Comparison**")
        st.image(str(comp_chart), use_container_width=True, caption="Model Performance Comparison")
    
    st.markdown("---")
    
    # Section 3: ROC Curves
    st.markdown("#### ROC Curves (Receiver Operating Characteristic)")
    st.caption("Measures model's ability to distinguish between classes. Higher curve = better model.")
    
    roc_files = {
        'XGBoost': outputs_dir / "xgboost_roc_curve.png",
        'Random Forest': outputs_dir / "rf_roc_curve.png"
    }
    
    # Display ROC curves side by side
    available_rocs = {name: path for name, path in roc_files.items() if path.exists()}
    
    if available_rocs:
        cols = st.columns(len(available_rocs))
        for idx, (model_name, roc_path) in enumerate(available_rocs.items()):
            with cols[idx]:
                st.image(str(roc_path), use_container_width=True, caption=f"{model_name} ROC Curve")
    else:
        st.info("ROC curves not available")
    
    st.markdown("---")
    
    # Section 4: Precision-Recall Curves
    st.markdown("#### Precision-Recall Curves")
    st.caption("Shows trade-off between precision and recall. Important for imbalanced datasets.")
    
    pr_files = {
        'XGBoost': outputs_dir / "xgboost_pr_curve.png",
        'Random Forest': outputs_dir / "rf_pr_curve.png",
        'Baseline': outputs_dir / "baseline_pr_curve.png"
    }
    
    # Display PR curves side by side
    available_prs = {name: path for name, path in pr_files.items() if path.exists()}
    
    if available_prs:
        cols = st.columns(len(available_prs))
        for idx, (model_name, pr_path) in enumerate(available_prs.items()):
            with cols[idx]:
                st.image(str(pr_path), use_container_width=True, caption=f"{model_name} PR Curve")
    else:
        st.info("Precision-Recall curves not available")
    
    st.markdown("---")
    
    # Section 5: Model Validation
    st.markdown("#### Model Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        overfitting_check = outputs_dir / "model_overfitting_check.png"
        if overfitting_check.exists():
            st.markdown("**Overfitting Check**")
            st.image(str(overfitting_check), use_container_width=True)
            st.caption("Train vs Test performance - lines should be close together")
        else:
            st.info("Overfitting check not available")
    
    with col2:
        # Additional validation chart if available
        risk_dist = outputs_dir / "risk_score_distribution.png"
        if risk_dist.exists():
            st.markdown("**Risk Score Distribution**")
            st.image(str(risk_dist), use_container_width=True)
            st.caption("Distribution of predicted risk scores")
    
    # Interpretation guide
    with st.expander("How to Interpret Model Performance", expanded=False):
        st.markdown("""
        **ROC Curve:**
        - Area Under Curve (AUC) closer to 1.0 = better model
        - Diagonal line = random guessing (AUC = 0.5)
        - Perfect model would hug top-left corner (AUC = 1.0)
        
        **Precision-Recall Curve:**
        - Important for imbalanced datasets (more Low risk than High risk)
        - Higher curve = better balance of precision and recall
        - Area under PR curve indicates overall performance
        
        **Overfitting Check:**
        - Train and test lines should be close
        - Large gap = overfitting (model memorized training data)
        - Similar performance on both = good generalization
        
        **Model Comparison:**
        - Compare metrics: Accuracy, Precision, Recall, F1-Score, AUC
        - Best model = highest scores across multiple metrics
        - Consider business requirements (e.g., prefer high recall to catch all risks)
        """)