# dashboard/tabs/tab_simulator.py
"""
Risk Simulator Tab - Interactive What-If Analysis
FIXED: Handles zero-value features, proper reset, model bundle support
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

from dashboard.config import (
    FEATURE_GLOSSARY, 
    FEATURE_NAMES_SHORT, 
    RISK_COLORS, 
    RISK_ORDER_4BAND,
    show_plot
)
from dashboard.simulator_engine import load_simulator_engine

# CRITICAL: Define the EXACT 24 features the model was trained with, in order
MODEL_FEATURES_ORDERED = [
    'degree_ma4', 'total_emails_ma4', 'degree', 'unique_contacts', 
    'betweenness', 'total_emails_delta', 'degree_delta', 'betweenness_ma4',
    'total_emails', 'betweenness_delta', 'kcore', 'out_degree',
    'clustering', 'after_hours_pct_delta', 'out_emails', 'after_hours_pct_ma4',
    'after_hours_pct', 'in_degree', 'in_contacts', 'out_contacts',
    'clustering_delta', 'clustering_ma4', 'in_emails', 'closeness'
]

def render(data: dict, filters: dict):
    """Render risk simulator tab"""
    
    st.subheader("Risk Simulator - Interactive What-If Analysis")
    st.caption("Modify features to see how risk predictions change in real-time")
    
    # Check if we have the required data
    preds = data.get('preds_filtered')
    features_df = data.get('features_filtered')
    
    if preds is None or preds.empty:
        st.warning("No prediction data available")
        return
    
    # Load simulator engine
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    if not models_dir.exists():
        st.error(f"Models directory not found: {models_dir}")
        st.info("Please ensure model files (.pkl) are in the 'models' folder at project root")
        return
    
    # Initialize simulator engine
    @st.cache_resource
    def load_engine():
        try:
            return load_simulator_engine(models_dir)
        except Exception as e:
            st.error(f"Failed to load simulator: {e}")
            return None
    
    engine = load_engine()
    
    if engine is None:
        st.error("Could not initialize simulator engine")
        return
    
    # Merge with nodes.csv first to get email column
    nodes_df = data.get('nodes')
    if nodes_df is not None and not nodes_df.empty:
        full_data = preds.merge(
            nodes_df[['node_id', 'email', 'domain']],
            on='node_id',
            how='left'
        )
    else:
        full_data = preds.copy()
    
    # Then merge with features for complete data
    if features_df is not None and not features_df.empty:
        full_data = full_data.merge(
            features_df,
            on=['node_id', 'week_start'],
            how='left',
            suffixes=('', '_feat')
        )
    
    if full_data.empty:
        st.warning("No data available after merging predictions with features")
        return
    
    # Validate that we have ALL required model features
    missing_features = [f for f in MODEL_FEATURES_ORDERED if f not in full_data.columns]
    
    if missing_features:
        st.error("❌ CRITICAL: Missing required model features!")
        st.error(f"Missing features: {', '.join(missing_features)}")
        st.info("The model was trained with 24 specific features. Your data is missing some of them.")
        return
    
    # Use the ordered model features
    available_features = MODEL_FEATURES_ORDERED
    
    # Create tabs for different simulation modes
    sim_tabs = st.tabs([
        "Individual Scenario",
        "Comparison View", 
        "Mass Simulation",
        "Saved Scenarios"
    ])
    
    # Tab 1: Individual Scenario Simulator
    with sim_tabs[0]:
        render_individual_simulator(full_data, engine, available_features)
    
    # Tab 2: Comparison View
    with sim_tabs[1]:
        render_comparison_view(full_data, engine, available_features)
    
    # Tab 3: Mass Simulation
    with sim_tabs[2]:
        render_mass_simulation(full_data, engine, available_features)
    
    # Tab 4: Saved Scenarios
    with sim_tabs[3]:
        render_saved_scenarios()


def render_individual_simulator(full_data: pd.DataFrame, engine, available_features: list):
    """Individual person risk simulation with feature adjustment"""
    
    st.markdown("### Select Individual for Simulation")
    
    # CRITICAL FIX: Initialize reset counter FIRST before any widgets
    if 'simulator_reset_counter' not in st.session_state:
        st.session_state.simulator_reset_counter = 0
    
    # Initialize reset flag BEFORE any widgets
    if 'simulator_reset_requested' not in st.session_state:
        st.session_state.simulator_reset_requested = False
    
    # Get unique individuals with email
    if 'email' in full_data.columns:
        individuals = sorted(full_data[full_data['email'].notna()]['email'].unique())
        
        if not individuals:
            st.warning("No individuals with email addresses found")
            return
        
        selected_email = st.selectbox(
            "Choose person to simulate",
            options=individuals,
            help="Select an individual to modify their features and see risk impact"
        )
        
        # Get their latest data
        person_data = full_data[full_data['email'] == selected_email]
        if 'week_start' in person_data.columns:
            latest_week = person_data['week_start'].max()
            person_row = person_data[person_data['week_start'] == latest_week].iloc[0]
        else:
            person_row = person_data.iloc[0]
        
        # Display current status
        st.markdown("---")
        st.markdown("### Current Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_risk = person_row.get('risk_score', 0.5)
            st.metric("Current Risk Score", f"{current_risk:.3f}")
        
        with col2:
            current_band = person_row.get('y_pred', 'Unknown')
            if current_band == 'High/Critical':
                color = RISK_COLORS['High/Critical']
            else:
                color = RISK_COLORS['Low/Medium']
            st.markdown(f"**Current Band**")
            st.markdown(f"<h3 style='color: {color};'>{current_band}</h3>", unsafe_allow_html=True)
        
        with col3:
            if 'risk_band_4' in person_row.index:
                band_4 = person_row['risk_band_4']
                st.metric("4-Band Classification", band_4)
        
        with col4:
            if 'trend' in person_row.index:
                st.metric("Trend", person_row['trend'])
        
        # Feature adjustment interface
        st.markdown("---")
        st.markdown("### Adjust Features")
        st.caption("Move sliders to modify feature values and see real-time risk predictions")
        
        # Initialize session state for modified features
        if 'modified_features' not in st.session_state:
            st.session_state.modified_features = {}
        
        # Clear slider values if reset was requested
        if st.session_state.simulator_reset_requested:
            print("🔍 RESET FLAG DETECTED!")
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith('slider_')]
            print(f"Keys found: {keys_to_clear}")
            for key in keys_to_clear:
                print(f"Deleting: {key}")
                del st.session_state[key]
            st.session_state.modified_features = {}
            st.session_state.run_prediction = False
            st.session_state.simulator_reset_counter += 1  # INCREMENT COUNTER
            st.session_state.simulator_reset_requested = False
            print(f"✓ Reset counter incremented to: {st.session_state.simulator_reset_counter}")
            st.rerun()  # CRITICAL: Rerun after clearing to redraw sliders
        
        # Create feature sliders in columns
        num_cols = 3
        features_to_adjust = [f for f in available_features if f in person_row.index]
        
        # Sort features by importance
        base_features = [f for f in features_to_adjust if '_ma4' not in f and '_delta' not in f]
        ma4_features = [f for f in features_to_adjust if '_ma4' in f]
        delta_features = [f for f in features_to_adjust if '_delta' in f]
        
        sorted_features = base_features + ma4_features + delta_features
        
        modified_values = {}
        
        # Organize into expandable sections
        with st.expander("Core Network Metrics", expanded=True):
            core_features = [f for f in sorted_features if any(x in f for x in ['degree', 'betweenness', 'closeness', 'clustering', 'kcore'])]
            if core_features:
                cols = st.columns(num_cols)
                for idx, feat in enumerate(core_features):
                    with cols[idx % num_cols]:
                        render_feature_slider(feat, person_row, modified_values)
        
        with st.expander("Communication Metrics"):
            comm_features = [f for f in sorted_features if any(x in f for x in ['email', 'contacts'])]
            if comm_features:
                cols = st.columns(num_cols)
                for idx, feat in enumerate(comm_features):
                    with cols[idx % num_cols]:
                        render_feature_slider(feat, person_row, modified_values)
        
        with st.expander("Work-Life Balance"):
            hours_features = [f for f in sorted_features if 'after_hours' in f]
            if hours_features:
                cols = st.columns(num_cols)
                for idx, feat in enumerate(hours_features):
                    with cols[idx % num_cols]:
                        render_feature_slider(feat, person_row, modified_values)
        
        # Store modified values
        st.session_state.modified_features = modified_values
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Predict Risk", type="primary", use_container_width=True):
                st.session_state.run_prediction = True
        
        with col2:
            if st.button("Reset to Original", use_container_width=True):
                print("🔄 RESET BUTTON CLICKED!")
                print(f"Slider keys: {[k for k in st.session_state.keys() if 'slider_' in k]}")
                print(f"Current counter: {st.session_state.simulator_reset_counter}")
                st.session_state.simulator_reset_requested = True
                st.rerun()
        
        with col3:
            if st.button("Save Scenario", use_container_width=True):
                save_scenario(selected_email, modified_values, person_row)
        
        # Run prediction if button clicked
        if st.session_state.get('run_prediction', False):
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            try:
                # Get original values for all 24 model features
                original_features = person_row[available_features].copy()
                modified_features = original_features.copy()
                
                # Apply modifications
                for feat, val in modified_values.items():
                    if feat in modified_features.index:
                        modified_features[feat] = val
                
                # Create DataFrame with features in EXACT model order
                pred_df = pd.DataFrame([modified_features[MODEL_FEATURES_ORDERED]])
                
                # Verify DataFrame structure
                if len(pred_df.columns) != 24:
                    st.error(f"❌ ERROR: Expected 24 features, got {len(pred_df.columns)}")
                    st.write("Columns:", list(pred_df.columns))
                    return
                
                # Debug info (optional)
                with st.expander("🔍 Debug Info - Feature Values"):
                    st.write("Prediction DataFrame shape:", pred_df.shape)
                    st.write("Columns:", list(pred_df.columns))
                    st.dataframe(pred_df)
                
                # Get predictions from all models
                with st.spinner("Running predictions..."):
                    all_predictions = engine.predict_all_models(pred_df)
                
                # Check if any predictions succeeded
                successful_preds = {k: v for k, v in all_predictions.items() if v is not None}
                
                if not successful_preds:
                    st.error("❌ All model predictions failed!")
                    st.error("This usually means:")
                    st.error("1. Feature names don't match model training")
                    st.error("2. Feature values are out of expected range")
                    st.error("3. Model files are corrupted")
                    return
                
                # Display results
                display_prediction_results(
                    original_features, 
                    modified_features, 
                    successful_preds,
                    engine,
                    person_row.get('risk_score', 0.5)
                )
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_feature_slider(feat: str, person_row: pd.Series, modified_values: dict):
    """Render slider for a single feature with tooltip"""
    
    original_val = person_row[feat]
    
    # Get feature description
    description = FEATURE_GLOSSARY.get(feat, "No description available")
    short_name = FEATURE_NAMES_SHORT.get(feat, feat)
    
    # Determine slider range based on feature type
    if 'pct' in feat or 'after_hours' in feat:
        min_val, max_val = 0.0, 100.0
        step = 1.0
    elif 'clustering' in feat or 'closeness' in feat or 'betweenness' in feat:
        min_val, max_val = 0.0, 1.0
        step = 0.01
    elif 'kcore' in feat:
        min_val, max_val = 0, 15
        step = 1
    elif 'delta' in feat:
        # CRITICAL FIX: Handle zero values for delta features
        if abs(original_val) < 0.01:  # Essentially zero
            min_val = -10.0
            max_val = 10.0
            step = 0.5
        else:
            min_val = -abs(original_val) * 2
            max_val = abs(original_val) * 2
            step = 0.1 if abs(original_val) < 10 else 1.0
    else:
        min_val = 0
        # CRITICAL FIX: Ensure max > min for regular features
        if original_val == 0:
            max_val = 100
        else:
            max_val = max(original_val * 3, 100)
        step = 1.0 if original_val > 10 else 0.1
    
    # Final safety check: ensure min < max
    if min_val >= max_val:
        min_val = 0.0
        max_val = 100.0
        step = 1.0
    
    # Ensure value is within range
    slider_value = float(original_val)
    if slider_value < min_val:
        slider_value = min_val
    if slider_value > max_val:
        slider_value = max_val
    
    # CRITICAL FIX: Add reset counter to key so sliders get new keys after reset
    new_val = st.slider(
        short_name,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(slider_value),
        step=float(step),
        help=description,
        key=f"slider_{feat}_{st.session_state.simulator_reset_counter}"  # COUNTER ADDED HERE
    )
    
    # Show change indicator
    if abs(new_val - original_val) > step * 0.1:
        change_pct = ((new_val - original_val) / original_val * 100) if original_val != 0 else 0
        if change_pct > 0:
            st.caption(f"⬆️ Up +{change_pct:.1f}% from original")
        else:
            st.caption(f"⬇️ Down {change_pct:.1f}% from original")
    
    modified_values[feat] = new_val


def display_prediction_results(original_features, modified_features, all_predictions, engine, original_risk):
    """Display comprehensive prediction results"""
    
    # Multi-model comparison
    st.markdown("#### Multi-Model Predictions")
    
    if not all_predictions:
        st.error("No predictions available from any model")
        return
    
    cols = st.columns(len(all_predictions))
    
    for idx, (model_name, pred_result) in enumerate(all_predictions.items()):
        if pred_result is None:
            with cols[idx]:
                st.markdown(f"**{model_name.upper()}**")
                st.error("Prediction failed")
            continue
        
        with cols[idx]:
            try:
                risk_score = pred_result['risk_score'][0]
                risk_band = engine.map_risk_score_to_4bands(risk_score)
                
                st.markdown(f"**{model_name.upper()}**")
                st.metric(
                    "Risk Score",
                    f"{risk_score:.3f}",
                    delta=f"{risk_score - original_risk:+.3f}",
                    delta_color="inverse"
                )
                
                color = RISK_COLORS.get(risk_band, '#95a5a6')
                st.markdown(f"<p style='color: {color}; font-weight: bold;'>{risk_band}</p>", 
                           unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying {model_name}: {e}")
    
    # Risk gauge visualization
    st.markdown("---")
    st.markdown("#### Risk Score Gauge")
    
    # Use primary model (xgboost) for main visualization
    primary_model = 'xgboost' if 'xgboost' in all_predictions else list(all_predictions.keys())[0]
    primary_pred = all_predictions[primary_model]
    
    if primary_pred and primary_pred['risk_score'] is not None:
        new_risk = primary_pred['risk_score'][0]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_risk,
            delta={'reference': original_risk, 'valueformat': '.3f'},
            title={'text': f"New Risk Score ({primary_model.upper()})"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkred" if new_risk > 0.75 else "orange" if new_risk > 0.5 else "yellow" if new_risk > 0.25 else "green"},
                'steps': [
                    {'range': [0, 0.25], 'color': "lightgreen"},
                    {'range': [0.25, 0.5], 'color': "lightyellow"},
                    {'range': [0.5, 0.75], 'color': "lightcoral"},
                    {'range': [0.75, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': original_risk
                }
            }
        ))
        
        fig.update_layout(height=300)
        show_plot(fig)
    
    # SHAP explanation
    st.markdown("---")
    st.markdown("#### Feature Impact Analysis (SHAP)")
    
    try:
        with st.spinner("Calculating SHAP values..."):
            pred_df = pd.DataFrame([modified_features[MODEL_FEATURES_ORDERED]])
            shap_values = engine.calculate_shap(pred_df, primary_model)
        
        if shap_values is not None:
            feature_names = MODEL_FEATURES_ORDERED
            
            if len(shap_values.shape) == 2:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            abs_shap = np.abs(shap_vals)
            top_indices = np.argsort(abs_shap)[-10:][::-1]
            
            top_features = [feature_names[i] for i in top_indices]
            top_shap = [shap_vals[i] for i in top_indices]
            
            fig = go.Figure(go.Bar(
                y=[FEATURE_NAMES_SHORT.get(f, f) for f in top_features],
                x=top_shap,
                orientation='h',
                marker=dict(
                    color=top_shap,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="SHAP Value")
                )
            ))
            
            fig.update_layout(
                title="Top 10 Features Driving Risk Prediction",
                xaxis_title="SHAP Value (Impact on Risk)",
                yaxis_title="Feature",
                height=400
            )
            
            show_plot(fig)
            
            st.caption("📊 Positive SHAP values (red) increase risk | Negative SHAP values (green) decrease risk")
            
    except Exception as e:
        st.warning(f"Could not calculate SHAP: {e}")
    
    # Recommendations
    st.markdown("---")
    st.markdown("#### Recommendations")
    
    try:
        recommendations = engine.generate_recommendations(
            original_features,
            modified_features,
            shap_values if 'shap_values' in locals() else None,
            MODEL_FEATURES_ORDERED
        )
        
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    except Exception as e:
        st.info("💡 Monitor for sustained changes over time")


def render_comparison_view(full_data, engine, available_features):
    """Side-by-side comparison of two scenarios"""
    
    st.markdown("### Compare Two Scenarios")
    st.caption("Select two individuals or scenarios to compare their risk profiles")
    
    col1, col2 = st.columns(2)
    
    if 'email' not in full_data.columns:
        st.warning("Email column not available")
        return
    
    individuals = sorted(full_data[full_data['email'].notna()]['email'].unique())
    
    with col1:
        st.markdown("**Scenario A**")
        email_a = st.selectbox("Select Person A", options=individuals, key="compare_a")
    
    with col2:
        st.markdown("**Scenario B**")
        email_b = st.selectbox("Select Person B", options=individuals, key="compare_b")
    
    if st.button("Compare Scenarios"):
        person_a = full_data[full_data['email'] == email_a].iloc[0]
        person_b = full_data[full_data['email'] == email_b].iloc[0]
        
        st.markdown("---")
        st.markdown("### Feature Comparison")
        
        comparison_data = []
        for feat in available_features[:15]:
            if feat in person_a.index and feat in person_b.index:
                comparison_data.append({
                    'Feature': FEATURE_NAMES_SHORT.get(feat, feat),
                    'Person A': f"{person_a[feat]:.2f}",
                    'Person B': f"{person_b[feat]:.2f}",
                    'Difference': f"{person_b[feat] - person_a[feat]:+.2f}"
                })
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


def render_mass_simulation(full_data, engine, available_features):
    """Apply changes to multiple people"""
    
    st.markdown("### Mass Simulation")
    st.caption("Apply percentage changes to groups of people and see aggregate impact")
    
    risk_group = st.selectbox(
        "Select group to simulate",
        options=['All', 'High/Critical Only', 'Low/Medium Only'],
        help="Choose which risk group to apply changes to"
    )
    
    feature_to_change = st.selectbox(
        "Select feature to modify across group",
        options=available_features,
        format_func=lambda x: FEATURE_NAMES_SHORT.get(x, x)
    )
    
    pct_change = st.slider(
        "Percentage change to apply",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Positive = increase, Negative = decrease"
    )
    
    if st.button("Run Mass Simulation"):
        st.markdown("---")
        st.markdown("### Mass Simulation Results")
        
        if risk_group == 'High/Critical Only':
            group_data = full_data[full_data['y_pred'] == 'High/Critical']
        elif risk_group == 'Low/Medium Only':
            group_data = full_data[full_data['y_pred'] == 'Low/Medium']
        else:
            group_data = full_data
        
        st.info(f"Simulating {len(group_data)} individuals with {pct_change:+d}% change in {FEATURE_NAMES_SHORT.get(feature_to_change, feature_to_change)}")
        st.caption("This feature is under development. Preview of mass simulation framework created.")


def render_saved_scenarios():
    """Display saved scenarios"""
    
    st.markdown("### Saved Scenarios")
    
    if 'saved_scenarios' not in st.session_state or not st.session_state.saved_scenarios:
        st.info("No saved scenarios yet. Create and save scenarios from the Individual Scenario tab.")
    else:
        for idx, scenario in enumerate(st.session_state.saved_scenarios):
            with st.expander(f"Scenario {idx+1}: {scenario['email']} - {scenario['timestamp']}"):
                st.json(scenario['modifications'])


def save_scenario(email: str, modifications: dict, original_data: pd.Series):
    """Save a scenario for later comparison"""
    
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []
    
    scenario = {
        'email': email,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'modifications': modifications,
        'original_risk': original_data.get('risk_score', 0)
    }
    
    st.session_state.saved_scenarios.append(scenario)
    st.success(f"✅ Scenario saved! Total saved: {len(st.session_state.saved_scenarios)}")