# dashboard/data_loader.py
"""
Load all data files (CSV, Parquet) with proper error handling and validation
"""
import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional

# === SAFE FILE READERS ===

def safe_read_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Safely read CSV with error handling"""
    try:
        if not filepath.exists():
            return None
        return pd.read_csv(filepath)
    except Exception as e:
        st.warning(f"Could not read {filepath.name}: {e}")
        return None

def safe_read_parquet(filepath: Path) -> Optional[pd.DataFrame]:
    """Safely read Parquet with error handling"""
    try:
        if not filepath.exists():
            return None
        return pd.read_parquet(filepath)
    except Exception as e:
        st.warning(f"Could not read {filepath.name}: {e}")
        return None

# === MAIN DATA LOADER ===

@st.cache_data(ttl=600, show_spinner=False)
def load_all_data(data_dir: Path, outputs_dir: Path, shap_dir: Path, filenames: dict) -> dict:
    """
    Load all required data files from correct locations
    
    Args:
        data_dir: Path to data/ folder (contains Sumit's enhanced files)
        outputs_dir: Path to outputs/ folder (contains your predictions + SHAP)
        shap_dir: Path to shap_analysis/ folder (contains SHAP PNGs)
        filenames: Dict of filename overrides from config
    
    Returns:
        dict: Dictionary containing all loaded dataframes + metadata
    """
    data = {}
    load_summary = []
    
    # === LOAD FROM DATA FOLDER ===
    
    # Core files
    nodes_path = data_dir / filenames['nodes']
    data['nodes'] = safe_read_csv(nodes_path)
    load_summary.append(('nodes', nodes_path, data['nodes'] is not None, len(data['nodes']) if data['nodes'] is not None else 0))
    
    idmap_path = data_dir / filenames['idmap']
    data['idmap'] = safe_read_csv(idmap_path)
    load_summary.append(('idmap', idmap_path, data['idmap'] is not None, len(data['idmap']) if data['idmap'] is not None else 0))
    
    community_path = data_dir / filenames['community_map']
    data['community_map'] = safe_read_csv(community_path)
    load_summary.append(('community_map', community_path, data['community_map'] is not None, len(data['community_map']) if data['community_map'] is not None else 0))
    
    # Enhanced files from Sumit (CRITICAL - must use enhanced versions!)
    edges_path = data_dir / filenames['edges']
    data['edges'] = safe_read_parquet(edges_path)
    load_summary.append(('edges', edges_path, data['edges'] is not None, len(data['edges']) if data['edges'] is not None else 0))
    
    features_path = data_dir / filenames['features']
    data['features'] = safe_read_parquet(features_path)
    load_summary.append(('features', features_path, data['features'] is not None, len(data['features']) if data['features'] is not None else 0))
    
    risk_path = data_dir / filenames['risk']
    data['risk'] = safe_read_parquet(risk_path)
    load_summary.append(('risk', risk_path, data['risk'] is not None, len(data['risk']) if data['risk'] is not None else 0))
    
    # === LOAD FROM OUTPUTS FOLDER ===
    
    # Your enhanced predictions (93.7 MB, 26 columns)
    preds_path = outputs_dir / filenames['preds']
    data['predictions'] = safe_read_csv(preds_path)
    load_summary.append(('predictions', preds_path, data['predictions'] is not None, len(data['predictions']) if data['predictions'] is not None else 0))
    
    # SHAP importance
    shap_imp_path = outputs_dir / filenames['feat_imp']
    data['feature_importance'] = safe_read_csv(shap_imp_path)
    load_summary.append(('feature_importance', shap_imp_path, data['feature_importance'] is not None, len(data['feature_importance']) if data['feature_importance'] is not None else 0))
    
    # === LOAD STATIC VISUALIZATIONS (for display only) ===
    
    # Check if SHAP PNGs exist
    shap_pngs = {
        'shap_summary': shap_dir / 'shap_summary_xgboost.png',
        'shap_waterfall': shap_dir / 'shap_waterfall_example.png',
    }
    
    outputs_pngs = {
        'model_comparison': outputs_dir / 'model_comparison_chart.png',
        'model_overfitting': outputs_dir / 'model_overfitting_check.png',
        'xgb_roc': outputs_dir / 'xgboost_roc_curve.png',
    }
    
    data['shap_pngs'] = {k: v for k, v in shap_pngs.items() if v.exists()}
    data['outputs_pngs'] = {k: v for k, v in outputs_pngs.items() if v.exists()}
    
    # === STORE LOAD SUMMARY ===
    data['_load_summary'] = load_summary
    data['_data_dir'] = data_dir
    data['_outputs_dir'] = outputs_dir
    data['_shap_dir'] = shap_dir
    
    return data


def display_load_summary(data: dict):
    """Display data loading summary in sidebar"""
    summary = data.get('_load_summary', [])
    
    st.caption("Data Load Status")
    
    for name, path, success, rows in summary:
        if success:
            st.success(f"OK: {name} - {rows:,} rows")
        else:
            st.error(f"MISSING: {name}")
    
    # PNG status
    shap_count = len(data.get('shap_pngs', {}))
    outputs_count = len(data.get('outputs_pngs', {}))
    
    if shap_count > 0:
        st.info(f"Found {shap_count} SHAP visualizations")
    if outputs_count > 0:
        st.info(f"Found {outputs_count} model charts")


def validate_critical_data(data: dict) -> bool:
    """
    Validate that critical data is loaded
    Returns True if OK, False if critical data missing
    """
    critical_keys = ['nodes', 'predictions']  # Minimum required
    missing = []
    
    for key in critical_keys:
        df = data.get(key)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            missing.append(key)
    
    if missing:
        st.error(f"Critical data missing: {', '.join(missing)}")
        st.error("Cannot proceed without nodes and predictions data!")
        return False
    
    return True


def validate_predictions_columns(predictions: pd.DataFrame) -> dict:
    """
    Validate predictions CSV has expected columns from your enhanced version
    Returns dict with column status
    """
    expected_columns = {
        'core': ['node_id', 'week_start', 'risk_score', 'y_pred'],
        'shap': ['top_features'],
        'comparison': ['percentile', 'risk_category'],
        'recommendations': ['recommendation'],
        'historical': ['avg_risk_last_4_weeks', 'max_risk_ever', 'risk_volatility'],
        'confidence': ['confidence_low', 'confidence_high'],
        'ensemble': ['risk_score_logreg', 'risk_score_rf', 'risk_score_xgb', 'risk_score_ensemble', 'model_agreement'],
        'trends': ['risk_score_last_week', 'risk_delta', 'trend']
    }
    
    status = {}
    for category, cols in expected_columns.items():
        found = [col for col in cols if col in predictions.columns]
        status[category] = {
            'expected': cols,
            'found': found,
            'complete': len(found) == len(cols)
        }
    
    return status