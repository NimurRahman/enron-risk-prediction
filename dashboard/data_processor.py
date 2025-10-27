# dashboard/data_processor.py
"""
Process and filter data based on user selections
Ensures risk consistency across all data sources
NOW WITH 4-BAND RISK MAPPING + PREDICTIONS RISK FILTER
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def map_risk_score_to_4bands(risk_score):
    """
    Map continuous risk score (0-1) to 4-band categorical system

    Thresholds:
    - Low: < 0.25
    - Elevated: 0.25 - 0.50
    - High: 0.50 - 0.75
    - Critical: >= 0.75
    """
    if pd.isna(risk_score):
        return None
    elif risk_score < 0.25:
        return 'Low'
    elif risk_score < 0.50:
        return 'Elevated'
    elif risk_score < 0.75:
        return 'High'
    else:
        return 'Critical'


def process_data(data_raw: dict, filters: dict) -> dict:
    """
    Apply filters and process data

    Args:
        data_raw: Dictionary of raw dataframes from data_loader
        filters: Dictionary of filter settings from sidebar

    Returns:
        dict: Processed and filtered data dictionary
    """
    processed = {}

    # Pass through unfiltered data
    processed['nodes'] = data_raw.get('nodes')
    processed['idmap'] = data_raw.get('idmap')
    processed['community_map'] = data_raw.get('community_map')
    processed['feature_importance'] = data_raw.get('feature_importance')
    processed['shap_pngs'] = data_raw.get('shap_pngs', {})
    processed['outputs_pngs'] = data_raw.get('outputs_pngs', {})

    # Get week lookback filter
    week_lookback = filters.get('week_lookback', 8)

    # ----------------------------------------------------------------------
    # Filter predictions (MAIN data source used by most tabs)
    # ----------------------------------------------------------------------
    preds = data_raw.get('predictions')
    if preds is not None and not preds.empty:
        preds_filtered = filter_by_weeks(preds, week_lookback, date_col='week_start')
        preds_filtered = preds_filtered.copy()

        # Add 4-BAND RISK CLASSIFICATION from risk_score if available
        if 'risk_score' in preds_filtered.columns:
            preds_filtered['risk_band_4'] = preds_filtered['risk_score'].apply(map_risk_score_to_4bands)

        # ✅ APPLY SIDEBAR RISK FILTER TO PREDICTIONS
        # Prefer 4-band; helper will fall back to y_pred / others if needed
        risk_bands = filters.get('risk_bands', [])
        if risk_bands:
            preds_filtered = filter_by_risk_bands(preds_filtered, risk_bands, risk_col='risk_band_4')

        processed['preds_filtered'] = preds_filtered
    else:
        processed['preds_filtered'] = None

    # ----------------------------------------------------------------------
    # Filter risk data (4-band system from node_week_risk_enhanced.parquet)
    # Kept for tabs that explicitly need the original risk parquet
    # ----------------------------------------------------------------------
    risk = data_raw.get('risk')
    if risk is not None and not risk.empty:
        risk_filtered = filter_by_weeks(risk, week_lookback, date_col='week_start')
        risk_filtered = risk_filtered.copy()

        # Apply risk band filter if provided (acts on 'risk_band' in this parquet)
        risk_bands = filters.get('risk_bands', [])
        if risk_bands:
            risk_filtered = filter_by_risk_bands(risk_filtered, risk_bands, risk_col='risk_band')

        processed['risk_filtered'] = risk_filtered
    else:
        processed['risk_filtered'] = None

    # ----------------------------------------------------------------------
    # Filter edges
    # ----------------------------------------------------------------------
    edges = data_raw.get('edges')
    if edges is not None and not edges.empty:
        edges_filtered = filter_by_weeks(edges, week_lookback, date_col='week_start')
        processed['edges_filtered'] = edges_filtered
    else:
        processed['edges_filtered'] = None

    # ----------------------------------------------------------------------
    # Filter features
    # ----------------------------------------------------------------------
    features = data_raw.get('features')
    if features is not None and not features.empty:
        features_filtered = filter_by_weeks(features, week_lookback, date_col='week_start')
        processed['features_filtered'] = features_filtered
    else:
        processed['features_filtered'] = None

    # Store filters for reference
    processed['filters'] = filters

    # Store metadata
    processed['_data_dir'] = data_raw.get('_data_dir')
    processed['_outputs_dir'] = data_raw.get('_outputs_dir')
    processed['_shap_dir'] = data_raw.get('_shap_dir')

    return processed


def filter_by_weeks(df: pd.DataFrame, weeks: int, date_col: str = 'week_start') -> pd.DataFrame:
    """
    Filter dataframe to last N weeks from the MOST RECENT date
    CRITICAL: Keeps LATEST weeks, removes OLDEST weeks

    Args:
        df: DataFrame to filter
        weeks: Number of weeks to keep (counting backwards from max date)
        date_col: Name of date column

    Returns:
        Filtered DataFrame with last N weeks only
    """
    if df is None or df.empty:
        return df

    # Check if date column exists
    if date_col not in df.columns:
        # Try alternate column names
        for alt in ['week', 'date', 'window_start']:
            if alt in df.columns:
                date_col = alt
                break
        else:
            # No date column found, return as-is
            return df

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

        # Drop rows with invalid dates
        df_copy = df_copy.dropna(subset=[date_col])

        if df_copy.empty:
            return df_copy

        # Get unique weeks and sort them (most recent first)
        unique_weeks = sorted(df_copy[date_col].unique(), reverse=True)

        # If weeks requested >= available weeks, return all
        if weeks >= len(unique_weeks):
            return df_copy

        # The cutoff is the Nth most recent unique week
        cutoff_date = unique_weeks[weeks - 1]

        # Keep dates >= cutoff (i.e., most recent N weeks)
        mask = df_copy[date_col] >= cutoff_date
        filtered = df_copy[mask]

        return filtered

    except Exception as e:
        print(f"Error filtering by weeks: {e}")
        return df


def filter_by_risk_bands(df: pd.DataFrame, risk_bands: list, risk_col: str = 'risk_band') -> pd.DataFrame:
    """
    Filter dataframe by selected risk bands

    Args:
        df: DataFrame to filter
        risk_bands: List of risk band names to keep
        risk_col: Name of risk band column

    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty or not risk_bands:
        return df

    # Ensure we don't mutate a view
    df = df.copy()

    # Check if risk column exists; fall back to common alternatives
    if risk_col not in df.columns:
        for alt in ['risk_band_4', 'y_pred', 'predicted_risk_band', 'risk_level']:
            if alt in df.columns:
                risk_col = alt
                break
        else:
            # No risk column found, return as-is
            return df

    try:
        mask = df[risk_col].isin(risk_bands)
        return df[mask]
    except Exception as e:
        print(f"Error filtering by risk bands: {e}")
        return df


def merge_risk_with_predictions(risk_df: pd.DataFrame, preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge risk data with predictions to create unified view
    Used when tabs need both 4-band (from risk) and 2-band/4-band (from preds) info

    Args:
        risk_df: DataFrame from node_week_risk_enhanced.parquet (4 bands, 'risk_band')
        preds_df: DataFrame from predictions_test_enhanced.csv (2 bands + optional 'risk_band_4')

    Returns:
        Merged DataFrame with both risk systems
    """
    if risk_df is None or preds_df is None:
        return preds_df if preds_df is not None else risk_df

    try:
        # Merge on node_id and week_start
        merged = preds_df.merge(
            risk_df[['node_id', 'week_start', 'risk_band']],
            on=['node_id', 'week_start'],
            how='left',
            suffixes=('', '_from_risk')
        )
        return merged
    except Exception as e:
        print(f"Error merging risk data: {e}")
        return preds_df
