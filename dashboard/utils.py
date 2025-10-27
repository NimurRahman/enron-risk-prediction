# dashboard/utils.py
"""
Utility functions for data processing and formatting
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

def safe_read_csv(filepath, **kwargs):
    """Safely read CSV with error handling"""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def safe_read_parquet(filepath):
    """Safely read Parquet with error handling"""
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def format_number(num: Optional[float], decimals: int = 2) -> str:
    """
    Format number with commas and decimals
    
    Args:
        num: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string or "N/A" if invalid
    """
    if num is None or pd.isna(num):
        return "N/A"
    try:
        if decimals == 0:
            return f"{int(num):,}"
        return f"{num:,.{decimals}f}"
    except:
        return str(num)

def parse_top_features(feature_str: str) -> List[Tuple[str, float]]:
    """
    Parse top_features string into list of (feature, value) tuples
    
    Example input: "degree_ma4=2.15; betweenness=0.34; after_hours_pct=0.71"
    Example output: [("degree_ma4", 2.15), ("betweenness", 0.34), ("after_hours_pct", 0.71)]
    
    Args:
        feature_str: String with format "feature1=value1; feature2=value2"
    
    Returns:
        List of (feature_name, feature_value) tuples
    """
    if pd.isna(feature_str) or not feature_str:
        return []
    
    try:
        pairs = []
        # Split by semicolon
        items = str(feature_str).split(';')
        
        for item in items:
            item = item.strip()
            if '=' in item:
                # Split on first = only (in case value contains =)
                key, val = item.split('=', 1)
                key = key.strip()
                val = val.strip()
                
                # Try to convert value to float
                try:
                    val_float = float(val)
                    pairs.append((key, val_float))
                except ValueError:
                    # If conversion fails, skip this item
                    continue
        
        return pairs
    except Exception as e:
        print(f"Error parsing features: {e}")
        return []

def calculate_percentile(value: float, series: pd.Series) -> float:
    """
    Calculate percentile rank of a value in a series
    
    Args:
        value: The value to rank
        series: Series of values to compare against
    
    Returns:
        Percentile (0-100) or NaN if invalid
    """
    if pd.isna(value) or series.empty:
        return np.nan
    
    try:
        # Remove NaN values from series
        clean_series = series.dropna()
        
        if clean_series.empty:
            return np.nan
        
        # Calculate percentile (percentage of values below this value)
        percentile = (clean_series < value).sum() / len(clean_series) * 100
        
        return percentile
    except Exception as e:
        print(f"Error calculating percentile: {e}")
        return np.nan

def get_risk_color(risk_level: str, color_map: dict) -> str:
    """
    Get color for risk level from color map
    
    Args:
        risk_level: Risk band name
        color_map: Dictionary mapping risk levels to colors
    
    Returns:
        Hex color code
    """
    return color_map.get(risk_level, "#95a5a6")

def filter_by_date_range(df: pd.DataFrame, date_col: str, start_date, end_date):
    """
    Filter dataframe by date range
    
    Args:
        df: DataFrame to filter
        date_col: Name of date column
        start_date: Start date (datetime-like)
        end_date: End date (datetime-like)
    
    Returns:
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        mask = (df_copy[date_col] >= pd.to_datetime(start_date)) & \
               (df_copy[date_col] <= pd.to_datetime(end_date))
        
        return df_copy[mask]
    except Exception as e:
        print(f"Error filtering dates: {e}")
        return df

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, return default if division fails
    
    Args:
        numerator: Top number
        denominator: Bottom number
        default: Value to return if division fails
    
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
            return default
        return numerator / denominator
    except:
        return default

def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate long text with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: String to add at end if truncated
    
    Returns:
        Truncated text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix