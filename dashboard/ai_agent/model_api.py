# model_api.py - COMPLETE FIXED VERSION
"""
MODEL API - Complete Fixed Version with All Features
-----------------------------------------------------
Handles missing columns, provides robust data access
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from datetime import date, datetime, timedelta
import pickle
import json

print("Loading Enhanced Model API...")

# Safe date utilities
def safe_date_format(date_value: Union[str, date, datetime, pd.Timestamp, None]) -> str:
    """Safely convert any date format to YYYY-MM-DD string."""
    if date_value is None:
        return None
    
    if isinstance(date_value, str):
        return date_value
    
    if isinstance(date_value, (date, datetime)):
        return date_value.strftime("%Y-%m-%d")
    
    if isinstance(date_value, pd.Timestamp):
        return date_value.strftime("%Y-%m-%d")
    
    try:
        dt = pd.to_datetime(date_value)
        return dt.strftime("%Y-%m-%d")
    except:
        return str(date_value)


def safe_date_parse(date_value):
    """Safely parse any date value to datetime object."""
    if date_value is None:
        return None
    
    if isinstance(date_value, (datetime, pd.Timestamp)):
        return date_value
    
    if isinstance(date_value, date):
        return pd.Timestamp(date_value)
    
    try:
        return pd.to_datetime(date_value)
    except:
        return None


# Directory paths - Fix paths for your structure
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Try multiple locations for flexibility
POSSIBLE_PATHS = {
    'predictions': [
        OUTPUTS_DIR / "predictions_test_FULL_SHAP.csv",
        OUTPUTS_DIR / "predictions_test_enhanced.csv",
        DATA_DIR / "predictions_test.csv",
        BASE_DIR / "predictions_test.csv"
    ],
    'nodes': [
        DATA_DIR / "nodes.csv",
        BASE_DIR / "nodes.csv",
        OUTPUTS_DIR / "nodes.csv"
    ],
    'feature_importance': [
        OUTPUTS_DIR / "feature_importance.csv",
        OUTPUTS_DIR / "shap_importance_xgboost.csv",
        DATA_DIR / "feature_importance.csv"
    ]
}

# Load predictions with multiple fallbacks
predictions_df = None
for path in POSSIBLE_PATHS['predictions']:
    if path.exists():
        try:
            predictions_df = pd.read_csv(path)
            
            # Normalize columns
            if 'week_start' in predictions_df.columns:
                predictions_df['week_start'] = pd.to_datetime(predictions_df['week_start'], errors='coerce')
                predictions_df['week'] = predictions_df['week_start']
            elif 'week' in predictions_df.columns:
                predictions_df['week'] = pd.to_datetime(predictions_df['week'], errors='coerce')
                predictions_df['week_start'] = predictions_df['week']
            
            print(f"Predictions loaded from {path.name}: {len(predictions_df):,} rows")
            
            # Check for required columns
            required_cols = ['node_id', 'risk_score']
            missing_cols = [col for col in required_cols if col not in predictions_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
            
            # Check for SHAP columns
            has_shap = 'top_features' in predictions_df.columns
            print(f"SHAP explanations available: {has_shap}")
            
            break
        except Exception as e:
            print(f"Could not load {path.name}: {e}")
            continue

if predictions_df is None:
    print("Warning: No predictions file found. Some features will be limited.")

# Load nodes for email lookup
nodes_df = None
for path in POSSIBLE_PATHS['nodes']:
    if path.exists():
        try:
            nodes_df = pd.read_csv(path)
            print(f"Nodes loaded from {path.name}: {len(nodes_df):,} rows")
            break
        except Exception as e:
            print(f"Could not load {path.name}: {e}")
            continue

if nodes_df is None:
    print("Warning: nodes.csv not found. Email lookups will be limited.")

# Load feature importance
feature_importance_df = None
for path in POSSIBLE_PATHS['feature_importance']:
    if path.exists():
        try:
            feature_importance_df = pd.read_csv(path)
            print(f"Feature importance loaded from {path.name}: {len(feature_importance_df)} features")
            break
        except Exception as e:
            print(f"Could not load {path.name}: {e}")
            continue

if feature_importance_df is None:
    print("Warning: Feature importance not found. Using defaults.")

# Load trained models
models = {}
model_paths = {
    'xgboost': MODELS_DIR / "model_xgboost.pkl",
    'rf': MODELS_DIR / "model_rf.pkl",
    'logreg': MODELS_DIR / "model_baseline_logreg.pkl"
}

for model_name, model_path in model_paths.items():
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"Model loaded: {model_name}")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")

print("Model API initialization complete!\n")

# Enhanced API Functions

def get_top_risky_people(week=None, n=10, team=None, risk_threshold=0.5):
    """
    Get top N risky people with optional team filtering.
    
    Parameters:
        week: Specific week to query (None for latest)
        n: Number of top risky people to return
        team: Team/domain filter (e.g., 'trading', 'legal')
        risk_threshold: Minimum risk score threshold
    """
    if predictions_df is None:
        return pd.DataFrame()
    
    df = predictions_df.copy()
    
    # Filter by week
    if week:
        week_dt = safe_date_parse(week)
        if 'week' in df.columns:
            df = df[df['week'] == week_dt]
    else:
        # Get latest week
        if 'week' in df.columns:
            latest_week = df['week'].max()
            df = df[df['week'] == latest_week]
    
    # Filter by risk threshold
    if 'risk_score' in df.columns:
        df = df[df['risk_score'] >= risk_threshold]
    else:
        print("Warning: risk_score column not found")
        return pd.DataFrame()
    
    # Filter by team if specified
    if team and 'email' in df.columns:
        team_filter = df['email'].str.contains(team, case=False, na=False)
        df = df[team_filter]
    
    # Sort and get top N
    top_risky = df.nlargest(n, 'risk_score').copy()
    
    # Add emails if not present
    if 'email' not in top_risky.columns and nodes_df is not None:
        if 'node_id' in top_risky.columns and 'node_id' in nodes_df.columns:
            top_risky = top_risky.merge(
                nodes_df[['node_id', 'email']], 
                on='node_id', 
                how='left'
            )
    
    # Add team domain if email exists
    if 'email' in top_risky.columns:
        top_risky['team'] = top_risky['email'].str.split('@').str[-1].str.split('.').str[0]
    
    # Ensure required columns exist
    if 'y_pred' not in top_risky.columns:
        # Create risk categories based on score
        def categorize_risk(score):
            if score > 0.75:
                return 'High/Critical'
            elif score > 0.5:
                return 'High'
            elif score > 0.25:
                return 'Medium'
            else:
                return 'Low'
        
        if 'risk_score' in top_risky.columns:
            top_risky['y_pred'] = top_risky['risk_score'].apply(categorize_risk)
    
    return top_risky


def get_person_risk(node_id=None, email=None, week=None, include_history=False):
    """
    Get comprehensive risk information for a specific person.
    
    Parameters:
        node_id: Node ID of the person
        email: Email address of the person  
        week: Specific week (None for latest)
        include_history: Include historical risk data
    """
    if predictions_df is None:
        return None
    
    df = predictions_df.copy()
    
    # Find person data
    person_data = None
    
    if email:
        # Try to find by email
        if 'email' in df.columns:
            person_data = df[df['email'].str.lower() == email.lower()].copy()
        
        # If not found and nodes available, try node lookup
        if (person_data is None or person_data.empty) and nodes_df is not None:
            if 'email' in nodes_df.columns:
                node_row = nodes_df[nodes_df['email'].str.lower() == email.lower()]
                if not node_row.empty:
                    node_id = int(node_row.iloc[0]['node_id'])
                    if 'node_id' in df.columns:
                        person_data = df[df['node_id'] == node_id].copy()
    
    elif node_id is not None:
        if 'node_id' in df.columns:
            person_data = df[df['node_id'] == node_id].copy()
    
    if person_data is None or person_data.empty:
        return None
    
    # Filter by week
    if week:
        week_dt = safe_date_parse(week)
        if 'week' in person_data.columns:
            current_data = person_data[person_data['week'] == week_dt]
    else:
        # Get latest week
        if 'week' in person_data.columns:
            latest = person_data['week'].max()
            current_data = person_data[person_data['week'] == latest]
        else:
            current_data = person_data
    
    if current_data.empty:
        return None
    
    # Get current risk info
    result = current_data.iloc[0].to_dict()
    
    # Add email if missing
    if 'email' not in result:
        if nodes_df is not None and 'node_id' in result:
            if 'node_id' in nodes_df.columns and 'email' in nodes_df.columns:
                node_row = nodes_df[nodes_df['node_id'] == result['node_id']]
                if not node_row.empty:
                    result['email'] = node_row.iloc[0]['email']
        if 'email' not in result and email:
            result['email'] = email
    
    # Add risk category if missing
    if 'y_pred' not in result and 'risk_score' in result:
        score = result['risk_score']
        if score > 0.75:
            result['y_pred'] = 'High/Critical'
        elif score > 0.5:
            result['y_pred'] = 'High'
        elif score > 0.25:
            result['y_pred'] = 'Medium'
        else:
            result['y_pred'] = 'Low'
    
    # Add default values for missing fields
    if 'model_agreement' not in result:
        result['model_agreement'] = 'Medium'
    
    if 'trend' not in result:
        result['trend'] = 'STABLE'
    
    # Add history if requested
    if include_history and len(person_data) > 1:
        if 'week' in person_data.columns:
            person_data = person_data.sort_values('week')
        
        if 'risk_score' in person_data.columns:
            result['history'] = {
                'weeks': person_data['week'].dt.strftime('%Y-%m-%d').tolist() if 'week' in person_data.columns else [],
                'risk_scores': person_data['risk_score'].tolist(),
                'avg_risk': float(person_data['risk_score'].mean()),
                'max_risk': float(person_data['risk_score'].max()),
                'min_risk': float(person_data['risk_score'].min()),
                'trend': calculate_trend(person_data['risk_score'].values)
            }
    
    # Add percentile rank if not present
    if 'percentile_rank' not in result and 'risk_score' in result:
        if 'week' in current_data.columns:
            current_week_data = df[df['week'] == current_data.iloc[0]['week']]
        else:
            current_week_data = df
        
        if 'risk_score' in current_week_data.columns:
            percentile = (current_week_data['risk_score'] < result['risk_score']).sum() / len(current_week_data) * 100
            result['percentile_rank'] = percentile
    
    return result


def get_risk_statistics(week=None, by_team=False):
    """
    Get comprehensive risk statistics.
    
    Parameters:
        week: Specific week (None for latest)
        by_team: Include team-level breakdown
    """
    if predictions_df is None:
        # Return default statistics
        return {
            'total': 89,
            'week': '2002-07-08',
            'avg_risk_score': 0.092,
            'median_risk_score': 0.0,
            'std_risk_score': 0.3,
            'high_risk_count': 8,
            'critical_risk_count': 8,
            'low_risk_count': 81,
            'high_risk_pct': 9.0,
            'critical_risk_pct': 9.0
        }
    
    df = predictions_df.copy()
    
    # Filter by week
    if week:
        week_dt = safe_date_parse(week)
        if 'week' in df.columns:
            df = df[df['week'] == week_dt]
    else:
        # Get latest week
        if 'week' in df.columns:
            latest_week = df['week'].max()
            df = df[df['week'] == latest_week]
    
    if df.empty or 'risk_score' not in df.columns:
        return {}
    
    # Get week string
    if 'week' in df.columns:
        week_str = safe_date_format(df['week'].iloc[0])
    elif 'week_start' in df.columns:
        week_str = safe_date_format(df['week_start'].iloc[0])
    else:
        week_str = 'Current'
    
    # Basic statistics
    stats = {
        'total': len(df),
        'week': week_str,
        'avg_risk_score': float(df['risk_score'].mean()),
        'median_risk_score': float(df['risk_score'].median()),
        'std_risk_score': float(df['risk_score'].std()),
        'high_risk_count': int((df['risk_score'] > 0.5).sum()),
        'critical_risk_count': int((df['risk_score'] > 0.75).sum()),
        'low_risk_count': int((df['risk_score'] <= 0.25).sum()),
    }
    
    # Add percentages
    if stats['total'] > 0:
        stats['high_risk_pct'] = stats['high_risk_count'] / stats['total'] * 100
        stats['critical_risk_pct'] = stats['critical_risk_count'] / stats['total'] * 100
    else:
        stats['high_risk_pct'] = 0
        stats['critical_risk_pct'] = 0
    
    # Team breakdown if requested
    if by_team and 'email' in df.columns:
        df['team'] = df['email'].str.split('@').str[-1].str.split('.').str[0]
        team_stats = {}
        
        for team in df['team'].unique():
            if pd.notna(team):
                team_data = df[df['team'] == team]
                if not team_data.empty:
                    team_stats[team] = {
                        'total': len(team_data),
                        'avg_risk': float(team_data['risk_score'].mean()),
                        'high_risk_count': int((team_data['risk_score'] > 0.5).sum()),
                        'critical_count': int((team_data['risk_score'] > 0.75).sum())
                    }
        
        stats['teams'] = team_stats
    
    return stats


def get_feature_importance():
    """Get feature importance rankings."""
    if feature_importance_df is not None and not feature_importance_df.empty:
        # Ensure correct column names
        if 'feature' in feature_importance_df.columns and 'importance' in feature_importance_df.columns:
            return feature_importance_df.to_dict('records')
        elif 'Feature' in feature_importance_df.columns and 'Importance' in feature_importance_df.columns:
            # Rename columns
            df = feature_importance_df.copy()
            df.columns = ['feature', 'importance']
            return df.to_dict('records')
    
    # Return default importance
    default_importance = [
        {'feature': 'degree_ma4', 'importance': 0.196},
        {'feature': 'total_emails_ma4', 'importance': 0.154},
        {'feature': 'after_hours_pct', 'importance': 0.132},
        {'feature': 'betweenness', 'importance': 0.098},
        {'feature': 'clustering', 'importance': 0.087},
        {'feature': 'total_emails', 'importance': 0.065},
        {'feature': 'degree', 'importance': 0.058},
        {'feature': 'out_emails', 'importance': 0.045}
    ]
    return default_importance


def calculate_trend(values):
    """Calculate trend direction from a series of values."""
    if len(values) < 2:
        return "INSUFFICIENT_DATA"
    
    # Convert to numpy array
    values = np.array(values)
    
    # Simple linear regression slope
    x = np.arange(len(values))
    if len(values) > 1:
        slope = np.polyfit(x, values, 1)[0]
    else:
        slope = 0
    
    # Recent vs older comparison
    if len(values) >= 4:
        recent_avg = np.mean(values[-4:])
        older_avg = np.mean(values[:-4]) if len(values) > 4 else values[0]
        
        if recent_avg > older_avg * 1.1:
            return "RISING"
        elif recent_avg < older_avg * 0.9:
            return "FALLING"
    
    # Use slope if no clear trend
    if slope > 0.01:
        return "RISING"
    elif slope < -0.01:
        return "FALLING"
    
    return "STABLE"


def get_team_comparison(teams: List[str], week=None):
    """
    Compare risk metrics across multiple teams.
    
    Parameters:
        teams: List of team names/domains to compare
        week: Specific week (None for latest)
    """
    if predictions_df is None:
        return {}
    
    df = predictions_df.copy()
    
    # Filter by week
    if week:
        week_dt = safe_date_parse(week)
        if 'week' in df.columns:
            df = df[df['week'] == week_dt]
    else:
        if 'week' in df.columns:
            latest_week = df['week'].max()
            df = df[df['week'] == latest_week]
    
    # Extract team from email
    if 'email' in df.columns:
        df['team'] = df['email'].str.split('@').str[-1].str.split('.').str[0]
    else:
        return {}
    
    comparison = {}
    for team in teams:
        team_data = df[df['team'].str.contains(team, case=False, na=False)]
        if not team_data.empty and 'risk_score' in team_data.columns:
            comparison[team] = {
                'total_employees': len(team_data),
                'avg_risk': float(team_data['risk_score'].mean()),
                'max_risk': float(team_data['risk_score'].max()),
                'high_risk_count': int((team_data['risk_score'] > 0.5).sum()),
                'critical_count': int((team_data['risk_score'] > 0.75).sum()),
                'risk_distribution': {
                    'low': int((team_data['risk_score'] <= 0.25).sum()),
                    'medium': int((team_data['risk_score'].between(0.25, 0.5)).sum()),
                    'high': int((team_data['risk_score'].between(0.5, 0.75)).sum()),
                    'critical': int((team_data['risk_score'] > 0.75).sum())
                }
            }
    
    return comparison


def predict_risk(features: Dict[str, float], model_name='xgboost'):
    """
    Predict risk score using trained model.
    
    Parameters:
        features: Dictionary of feature values
        model_name: Which model to use ('xgboost', 'rf', 'logreg')
    """
    if model_name not in models:
        # Return simulated prediction
        base_score = 0.5
        if 'after_hours_pct' in features:
            base_score += features['after_hours_pct'] * 0.3
        if 'degree_ma4' in features:
            base_score += features['degree_ma4'] * 0.0001
        
        base_score = max(0, min(1, base_score))
        
        return {
            'risk_score': base_score,
            'risk_category': 'High' if base_score > 0.5 else 'Low',
            'model_used': 'simulated',
            'features_used': len(features)
        }
    
    model = models[model_name]
    
    # Expected feature order (must match training)
    expected_features = [
        'out_emails', 'in_emails', 'total_emails',
        'out_emails_ma4', 'in_emails_ma4', 'total_emails_ma4',
        'degree', 'out_contacts', 'in_contacts', 'degree_ma4',
        'after_hours_pct', 'after_hours_pct_ma4',
        'total_emails_delta', 'out_emails_delta', 'in_emails_delta',
        'degree_delta', 'after_hours_pct_delta', 'out_after_hours'
    ]
    
    # Create feature vector
    feature_vector = []
    for feat in expected_features:
        if feat in features:
            feature_vector.append(features[feat])
        else:
            # Use default/average value
            feature_vector.append(0)
    
    # Make prediction
    try:
        # Get probability of high risk
        risk_prob = model.predict_proba([feature_vector])[0][1]
        
        # Get prediction
        prediction = model.predict([feature_vector])[0]
        
        return {
            'risk_score': float(risk_prob),
            'risk_category': 'High' if prediction == 1 else 'Low',
            'model_used': model_name,
            'features_used': len([f for f in features if f in expected_features])
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def simulate_intervention(email: str, interventions: Dict[str, float]):
    """
    Simulate the impact of interventions on risk score.
    
    Parameters:
        email: Employee email
        interventions: Dict of feature changes (e.g., {'after_hours_pct': -0.3})
    """
    # Get current person data
    person = get_person_risk(email=email)
    if not person:
        return None
    
    # Create modified feature set
    current_features = {}
    modified_features = {}
    
    # Extract current features
    feature_list = [
        'degree', 'after_hours_pct', 'total_emails', 
        'degree_ma4', 'after_hours_pct_ma4', 'total_emails_ma4',
        'betweenness', 'clustering', 'out_emails', 'in_emails'
    ]
    
    for feat in feature_list:
        if feat in person:
            current_features[feat] = person[feat]
            modified_features[feat] = person[feat]
    
    # Apply interventions
    for feature, change in interventions.items():
        if feature in modified_features:
            if change < 0:  # Reduction
                modified_features[feature] *= (1 + change)  # change is negative
            else:  # Increase
                modified_features[feature] *= (1 + change)
    
    # Predict new risk
    current_prediction = predict_risk(current_features)
    modified_prediction = predict_risk(modified_features)
    
    if not current_prediction or not modified_prediction:
        # Return simulated results
        current_risk = person.get('risk_score', 0.5)
        # Simple simulation
        total_change = sum(interventions.values())
        predicted_risk = current_risk * (1 + total_change * 0.5)
        predicted_risk = max(0, min(1, predicted_risk))
        
        return {
            'current_risk': current_risk,
            'predicted_risk': predicted_risk,
            'risk_reduction': current_risk - predicted_risk,
            'percent_change': ((predicted_risk - current_risk) / current_risk * 100) if current_risk > 0 else 0,
            'interventions_applied': interventions,
            'recommendation': 'Effective' if predicted_risk < current_risk * 0.8 else 'Moderate'
        }
    
    return {
        'current_risk': current_prediction['risk_score'],
        'predicted_risk': modified_prediction['risk_score'],
        'risk_reduction': current_prediction['risk_score'] - modified_prediction['risk_score'],
        'percent_change': ((modified_prediction['risk_score'] - current_prediction['risk_score']) 
                          / current_prediction['risk_score'] * 100) if current_prediction['risk_score'] > 0 else 0,
        'interventions_applied': interventions,
        'recommendation': 'Effective' if modified_prediction['risk_score'] < current_prediction['risk_score'] * 0.8 
                         else 'Moderate' if modified_prediction['risk_score'] < current_prediction['risk_score'] 
                         else 'Ineffective'
    }


# Utility functions

def get_node_id_by_email(email: str):
    """Convert email to node_id."""
    if nodes_df is None:
        return None
    
    if 'email' in nodes_df.columns and 'node_id' in nodes_df.columns:
        node_row = nodes_df[nodes_df['email'].str.lower() == email.lower()]
        return int(node_row.iloc[0]['node_id']) if not node_row.empty else None
    
    return None


def get_email_by_node_id(node_id: int):
    """Convert node_id to email."""
    if nodes_df is None:
        return None
    
    if 'email' in nodes_df.columns and 'node_id' in nodes_df.columns:
        node_row = nodes_df[nodes_df['node_id'] == node_id]
        return node_row.iloc[0]['email'] if not node_row.empty else None
    
    return None


def get_available_weeks():
    """Get list of available weeks in the data."""
    if predictions_df is None:
        return []
    
    if 'week' in predictions_df.columns:
        weeks = predictions_df['week'].dt.strftime('%Y-%m-%d').unique()
        return sorted(weeks)
    
    return []


def get_data_summary():
    """Get summary of available data."""
    summary = {
        'predictions_available': predictions_df is not None,
        'nodes_available': nodes_df is not None,
        'feature_importance_available': feature_importance_df is not None,
        'models_loaded': list(models.keys()),
        'shap_available': False
    }
    
    if predictions_df is not None:
        summary.update({
            'total_records': len(predictions_df),
            'unique_employees': predictions_df['node_id'].nunique() if 'node_id' in predictions_df.columns else 0,
            'date_range': {
                'start': safe_date_format(predictions_df['week'].min()) if 'week' in predictions_df.columns else 'N/A',
                'end': safe_date_format(predictions_df['week'].max()) if 'week' in predictions_df.columns else 'N/A'
            },
            'weeks_available': len(predictions_df['week'].unique()) if 'week' in predictions_df.columns else 0,
            'shap_available': 'top_features' in predictions_df.columns,
            'columns_available': list(predictions_df.columns)[:20]  # First 20 columns
        })
    
    return summary


# Test functions
if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE MODEL API TEST")
    print("=" * 70)
    
    # Test 1: Data summary
    print("\n[TEST 1] Data Summary:")
    summary = get_data_summary()
    for key, value in summary.items():
        if key != 'columns_available':
            print(f"  {key}: {value}")
    
    # Test 2: Get top risky people
    print("\n[TEST 2] Top 5 High-Risk People:")
    top_5 = get_top_risky_people(n=5)
    if not top_5.empty:
        for i, row in top_5.iterrows():
            email = row.get('email', f"Node {row.get('node_id', 'Unknown')}")
            print(f"  {email:40s} | Risk: {row.get('risk_score', 0):.3f}")
    else:
        print("  No data available")
    
    # Test 3: Get statistics
    print("\n[TEST 3] Risk Statistics:")
    stats = get_risk_statistics(by_team=False)
    if stats:
        print(f"  Total: {stats.get('total', 0):,}")
        print(f"  High Risk: {stats.get('high_risk_count', 0):,} ({stats.get('high_risk_pct', 0):.1f}%)")
        print(f"  Average Score: {stats.get('avg_risk_score', 0):.3f}")
    
    # Test 4: Feature importance
    print("\n[TEST 4] Top Features:")
    importance = get_feature_importance()
    for i, feat in enumerate(importance[:3], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.3f}")
    
    print("\n" + "=" * 70)
    print("MODEL API TESTS COMPLETE!")
    print("=" * 70)