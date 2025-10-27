# dashboard/simulator_engine.py
"""
Risk Simulator Engine - Model Loading and Prediction
FIXED: Properly extracts models and scalers from bundles, scales features before prediction
ENHANCED: Added extensive debugging to diagnose prediction failures
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import shap

class RiskSimulatorEngine:
    """Engine for loading models and simulating risk predictions"""
    
    def __init__(self, models_dir: Path):
        """
        Initialize simulator with trained models
        
        Args:
            models_dir: Path to directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.model_features = {}
        self.explainers = {}
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load all available models and extract from bundles"""
        try:
            # Load XGBoost model bundle
            xgb_path = self.models_dir / "model_xgboost.pkl"
            if xgb_path.exists():
                print(f"\n🔍 Loading XGBoost from: {xgb_path}")
                bundle = joblib.load(xgb_path)
                print(f"   Type loaded: {type(bundle)}")
                
                if isinstance(bundle, dict):
                    print(f"   Keys in bundle: {list(bundle.keys())}")
                    self.models['xgboost'] = bundle['model']
                    self.scalers['xgboost'] = bundle['scaler']
                    self.model_features['xgboost'] = bundle['features']
                    print(f"   ✓ Extracted model type: {type(self.models['xgboost'])}")
                    print(f"   ✓ Extracted scaler type: {type(self.scalers['xgboost'])}")
                    print(f"   ✓ Features count: {len(self.model_features['xgboost'])}")
                    print(f"✓ Loaded XGBoost model (with scaler)")
                else:
                    self.models['xgboost'] = bundle
                    print(f"⚠ Loaded XGBoost model (no bundle) - Type: {type(bundle)}")
            
            # Load Random Forest model bundle
            rf_path = self.models_dir / "model_rf.pkl"
            if rf_path.exists():
                print(f"\n🔍 Loading Random Forest from: {rf_path}")
                bundle = joblib.load(rf_path)
                print(f"   Type loaded: {type(bundle)}")
                
                if isinstance(bundle, dict):
                    print(f"   Keys in bundle: {list(bundle.keys())}")
                    self.models['rf'] = bundle['model']
                    self.scalers['rf'] = bundle['scaler']
                    self.model_features['rf'] = bundle['features']
                    print(f"   ✓ Extracted model type: {type(self.models['rf'])}")
                    print(f"✓ Loaded Random Forest model (with scaler)")
                else:
                    self.models['rf'] = bundle
                    print(f"⚠ Loaded RF model (no bundle) - Type: {type(bundle)}")
            
            # Load Baseline model bundle
            baseline_path = self.models_dir / "model_baseline_logreg.pkl"
            if baseline_path.exists():
                print(f"\n🔍 Loading Baseline from: {baseline_path}")
                bundle = joblib.load(baseline_path)
                print(f"   Type loaded: {type(bundle)}")
                
                if isinstance(bundle, dict):
                    print(f"   Keys in bundle: {list(bundle.keys())}")
                    self.models['baseline'] = bundle['model']
                    self.scalers['baseline'] = bundle['scaler']
                    self.model_features['baseline'] = bundle['features']
                    print(f"   ✓ Extracted model type: {type(self.models['baseline'])}")
                    print(f"✓ Loaded Baseline model (with scaler)")
                else:
                    self.models['baseline'] = bundle
                    print(f"⚠ Loaded Baseline model (no bundle) - Type: {type(bundle)}")
            
            if not self.models:
                raise Exception("No models loaded!")
            
            print(f"\n✓ Simulator engine ready with {len(self.models)} models")
            print(f"  Models with scalers: {len(self.scalers)}")
            print(f"  Model types: {[type(m).__name__ for m in self.models.values()]}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_feature_names(self, model_name: str = 'xgboost') -> list:
        """Get expected feature names from model"""
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
        
        # Try to get from bundle first
        if model_name in self.model_features:
            return self.model_features[model_name]
        
        # Fallback: try model object
        model = self.models[model_name]
        
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'get_booster'):
            return model.get_booster().feature_names
        elif hasattr(model, 'feature_name_'):
            return model.feature_name_
        else:
            # Last resort
            return [
                'degree', 'degree_ma4', 'in_degree', 'out_degree',
                'betweenness', 'betweenness_ma4', 'closeness',
                'clustering', 'clustering_ma4', 'kcore',
                'total_emails', 'total_emails_ma4', 'out_emails', 'in_emails',
                'unique_contacts', 'out_contacts', 'in_contacts',
                'after_hours_pct', 'after_hours_pct_ma4'
            ]
    
    def predict(self, features_df: pd.DataFrame, model_name: str = 'xgboost') -> Dict:
        """
        Make prediction using specified model
        
        Args:
            features_df: DataFrame with feature values (single row or multiple)
            model_name: Which model to use ('xgboost', 'rf', 'baseline')
        
        Returns:
            dict with predictions and probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        print(f"\n🔍 PREDICT DEBUG for {model_name}:")
        print(f"   Model type: {type(model)}")
        print(f"   Model class: {model.__class__.__name__}")
        print(f"   Has predict_proba: {hasattr(model, 'predict_proba')}")
        print(f"   Input shape: {features_df.shape}")
        print(f"   Input columns: {list(features_df.columns)}")
        
        # CRITICAL: Scale features if scaler exists
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            print(f"   Scaler type: {type(scaler)}")
            try:
                features_scaled = scaler.transform(features_df)
                features_to_use = pd.DataFrame(
                    features_scaled, 
                    columns=features_df.columns,
                    index=features_df.index
                )
                print(f"   ✓ Scaled features for {model_name}")
                print(f"   Scaled shape: {features_to_use.shape}")
            except Exception as e:
                print(f"   ⚠ Scaling failed for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                features_to_use = features_df
        else:
            print(f"   ⚠ No scaler for {model_name}")
            features_to_use = features_df
        
        # Check if model expects specific features
        if model_name in self.model_features:
            expected_features = self.model_features[model_name]
            print(f"   Expected features count: {len(expected_features)}")
            print(f"   Expected features: {expected_features[:5]}...")
            
            # Check if we have all expected features
            missing = set(expected_features) - set(features_to_use.columns)
            if missing:
                print(f"   ⚠ Missing features: {missing}")
            
            # Reorder columns to match expected order
            try:
                features_to_use = features_to_use[expected_features]
                print(f"   ✓ Reordered features to match model training")
            except KeyError as e:
                print(f"   ❌ Cannot reorder features: {e}")
        
        # Get prediction probabilities
        try:
            print(f"   Attempting prediction...")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_to_use)
                print(f"   ✓ predict_proba returned shape: {proba.shape}")
                risk_score = proba[:, 1]
                prediction = (risk_score >= 0.5).astype(int)
            else:
                prediction = model.predict(features_to_use)
                risk_score = prediction
            
            print(f"   ✓ Prediction successful for {model_name}: risk={risk_score[0]:.3f}")
            
            return {
                'risk_score': risk_score,
                'prediction': prediction,
                'proba': proba if hasattr(model, 'predict_proba') else None
            }
        except Exception as e:
            print(f"   ❌ Prediction failed for {model_name}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {e}")
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            return None
    
    def predict_all_models(self, features_df: pd.DataFrame) -> Dict:
        """Get predictions from all available models"""
        print(f"\n🎯 Running predictions for ALL models...")
        results = {}
        for model_name in self.models.keys():
            try:
                results[model_name] = self.predict(features_df, model_name)
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results[model_name] = None
        
        # Summary
        success_count = sum(1 for r in results.values() if r is not None)
        print(f"\n📊 Prediction Summary: {success_count}/{len(results)} models succeeded")
        
        return results
    
    def calculate_shap(self, features_df: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """
        Calculate SHAP values for feature importance
        
        Args:
            features_df: Features to explain
            model_name: Which model to use
        
        Returns:
            SHAP values array
        """
        if model_name not in self.models:
            return None
        
        try:
            model = self.models[model_name]
            
            # CRITICAL: Scale features if scaler exists
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                features_scaled = scaler.transform(features_df)
                features_to_use = pd.DataFrame(
                    features_scaled, 
                    columns=features_df.columns,
                    index=features_df.index
                )
            else:
                features_to_use = features_df
            
            # Reorder features if needed
            if model_name in self.model_features:
                expected_features = self.model_features[model_name]
                features_to_use = features_to_use[expected_features]
            
            # Create or get explainer
            if model_name not in self.explainers:
                if model_name == 'xgboost':
                    self.explainers[model_name] = shap.TreeExplainer(model)
                elif model_name == 'rf':
                    self.explainers[model_name] = shap.TreeExplainer(model)
                else:
                    background = shap.sample(features_to_use, 100)
                    self.explainers[model_name] = shap.KernelExplainer(
                        model.predict_proba, 
                        background
                    )
            
            explainer = self.explainers[model_name]
            shap_values = explainer.shap_values(features_to_use)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            return shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP: {e}")
            return None
    
    def map_risk_score_to_4bands(self, risk_score: float) -> str:
        """Map continuous risk score to 4-band system"""
        if risk_score < 0.25:
            return 'Low'
        elif risk_score < 0.50:
            return 'Elevated'
        elif risk_score < 0.75:
            return 'High'
        else:
            return 'Critical'
    
    def generate_recommendations(self, 
                                 original_features: pd.Series,
                                 modified_features: pd.Series,
                                 shap_values: np.ndarray,
                                 feature_names: list) -> list:
        """
        Generate actionable recommendations based on simulation
        
        Args:
            original_features: Original feature values
            modified_features: Modified feature values  
            shap_values: SHAP values for modified prediction
            feature_names: List of feature names
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if shap_values is not None and len(shap_values.shape) > 0:
            if len(shap_values.shape) == 2:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            abs_shap = np.abs(shap_vals)
            top_indices = np.argsort(abs_shap)[-5:][::-1]
            
            for idx in top_indices:
                feat = feature_names[idx]
                shap_val = shap_vals[idx]
                orig_val = original_features[feat] if feat in original_features.index else 0
                mod_val = modified_features[feat] if feat in modified_features.index else 0
                
                if shap_val > 0.05:
                    if 'after_hours' in feat:
                        recommendations.append(
                            f"⚠️ Reduce after-hours work from {orig_val:.1f}% to improve work-life balance"
                        )
                    elif 'total_emails' in feat:
                        recommendations.append(
                            f"📧 High email volume ({orig_val:.0f}) may indicate overwork - consider delegation"
                        )
                    elif 'betweenness' in feat:
                        recommendations.append(
                            f"🔗 High bridging position ({orig_val:.2f}) creates bottleneck risk - develop redundancy"
                        )
                    elif 'unique_contacts' in feat and mod_val > orig_val:
                        recommendations.append(
                            f"👥 Rapidly expanding network (to {mod_val:.0f} contacts) - ensure appropriate access controls"
                        )
        
        if not recommendations:
            recommendations.append("✓ Current feature values appear within normal ranges")
            recommendations.append("💡 Monitor for sustained changes over time")
        
        return recommendations


def load_simulator_engine(models_dir: Path) -> Optional[RiskSimulatorEngine]:
    """Helper function to load simulator engine with error handling"""
    try:
        engine = RiskSimulatorEngine(models_dir)
        return engine
    except Exception as e:
        print(f"Failed to load simulator engine: {e}")
        import traceback
        traceback.print_exc()
        return None