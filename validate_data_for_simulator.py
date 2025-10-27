"""
Data Validation Script for Simulator Fix
Run this BEFORE deploying the fixed simulator to verify your data has all required features

Usage:
    python validate_data_for_simulator.py

This will check if your parquet files have all 24 required features.
"""
import pandas as pd
from pathlib import Path
import sys

# The EXACT 24 features the XGBoost model was trained with, in order
MODEL_FEATURES_ORDERED = [
    'degree_ma4', 'total_emails_ma4', 'degree', 'unique_contacts', 
    'betweenness', 'total_emails_delta', 'degree_delta', 'betweenness_ma4',
    'total_emails', 'betweenness_delta', 'kcore', 'out_degree',
    'clustering', 'after_hours_pct_delta', 'out_emails', 'after_hours_pct_ma4',
    'after_hours_pct', 'in_degree', 'in_contacts', 'out_contacts',
    'clustering_delta', 'clustering_ma4', 'in_emails', 'closeness'
]

def validate_parquet_files():
    """Check if parquet files have all required features"""
    
    print("=" * 70)
    print("ENRON SIMULATOR - DATA VALIDATION")
    print("=" * 70)
    print()
    
    # Check if we're in the right directory
    if not Path('data').exists():
        print("❌ ERROR: 'data' folder not found!")
        print("   Please run this script from the project root: I:\\enron_modeling\\")
        return False
    
    # Files to check
    files_to_check = {
        'features': 'data/node_week_features_enhanced.parquet',
        'risk': 'data/node_week_risk_enhanced.parquet',
        'edges': 'data/edges_weekly_weighted.parquet',
    }
    
    all_good = True
    
    for file_type, filepath in files_to_check.items():
        print(f"Checking {file_type}: {filepath}")
        print("-" * 70)
        
        file_path = Path(filepath)
        
        if not file_path.exists():
            print(f"   ❌ FILE NOT FOUND: {filepath}")
            print(f"   → This file is required for the simulator to work")
            all_good = False
            print()
            continue
        
        try:
            # Load the file
            df = pd.read_parquet(file_path)
            print(f"   ✅ File loaded successfully")
            print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Check for required features (only for features file)
            if file_type == 'features':
                print(f"\n   Checking for 24 required model features:")
                
                missing_features = []
                present_features = []
                
                for feat in MODEL_FEATURES_ORDERED:
                    if feat in df.columns:
                        present_features.append(feat)
                    else:
                        missing_features.append(feat)
                
                print(f"   ✅ Present: {len(present_features)}/24 features")
                
                if missing_features:
                    print(f"   ❌ MISSING: {len(missing_features)} features")
                    print(f"\n   Missing features:")
                    for feat in missing_features:
                        print(f"      - {feat}")
                    all_good = False
                else:
                    print(f"   ✅ ALL 24 FEATURES PRESENT! Data is ready!")
            
            print()
            
        except Exception as e:
            print(f"   ❌ ERROR loading file: {e}")
            all_good = False
            print()
    
    print("=" * 70)
    
    if all_good:
        print("✅ VALIDATION PASSED!")
        print()
        print("Your data has all required features. You can safely deploy the fixed simulator.")
        print()
        print("Next steps:")
        print("1. Backup your current tab_simulator.py")
        print("2. Replace it with tab_simulator_FIXED.py")
        print("3. Restart your Streamlit dashboard")
        print("4. Test the reset and predict buttons")
        return True
    else:
        print("❌ VALIDATION FAILED!")
        print()
        print("Your data is missing required features or files.")
        print()
        print("Solutions:")
        print("1. If features are missing: Re-run script 02_prepare_modeling_data.py")
        print("2. If files are missing: Check your data folder structure")
        print("3. Contact the data team for the correct parquet files")
        return False

def check_model_files():
    """Check if model files exist"""
    print()
    print("=" * 70)
    print("CHECKING MODEL FILES")
    print("=" * 70)
    print()
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("❌ ERROR: 'models' folder not found!")
        print("   The simulator needs trained model files to make predictions.")
        return False
    
    required_models = {
        'XGBoost': 'model_xgboost.pkl',
        'Random Forest': 'model_rf.pkl',
        'Baseline': 'model_baseline_logreg.pkl'
    }
    
    all_models_present = True
    
    for model_name, filename in required_models.items():
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✅ {model_name}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ MISSING: {model_name} ({filename})")
            all_models_present = False
    
    print()
    
    if all_models_present:
        print("✅ All 3 model files present!")
    else:
        print("❌ Some model files are missing!")
        print("   Run the training scripts to generate missing models:")
        print("   - 07_train_xgboost.py")
        print("   - 06_train_random_forest.py")
        print("   - 03_baseline_model.py")
    
    return all_models_present

def check_predictions_file():
    """Check if enhanced predictions file exists and has required columns"""
    print()
    print("=" * 70)
    print("CHECKING PREDICTIONS FILE")
    print("=" * 70)
    print()
    
    pred_file = Path('outputs/predictions_test_enhanced.csv')
    
    if not pred_file.exists():
        print("❌ ERROR: predictions_test_enhanced.csv not found!")
        print("   This file is required for the dashboard to work.")
        print("   Run: 11_generate_predictions_enhanced.py")
        return False
    
    try:
        # Load just the first few rows to check columns
        df = pd.read_csv(pred_file, nrows=5)
        
        size_mb = pred_file.stat().st_size / (1024 * 1024)
        print(f"✅ File found: {size_mb:.1f} MB")
        print(f"📊 Columns: {len(df.columns)}")
        
        # Check for critical columns
        critical_cols = ['node_id', 'week_start', 'risk_score', 'y_pred']
        missing_critical = [col for col in critical_cols if col not in df.columns]
        
        if missing_critical:
            print(f"❌ Missing critical columns: {missing_critical}")
            return False
        else:
            print(f"✅ All critical columns present")
            return True
        
    except Exception as e:
        print(f"❌ ERROR reading predictions file: {e}")
        return False

if __name__ == "__main__":
    print()
    print("🔍 Starting validation...")
    print()
    
    # Run all validations
    data_ok = validate_parquet_files()
    models_ok = check_model_files()
    preds_ok = check_predictions_file()
    
    # Final summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"Data files: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Model files: {'✅ PASS' if models_ok else '❌ FAIL'}")
    print(f"Predictions file: {'✅ PASS' if preds_ok else '❌ FAIL'}")
    print()
    
    if data_ok and models_ok and preds_ok:
        print("🎉 ALL CHECKS PASSED! Your simulator is ready to be fixed!")
        print()
        print("📋 Deployment Checklist:")
        print("   1. ✅ Data has all 24 features")
        print("   2. ✅ All 3 model files present")
        print("   3. ✅ Predictions file exists")
        print("   4. ⏳ Replace tab_simulator.py with fixed version")
        print("   5. ⏳ Restart Streamlit dashboard")
        print("   6. ⏳ Test reset and predict buttons")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Fix the issues above before deploying")
        sys.exit(1)
