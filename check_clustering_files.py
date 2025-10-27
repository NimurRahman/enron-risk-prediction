# check_clustering_files.py
"""
Verify all files needed for clustering exist
"""
import pandas as pd
from pathlib import Path

BASE = Path(r"I:\enron_fresh")
DATA_DIR = BASE / "data"

print("="*60)
print("CLUSTERING FILES CHECK")
print("="*60)

# Check files
files_to_check = {
    'community_map.csv': 'REQUIRED for Script 12',
    'nodes.csv': 'REQUIRED for both scripts',
    'node_week_features_enhanced.parquet': 'REQUIRED for both scripts',
    'edges_weekly_weighted.parquet': 'OPTIONAL for visualization'
}

all_good = True

for filename, purpose in files_to_check.items():
    filepath = DATA_DIR / filename
    
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"✅ {filename}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Purpose: {purpose}")
        
        # Show row count
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
        
        print()
    else:
        print(f"❌ MISSING: {filename}")
        print(f"   Purpose: {purpose}")
        print()
        all_good = False

if all_good:
    print("\n✅ ALL REQUIRED FILES PRESENT!")
    print("Ready to run clustering scripts.")
else:
    print("\n⚠️ SOME FILES MISSING!")
    print("Check file paths and names.")