"""
Verify what columns actually exist in features.parquet
"""
import pandas as pd

print("="*80)
print("CHECKING FEATURES.PARQUET COLUMNS")
print("="*80)

features = pd.read_parquet('data/node_week_features_enhanced.parquet')

print(f"\nTotal rows: {len(features):,}")
print(f"Unique nodes: {features['node_id'].nunique():,}")
print(f"\nColumns ({len(features.columns)}):")
for i, col in enumerate(features.columns, 1):
    dtype = features[col].dtype
    print(f"  {i:2d}. {col:30s} ({dtype})")

# Check for specific columns we need
required_for_clustering = [
    'node_id', 'week_start',
    'degree', 'betweenness', 'clustering', 
    'total_emails', 'after_hours_pct',
    'degree_ma4', 'betweenness_ma4'
]

print(f"\n{'='*80}")
print("REQUIRED COLUMNS CHECK:")
print("="*80)

missing = []
for col in required_for_clustering:
    if col in features.columns:
        print(f"  ✅ {col}")
    else:
        print(f"  ❌ {col} - MISSING!")
        missing.append(col)

if missing:
    print(f"\n❌ MISSING {len(missing)} REQUIRED COLUMNS!")
    print(f"   Scripts will need modification!")
else:
    print(f"\n✅ ALL REQUIRED COLUMNS PRESENT!")
    
# Show sample data
print(f"\n{'='*80}")
print("SAMPLE DATA (first 3 rows):")
print("="*80)
print(features.head(3).to_string())

# Show latest week per node
print(f"\n{'='*80}")
print("LATEST WEEK EXTRACTION TEST:")
print("="*80)
latest = features.groupby('node_id').last().reset_index()
print(f"  Unique nodes: {len(latest):,}")
print(f"  This is what clustering will use!")