"""
DATA INTEGRITY VALIDATION
Verify node_id consistency across all clustering files
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("DATA INTEGRITY VALIDATION - NODE_ID CONSISTENCY CHECK")
print("="*80)

BASE = Path(r"I:\enron_fresh")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"

# ============================================================================
# STEP 1: LOAD ALL FILES
# ============================================================================

print("\n[STEP 1/5] Loading all files...")

files_loaded = {}

try:
    # Core files
    nodes = pd.read_csv(DATA_DIR / "nodes.csv")
    files_loaded['nodes.csv'] = nodes
    print(f"  ✅ nodes.csv: {len(nodes):,} rows")
    
    community_map = pd.read_csv(DATA_DIR / "community_map.csv")
    files_loaded['community_map.csv'] = community_map
    print(f"  ✅ community_map.csv: {len(community_map):,} rows")
    
    features = pd.read_parquet(DATA_DIR / "node_week_features_enhanced.parquet")
    files_loaded['features.parquet'] = features
    print(f"  ✅ features.parquet: {len(features):,} rows")
    
    # Optional files
    try:
        idmap = pd.read_csv(DATA_DIR / "idmap.csv")
        files_loaded['idmap.csv'] = idmap
        print(f"  ✅ idmap.csv: {len(idmap):,} rows")
    except:
        print(f"  ⚠️  idmap.csv: Not found (optional)")
    
    try:
        preds = pd.read_csv(OUTPUT_DIR / "predictions_test_enhanced.csv")
        files_loaded['predictions.csv'] = preds
        print(f"  ✅ predictions.csv: {len(preds):,} rows")
    except:
        print(f"  ⚠️  predictions.csv: Not found (optional)")
        
except Exception as e:
    print(f"  ❌ ERROR loading files: {e}")
    exit(1)

# ============================================================================
# STEP 2: EXTRACT UNIQUE NODE_IDS FROM EACH FILE
# ============================================================================

print("\n[STEP 2/5] Extracting unique node_ids from each file...")

node_id_sets = {}

# From nodes.csv
if 'nodes.csv' in files_loaded:
    node_id_sets['nodes.csv'] = set(files_loaded['nodes.csv']['node_id'].unique())
    print(f"  nodes.csv: {len(node_id_sets['nodes.csv']):,} unique node_ids")

# From community_map.csv
if 'community_map.csv' in files_loaded:
    node_id_sets['community_map.csv'] = set(files_loaded['community_map.csv']['node_id'].unique())
    print(f"  community_map.csv: {len(node_id_sets['community_map.csv']):,} unique node_ids")

# From features.parquet
if 'features.parquet' in files_loaded:
    node_id_sets['features.parquet'] = set(files_loaded['features.parquet']['node_id'].unique())
    print(f"  features.parquet: {len(node_id_sets['features.parquet']):,} unique node_ids")

# From idmap.csv
if 'idmap.csv' in files_loaded:
    node_id_sets['idmap.csv'] = set(files_loaded['idmap.csv']['node_id'].unique())
    print(f"  idmap.csv: {len(node_id_sets['idmap.csv']):,} unique node_ids")

# From predictions.csv
if 'predictions.csv' in files_loaded:
    node_id_sets['predictions.csv'] = set(files_loaded['predictions.csv']['node_id'].unique())
    print(f"  predictions.csv: {len(node_id_sets['predictions.csv']):,} unique node_ids")

# ============================================================================
# STEP 3: CHECK FOR MISMATCHES
# ============================================================================

print("\n[STEP 3/5] Checking for mismatches...")

# Define master set (nodes.csv should be the master)
master_file = 'nodes.csv'
master_set = node_id_sets[master_file]

print(f"\n  Using {master_file} as MASTER ({len(master_set):,} node_ids)")
print("  " + "="*70)

all_consistent = True

for filename, node_set in node_id_sets.items():
    if filename == master_file:
        continue
    
    # Check if all node_ids in this file exist in master
    missing_in_master = node_set - master_set
    extra_in_master = master_set - node_set
    
    print(f"\n  {filename}:")
    
    if len(missing_in_master) == 0 and len(extra_in_master) == 0:
        print(f"    ✅ PERFECT MATCH - All node_ids align with master")
    else:
        all_consistent = False
        
        if len(missing_in_master) > 0:
            print(f"    ❌ {len(missing_in_master):,} node_ids in {filename} NOT in master!")
            print(f"       Examples: {list(missing_in_master)[:5]}")
        
        if len(extra_in_master) > 0:
            print(f"    ⚠️  {len(extra_in_master):,} node_ids in master NOT in {filename}")
            print(f"       (This may be OK if file is subset)")
            print(f"       Examples: {list(extra_in_master)[:5]}")
        
        # Calculate overlap percentage
        overlap = len(node_set & master_set)
        overlap_pct = (overlap / len(node_set)) * 100 if len(node_set) > 0 else 0
        print(f"    📊 Overlap: {overlap:,} / {len(node_set):,} ({overlap_pct:.1f}%)")

# ============================================================================
# STEP 4: DETAILED MISMATCH ANALYSIS
# ============================================================================

print("\n[STEP 4/5] Detailed mismatch analysis...")

# Check for duplicates
print("\n  Checking for duplicate node_ids:")
for filename, df in files_loaded.items():
    if 'node_id' in df.columns:
        duplicates = df['node_id'].duplicated().sum()
        if duplicates > 0:
            print(f"    ❌ {filename}: {duplicates:,} duplicate node_ids found!")
            all_consistent = False
        else:
            print(f"    ✅ {filename}: No duplicates")

# Check for null node_ids
print("\n  Checking for null node_ids:")
for filename, df in files_loaded.items():
    if 'node_id' in df.columns:
        nulls = df['node_id'].isnull().sum()
        if nulls > 0:
            print(f"    ❌ {filename}: {nulls:,} null node_ids found!")
            all_consistent = False
        else:
            print(f"    ✅ {filename}: No nulls")

# ============================================================================
# STEP 5: CROSS-FILE MERGE TEST
# ============================================================================

print("\n[STEP 5/5] Testing merge compatibility...")

merge_results = []

# Test merge: nodes + community_map
if 'nodes.csv' in files_loaded and 'community_map.csv' in files_loaded:
    merged = nodes.merge(community_map, on='node_id', how='inner')
    expected = len(community_map)
    actual = len(merged)
    
    result = {
        'merge': 'nodes + community_map',
        'expected': expected,
        'actual': actual,
        'success': actual == expected
    }
    merge_results.append(result)
    
    if result['success']:
        print(f"  ✅ nodes + community_map: {actual:,} rows (PERFECT)")
    else:
        print(f"  ❌ nodes + community_map: Expected {expected:,}, got {actual:,}")
        all_consistent = False

# Test merge: nodes + features (get unique node_ids first)
if 'nodes.csv' in files_loaded and 'features.parquet' in files_loaded:
    features_unique = features[['node_id']].drop_duplicates()
    merged = nodes.merge(features_unique, on='node_id', how='inner')
    expected = len(features_unique)
    actual = len(merged)
    
    result = {
        'merge': 'nodes + features',
        'expected': expected,
        'actual': actual,
        'success': actual == expected
    }
    merge_results.append(result)
    
    if result['success']:
        print(f"  ✅ nodes + features: {actual:,} unique nodes (PERFECT)")
    else:
        print(f"  ❌ nodes + features: Expected {expected:,}, got {actual:,}")
        all_consistent = False

# Test merge: community_map + features
if 'community_map.csv' in files_loaded and 'features.parquet' in files_loaded:
    features_unique = features[['node_id']].drop_duplicates()
    merged = community_map.merge(features_unique, on='node_id', how='inner')
    expected_min = min(len(community_map), len(features_unique))
    actual = len(merged)
    
    result = {
        'merge': 'community_map + features',
        'expected_min': expected_min,
        'actual': actual,
        'success': actual >= expected_min * 0.95  # Allow 5% tolerance
    }
    merge_results.append(result)
    
    if result['success']:
        print(f"  ✅ community_map + features: {actual:,} rows (OK)")
    else:
        print(f"  ❌ community_map + features: Expected ~{expected_min:,}, got {actual:,}")
        all_consistent = False

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("FINAL VALIDATION REPORT")
print("="*80)

if all_consistent:
    print("\n✅ ALL CHECKS PASSED!")
    print("   Data integrity verified - safe to proceed with clustering.")
    print("\n   Summary:")
    print(f"   • All node_ids consistent across files")
    print(f"   • No duplicates or nulls found")
    print(f"   • All merges successful")
    print("\n   👍 You can safely run clustering scripts!")
    
else:
    print("\n❌ DATA INTEGRITY ISSUES FOUND!")
    print("   DO NOT proceed with clustering until issues are resolved.")
    print("\n   Issues to fix:")
    print("   • Mismatched node_ids between files")
    print("   • OR duplicates found")
    print("   • OR merge failures")
    print("\n   ⚠️  Contact your data source or review preprocessing steps!")

# Save detailed report
report = {
    'all_consistent': all_consistent,
    'node_id_counts': {k: len(v) for k, v in node_id_sets.items()},
    'merge_results': merge_results
}

import json
with open(OUTPUT_DIR / 'data_integrity_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Detailed report saved: {OUTPUT_DIR / 'data_integrity_report.json'}")

# ============================================================================
# BONUS: SHOW INTERSECTION OF ALL FILES
# ============================================================================

print("\n" + "="*80)
print("BONUS: COMPLETE OVERLAP ANALYSIS")
print("="*80)

if len(node_id_sets) > 1:
    # Find intersection (node_ids present in ALL files)
    all_files_intersection = set.intersection(*node_id_sets.values())
    
    print(f"\n  Node_ids present in ALL files: {len(all_files_intersection):,}")
    
    # Find union (node_ids present in ANY file)
    all_files_union = set.union(*node_id_sets.values())
    
    print(f"  Node_ids present in ANY file: {len(all_files_union):,}")
    
    coverage_pct = (len(all_files_intersection) / len(all_files_union)) * 100
    print(f"  Coverage: {coverage_pct:.1f}%")
    
    if coverage_pct < 90:
        print(f"\n  ⚠️  WARNING: Only {coverage_pct:.1f}% coverage!")
        print(f"     Consider investigating why so many node_ids don't appear in all files.")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)