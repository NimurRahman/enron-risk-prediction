"""
Quick fix: Remove 37 phantom node_ids from community_map.csv
"""
import pandas as pd

print("="*80)
print("FIXING COMMUNITY_MAP.CSV")
print("="*80)

# Load files
nodes = pd.read_csv('data/nodes.csv')
community_map = pd.read_csv('data/community_map.csv')

print(f"\nBefore fix:")
print(f"  nodes.csv: {len(nodes):,} node_ids")
print(f"  community_map.csv: {len(community_map):,} rows")

# Find valid node_ids
valid_node_ids = set(nodes['node_id'])
phantom_ids = set(community_map['node_id']) - valid_node_ids

print(f"\n🚨 Found {len(phantom_ids)} phantom node_ids:")
print(f"   {sorted(list(phantom_ids))[:10]}...")

# Filter to only valid node_ids
community_map_fixed = community_map[community_map['node_id'].isin(valid_node_ids)]

print(f"\nAfter fix:")
print(f"  community_map_fixed: {len(community_map_fixed):,} rows")
print(f"  Removed: {len(community_map) - len(community_map_fixed)} rows")

# Save fixed version
community_map_fixed.to_csv('data/community_map.csv', index=False)

print(f"\n✅ Fixed file saved: data/community_map.csv")
print(f"✅ All node_ids now match nodes.csv!")

# Verify
merged = nodes.merge(community_map_fixed, on='node_id', how='inner')
print(f"\n✅ Verification: {len(merged):,} nodes can be merged successfully")

print("\n" + "="*80)
print("FIX COMPLETE!")
print("="*80)