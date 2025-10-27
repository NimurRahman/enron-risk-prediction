"""
Investigate where the 37 phantom node_ids came from
"""
import pandas as pd

nodes = pd.read_csv('data/nodes.csv')
community_map = pd.read_csv('data/community_map.csv')
idmap = pd.read_csv('data/idmap.csv')

valid_node_ids = set(nodes['node_id'])
phantom_ids = set(community_map['node_id']) - valid_node_ids

print("="*80)
print("INVESTIGATING PHANTOM NODE_IDS")
print("="*80)

print(f"\nPhantom node_ids: {sorted(list(phantom_ids))}")

# Check if they exist in idmap
if 'node_id' in idmap.columns:
    phantom_in_idmap = [nid for nid in phantom_ids if nid in idmap['node_id'].values]
    print(f"\nFound in idmap.csv: {len(phantom_in_idmap)} / {len(phantom_ids)}")
    
    if phantom_in_idmap:
        print("\nPhantom node details from idmap:")
        print(idmap[idmap['node_id'].isin(phantom_in_idmap)])

# Check edges file
try:
    edges = pd.read_parquet('data/edges_weekly_weighted.parquet')
    phantom_in_edges = []
    for col in ['source', 'target']:
        if col in edges.columns:
            phantom_in_edges.extend([nid for nid in phantom_ids if nid in edges[col].values])
    
    phantom_in_edges = list(set(phantom_in_edges))
    print(f"\nFound in edges.parquet: {len(phantom_in_edges)} / {len(phantom_ids)}")
except:
    print("\nCouldn't check edges.parquet")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("These phantom node_ids likely came from:")
print("  1. Different version of preprocessing")
print("  2. OR nodes that were removed during cleaning")
print("\nRecommendation: Use Option 1 (Quick Fix) to remove them")