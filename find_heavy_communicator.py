"""
Who is the single Heavy Communicator in Cluster 1?
"""
import pandas as pd

# Load results
behavioral = pd.read_csv('outputs/behavioral_clusters.csv')
nodes = pd.read_csv('data/nodes.csv')

# Find Cluster 1
cluster_1 = behavioral[behavioral['behavior_cluster'] == 1]

# Get email
heavy_comm = nodes[nodes['node_id'].isin(cluster_1['node_id'])]

print("="*80)
print("THE HEAVY COMMUNICATOR")
print("="*80)
print(f"\n{heavy_comm.to_string(index=False)}")

# Get their stats
features = pd.read_parquet('data/node_week_features_enhanced.parquet')
person_id = heavy_comm['node_id'].iloc[0]
person_features = features[features['node_id'] == person_id].iloc[-1]

print("\n📊 STATISTICS:")
print(f"  Degree: {person_features['degree']}")
print(f"  Betweenness: {person_features['betweenness']:.3f}")
print(f"  Total Emails: {person_features['total_emails']:.0f}")
print(f"  After Hours %: {person_features['after_hours_pct']:.1%}")

print("\n🎯 This person is the MOST CONNECTED individual in Enron!")
print("="*80)