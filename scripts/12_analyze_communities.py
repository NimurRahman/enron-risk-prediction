"""
SCRIPT 12: COMMUNITY DETECTION ANALYSIS
Analyze existing community assignments from community_map.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("="*80)
print("COMMUNITY DETECTION ANALYSIS")
print("="*80)

# Paths
BASE = Path(r"I:\enron_fresh")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1/6] Loading data...")

community_map = pd.read_csv(DATA_DIR / "community_map.csv")
nodes = pd.read_csv(DATA_DIR / "nodes.csv")
features = pd.read_parquet(DATA_DIR / "node_week_features_enhanced.parquet")

print(f"  ✅ community_map.csv: {len(community_map):,} nodes with communities")
print(f"  ✅ nodes.csv: {len(nodes):,} total nodes")
print(f"  ✅ features.parquet: {len(features):,} rows")

# ============================================================================
# STEP 2: BASIC STATISTICS
# ============================================================================

print("\n[STEP 2/6] Computing community statistics...")

n_communities = community_map['community_id'].nunique()
community_sizes = community_map.groupby('community_id').size()

print(f"\n  📊 COMMUNITY OVERVIEW:")
print(f"     Total communities: {n_communities}")
print(f"     Mean size: {community_sizes.mean():.1f} members")
print(f"     Median size: {community_sizes.median():.0f} members")
print(f"     Largest community: {community_sizes.max()} members")
print(f"     Smallest community: {community_sizes.min()} members")
print(f"     Std dev: {community_sizes.std():.1f}")

# ============================================================================
# STEP 3: MERGE WITH NODE INFO
# ============================================================================

print("\n[STEP 3/6] Merging with node information...")

nodes_with_community = nodes.merge(community_map, on='node_id', how='inner')
print(f"  ✅ Merged: {len(nodes_with_community):,} nodes")

# Extract domain from email
nodes_with_community['domain'] = nodes_with_community['email'].str.split('@').str[1]

# ============================================================================
# STEP 4: COMMUNITY CHARACTERISTICS
# ============================================================================

print("\n[STEP 4/6] Analyzing community characteristics...")

# Get latest features for each node
latest_features = features.groupby('node_id').last().reset_index()

# Merge with community assignments
community_features = nodes_with_community.merge(
    latest_features[['node_id', 'degree', 'betweenness', 'clustering', 
                     'total_emails', 'after_hours_pct']], 
    on='node_id',
    how='left'
)

# Calculate average metrics per community
community_stats = community_features.groupby('community_id').agg({
    'node_id': 'count',
    'degree': 'mean',
    'betweenness': 'mean',
    'clustering': 'mean',
    'total_emails': 'mean',
    'after_hours_pct': 'mean'
}).round(3)

community_stats.columns = ['size', 'avg_degree', 'avg_betweenness', 
                            'avg_clustering', 'avg_total_emails', 'avg_after_hours_pct']

# Sort by size
community_stats = community_stats.sort_values('size', ascending=False)

print(f"\n  📊 TOP 10 LARGEST COMMUNITIES:")
print(community_stats.head(10).to_string())

# Save
community_stats.to_csv(OUTPUT_DIR / "community_characteristics.csv")
print(f"\n  ✅ Saved: community_characteristics.csv")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

print("\n[STEP 5/6] Creating visualizations...")

# 1. Community size distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(community_sizes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Community Size (members)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Community Size Distribution', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Add statistics text
stats_text = f'Total: {n_communities}\nMean: {community_sizes.mean():.1f}\nMedian: {community_sizes.median():.0f}'
axes[0].text(0.97, 0.97, stats_text, transform=axes[0].transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

# 2. Box plot
axes[1].boxplot(community_sizes, vert=True)
axes[1].set_ylabel('Community Size', fontsize=11)
axes[1].set_title('Community Size - Box Plot', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "community_size_distribution.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: community_size_distribution.png")
plt.close()

# 3. Top 20 communities bar chart
fig, ax = plt.subplots(figsize=(12, 6))
top_20 = community_stats.head(20)
ax.barh(range(len(top_20)), top_20['size'], color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels([f"Community {idx}" for idx in top_20.index])
ax.set_xlabel('Number of Members', fontsize=11)
ax.set_ylabel('Community ID', fontsize=11)
ax.set_title('Top 20 Largest Communities', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_20_communities.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: top_20_communities.png")
plt.close()

# 4. Community characteristics heatmap (top 15 communities)
fig, ax = plt.subplots(figsize=(10, 8))
top_15 = community_stats.head(15)
# Normalize for better visualization
top_15_norm = top_15.copy()
for col in ['avg_degree', 'avg_betweenness', 'avg_clustering', 'avg_total_emails', 'avg_after_hours_pct']:
    if top_15_norm[col].std() > 0:
        top_15_norm[col] = (top_15_norm[col] - top_15_norm[col].mean()) / top_15_norm[col].std()

sns.heatmap(
    top_15_norm[['avg_degree', 'avg_betweenness', 'avg_clustering', 
                 'avg_total_emails', 'avg_after_hours_pct']].T,
    annot=top_15[['avg_degree', 'avg_betweenness', 'avg_clustering', 
                  'avg_total_emails', 'avg_after_hours_pct']].T,
    fmt='.2f',
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Normalized Value'},
    xticklabels=[f"Comm {idx}" for idx in top_15.index],
    yticklabels=['Degree', 'Betweenness', 'Clustering', 'Emails', 'After Hours %'],
    ax=ax
)
ax.set_title('Top 15 Communities - Network Characteristics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "community_characteristics_heatmap.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: community_characteristics_heatmap.png")
plt.close()

# ============================================================================
# STEP 6: DOMAIN ANALYSIS
# ============================================================================

print("\n[STEP 6/6] Analyzing community-domain relationships...")

# Find dominant domain per community
community_domains = []
for comm_id in community_map['community_id'].unique():
    comm_nodes = nodes_with_community[nodes_with_community['community_id'] == comm_id]
    if len(comm_nodes) > 0:
        domain_counts = comm_nodes['domain'].value_counts()
        dominant_domain = domain_counts.index[0]
        dominant_count = domain_counts.iloc[0]
        total_count = len(comm_nodes)
        purity = (dominant_count / total_count) * 100
        
        community_domains.append({
            'community_id': comm_id,
            'size': total_count,
            'dominant_domain': dominant_domain,
            'domain_count': dominant_count,
            'purity_pct': purity
        })

community_domains_df = pd.DataFrame(community_domains)
community_domains_df = community_domains_df.sort_values('size', ascending=False)

print(f"\n  📊 TOP 10 COMMUNITIES BY DOMAIN PURITY:")
print(community_domains_df.head(10)[['community_id', 'size', 'dominant_domain', 'purity_pct']].to_string(index=False))

# Save
community_domains_df.to_csv(OUTPUT_DIR / "community_domain_analysis.csv", index=False)
print(f"\n  ✅ Saved: community_domain_analysis.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("CLUSTERING ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"   • {n_communities} communities identified")
print(f"   • {len(community_map):,} nodes with community assignments")
print(f"   • {len(nodes) - len(community_map):,} isolated nodes (no community)")
print(f"   • Average community size: {community_sizes.mean():.1f} members")
print(f"   • Coverage: {len(community_map)/len(nodes)*100:.1f}% of total nodes")

print(f"\n📁 OUTPUT FILES CREATED:")
print(f"   1. community_characteristics.csv")
print(f"   2. community_domain_analysis.csv")
print(f"   3. community_size_distribution.png")
print(f"   4. top_20_communities.png")
print(f"   5. community_characteristics_heatmap.png")

print(f"\n✅ Script 12 complete!")
print("="*80)