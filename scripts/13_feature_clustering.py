"""
SCRIPT 13: BEHAVIORAL CLUSTERING (K-MEANS)
Cluster employees by communication patterns (not network topology)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

print("="*80)
print("BEHAVIORAL CLUSTERING (K-MEANS)")
print("="*80)

# Paths
BASE = Path(r"I:\enron_fresh")
DATA_DIR = BASE / "data"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1/7] Loading data...")

features = pd.read_parquet(DATA_DIR / "node_week_features_enhanced.parquet")
nodes = pd.read_csv(DATA_DIR / "nodes.csv")

print(f"  ✅ features.parquet: {len(features):,} rows")
print(f"  ✅ nodes.csv: {len(nodes):,} nodes")

# ============================================================================
# STEP 2: EXTRACT LATEST WEEK
# ============================================================================

print("\n[STEP 2/7] Extracting latest week per person...")

# Get latest week for each employee
latest_features = features.groupby('node_id').last().reset_index()

print(f"  ✅ Extracted: {len(latest_features):,} employees with latest features")

# ============================================================================
# STEP 3: SELECT FEATURES FOR CLUSTERING
# ============================================================================

print("\n[STEP 3/7] Selecting features for clustering...")

# Features to use (behavioral + network metrics)
cluster_features = [
    'degree',           # Network connections
    'betweenness',      # Information broker score
    'clustering',       # Team cohesion
    'total_emails',     # Email volume
    'after_hours_pct',  # Work-life balance
    'degree_ma4',       # Network trend
    'betweenness_ma4'   # Broker trend
]

print(f"  📊 Using {len(cluster_features)} features:")
for i, feat in enumerate(cluster_features, 1):
    print(f"     {i}. {feat}")

# Prepare data
X = latest_features[cluster_features].fillna(0)

print(f"\n  ✅ Data shape: {X.shape}")
print(f"  ✅ Missing values filled with 0")

# ============================================================================
# STEP 4: STANDARDIZE FEATURES
# ============================================================================

print("\n[STEP 4/7] Standardizing features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✅ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 5: DETERMINE OPTIMAL K (ELBOW METHOD)
# ============================================================================

print("\n[STEP 5/7] Finding optimal number of clusters...")

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"  k={k}: inertia={kmeans.inertia_:.0f}")

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "clustering_elbow_curve.png", dpi=300, bbox_inches='tight')
print(f"\n  ✅ Saved: clustering_elbow_curve.png")
plt.close()

# ============================================================================
# STEP 6: APPLY K-MEANS WITH OPTIMAL K
# ============================================================================

print("\n[STEP 6/7] Applying k-means clustering...")

# Choose optimal k (you can adjust this based on elbow curve)
optimal_k = 5
print(f"  Using k={optimal_k} clusters")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
latest_features['behavior_cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n  📊 CLUSTER SIZES:")
cluster_sizes = latest_features['behavior_cluster'].value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    pct = (size / len(latest_features)) * 100
    print(f"     Cluster {cluster_id}: {size:,} employees ({pct:.1f}%)")

# ============================================================================
# STEP 7: ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================

print("\n[STEP 7/7] Analyzing cluster characteristics...")

# Calculate cluster profiles (mean of each feature)
cluster_profiles = latest_features.groupby('behavior_cluster')[cluster_features].mean()

print(f"\n  📈 CLUSTER PROFILES (average values):")
print(cluster_profiles.round(2).to_string())

# Interpret clusters (based on characteristics)
cluster_names = {
    0: "Unknown",
    1: "Unknown",
    2: "Unknown",
    3: "Unknown",
    4: "Unknown"
}

# Auto-name based on characteristics
for cluster_id in range(optimal_k):
    profile = cluster_profiles.loc[cluster_id]
    
    # Simple naming logic
    if profile['total_emails'] > cluster_profiles['total_emails'].mean() * 1.5:
        cluster_names[cluster_id] = "Heavy Communicators"
    elif profile['betweenness'] > cluster_profiles['betweenness'].mean() * 1.5:
        cluster_names[cluster_id] = "Information Brokers"
    elif profile['clustering'] > cluster_profiles['clustering'].mean() * 1.2:
        cluster_names[cluster_id] = "Team Players"
    elif profile['degree'] > cluster_profiles['degree'].mean() * 1.5:
        cluster_names[cluster_id] = "Network Hubs"
    else:
        cluster_names[cluster_id] = "Peripheral Workers"

latest_features['cluster_name'] = latest_features['behavior_cluster'].map(cluster_names)

print(f"\n  🏷️  CLUSTER NAMES (auto-assigned):")
for cluster_id, name in cluster_names.items():
    size = (latest_features['behavior_cluster'] == cluster_id).sum()
    print(f"     Cluster {cluster_id}: {name} ({size:,} employees)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[Visualization] Creating visualizations...")

# 1. PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=latest_features['behavior_cluster'],
    cmap='tab10',
    alpha=0.6,
    s=30,
    edgecolors='black',
    linewidth=0.5
)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('Employee Behavioral Clusters (PCA Projection)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster', ax=ax)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "behavioral_clusters_pca.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: behavioral_clusters_pca.png")
plt.close()

# 2. Cluster profiles heatmap
fig, ax = plt.subplots(figsize=(12, 8))
# Normalize for heatmap
cluster_profiles_norm = cluster_profiles.copy()
for col in cluster_profiles_norm.columns:
    if cluster_profiles_norm[col].std() > 0:
        cluster_profiles_norm[col] = (cluster_profiles_norm[col] - cluster_profiles_norm[col].mean()) / cluster_profiles_norm[col].std()

sns.heatmap(
    cluster_profiles_norm.T,
    annot=cluster_profiles.T,
    fmt='.2f',
    cmap='RdYlGn',
    cbar_kws={'label': 'Normalized Value'},
    xticklabels=[f"Cluster {i}\n{cluster_names[i]}" for i in range(optimal_k)],
    yticklabels=cluster_features,
    ax=ax
)
ax.set_title('Behavioral Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_profiles_heatmap.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: cluster_profiles_heatmap.png")
plt.close()

# 3. Cluster size bar chart
fig, ax = plt.subplots(figsize=(10, 6))
cluster_counts = latest_features['cluster_name'].value_counts()
ax.bar(range(len(cluster_counts)), cluster_counts.values, color='steelblue', edgecolor='black')
ax.set_xticks(range(len(cluster_counts)))
ax.set_xticklabels(cluster_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Employees', fontsize=12)
ax.set_title('Behavioral Cluster Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cluster_counts.values):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_distribution.png", dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: cluster_distribution.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[Saving] Saving cluster assignments...")

# Save cluster assignments (node_id, cluster_id, cluster_name)
cluster_assignments = latest_features[['node_id', 'behavior_cluster', 'cluster_name']].copy()
cluster_assignments.to_csv(OUTPUT_DIR / "behavioral_clusters.csv", index=False)
print(f"  ✅ Saved: behavioral_clusters.csv")

# Merge with node info (email addresses)
nodes_with_clusters = nodes.merge(cluster_assignments, on='node_id', how='left')
nodes_with_clusters.to_csv(OUTPUT_DIR / "nodes_with_clusters.csv", index=False)
print(f"  ✅ Saved: nodes_with_clusters.csv")

# Save cluster profiles
cluster_profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
print(f"  ✅ Saved: cluster_profiles.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BEHAVIORAL CLUSTERING COMPLETE")
print("="*80)

print(f"\n📊 SUMMARY:")
print(f"   • {optimal_k} behavioral clusters identified")
print(f"   • {len(latest_features):,} employees clustered")
print(f"   • {len(cluster_features)} features used")
print(f"   • PCA variance explained: {sum(pca.explained_variance_ratio_):.1%}")

print(f"\n🏷️  CLUSTER TYPES:")
for cluster_id, name in cluster_names.items():
    size = (latest_features['behavior_cluster'] == cluster_id).sum()
    pct = (size / len(latest_features)) * 100
    print(f"   • {name}: {size:,} employees ({pct:.1f}%)")

print(f"\n📁 OUTPUT FILES CREATED:")
print(f"   1. behavioral_clusters.csv")
print(f"   2. nodes_with_clusters.csv")
print(f"   3. cluster_profiles.csv")
print(f"   4. clustering_elbow_curve.png")
print(f"   5. behavioral_clusters_pca.png")
print(f"   6. cluster_profiles_heatmap.png")
print(f"   7. cluster_distribution.png")

print(f"\n✅ Script 13 complete!")
print("="*80)