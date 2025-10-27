# dashboard/tabs/tab_clustering.py
"""
Clustering Analysis Tab - Shows behavioral clusters and communities
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def render(data: dict, filters: dict):
    """Render clustering analysis tab"""
    
    st.subheader("Clustering Analysis")
    st.caption("Organizational structure discovered through network analysis and behavioral patterns")
    
    # Load clustering data
    try:
        outputs_dir = data.get('_outputs_dir', Path('outputs'))
        data_dir = data.get('_data_dir', Path('data'))
        
        behavioral = pd.read_csv(outputs_dir / 'behavioral_clusters.csv')
        community_map = pd.read_csv(data_dir / 'community_map.csv')
        
    except Exception as e:
        st.error(f"Could not load clustering data: {e}")
        st.info("Make sure behavioral_clusters.csv and community_map.csv exist")
        return
    
    # === STATISTICS GRID ===
    col1, col2 = st.columns(2)
    
    # BEHAVIORAL CLUSTERS
    with col1:
        st.markdown("### Behavioral Clusters")
        st.caption("Groups based on work patterns (k-means)")
        
        cluster_counts = behavioral['cluster_name'].value_counts().sort_values(ascending=False)
        
        st.metric("Total Clusters", len(cluster_counts))
        
        st.markdown("**Cluster Breakdown:**")
        for cluster, count in cluster_counts.items():
            pct = count / len(behavioral) * 100
            st.write(f"- **{cluster}**: {count:,} people ({pct:.1f}%)")
        
        # Pie chart
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Behavioral Cluster Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # NETWORK COMMUNITIES
    with col2:
        st.markdown("### Network Communities")
        st.caption("Groups based on communication (Louvain)")
        
        community_counts = community_map['community_id'].value_counts().sort_index()
        
        st.metric("Total Communities", len(community_counts))
        
        st.markdown("**Top 5 Largest:**")
        for comm_id, count in community_counts.head(5).items():
            pct = count / len(community_map) * 100
            st.write(f"- **Community {int(comm_id)}**: {count:,} ({pct:.1f}%)")
        
        # Bar chart
        fig = px.bar(
            x=community_counts.index[:14],
            y=community_counts.values[:14],
            title="Community Size Distribution",
            labels={'x': 'Community ID', 'y': 'Members'},
            color=community_counts.values[:14],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === KEY FINDINGS ===
    st.markdown("---")
    st.markdown("### Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        info_brokers = behavioral[behavioral['cluster_name'] == 'Information Brokers']
        broker_pct = len(info_brokers) / len(behavioral) * 100
        st.metric(
            "Information Brokers",
            f"{len(info_brokers)} people",
            delta=f"{broker_pct:.2f}%",
            help="Critical communication hubs"
        )
    
    with col2:
        peripheral = behavioral[behavioral['cluster_name'] == 'Peripheral Workers']
        pct = len(peripheral) / len(behavioral) * 100
        st.metric(
            "Peripheral Workers",
            f"{len(peripheral):,}",
            delta=f"{pct:.0f}%",
            help="Low activity employees"
        )
    
    with col3:
        st.metric(
            "Network Coverage",
            f"{len(community_map):,}",
            delta="29% of total",
            help="Active communicators"
        )
    
    # === INSIGHT BOX ===
    st.info("""
    **Critical Discovery:** Only 33 people (0.2% of workforce) serve as Information Brokers 
    connecting the entire organization. These individuals are critical single points of failure.
    """)
    
    # === DETAILED TABLES ===
    st.markdown("---")
    st.markdown("### Detailed Statistics")
    
    tab1, tab2, tab3 = st.tabs(["Behavioral Clusters", "Network Communities", "Cross-Analysis"])
    
    with tab1:
        cluster_stats = behavioral.groupby('cluster_name').size().reset_index(name='Count')
        cluster_stats['Percentage'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(1)
        cluster_stats = cluster_stats.sort_values('Count', ascending=False)
        st.dataframe(cluster_stats, use_container_width=True, hide_index=True)
    
    with tab2:
        comm_stats = community_map.groupby('community_id').size().reset_index(name='Count')
        comm_stats['Percentage'] = (comm_stats['Count'] / comm_stats['Count'].sum() * 100).round(1)
        comm_stats = comm_stats.sort_values('Count', ascending=False).head(20)
        st.caption("Showing top 20 communities by size")
        st.dataframe(comm_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        # Cross-tabulation of clusters and communities
        try:
            merged = behavioral.merge(community_map, on='node_id', how='inner')
            cross_tab = pd.crosstab(
                merged['cluster_name'], 
                merged['community_id'],
                margins=True
            )
            st.markdown("**Behavioral Clusters vs Network Communities**")
            st.caption("Shows how behavioral patterns map to network structure")
            st.dataframe(cross_tab, use_container_width=True)
        except Exception as e:
            st.warning("Cross-analysis requires matching node IDs in both datasets")