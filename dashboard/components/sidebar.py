# dashboard/components/sidebar.py
"""
Sidebar with filters and settings
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.config import DEFAULT_FILENAMES, RISK_ORDER_4BAND

def render_sidebar(data_dir: str):
    """Render sidebar and return filters + filenames"""
    
    with st.sidebar:
        st.header("Data & Settings")
        
        # Data directory input
        data_dir_input = st.text_input("Data folder", value=str(data_dir))
        
        # File name overrides
        with st.expander("File names (override if different)", expanded=False):
            filenames = {}
            for k, v in DEFAULT_FILENAMES.items():
                filenames[k] = st.text_input(k, value=v, key=f"file_{k}")
        
        st.divider()
        
        # === FILTERS ===
        st.subheader("Filters")
        
        # Week range slider (KEEP - WORKING)
        st.markdown("**Date Range**")
        week_lookback = st.slider(
            "Weeks to show",
            min_value=1,
            max_value=132,
            value=4,
            step=1,
            help="Filter data to last N weeks"
        )
        
        # Risk band filter (KEEP - WORKING)
        st.markdown("**Risk Levels**")
        risk_bands = st.multiselect(
            "Select risk bands",
            options=RISK_ORDER_4BAND,
            default=RISK_ORDER_4BAND,
            help="Filter by risk level (applies to Risk tab only)"
        )
        
        # === NEW: CLUSTERING FILTERS (REPLACES group_by) ===
        st.markdown("**Clustering**")
        
        # Try to load clustering data
        clustering_available = False
        selected_clusters = ['All']
        selected_communities = ['All']
        
        try:
            DATA_DIR = Path(data_dir_input) / "data"
            OUTPUTS_DIR = Path(data_dir_input) / "outputs"
            
            # Check if files exist
            community_file = DATA_DIR / "community_map.csv"
            behavioral_file = OUTPUTS_DIR / "behavioral_clusters.csv"
            
            if community_file.exists() and behavioral_file.exists():
                community_map = pd.read_csv(community_file)
                behavioral = pd.read_csv(behavioral_file)
                
                # Behavioral cluster filter
                cluster_options = ['All'] + sorted(behavioral['cluster_name'].dropna().unique().tolist())
                selected_clusters = st.multiselect(
                    "Behavioral Cluster",
                    options=cluster_options,
                    default=['All'],
                    help="Filter by employee behavior type (Information Brokers, Team Players, etc.)"
                )
                
                # Community filter  
                community_ids = sorted(community_map['community_id'].unique())
                community_options = ['All'] + [f"Community {int(c)}" for c in community_ids]
                selected_communities = st.multiselect(
                    "Network Community",
                    options=community_options,
                    default=['All'],
                    help="Filter by communication group (14 natural teams discovered)"
                )
                
                clustering_available = True
                
                # Show info about clustering
                if clustering_available:
                    with st.expander("ℹ️ What are these filters?", expanded=False):
                        st.markdown("""
                        **Behavioral Cluster:**
                        Groups employees by work patterns:
                        - Information Brokers (33 people)
                        - Team Players (86 people)
                        - Active Employees (1,385 people)
                        - Peripheral Workers (19,695 people)
                        
                        **Network Community:**
                        Groups based on who emails whom:
                        - 14 natural teams discovered
                        - Ranges from 1 to 1,644 members
                        """)
            else:
                st.info("ℹ️ Clustering data not found. Run clustering scripts first.")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load clustering data: {str(e)[:50]}...")
        
        # === REMOVED: group_by (was broken) ===
        # This section has been replaced with clustering filters above
        
        st.divider()
        
        # Search filter (NEW - BONUS)
        st.markdown("**Search**")
        email_search = st.text_input(
            "Search by email",
            value="",
            placeholder="e.g., john@enron.com",
            help="Filter to specific email addresses"
        )
        
        st.divider()
        
        # Info section (KEEP)
        with st.expander("About", expanded=False):
            st.markdown("""
            **Enhanced SNA Risk Dashboard**
            
            This dashboard analyzes social network activity 
            and predicts risk levels using machine learning.
            
            **Features:**
            - SHAP-based feature explanations
            - Multi-model ensemble predictions
            - Peer comparison analysis
            - Trend detection
            - Behavioral clustering analysis
            - Community detection
            """)
    
    # Return filters as dict
    # IMPORTANT: Keep backward compatibility!
    return {
        "data_dir": data_dir_input,
        "filenames": filenames,
        "week_lookback": week_lookback,
        "risk_bands": risk_bands,
        "group_by": "individual",  # ← KEEP THIS for backward compatibility (default value)
        # New filters:
        "selected_clusters": selected_clusters,
        "selected_communities": selected_communities,
        "clustering_available": clustering_available,
        "email_search": email_search
    }