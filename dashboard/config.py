# dashboard/config.py
"""
Configuration, constants, and page setup
Enhanced with comprehensive feature descriptions for simulator
"""
import streamlit as st
from pathlib import Path
import os

# === PATHS ===
APP_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_DATA_DIR = Path(os.getenv("SNA_DATA_DIR", APP_DIR / "data"))
OUTPUTS_DIR = APP_DIR / "outputs"
SHAP_DIR = APP_DIR / "shap_analysis"

# === DEFAULT FILE NAMES ===
DEFAULT_FILENAMES = {
    "nodes": "nodes.csv",
    "idmap": "idmap.csv",
    "community_map": "community_map.csv",
    "edges": "edges_weekly_weighted.parquet",
    "features": "node_week_features_enhanced.parquet",
    "risk": "node_week_risk_enhanced.parquet",
    "preds": "predictions_test_enhanced.csv",
    "feat_imp": "shap_importance_xgboost.csv",
}

# === RISK CATEGORIES ===
RISK_ORDER_2BAND = ["Low/Medium", "High/Critical"]
RISK_ORDER_4BAND = ["Low", "Elevated", "High", "Critical"]

RISK_COLORS = {
    "Low": "#27ae60",
    "Elevated": "#f39c12",
    "High": "#e67e22",
    "Critical": "#c0392b",
    "Low/Medium": "#27ae60",
    "High/Critical": "#c0392b"
}

RISK_THRESHOLDS = {
    "Low": 0.0,
    "Elevated": 0.25,
    "High": 0.50,
    "Critical": 0.75
}

RISK_THRESHOLDS_4BAND = {
    'Low': 0.25,
    'Elevated': 0.50,
    'High': 0.75,
    'Critical': 1.0
}

# === COMPREHENSIVE FEATURE GLOSSARY ===
FEATURE_GLOSSARY = {
    # Network Centrality Metrics
    "degree": """Network Connections: Number of unique people you email with directly. 
    High (>50): Well-connected hub, broad reach across organization. 
    Low (<10): Specialized role or isolated position. 
    Example: degree=45 means you communicate with 45 different colleagues.""",
    
    "degree_ma4": """Network Connections (4-week average): Smoothed count of unique email contacts, removing daily noise to show true connectivity trends. 
    High (>50): Sustained broad organizational reach. 
    Low (<10): Consistently narrow communication network. 
    Example: degree_ma4=42 means averaging 42 contacts over the past month.""",
    
    "in_degree": """Incoming Connections: Number of distinct people who email you. 
    High: Sought-after for advice, leadership, or information. 
    Low: Peripheral position or junior role. 
    Example: in_degree=35 means 35 different people email you.""",
    
    "out_degree": """Outgoing Connections: Number of distinct people you send emails to. 
    High: Proactive communicator, coordinator, or broadcaster. 
    Low: Reactive communication style. 
    Example: out_degree=28 means you email 28 different colleagues.""",
    
    "betweenness": """Information Broker Score: How often you lie on the shortest path between other people - measures control over information flow. 
    High (>0.6): Critical bridge connecting teams/departments, gatekeeper position. 
    Low (<0.2): Not on critical paths, within single tight group. 
    Example: betweenness=0.52 means you're a key intermediary in 52% of communication paths.""",
    
    "betweenness_ma4": """Information Broker Score (4-week average): Smoothed measure of bridging position in network. 
    High (>0.6): Sustained gatekeeper role. 
    Low (<0.2): Consistently within single cluster. 
    Example: betweenness_ma4=0.48 indicates stable bridging position.""",
    
    "closeness": """Network Distance: How quickly you can reach everyone else (fewer hops = higher closeness). 
    High (>0.5): Central position, efficient information access across organization. 
    Low (<0.3): Peripheral, requires many intermediaries to reach others. 
    Example: closeness=0.62 means you're 1-2 hops from most people.""",
    
    "clustering": """Team Cohesion: How much your contacts also email each other (tightly-knit team vs diverse bridging). 
    High (>0.7): Part of cohesive team where everyone communicates. 
    Low (<0.3): Bridge between disconnected groups, cross-functional role. 
    Example: clustering=0.85 means your contacts form a tight-knit group.""",
    
    "clustering_ma4": """Team Cohesion (4-week average): Smoothed measure of how interconnected your contacts are. 
    High (>0.7): Sustained tight team membership. 
    Low (<0.3): Consistently bridging separate groups. 
    Example: clustering_ma4=0.78 shows stable cohesive team environment.""",
    
    "kcore": """Core Membership Level: Which "shell" of network density you belong to (higher = more deeply embedded). 
    High (8-12): Part of organizational core, tightly integrated teams. 
    Low (1-3): Peripheral position, easily disconnected. 
    Example: kcore=9 means you're in the deeply connected inner core.""",
    
    # Communication Volume Metrics
    "total_emails": """Weekly Email Volume: Total emails sent + received in one week. 
    High (>150): Heavy workload or central communication role. 
    Low (<20): Limited engagement or specialized quiet role. 
    Example: total_emails=112 means 112 emails this week.""",
    
    "total_emails_ma4": """Email Volume (4-week average): Smoothed weekly email count showing sustained activity level. 
    High (>140): Consistently high communication load. 
    Low (<25): Persistently low engagement. 
    Example: total_emails_ma4=95 means averaging 95 emails weekly.""",
    
    "out_emails": """Emails Sent: Number of emails you initiated this period. 
    High (>80): Proactive communicator, directive leadership style. 
    Low (<15): Reactive, support, or advisory role. 
    Example: out_emails=65 means you sent 65 emails.""",
    
    "in_emails": """Emails Received: Number of emails sent to you this period. 
    High (>80): Sought for decisions, advice, or approvals. 
    Low (<15): Limited incoming requests. 
    Example: in_emails=78 means you received 78 emails.""",
    
    # Contact Diversity Metrics
    "unique_contacts": """Contact Breadth: Total distinct people you communicated with (sent + received). 
    High (>50): Broad organizational influence and diverse information access. 
    Low (<10): Narrow focus or isolated position. 
    Example: unique_contacts=42 means you emailed with 42 different people.""",
    
    "out_contacts": """Outreach Breadth: Number of distinct people you sent emails to. 
    High (>40): Wide broadcasting or coordination across many people. 
    Low (<8): Focused outreach to small group. 
    Example: out_contacts=35 means you emailed 35 different colleagues.""",
    
    "in_contacts": """Inbound Diversity: Number of distinct people who emailed you. 
    High (>40): Sought by many different people for various reasons. 
    Low (<8): Limited incoming contact diversity. 
    Example: in_contacts=28 means 28 different people contacted you.""",
    
    # Work-Life Balance Metric
    "after_hours_pct": """After-Hours Activity: Percentage of emails sent/received outside 8AM-6PM or on weekends. 
    High (>30%): Poor work-life boundaries, burnout risk, or deliberate concealment. 
    Low (<15%): Healthy boundaries, traditional work hours. 
    Example: after_hours_pct=22% means 22% of activity outside normal hours.""",
    
    "after_hours_pct_ma4": """After-Hours Activity (4-week average): Smoothed percentage showing sustained boundary patterns. 
    High (>30%): Chronic overwork or systematic off-hours activity. 
    Low (<15%): Consistently healthy work hours. 
    Example: after_hours_pct_ma4=26% shows trend toward evening/weekend work.""",
    
    # Change Metrics (Deltas)
    "degree_delta": """Change in Network Size: How your unique contacts increased/decreased from last period. 
    Positive (+15): Expanding network, growing influence. 
    Negative (-15): Contracting network, possible disengagement. 
    Example: degree_delta=+12 means gained 12 new contacts.""",
    
    "betweenness_delta": """Change in Bridging Position: How your gatekeeper role increased/decreased. 
    Positive (+0.15): Becoming more critical bridge. 
    Negative (-0.15): Losing intermediary position. 
    Example: betweenness_delta=+0.18 means significant increase in brokerage.""",
    
    "total_emails_delta": """Change in Email Volume: How your activity increased/decreased from last period. 
    Positive (+40): Workload increase or new responsibilities. 
    Negative (-40): Activity reduction, possible disengagement. 
    Example: total_emails_delta=+35 means 35 more emails than last week.""",
    
    "clustering_delta": """Change in Team Cohesion: How interconnectedness of your contacts changed. 
    Positive (+0.2): Moving toward tighter team. 
    Negative (-0.2): Moving toward bridging role. 
    Example: clustering_delta=-0.15 means shift from team member to coordinator.""",
    
    "after_hours_pct_delta": """Change in After-Hours Work: How off-hours activity increased/decreased. 
    Positive (+10 points): Deteriorating work-life boundaries. 
    Negative (-10 points): Improving boundaries or reduced engagement. 
    Example: after_hours_pct_delta=+8 means 8% more after-hours activity.""",
}

# Shorter descriptions for tables/dropdowns
FEATURE_NAMES_SHORT = {
    "degree": "Network Connections",
    "degree_ma4": "Network Connections (4-wk avg)",
    "in_degree": "Incoming Connections",
    "out_degree": "Outgoing Connections",
    "betweenness": "Information Broker Score",
    "betweenness_ma4": "Broker Score (4-wk avg)",
    "closeness": "Network Distance",
    "clustering": "Team Cohesion",
    "clustering_ma4": "Team Cohesion (4-wk avg)",
    "kcore": "Core Membership",
    "total_emails": "Weekly Email Volume",
    "total_emails_ma4": "Email Volume (4-wk avg)",
    "out_emails": "Emails Sent",
    "in_emails": "Emails Received",
    "unique_contacts": "Contact Breadth",
    "out_contacts": "Outreach Breadth",
    "in_contacts": "Inbound Diversity",
    "after_hours_pct": "After-Hours %",
    "after_hours_pct_ma4": "After-Hours % (4-wk avg)",
    "degree_delta": "Network Change",
    "betweenness_delta": "Broker Change",
    "total_emails_delta": "Volume Change",
    "clustering_delta": "Cohesion Change",
    "after_hours_pct_delta": "Hours Change",
}

# === PAGE CONFIGURATION ===
def setup_page():
    """Configure Streamlit page settings and custom CSS"""
    st.set_page_config(
        page_title="SNA Risk Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { 
            padding: 8px 16px; 
            background-color: #f0f2f6;
            border-radius: 4px;
        }
        .stTabs [aria-selected="true"] { 
            background-color: #1f77b4 !important; 
            color: white !important;
        }
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .risk-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .risk-low { background-color: #d4edda; color: #155724; }
        .risk-elevated { background-color: #fff3cd; color: #856404; }
        .risk-high { background-color: #f8d7da; color: #721c24; }
        .risk-critical { background-color: #f5c6cb; color: #721c24; font-weight: 700; }
        .trend-up { color: #c0392b; font-weight: 700; }
        .trend-down { color: #27ae60; font-weight: 700; }
        .trend-stable { color: #7f8c8d; }
        </style>
    """, unsafe_allow_html=True)

def show_plot(fig, use_container_width: bool = True):
    """Display plotly figure with consistent settings"""
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=use_container_width)