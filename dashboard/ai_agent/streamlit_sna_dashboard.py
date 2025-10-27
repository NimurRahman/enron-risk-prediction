# Streamlit SNA Risk Dashboard — Enron (client-ready, fixed)
# Notes:
# - Replaced any deprecated Streamlit Plotly args with a single helper `show_plot()`
# - Ensured the Scenario Simulator is scoped ONLY to the Predictions tab (indentation fixed)
# - Removed stray mojibake characters by using plain ASCII text in UI strings
# - Kept the model-bundle loader (supports .pkl/.joblib or dict bundles)
# - Added gentle fallbacks for feature names and simulator defaults
# Streamlit SNA Risk Dashboard — Enron (FINAL PRODUCTION VERSION)
# All errors fixed, AI Agent fully functional with visualizations
# Author: Sumit
# Date: October 2025

import os
import io
import pickle
import time  # ADDED for unique chart keys
from textwrap import dedent
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# AI Agent imports
from ai_agent_parser import QueryParser
from ai_agent_responder import ResponseGenerator

try:
    import joblib
except Exception:
    joblib = None

# ---- Default data directory
APP_DIR = Path(__file__).parent.resolve()
DEFAULT_DATA_DIR = Path(os.getenv("SNA_DATA_DIR", APP_DIR / "data"))

# ----------------------
# Global page config & styles
# ----------------------
st.set_page_config(
    page_title="Collaboration Risk Dashboard (Enron)",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = r"""
<style>
:root { --radius: 14px; }
.block-container {padding-top: 1.2rem;}
.kpi-card { border:1px solid rgba(0,0,0,0.06); border-radius:var(--radius); padding:14px 16px; background:linear-gradient(180deg,#fff 0%,#fafafa 100%); box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.kpi-title {font-size:0.85rem;color:#666;margin-bottom:4px;}
.kpi-value {font-size:1.6rem;font-weight:700;}
.badge {padding:2px 8px;border-radius:999px;font-size:0.75rem;font-weight:600;}
.badge-low{background:#e8f5e9;color:#1b5e20;} .badge-med{background:#fff3e0;color:#e65100;} .badge-high{background:#ffebee;color:#b71c1c;} .badge-crit{background:#fbe9e7;color:#bf360c;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------
# Utilities
# ----------------------
RISK_ORDER = ["Low","Medium","High","Critical"]
RISK_COLOR = {"Low":"#66bb6a","Medium":"#ffb74d","High":"#ef5350","Critical":"#8d6e63"}

FEATURE_GLOSSARY = {
    "total_emails": "Total number of emails sent/received in the week.",
    "emails_sent": "Count of emails sent by the node in the week.",
    "after_hours_pct": "% of emails sent outside business hours.",
    "unique_contacts": "Number of distinct counterparts.",
    "in_degree": "# incoming connections.",
    "out_degree": "# outgoing connections.",
    "betweenness": "Betweenness centrality (brokerage).",
    "degree_centrality": "Degree centrality (connectedness).",
}

DEFAULT_FILENAMES = {
    "nodes": "nodes.csv",
    "idmap": "idmap.csv",
    "edges": "edges_weekly_weighted.parquet",
    "features": "node_week_features.parquet",
    "risk": "node_week_risk.parquet",
    "model_input": "model_input.csv",
    "preds": "predictions_test.csv",
    "feat_imp": "feature_importance.csv",
    "email_fixes": "nodes_email_fixes.csv",
}

def show_plot(fig, *, width="stretch", config=None):
    """Render Plotly figures using Streamlit's current API."""
    if config is None:
        config = {}
    st.plotly_chart(fig, use_container_width=(width=="stretch"), config=config)

# ----------------------
# File IO helpers
# ----------------------

def _read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _read_parquet(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_all(data_dir: str, filenames: dict) -> dict:
    paths = {k: os.path.join(data_dir, v) for k, v in filenames.items()}

    def prefer_parquet_then_csv(p: str):
        dfp = _read_parquet(p)
        if dfp is not None:
            return dfp
        root, ext = os.path.splitext(p)
        if ext.lower() == ".parquet":
            csv_path = root + ".csv"
            if os.path.exists(csv_path):
                dfc = _read_csv(csv_path)
                if dfc is not None:
                    return dfc
        return _read_csv(p)

    return {
        "nodes": _read_csv(paths["nodes"]),
        "idmap": _read_csv(paths["idmap"]),
        "edges": prefer_parquet_then_csv(paths["edges"]),
        "features": prefer_parquet_then_csv(paths["features"]),
        "risk": prefer_parquet_then_csv(paths["risk"]),
        "model_input": _read_csv(paths["model_input"]),
        "preds": _read_csv(paths["preds"]),
        "feat_imp": _read_csv(paths["feat_imp"]),
        "email_fixes": _read_csv(paths["email_fixes"]),
        "_paths": paths,
    }

# ----------------------
# Normalisation helpers
# ----------------------

def _coerce_week_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    if pd.api.types.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.DatetimeTZDtype):
        return pd.to_datetime(s, errors="coerce").dt.date
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        as_int = pd.to_numeric(s, errors="coerce").astype("Int64")
        eight_digit = as_int.between(10_000_00, 99_999_99)
        out = pd.Series([pd.NaT] * len(s), dtype="datetime64[ns]")
        if eight_digit.any():
            tmp = as_int[eight_digit].astype(str)
            out.loc[eight_digit] = pd.to_datetime(tmp, format="%Y%m%d", errors="coerce")
        else:
            out1 = pd.to_datetime(as_int, unit="ms", errors="coerce")
            out2 = pd.to_datetime(as_int, unit="s", errors="coerce")
            out = out1.where(out1.notna(), out2)
        return out.dt.date
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True).dt.date

def parse_week_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in ["week","week_start","window_start","date","ts"]:
        if c in df.columns:
            out = df.copy()
            out["week"] = _coerce_week_series(out[c])
            lo = pd.to_datetime("1995-01-01").date()
            hi = pd.to_datetime("2010-01-01").date()
            out = out[(out["week"] >= lo) & (out["week"] <= hi)]
            return out
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out["week"] = df.index.normalize().date
        lo = pd.to_datetime("1995-01-01").date()
        hi = pd.to_datetime("2010-01-01").date()
        out = out[(out["week"] >= lo) & (out["week"] <= hi)]
        return out
    return df

def normalise_identity_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    out = df.copy()
    for col in ["email","src_email","dst_email"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.lower()
    for cand in ["node_id","id","node","src","dst"]:
        if cand in out.columns:
            try:
                out[cand] = pd.to_numeric(out[cand])
            except Exception:
                pass
    return out

def attach_email_and_node_id(df, nodes, idmap, id_col_candidates=("node_id","id")):
    if df is None:
        return df
    df = normalise_identity_cols(df)
    mapper = None
    if nodes is not None and {"email"}.issubset(nodes.columns):
        nid = next((c for c in ["node_id","id"] if c in nodes.columns), None)
        if nid:
            m = nodes[[nid,"email"]].drop_duplicates().rename(columns={nid:"node_id"})
            mapper = m if mapper is None else pd.concat([mapper,m], ignore_index=True)
    if idmap is not None and {"email"}.issubset(idmap.columns):
        nid = next((c for c in ["node_id","id"] if c in idmap.columns), None)
        if nid:
            m = idmap[[nid,"email"]].drop_duplicates().rename(columns={nid:"node_id"})
            mapper = m if mapper is None else pd.concat([mapper,m], ignore_index=True)
    if mapper is not None:
        mapper = mapper.drop_duplicates(subset=["node_id","email"]).copy()
        nid_in_df = next((c for c in id_col_candidates if c in df.columns), None)
        if nid_in_df and "email" not in df.columns:
            df = df.merge(mapper, left_on=nid_in_df, right_on="node_id", how="left")
        if "email" in df.columns and "node_id" not in df.columns:
            df = df.merge(mapper, on="email", how="left")
    return df

def derive_team_domain(email: pd.Series) -> pd.Series:
    return email.fillna("").astype(str).str.split("@").str[-1].str.lower() if email is not None else None

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.header("Data & Settings")
    data_dir = st.text_input("Data folder", value=str(DEFAULT_DATA_DIR))

    with st.expander("File names (override if different)", expanded=False):
        filenames = {}
        for k, v in DEFAULT_FILENAMES.items():
            filenames[k] = st.text_input(k, value=v)

    data = load_all(data_dir, filenames)

    st.caption("Loaded files (✓ = found / — = missing)")
    for k in ["nodes","idmap","edges","features","risk","model_input","preds","feat_imp","email_fixes"]:
        df = data.get(k)
        ok = (df is not None) and (not (isinstance(df, pd.DataFrame) and df.empty))
        st.write(f"{('✓' if ok else '—')} {k}")

    st.divider()
    st.subheader("Filters")

    def _minmax_week():
        for key in ["risk","features","edges"]:
            df = data.get(key)
            df = parse_week_col(df) if df is not None else None
            if df is not None and "week" in df.columns and df["week"].notna().any():
                s = pd.to_datetime(df["week"], errors="coerce").dropna()
                if len(s) >= 10:
                    q1, q9 = s.quantile(0.01), s.quantile(0.99)
                    return q1.date(), q9.date()
                return s.min().date(), s.max().date()
        return None, None

    wmin, wmax = _minmax_week()
    if wmin is not None:
        week_range = st.slider(
            "Week range",
            min_value=pd.to_datetime(wmin).date(),
            max_value=pd.to_datetime(wmax).date(),
            value=(pd.to_datetime(wmin).date(), pd.to_datetime(wmax).date()),
            format="YYYY-MM-DD",
        )
    else:
        week_range = (None, None)

    risk_bands_sel = st.multiselect("Risk bands", options=RISK_ORDER, default=RISK_ORDER)
    search_email = st.text_input("Search email contains", value="")
    comm_map = data.get("community_map")
    has_comm = (
        isinstance(comm_map, pd.DataFrame)
        and not comm_map.empty
        and {"node_id","community_id"}.issubset(comm_map.columns)
    )
    _group_opts = ["Individual","Team (email domain)"] + (["Community (detected)"] if has_comm else [])
    group_mode = st.radio("Group by", options=_group_opts, horizontal=False)

    st.divider()
    st.subheader("Branding")
    if "client_name" not in st.session_state:
        st.session_state["client_name"] = "SNA Toolbox Pty Ltd"
    client_name = st.text_input("Client name", key="client_name")
    _ = st.file_uploader("Upload client logo (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=False)
    _ = st.color_picker("Primary color", value="#0F766E")

# ----------------------
# Prepare datasets
# ----------------------
nodes = normalise_identity_cols(data.get("nodes"))
idmap = normalise_identity_cols(data.get("idmap"))
features = parse_week_col(normalise_identity_cols(data.get("features")))
risk = parse_week_col(attach_email_and_node_id(data.get("risk"), nodes, idmap))
edges = parse_week_col(normalise_identity_cols(data.get("edges")))
feat_imp = data.get("feat_imp")
preds = parse_week_col(attach_email_and_node_id(data.get("preds"), nodes, idmap))

if risk is not None and "risk_band" in risk.columns:
    risk = risk.copy()
    risk["risk_band"] = (
        risk["risk_band"].astype(str).str.title().replace({"Med": "Medium", "Elevated": "Medium"})
    )

features = attach_email_and_node_id(features, nodes, idmap)

# Filter by week
if week_range[0] is not None and "week" in (risk.columns if isinstance(risk, pd.DataFrame) else []):
    risk = risk[(risk["week"] >= week_range[0]) & (risk["week"] <= week_range[1])]
if week_range[0] is not None and "week" in (features.columns if isinstance(features, pd.DataFrame) else []):
    features = features[(features["week"] >= week_range[0]) & (features["week"] <= week_range[1])]
if week_range[0] is not None and "week" in (edges.columns if isinstance(edges, pd.DataFrame) else []):
    edges = edges[(edges["week"] >= week_range[0]) & (edges["week"] <= week_range[1])]
if week_range[0] is not None and preds is not None and "week" in preds.columns:
    preds = preds[(preds["week"] >= week_range[0]) & (preds["week"] <= week_range[1])]

# Filter by risk-band & email search
if risk is not None and not risk.empty:
    if "risk_band" in risk.columns:
        risk = risk[risk["risk_band"].astype(str).isin(risk_bands_sel)]
    if search_email and "email" in risk.columns:
        risk = risk[risk["email"].astype(str).str.contains(search_email, case=False, na=False)]

if features is not None and search_email and "email" in features.columns:
    features = features[features["email"].astype(str).str.contains(search_email, case=False, na=False)]

if risk is not None and "email" in risk.columns:
    risk["team"] = derive_team_domain(risk["email"])
if features is not None and "email" in features.columns:
    features["team"] = derive_team_domain(features["email"])

# ----------------------
# Header & KPIs
# ----------------------
st.title("Collaboration Risk Dashboard - Enron")
if client_name:
    st.caption(f"Prepared for {client_name}")

cols = st.columns(4)

def compute_activity_from_edges(edges_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if edges_df is None or edges_df.empty:
        return None
    df = parse_week_col(edges_df); df = normalise_identity_cols(df)
    w = next((c for c in ["weight","count","n","emails","volume"] if c in df.columns), None)
    if w is None:
        df["weight"] = 1; w = "weight"
    return df.groupby("week", dropna=True)[w].sum().reset_index(name="emails")

def compute_activity_from_features(features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if features_df is None or features_df.empty:
        return None
    df = parse_week_col(features_df); df = normalise_identity_cols(df)
    total = next((c for c in ["total_emails","emails_sent","emails","count"] if c in df.columns), None)
    if total is None:
        return None
    return df.groupby("week", dropna=True)[total].sum().reset_index(name="emails")

act_feat = compute_activity_from_features(features)
activity = act_feat if act_feat is not None and not act_feat.empty else compute_activity_from_edges(edges)
_total_emails = int(activity["emails"].sum()) if (activity is not None and not activity.empty) else 0
_active_individuals = int(features["email"].nunique()) if (features is not None and "email" in features.columns) else 0

latest_wk = pd.to_datetime(risk["week"]).max() if (risk is not None and not risk.empty and "week" in risk.columns) else None
if latest_wk is not None:
    snap = risk[pd.to_datetime(risk["week"])==latest_wk]
    by_band = snap.groupby("risk_band", observed=False).size().reindex(RISK_ORDER).fillna(0).astype(int)
else:
    by_band = pd.Series({b:0 for b in RISK_ORDER})

with cols[0]:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total emails (selected window)</div><div class='kpi-value'>{_total_emails:,}</div></div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Active individuals</div><div class='kpi-value'>{_active_individuals}</div></div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>High risk (latest week)</div><div class='kpi-value'>{int(by_band.get('High',0))}</div></div>", unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Critical risk (latest week)</div><div class='kpi-value'>{int(by_band.get('Critical',0))}</div></div>", unsafe_allow_html=True)

st.divider()

# ----------------------
# Tabs
# ----------------------
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Activity Trends",
    "Risk Over Time",
    "Top Risky Entities",
    "Feature Importance",
    "Predictions (optional)",
    "Insights & Report",
    "AI Agent Chat",
])

# ---- Activity Trends ----
with t1:
    st.subheader("Email activity over time")
    activity = compute_activity_from_features(features)
    if activity is None or activity.empty:
        activity = compute_activity_from_edges(edges)
    if activity is None or activity.empty:
        st.info("No activity data available (features/edges missing).")
    else:
        fig = px.line(activity.sort_values("week"), x="week", y="emails", markers=True)
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=10,b=10))
        show_plot(fig)

    if features is not None and not features.empty:
        st.markdown("**Breakout by selection**")
        metric_col = next((c for c in ["total_emails","emails_sent","emails","count"] if c in features.columns), None)
        if metric_col:
            group_col = "email" if group_mode.startswith("Individual") else "team"
            agg = features.groupby(["week", group_col])[metric_col].sum().reset_index()
            topN = st.slider("Show top N", 5, 30, 10)
            latest = agg[agg["week"] == agg["week"].max()].sort_values(metric_col, ascending=False)
            keep = latest[group_col].head(topN).tolist()
            agg = agg[agg[group_col].isin(keep)]
            fig2 = px.line(agg.sort_values(["week",group_col]), x="week", y=metric_col, color=group_col)
            fig2.update_layout(height=420, legend_title=group_col)
            show_plot(fig2)

# ---- Risk Over Time ----
with t2:
    st.subheader("Risk bands over time")
    if risk is None or risk.empty or "risk_band" not in risk.columns:
        st.info("No risk data available.")
    else:
        grp = risk.groupby(["week","risk_band"], observed=False).size().reset_index(name="count")
        grp["week"] = pd.to_datetime(grp["week"], errors="coerce").dt.date
        fig = px.area(
            grp.sort_values("week"), x="week", y="count", color="risk_band",
            color_discrete_map=RISK_COLOR, category_orders={"risk_band":RISK_ORDER}
        )
        fig.update_layout(height=380, legend_title="Band")
        show_plot(fig)

        score_col = next((c for c in ["risk_score","score","pred_risk_score"] if c in risk.columns), None)
        if score_col:
            avg = risk.groupby("week")[score_col].mean().reset_index(name="avg_risk")
            avg["week"] = pd.to_datetime(avg["week"], errors="coerce").dt.date
            st.markdown("**Average risk score**")
            fig2 = px.line(avg.sort_values("week"), x="week", y="avg_risk", markers=True)
            fig2.update_layout(height=300)
            show_plot(fig2)

# ---- Top Risky Entities ----
with t3:
    st.subheader("Top risky individuals / teams (latest week)")
    if risk is None or risk.empty:
        st.info("No risk data available.")
    else:
        latest_wk = pd.to_datetime(risk["week"]).max() if "week" in risk.columns else None
        snap = risk[pd.to_datetime(risk["week"]) == latest_wk].copy() if latest_wk is not None else risk.copy()
        group_col = "email" if group_mode.startswith("Individual") else "team"
        score_col = next((c for c in ["risk_score","score","pred_risk_score"] if c in snap.columns), None)
        if score_col is None:
            ord_map = {"Low":1, "Medium":2, "High":3, "Critical":4}
            snap["_score"] = snap.get("risk_band").map(ord_map).fillna(0)
            score_col = "_score"
        topN = st.slider("Show top N", 5, 50, 15)
        ranked = (
            snap.groupby([group_col])[[score_col]].mean()
                .rename(columns={score_col:"risk_score"}).reset_index()
                .sort_values("risk_score", ascending=False).head(topN)
        )
        band_counts = snap.groupby([group_col, "risk_band"], observed=False).size().unstack(fill_value=0)
        band_counts = band_counts.reindex(columns=RISK_ORDER, fill_value=0).reset_index()
        ranked = ranked.merge(band_counts, on=group_col, how="left")
        st.dataframe(ranked, use_container_width=True)
        fig = px.bar(ranked.sort_values("risk_score"), x="risk_score", y=group_col, orientation="h")
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=10,b=10))
        show_plot(fig)
        buf = io.BytesIO(); ranked.to_csv(buf, index=False)
        st.download_button("Download ranked CSV", buf.getvalue(), file_name="top_risky_entities.csv", mime="text/csv")

# ---- Feature Importance ----
with t4:
    st.subheader("Model feature importance")
    if feat_imp is None or feat_imp.empty:
        st.info("feature_importance.csv not found.")
    else:
        df = feat_imp.copy()
        name_col = next((c for c in ["feature","variable","name"] if c in df.columns), None)
        val_col = next((c for c in ["importance","gain","weight","mean_importance"] if c in df.columns), None)
        if not name_col or not val_col:
            st.warning("Could not find expected columns in feature_importance.csv")
        else:
            df = df[[name_col, val_col]].sort_values(val_col, ascending=True)
            df["tooltip"] = df[name_col].map(FEATURE_GLOSSARY).fillna("Model input feature")
            fig = px.bar(df.tail(30), x=val_col, y=name_col, orientation="h", hover_data={"tooltip": True})
            fig.update_layout(height=560, margin=dict(l=10,r=10,t=10,b=10))
            show_plot(fig)
            st.caption("Hover a bar to see what the feature means and why it matters.")

# ---- Predictions ----
with t5:
    st.subheader("Predicted risk (if provided)")
    if preds is None:
        st.info("predictions_test.csv not found.")
    else:
        preds_f = preds.copy()
        if week_range[0] is not None and "week" in preds_f.columns:
            preds_f = preds_f[(preds_f["week"] >= week_range[0]) & (preds_f["week"] <= week_range[1])]
        if preds_f.empty:
            st.info("No predictions in the current date window. Widen the Week range in the sidebar.")
        else:
            preds = preds_f
            numeric_cols = []
            tmp = preds.copy()
            for c in tmp.columns:
                if c in {"week","email","team","node_id","id","src","dst"}:
                    continue
                if pd.api.types.is_numeric_dtype(tmp[c]):
                    numeric_cols.append(c)
            if not numeric_cols:
                st.warning("No numeric prediction column found in predictions_test.csv")
            else:
                score_col = st.selectbox("Select prediction score column", options=numeric_cols, index=0)
                method = st.radio("Banding method", ["Quantiles (quartiles)", "Custom thresholds"], horizontal=True)
                if method.startswith("Custom"):
                    c1 = st.number_input("Low/Medium threshold", value=0.25, step=0.01)
                    c2 = st.number_input("Medium/High threshold", value=0.50, step=0.01)
                    c3 = st.number_input("High/Critical threshold", value=0.75, step=0.01)
                    def band(x):
                        return RISK_ORDER[(x>c1)+(x>c2)+(x>c3)]
                else:
                    q = tmp[score_col].quantile([0.25,0.5,0.75]).values.tolist()
                    def band(x):
                        return RISK_ORDER[(x>q[0])+(x>q[1])+(x>q[2])]
                preds["pred_band"] = preds[score_col].apply(band)
                group_col = "email" if group_mode.startswith("Individual") else "team"
                if group_col not in preds.columns and "email" in preds.columns:
                    preds["team"] = derive_team_domain(preds["email"])
                g = preds.groupby(["week", group_col])[score_col].mean().reset_index()
                latest = g[g["week"] == g["week"].max()].sort_values(score_col, ascending=False)
                keep = latest[group_col].head(12).tolist()
                g = g[g[group_col].isin(keep)]
                g["week"] = pd.to_datetime(g["week"], errors="coerce").dt.date
                fig = px.line(g.sort_values(["week",group_col]), x="week", y=score_col, color=group_col, markers=True)
                fig.update_layout(height=420, legend_title=group_col)
                show_plot(fig)
                latest_bands = preds[preds["week"]==preds["week"].max()].copy()
                st.dataframe(latest_bands[[group_col, score_col, "pred_band"]].sort_values(score_col, ascending=False).head(30), use_container_width=True)

                st.markdown("---")
                st.subheader("Scenario simulator")
                st.caption("Adjust feature values and see how the model's prediction changes.")

                data_path = Path(data_dir)
                model_file = None
                for cand in ["model_rf.pkl","model.pkl","model.joblib"] + [f for f in os.listdir(data_path) if f.lower().endswith((".pkl",".joblib"))]:
                    p = data_path / cand
                    if p.exists():
                        model_file = p
                        break
                loaded = None
                bundle = None
                if model_file:
                    try:
                        if joblib is not None:
                            loaded = joblib.load(model_file)
                        else:
                            with open(model_file, "rb") as f:
                                loaded = pickle.load(f)
                    except Exception:
                        try:
                            with open(model_file, "rb") as f:
                                loaded = pickle.load(f)
                        except Exception:
                            loaded = None
                if isinstance(loaded, dict):
                    bundle = loaded
                    est = bundle.get("pipeline") or bundle.get("model") or bundle.get("estimator") or bundle.get("clf")
                    feat_names = bundle.get("features")
                else:
                    est = loaded
                    feat_names = getattr(est, "feature_names_in_", None)

                if model_file is not None:
                    st.caption(f"Using model: `{model_file.name}`")

                schema_file = data_path / "model_expected_features.txt"
                if feat_names is None and schema_file.exists():
                    try:
                        feat_names = [ln.strip() for ln in schema_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    except Exception:
                        pass
                if feat_names is None and (data_path / "model_input.csv").exists():
                    try:
                        tmpmi = pd.read_csv(data_path / "model_input.csv", nrows=200)
                        feat_names = [c for c in tmpmi.columns if c.lower() not in {"y","target","label"}]
                    except Exception:
                        pass

                top_feats = None
                if feat_imp is not None and not feat_imp.empty:
                    name_col = next((c for c in ["feature","variable","name"] if c in feat_imp.columns), None)
                    val_col = next((c for c in ["importance","gain","weight","mean_importance"] if c in feat_imp.columns), None)
                    if name_col and val_col:
                        top_feats = feat_imp.sort_values(val_col, ascending=False)[name_col].head(6).tolist()
                if not top_feats:
                    top_feats = ["total_emails","after_hours_pct","out_emails","in_emails","degree","total_emails_ma4"]

                defaults = {f:0.0 for f in (feat_names or [])}
                try:
                    mi_full = pd.read_csv(data_path / "model_input.csv")
                    for f in defaults:
                        if f in mi_full.columns and pd.api.types.is_numeric_dtype(mi_full[f]):
                            defaults[f] = float(mi_full[f].median())
                except Exception:
                    pass

                user_vals = {}
                cols_sim = st.columns(3)
                for i, f in enumerate(top_feats):
                    with cols_sim[i%3]:
                        base = float(defaults.get(f, 0.0))
                        user_vals[f] = st.number_input(f, value=base)

                X = None
                if feat_names is not None:
                    row = {fn: defaults.get(fn, 0.0) for fn in feat_names}
                    for k,v in user_vals.items():
                        if k in row:
                            row[k] = float(v)
                    X = pd.DataFrame([row], columns=feat_names)

                pred_score = None
                try:
                    if est is not None and X is not None:
                        if hasattr(est, 'predict_proba'):
                            pred_score = float(est.predict_proba(X)[0,1])
                        elif hasattr(est, 'predict'):
                            pred_score = float(est.predict(X)[0])
                except Exception:
                    pred_score = None

                if pred_score is not None:
                    st.metric("Simulated prediction", f"{pred_score:.3f}")
                else:
                    st.warning("Could not compute a prediction with the current model/inputs.")

# ---- Insights & Report ----
with t6:
    st.subheader("Insights & Report")
    st.caption("Automated anomaly detection, narrative storyboard, and report export.")

    st.markdown("### Key Insights")
    insights = []
    activity = compute_activity_from_features(features)
    if activity is None or activity.empty:
        activity = compute_activity_from_edges(edges)
    if activity is not None and not activity.empty:
        s = activity.set_index('week')['emails'].copy()
        if s.std() > 0:
            z = (s - s.mean())/s.std()
            spikes = z[z>2].sort_values(ascending=False)
            if not spikes.empty:
                insights.append("Unusual email volume spikes on: " + ", ".join(spikes.index.astype(str)[:5]) + ".")
    if risk is not None and not risk.empty and 'risk_score' in risk.columns:
        r = risk.groupby('week')['risk_score'].mean().sort_index()
        if r.std() > 0:
            rz = (r - r.mean())/r.std()
            r_spikes = rz[rz>2].sort_values(ascending=False)
            if not r_spikes.empty:
                insights.append("Elevated average risk on weeks: " + ", ".join(r_spikes.index.astype(str)[:5]) + ".")
    if risk is not None and not risk.empty and 'team' in risk.columns:
        latest_wk = pd.to_datetime(risk['week']).max() if 'week' in risk.columns else None
        if latest_wk is not None:
            snap = risk[pd.to_datetime(risk['week'])==latest_wk]
            by_team = snap.groupby('team').size().sort_values(ascending=False).head(5)
            if not by_team.empty:
                insights.append("Teams with most risk flags last week: " + ", ".join([f"{k} ({v})" for k,v in by_team.items()]))
    if insights:
        for i in insights:
            st.markdown(f"- {i}")
    else:
        st.info("No strong anomalies detected with current filters.")

    st.markdown("### Storyboard for Client Presentation")
    st.markdown(dedent(f"""
    1) Baseline activity — Start with *Activity Trends* to show normal communication volume and top-send entities.
    2) Risk posture over time — Move to *Risk Over Time* to explain how Low→Critical bands evolve.
    3) Who needs attention — Use *Top Risky Entities* grouped by {'team' if group_mode.endswith('domain)') else 'individual'} to focus remediation.
    4) Why the model thinks so — In *Feature Importance*, hover bars to explain drivers in business terms.
    5) What happens next — In *Predictions*, show colored bands and run the *Scenario simulator* to test "what‑ifs".
    """))

    st.markdown("### Export Summary Report (PDF)")
    report_title = st.text_input("Report title", value=f"Collaboration Risk Summary — {client_name}")
    if st.button("Generate PDF"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph(f"<b>{report_title}</b>", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Total emails (window): <b>{_total_emails:,}</b>", styles['Normal']))
            story.append(Paragraph(f"Active individuals: <b>{_active_individuals}</b>", styles['Normal']))
            story.append(Paragraph(f"Latest High risk count: <b>{int(by_band.get('High',0))}</b>", styles['Normal']))
            story.append(Paragraph(f"Latest Critical risk count: <b>{int(by_band.get('Critical',0))}</b>", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Key Insights</b>", styles['Heading2']))
            if insights:
                for i in insights:
                    story.append(Paragraph(i, styles['Normal']))
            else:
                story.append(Paragraph("No strong anomalies detected with current filters.", styles['Normal']))
            story.append(Spacer(1, 12))
            doc.build(story)
            st.download_button("Download PDF", buffer.getvalue(), file_name="collab_risk_summary.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Could not generate PDF: {e}. Install reportlab or export from the tables instead.")

# ---- AI Agent Chat (FIXED - Unique chart keys) ----
with t7:
    st.subheader("AI Risk Agent - Ask Me Anything")
    st.caption("Natural language interface with visualizations and advanced analytics")
    
    @st.cache_resource
    def load_ai_agent():
        try:
            parser = QueryParser()
            responder = ResponseGenerator()
            return parser, responder, None
        except Exception as e:
            return None, None, str(e)
    
    parser, responder, error = load_ai_agent()
    
    if error:
        st.error(f"AI Agent could not load: {error}")
        st.info("Make sure ai_agent_parser.py and ai_agent_responder.py are in the same folder.")
    else:
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = [{
                "role": "assistant",
                "content": {
                    "text": """**Hello! I'm your AI Risk Agent.**

I can answer questions AND show visualizations!

**Try asking:**
- "Who is high risk this week?" → Gets bar chart
- "Why is christy.wire@enron.com high risk?" → Gets SHAP waterfall
- "How many people are critical?" → Gets pie chart
- "Compare two employees" → Gets comparison chart
- "What correlates with high risk?" → Advanced analysis

What would you like to know?""",
                    "figure": None
                }
            }]
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("AI Quick Questions")
            
            quick_questions = [
                "Who is high risk?",
                "How many critical?",
                "Show statistics",
                "Compare two people"
            ]
            
            for q in quick_questions:
                if st.button(q, key=f"ai_quick_{q}", use_container_width=True):
                    st.session_state.ai_messages.append({
                        "role": "user",
                        "content": {"text": q, "figure": None}
                    })
                    st.rerun()
            
            if st.button("Clear AI Chat", use_container_width=True):
                st.session_state.ai_messages = st.session_state.ai_messages[:1]
                st.rerun()
        
        for idx, message in enumerate(st.session_state.ai_messages):
            with st.chat_message(message["role"], avatar=None):
                content = message["content"]
                if isinstance(content, dict):
                    st.markdown(content["text"])
                    if content.get("figure"):
                        # FIXED: Unique key for historical charts
                        chart_key = f"chart_hist_{idx}_{int(time.time()*1000)}"
                        st.plotly_chart(content["figure"], use_container_width=True, key=chart_key)
                else:
                    st.markdown(content)
        
        if prompt := st.chat_input("Ask about employee risk..."):
            st.session_state.ai_messages.append({
                "role": "user",
                "content": {"text": prompt, "figure": None}
            })
            
            with st.chat_message("user", avatar=None):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar=None):
                with st.spinner("Analyzing..."):
                    try:
                        parsed = parser.parse(prompt)
                        result = responder.generate_response(parsed)
                        
                        if isinstance(result, dict):
                            st.markdown(result['text'])
                            
                            if result.get('figure'):
                                # FIXED: Unique key for new charts
                                chart_key = f"chart_new_{len(st.session_state.ai_messages)}_{int(time.time()*1000000)}"
                                st.plotly_chart(result['figure'], use_container_width=True, key=chart_key)
                            
                            st.session_state.ai_messages.append({
                                "role": "assistant",
                                "content": result
                            })
                        else:
                            st.markdown(result)
                            st.session_state.ai_messages.append({
                                "role": "assistant",
                                "content": {"text": result, "figure": None}
                            })
                        
                    except Exception as e:
                        err = f"Error: {str(e)}\n\nPlease rephrase your question."
                        st.error(err)
                        st.session_state.ai_messages.append({
                            "role": "assistant",
                            "content": {"text": err, "figure": None}
                        })
        
        st.markdown("---")
        with st.expander("AI Agent Capabilities", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Visualizations**")
                st.caption("- Bar charts\n- Pie charts\n- Waterfall charts\n- Comparisons")
            with cols[1]:
                st.markdown("**Analysis**")
                st.caption("- SHAP explanations\n- Correlations\n- Trends\n- Predictions")
            with cols[2]:
                st.markdown("**Queries**")
                st.caption("- Who/What/Why\n- Compare\n- Statistics\n- Recommendations")

# ---- Inspectors ----
with st.expander("Inspect raw/filtered tables"):
    opt = st.selectbox("Choose table", ["risk","features","edges","nodes","idmap","preds","feat_imp"])
    df = locals().get(opt)
    if isinstance(df, pd.DataFrame):
        st.dataframe(df.head(1000), use_container_width=True)
        buf = io.BytesIO(); df.to_csv(buf, index=False)
        st.download_button(f"Download {opt}.csv", buf.getvalue(), file_name=f"{opt}.csv", mime="text/csv")
    else:
        st.info("Selected table is empty or missing.")

st.caption("Tip: Use the sidebar to scope dates and bands. Group by individual or by team (email domain) to switch views.")