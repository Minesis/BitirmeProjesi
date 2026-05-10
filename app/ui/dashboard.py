"""
app/ui/dashboard.py
--------------------
Streamlit analytics dashboard.

Run:
    streamlit run app/ui/dashboard.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import config

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS – dark industrial theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
}
.metric-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #F59E0B;
}
.metric-label {
    font-size: 0.78rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.stApp { background-color: #0F172A; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

try:
    import config
    API_URL = config.get("dashboard.api_url", default="http://localhost:8000")
    REFRESH = config.get("dashboard.refresh_interval", default=5)
except Exception:
    API_URL = "http://localhost:8000"
    REFRESH = 5

BASE = f"{API_URL}/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
#  Data fetching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH)
def fetch_statistics(from_dt: str | None, to_dt: str | None) -> dict:
    params = {}
    if from_dt:
        params["from_dt"] = from_dt
    if to_dt:
        params["to_dt"] = to_dt
    try:
        r = requests.get(f"{BASE}/statistics", params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {}


@st.cache_data(ttl=REFRESH)
def fetch_visits(from_dt: str | None, to_dt: str | None, limit: int = 500, camera_id: str | None = None) -> list[dict]:
    params = {"limit": limit}
    if from_dt:
        params["from_dt"] = from_dt
    if to_dt:
        params["to_dt"] = to_dt
    if camera_id:
        params["camera_id"] = camera_id
    try:
        r = requests.get(f"{BASE}/visits", params=params, timeout=4)
        r.raise_for_status()
        payload = r.json()
        return payload.get("visits", payload.get("events", []))
    except Exception:
        return []


@st.cache_data(ttl=REFRESH)
def fetch_active(camera_id: str | None = None) -> dict:
    params = {}
    if camera_id:
        params["camera_id"] = camera_id
    try:
        r = requests.get(f"{BASE}/active", params=params, timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def api_is_live() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛒 Retail Analytics")
    st.markdown("---")

    status = "🟢 Online" if api_is_live() else "🔴 Offline"
    st.markdown(f"**API Status:** {status}")
    st.markdown(f"**Endpoint:** `{API_URL}`")
    st.markdown("---")

    st.markdown("### Date Filter")
    today = datetime.utcnow().date()
    preset = st.selectbox("Quick range", ["Today", "Last 7 days", "Last 30 days", "All time"])
    if preset == "Today":
        date_from = datetime.combine(today, datetime.min.time())
        date_to = None
    elif preset == "Last 7 days":
        date_from = datetime.combine(today - timedelta(days=7), datetime.min.time())
        date_to = None
    elif preset == "Last 30 days":
        date_from = datetime.combine(today - timedelta(days=30), datetime.min.time())
        date_to = None
    else:
        date_from = None
        date_to = None

    from_str = date_from.isoformat() if date_from else None
    to_str = date_to.isoformat() if date_to else None

    st.markdown("---")
    camera_id_filter_raw = st.text_input("Camera filter (optional)", value="", placeholder="e.g. cam_1")
    camera_id_filter = camera_id_filter_raw.strip() or None

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📊 Shelf Analytics Dashboard")
st.markdown(f"*Last updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}*")
st.markdown("---")

stats = fetch_statistics(from_str, to_str)
active_payload = fetch_active(camera_id_filter)
visits_raw = fetch_visits(from_str, to_str, camera_id=camera_id_filter)

if not stats:
    st.warning("⚠️ Cannot reach the API. Make sure `python main.py api` is running.")
    st.stop()

# ── KPI Metrics ──────────────────────────────────────────────────────────── #

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Unique Visitors</div>
        <div class="metric-value">{stats.get('unique_visitors', 0)}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Total Visits</div>
        <div class="metric-value">{stats.get('total_visits', 0)}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    avg_dur = stats.get("overall_avg_duration_seconds", 0)
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Avg Dwell Time</div>
        <div class="metric-value">{avg_dur:.1f}s</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Active Visitors</div>
        <div class="metric-value">{active_payload.get('count', 0)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ─────────────────────────────────────────────────────────── #

left, right = st.columns(2)

with left:
    st.markdown("### Gender Distribution")
    gd = stats.get("gender_distribution", {})
    if gd:
        fig = px.pie(
            names=list(gd.keys()),
            values=list(gd.values()),
            color_discrete_sequence=["#F59E0B", "#3B82F6", "#9CA3AF"],
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            legend=dict(font=dict(color="#E2E8F0")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

with right:
    st.markdown("### Age Group Distribution")
    ad = stats.get("age_group_distribution", {})
    # Sort by age group label order
    base_order = list(config.get("age_gender.age_labels", default=[])) or []
    extras = [k for k in ad.keys() if k not in base_order and k != "Unknown"]
    age_order = base_order + sorted(extras) + (["Unknown"] if "Unknown" in ad else [])
    ad_sorted = {k: ad.get(k, 0) for k in age_order if k in ad}
    if ad_sorted:
        fig = px.bar(
            x=list(ad_sorted.keys()),
            y=list(ad_sorted.values()),
            labels={"x": "Age Group", "y": "Count"},
            color=list(ad_sorted.keys()),
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            showlegend=False,
            margin=dict(t=20, b=20),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet")

# ── Charts row 2 ─────────────────────────────────────────────────────────── #

st.markdown("### Shelf Performance")

shelf_rows = stats.get("shelf_stats", [])
if shelf_rows:
    df_shelf = pd.DataFrame(shelf_rows)
    df_shelf["label"] = df_shelf["shelf_name"].astype(str) + " (" + df_shelf["camera_id"].astype(str) + ")"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_shelf["label"],
        y=df_shelf["visit_count"],
        name="Visits",
        marker_color="#3B82F6",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df_shelf["label"],
        y=df_shelf["avg_duration_seconds"],
        name="Avg Dwell (s)",
        mode="lines+markers",
        marker=dict(color="#F59E0B", size=10),
        line=dict(color="#F59E0B", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        yaxis=dict(title="Visits", gridcolor="#1E293B"),
        yaxis2=dict(title="Avg Dwell Time (s)", overlaying="y", side="right"),
        legend=dict(font=dict(color="#E2E8F0")),
        margin=dict(t=20, b=20),
        xaxis=dict(gridcolor="#1E293B"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Active Visitors by Shelf")
    df_active = df_shelf.sort_values("active_visitors", ascending=False)
    fig2 = px.bar(
        df_active,
        x="label",
        y="active_visitors",
        labels={"label": "Shelf (Camera)", "active_visitors": "Active Visitors"},
        color_discrete_sequence=["#F59E0B"],
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        showlegend=False,
        margin=dict(t=20, b=20),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No shelf data yet")

# ── Recent Events Table ───────────────────────────────────────────────────── #

st.markdown("### Active Visitors (Real-time)")
active_visits = active_payload.get("active_visits", []) if isinstance(active_payload, dict) else []
if active_visits:
    df_active_visits = pd.DataFrame(active_visits)
    cols = [c for c in [
        "camera_id",
        "shelf_name",
        "visitor_id",
        "gender",
        "age_group",
        "enter_time",
        "duration_seconds",
    ] if c in df_active_visits.columns]
    st.dataframe(df_active_visits[cols].head(200), use_container_width=True, hide_index=True)
else:
    st.info("No active visitors right now.")

st.markdown("### Recent Visits")
if visits_raw:
    df = pd.DataFrame(visits_raw)
    display_cols = [c for c in [
        "camera_id",
        "shelf_name",
        "visitor_id",
        "gender",
        "age_group",
        "enter_time",
        "exit_time",
        "duration_seconds",
    ] if c in df.columns]
    df = df[display_cols].rename(columns={
        "camera_id": "Camera",
        "shelf_name": "Shelf",
        "visitor_id": "Visitor ID",
        "gender": "Gender",
        "age_group": "Age Group",
        "enter_time": "Enter Time",
        "exit_time": "Exit Time",
        "duration_seconds": "Duration (s)",
    })
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
else:
    st.info("No visits recorded yet. Start the multi-camera detection system to collect data.")

# ── Timeline ─────────────────────────────────────────────────────────────── #

if visits_raw:
    df_all = pd.DataFrame(visits_raw)
    if "enter_time" in df_all.columns:
        df_all["enter_time"] = pd.to_datetime(df_all["enter_time"])
        df_all["hour"] = df_all["enter_time"].dt.floor("h")
        hourly = df_all.groupby("hour").size().reset_index(name="visits")
        st.markdown("### Hourly Traffic")
        fig = px.area(hourly, x="hour", y="visits",
                      color_discrete_sequence=["#F59E0B"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(REFRESH)
    st.cache_data.clear()
    st.rerun()
