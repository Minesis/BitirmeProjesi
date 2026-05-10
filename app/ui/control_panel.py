"""
app/ui/control_panel.py
--------------
UI-controlled multi-camera retail analytics control panel.

Launch (recommended):
    python run_ui.py

Fallback:
    streamlit run app/ui/control_panel.py
"""

from __future__ import annotations

import time
from datetime import datetime, time as dt_time, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from app.services.dashboard_service import DashboardService
from app.vision.camera_manager import CameraManager


# --------------------------------------------------------------------- #
#  Streamlit setup
# --------------------------------------------------------------------- #

st.set_page_config(page_title="Retail Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.stApp { background-color: #0B1220; color: #E5E7EB; }
section[data-testid="stSidebar"] { background-color: #0F172A; }
.card {
  background: #0F172A;
  border: 1px solid #1F2937;
  border-radius: 12px;
  padding: 14px 14px 10px 14px;
}
.muted { color: #9CA3AF; font-size: 0.9rem; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85rem; }
.pill-running { background: rgba(16,185,129,0.12); color: #34D399; border: 1px solid rgba(16,185,129,0.35); }
.pill-offline { background: rgba(148,163,184,0.10); color: #CBD5E1; border: 1px solid rgba(148,163,184,0.25); }
.pill-stopped { background: rgba(245,158,11,0.10); color: #FBBF24; border: 1px solid rgba(245,158,11,0.25); }
.pill-error { background: rgba(239,68,68,0.12); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.35); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_runtime() -> tuple[CameraManager, DashboardService]:
    db_path = config.get("database.path", default="data/analytics.db")
    mgr = CameraManager()
    dash = DashboardService(db_path)
    return mgr, dash


def _combine_date_time(d, t: dt_time) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)


def _status_pill(status: str) -> str:
    status = (status or "offline").lower()
    cls = {
        "running": "pill-running",
        "offline": "pill-offline",
        "stopped": "pill-stopped",
        "error": "pill-error",
    }.get(status, "pill-offline")
    label = status.upper()
    return f'<span class="pill {cls}">{label}</span>'


# --------------------------------------------------------------------- #
#  Sidebar navigation
# --------------------------------------------------------------------- #

mgr, dash = get_runtime()
stream = mgr.get_stream_manager()

st.sidebar.markdown("## Control Panel")
page = st.sidebar.radio("Pages", ["Dashboard", "Cameras", "Analytics", "Settings"], index=0)

auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
refresh_s = st.sidebar.slider("Refresh (s)", min_value=1, max_value=10, value=3, step=1)


# --------------------------------------------------------------------- #
#  Dashboard page
# --------------------------------------------------------------------- #

if page == "Dashboard":
    st.markdown("# Retail Analytics")
    st.markdown('<div class="muted">Multi-camera shelf analytics (UI-controlled)</div>', unsafe_allow_html=True)

    cams = mgr.list_camera_configs()
    cam_ids = [str(c.get("id")) for c in cams if c.get("id")]
    shelf_names = sorted({str(c.get("shelf_name")) for c in cams if c.get("shelf_name")})

    colf1, colf2, colf3, colf4 = st.columns([2, 2, 2, 2])
    with colf1:
        camera_filter = st.selectbox("Camera", options=["All"] + cam_ids, index=0)
    with colf2:
        shelf_filter = st.selectbox("Shelf", options=["All"] + shelf_names, index=0)
    with colf3:
        today = datetime.now().date()
        from_date = st.date_input("From date", value=today - timedelta(days=1))
    with colf4:
        to_date = st.date_input("To date", value=today)

    colft1, colft2 = st.columns(2)
    with colft1:
        from_time = st.time_input("From time", value=dt_time(0, 0))
    with colft2:
        to_time = st.time_input("To time", value=dt_time(23, 59))

    from_dt = _combine_date_time(from_date, from_time)
    to_dt = _combine_date_time(to_date, to_time)

    cam_arg = None if camera_filter == "All" else camera_filter
    shelf_arg = None if shelf_filter == "All" else shelf_filter

    stats = dash.get_statistics(from_dt=from_dt, to_dt=to_dt, camera_id=cam_arg, shelf_name=shelf_arg)
    active = dash.get_active_visits(camera_id=cam_arg, shelf_name=shelf_arg)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total visitors", int(stats.get("total_visits", 0)))
    m2.metric("Unique visitors", int(stats.get("unique_visitors", 0)))
    m3.metric("Active now", int(len(active)))
    m4.metric("Avg dwell (s)", float(stats.get("overall_avg_duration_seconds", 0.0)))

    # Shelf stats chart
    shelf_stats = stats.get("shelf_stats", []) or []
    if shelf_stats:
        df_shelf = pd.DataFrame(shelf_stats)
        st.markdown("## Shelf Overview")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df_shelf,
                x="shelf_name",
                y="visit_count",
                color="camera_id",
                title="Shelf Popularity (visits)",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(
                df_shelf,
                x="shelf_name",
                y="active_visitors",
                color="camera_id",
                title="Active Visitors per Shelf",
                template="plotly_dark",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data yet. Start cameras from the Cameras page.")

    # Distributions
    c3, c4 = st.columns(2)
    with c3:
        gd = stats.get("gender_distribution", {}) or {}
        if gd:
            df = pd.DataFrame({"gender": list(gd.keys()), "count": list(gd.values())})
            fig = px.pie(df, names="gender", values="count", title="Gender Distribution", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gender distribution: no data")
    with c4:
        ad = stats.get("age_group_distribution", {}) or {}
        if ad:
            df = pd.DataFrame({"age_group": list(ad.keys()), "count": list(ad.values())})
            base_order = list(config.get("age_gender.age_labels", default=[])) or []
            present = list(df["age_group"].astype(str).tolist())
            extras = [k for k in present if k not in base_order and k != "Unknown"]
            order = base_order + sorted(set(extras)) + (["Unknown"] if "Unknown" in present else [])
            fig = px.bar(
                df,
                x="age_group",
                y="count",
                title="Age Group Distribution",
                template="plotly_dark",
                category_orders={"age_group": order},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Age distribution: no data")

    st.markdown("## Active Visitors (Real-time)")
    if active:
        df_active = pd.DataFrame(active)
        cols = [c for c in ["camera_id", "shelf_name", "visitor_id", "gender", "age_group", "enter_time", "duration_seconds"] if c in df_active.columns]
        st.dataframe(df_active[cols].head(500), use_container_width=True, hide_index=True)
    else:
        st.info("No active visitors right now.")


# --------------------------------------------------------------------- #
#  Cameras page (management + live view)
# --------------------------------------------------------------------- #

if page == "Cameras":
    st.markdown("# Cameras")
    st.markdown('<div class="muted">Start/stop cameras and open live previews.</div>', unsafe_allow_html=True)

    if "open_camera_id" not in st.session_state:
        st.session_state.open_camera_id = None

    topc1, topc2, topc3, topc4 = st.columns([1.3, 1.3, 1.3, 2.2])
    with topc1:
        if st.button("Start All", use_container_width=True):
            try:
                mgr.start_all(mode="analytics")
            except Exception as exc:
                st.error(f"Start All failed: {exc}")
    with topc2:
        if st.button("Stop All", use_container_width=True):
            mgr.stop_all()
    with topc3:
        if st.button("Reload Config", use_container_width=True):
            mgr.reload_config()
    with topc4:
        st.caption("Tip: Test mode runs without DB writes and auto-stops.")

    cams = mgr.list_camera_configs()
    if not cams:
        st.warning("No cameras found in config/config.yaml (cameras: ...).")
    else:
        # Live view panel
        with st.expander("Live Camera View", expanded=bool(st.session_state.open_camera_id)):
            cam_ids = [str(c.get("id")) for c in cams if c.get("id")]
            selected = st.selectbox(
                "Select camera",
                options=cam_ids,
                index=cam_ids.index(st.session_state.open_camera_id) if st.session_state.open_camera_id in cam_ids else 0,
            )
            st.session_state.open_camera_id = selected

            s = mgr.get_status(selected)
            st.markdown(_status_pill(str(s.get("status"))), unsafe_allow_html=True)
            st.write(f"**{selected}** • Shelf: `{s.get('shelf_name')}` • Source: `{s.get('source')}` • FPS: `{s.get('fps')}`")
            if s.get("last_error"):
                st.error(str(s.get("last_error")))

            act1, act2, act3, act4 = st.columns(4)
            with act1:
                if st.button("Start (analytics)", key=f"live_start_{selected}", use_container_width=True):
                    try:
                        mgr.start_camera(selected, mode="analytics")
                    except Exception as exc:
                        st.error(f"Start failed: {exc}")
            with act2:
                if st.button("Stop", key=f"live_stop_{selected}", use_container_width=True):
                    mgr.stop_camera(selected)
            with act3:
                if st.button("Restart", key=f"live_restart_{selected}", use_container_width=True):
                    mgr.restart_camera(selected, mode="analytics")
            with act4:
                if st.button("Test (10s)", key=f"live_test_{selected}", use_container_width=True):
                    mgr.test_camera(selected, seconds=10.0)

            live_auto = st.toggle("Live auto-refresh", value=True)
            live_interval = st.slider("Live refresh (s)", min_value=0.1, max_value=2.0, value=0.25, step=0.05)

            frame = stream.get_latest_frame(selected)
            if frame is not None:
                st.image(frame.jpeg_bytes, caption=f"Last frame: {datetime.fromtimestamp(frame.timestamp).strftime('%H:%M:%S')}", use_container_width=True)
            else:
                st.info("No frame yet. Start or Test this camera to see preview.")

            if live_auto:
                time.sleep(float(live_interval))
                st.rerun()

        # Camera cards
        st.markdown("## Configured Cameras")
        cols = st.columns(3)
        for idx, cam in enumerate(cams):
            cid = str(cam.get("id", "")).strip()
            if not cid:
                continue
            shelf = str(cam.get("shelf_name", "")).strip()
            source = cam.get("source", "")
            status = mgr.get_status(cid)
            frame = stream.get_latest_frame(cid)

            with cols[idx % 3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"### `{cid}`")
                st.markdown(f"<div class='muted'>Shelf: <b>{shelf}</b><br/>Source: <code>{source}</code></div>", unsafe_allow_html=True)
                st.markdown(_status_pill(str(status.get("status"))), unsafe_allow_html=True)
                st.caption(f"FPS: {status.get('fps', 0.0)} • Active: {status.get('active_visitors', 0)}")
                if status.get("last_error"):
                    st.caption(f"Error: {status.get('last_error')}")

                if frame is not None:
                    st.image(frame.jpeg_bytes, use_container_width=True)
                else:
                    st.caption("No preview yet")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    if st.button("Start", key=f"start_{cid}", use_container_width=True):
                        mgr.start_camera(cid, mode="analytics")
                with b2:
                    if st.button("Stop", key=f"stop_{cid}", use_container_width=True):
                        mgr.stop_camera(cid)
                with b3:
                    if st.button("Restart", key=f"restart_{cid}", use_container_width=True):
                        mgr.restart_camera(cid, mode="analytics")
                with b4:
                    if st.button("Test", key=f"test_{cid}", use_container_width=True):
                        st.session_state.open_camera_id = cid
                        mgr.test_camera(cid, seconds=10.0)

                if st.button("Open Camera", key=f"open_{cid}", use_container_width=True):
                    st.session_state.open_camera_id = cid
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------- #
#  Analytics page (detailed)
# --------------------------------------------------------------------- #

if page == "Analytics":
    st.markdown("# Analytics")
    st.markdown('<div class="muted">Explore visits and statistics.</div>', unsafe_allow_html=True)

    cams = mgr.list_camera_configs()
    cam_ids = [str(c.get("id")) for c in cams if c.get("id")]
    shelf_names = sorted({str(c.get("shelf_name")) for c in cams if c.get("shelf_name")})

    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        camera_filter = st.selectbox("Camera", options=["All"] + cam_ids, index=0)
    with f2:
        shelf_filter = st.selectbox("Shelf", options=["All"] + shelf_names, index=0)
    with f3:
        limit = st.slider("Rows", min_value=50, max_value=2000, value=300, step=50)

    today = datetime.now().date()
    d1, d2 = st.columns(2)
    with d1:
        from_date = st.date_input("From", value=today - timedelta(days=7), key="ana_from_date")
    with d2:
        to_date = st.date_input("To", value=today, key="ana_to_date")

    from_dt = datetime(from_date.year, from_date.month, from_date.day, 0, 0, 0)
    to_dt = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59)

    cam_arg = None if camera_filter == "All" else camera_filter
    shelf_arg = None if shelf_filter == "All" else shelf_filter

    visits = dash.get_recent_visits(limit=int(limit), from_dt=from_dt, to_dt=to_dt, camera_id=cam_arg, shelf_name=shelf_arg)
    if visits:
        df = pd.DataFrame(visits)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No visits for selected filters.")


# --------------------------------------------------------------------- #
#  Settings page
# --------------------------------------------------------------------- #

if page == "Settings":
    st.markdown("# Settings")

    db_path = config.get("database.path", default="data/analytics.db")
    st.write(f"Database: `{db_path}`")

    st.markdown("## Config File")
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            yaml_text = f.read()
        st.code(yaml_text, language="yaml")
    except Exception as exc:
        st.error(f"Failed to read config/config.yaml: {exc}")

    s1, s2, s3 = st.columns(3)
    with s1:
        if st.button("Reload Config", use_container_width=True):
            mgr.reload_config()
            st.success("Config reloaded.")
    with s2:
        if st.button("Stop All Cameras", use_container_width=True):
            mgr.stop_all()
            st.success("All cameras stopped.")
    with s3:
        if st.button("Shutdown Runtime", use_container_width=True):
            mgr.shutdown()
            st.success("Runtime shutdown complete.")


# --------------------------------------------------------------------- #
#  Global auto-refresh
# --------------------------------------------------------------------- #

if auto_refresh and page in {"Dashboard"}:
    time.sleep(float(refresh_s))
    st.rerun()
