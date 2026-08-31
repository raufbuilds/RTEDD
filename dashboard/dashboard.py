import os
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from rtedd_agentic_analyst_ui import render_rtedd_agentic_analyst


SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
BASE_URL = f"http://{SERVER_IP}:8000"
RECORD_COUNT_URL = f"{BASE_URL}/records/count"
DASHBOARD_DATA_URL = f"{BASE_URL}/dashboard/data"
FORECAST_URL = f"{BASE_URL}/forecast/latest"
FORECAST_REFRESH_URL = f"{BASE_URL}/forecast/refresh"
METRICS_2025_URL = f"{BASE_URL}/forecast/2025-metrics"
FORECAST_FRESH_POLL_SECONDS = 30
FORECAST_POLL_SECONDS = 5
HOURS = list(range(1, 25))


st.set_page_config(
    page_title="RTEDD | Real-Time Electricity Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg:#07111f; --panel:#0d1a2a; --panel2:#101f31;
        --line:#203247; --text:#e8eef7; --muted:#93a4b8;
        --blue:#4f9cff; --green:#54d88b; --amber:#f4b942; --red:#ff6868;
    }
    .stApp { background:var(--bg); color:var(--text); }
    .block-container { max-width:1600px; padding-top:1.2rem; padding-bottom:2.5rem; }
    [data-testid="stSidebar"] { background:#081321; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--text); }
    [data-testid="stSidebar"] .stRadio label { border-radius:8px; padding:4px 8px; }
    .brand-card {
        background:linear-gradient(120deg,#0b1a2c 0%,#102b46 55%,#0c473f 100%);
        border:1px solid #24415b; border-radius:18px; padding:24px 28px;
        color:var(--text); margin-bottom:16px; box-shadow:0 12px 30px rgba(0,0,0,.22);
    }
    .brand-card h1 { margin:0; font-size:2rem; letter-spacing:-.03em; }
    .brand-card p { margin:.55rem 0 0; color:#b8c7d7; }
    .eyebrow {
        color:#91a3b7; font-size:.73rem; font-weight:750;
        letter-spacing:.12em; text-transform:uppercase; margin:1.15rem 0 .55rem;
    }
    .insight-card {
        background:linear-gradient(145deg,#0e1d2e,#0a1726);
        border:1px solid #22384e; border-radius:15px; padding:17px 18px;
        min-height:116px; box-shadow:0 6px 18px rgba(0,0,0,.16);
    }
    .insight-card .label {
        color:#9eafc2; font-size:.73rem; font-weight:750;
        text-transform:uppercase; letter-spacing:.07em;
    }
    .insight-card .value { color:#eaf3ff; font-size:1.75rem; font-weight:750; margin-top:7px; }
    .insight-card .sub { color:#9db0c3; font-size:.82rem; margin-top:5px; }
    .forecast-card {
        background:linear-gradient(145deg,#102239,#0b1a2a);
        border:1px solid #28517a; border-radius:15px; padding:17px 18px; min-height:116px;
    }
    .forecast-card .label { color:#9eafc2; font-size:.73rem; font-weight:750; letter-spacing:.07em; }
    .forecast-card .value { color:#65e09a; font-size:1.7rem; font-weight:750; margin:7px 0 4px; }
    .forecast-card .models { color:#b7c7d7; font-size:.78rem; line-height:1.6; }
    [data-testid="stMetric"] {
        background:#0d1a2a; border:1px solid #22384e; border-radius:12px; padding:14px;
    }
    [data-testid="stPlotlyChart"] { border:1px solid #1d3044; border-radius:14px; background:#0b1828; padding:4px; }
    .stTabs [data-baseweb="tab"] { color:#9fb1c3; border-radius:8px 8px 0 0; padding:10px 16px; }
    .stTabs [aria-selected="true"] { color:#63a8ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_2025_test_data_warning(df):
    """Show a warning if 2025 data is present, indicating test mode."""
    if df.empty:
        return
    
    if df["Date"].dt.year.max() >= 2025 and df["Date"].dt.year.min() <= 2024:
        st.warning(
            "⚠️ **TEST MODE ACTIVE**: 2025 is configured as test data. "
            "Forecast models are trained only on 2020-2024 data. "
            "Use the 'Forecast Training Comparison' view to evaluate performance against 2025 actual demand."
        )


def ensure_state():
    if "dashboard_df" not in st.session_state:
        st.session_state.dashboard_df = pd.DataFrame()
    if "dashboard_baseline" not in st.session_state:
        st.session_state.dashboard_baseline = pd.DataFrame()
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "selected_anomaly_id" not in st.session_state:
        st.session_state.selected_anomaly_id = None
    if "last_received_epoch" not in st.session_state:
        st.session_state.last_received_epoch = None
    if "last_received_record" not in st.session_state:
        st.session_state.last_received_record = None
    if "refresh_seconds" not in st.session_state:
        st.session_state.refresh_seconds = 2
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    if "scope" not in st.session_state:
        st.session_state.scope = "Today"
    if "date_range" not in st.session_state:
        st.session_state.date_range = None
    if "hour_range" not in st.session_state:
        st.session_state.hour_range = (1, 24)
    if "show_normal_rows" not in st.session_state:
        st.session_state.show_normal_rows = False
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Today"
    elif st.session_state.view_mode == "Today vs Average":
        st.session_state.view_mode = "Today vs Forecast Benchmarks"
    elif st.session_state.view_mode in {
        "Today vs Forecast Benchmarks",
        "Today vs Live-Trained Forecast",
    }:
        st.session_state.view_mode = "Forecast Training Comparison"
    if "forecast_status" not in st.session_state:
        st.session_state.forecast_status = None
    if "forecast_trained_at" not in st.session_state:
        st.session_state.forecast_trained_at = None
    if "forecast_requested_at" not in st.session_state:
        st.session_state.forecast_requested_at = None
    if "forecast_training_seconds" not in st.session_state:
        st.session_state.forecast_training_seconds = None
    if "forecast_message" not in st.session_state:
        st.session_state.forecast_message = None
    if "forecast_summary" not in st.session_state:
        st.session_state.forecast_summary = None
    if "forecast_cache" not in st.session_state:
        st.session_state.forecast_cache = {}
    if "total_records_received" not in st.session_state:
        st.session_state.total_records_received = 0
    if "last_record_count_fetch" not in st.session_state:
        st.session_state.last_record_count_fetch = 0


def coerce_date_range_value(value):
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            start = pd.to_datetime(value[0]).date()
            end = pd.to_datetime(value[1]).date()
        elif len(value) == 1:
            start = end = pd.to_datetime(value[0]).date()
        else:
            return None
    elif value is not None:
        start = end = pd.to_datetime(value).date()
    else:
        return None

    if start > end:
        start, end = end, start
    return (start, end)


def clamp_date_range(date_range, min_date, max_date):
    normalized = coerce_date_range_value(date_range)
    if normalized is None:
        return (min_date, max_date)

    start, end = normalized
    if start < min_date:
        start = min_date
    if end > max_date:
        end = max_date
    if start > end:
        start, end = min_date, max_date
    return (start, end)


def fetch_total_records_received(force=False):
    now = time.time()
    if not force and now - st.session_state.last_record_count_fetch < 5:
        return st.session_state.total_records_received

    try:
        response = requests.get(RECORD_COUNT_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return max(st.session_state.total_records_received, len(st.session_state.dashboard_df))

    total_records = int(payload.get("total_records", 0))
    st.session_state.total_records_received = total_records
    st.session_state.last_record_count_fetch = now
    return total_records


def fetch_dashboard_data():
    try:
        response = requests.get(DASHBOARD_DATA_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.session_state.last_error = f"Dashboard data load failed: {exc}"
        return (
            st.session_state.dashboard_df.copy(),
            st.session_state.dashboard_baseline.copy(),
            st.session_state.total_records_received,
        )

    records = payload.get("records") or []
    baseline_rows = payload.get("baseline") or []
    total_records = int(payload.get("total_records", len(records)))

    df = pd.DataFrame(records)
    if not df.empty:
        for column in ["Date", "Timestamp"]:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")
        numeric_cols = [
            "id",
            "Hour",
            "Ontario Demand",
            "Expected Demand",
            "Deviation",
            "Anomaly Score",
        ]
        for column in numeric_cols:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        if "Anomaly" in df.columns:
            df["Anomaly"] = df["Anomaly"].astype(bool)
        df = df.dropna(subset=["Date", "Hour", "Ontario Demand", "Timestamp"])
        df["Hour"] = df["Hour"].astype(int)
        if "Date Label" not in df.columns:
            df["Date Label"] = df["Date"].dt.strftime("%Y-%m-%d")

    baseline = pd.DataFrame(baseline_rows)
    if not baseline.empty:
        for column in ["Hour", "Expected", "Scale", "Lower", "Upper"]:
            if column in baseline.columns:
                baseline[column] = pd.to_numeric(baseline[column], errors="coerce")
        baseline = baseline.dropna(subset=["Hour", "Expected", "Lower", "Upper"])
        baseline["Hour"] = baseline["Hour"].astype(int)

    st.session_state.dashboard_df = df
    st.session_state.dashboard_baseline = baseline
    st.session_state.total_records_received = total_records
    st.session_state.last_received_epoch = time.time()
    st.session_state.last_received_record = (
        df.sort_values(["Date", "Hour", "id"]).tail(1).to_dict(orient="records")[0]
        if not df.empty
        else None
    )
    st.session_state.last_error = None
    return df.copy(), baseline.copy(), total_records


def request_forecast_refresh(target_date=None, include_target_date=False):
    params: dict[str, Any] = {"include_target_date": include_target_date}
    if target_date is not None:
        params["target_date"] = pd.Timestamp(target_date).date().isoformat()

    try:
        response = requests.post(FORECAST_REFRESH_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
        st.session_state.forecast_status = payload.get("status")
        st.session_state.forecast_message = "Forecast refresh queued"
        if target_date is None:
            st.session_state.forecast_cache.clear()
        else:
            st.session_state.forecast_cache.pop(
                forecast_client_cache_key(target_date, include_target_date),
                None,
            )
        st.session_state.last_error = None
        return True
    except requests.RequestException as exc:
        st.session_state.last_error = f"Forecast refresh failed: {exc}"
        return False


def forecast_client_cache_key(target_date=None, include_target_date=False):
    if target_date is None:
        target_key = None
    else:
        target_key = pd.Timestamp(target_date).date().isoformat()
    return (target_key, bool(include_target_date))


def empty_forecast_frame():
    return pd.DataFrame(
        columns=[
            "Hour",
            "Prophet",
            "LightGBM",
            "Ensemble",
            "Ensemble_P10",
            "Ensemble_P50",
            "Ensemble_P90",
        ]
    )


def forecast_methods_for_view(include_today_in_training=False):
    """Return the methods to display for the requested forecast view."""
    if include_today_in_training:
        return ["Prophet", "LightGBM", "Ensemble"]
    return ["Average", "Expected Median", "Prophet", "LightGBM", "Ensemble"]


def forecast_frame_from_rows(forecast_rows):
    if not forecast_rows:
        return empty_forecast_frame()

    forecast_df = pd.DataFrame(forecast_rows)
    legacy_lightgbm_columns = {
        "XGBoost": "LightGBM",
        "XGBoost_P10": "LightGBM_P10",
        "XGBoost_P50": "LightGBM_P50",
        "XGBoost_P90": "LightGBM_P90",
    }
    for legacy_name, current_name in legacy_lightgbm_columns.items():
        if legacy_name in forecast_df.columns and current_name not in forecast_df.columns:
            forecast_df[current_name] = forecast_df[legacy_name]

    forecast_columns = [
        "Hour",
        "Prophet",
        "LightGBM",
        "Ensemble",
        "Ensemble_P10",
        "Ensemble_P50",
        "Ensemble_P90",
    ]
    for column in forecast_columns:
        if column not in forecast_df.columns:
            forecast_df[column] = pd.NA
        forecast_df[column] = pd.to_numeric(forecast_df[column], errors="coerce")

    forecast_df = forecast_df.dropna(subset=["Hour"])
    forecast_df["Hour"] = forecast_df["Hour"].astype(int)
    return forecast_df[forecast_columns].sort_values("Hour")


def forecast_poll_interval(status):
    if status in {"fresh", "failed", "empty"}:
        return FORECAST_FRESH_POLL_SECONDS
    return FORECAST_POLL_SECONDS


def fetch_cached_forecast(target_date=None, include_target_date=False):
    cache_key = forecast_client_cache_key(target_date, include_target_date)
    cached = st.session_state.forecast_cache.get(cache_key)
    now = time.time()
    if cached is not None:
        age = now - cached.get("fetched_at", 0)
        interval = forecast_poll_interval(cached.get("status"))
        if age < interval:
            return forecast_frame_from_rows(cached.get("forecast") or [])

    params: dict[str, Any] = {"include_target_date": include_target_date}
    if target_date is not None:
        params["target_date"] = pd.Timestamp(target_date).date().isoformat()

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.session_state.last_error = f"Forecast fetch failed: {exc}"
        if cached is not None:
            return forecast_frame_from_rows(cached.get("forecast") or [])
        return empty_forecast_frame()

    st.session_state.forecast_status = payload.get("status")
    st.session_state.forecast_trained_at = payload.get("trained_at")
    st.session_state.forecast_requested_at = payload.get("requested_at")
    st.session_state.forecast_training_seconds = payload.get("training_seconds")
    st.session_state.forecast_message = payload.get("message")
    st.session_state.forecast_summary = payload.get("summary")
    st.session_state.forecast_cache[cache_key] = {
        "fetched_at": now,
        "status": payload.get("status"),
        "forecast": payload.get("forecast") or [],
    }

    if payload.get("error"):
        st.session_state.last_error = f"Forecast error: {payload['error']}"

    return forecast_frame_from_rows(payload.get("forecast") or [])


def compute_ensemble_forecast(df, target_date=None, include_target_date=False):
    """
    Fetch the latest cached Prophet/LightGBM ensemble forecast from FastAPI.
    """
    return fetch_cached_forecast(target_date, include_target_date)


def add_baseline_to_figure(fig, baseline, hours, title_suffix=""):
    if baseline.empty:
        return fig

    band = baseline[baseline["Hour"].isin(hours)].sort_values("Hour")
    if band.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=band["Hour"],
            y=band["Upper"],
            mode="lines",
            line=dict(color="rgba(160,160,160,0.35)"),
            name="Expected + band",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["Hour"],
            y=band["Lower"],
            mode="lines",
            line=dict(color="rgba(160,160,160,0.35)"),
            fill="tonexty",
            fillcolor="rgba(160,160,160,0.15)",
            name="Expected band",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=band["Hour"],
            y=band["Expected"],
            mode="lines",
            line=dict(color="rgba(90,90,90,0.85)", dash="dot"),
            name="Expected (median)",
        )
    )

    if title_suffix:
        fig.update_layout(title=f"{fig.layout.title.text}{title_suffix}")
    return fig


def fix_hour_axis(fig):
    fig.update_xaxes(
        title_text="Hour",
        range=[-0.5, 23.5],
        tickmode="array",
        tickvals=HOURS,
        ticktext=[str(hour) for hour in HOURS],
        dtick=1,
    )
    return fig


def add_anomaly_markers(fig, df_with_anomaly, label_col=None):
    anomalies = df_with_anomaly[df_with_anomaly["Anomaly"]].copy()
    if anomalies.empty:
        return fig

    hover_text = None
    if label_col and label_col in anomalies.columns:
        hover_text = anomalies[label_col]

    fig.add_trace(
        go.Scatter(
            x=anomalies["Hour"],
            y=anomalies["Ontario Demand"],
            mode="markers",
            marker=dict(size=11, color="#d62728", symbol="x"),
            name="Anomaly",
            text=hover_text,
            customdata=anomalies[["Expected Demand", "Deviation", "Anomaly Score"]].to_numpy(),
            hovertemplate=(
                "Hour: %{x}<br>"
                "Demand: %{y:.1f} MW<br>"
                "Expected: %{customdata[0]:.1f} MW<br>"
                "Deviation: %{customdata[1]:.1f} MW<br>"
                "Score: %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    return fig


def sidebar_controls(df):
    st.sidebar.markdown("# ⚡ GridPulse")
    st.sidebar.caption("ENERGY INTELLIGENCE PLATFORM")
    st.sidebar.divider()

    workspace = st.sidebar.radio(
        "WORKSPACE",
        ["Command Center", "Forecast Studio", "Alert Center", "Data Explorer", "🤖 Agentic Analyst"],
        label_visibility="visible",
    )

    st.sidebar.markdown("### View controls")
    scope = st.sidebar.selectbox(
        "Period",
        ["Today", "Last 7 days", "All data", "Custom date range"],
        key="scope",
    )

    if df.empty:
        min_date = max_date = pd.Timestamp.today().date()
    else:
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()

    date_range = st.session_state.get("date_range") or (min_date, max_date)
    if scope == "Custom date range":
        selected = st.sidebar.date_input(
            "Custom range", value=date_range, min_value=min_date, max_value=max_date
        )
        date_range = coerce_date_range_value(selected) or (min_date, max_date)

    hour_range = st.sidebar.slider("Operating hours", 1, 24, (1, 24))

    st.sidebar.divider()
    st.sidebar.markdown("### Live connection")
    auto_refresh_enabled = st.sidebar.checkbox("Live updates", value=True)
    refresh_seconds = st.sidebar.select_slider(
        "Refresh interval", options=[2, 5, 10, 30, 60], value=5,
        format_func=lambda x: f"{x}s"
    )
    if st.sidebar.button("↻ Refresh data", use_container_width=True):
        fetch_dashboard_data()

    st.sidebar.divider()
    st.sidebar.caption(f"Connected records: {len(df):,}")
    st.sidebar.caption("● API connection active")

    return (
        workspace,
        refresh_seconds,
        auto_refresh_enabled,
        scope,
        date_range,
        hour_range,
        True,
    )


def apply_scope_and_filters(df, scope, date_range, hour_range):
    if df.empty:
        return df

    df_view = df.copy()

    if scope == "Today":
        latest_date = df_view["Date"].max()
        df_view = df_view[df_view["Date"] == latest_date]
    elif scope == "Last 7 days":
        latest_date = df_view["Date"].max()
        cutoff = latest_date - pd.Timedelta(days=6)
        df_view = df_view[df_view["Date"] >= cutoff]
    elif scope == "Custom date range":
        start = None
        end = None
        if isinstance(date_range, (list, tuple)):
            if len(date_range) >= 2:
                start, end = date_range[0], date_range[1]
            elif len(date_range) == 1:
                start = end = date_range[0]
        elif date_range is not None:
            start = end = date_range

        if start is not None and end is not None:
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            if start_ts > end_ts:
                start_ts, end_ts = end_ts, start_ts
            df_view = df_view[(df_view["Date"] >= start_ts) & (df_view["Date"] <= end_ts)]

    hr0, hr1 = hour_range
    df_view = df_view[(df_view["Hour"] >= hr0) & (df_view["Hour"] <= hr1)]
    return df_view


def build_scope_label(df_view, scope, hour_range):
    hour_label = f"Hours {hour_range[0]}-{hour_range[1]}"
    if df_view.empty:
        return f"{scope} | {hour_label}"

    min_date = df_view["Date"].min().date()
    max_date = df_view["Date"].max().date()

    if scope == "Today":
        scope_label = f"Today ({max_date})"
    elif scope == "Last 7 days":
        scope_label = f"Last 7 days ({min_date} to {max_date})"
    elif scope == "Custom date range":
        scope_label = f"Custom range ({min_date} to {max_date})"
    else:
        scope_label = f"All data ({min_date} to {max_date})"

    return f"{scope_label} | {hour_label}"


def render_today(df, scope_label, baseline=None):
    latest_date = df["Date"].max()
    df_today = df[df["Date"] == latest_date]

    if df_today.empty:
        st.info("No data for the latest date yet")
        return

    fig = px.line(
        df_today,
        x="Hour",
        y="Ontario Demand",
        title=f"Ontario Demand - {latest_date.date()} | {scope_label}",
        markers=True,
    )
    baseline_df = baseline if baseline is not None else pd.DataFrame()
    fig = add_baseline_to_figure(fig, baseline_df, HOURS)
    fig = add_anomaly_markers(fig, df_today)
    fig = fix_hour_axis(fig)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_all_dates(df, scope_label):
    daily_count = df["Date"].nunique()
    if daily_count <= 1:
        render_today(df, scope_label)
        return

    st.subheader("All Dates Overview")

    hourly = (
        df.groupby(["Date", "Date Label", "Hour"], as_index=False)["Ontario Demand"]
        .mean()
        .sort_values(["Date", "Hour"])
    )
    heatmap_data = hourly.pivot(
        index="Date Label",
        columns="Hour",
        values="Ontario Demand",
    )
    heatmap_data = heatmap_data.reindex(HOURS, axis=1)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            x=HOURS,
            y=heatmap_data.index,
            z=heatmap_data.values,
            colorscale="Viridis",
            colorbar=dict(title="MW"),
            hovertemplate="Date: %{y}<br>Hour: %{x}:00<br>Demand: %{z:.1f} MW<extra></extra>",
        )
    )
    anomaly_points = df[df["Anomaly"]].copy()
    if not anomaly_points.empty:
        fig_heatmap.add_trace(
            go.Scatter(
                x=anomaly_points["Hour"],
                y=anomaly_points["Date Label"],
                mode="markers",
                marker=dict(size=8, color="#d62728", symbol="x"),
                name="Anomaly",
                customdata=anomaly_points[
                    ["Ontario Demand", "Expected Demand", "Deviation", "Anomaly Score"]
                ].to_numpy(),
                hovertemplate=(
                    "Date: %{y}<br>"
                    "Hour: %{x}:00<br>"
                    "Demand: %{customdata[0]:.1f} MW<br>"
                    "Expected: %{customdata[1]:.1f} MW<br>"
                    "Deviation: %{customdata[2]:.1f} MW<br>"
                    "Score: %{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )
    fig_heatmap.update_layout(
        title=f"Demand Heatmap by Date and Hour | {scope_label}",
        xaxis_title="Hour",
        yaxis_title="Date",
        yaxis=dict(autorange="reversed"),
        height=max(420, min(900, 26 * daily_count + 180)),
        hovermode="closest",
    )
    fig_heatmap = fix_hour_axis(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    left_col, right_col = st.columns(2)

    daily_summary = (
        df.groupby(["Date", "Date Label"], as_index=False)
        .agg(
            Average_Demand=("Ontario Demand", "mean"),
            Peak_Demand=("Ontario Demand", "max"),
            Minimum_Demand=("Ontario Demand", "min"),
            Records=("Ontario Demand", "size"),
        )
        .sort_values("Date")
    )
    daily_summary_melted = daily_summary.melt(
        id_vars=["Date", "Date Label"],
        value_vars=["Average_Demand", "Peak_Demand", "Minimum_Demand"],
        var_name="Series",
        value_name="Demand",
    )
    daily_summary_melted["Series"] = daily_summary_melted["Series"].str.replace("_", " ")

    with left_col:
        fig_daily = px.line(
            daily_summary_melted,
            x="Date",
            y="Demand",
            color="Series",
            markers=True,
            title="Daily Demand Summary",
            color_discrete_map={
                "Average Demand": "#1f77b4",
                "Peak Demand": "#d62728",
                "Minimum Demand": "#2ca02c",
            },
        )
        fig_daily.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="MW")
        st.plotly_chart(fig_daily, use_container_width=True)

    hourly_profile = (
        df.groupby("Hour")["Ontario Demand"]
        .agg(
            Median="median",
            P10=lambda series: series.quantile(0.10),
            P90=lambda series: series.quantile(0.90),
            Minimum="min",
            Maximum="max",
        )
        .reset_index()
        .sort_values("Hour")
    )
    latest_date = df["Date"].max()
    latest_profile = (
        df[df["Date"] == latest_date]
        .groupby("Hour", as_index=False)["Ontario Demand"]
        .mean()
        .sort_values("Hour")
    )

    with right_col:
        fig_profile = go.Figure()
        fig_profile.add_trace(
            go.Scatter(
                x=hourly_profile["Hour"],
                y=hourly_profile["P90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_profile.add_trace(
            go.Scatter(
                x=hourly_profile["Hour"],
                y=hourly_profile["P10"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                line=dict(width=0),
                name="10-90% range",
                hovertemplate="Hour: %{x}<br>10th percentile: %{y:.1f} MW<extra></extra>",
            )
        )
        fig_profile.add_trace(
            go.Scatter(
                x=hourly_profile["Hour"],
                y=hourly_profile["Median"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=6),
                name="Median",
                hovertemplate="Hour: %{x}<br>Median: %{y:.1f} MW<extra></extra>",
            )
        )
        fig_profile.add_trace(
            go.Scatter(
                x=latest_profile["Hour"],
                y=latest_profile["Ontario Demand"],
                mode="lines+markers",
                line=dict(color="#d62728", width=3),
                marker=dict(size=7),
                name=f"Latest date ({latest_date.date()})",
                hovertemplate="Hour: %{x}<br>Latest: %{y:.1f} MW<extra></extra>",
            )
        )
        fig_profile.update_layout(
            title="Typical Hourly Range vs Latest Date",
            yaxis_title="MW",
            hovermode="x unified",
        )
        fig_profile = fix_hour_axis(fig_profile)
        st.plotly_chart(fig_profile, use_container_width=True)

    with st.expander("Daily summary table"):
        display_summary = daily_summary.rename(
            columns={
                "Average_Demand": "Average Demand",
                "Peak_Demand": "Peak Demand",
                "Minimum_Demand": "Minimum Demand",
            }
        )
        st.dataframe(
            display_summary[
                ["Date Label", "Average Demand", "Peak Demand", "Minimum Demand", "Records"]
            ].round(1),
            use_container_width=True,
            hide_index=True,
        )


def render_average(df, scope_label):
    df_avg = df.groupby("Hour", as_index=False)["Ontario Demand"].mean()
    fig = px.line(df_avg, x="Hour", y="Ontario Demand", title=f"Average Demand | {scope_label}", markers=True)
    fig = fix_hour_axis(fig)
    st.plotly_chart(fig, use_container_width=True)


def calculate_accuracy_percentage(actual, predicted):
    comparison = pd.DataFrame({"Actual": actual, "Predicted": predicted}).dropna()
    comparison = comparison[comparison["Actual"] != 0]
    if comparison.empty:
        return None

    absolute_percentage_error = (
        (comparison["Actual"] - comparison["Predicted"]).abs() / comparison["Actual"].abs()
    )
    accuracy = 100 - (absolute_percentage_error.mean() * 100)
    return max(0.0, min(100.0, float(accuracy)))


def calculate_error_metrics(actual, predicted):
    comparison = pd.DataFrame({"Actual": actual, "Predicted": predicted}).dropna()
    if comparison.empty:
        return None

    error = comparison["Actual"] - comparison["Predicted"]
    mae = error.abs().mean()
    rmse = (error.pow(2).mean()) ** 0.5
    nonzero = comparison[comparison["Actual"] != 0]
    mape = None
    if not nonzero.empty:
        mape = ((nonzero["Actual"] - nonzero["Predicted"]).abs() / nonzero["Actual"].abs()).mean() * 100

    return {
        "MAE (MW)": float(mae),
        "RMSE (MW)": float(rmse),
        "MAPE %": None if mape is None else float(mape),
        "Compared Hours": len(comparison),
    }


def fetch_2025_metrics():
    """Fetch 2025 test data metrics from the server."""
    try:
        response = requests.get(METRICS_2025_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Failed to fetch 2025 metrics: {exc}")
        return None


def render_2025_metrics():
    """Render the 2025 test data metrics panel."""
    st.subheader("📊 2025 Test Data: Forecast Accuracy Metrics")
    
    metrics_data = fetch_2025_metrics()
    
    if metrics_data is None:
        st.info("Unable to fetch 2025 metrics")
        return
    
    if not metrics_data.get("available"):
        st.warning(f"⚠️ {metrics_data.get('message', 'No 2025 data available')}")
        return
    
    # Display key metrics as cards
    col1, col2, col3 = st.columns(3)
    
    metrics_info = metrics_data.get("metrics", {})
    mae = metrics_info.get("MAE", None)
    mape = metrics_info.get("MAPE", None)
    rmse = metrics_info.get("RMSE", None)
    
    with col1:
        if mae is not None:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} MW")
        else:
            st.metric("Mean Absolute Error (MAE)", "—")
    
    with col2:
        if mape is not None:
            st.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
        else:
            st.metric("Mean Absolute % Error (MAPE)", "—")
    
    with col3:
        if rmse is not None:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} MW")
        else:
            st.metric("Root Mean Squared Error (RMSE)", "—")
    
    # Display status and data info
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("2025 Records", f"{metrics_data.get('total_records', 0)}")
    with col_b:
        st.metric("Forecast Records Matched", f"{metrics_data.get('forecast_records', 0)}")
    
    # Display hourly errors table if available
    hourly_errors = metrics_data.get("hourly_errors", {})
    if hourly_errors:
        st.subheader("Hourly Error Breakdown")
        hourly_df = pd.DataFrame([
            {
                "Hour": int(hour.replace("H", "")),
                "Mean Error (MW)": v.get("mean_error", 0),
                "MAE (MW)": v.get("mae", 0),
                "MAPE (%)": v.get("mape", 0),
                "Count": v.get("count", 0),
            }
            for hour, v in sorted(hourly_errors.items())
        ])
        hourly_df = hourly_df.sort_values("Hour")
        st.dataframe(hourly_df.round(2), use_container_width=True, hide_index=True)
    
    # Display status info
    st.caption(
        f"Forecast status: {metrics_data.get('forecast_status', 'unknown')} | "
        f"Last trained: {metrics_data.get('forecast_trained_at', 'N/A')}"
    )


def render_today_vs_forecast(df, scope_label, baseline=None, include_today_in_training=False):
    """
    Preserve RTEDD's original two logical modes:
    - include_today_in_training=False: benchmark comparison
    - include_today_in_training=True: live-trained forecast comparison
    Only the presentation is redesigned.
    """
    latest_date = df["Date"].max()
    today_df = (
        df[df["Date"] == latest_date]
        .groupby("Hour", as_index=False)["Ontario Demand"].mean()
        .rename(columns={"Ontario Demand": "Today"})
    )

    historical_df = df[df["Date"] < latest_date].copy()
    comparison_source = df if include_today_in_training else (
        historical_df if not historical_df.empty else df
    )
    avg_df = (
        comparison_source.groupby("Hour", as_index=False)["Ontario Demand"].mean()
        .rename(columns={"Ontario Demand": "Average"})
    )

    forecast_df = compute_ensemble_forecast(
        df,
        latest_date,
        include_target_date=include_today_in_training,
    )

    merged = pd.merge(today_df, avg_df, on="Hour", how="outer").sort_values("Hour")
    if not forecast_df.empty:
        merged = pd.merge(merged, forecast_df, on="Hour", how="outer")

    baseline_df = baseline if baseline is not None else pd.DataFrame()
    if not baseline_df.empty:
        expected = baseline_df[["Hour", "Expected"]].rename(
            columns={"Expected": "Expected Median"}
        )
        merged = pd.merge(merged, expected, on="Hour", how="outer")

    fig = go.Figure()

    # Benchmark view: compare actual demand against the prior-day trained model outputs
    # and the operating baseline, while keeping the historical average visible.
    if not include_today_in_training:
        if not baseline_df.empty:
            band = baseline_df.sort_values("Hour")
            fig.add_trace(go.Scatter(
                x=band["Hour"], y=band["Upper"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=band["Hour"], y=band["Lower"], mode="lines",
                fill="tonexty", fillcolor="rgba(79,156,255,0.10)",
                line=dict(width=0), name="Expected operating range",
                hovertemplate="Hour %{x}:00<br>Lower: %{y:,.0f} MW<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=band["Hour"], y=band["Expected"], mode="lines",
                line=dict(width=2, dash="dot", color="#94a3b8"),
                name="Expected benchmark",
            ))

        if {"Ensemble_P10", "Ensemble_P90"}.issubset(merged.columns):
            band = merged[["Hour", "Ensemble_P10", "Ensemble_P90"]].dropna().sort_values("Hour")
            if not band.empty:
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Ensemble_P90"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Ensemble_P10"], mode="lines",
                    fill="tonexty", fillcolor="rgba(84,216,139,0.13)",
                    line=dict(width=0), name="Ensemble P10–P90 range",
                ))

        if "Average" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["Average"], mode="lines",
                line=dict(width=1.5, dash="dash", color="#6b7c93"),
                name="Historical hourly average",
            ))

        if "Prophet" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["Prophet"], mode="lines",
                line=dict(width=2, dash="dash", color="#f4b942"),
                name="Prophet",
            ))
        if "LightGBM" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["LightGBM"], mode="lines",
                line=dict(width=2, dash="dot", color="#b17cff"),
                name="LightGBM",
            ))
        if "Ensemble" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["Ensemble"], mode="lines",
                line=dict(width=3, color="#54d88b"),
                name="Ensemble P50",
            ))

        fig.add_trace(go.Scatter(
            x=merged["Hour"], y=merged["Today"], mode="lines+markers",
            line=dict(width=4, color="#4f9cff"),
            marker=dict(size=6), name="Today's actual demand",
        ))
        title = f"Benchmark comparison · {latest_date.date()}"

    # Live-trained view: model outputs are generated using all data available up to the
    # latest observed hour for the selected target date.
    else:
        if {"Ensemble_P10", "Ensemble_P90"}.issubset(merged.columns):
            band = merged[["Hour", "Ensemble_P10", "Ensemble_P90"]].dropna().sort_values("Hour")
            if not band.empty:
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Ensemble_P90"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Ensemble_P10"], mode="lines",
                    fill="tonexty", fillcolor="rgba(84,216,139,0.13)",
                    line=dict(width=0), name="Ensemble P10–P90 range",
                ))

        if "Prophet" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["Prophet"], mode="lines",
                line=dict(width=2, dash="dash", color="#f4b942"),
                name="Prophet",
            ))
        if "LightGBM" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["LightGBM"], mode="lines",
                line=dict(width=2, dash="dot", color="#b17cff"),
                name="LightGBM",
            ))
        if "Ensemble" in merged:
            fig.add_trace(go.Scatter(
                x=merged["Hour"], y=merged["Ensemble"], mode="lines",
                line=dict(width=3, color="#54d88b"),
                name="Ensemble P50",
            ))

        fig.add_trace(go.Scatter(
            x=merged["Hour"], y=merged["Today"], mode="lines+markers",
            line=dict(width=4, color="#4f9cff"),
            marker=dict(size=6), name="Today's actual demand",
        ))
        title = f"Live-trained model comparison · {latest_date.date()}"

    # Keep anomaly logic in both views.
    anomalies = df[(df["Date"] == latest_date) & (df["Anomaly"])].copy()
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["Hour"], y=anomalies["Ontario Demand"],
            mode="markers", marker=dict(size=11, color="#ff6868", symbol="diamond"),
            name="Detected anomaly",
            customdata=anomalies[["Expected Demand", "Deviation", "Anomaly Score"]].to_numpy(),
            hovertemplate=(
                "Hour: %{x}:00<br>Demand: %{y:,.0f} MW<br>"
                "Expected: %{customdata[0]:,.0f} MW<br>"
                "Deviation: %{customdata[1]:,.0f} MW<br>"
                "Score: %{customdata[2]:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=20, r=20, t=55, b=25),
        hovermode="x unified",
        paper_bgcolor="#0b1828",
        plot_bgcolor="#0b1828",
        font=dict(color="#dbe7f3"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(
        title="Hour of day",
        tickmode="array",
        tickvals=list(range(1, 25)),
        gridcolor="rgba(120,150,180,0.12)",
        zeroline=False,
    )
    fig.update_yaxes(
        title="Demand (MW)",
        gridcolor="rgba(120,150,180,0.12)",
        zeroline=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy stays available but does not clutter the chart.
    rows = []
    for method in forecast_methods_for_view(include_today_in_training):
        if method not in merged.columns:
            continue
        metrics = calculate_error_metrics(merged["Today"], merged[method])
        accuracy = calculate_accuracy_percentage(merged["Today"], merged[method])
        if metrics is not None:
            rows.append({
                "Method": "Ensemble P50" if method == "Ensemble" else method,
                "Accuracy %": accuracy,
                **metrics,
            })

    if rows:
        accuracy_df = pd.DataFrame(rows)
        accuracy_df["Accuracy %"] = accuracy_df["Accuracy %"].map(
            lambda x: "N/A" if x is None else f"{x:.1f}%"
        )
        with st.expander("View accuracy and error metrics"):
            st.dataframe(
                accuracy_df.round({"MAE (MW)": 1, "RMSE (MW)": 1, "MAPE %": 2}),
                use_container_width=True,
                hide_index=True,
            )


def render_forecast_training_comparison(df, scope_label, baseline=None):
    """Keep RTEDD's two forecast-training views separate and visually distinct."""
    latest_date = df["Date"].max()

    st.markdown(
        '<div class="eyebrow">Two complementary forecast views</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The benchmark view answers whether today's demand follows the expected operating pattern. "
        "The live-trained view evaluates the current Prophet + LightGBM forecasting pipeline."
    )

    # 1) Stable benchmark comparison.
    st.markdown("### 1. Today vs Forecast Benchmark")
    st.caption("Actual demand against the historical/expected benchmark. This is the operational baseline view.")
    render_today_vs_forecast(
        df,
        scope_label,
        baseline=baseline,
        include_today_in_training=False,
    )

    st.divider()

    # 2) Live-trained model comparison.
    st.markdown("### 2. Today vs Live-Trained Forecast")
    st.caption(
        "Actual demand against forecasts generated by the live training pipeline. "
        "Prophet, LightGBM and Ensemble remain visible as separate RTEDD model outputs."
    )
    render_today_vs_forecast(
        df,
        scope_label,
        baseline=baseline,
        include_today_in_training=True,
    )


def render_latest_7_days(df, scope_label):
    dates = sorted(df["Date"].dt.normalize().dropna().unique())[-7:]
    recent_df = df[df["Date"].isin(dates)]
    if recent_df.empty:
        st.info("Not enough recent data yet")
        return
    fig = px.line(
        recent_df,
        x="Hour",
        y="Ontario Demand",
        color="Date Label",
        title=f"Latest 7 Dates | {scope_label}",
        markers=True,
    )
    fig = fix_hour_axis(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_latest_records(df, scope_label):
    recent = df.tail(50).copy()
    if recent.empty:
        st.info("No recent records available")
        return
    recent["Label"] = recent["Timestamp"].dt.strftime("%Y-%m-%d %H:00")
    fig = px.bar(
        recent,
        x="Label",
        y="Ontario Demand",
        color="Status",
        title=f"Latest Records | {scope_label}",
        hover_data=["Hour", "Deviation", "Anomaly Score"],
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_details(df, scope_label):
    anomaly_df = df[df["Anomaly"]].copy()
    st.subheader(f"Anomaly Details - {scope_label}")

    if anomaly_df.empty:
        st.success("System normal - no anomaly markers were detected")
        return

    anomaly_df["Label"] = anomaly_df["Timestamp"].dt.strftime("%Y-%m-%d %H:00")
    st.dataframe(
        anomaly_df[
            [
                "Label",
                "Hour",
                "Ontario Demand",
                "Expected Demand",
                "Deviation",
                "Anomaly Score",
                "Status",
            ]
        ],
        use_container_width=True,
    )


def render_current_demand_card(df):
    """Show current actual demand plus RTEDD's Prophet, LightGBM and Ensemble outputs."""
    if df.empty:
        return

    latest_date = df["Date"].max()
    latest_rows = df[df["Date"] == latest_date].sort_values("Hour")
    latest = latest_rows.iloc[-1]
    current_hour = int(latest["Hour"])
    actual = float(latest["Ontario Demand"])

    forecast_df = compute_ensemble_forecast(
        df, latest_date, include_target_date=True
    )
    row = (
        forecast_df[forecast_df["Hour"] == current_hour]
        if forecast_df is not None and not forecast_df.empty
        else pd.DataFrame()
    )

    model_lines = []
    if not row.empty:
        item = row.iloc[0]
        for label, column in [
            ("Prophet", "Prophet"),
            ("LightGBM", "LightGBM"),
            ("Ensemble", "Ensemble"),
        ]:
            value = item.get(column)
            if pd.notna(value):
                model_lines.append(f"{label}: {float(value):,.0f} MW")

    expected = float(latest.get("Expected Demand", actual))
    deviation_pct = ((actual - expected) / expected * 100) if expected else 0
    details = "<br>".join(model_lines) if model_lines else f"{deviation_pct:+.1f}% vs expected"

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="label">CURRENT DEMAND · {current_hour:02d}:00</div>'
        f'<div class="value">{actual:,.0f} MW</div>'
        f'<div class="models">{details}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def render_next_hour_forecast_card(df):
    """Render RTEDD's next-hour forecast using both model outputs and ensemble when available."""
    if df.empty:
        st.markdown(
            '<div class="forecast-card"><div class="label">NEXT HOUR FORECAST</div>'
            '<div class="value">—</div><div class="models">No data available.</div></div>',
            unsafe_allow_html=True,
        )
        return

    latest_date = df["Date"].max()
    latest_rows = df[df["Date"] == latest_date]
    latest_hour = int(latest_rows["Hour"].max()) if not latest_rows.empty else 24
    next_hour = 1 if latest_hour >= 24 else latest_hour + 1

    forecast_df = compute_ensemble_forecast(
        df, latest_date, include_target_date=True
    )

    row = (
        forecast_df[forecast_df["Hour"] == next_hour]
        if forecast_df is not None and not forecast_df.empty
        else pd.DataFrame()
    )

    if row.empty:
        st.markdown(
            f'<div class="forecast-card"><div class="label">NEXT HOUR FORECAST · {next_hour:02d}:00</div>'
            '<div class="value">—</div>'
            '<div class="models">Live model output is not available for this hour.</div></div>',
            unsafe_allow_html=True,
        )
        return

    item = row.iloc[0]
    prophet = item.get("Prophet")
    lgbm = item.get("LightGBM")
    ensemble = item.get("Ensemble")
    p10 = item.get("Ensemble_P10")
    p90 = item.get("Ensemble_P90")

    values = [v for v in [ensemble, lgbm, prophet] if pd.notna(v)]
    if not values:
        main_html = "—"
    else:
        main_html = f"{float(values[0]):,.0f} MW"

    model_lines = []
    if pd.notna(prophet):
        model_lines.append(f"Prophet: {float(prophet):,.0f} MW")
    if pd.notna(lgbm):
        model_lines.append(f"LightGBM: {float(lgbm):,.0f} MW")
    if pd.notna(p10) and pd.notna(p90):
        model_lines.append(f"Expected range: {float(p10):,.0f}–{float(p90):,.0f} MW")

    details = "<br>".join(model_lines) if model_lines else "No model values available."

    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="label">NEXT HOUR FORECAST · {next_hour:02d}:00</div>'
        f'<div class="value">{main_html}</div>'
        f'<div class="models">{details}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )




def render_system_health_panel(latest_date, latest_hour, total_records_received):
    st.markdown("### SYSTEM HEALTH")
    st.markdown(
        f"""
        🟢 **Data Pipeline**  
        Receiving demand records

        🟢 **Forecast Engine**  
        Prophet + LightGBM available

        🟢 **API Connection**  
        Connected

        🟢 **Latest Data**  
        {latest_date.date()} · {int(latest_hour):02d}:00

        📍 **Timezone**  
        Ontario Time (ET)

        **Total records:** {total_records_received:,}
        """
    )


def render_recent_intelligence(df):
    st.markdown("### 🚨 RECENT INTELLIGENCE")
    if "Anomaly" not in df.columns or not df["Anomaly"].any():
        st.success("🟢 System Normal\n\nAll selected observations are within the expected operating pattern.")
        return

    events = df[df["Anomaly"]].sort_values(["Date", "Hour"], ascending=False).head(4)
    for _, event in events.iterrows():
        demand = float(event["Ontario Demand"])
        expected = float(event.get("Expected Demand", demand))
        deviation = demand - expected
        pct = (deviation / expected * 100) if expected else 0
        score = float(event.get("Anomaly Score", 0))
        severity = "🔴 High" if abs(score) >= 3 else "🟠 Medium"

        direction = "higher" if deviation >= 0 else "lower"
        st.markdown(
            f"**{severity} demand deviation**  \n"
            f"{event['Date'].date()} · {int(event['Hour']):02d}:00 — "
            f"Demand is **{abs(pct):.1f}% {direction}** than expected."
        )
        st.caption(f"Actual {demand:,.0f} MW · Expected {expected:,.0f} MW")
        st.divider()


def render_supporting_metrics(df):
    values = df["Ontario Demand"].dropna()
    if values.empty:
        return

    avg = float(values.mean())
    peak_idx = values.idxmax()
    low_idx = values.idxmin()
    peak_row = df.loc[peak_idx]
    low_row = df.loc[low_idx]

    # Hourly MW readings are treated as hourly demand observations for an approximate
    # GWh indicator; the label intentionally says estimated.
    estimated_gwh = float(values.sum() / 1000.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AVERAGE DEMAND", f"{avg:,.0f} MW")
    c2.metric("PEAK DEMAND", f"{float(peak_row['Ontario Demand']):,.0f} MW",
              f"{peak_row['Date'].date()} · {int(peak_row['Hour']):02d}:00")
    c3.metric("LOW DEMAND", f"{float(low_row['Ontario Demand']):,.0f} MW",
              f"{low_row['Date'].date()} · {int(low_row['Hour']):02d}:00")
    c4.metric("ESTIMATED DEMAND TOTAL", f"{estimated_gwh:,.1f} GWh",
              "Selected period")


def render_seven_day_trend(df):
    daily = (
        df.groupby("Date", as_index=False)["Ontario Demand"]
        .mean()
        .tail(7)
    )
    if daily.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Ontario Demand"],
        mode="lines+markers", name="Average daily demand",
        line=dict(width=3, color="#4f9cff"),
        fill="tozeroy", fillcolor="rgba(79,156,255,0.10)",
    ))
    fig.update_layout(
        height=310, margin=dict(l=20, r=20, t=45, b=20),
        title="7-Day Demand Trend",
        paper_bgcolor="#0b1828", plot_bgcolor="#0b1828",
        font=dict(color="#dbe7f3"), showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(120,150,180,0.12)")
    fig.update_yaxes(title="Average demand (MW)", gridcolor="rgba(120,150,180,0.12)")
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard_content(view, scope, date_range, hour_range, show_normal_rows):
    df, baseline, total_records_received = fetch_dashboard_data()

    if df.empty:
        st.markdown('<div class="brand-card"><h1>GridPulse</h1><p>Energy intelligence is waiting for live demand data.</p></div>', unsafe_allow_html=True)
        st.info("Waiting for the data pipeline to deliver records.")
        return

    df_view = apply_scope_and_filters(df, scope, date_range, hour_range)
    if df_view.empty:
        st.warning("No records match the current view.")
        return

    latest_date = df_view["Date"].max()
    latest_rows = df_view[df_view["Date"] == latest_date].sort_values("Hour")
    latest = latest_rows.iloc[-1]
    current = float(latest["Ontario Demand"])
    expected = float(latest.get("Expected Demand", current))
    deviation_pct = ((current - expected) / expected * 100) if expected else 0
    anomaly_count = int(df_view["Anomaly"].sum()) if "Anomaly" in df_view else 0
    status = "ATTENTION" if anomaly_count else "NORMAL"
    status_icon = "🟠" if anomaly_count else "🟢"

    st.markdown(
        f"""
        <div class="brand-card">
            <h1>⚡ RTEDD <span style="font-weight:400;opacity:.78;font-size:1.05rem;">Real-Time Electricity Demand Intelligence</span></h1>
            <p>{status_icon} System status: <b>{status}</b> &nbsp;•&nbsp; Latest observation: {latest_date.date()} at hour {int(latest['Hour'])}:00</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if view == "Command Center":
        # Product hierarchy: status → key numbers → main activity → problems → supporting analysis.
        st.markdown('<div class="eyebrow">Live system snapshot</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        peak = float(latest_rows["Ontario Demand"].max())
        peak_hour = int(latest_rows.loc[latest_rows["Ontario Demand"].idxmax(), "Hour"])

        with c1:
            render_current_demand_card(df_view)
        with c2:
            render_next_hour_forecast_card(df_view)
        c3.markdown(
            f'<div class="insight-card"><div class="label">TODAY’S PEAK</div>'
            f'<div class="value">{peak:,.0f} MW</div>'
            f'<div class="sub">Observed at {peak_hour:02d}:00</div></div>',
            unsafe_allow_html=True,
        )

        # Keep alert severity honest: only classify from anomaly score when it exists.
        high_alerts = 0
        medium_alerts = anomaly_count
        if "Anomaly Score" in df_view.columns:
            scores = df_view.loc[df_view["Anomaly"], "Anomaly Score"].abs()
            high_alerts = int((scores >= 3).sum())
            medium_alerts = int((scores < 3).sum())

        c4.markdown(
            f'<div class="insight-card"><div class="label">ACTIVE ALERTS</div>'
            f'<div class="value">{anomaly_count}</div>'
            f'<div class="sub">🔴 {high_alerts} High · 🟠 {medium_alerts} Medium</div></div>',
            unsafe_allow_html=True,
        )

        # Main intelligence row: signature chart + system trust panel.
        main_left, main_right = st.columns([2.2, 1])
        with main_left:
            st.markdown('<div class="eyebrow">Demand intelligence</div>', unsafe_allow_html=True)
            fig = go.Figure()

            # Expected operating band.
            if baseline is not None and not baseline.empty:
                band = baseline.sort_values("Hour")
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Upper"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=band["Hour"], y=band["Lower"], mode="lines",
                    fill="tonexty", fillcolor="rgba(79,156,255,0.10)",
                    line=dict(width=0), name="Expected range"
                ))
                if "Expected" in band.columns:
                    fig.add_trace(go.Scatter(
                        x=band["Hour"], y=band["Expected"], mode="lines",
                        line=dict(width=2, dash="dot", color="#94a3b8"),
                        name="Expected demand"
                    ))

            # Actual and model forecasts for today's available hours.
            fig.add_trace(go.Scatter(
                x=latest_rows["Hour"], y=latest_rows["Ontario Demand"],
                mode="lines+markers", name="Actual demand",
                line=dict(width=4, color="#4f9cff"), marker=dict(size=6)
            ))

            forecast_today = compute_ensemble_forecast(
                df_view, latest_date, include_target_date=True
            )
            if forecast_today is not None and not forecast_today.empty:
                if "Ensemble" in forecast_today.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_today["Hour"], y=forecast_today["Ensemble"],
                        mode="lines", name="Ensemble forecast",
                        line=dict(width=3, color="#54d88b")
                    ))
                if {"Ensemble_P10", "Ensemble_P90"}.issubset(forecast_today.columns):
                    band = forecast_today.dropna(subset=["Ensemble_P10", "Ensemble_P90"]).sort_values("Hour")
                    if not band.empty:
                        fig.add_trace(go.Scatter(
                            x=band["Hour"], y=band["Ensemble_P90"], mode="lines",
                            line=dict(width=0), showlegend=False, hoverinfo="skip"
                        ))
                        fig.add_trace(go.Scatter(
                            x=band["Hour"], y=band["Ensemble_P10"], mode="lines",
                            fill="tonexty", fillcolor="rgba(84,216,139,0.10)",
                            line=dict(width=0), name="Ensemble P10–P90"
                        ))

            anomalies = latest_rows[latest_rows["Anomaly"]] if "Anomaly" in latest_rows else pd.DataFrame()
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies["Hour"], y=anomalies["Ontario Demand"],
                    mode="markers", name="Detected anomaly",
                    marker=dict(size=12, color="#ff6868", symbol="diamond")
                ))

            fig.update_layout(
                height=470, margin=dict(l=20, r=20, t=50, b=20),
                title="Actual demand · Expected pattern · Live forecast",
                hovermode="x unified",
                paper_bgcolor="#0b1828", plot_bgcolor="#0b1828",
                font=dict(color="#dbe7f3"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            fig.update_xaxes(
                title="Hour of day", tickmode="array", tickvals=list(range(1, 25)),
                gridcolor="rgba(120,150,180,0.12)"
            )
            fig.update_yaxes(title="Demand (MW)", gridcolor="rgba(120,150,180,0.12)")
            st.plotly_chart(fig, use_container_width=True)

        with main_right:
            st.markdown('<div class="eyebrow">Operational trust</div>', unsafe_allow_html=True)
            render_system_health_panel(
                latest_date, latest["Hour"], total_records_received
            )

        # Problems and trend are intentionally separated.
        lower_left, lower_right = st.columns([1.25, .95])
        with lower_left:
            st.markdown('<div class="eyebrow">Historical movement</div>', unsafe_allow_html=True)
            render_seven_day_trend(df_view)
        with lower_right:
            st.markdown('<div class="eyebrow">What needs attention?</div>', unsafe_allow_html=True)
            render_recent_intelligence(df_view)

        st.markdown('<div class="eyebrow">Supporting intelligence</div>', unsafe_allow_html=True)
        render_supporting_metrics(df_view)

    elif view == "Forecast Studio":
        st.markdown('<div class="eyebrow">Forecast outlook</div>', unsafe_allow_html=True)
        render_forecast_training_comparison(df_view, build_scope_label(df_view, scope, hour_range), baseline=baseline)
        st.divider()
        st.markdown('<div class="eyebrow">Model evaluation</div>', unsafe_allow_html=True)
        render_2025_metrics()

    elif view == "Alert Center":
        st.markdown('<div class="eyebrow">Operational alerts</div>', unsafe_allow_html=True)
        if anomaly_count == 0:
            st.success("No anomalies detected for the selected period.")
        else:
            st.warning(f"{anomaly_count} events are outside the expected operating pattern.")
        render_anomaly_details(df_view, build_scope_label(df_view, scope, hour_range))

    elif view == "🤖 Agentic Analyst":
        render_rtedd_agentic_analyst(df=df_view)

    else:
        st.markdown('<div class="eyebrow">Data explorer</div>', unsafe_allow_html=True)
        st.caption("Use this workspace for validation and detailed analysis—not as the primary operational view.")
        st.dataframe(
            df_view[["Date","Hour","Ontario Demand","Expected Demand","Deviation","Anomaly Score","Status"]].sort_values(["Date","Hour"], ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download selected data",
            df_view.to_csv(index=False).encode("utf-8"),
            "gridpulse_export.csv",
            "text/csv",
            key="gridpulse_export",
        )


ensure_state()

if st.session_state.dashboard_df.empty:
    fetch_dashboard_data()

df_sidebar = st.session_state.dashboard_df.copy()
(
    view,
    refresh_seconds,
    auto_refresh_enabled,
    scope,
    date_range,
    hour_range,
    show_normal_rows,
) = sidebar_controls(df_sidebar)

if hasattr(st, "fragment"):
    @st.fragment(run_every=refresh_seconds if auto_refresh_enabled else None)
    def live_dashboard():
        render_dashboard_content(view, scope, date_range, hour_range, show_normal_rows)


    live_dashboard()
else:
    render_dashboard_content(view, scope, date_range, hour_range, show_normal_rows)
    if auto_refresh_enabled:
        time.sleep(refresh_seconds)
        st.rerun()
