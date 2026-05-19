import os
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
BASE_URL = f"http://{SERVER_IP}:8000"
RECORD_COUNT_URL = f"{BASE_URL}/records/count"
DASHBOARD_DATA_URL = f"{BASE_URL}/dashboard/data"
FORECAST_URL = f"{BASE_URL}/forecast/latest"
FORECAST_REFRESH_URL = f"{BASE_URL}/forecast/refresh"
HOURS = list(range(24))
FORECAST_POLL_SECONDS = 15
FORECAST_FRESH_POLL_SECONDS = 60


st.set_page_config(
    page_title="Real-Time Electricity Demand Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="staleElement"],
    [data-testid="stale-element"],
    .stale-element,
    .stApp [style*="opacity: 0"],
    .stApp [style*="opacity:0"],
    .stApp [style*="opacity: 0."],
    .stApp [style*="opacity:0."] {
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Real-Time Electricity Demand Dashboard")
st.caption(f"Connected to {BASE_URL}")


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
        st.session_state.hour_range = (0, 23)
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
            "XGBoost",
            "Ensemble",
            "Ensemble_P10",
            "Ensemble_P50",
            "Ensemble_P90",
        ]
    )


def forecast_frame_from_rows(forecast_rows):
    if not forecast_rows:
        return empty_forecast_frame()

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_columns = [
        "Hour",
        "Prophet",
        "XGBoost",
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


def forecast_status_text():
    status = st.session_state.forecast_status
    trained_at = st.session_state.forecast_trained_at
    training_seconds = st.session_state.forecast_training_seconds
    message = st.session_state.forecast_message
    summary = st.session_state.forecast_summary

    parts = []
    if status:
        parts.append(f"Forecast status: {status}")
    if status == "training" and training_seconds is not None:
        parts.append(f"Training for {float(training_seconds):.0f}s")
    if trained_at:
        trained_time = pd.to_datetime(float(trained_at), unit="s").strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"last trained {trained_time}")
    if message:
        parts.append(message)

    status_line = " | ".join(parts)
    if summary:
        return f"{status_line}\n{summary}"
    return status_line


def compute_ensemble_forecast(df, target_date=None, include_target_date=False):
    """
    Fetch the latest cached Prophet/XGBoost ensemble forecast from FastAPI.
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
    st.sidebar.header("Controls")
    refresh_seconds = st.sidebar.slider(
        "Dashboard refresh interval (seconds)",
        1,
        10,
        key="refresh_seconds",
    )
    auto_refresh_enabled = st.sidebar.checkbox(
        "Auto refresh",
        key="auto_refresh_enabled",
    )
    if st.sidebar.button("Refresh now", key="refresh_now"):
        st.rerun()
    if st.sidebar.button("Refresh forecast", key="refresh_forecast"):
        request_forecast_refresh()

    st.sidebar.subheader("Scope")
    scope = st.sidebar.selectbox(
        "Data scope",
        ["All data", "Today", "Last 7 days", "Custom date range"],
        key="scope",
    )

    if not df.empty:
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
    else:
        today = pd.Timestamp.today().date()
        min_date, max_date = today, today

    date_range = clamp_date_range(
        st.session_state.date_range,
        min_date,
        max_date,
    )
    st.session_state.date_range = date_range

    if scope == "Custom date range":
        selected_date_range = st.sidebar.date_input(
            "Date range",
            value=date_range,
            min_value=min_date,
            max_value=max_date,
            key="date_range_input",
        )
        date_range = clamp_date_range(selected_date_range, min_date, max_date)
        st.session_state.date_range = date_range

    st.sidebar.subheader("Filters")
    hour_range = st.sidebar.slider("Hour range", 0, 23, key="hour_range")
    show_normal_rows = st.sidebar.checkbox("Show normal rows", key="show_normal_rows")

    st.sidebar.caption(f"Loaded records: {len(df)}")
    st.sidebar.caption("Connection: Server-backed")

    if st.session_state.last_error:
        st.sidebar.warning(st.session_state.last_error)

    if not df.empty:
        latest_date = df["Date"].max().date()
        earliest_date = df["Date"].min().date()
        st.sidebar.caption(f"Available dates: {earliest_date} to {latest_date}")

    if st.session_state.last_received_epoch is None:
        st.sidebar.caption("Last update: N/A")
    else:
        age_s = max(0.0, time.time() - st.session_state.last_received_epoch)
        st.sidebar.caption(f"Last update: {age_s:.0f}s ago")

    st.sidebar.subheader("View")
    view = st.sidebar.selectbox(
        "View Mode",
        [
            "Today",
            "All Dates",
            "Average",
            "Forecast Training Comparison",
            "Latest 7 Days",
            "Latest Records",
        ],
        key="view_mode",
    )

    return (
        view,
        refresh_seconds,
        auto_refresh_enabled,
        scope,
        date_range,
        hour_range,
        show_normal_rows,
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


def render_metrics(df, total_records_received=None):
    peak = df["Ontario Demand"].max()
    avg = df["Ontario Demand"].mean()
    if total_records_received is None:
        total_records_received = len(df)

    cols = st.columns(3)
    cols[0].metric("Peak Demand", f"{peak:.0f} MW" if pd.notna(peak) else "N/A")
    cols[1].metric("Avg Demand", f"{avg:.0f} MW" if pd.notna(avg) else "N/A")
    cols[2].metric("Total Records", f"{total_records_received}")

    if df["Anomaly"].any():
        st.warning(f"{int(df['Anomaly'].sum())} anomalies detected")
    else:
        st.success("System normal")


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


def style_today_trace(fig):
    for trace in fig.data:
        line_width = 5 if getattr(trace, "name", None) == "Today" else 2
        marker_size = 9 if getattr(trace, "name", None) == "Today" else None
        if marker_size is None:
            trace.update(line={"width": line_width})
        else:
            trace.update(line={"width": line_width}, marker={"size": marker_size})


def render_today_vs_forecast(df, scope_label, baseline=None, include_today_in_training=False):
    latest_date = df["Date"].max()
    today_df = df[df["Date"] == latest_date].groupby("Hour", as_index=False)["Ontario Demand"].mean()
    if include_today_in_training:
        comparison_source = df
        title_prefix = "Today vs Live-Trained Forecast"
    else:
        historical_df = df[df["Date"] < latest_date].copy()
        comparison_source = historical_df if not historical_df.empty else df
        title_prefix = "Today vs Forecast Benchmarks"

    avg_df = comparison_source.groupby("Hour", as_index=False)["Ontario Demand"].mean()
    today_df = today_df.rename(columns={"Ontario Demand": "Today"})
    avg_df = avg_df.rename(columns={"Ontario Demand": "Average"})

    # Get Prophet, XGBoost, and Ensemble forecasts
    forecast_df = compute_ensemble_forecast(
        df,
        latest_date,
        include_target_date=include_today_in_training,
    )
    status_text = forecast_status_text()
    if status_text:
        st.caption(status_text)
    if forecast_df.empty and st.session_state.forecast_status in {"training", "stale"}:
        st.info("Forecast training is running in FastAPI. Cached forecast data will appear after it finishes.")

    # Merge all forecasts
    merged = pd.merge(today_df, avg_df, on="Hour", how="outer").sort_values("Hour")

    if not forecast_df.empty:
        merged = pd.merge(merged, forecast_df, on="Hour", how="outer")

    baseline_df = baseline if baseline is not None else pd.DataFrame()
    if not baseline_df.empty:
        expected_median = baseline_df[["Hour", "Expected"]].rename(
            columns={"Expected": "Expected Median"}
        )
        merged = pd.merge(merged, expected_median, on="Hour", how="outer")

    # Melt for plotting
    if not forecast_df.empty:
        value_vars = [
            "Today",
            "Average",
            "Prophet",
            "XGBoost",
            "Ensemble",
        ]
        value_vars = [v for v in value_vars if v in merged.columns]
    else:
        value_vars = ["Today", "Average"]
        value_vars = [v for v in value_vars if v in merged.columns]

    melted = merged.melt(id_vars="Hour", value_vars=value_vars, var_name="Series", value_name="Demand")

    # Define colors for each series
    color_map = {
        "Today": "#1f77b4",
        "Average": "#7f7f7f",
        "Prophet": "#ff7f0e",
        "XGBoost": "#2ca02c",
        "Ensemble": "#d62728",
        "Expected Median": "#4d4d4d",
    }

    fig = px.line(
        melted,
        x="Hour",
        y="Demand",
        color="Series",
        color_discrete_map=color_map,
        title=f"{title_prefix} - {latest_date.date()} | {scope_label}",
        markers=True,
    )

    style_today_trace(fig)

    # Add baseline band if available
    fig = add_baseline_to_figure(fig, baseline_df, sorted(merged["Hour"].dropna().unique()))

    # Highlight anomalies for the latest date
    anomalies = df[(df["Date"] == latest_date) & (df["Anomaly"])].copy()
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["Hour"],
                y=anomalies["Ontario Demand"],
                mode="markers",
                marker=dict(size=11, color="#d62728", symbol="x"),
                name="Anomaly",
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

    fig = fix_hour_axis(fig)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Today vs Expected Accuracy")
    methods = ["Average", "Prophet", "XGBoost", "Ensemble", "Expected Median"]
    accuracy_rows: list[dict[str, Any]] = []
    for method in methods:
        if method not in merged.columns:
            continue

        comparison = merged[["Today", method]].dropna()
        if comparison.empty:
            continue

        mae = (comparison["Today"] - comparison[method]).abs().mean()
        accuracy = calculate_accuracy_percentage(comparison["Today"], comparison[method])
        accuracy_rows.append(
            {
                "Method": method,
                "Accuracy %": accuracy,
                "MAE (MW)": mae,
                "Compared Hours": len(comparison),
            }
        )

    if accuracy_rows:
        accuracy_df = pd.DataFrame(accuracy_rows)
        accuracy_df["Accuracy %"] = accuracy_df["Accuracy %"].map(
            lambda value: "N/A" if value is None else f"{value:.1f}%"
        )
        accuracy_df["MAE (MW)"] = accuracy_df["MAE (MW)"].round(1)
        st.dataframe(accuracy_df, use_container_width=True, hide_index=True)

    comparison_columns = ["Hour"] + [column for column in methods if column in merged.columns]
    if len(comparison_columns) > 1:
        st.subheader("Expected Demand Comparison")
        st.dataframe(merged[comparison_columns].round(1), use_container_width=True)


def render_forecast_training_comparison(df, scope_label, baseline=None):
    left_col, right_col = st.columns(2)

    with left_col:
        render_today_vs_forecast(
            df,
            scope_label,
            baseline=baseline,
            include_today_in_training=False,
        )

    with right_col:
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


def render_chart(df, view_mode, baseline, scope_label):
    if view_mode == "Today":
        render_today(df, scope_label, baseline=baseline)
    elif view_mode == "All Dates":
        render_all_dates(df, scope_label)
    elif view_mode == "Average":
        render_average(df, scope_label)
    elif view_mode == "Forecast Training Comparison":
        render_forecast_training_comparison(df, scope_label, baseline=baseline)
    elif view_mode == "Latest 7 Days":
        render_latest_7_days(df, scope_label)
    elif view_mode == "Latest Records":
        render_latest_records(df, scope_label)


def render_dashboard_content(view, scope, date_range, hour_range, show_normal_rows):
    df, baseline, total_records_received = fetch_dashboard_data()

    if df.empty:
        st.info("Waiting for data from the server...")
        return

    df_view = apply_scope_and_filters(df, scope, date_range, hour_range)
    scope_label = build_scope_label(df_view, scope, hour_range)

    if df_view.empty:
        st.warning(f"No data matches the selected scope/filters. Active view: {scope_label}")
    else:
        render_metrics(df_view, total_records_received=total_records_received)
        st.caption(f"Active scope: {scope_label}")
        render_chart(df_view, view, baseline, scope_label)

        st.divider()
        render_anomaly_details(df_view, scope_label)

    df_table = df_view
    if not show_normal_rows:
        df_table = df_table[df_table["Anomaly"]]

    table_scope_label = build_scope_label(df_view, scope, hour_range)
    st.subheader(f"Latest Records - {table_scope_label}")

    if df_table.empty:
        if show_normal_rows:
            st.info(f"No records available for {table_scope_label}.")
        else:
            st.info(f"No anomaly rows available for {table_scope_label}. Turn on 'Show normal rows' to see all records.")

    st.dataframe(
        df_table[
            [
                "Date",
                "Hour",
                "Ontario Demand",
                "Expected Demand",
                "Deviation",
                "Anomaly Score",
                "Status",
            ]
        ].tail(25),
        use_container_width=True,
    )

    st.download_button(
        f"Download current view ({scope_label})",
        data=df_view.to_csv(index=False).encode("utf-8"),
        file_name="pub_dashboard_view.csv",
        mime="text/csv",
        key="download_current_view",
        on_click="ignore",
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
