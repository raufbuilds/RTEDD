from contextlib import contextmanager
import asyncio
import json
import os
from queue import Queue
import sqlite3
import threading
import time
from typing import Any, Optional, cast

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse


app = FastAPI()


class RequestSizeLimitMiddleware:
    def __init__(self, app, max_size: int):
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = 0
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"content-length":
                try:
                    content_length = int(header_value.decode("latin1"))
                except ValueError:
                    content_length = 0
                break

        if content_length > self.max_size:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body too large"},
            )
            await response(scope, receive, send)
            return

        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        body = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                more_body = message.get("more_body", False)
                if len(body) > self.max_size:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large"},
                    )
                    await response(scope, receive, send)
                    return
            else:
                await self.app(scope, receive, send)
                return

        body_sent = False

        async def replay_receive():
            nonlocal body_sent
            if body_sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)


def load_env_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

DB_PATH = os.getenv("DB_PATH", "data.db")
DB_POOL_SIZE = max(1, int(os.getenv("DB_POOL_SIZE", "5")))
MAX_REQUEST_SIZE_BYTES = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(1024 * 1024)))
MAX_BULK_INGEST_ROWS = int(os.getenv("MAX_BULK_INGEST_ROWS", "5000"))
FORECAST_REFRESH_SECONDS = int(os.getenv("FORECAST_REFRESH_SECONDS", "300"))
FORECAST_XGBOOST_N_JOBS = int(os.getenv("FORECAST_XGBOOST_N_JOBS", "2"))
FORECAST_LIGHTGBM_RETRAIN_ROWS = int(os.getenv("FORECAST_LIGHTGBM_RETRAIN_ROWS", "168"))
WEATHER_LATITUDE = float(os.getenv("WEATHER_LATITUDE", "43.6532"))
WEATHER_LONGITUDE = float(os.getenv("WEATHER_LONGITUDE", "-79.3832"))

connection_pool = Queue(maxsize=DB_POOL_SIZE)
forecast_training_lock = threading.Lock()
forecast_training_keys: set[str] = set()
app.add_middleware(RequestSizeLimitMiddleware, max_size=MAX_REQUEST_SIZE_BYTES)


def create_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


for _ in range(DB_POOL_SIZE):
    connection_pool.put(create_connection())


@contextmanager
def get_connection():
    conn = connection_pool.get()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        connection_pool.put(conn)


def init_db():
    with get_connection() as conn:
        curr = conn.cursor()
        curr.execute(
            "CREATE TABLE IF NOT EXISTS demand ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "date TEXT, hour INTEGER, demand REAL)"
        )
        curr.execute(
            "DELETE FROM demand "
            "WHERE id NOT IN ("
            "SELECT MIN(id) FROM demand GROUP BY date, hour"
            ")"
        )
        curr.execute("DROP INDEX IF EXISTS idx_demand_date_hour")
        curr.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_demand_date_hour "
            "ON demand(date, hour)"
        )
        curr.execute(
            "CREATE TABLE IF NOT EXISTS forecast_cache ("
            "cache_key TEXT PRIMARY KEY, "
            "target_date TEXT, "
            "include_target_date INTEGER, "
            "signature TEXT, "
            "result_json TEXT, "
            "status TEXT, "
            "error TEXT, "
            "trained_at REAL, "
            "requested_at REAL)"
        )
        curr.execute(
            "CREATE TABLE IF NOT EXISTS weather ("
            "date TEXT, hour INTEGER, temp REAL, humidity REAL, wind REAL, solar REAL, "
            "PRIMARY KEY(date, hour))"
        )
        conn.commit()


init_db()


def fetch_rows(after_id: int = 0, limit: Optional[int] = None):
    with get_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT id, date, hour, demand FROM demand WHERE id > ? ORDER BY id ASC"
        params = [after_id]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        return cursor.fetchall()


def fetch_record_count():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM demand")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def fetch_latest_progress():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, date, hour, demand FROM demand "
            "ORDER BY date DESC, hour DESC, id DESC LIMIT 1"
        )
        return cursor.fetchone()


def fetch_all_demand_rows():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, date, hour, demand FROM demand ORDER BY date ASC, hour ASC, id ASC")
        return cursor.fetchall()


def fetch_weather_rows():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT date, hour, temp, humidity, wind, solar FROM weather")
        return cursor.fetchall()


def demand_dataframe():
    rows = fetch_all_demand_rows()
    if not rows:
        return pd.DataFrame(columns=["id", "Date", "Hour", "Ontario Demand", "Timestamp"])

    df = pd.DataFrame(rows, columns=["id", "Date", "Hour", "Ontario Demand"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df["Ontario Demand"] = pd.to_numeric(df["Ontario Demand"], errors="coerce")
    df = df.dropna(subset=["Date", "Hour", "Ontario Demand"])
    df["Hour"] = df["Hour"].astype(int)
    df = df[(df["Hour"] >= 0) & (df["Hour"] <= 23)]
    df = df.sort_values(["Date", "Hour", "id"]).reset_index(drop=True)
    df["Timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df["Date Label"] = df["Date"].dt.strftime("%Y-%m-%d")
    weather_rows = fetch_weather_rows()
    if weather_rows:
        weather_df = pd.DataFrame(
            weather_rows,
            columns=["Date", "Hour", "temp", "humidity", "wind", "solar"],
        )
        weather_df["Date"] = pd.to_datetime(weather_df["Date"], errors="coerce")
        weather_df["Hour"] = pd.to_numeric(weather_df["Hour"], errors="coerce")
        weather_df = weather_df.dropna(subset=["Date", "Hour"])
        weather_df["Hour"] = weather_df["Hour"].astype(int)
        df = pd.merge(df, weather_df, on=["Date", "Hour"], how="left")
    return df


def calculate_anomalies(df):
    if df.empty:
        return df

    df = df.copy()
    hour_median = df.groupby("Hour")["Ontario Demand"].transform("median")
    hour_mad = df.groupby("Hour")["Ontario Demand"].transform(
        lambda series: (series - series.median()).abs().median()
    )

    global_mad = (df["Ontario Demand"] - df["Ontario Demand"].median()).abs().median()
    global_std = df["Ontario Demand"].std()
    fallback_scale = max(
        [value for value in [global_mad * 1.4826, global_std] if pd.notna(value) and value > 0]
        or [1.0]
    )

    scale = hour_mad.fillna(fallback_scale) * 1.4826
    scale = scale.replace(0, fallback_scale)

    df["Expected Demand"] = hour_median
    df["Deviation"] = df["Ontario Demand"] - df["Expected Demand"]
    df["Anomaly Score"] = (df["Deviation"].abs() / scale).fillna(0)
    df["Anomaly"] = df["Anomaly Score"] >= 3
    df["Status"] = df["Anomaly"].map(lambda value: "Anomaly detected" if value else "System normal")
    return df


def compute_hourly_baseline(df, threshold=3.0, target_date=None, min_points_per_hour=2):
    if df.empty:
        return pd.DataFrame(columns=["Hour", "Expected", "Scale", "Lower", "Upper"])

    baseline_source = df.copy()
    baseline_source["Date"] = pd.to_datetime(baseline_source["Date"], errors="coerce")
    baseline_source["Hour"] = pd.to_numeric(baseline_source["Hour"], errors="coerce")
    baseline_source["Ontario Demand"] = pd.to_numeric(
        baseline_source["Ontario Demand"],
        errors="coerce",
    )
    baseline_source = baseline_source.dropna(subset=["Date", "Hour", "Ontario Demand"])
    if baseline_source.empty:
        return pd.DataFrame(columns=["Hour", "Expected", "Scale", "Lower", "Upper"])

    baseline_source["Hour"] = baseline_source["Hour"].astype(int)
    baseline_source = baseline_source[(baseline_source["Hour"] >= 0) & (baseline_source["Hour"] <= 23)]
    if baseline_source.empty:
        return pd.DataFrame(columns=["Hour", "Expected", "Scale", "Lower", "Upper"])

    if target_date is None:
        target_date = baseline_source["Date"].max()
    target_month = pd.Timestamp(target_date).month
    target_quarter = pd.Timestamp(target_date).quarter

    same_month = baseline_source[baseline_source["Date"].dt.month == target_month]
    same_quarter = baseline_source[baseline_source["Date"].dt.quarter == target_quarter]

    monthly_counts = same_month.groupby("Hour")["Ontario Demand"].size()
    enough_monthly_hours = (monthly_counts >= min_points_per_hour).sum()
    if enough_monthly_hours >= 18:
        seasonal_source = same_month
    elif not same_quarter.empty:
        seasonal_source = same_quarter
    else:
        seasonal_source = baseline_source

    def mad(series: pd.Series) -> float:
        med = series.median()
        return float((series - med).abs().median())

    global_median = baseline_source["Ontario Demand"].median()
    global_mad = float((baseline_source["Ontario Demand"] - global_median).abs().median())
    global_std = baseline_source["Ontario Demand"].std()
    global_std = float(global_std) if pd.notna(global_std) else 0.0
    fallback_scale = max([v for v in [global_mad * 1.4826, global_std] if v and v > 0] or [1.0])

    seasonal_expected = seasonal_source.groupby("Hour")["Ontario Demand"].median()
    seasonal_mad = seasonal_source.groupby("Hour")["Ontario Demand"].apply(mad) * 1.4826
    all_hour_expected = baseline_source.groupby("Hour")["Ontario Demand"].median()
    all_hour_mad = baseline_source.groupby("Hour")["Ontario Demand"].apply(mad) * 1.4826

    available_hours = sorted(set(all_hour_expected.index).union(set(seasonal_expected.index)))
    expected = seasonal_expected.reindex(available_hours).combine_first(
        all_hour_expected.reindex(available_hours)
    )
    scale = seasonal_mad.reindex(available_hours).combine_first(
        all_hour_mad.reindex(available_hours)
    )
    scale = scale.replace(0, fallback_scale).fillna(fallback_scale)

    baseline = pd.DataFrame(
        {
            "Hour": expected.index.astype(int),
            "Expected": expected.values.astype(float),
            "Scale": scale.values.astype(float),
        }
    )
    baseline["Lower"] = baseline["Expected"] - threshold * baseline["Scale"]
    baseline["Upper"] = baseline["Expected"] + threshold * baseline["Scale"]
    return baseline


def forecast_training_frame(df, target_date=None, include_target_date=False):
    if df.empty:
        return df

    if target_date is None:
        target_date = df["Date"].max()

    if include_target_date:
        return df.copy()

    historical_df = df[df["Date"] < pd.Timestamp(target_date)].copy()
    if len(historical_df) >= 48:
        return historical_df

    return df.copy()


def forecast_cache_signature(df, target_date=None, include_target_date=False):
    if df.empty:
        return (0, None, None, None, include_target_date)

    if target_date is None:
        target_date = df["Date"].max()

    latest_ts = df["Timestamp"].max() if "Timestamp" in df.columns else None
    earliest_ts = df["Timestamp"].min() if "Timestamp" in df.columns else None
    return (
        len(df),
        pd.Timestamp(earliest_ts).isoformat() if pd.notna(earliest_ts) else None,
        pd.Timestamp(latest_ts).isoformat() if pd.notna(latest_ts) else None,
        pd.Timestamp(target_date).date().isoformat(),
        include_target_date,
    )


def forecast_with_prophet(df, target_date=None):
    if df.empty or len(df) < 24:
        return pd.DataFrame(columns=["Hour", "Prophet Predicted"])

    from prophet import Prophet

    prophet_df = df[["Timestamp", "Ontario Demand"]].copy()
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df.rename(columns={"Timestamp": "ds", "Ontario Demand": "y"})

    if len(prophet_df) < 24:
        return pd.DataFrame(columns=["Hour", "Prophet Predicted"])

    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    if target_date is None:
        target_date = df["Date"].max()

    future_df = pd.DataFrame(
        {"ds": [pd.Timestamp(target_date).replace(hour=hour) for hour in range(24)]}
    )
    forecast = model.predict(future_df)
    return pd.DataFrame(
        {
            "Hour": range(24),
            "Prophet Predicted": forecast["yhat"].values,
        }
    )


def add_prophet_components(df):
    component_cols = ["prophet_trend", "prophet_yearly", "prophet_weekly", "prophet_daily"]
    if df.empty or len(df) < 24:
        result = df.copy()
        for column in component_cols:
            result[column] = 0.0
        return result

    try:
        from prophet import Prophet
    except ImportError:
        result = df.copy()
        for column in component_cols:
            result[column] = 0.0
        return result

    prophet_df = df[["Timestamp", "Ontario Demand"]].copy()
    prophet_df = prophet_df.dropna()
    prophet_df = prophet_df.rename(columns={"Timestamp": "ds", "Ontario Demand": "y"})
    if len(prophet_df) < 24:
        result = df.copy()
        for column in component_cols:
            result[column] = 0.0
        return result

    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)
    components = model.predict(prophet_df[["ds"]])
    component_df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(prophet_df["ds"]).values,
            "prophet_trend": components.get("trend", 0.0),
            "prophet_yearly": components.get("yearly", 0.0),
            "prophet_weekly": components.get("weekly", 0.0),
            "prophet_daily": components.get("daily", 0.0),
        }
    )

    result = pd.merge(df.copy(), component_df, on="Timestamp", how="left")
    for column in component_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def get_model_metadata_path(model_path):
    """Get the metadata file path for a model."""
    return model_path.replace(".pkl", ".meta.json")


def load_model_train_rows(model_path):
    """Load the number of training rows from model metadata."""
    meta_path = get_model_metadata_path(model_path)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f).get("train_rows", None)
        except Exception:
            return None
    return None


def save_model_train_rows(model_path, train_rows):
    """Save the number of training rows to model metadata."""
    meta_path = get_model_metadata_path(model_path)
    try:
        with open(meta_path, "w") as f:
            json.dump({"train_rows": train_rows}, f)
    except Exception:
        pass


def forecast_with_xgboost(df, target_date=None):
    import os

    import joblib
    try:
        import lightgbm as lgb
    except ImportError:
        return pd.DataFrame(columns=["Hour", "XGBoost Predicted"])
    try:
        from server.features import engineer_features, get_feature_cols
    except ImportError:
        from features import engineer_features, get_feature_cols

    columns = ["Hour", "XGBoost Predicted", "XGBoost_P10", "XGBoost_P50", "XGBoost_P90"]
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    if len(df) < 500:
        return pd.DataFrame(columns=columns)

    df_with_prophet = add_prophet_components(df)
    df_feat = engineer_features(df_with_prophet)
    feature_cols = get_feature_cols(df_feat)

    last_timestamp = pd.Timestamp(df_feat.iloc[-1]["Timestamp"])
    predictions = []
    for horizon in range(1, 25):
        train = df_feat.copy()
        train["target"] = train["Ontario Demand"].shift(-horizon)
        train = train.dropna(subset=["target"])
        current_train_rows = len(train)
        if current_train_rows < 200:
            continue

        X_pred = df_feat.iloc[-1:][feature_cols]
        horizon_preds = {}
        for label, alpha in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]:
            model_path = os.path.join(MODEL_DIR, f"lgb_h{horizon}_{label}.pkl")
            model = None
            retrain = True

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                train_rows = load_model_train_rows(model_path)
                if (
                    train_rows is not None
                    and (current_train_rows - train_rows) < FORECAST_LIGHTGBM_RETRAIN_ROWS
                ):
                    retrain = False

            if model is None or retrain:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.04,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.05,
                    reg_lambda=0.05,
                    random_state=42,
                    n_jobs=FORECAST_XGBOOST_N_JOBS,
                    verbose=-1,
                )
                model.fit(train[feature_cols], train["target"])
                joblib.dump(model, model_path)
                save_model_train_rows(model_path, len(train))

            pred_result = cast(Any, model.predict(X_pred))
            horizon_preds[label] = float(pred_result[0])

        predictions.append(
            {
                "Hour": int((last_timestamp + pd.Timedelta(hours=horizon)).hour),
                "XGBoost Predicted": horizon_preds["p50"],
                "XGBoost_P10": horizon_preds["p10"],
                "XGBoost_P50": horizon_preds["p50"],
                "XGBoost_P90": horizon_preds["p90"],
            }
        )

    return pd.DataFrame(predictions, columns=columns)


def compute_ensemble_forecast(df, target_date=None, include_target_date=False):
    training_df = forecast_training_frame(df, target_date, include_target_date)
    prophet_forecast = forecast_with_prophet(training_df, target_date)
    xgboost_forecast = forecast_with_xgboost(training_df, target_date)

    forecast_columns = [
        "Hour",
        "Prophet",
        "XGBoost",
        "Ensemble",
        "Ensemble_P10",
        "Ensemble_P50",
        "Ensemble_P90",
    ]
    if prophet_forecast.empty and xgboost_forecast.empty:
        return pd.DataFrame(columns=forecast_columns)

    if prophet_forecast.empty:
        result = xgboost_forecast.rename(columns={"XGBoost Predicted": "Ensemble"})
        result["Prophet"] = result["Ensemble"]
        result["XGBoost"] = result["Ensemble"]
        result["Ensemble_P10"] = result.get("XGBoost_P10", result["Ensemble"])
        result["Ensemble_P50"] = result.get("XGBoost_P50", result["Ensemble"])
        result["Ensemble_P90"] = result.get("XGBoost_P90", result["Ensemble"])
    elif xgboost_forecast.empty:
        result = prophet_forecast.rename(columns={"Prophet Predicted": "Ensemble"})
        result["Prophet"] = result["Ensemble"]
        result["XGBoost"] = result["Ensemble"]
        result["Ensemble_P10"] = result["Ensemble"]
        result["Ensemble_P50"] = result["Ensemble"]
        result["Ensemble_P90"] = result["Ensemble"]
    else:
        merged = pd.merge(prophet_forecast, xgboost_forecast, on="Hour", how="outer")
        merged["Ensemble"] = 0.05 * merged["Prophet Predicted"] + 0.95 * merged["XGBoost Predicted"]
        merged["Ensemble_P10"] = 0.05 * merged["Prophet Predicted"] + 0.95 * merged.get(
            "XGBoost_P10",
            merged["XGBoost Predicted"],
        )
        merged["Ensemble_P50"] = 0.05 * merged["Prophet Predicted"] + 0.95 * merged.get(
            "XGBoost_P50",
            merged["XGBoost Predicted"],
        )
        merged["Ensemble_P90"] = 0.05 * merged["Prophet Predicted"] + 0.95 * merged.get(
            "XGBoost_P90",
            merged["XGBoost Predicted"],
        )
        result = merged.rename(
            columns={
                "Prophet Predicted": "Prophet",
                "XGBoost Predicted": "XGBoost",
            }
        )

    for column in forecast_columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[forecast_columns]


def normalize_target_date(df, target_date=None):
    if target_date is not None:
        return pd.Timestamp(target_date).date().isoformat()
    if df.empty:
        return None
    return pd.Timestamp(df["Date"].max()).date().isoformat()


def forecast_cache_key(target_date: str, include_target_date: bool):
    return f"{target_date}|include_today={int(include_target_date)}"


def get_cached_forecast(cache_key: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cache_key, target_date, include_target_date, signature, result_json, "
            "status, error, trained_at, requested_at FROM forecast_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    result = json.loads(row[4]) if row[4] else []
    return {
        "cache_key": row[0],
        "target_date": row[1],
        "include_target_date": bool(row[2]),
        "signature": json.loads(row[3]) if row[3] else None,
        "forecast": result,
        "status": row[5],
        "error": row[6],
        "trained_at": row[7],
        "requested_at": row[8],
    }


def get_latest_fresh_forecast(include_target_date: bool):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cache_key, target_date, include_target_date, signature, result_json, "
            "status, error, trained_at, requested_at FROM forecast_cache "
            "WHERE include_target_date = ? AND status = 'fresh' AND result_json IS NOT NULL "
            "ORDER BY trained_at DESC LIMIT 1",
            (int(include_target_date),),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "cache_key": row[0],
        "target_date": row[1],
        "include_target_date": bool(row[2]),
        "signature": json.loads(row[3]) if row[3] else None,
        "forecast": json.loads(row[4]) if row[4] else [],
        "status": row[5],
        "error": row[6],
        "trained_at": row[7],
        "requested_at": row[8],
    }


def forecast_training_seconds(cached, now=None):
    if not cached or cached.get("requested_at") is None:
        return None

    end_time = cached.get("trained_at") if cached.get("status") in {"fresh", "failed", "empty"} else now
    if end_time is None:
        end_time = time.time()

    return max(0.0, float(end_time) - float(cached["requested_at"]))


def forecast_summary(status, stale=False, error=None, training_seconds=None):
    if status == "training":
        if training_seconds is None:
            return "Training is queued or starting."
        return f"Training is running for {training_seconds:.0f}s."
    if status == "stale":
        return "Showing cached forecast while a refresh is queued."
    if status == "fresh":
        return "Forecast is ready and using cached server results."
    if status == "failed":
        return f"Training failed: {error}" if error else "Training failed."
    if status == "empty":
        return "No forecast is available because there are no demand records."
    if stale:
        return "Forecast cache is stale and refresh has been queued."
    return "Forecast status is pending."


def save_forecast_cache(
    cache_key: str,
    target_date: str,
    include_target_date: bool,
    signature,
    result: pd.DataFrame,
    status: str,
    error: Optional[str] = None,
):
    result_json = result.to_json(orient="records") if result is not None else None
    now = time.time()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO forecast_cache "
            "(cache_key, target_date, include_target_date, signature, result_json, status, error, trained_at, requested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET "
            "target_date = excluded.target_date, "
            "include_target_date = excluded.include_target_date, "
            "signature = excluded.signature, "
            "result_json = excluded.result_json, "
            "status = excluded.status, "
            "error = excluded.error, "
            "trained_at = excluded.trained_at, "
            "requested_at = excluded.requested_at",
            (
                cache_key,
                target_date,
                int(include_target_date),
                json.dumps(signature),
                result_json,
                status,
                error,
                now,
                now,
            ),
        )
        conn.commit()


def mark_forecast_requested(cache_key: str, target_date: str, include_target_date: bool, status: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO forecast_cache "
            "(cache_key, target_date, include_target_date, status, requested_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET "
            "status = excluded.status, requested_at = excluded.requested_at",
            (cache_key, target_date, int(include_target_date), status, time.time()),
        )
        conn.commit()


def refresh_forecast_cache(target_date: str, include_target_date: bool):
    cache_key = forecast_cache_key(target_date, include_target_date)

    try:
        mark_forecast_requested(cache_key, target_date, include_target_date, "training")
        df = demand_dataframe()
        if df.empty:
            save_forecast_cache(
                cache_key,
                target_date,
                include_target_date,
                forecast_cache_signature(df, target_date, include_target_date),
                pd.DataFrame(columns=["Hour", "Prophet", "XGBoost", "Ensemble"]),
                "empty",
                "No demand records are available.",
            )
            return

        training_df = forecast_training_frame(df, target_date, include_target_date)
        signature = forecast_cache_signature(training_df, target_date, include_target_date)
        result = compute_ensemble_forecast(df, target_date, include_target_date)
        save_forecast_cache(cache_key, target_date, include_target_date, signature, result, "fresh")
    except Exception as exc:
        save_forecast_cache(
            cache_key,
            target_date,
            include_target_date,
            None,
            pd.DataFrame(columns=["Hour", "Prophet", "XGBoost", "Ensemble"]),
            "failed",
            str(exc),
        )
    finally:
        with forecast_training_lock:
            forecast_training_keys.discard(cache_key)


def schedule_forecast_refresh(
    background_tasks: BackgroundTasks,
    target_date: str,
    include_target_date: bool,
):
    cache_key = forecast_cache_key(target_date, include_target_date)
    with forecast_training_lock:
        if cache_key in forecast_training_keys:
            return
        forecast_training_keys.add(cache_key)

    background_tasks.add_task(refresh_forecast_cache, target_date, include_target_date)


@app.post("/ingest")
async def ingest(data: dict):
    date_value = data.get("Date")
    hour_value = data.get("Hour")
    demand_value = data.get("Ontario Demand")

    with get_connection() as conn:
        curr = conn.cursor()
        curr.execute(
            "INSERT OR IGNORE INTO demand (date, hour, demand) VALUES (?,?,?)",
            (date_value, hour_value, demand_value),
        )
        conn.commit()
        if curr.rowcount:
            return {"status": "saved", "id": curr.lastrowid}

        curr.execute(
            "SELECT id FROM demand WHERE date = ? AND hour = ? LIMIT 1",
            (date_value, hour_value),
        )
        existing = curr.fetchone()
        existing_id = existing[0] if existing else None
        return {"status": "skipped", "reason": "duplicate", "id": existing_id}


def normalize_ingest_record(data: dict):
    return (
        data.get("Date"),
        data.get("Hour"),
        data.get("Ontario Demand"),
    )


@app.post("/ingest/bulk")
async def ingest_bulk(data: dict):
    rows = data.get("rows")
    if not isinstance(rows, list):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "detail": "Expected a JSON body with a rows list."},
        )

    if len(rows) > MAX_BULK_INGEST_ROWS:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "detail": f"Bulk ingest accepts at most {MAX_BULK_INGEST_ROWS} rows.",
            },
        )

    normalized_rows = []
    invalid = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid += 1
            continue

        date_value, hour_value, demand_value = normalize_ingest_record(row)
        if date_value is None or hour_value is None or demand_value is None:
            invalid += 1
            continue

        normalized_rows.append((date_value, hour_value, demand_value))

    saved = 0
    skipped = 0
    with get_connection() as conn:
        curr = conn.cursor()
        for date_value, hour_value, demand_value in normalized_rows:
            curr.execute(
                "INSERT OR IGNORE INTO demand (date, hour, demand) VALUES (?,?,?)",
                (date_value, hour_value, demand_value),
            )
            if curr.rowcount:
                saved += 1
            else:
                skipped += 1

        conn.commit()

    return {
        "status": "saved",
        "received": len(rows),
        "valid": len(normalized_rows),
        "saved": saved,
        "skipped": skipped,
        "invalid": invalid,
    }


@app.get("/records")
async def records(
    after_id: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1, le=10000),
):
    rows = fetch_rows(after_id=after_id, limit=limit)
    return [
        {
            "id": row[0],
            "Date": row[1],
            "Hour": row[2],
            "Ontario Demand": row[3],
        }
        for row in rows
    ]


@app.get("/records/count")
async def records_count():
    return {"total_records": fetch_record_count()}


@app.get("/dashboard/data")
async def dashboard_data():
    df = demand_dataframe()
    total_records = len(df)
    if df.empty:
        return {"records": [], "baseline": [], "total_records": total_records}

    df = calculate_anomalies(df)
    baseline = compute_hourly_baseline(df)

    records_df = df.copy()
    records_df["Date"] = records_df["Date"].dt.strftime("%Y-%m-%d")
    records_df["Timestamp"] = records_df["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "records": records_df.to_dict(orient="records"),
        "baseline": baseline.to_dict(orient="records"),
        "total_records": total_records,
    }


@app.post("/weather/backfill")
async def weather_backfill(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    try:
        from server.weather import fetch_weather_frame, upsert_weather_rows
    except ImportError:
        from weather import fetch_weather_frame, upsert_weather_rows

    df = demand_dataframe()
    if df.empty and (start_date is None or end_date is None):
        return {"status": "empty", "saved": 0, "message": "No demand rows or date range available."}

    start = start_date or pd.Timestamp(df["Date"].min()).date().isoformat()
    end = end_date or pd.Timestamp(df["Date"].max()).date().isoformat()
    weather_df = fetch_weather_frame(WEATHER_LATITUDE, WEATHER_LONGITUDE, start, end)
    with get_connection() as conn:
        saved = upsert_weather_rows(conn, weather_df.to_dict(orient="records"))

    return {
        "status": "saved",
        "saved": saved,
        "start_date": start,
        "end_date": end,
        "latitude": WEATHER_LATITUDE,
        "longitude": WEATHER_LONGITUDE,
    }


@app.get("/latest")
async def latest():
    row = fetch_latest_progress()
    if row is None:
        return {"id": None, "Date": None, "Hour": None, "Ontario Demand": None}

    return {
        "id": row[0],
        "Date": row[1],
        "Hour": row[2],
        "Ontario Demand": row[3],
    }


@app.get("/forecast/latest")
async def forecast_latest(
    background_tasks: BackgroundTasks,
    target_date: Optional[str] = Query(None),
    include_target_date: bool = Query(False),
):
    df = demand_dataframe()
    normalized_target_date = normalize_target_date(df, target_date)
    if normalized_target_date is None:
        return {
            "status": "empty",
            "target_date": None,
            "include_target_date": include_target_date,
            "forecast": [],
            "trained_at": None,
            "requested_at": None,
            "training_seconds": None,
            "stale": False,
            "message": "No demand records are available.",
            "summary": forecast_summary("empty"),
        }

    cache_key = forecast_cache_key(normalized_target_date, include_target_date)
    cached = get_cached_forecast(cache_key)

    training_df = forecast_training_frame(df, normalized_target_date, include_target_date)
    current_signature = forecast_cache_signature(
        training_df,
        normalized_target_date,
        include_target_date,
    )
    now = time.time()

    is_missing = cached is None or not cached.get("forecast")
    is_stale = cached is not None and (
        cached.get("signature") != list(current_signature)
        or cached.get("trained_at") is None
    )

    if is_missing or is_stale or (cached and cached.get("status") == "failed"):
        schedule_forecast_refresh(background_tasks, normalized_target_date, include_target_date)

    if cached is None:
        fallback = get_latest_fresh_forecast(include_target_date)
        if fallback is not None and fallback.get("forecast"):
            return {
                "status": "training",
                "target_date": normalized_target_date,
                "include_target_date": include_target_date,
                "forecast": fallback["forecast"],
                "trained_at": fallback["trained_at"],
                "requested_at": None,
                "training_seconds": None,
                "stale": True,
                "error": None,
                "message": "Forecast training has been queued.",
                "summary": (
                    f"Showing last saved forecast for {fallback['target_date']} "
                    f"while {normalized_target_date} trains."
                ),
            }
        return {
            "status": "training",
            "target_date": normalized_target_date,
            "include_target_date": include_target_date,
            "forecast": [],
            "trained_at": None,
            "requested_at": None,
            "training_seconds": None,
            "stale": True,
            "message": "Forecast training has been queued.",
            "summary": "Forecast training has been queued.",
        }

    if is_missing:
        fallback = get_latest_fresh_forecast(include_target_date)
        if fallback is not None and fallback.get("forecast"):
            training_seconds = forecast_training_seconds(cached, now)
            return {
                "status": cached["status"],
                "target_date": normalized_target_date,
                "include_target_date": include_target_date,
                "forecast": fallback["forecast"],
                "trained_at": fallback["trained_at"],
                "requested_at": cached["requested_at"],
                "training_seconds": training_seconds,
                "stale": True,
                "error": cached["error"],
                "message": "Forecast refresh has been queued.",
                "summary": (
                    f"Showing last saved forecast for {fallback['target_date']} "
                    f"while {normalized_target_date} trains."
                ),
            }

    status = cached["status"]
    if is_stale and status == "fresh":
        status = "stale"
    training_seconds = forecast_training_seconds(cached, now)

    return {
        "status": status,
        "target_date": normalized_target_date,
        "include_target_date": include_target_date,
        "forecast": cached["forecast"],
        "trained_at": cached["trained_at"],
        "requested_at": cached["requested_at"],
        "training_seconds": training_seconds,
        "stale": is_stale,
        "error": cached["error"],
        "message": "Forecast refresh has been queued." if is_stale else None,
        "summary": forecast_summary(
            status,
            stale=is_stale,
            error=cached["error"],
            training_seconds=training_seconds,
        ),
    }


@app.post("/forecast/refresh")
async def forecast_refresh(
    background_tasks: BackgroundTasks,
    target_date: Optional[str] = Query(None),
    include_target_date: bool = Query(False),
):
    df = demand_dataframe()
    normalized_target_date = normalize_target_date(df, target_date)
    if normalized_target_date is None:
        return {
            "status": "empty",
            "target_date": None,
            "include_target_date": include_target_date,
            "message": "No demand records are available.",
        }

    schedule_forecast_refresh(background_tasks, normalized_target_date, include_target_date)
    return {
        "status": "queued",
        "target_date": normalized_target_date,
        "include_target_date": include_target_date,
    }


@app.get("/forecast/status")
async def forecast_status(
    target_date: Optional[str] = Query(None),
    include_target_date: bool = Query(False),
):
    df = demand_dataframe()
    normalized_target_date = normalize_target_date(df, target_date)
    if normalized_target_date is None:
        return {
            "status": "empty",
            "target_date": None,
            "include_target_date": include_target_date,
        }

    cache_key = forecast_cache_key(normalized_target_date, include_target_date)
    cached = get_cached_forecast(cache_key)
    if cached is None:
        return {
            "status": "missing",
            "target_date": normalized_target_date,
            "include_target_date": include_target_date,
        }

    return {
        "status": cached["status"],
        "target_date": cached["target_date"],
        "include_target_date": cached["include_target_date"],
        "trained_at": cached["trained_at"],
        "error": cached["error"],
    }


@app.get("/stream")
async def stream(request: Request):
    async def event_generator():
        last_id = 0

        while True:
            if await request.is_disconnected():
                break

            try:
                rows = fetch_rows(after_id=last_id)
                for row in rows:
                    if await request.is_disconnected():
                        return

                    last_id = row[0]
                    record = {
                        "id": row[0],
                        "Date": row[1],
                        "Hour": row[2],
                        "Ontario Demand": row[3],
                    }
                    yield f"data: {json.dumps(record)}\n\n"
            except Exception as exc:
                error_payload = {"error": "stream_fetch_failed", "detail": str(exc)}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
