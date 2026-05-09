from contextlib import contextmanager
import asyncio
import json
import os
from queue import Queue
import sqlite3
import threading
import time
from typing import Optional

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
            "CREATE INDEX IF NOT EXISTS idx_demand_date_hour "
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
    return df


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


def forecast_with_xgboost(df, target_date=None):
    if df.empty or len(df) < 48:
        return pd.DataFrame(columns=["Hour", "XGBoost Predicted"])

    from xgboost import XGBRegressor

    df_features = df.copy()
    df_features = df_features.dropna(subset=["Timestamp", "Ontario Demand"])

    if len(df_features) < 48:
        return pd.DataFrame(columns=["Hour", "XGBoost Predicted"])

    df_features["hour"] = df_features["Timestamp"].dt.hour
    df_features["day_of_week"] = df_features["Timestamp"].dt.dayofweek
    df_features["day_of_month"] = df_features["Timestamp"].dt.day
    df_features["month"] = df_features["Timestamp"].dt.month
    df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)

    df_features = df_features.sort_values("Timestamp")
    df_features["demand_lag_1"] = df_features["Ontario Demand"].shift(1)
    df_features["demand_lag_24"] = df_features["Ontario Demand"].shift(24)
    df_features["rolling_mean_24"] = df_features["Ontario Demand"].rolling(24).mean()
    df_features["rolling_std_24"] = df_features["Ontario Demand"].rolling(24).std()
    df_features = df_features.dropna()

    if len(df_features) < 24:
        return pd.DataFrame(columns=["Hour", "XGBoost Predicted"])

    feature_cols = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "demand_lag_1",
        "demand_lag_24",
        "rolling_mean_24",
        "rolling_std_24",
    ]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=FORECAST_XGBOOST_N_JOBS,
    )
    model.fit(df_features[feature_cols], df_features["Ontario Demand"])

    if target_date is None:
        target_date = df["Date"].max()

    target_ts = pd.Timestamp(target_date)
    predictions = []
    for hour in range(24):
        predictions.append(
            {
                "hour": hour,
                "day_of_week": target_ts.dayofweek,
                "day_of_month": target_ts.day,
                "month": target_ts.month,
                "is_weekend": 1 if target_ts.dayofweek >= 5 else 0,
                "demand_lag_1": df_features["Ontario Demand"].iloc[-1],
                "demand_lag_24": (
                    df_features[df_features["Timestamp"] < target_ts - pd.Timedelta(hours=24)][
                        "Ontario Demand"
                    ].mean()
                    if len(df_features) > 24
                    else 0
                ),
                "rolling_mean_24": df_features["Ontario Demand"].tail(24).mean(),
                "rolling_std_24": df_features["Ontario Demand"].tail(24).std(),
            }
        )

    pred_df = pd.DataFrame(predictions)[feature_cols]
    forecasted = model.predict(pred_df)
    return pd.DataFrame({"Hour": range(24), "XGBoost Predicted": forecasted})


def compute_ensemble_forecast(df, target_date=None, include_target_date=False):
    training_df = forecast_training_frame(df, target_date, include_target_date)
    prophet_forecast = forecast_with_prophet(training_df, target_date)
    xgboost_forecast = forecast_with_xgboost(training_df, target_date)

    if prophet_forecast.empty and xgboost_forecast.empty:
        return pd.DataFrame(columns=["Hour", "Prophet", "XGBoost", "Ensemble"])

    if prophet_forecast.empty:
        result = xgboost_forecast.rename(columns={"XGBoost Predicted": "Ensemble"})
        result["Prophet"] = result["Ensemble"]
        result["XGBoost"] = result["Ensemble"]
    elif xgboost_forecast.empty:
        result = prophet_forecast.rename(columns={"Prophet Predicted": "Ensemble"})
        result["Prophet"] = result["Ensemble"]
        result["XGBoost"] = result["Ensemble"]
    else:
        merged = pd.merge(prophet_forecast, xgboost_forecast, on="Hour", how="outer")
        merged["Ensemble"] = 0.4 * merged["Prophet Predicted"] + 0.6 * merged["XGBoost Predicted"]
        result = merged.rename(
            columns={
                "Prophet Predicted": "Prophet",
                "XGBoost Predicted": "XGBoost",
            }
        )

    return result[["Hour", "Prophet", "XGBoost", "Ensemble"]]


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
            "SELECT id FROM demand WHERE date = ? AND hour = ? LIMIT 1",
            (date_value, hour_value),
        )
        existing = curr.fetchone()

        if existing:
            return {"status": "skipped", "reason": "duplicate", "id": existing[0]}

        curr.execute(
            "INSERT INTO demand (date, hour, demand) VALUES (?,?,?)",
            (date_value, hour_value, demand_value),
        )
        conn.commit()
        inserted_id = curr.lastrowid
        return {"status": "saved", "id": inserted_id}


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
                "SELECT id FROM demand WHERE date = ? AND hour = ? LIMIT 1",
                (date_value, hour_value),
            )
            if curr.fetchone():
                skipped += 1
                continue

            curr.execute(
                "INSERT INTO demand (date, hour, demand) VALUES (?,?,?)",
                (date_value, hour_value, demand_value),
            )
            saved += 1

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
            "stale": False,
            "message": "No demand records are available.",
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
    is_stale = (
        cached is not None
        and (
            cached.get("signature") != list(current_signature)
            or cached.get("trained_at") is None
            or now - float(cached["trained_at"]) > FORECAST_REFRESH_SECONDS
        )
    )

    if is_missing or is_stale or (cached and cached.get("status") == "failed"):
        schedule_forecast_refresh(background_tasks, normalized_target_date, include_target_date)

    if cached is None:
        return {
            "status": "training",
            "target_date": normalized_target_date,
            "include_target_date": include_target_date,
            "forecast": [],
            "trained_at": None,
            "stale": True,
            "message": "Forecast training has been queued.",
        }

    status = cached["status"]
    if is_stale and status == "fresh":
        status = "stale"

    return {
        "status": status,
        "target_date": normalized_target_date,
        "include_target_date": include_target_date,
        "forecast": cached["forecast"],
        "trained_at": cached["trained_at"],
        "stale": is_stale,
        "error": cached["error"],
        "message": "Forecast refresh has been queued." if is_stale else None,
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
