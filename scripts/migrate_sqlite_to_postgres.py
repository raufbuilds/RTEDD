import argparse
import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from server.database import async_session_factory
from server.models import Demand, ForecastCache, Weather


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_json_value(value):
    if value in (None, ""):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def parse_epoch_or_datetime(value):
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError):
        return pd.Timestamp(value).to_pydatetime()


def read_sqlite_rows(sqlite_path: Path, table: str):
    with sqlite3.connect(sqlite_path) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            return []
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


async def migrate(sqlite_path: Path):
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    demand_rows = read_sqlite_rows(sqlite_path, "demand")
    weather_rows = read_sqlite_rows(sqlite_path, "weather")
    forecast_rows = read_sqlite_rows(sqlite_path, "forecast_cache")

    async with async_session_factory() as db:
        if demand_rows:
            values = [
                {
                    "date": pd.Timestamp(row["date"]).date(),
                    "hour": int(row["hour"]),
                    "demand": float(row["demand"]),
                }
                for row in demand_rows
                if row.get("date") is not None and row.get("hour") is not None and row.get("demand") is not None
            ]
            statement = pg_insert(Demand).values(values).on_conflict_do_update(
                index_elements=["date", "hour"],
                set_={"demand": pg_insert(Demand).excluded.demand},
            )
            await db.execute(statement)
            logger.info("Migrated %s demand rows", len(values))

        if weather_rows:
            values = [
                {
                    "date": pd.Timestamp(row["date"]).date(),
                    "hour": int(row["hour"]),
                    "temp": row.get("temp"),
                    "humidity": row.get("humidity"),
                    "wind": row.get("wind"),
                    "solar": row.get("solar"),
                    "data_source": row.get("data_source") or "open-meteo",
                }
                for row in weather_rows
                if row.get("date") is not None and row.get("hour") is not None
            ]
            statement = pg_insert(Weather).values(values)
            await db.execute(
                statement.on_conflict_do_update(
                    index_elements=["date", "hour"],
                    set_={
                        "temp": statement.excluded.temp,
                        "humidity": statement.excluded.humidity,
                        "wind": statement.excluded.wind,
                        "solar": statement.excluded.solar,
                        "data_source": statement.excluded.data_source,
                    },
                )
            )
            logger.info("Migrated %s weather rows", len(values))

        if forecast_rows:
            values = [
                {
                    "cache_key": row["cache_key"],
                    "target_date": pd.Timestamp(row["target_date"]).date(),
                    "include_target_date": bool(row.get("include_target_date")),
                    "signature": parse_json_value(row.get("signature")),
                    "result_json": parse_json_value(row.get("result_json")),
                    "status": row.get("status") or "training",
                    "error": row.get("error"),
                    "trained_at": parse_epoch_or_datetime(row.get("trained_at")),
                    "requested_at": parse_epoch_or_datetime(row.get("requested_at")),
                }
                for row in forecast_rows
                if row.get("cache_key") and row.get("target_date")
            ]
            statement = pg_insert(ForecastCache).values(values)
            await db.execute(
                statement.on_conflict_do_update(
                    index_elements=["cache_key"],
                    set_={
                        "target_date": statement.excluded.target_date,
                        "include_target_date": statement.excluded.include_target_date,
                        "signature": statement.excluded.signature,
                        "result_json": statement.excluded.result_json,
                        "status": statement.excluded.status,
                        "error": statement.excluded.error,
                        "trained_at": statement.excluded.trained_at,
                        "requested_at": statement.excluded.requested_at,
                    },
                )
            )
            logger.info("Migrated %s forecast cache rows", len(values))

        await db.commit()


def main():
    parser = argparse.ArgumentParser(description="Migrate RTEDD SQLite data.db into PostgreSQL.")
    parser.add_argument("--sqlite-path", default="data.db", help="Path to the old SQLite data.db file.")
    args = parser.parse_args()
    asyncio.run(migrate(Path(args.sqlite_path)))


if __name__ == "__main__":
    main()
