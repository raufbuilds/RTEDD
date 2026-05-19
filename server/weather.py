import datetime as dt
from typing import Iterable

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
]


def _weather_params(latitude, longitude, start_date, end_date):
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/Toronto",
    }


def fetch_weather_frame(latitude, longitude, start_date, end_date, timeout=30):
    today = dt.date.today()
    end = pd.Timestamp(end_date).date()
    url = FORECAST_URL if end >= today else ARCHIVE_URL
    response = requests.get(
        url,
        params=_weather_params(latitude, longitude, start_date, end_date),
        timeout=timeout,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame(columns=["date", "hour", "temp", "humidity", "wind", "solar"])

    weather = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(hourly["time"], errors="coerce"),
            "temp": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "wind": hourly.get("wind_speed_10m"),
            "solar": hourly.get("shortwave_radiation"),
        }
    )
    weather = weather.dropna(subset=["Timestamp"])
    weather["date"] = weather["Timestamp"].dt.strftime("%Y-%m-%d")
    weather["hour"] = weather["Timestamp"].dt.hour.astype(int)
    return weather[["date", "hour", "temp", "humidity", "wind", "solar"]]


def upsert_weather_rows(conn, rows: Iterable[dict]):
    cursor = conn.cursor()
    saved = 0
    for row in rows:
        cursor.execute(
            "INSERT INTO weather (date, hour, temp, humidity, wind, solar) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date, hour) DO UPDATE SET "
            "temp = excluded.temp, "
            "humidity = excluded.humidity, "
            "wind = excluded.wind, "
            "solar = excluded.solar",
            (
                row["date"],
                int(row["hour"]),
                row.get("temp"),
                row.get("humidity"),
                row.get("wind"),
                row.get("solar"),
            ),
        )
        saved += 1
    conn.commit()
    return saved
