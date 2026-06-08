import logging

from holidays import country_holidays
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
CRITICAL_FEATURE_COLUMNS = ["Ontario Demand", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def _feature_years(timestamp):
    years = pd.to_datetime(timestamp, errors="coerce").dt.year.dropna()
    if years.empty:
        current_year = pd.Timestamp.today().year
        return range(current_year - 1, current_year + 2)

    start_year = int(years.min()) - 1
    end_year = int(years.max()) + 2
    return range(start_year, end_year)


def _days_to_nearest_holiday(date_value, holidays):
    current_date = pd.Timestamp(date_value).date()
    distances = [
        abs(offset)
        for offset in range(-7, 8)
        if current_date + pd.Timedelta(days=offset) in holidays
    ]
    return min(distances) if distances else 8


def _dst_transition_dates(years):
    dates = set()
    for year in years:
        march = pd.Timestamp(year=year, month=3, day=1)
        november = pd.Timestamp(year=year, month=11, day=1)

        days_until_first_sunday_march = (6 - march.dayofweek) % 7
        days_until_first_sunday_november = (6 - november.dayofweek) % 7
        second_sunday_march = march + pd.Timedelta(days=days_until_first_sunday_march + 7)
        first_sunday_november = november + pd.Timedelta(days=days_until_first_sunday_november)

        dates.add(second_sunday_march.date())
        dates.add(first_sunday_november.date())
    return dates


def engineer_features(df):
    df = df.copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    timestamp = pd.to_datetime(df["Timestamp"], errors="coerce")
    demand = pd.to_numeric(df["Ontario Demand"], errors="coerce")

    hour = timestamp.dt.hour
    dow = timestamp.dt.dayofweek
    month = timestamp.dt.month
    dayofyear = timestamp.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    for order in range(1, 4):
        df[f"year_sin_{order}"] = np.sin(2 * np.pi * order * dayofyear / 365.25)
        df[f"year_cos_{order}"] = np.cos(2 * np.pi * order * dayofyear / 365.25)

    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)
    df["month"] = month
    df["dow"] = dow
    df["days_since_start"] = (timestamp.dt.normalize() - timestamp.min().normalize()).dt.days

    years = _feature_years(timestamp)
    ontario_holidays = country_holidays("CA", subdiv="ON", years=years)
    ontario_dst_transitions = _dst_transition_dates(years)

    dates = timestamp.dt.date
    df["is_holiday"] = dates.isin(ontario_holidays).astype(int)
    df["days_to_nearest_holiday"] = dates.map(
        lambda date_value: _days_to_nearest_holiday(date_value, ontario_holidays)
    )
    df["is_dst"] = dates.isin(ontario_dst_transitions).astype(int)

    weather_defaults = {
        "temp": 18.0,
        "humidity": 50.0,
        "wind": 0.0,
        "solar": 0.0,
    }
    for column, default in weather_defaults.items():
        if column not in df.columns:
            df[column] = default
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    df["temp_squared"] = df["temp"] ** 2
    df["temp_lag_1"] = df["temp"].shift(1)
    df["temp_lag_24"] = df["temp"].shift(24)
    df["humidity"] = df["humidity"]
    df["wind_speed"] = df["wind"]
    df["solar_radiation"] = df["solar"]
    df["cdd"] = (df["temp"] - 18).clip(lower=0)
    df["hdd"] = (18 - df["temp"]).clip(lower=0)

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f"lag_{lag}"] = demand.shift(lag)

    shifted_demand = demand.shift(1)
    shifted_same_hour = shifted_demand.groupby(df["Hour"])
    for window in [7, 14, 21, 28]:
        df[f"samehour_mean_{window}"] = shifted_same_hour.transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        df[f"samehour_std_{window}"] = shifted_same_hour.transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )

    same_dow_keys = [df["Hour"], df["dow"]]
    shifted_same_dow = shifted_demand.groupby(same_dow_keys)
    for window in [4, 8, 12]:
        df[f"samedow_mean_{window}"] = shifted_same_dow.transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        df[f"samedow_std_{window}"] = shifted_same_dow.transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )

    df["roll_mean_24"] = shifted_demand.rolling(24, min_periods=1).mean()
    df["roll_std_24"] = shifted_demand.rolling(24, min_periods=2).std()
    df["roll_mean_168"] = shifted_demand.rolling(168, min_periods=1).mean()

    df["expected_demand"] = shifted_demand.groupby(df["Hour"]).transform(
        lambda series: series.rolling(14, min_periods=1).median()
    )
    df["demand_vs_expected"] = demand - df["expected_demand"]
    df["expected_dow"] = shifted_demand.groupby(same_dow_keys).transform(
        lambda series: series.rolling(8, min_periods=1).median()
    )
    df["demand_vs_dow"] = demand - df["expected_dow"]

    for column in ["prophet_trend", "prophet_yearly", "prophet_weekly", "prophet_daily"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if not nan_counts.empty:
        affected_rows = int(df[nan_counts.index].isna().any(axis=1).sum())
        logger.warning(
            "Feature engineering produced NaN values in %s columns affecting %s rows: %s",
            len(nan_counts),
            affected_rows,
            nan_counts.to_dict(),
        )

    return df


def prepare_training_features(df):
    """Drop rows with NaN values in critical feature columns."""
    before = len(df)
    df = df.dropna(subset=CRITICAL_FEATURE_COLUMNS)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Dropped %s rows with NaN in critical feature columns: %s",
            dropped,
            CRITICAL_FEATURE_COLUMNS,
        )
    return df


def get_feature_cols(df):
    excluded_cols = {
        "id",
        "Date",
        "Hour",
        "Ontario Demand",
        "Timestamp",
        "Date Label",
        "Status",
        "Expected Demand",
        "Deviation",
        "Anomaly Score",
        "Anomaly",
        "expected_demand",
        "expected_dow",
    }
    return [column for column in df.columns if column not in excluded_cols]
