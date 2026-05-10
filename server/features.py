import holidays
import numpy as np
import pandas as pd


ONTARIO_HOLIDAYS = holidays.CA(prov="ON", years=range(2020, 2026))


def _days_to_nearest_holiday(date_value):
    current_date = pd.Timestamp(date_value).date()
    distances = [
        abs(offset)
        for offset in range(-7, 8)
        if current_date + pd.Timedelta(days=offset) in ONTARIO_HOLIDAYS
    ]
    return min(distances) if distances else 8


def _dst_transition_dates():
    dates = set()
    for year in range(2020, 2026):
        march = pd.Timestamp(year=year, month=3, day=1)
        november = pd.Timestamp(year=year, month=11, day=1)

        days_until_first_sunday_march = (6 - march.dayofweek) % 7
        days_until_first_sunday_november = (6 - november.dayofweek) % 7
        second_sunday_march = march + pd.Timedelta(days=days_until_first_sunday_march + 7)
        first_sunday_november = november + pd.Timedelta(days=days_until_first_sunday_november)

        dates.add(second_sunday_march.date())
        dates.add(first_sunday_november.date())
    return dates


ONTARIO_DST_TRANSITIONS = _dst_transition_dates()


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

    dates = timestamp.dt.date
    df["is_holiday"] = dates.isin(ONTARIO_HOLIDAYS).astype(int)
    df["days_to_nearest_holiday"] = dates.map(_days_to_nearest_holiday)
    df["is_dst"] = dates.isin(ONTARIO_DST_TRANSITIONS).astype(int)

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

    return df.dropna()


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
