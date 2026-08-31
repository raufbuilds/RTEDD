import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dashboard'))

import dashboard.dashboard as dashboard
import server.app as app


def test_benchmark_forecast_methods_include_model_series():
    methods = dashboard.forecast_methods_for_view(include_today_in_training=False)

    assert "Average" in methods
    assert "Expected Median" in methods
    assert "Prophet" in methods
    assert "LightGBM" in methods
    assert "Ensemble" in methods


def test_forecast_training_frame_matches_expected_windowing():
    df = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Hour": 1, "Ontario Demand": 10.0, "Timestamp": "2024-01-01 01:00:00"},
            {"Date": "2024-01-01", "Hour": 2, "Ontario Demand": 12.0, "Timestamp": "2024-01-01 02:00:00"},
            {"Date": "2024-01-02", "Hour": 1, "Ontario Demand": 15.0, "Timestamp": "2024-01-02 01:00:00"},
            {"Date": "2024-01-02", "Hour": 2, "Ontario Demand": 17.0, "Timestamp": "2024-01-02 02:00:00"},
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    benchmark = app.forecast_training_frame(df, target_date="2024-01-02", include_target_date=False)
    live = app.forecast_training_frame(df, target_date="2024-01-02", include_target_date=True)

    assert sorted(benchmark["Date"].unique()) == [pd.Timestamp("2024-01-01")]
    assert sorted(live["Date"].unique()) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
