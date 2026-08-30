import pandas as pd

import server.app as app


def test_calculate_2025_metrics_matches_forecast_target_date(monkeypatch):
    actual_df = pd.DataFrame(
        [
            {"Date": "2025-01-01", "Hour": 1, "Ontario Demand": 10.0},
            {"Date": "2025-01-01", "Hour": 2, "Ontario Demand": 20.0},
            {"Date": "2025-01-02", "Hour": 1, "Ontario Demand": 30.0},
            {"Date": "2025-01-02", "Hour": 2, "Ontario Demand": 40.0},
        ]
    )
    actual_df["Date"] = pd.to_datetime(actual_df["Date"])

    forecast_payload = [
        {"Hour": 1, "Ensemble": 15.0, "Ensemble_P10": 10.0, "Ensemble_P90": 20.0},
        {"Hour": 2, "Ensemble": 25.0, "Ensemble_P10": 20.0, "Ensemble_P90": 30.0},
    ]

    monkeypatch.setattr(app, "demand_dataframe", lambda: actual_df)
    monkeypatch.setattr(
        app,
        "get_latest_fresh_forecast",
        lambda include_target_date: {
            "target_date": "2025-01-01",
            "forecast": forecast_payload,
            "status": "fresh",
            "trained_at": 1.0,
        },
    )

    result = app.calculate_2025_metrics()

    assert result["available"] is True
    assert result["forecast_records"] == 2
    assert result["forecast_target_date"] == "2025-01-01"
    assert result["metrics"]["MAE"] == 5.0
