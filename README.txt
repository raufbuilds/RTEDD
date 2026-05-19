============================================================
Real-Time Electricity Demand Dashboard - SETUP & USER GUIDE
============================================================

This project runs a FastAPI server, a data sender, and a Streamlit dashboard for
real-time electricity demand monitoring and forecasting.

The current forecasting setup is server-backed:
- FastAPI stores all demand rows in data.db.
- FastAPI prepares dashboard records, anomaly fields, and the baseline band.
- FastAPI trains and caches Prophet + LightGBM forecast results.
- Streamlit only fetches prepared data and renders charts, filters, tables, and controls.

-----------------------------
PREREQUISITES
-----------------------------
1. Python 3.10 or higher.
   - Download from: https://www.python.org/downloads/
   - During installation, enable "Add Python to PATH".

2. Processed CSV files in:
   - cleaner/processed_.csv_file/

Expected file names look like:
   - PUB_Demand_2020.P.csv
   - PUB_Demand_2021.P.csv
   - PUB_Demand_2022.P.csv
   - PUB_Demand_2023.P.csv
   - PUB_Demand_2024.P.csv
   - PUB_Demand_2025.P.csv

-----------------------------
STEP 1: INITIAL SETUP
-----------------------------
Double-click:

   setup_env.bat

This creates the local Python environment in venv and installs packages from:

   requirements.txt

Main packages include FastAPI, Streamlit, Pandas, Plotly, Prophet, LightGBM,
holidays, and joblib.

-----------------------------
STEP 2: CURRENT TRAIN/TEST PLAN
-----------------------------
The current database is prepared for this workflow:

   Train forecast models with: 2020-2024 data
   Test outcomes with:        2025 data


Keep 2025 out of the database until after the server has trained forecasts from
the 2020-2024 data.

-----------------------------
STEP 3: START LAPTOP 1
-----------------------------
Double-click:

   start_laptop1.bat

This starts:
   - FastAPI server: server/app.py
   - Sender script:  client/sender.py

FastAPI endpoints used by the dashboard include:
   - /dashboard/data
   - /forecast/latest
   - /forecast/refresh
   - /records/count

-----------------------------
STEP 4: VIEW THE DASHBOARD
-----------------------------
On the same laptop, or another laptop on the same network, run:

   dashboard/start_laptop2.bat

The Streamlit dashboard opens at:

   http://localhost:8501

If it does not open automatically, copy the URL from the terminal into your
browser.

-----------------------------
FORECASTING NOTES
-----------------------------
Forecast training happens on the server, not in the dashboard.

The server uses:
   - Prophet
   - LightGBM direct multi-step models
   - server/features.py for feature engineering

LightGBM model files are cached in:

   models/

This folder is ignored by Git. If you delete it, the server will retrain models
the next time forecasts are refreshed.

LightGBM retraining is controlled by:

   FORECAST_LIGHTGBM_RETRAIN_ROWS=168

The default 168 means the server reuses cached LightGBM models until at least
168 new training rows exist. For hourly data, 168 rows is 7 days.

The forecast ensemble currently uses:

   5% Prophet + 95% LightGBM

-----------------------------
HOW TO TEST WITH 2025 DATA
-----------------------------
1. Start the server with only 2020-2024 data in data.db.
2. Open the dashboard and refresh/trigger forecasts.
3. Wait for forecast training to finish.
4. Ingest the 2025 file afterward:

   cleaner/processed_.csv_file/PUB_Demand_2025.P.csv

5. Use the dashboard to compare forecasted values against the 2025 actual rows.

If 2025 was already sent by mistake, remove it from data.db and clear forecast
cache before training again.

-----------------------------
SENDER MODE
-----------------------------
The sender mode is controlled in .env:

   CLIENT_SEND_MODE=bulk

Common options:
   - bulk: sends rows quickly in chunks
   - realtime: sends one row at a time with a delay

Useful .env settings:
   - CLIENT_BULK_CHUNK_SIZE
   - CLIENT_REALTIME_DELAY_SECONDS
   - MAX_BULK_INGEST_ROWS

-----------------------------
RUNNING ON TWO LAPTOPS
-----------------------------
If the dashboard runs on Laptop 2 and the server runs on Laptop 1:

1. On Laptop 1, find the IPv4 address:

   ipconfig

2. On Laptop 2, set SERVER_IP for the dashboard.
   The dashboard reads:

   SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")

You can set SERVER_IP in the environment before starting Streamlit, or edit the
startup script to set it.

-----------------------------
TROUBLESHOOTING
-----------------------------
- Dashboard says "Waiting for data from the server..."
  - Make sure FastAPI is running.
  - Check that data.db has rows.
  - Check the terminal for server errors.

- Forecast stays empty
  - Make sure at least 500 training rows exist.
  - Check that LightGBM is installed.
  - Delete models/ and refresh forecasts to retrain.

- 2025 appears before testing
  - Delete 2025 rows from data.db.
  - Clear forecast_cache.
  - Remove PUB_Demand_2025.P.csv from sent_files.txt.

- Port already in use
  - Close old server/dashboard windows or restart the computer.

-----------------------------
END OF GUIDE
-----------------------------
