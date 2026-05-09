
============================================================
Real-Time Electricity Demand Dashboard - SETUP & USER GUIDE
============================================================

This guide explains how to set up and run the Real-Time Electricity Demand Dashboard system.

-----------------------------
PREREQUISITES
-----------------------------
1. Python 3.10 or higher must be installed.
   - Download from: https://www.python.org/downloads/
   - During installation, make sure to check "Add Python to PATH".

-----------------------------
STEP 1: INITIAL SETUP
-----------------------------
1. Double-click setup_env.bat in the project folder.
   - This will:
     - Allow scripts to run on Windows
     - Create a folder named venv (your Python environment)
     - Install all required packages (FastAPI, Streamlit, Pandas, etc.)
2. Wait until the window indicates setup is complete.
3. You should now see a new folder named "venv" in your project directory.

-----------------------------
STEP 2: DATA CHECK
-----------------------------
1. Make sure your processed CSV file is present at:
   - cleaner/processed_.csv_file/PUB_Demand_YYYY.P.csv (where YYYY is the year)
2. If you have multiple years, you can have several files (e.g., PUB_Demand_2023.P.csv).

-----------------------------
SENDER MODE: BULK OR REALTIME
-----------------------------
The sender behavior is controlled in the .env file:

   CLIENT_SEND_MODE=bulk

Current config lines:
   - .env line 9: CLIENT_SEND_MODE=bulk ; =realtime, for live-style behavior
   - .env line 10: CLIENT_BULK_CHUNK_SIZE=1000 ; The number of rows sent per request
   - .env line 11: CLIENT_REALTIME_DELAY_SECONDS=30 ; wait time between rows
   - .env line 12: MAX_BULK_INGEST_ROWS=5000

Use bulk mode when you want to load historical data quickly for forecasting.
In bulk mode, client/sender.py sends many rows at once to the server using:

   /ingest/bulk

Related code lines:
   - client/sender.py line 28: BULK_API_URL points to /ingest/bulk
   - client/sender.py line 35: reads CLIENT_SEND_MODE
   - client/sender.py line 36: reads CLIENT_BULK_CHUNK_SIZE
   - client/sender.py line 218: send_dataframe_bulk sends chunks of rows
   - client/sender.py line 308: bulk mode calls send_dataframe_bulk
   - server/app.py line 577: defines the /ingest/bulk endpoint
   - server/app.py line 98: reads MAX_BULK_INGEST_ROWS

Related code lines:
   - client/sender.py line 37: reads CLIENT_REALTIME_DELAY_SECONDS
   - client/sender.py line 227: send_dataframe_realtime sends one row at a time
   - client/sender.py line 306: realtime mode calls send_dataframe_realtime

So with CLIENT_SEND_MODE=realtime and CLIENT_REALTIME_DELAY_SECONDS=30, the sender
will send one row every 30 seconds.

-----------------------------
STEP 3: START THE SYSTEM (LAPTOP 1)
-----------------------------
1. Double-click start_laptop1.bat
   - This will start:
     - The FastAPI Server (server/app.py)
     - The Sender Script (client/sender.py)
2. Wait for the terminal to show messages like "Uvicorn running..." and "Sender started..."

-----------------------------
STEP 4: VIEW THE DASHBOARD (LAPTOP 1 or 2)
-----------------------------
1. On the same laptop or a different one on the same network:
   - Double-click start_laptop2.bat
   - This will open the dashboard in your web browser (Streamlit UI)
2. If it does not open automatically, look for a link like http://localhost:8501 in the terminal and open it in your browser.

-----------------------------
RUNNING ON TWO LAPTOPS
-----------------------------
If you want to view the dashboard from a different laptop:
1. Find the local IP address of Laptop 1 (the server):
   - Open Command Prompt and type: ipconfig
   - Look for the "IPv4 Address" (e.g., 192.168.1.10)
2. On Laptop 2, open the file dashboard/dashboard.py and set:
   - SERVER_IP = "Laptop 1's IP address"
3. Save the file and run start_laptop2.bat on Laptop 2.

-----------------------------
TROUBLESHOOTING
-----------------------------
- Sender says 'FileNotFound':
  - Check that the CSV filename matches the one set in client/sender.py.
- Dashboard says 'Waiting for data from the server...':
  - Make sure the server is running (see Step 3)
  - Check for errors in the terminal window
- Connection errors:
  - If running on two laptops, make sure both are on the same network
  - Double-check the SERVER_IP setting
- Port already in use:
  - Another program may be using the same port. Restart your computer or change the port in the code (advanced users).
- Still stuck?
  - Try restarting both the server and the dashboard
  - Check the main project README or ask for help

-----------------------------
WHERE TO GET HELP
-----------------------------
- Read the full user guide: dashboard/README_Dashboard.md

-----------------------------
END OF GUIDE
-----------------------------

