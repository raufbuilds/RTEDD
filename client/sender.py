import glob
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{BASE_URL}/ingest"
BULK_API_URL = f"{BASE_URL}/ingest/bulk"
LATEST_URL = f"{BASE_URL}/latest"
LOG_FILE = "sent_files.txt"
RETRY_TOTAL = int(os.getenv("CLIENT_RETRY_TOTAL", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("CLIENT_RETRY_BACKOFF_FACTOR", "0.5"))
ROW_RETRY_TOTAL = int(os.getenv("CLIENT_ROW_RETRY_TOTAL", "10"))
REQUEST_TIMEOUT = float(os.getenv("CLIENT_REQUEST_TIMEOUT", "30"))
SEND_MODE = os.getenv("CLIENT_SEND_MODE", "bulk").strip().lower()
BULK_CHUNK_SIZE = max(1, int(os.getenv("CLIENT_BULK_CHUNK_SIZE", "1000")))
REALTIME_DELAY_SECONDS = float(os.getenv("CLIENT_REALTIME_DELAY_SECONDS", "30"))


def create_retry_session():
    session = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        connect=RETRY_TOTAL,
        read=RETRY_TOTAL,
        status=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = create_retry_session()


def wait_for_server():
    while True:
        try:
            response = session.get(LATEST_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            print(f"Waiting for API server: {exc}")
            time.sleep(3)


def get_processed_files():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as file_obj:
        return set(line.strip() for line in file_obj)


def mark_as_processed(filename):
    with open(LOG_FILE, "a", encoding="utf-8") as file_obj:
        file_obj.write(filename + "\n")


def get_latest_progress():
    try:
        response = session.get(LATEST_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"Could not fetch latest server progress: {exc}")
        return None

    date_value = payload.get("Date")
    hour_value = payload.get("Hour")
    if date_value is None or hour_value is None:
        return None

    date_value = pd.to_datetime(date_value, errors="coerce")
    hour_value = pd.to_numeric(hour_value, errors="coerce")
    if pd.isna(date_value) or pd.isna(hour_value):
        return None

    return date_value.normalize(), int(hour_value)


def format_progress(progress):
    if progress is None:
        return "empty database"
    return f"{progress[0].date()} hour {progress[1]}"


def normalize_sender_dataframe(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "Ontario_Demand": "Ontario Demand",
            "ontario demand": "Ontario Demand",
        }
    )
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["Hour"] = pd.to_numeric(df.get("Hour"), errors="coerce")
    df["Ontario Demand"] = pd.to_numeric(df.get("Ontario Demand"), errors="coerce")
    df = df.dropna(subset=["Date", "Hour", "Ontario Demand"])
    df["Hour"] = df["Hour"].astype(int)
    df = df[(df["Hour"] >= 1) & (df["Hour"] <= 24)]
    df["Date"] = df["Date"].dt.normalize()
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    return df


def filter_rows_after_progress(df, latest_progress):
    if latest_progress is None or df.empty:
        return df

    latest_date, latest_hour = latest_progress
    mask = (df["Date"] > latest_date) | (
        (df["Date"] == latest_date) & (df["Hour"] > latest_hour)
    )
    return df[mask].reset_index(drop=True)


def send_row(row_dict, row_index):
    for attempt in range(1, ROW_RETRY_TOTAL + 1):
        try:
            response = session.post(API_URL, json=row_dict, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                print(
                    f"Row {row_index} failed with status {response.status_code} "
                    f"(attempt {attempt}/{ROW_RETRY_TOTAL})"
                )
                time.sleep(min(attempt * 2, 15))
                continue

            result = response.json()
            status = result.get("status")
            date_str = row_dict.get("Date", "N/A")
            hour_str = row_dict.get("Hour", "N/A")
            if status == "saved":
                print(f"Sent row {row_index} successfully ({date_str} hour {hour_str})")
            elif status == "skipped":
                print(f"Skipped row {row_index}, already exists ({date_str} hour {hour_str})")
            else:
                print(f"Processed row {row_index} with status {status} ({date_str} hour {hour_str})")
            return True
        except requests.RequestException as exc:
            print(
                f"Error sending row {row_index} "
                f"(attempt {attempt}/{ROW_RETRY_TOTAL}): {exc}"
            )
            time.sleep(min(attempt * 2, 15))

    return False


def dataframe_to_payload_rows(df):
    payload_rows = []
    for row in df.to_dict(orient="records"):
        row["Date"] = row["Date"].isoformat()
        row["Hour"] = int(row["Hour"])
        row["Ontario Demand"] = float(row["Ontario Demand"])
        payload_rows.append(row)
    return payload_rows


def send_bulk_rows(rows, start_index):
    for attempt in range(1, ROW_RETRY_TOTAL + 1):
        try:
            response = session.post(
                BULK_API_URL,
                json={"rows": rows},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                print(
                    f"Bulk chunk starting at row {start_index} failed with status "
                    f"{response.status_code} (attempt {attempt}/{ROW_RETRY_TOTAL})"
                )
                time.sleep(min(attempt * 2, 15))
                continue

            result = response.json()
            print(
                f"Bulk chunk starting at row {start_index}: "
                f"saved={result.get('saved', 0)}, "
                f"skipped={result.get('skipped', 0)}, "
                f"invalid={result.get('invalid', 0)}"
            )
            return True
        except requests.RequestException as exc:
            print(
                f"Error sending bulk chunk starting at row {start_index} "
                f"(attempt {attempt}/{ROW_RETRY_TOTAL}): {exc}"
            )
            time.sleep(min(attempt * 2, 15))

    return False


def send_dataframe_bulk(df):
    payload_rows = dataframe_to_payload_rows(df)
    for start in range(0, len(payload_rows), BULK_CHUNK_SIZE):
        chunk = payload_rows[start : start + BULK_CHUNK_SIZE]
        if not send_bulk_rows(chunk, start):
            return False
    return True


def send_dataframe_realtime(df):
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict["Date"] = row_dict["Date"].isoformat()
        if send_row(row_dict, idx):
            time.sleep(REALTIME_DELAY_SECONDS)
            continue

        print(f"Stopping current file; row {idx} was not confirmed by the server")
        return False
    return True


wait_for_server()
processed_files = get_processed_files()
latest_progress = get_latest_progress()
base_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "..", "cleaner", "processed_.csv_file")

print(f"Looking for CSV files in: {folder_path}")
print(f"Folder exists: {os.path.exists(folder_path)}")
print(f"Send mode: {SEND_MODE}")
if SEND_MODE == "bulk":
    print(f"Bulk chunk size: {BULK_CHUNK_SIZE}")
else:
    print(f"Realtime delay: {REALTIME_DELAY_SECONDS} seconds")

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files.sort()
print(f"Found {len(csv_files)} CSV files")

if latest_progress is None:
    print("Server progress: empty database or progress unavailable")
    print("Starting from the beginning of the available CSV history")
else:
    print(f"Server progress: {format_progress(latest_progress)}")
    print(f"Starting after {format_progress(latest_progress)}")

if not csv_files:
    print("ERROR: No CSV files found!")
else:
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")

        if filename in processed_files:
            print(f"Skipping {filename}, already sent.")
            continue

        try:
            df = pd.read_csv(file_path)
            df = normalize_sender_dataframe(df)
            valid_count = len(df)
            if not df.empty:
                file_start = (df.iloc[0]["Date"], int(df.iloc[0]["Hour"]))
                file_end = (df.iloc[-1]["Date"], int(df.iloc[-1]["Hour"]))
                print(
                    f"File range: {format_progress(file_start)} -> "
                    f"{format_progress(file_end)}"
                )
            df = filter_rows_after_progress(df, latest_progress)
            print(f"Read {valid_count} valid rows from {filename}")

            if df.empty:
                print(
                    f"Skipping {filename}, all rows are at or before "
                    f"{format_progress(latest_progress)}"
                )
                mark_as_processed(filename)
                continue

            start_progress = (df.iloc[0]["Date"], int(df.iloc[0]["Hour"]))
            end_progress = (df.iloc[-1]["Date"], int(df.iloc[-1]["Hour"]))
            print(
                f"Sending {len(df)} new rows from {filename} "
                f"({format_progress(start_progress)} -> {format_progress(end_progress)})"
            )

            if SEND_MODE == "realtime":
                file_successful = send_dataframe_realtime(df)
            else:
                file_successful = send_dataframe_bulk(df)

            if file_successful:
                latest_progress = (df.iloc[-1]["Date"], int(df.iloc[-1]["Hour"]))

            if file_successful:
                mark_as_processed(filename)
                print(f"Marked {filename} as processed")
        except Exception as exc:
            print(f"Error reading {file_path}: {exc}")

print("\nAll done!")
