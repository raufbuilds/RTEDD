@echo off
setlocal
:: Set script to run from the folder it is located in
cd /D "%~dp0"

:: 1. Verify VENV exists before starting
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at: %~dp0venv
    echo Please run 'setup_env.bat' first.
    pause
    exit /b 1
)

:: 2. Activate VENV using double quotes to handle spaces in path (e.g., "IT BD")
call "venv\Scripts\activate.bat"

:: 3. Load API_PORT from .env, defaulting to 8000.
set "API_PORT=8000"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if /i "%%a"=="API_PORT" set "API_PORT=%%b"
    )
)
set "API_PORT=%API_PORT: =%"
set "API_PORT=%API_PORT:"=%"

:: 4. Check the configured port before starting FastAPI.
echo Checking port %API_PORT%...
set "PORT_PIDS="
set "PORT_FILE=%TEMP%\pub_port_pids.txt"
powershell -NoLogo -NoProfile -Command "Try { (Get-NetTCPConnection -LocalPort %API_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -Expand OwningProcess) -join ' ' } Catch { }" > "%PORT_FILE%"
if exist "%PORT_FILE%" set /p PORT_PIDS=<"%PORT_FILE%"
del "%PORT_FILE%" 2>nul

if defined PORT_PIDS (
    echo Port %API_PORT% is already in use by PIDs:%PORT_PIDS%
    choice /m "Stop these processes so FastAPI can start"
    if errorlevel 2 (
        echo FastAPI startup cancelled. Free port %API_PORT% and run this script again.
        pause
        exit /b 1
    )
    for %%a in (%PORT_PIDS%) do (
        taskkill /f /pid %%a >nul 2>&1
    )
)

echo Starting FastAPI server...
:: Reload only when server code changes. Dashboard edits should not restart the API.
start "FASTAPI" cmd /k "uvicorn server.app:app --host 0.0.0.0 --port %API_PORT% --reload --reload-dir server"

echo Starting sender...
start "SENDER" cmd /k "python client/sender.py"

echo All systems running.
pause
