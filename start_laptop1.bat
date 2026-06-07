@echo off
setlocal EnableDelayedExpansion
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

:: 3. Check port 8000 before starting FastAPI.
echo Checking port 8000...
set "PORT_PIDS="
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r /c:":8000 .*LISTENING"') do (
    set "PORT_PIDS=!PORT_PIDS! %%a"
)

if defined PORT_PIDS (
    echo Port 8000 is already in use by PID(s):!PORT_PIDS!
    choice /m "Stop these process(es) so FastAPI can start"
    if errorlevel 2 (
        echo FastAPI startup cancelled. Free port 8000 and run this script again.
        pause
        exit /b 1
    )
    for %%a in (!PORT_PIDS!) do (
        taskkill /f /pid %%a >nul 2>&1
    )
)

echo Starting FastAPI server...
:: Reload only when server code changes. Dashboard edits should not restart the API.
start "FASTAPI" cmd /k "uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir server"


echo Starting sender...
start "SENDER" cmd /k "python client/sender.py"

echo All systems running.
pause
