@echo off
setlocal EnableDelayedExpansion
:: Move to the dashboard directory where this .bat file sits
cd /D "%~dp0"

:: Use quotes around the path to handle the space in "IT BD"
:: Also, we go up one level if the bat is inside the dashboard folder
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else (
    call "venv\Scripts\activate.bat"
)

:: Load API settings from the root .env so the dashboard uses the same port as FastAPI.
if exist "..\.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("..\.env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
    )
)

echo Starting dashboard...
echo Dashboard API: !API_BASE_URL!
:: Use 'python -m' as a backup way to trigger streamlit
python -m streamlit run dashboard.py
pause
