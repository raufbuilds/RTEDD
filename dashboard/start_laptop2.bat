@echo off
:: Move to the dashboard directory where this .bat file sits
cd /D "%~dp0"

:: Use quotes around the path to handle the space in "IT BD"
:: Also, we go up one level if the bat is inside the dashboard folder
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else (
    call "venv\Scripts\activate.bat"
)

echo Starting dashboard...
:: Use 'python -m' as a backup way to trigger streamlit
python -m streamlit run dashboard.py --server.address 127.0.0.1
pause
