@echo off
echo ----------------------------------------------
echo Setting up PUB Realtime System Environment...
echo ----------------------------------------------

:: Fix PowerShell execution policy for the current user
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo venv already exists. skipping...
)

:: Activate and install requirements
echo Installing dependencies from requirements.txt...
cd /d "%~dp0"

if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
    pip install --upgrade pip
    pip install -r "%~dp0requirements.txt"
    
    echo Loading environment variables from .env...
    powershell -Command "& '%~dp0load-env.ps1'"
) else (
    echo [ERROR] Activation script not found in venv.
    echo This usually means the virtual environment is corrupted or still in use.
    echo Close any running terminals, editors, or Python processes that may be using the venv, then delete the venv folder and rerun setup_env.bat.
    echo.
    echo To delete the folder manually, run:
    echo     rmdir /s /q "%~dp0venv"
    pause
    exit /b 1
)

echo ----------------------------------------------
echo SETUP COMPLETE!
echo You can now run your start_laptop scripts.
echo ----------------------------------------------
pause