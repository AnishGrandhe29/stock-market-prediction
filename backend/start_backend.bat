@echo off
echo Starting Backend Server...
cd /d "%~dp0"

:: Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
