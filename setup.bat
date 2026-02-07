@echo off
echo ================================
echo Setup NIFTY 50 Prediction System
echo ================================
echo.

:: Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

:: Check Node.js
node --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

echo [1/4] Creating Python virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing Python dependencies...
pip install -r requirements.txt

echo [3/4] Setting up environment file...
if not exist ".env" (
    copy .env.example .env
    echo Created .env file - please edit with your settings!
)

cd ..

echo [4/4] Installing frontend dependencies...
cd frontend
call npm install

cd ..

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Edit backend\.env with your settings
echo 2. Set up PostgreSQL database
echo 3. Run: run_all.bat
echo.
pause
