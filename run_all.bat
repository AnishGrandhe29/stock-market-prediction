@echo off
echo ================================
echo NIFTY 50 Prediction System
echo ================================
echo.

echo Starting backend server...
cd /d "%~dp0"
start "Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting frontend server...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ================================
echo System Started Successfully!
echo ================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit this window...
pause > nul
