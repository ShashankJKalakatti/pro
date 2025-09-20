@echo off
TITLE 🚀 E-commerce Recommendation System Full Launcher

echo ================================
echo Starting Federated Server...
echo ================================
start cmd /k "cd backend && python scripts\server.py"

echo ================================
echo Starting Federated Learning...
echo ================================
start cmd /k "cd backend && python scripts\federated.py"

echo ================================
echo Starting Backend (Flask API)...
echo ================================
start cmd /k "cd backend && set FLASK_APP=app.py && set FLASK_ENV=development && flask run"

echo.
echo ================================
echo Starting Frontend (React.js)...
echo ================================
start cmd /k "cd frontend && npm start"

echo.
echo 🟢 All systems launching!
REM Removed browser auto-open to prevent duplicate tabs

cd ..
