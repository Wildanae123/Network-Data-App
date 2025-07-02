@REM START-HERE.bat
@echo off
title Network Data App - Development Environment

echo ================================================================
echo.
echo   Network Data App v1.0
echo   Starting Development Environment...
echo.
echo ================================================================

REM Check if backend virtual environment exists
if not exist "backend\venv\Scripts\python.exe" (
    echo ERROR: Backend virtual environment not found!
    echo.
    echo Please run INSTALL.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

REM Check if backend server exists
if not exist "backend\server.py" (
    echo ERROR: backend server not found!
    echo.
    echo Please ensure server.py is in the backend folder.
    echo.
    pause
    exit /b 1
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install frontend dependencies!
        pause
        exit /b 1
    )
    cd ..
    echo Frontend dependencies installed successfully!
    echo.
)

REM Check if App component exists
if not exist "frontend\src\App.jsx" (
    echo WARNING: App.jsx not found, using standard App.jsx
    echo.
) else (
    echo Switching to App component...
    copy /Y "frontend\src\App.jsx" "frontend\src\App.jsx" >nul 2>&1
    if not exist "frontend\src\App.css" (
        echo WARNING: App.css not found
    ) else (
        copy /Y "frontend\src\App.css" "frontend\src\App.css" >nul 2>&1
    )
    echo Components activated!
    echo.
)

echo Starting Frontend Development Server...
START "Frontend (React + Vite)" cmd /c "cd frontend && npm run dev && echo. && echo Frontend server stopped. && pause"

REM Wait for frontend to start
echo Waiting for frontend to initialize...
timeout /t 8 /nobreak >nul

echo Starting Backend Development Server...
START "Backend (Flask)" cmd /c "cd backend && echo Starting Flask Server... && venv\Scripts\python.exe server.py && echo. && echo Backend server stopped. && pause"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ================================================================
echo   Development Environment Started Successfully!
echo.
echo   Frontend URL: http://localhost:5173
echo   Backend API:  http://localhost:5000
echo.
echo   Quick Start Guide:
echo   1. Open http://localhost:5173 in your browser
echo   2. Enter your SSH credentials
echo   3. Click "Start & Select Device File"
echo   4. Choose your CSV file with device list
echo   5. Monitor real-time progress
echo   6. Export results when complete
echo.
echo   For detailed help, see README.md and TUTORIAL.md
echo   For troubleshooting, check the log files
echo.
echo   Opening browser automatically...
echo ================================================================
echo.

REM Open the frontend URL in default browser
echo Opening http://localhost:5173 in your default browser...
start http://localhost:5173

REM Wait 3 seconds then close this window
echo.
echo This window will close automatically in 3 seconds...
timeout /t 3 /nobreak >nul