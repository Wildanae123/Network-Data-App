@REM INSTALL.bat
@echo off
title Development Environment Setup

echo ================================================
echo.
echo  Setting up Development Environment...
echo.
echo ================================================

REM Check if we're in the right directory
if not exist "frontend" (
    echo ERROR: frontend folder not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

if not exist "backend" (
    echo ERROR: backend folder not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

echo Step 1: Setting up Frontend...
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: Frontend dependency installation failed!
        pause
        exit /b 1
    )
) else (
    echo Frontend dependencies already installed.
)

cd ..

echo.
echo Step 2: Setting up Backend...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python is installed and available in PATH.
        pause
        exit /b 1
    )
)

REM Install backend dependencies
echo Installing backend dependencies...
venv\Scripts\pip.exe install flask flask-cors netmiko pandas plotly pyyaml openpyxl pywebview
if errorlevel 1 (
    echo ERROR: Backend dependency installation failed!
    pause
    exit /b 1
)

cd ..

echo.
echo ================================================
echo  Setup Complete!
echo.
echo  To start the development environment, run:
echo  START-HERE.bat
echo ================================================
echo.
pause