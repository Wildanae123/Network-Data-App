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
echo Current directory: %CD%
cd frontend
if errorlevel 1 (
    echo ERROR: Failed to change to frontend directory!
    pause
    exit /b 1
)

echo Changed to frontend directory: %CD%

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: Frontend dependency installation failed!
        cd ..
        pause
        exit /b 1
    )
    echo Frontend dependencies installed successfully!
) else (
    echo Frontend dependencies already installed.
)

echo Returning to root directory...
cd ..
if errorlevel 1 (
    echo ERROR: Failed to return to root directory!
    pause
    exit /b 1
)

echo Step 1 completed successfully!
echo Current directory: %CD%
echo.

echo Step 2: Setting up Backend...
echo Changing to backend directory...
cd backend
if errorlevel 1 (
    echo ERROR: Failed to change to backend directory!
    pause
    exit /b 1
)

echo Changed to backend directory: %CD%

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python is installed and available in PATH.
        cd ..
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)

REM Verify virtual environment was created properly
if not exist "venv\Scripts\pip.exe" (
    echo ERROR: Virtual environment was not created properly!
    echo pip.exe not found in venv\Scripts\
    echo Trying to recreate virtual environment...
    rmdir /s /q venv 2>nul
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Still failed to create virtual environment!
        echo Please check if Python is properly installed.
        cd ..
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    cd ..
    pause
    exit /b 1
)

echo Installing backend dependencies...
if exist "requirements.txt" (
    echo Installing from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Backend dependency installation from requirements.txt failed!
        cd ..
        pause
        exit /b 1
    )
) else (
    echo Installing individual packages...
    pip install flask>=3.0.0 flask-cors>=4.0.0 requests>=2.31.0 pandas>=2.1.0 numpy>=1.25.0 plotly>=5.17.0 pyyaml>=6.0.1 jsonrpclib-pelix>=0.4.3.2 openpyxl>=3.1.2 werkzeug>=3.0.0 urllib3>=2.0.7
    if errorlevel 1 (
        echo ERROR: Backend dependency installation failed!
        cd ..
        pause
        exit /b 1
    )
)

echo Backend dependencies installed successfully!

echo Deactivating virtual environment...
deactivate

echo Returning to root directory...
cd ..
if errorlevel 1 (
    echo ERROR: Failed to return to root directory!
    pause
    exit /b 1
)

echo Step 2 completed successfully!
echo.
echo ================================================
echo  Setup Complete!
echo.
echo  To start the development environment, run:
echo  START-HERE.bat
echo ================================================
echo.
pause